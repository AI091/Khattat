"""Morphing optimization (paper §4.5): L_SDS + 0.5 n L_OCR + 0.5 L_ACAP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from khattat.fonts.shaping import ShapedWord, shape_word
from khattat.losses.acap import ACAPLoss
from khattat.losses.ocr import OCRLoss, ocr_loss_weight
from khattat.losses.sds import SDSConfig, SDSLoss, build_caption, encode_captions
from khattat.losses.tone import ToneLoss
from khattat.render.scene import (
    GlyphScene,
    build_region_scene,
    build_word_scene,
    render,
    to_nchw,
)


@dataclass
class MorphConfig:
    """Optimization settings. Defaults are the paper's."""

    num_iter: int = 500
    render_size: int = 600
    control_points_per_glyph: int = 120
    lr_init: float = 1.0
    lr_final: float = 0.4
    lr_delay_steps: int = 100
    lr_delay_mult: float = 0.1
    acap_weight: float = 0.5
    ocr_weight_base: float = 0.5
    use_ocr: bool = True
    use_acap: bool = True
    use_tone: bool = False
    """Word-as-Image's tone loss, for the baseline."""
    seed: int = 0
    save_every: int = 0
    """Save a frame every k steps; 0 disables."""
    release_cache_between_branches: bool = True
    """Empty the CUDA cache between the SDS and OCR backward passes."""
    free_text_encoder: bool = True
    """Free the SD text encoder after embedding the caption.

    Embed any later captions first with `encode_captions`."""
    sds: SDSConfig = field(default_factory=SDSConfig)


def word_as_image_config(**overrides) -> MorphConfig:
    """The Word-as-Image baseline: SDS + ACAP + tone, no OCR loss."""
    return replace(MorphConfig(use_ocr=False, use_tone=True), **overrides)


@dataclass
class MorphResult:
    region_scene: GlyphScene
    region_image: torch.Tensor
    region_initial: torch.Tensor
    word_scene: GlyphScene
    word_image: torch.Tensor
    word_initial: torch.Tensor
    history: list[dict[str, float]]
    caption: str
    region: tuple[int, int]

    @property
    def final_losses(self) -> dict[str, float]:
        return self.history[-1] if self.history else {}


def learning_rate_decay(
    step: int,
    lr_init: float,
    lr_final: float,
    max_steps: int,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
) -> float:
    """Log-linear decay with a warm-up, as in Word-as-Image."""
    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / lr_delay_steps, 0), 1)
        )
    else:
        delay_rate = 1.0
    t = min(max(step / max_steps, 0), 1)
    log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return delay_rate * log_lerp


def morph(
    *,
    word: str | ShapedWord,
    concept: str,
    region: tuple[int, int],
    pipe,
    font_path: str | Path | None = None,
    ocr_encoder=None,
    config: MorphConfig | None = None,
    device: torch.device | str = "cuda",
    output_dir: Path | str | None = None,
    progress: bool = True,
) -> MorphResult:
    """Morph the character range `region` of `word` towards `concept`."""
    import pydiffvg

    cfg = config or MorphConfig()
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    pydiffvg.set_use_gpu(device.type == "cuda")

    if isinstance(word, ShapedWord):
        shaped = word
    else:
        if font_path is None:
            raise ValueError("font_path is required when word is a string")
        shaped = shape_word(word, font_path, canvas_size=cfg.render_size)

    scene = build_region_scene(
        shaped,
        region,
        size=cfg.render_size,
        control_points_per_glyph=cfg.control_points_per_glyph,
        device=device,
    )

    with torch.no_grad():
        region_initial = render(scene, seed=cfg.seed).detach()

    caption = build_caption(concept)
    encode_captions(pipe, [caption], device=device, free_encoder=cfg.free_text_encoder)
    sds = SDSLoss(pipe, caption, cfg.sds, device=device)

    ocr = None
    if cfg.use_ocr:
        if ocr_encoder is None:
            raise ValueError("use_ocr is set but no ocr_encoder was provided")
        ocr = OCRLoss(ocr_encoder, to_nchw(region_initial), device=device)
    w_ocr = ocr_loss_weight(region[1] - region[0], cfg.ocr_weight_base)

    acap = ACAPLoss(scene, device=device) if cfg.use_acap else None
    tone = ToneLoss(to_nchw(region_initial)) if cfg.use_tone else None

    optim = torch.optim.Adam(scene.parameters(), lr=1.0, betas=(0.9, 0.9), eps=1e-6)
    scheduler = LambdaLR(
        optim,
        lr_lambda=lambda step: learning_rate_decay(
            step,
            cfg.lr_init,
            cfg.lr_final,
            cfg.num_iter,
            cfg.lr_delay_steps,
            cfg.lr_delay_mult,
        ),
    )

    out = Path(output_dir) if output_dir else None
    if out and cfg.save_every:
        (out / "frames").mkdir(parents=True, exist_ok=True)

    bar = task = None
    if progress:
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

        bar = Progress(
            TextColumn(f"[cyan]{concept}[/cyan] {region}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.fields[loss]}"),
            TimeRemainingColumn(),
        )
        task = bar.add_task("morph", total=cfg.num_iter, loss="")
        bar.start()

    history: list[dict[str, float]] = []
    try:
        for step in range(cfg.num_iter):
            optim.zero_grad(set_to_none=True)
            img = render(scene, seed=step)

            # Backpropagate each image loss separately into a detached render, then
            # push the summed gradient through the rasterizer once. Same update,
            # but the VAE and OCR graphs never share memory.
            img_leaf = img.detach().requires_grad_(True)
            img_nchw = to_nchw(img_leaf)
            entry: dict[str, float] = {}

            loss_sds = sds(sds.augment_image(img_nchw))
            loss_sds.backward()
            entry["sds"] = float(loss_sds.detach())
            del loss_sds
            if cfg.release_cache_between_branches:
                torch.cuda.empty_cache()

            if ocr is not None:
                loss_ocr = ocr(img_nchw)
                (w_ocr * loss_ocr).backward()
                entry["ocr"] = float(loss_ocr.detach())

            if tone is not None:
                loss_tone = tone(img_nchw, step)
                loss_tone.backward()
                entry["tone"] = float(loss_tone.detach())

            img.backward(img_leaf.grad)

            if acap is not None:
                loss_acap = acap()
                (cfg.acap_weight * loss_acap).backward()
                entry["acap"] = float(loss_acap.detach())

            entry["total"] = (
                entry["sds"]
                + w_ocr * entry.get("ocr", 0.0)
                + cfg.acap_weight * entry.get("acap", 0.0)
                + entry.get("tone", 0.0)
            )
            history.append(entry)

            optim.step()
            scheduler.step()

            if out and cfg.save_every and step % cfg.save_every == 0:
                save_png(img, out / "frames" / f"iter{step:04d}.png")
            if bar is not None and task is not None:
                bar.update(
                    task,
                    advance=1,
                    loss=" ".join(f"{k}={v:.3g}" for k, v in entry.items() if k != "total"),
                )
    finally:
        if bar is not None:
            bar.stop()

    with torch.no_grad():
        region_image = render(scene, seed=cfg.seed).detach()
        initial_word = build_word_scene(shaped, device=device)
        word_initial = render(initial_word, seed=cfg.seed).detach()
        word_scene = build_word_scene(shaped, region=region, region_scene=scene, device=device)
        word_image = render(word_scene, seed=cfg.seed).detach()

    if out:
        out.mkdir(parents=True, exist_ok=True)
        save_png(region_initial, out / "region_init.png")
        save_png(region_image, out / "region_output.png")
        save_png(word_initial, out / "init.png")
        save_png(word_image, out / "output.png")
        save_svg(word_scene, out / "output.svg")
        save_svg(scene, out / "region_output.svg")

    return MorphResult(
        region_scene=scene,
        region_image=region_image,
        region_initial=region_initial,
        word_scene=word_scene,
        word_image=word_image,
        word_initial=word_initial,
        history=history,
        caption=caption,
        region=region,
    )


def save_png(img: torch.Tensor, path: Path) -> None:
    import torchvision.utils as vutils

    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(img.detach().permute(2, 0, 1).cpu().clamp(0, 1), str(path))


def save_svg(scene: GlyphScene, path: Path) -> None:
    """Write a scene as SVG, skipping paths no group references."""
    import pydiffvg

    path.parent.mkdir(parents=True, exist_ok=True)
    used = sorted({int(i) for g in scene.groups for i in g.shape_ids.tolist()})
    remap = {old: new for new, old in enumerate(used)}
    paths = [scene.paths[i] for i in used]
    groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor(
                [remap[int(i)] for i in g.shape_ids.tolist()], dtype=torch.int32
            ),
            fill_color=g.fill_color.detach().cpu(),
            use_even_odd_rule=g.use_even_odd_rule,
        )
        for g in scene.groups
    ]
    pydiffvg.save_svg(str(path), scene.width, scene.height, paths, groups)
