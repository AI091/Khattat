"""End-to-end pipeline: prompt engine, font selection, region selection, morph."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from khattat.engine.morph import MorphConfig, morph
from khattat.engine.prompts import ConceptPlan, PromptEngine
from khattat.fonts.db import script_of
from khattat.fonts.shaping import shape_word


@dataclass
class RunRequest:
    word: str
    concept: str
    output_dir: Path
    font: Path | None = None
    region: tuple[int, int] | None = None
    targets: list[str] | None = None
    """Overrides the prompt engine's targets."""
    lite_iters: int = 100
    max_region_len: int | None = None
    select_target: bool = True
    """Search every target during region selection. If False, keep only the target
    closest to the concept in CLIP text space."""
    font_dirs: list[Path] | None = None
    morph: MorphConfig = field(default_factory=MorphConfig)
    device: str = "cuda"


@dataclass
class RunRecord:
    """Written to run.json."""

    word: str
    concept: str
    plan: dict
    font: str
    font_similarity: float | None
    region: tuple[int, int]
    region_text: str
    target: str
    selection: list[dict] | None
    final_losses: dict
    seconds: dict
    config: dict


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(req: RunRequest, *, engine: PromptEngine | None = None, log=print) -> RunRecord:
    out = Path(req.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    script = script_of(req.word)

    t = time.time()
    engine = engine or PromptEngine()
    plan: ConceptPlan = engine.plan(req.concept)
    targets = list(dict.fromkeys(req.targets or plan.targets))  # dedupe, keep order
    if not req.select_target and len(targets) > 1:
        from khattat.models.clipscore import CLIPScorer

        scorer = CLIPScorer.load(device="cpu")
        sims = (
            (scorer.text([f"a {t}" for t in targets]) @ scorer.text([f"a {req.concept}"]).T)
            .squeeze(1)
            .tolist()
        )
        ranked = sorted(zip(targets, sims, strict=True), key=lambda x: -x[1])
        log("target ranking: " + ", ".join(f"{t} {s:.3f}" for t, s in ranked))
        targets = [ranked[0][0]]
        del scorer
    timings["prompt_engine"] = time.time() - t
    log(f"plan ({plan.source}): targets={targets} attributes={plan.attributes}")
    if plan.source == "fallback":
        log(
            "warning: no LLM backend worked; the concept stands in for its own imagery. "
            f"({engine.last_error or 'no backend available'}) See KHATTAT_LLM in engine/prompts.py."
        )

    t = time.time()
    font_similarity = None
    if req.font:
        font_path = Path(req.font)
    else:
        from khattat.engine.font_select import FontIndex, select_font
        from khattat.models.fontclip import FontCLIP

        fc = FontCLIP.load(device=req.device)
        index = FontIndex.build(fc, script=script, font_dirs=req.font_dirs)
        match = select_font(fc, index, plan.attributes)
        font_path, font_similarity = match.font.path, match.similarity
        del fc, index
        _free()
    timings["font_selection"] = time.time() - t
    log(f"font: {font_path.name}")

    shaped = shape_word(req.word, font_path, canvas_size=req.morph.render_size)

    t = time.time()
    from khattat.models.sd import load_sd_pipeline
    from khattat.models.surya import load_ocr_encoder

    pipe = load_sd_pipeline(device=req.device)
    ocr_encoder = load_ocr_encoder(device=req.device) if req.morph.use_ocr else None
    timings["model_load"] = time.time() - t

    t = time.time()
    selection = None
    if req.region is not None:
        region, target = tuple(req.region), targets[0]
    else:
        from khattat.engine.region_select import select_region
        from khattat.models.clipscore import CLIPScorer

        scorer = CLIPScorer.load(device="cpu")
        best, cands = select_region(
            word=shaped,
            concept=req.concept,
            targets=targets,
            pipe=pipe,
            ocr_encoder=ocr_encoder,
            clip_scorer=scorer,
            lite_iters=req.lite_iters,
            max_len=req.max_region_len,
            base_config=req.morph,
            device=req.device,
            output_dir=out / "selection",
            log=log,
        )
        region, target = best.region, best.target
        selection = [asdict(c) for c in cands]
        del scorer
        _free()
    timings["region_selection"] = time.time() - t
    log(f"region: {region} {req.word[region[0] : region[1]]!r}  target: {target!r}")

    t = time.time()
    result = morph(
        word=shaped,
        concept=target,
        region=region,
        pipe=pipe,
        ocr_encoder=ocr_encoder,
        config=req.morph,
        device=req.device,
        output_dir=out,
        progress=True,
    )
    timings["morph"] = time.time() - t

    record = RunRecord(
        word=req.word,
        concept=req.concept,
        plan=asdict(plan),
        font=str(font_path),
        font_similarity=font_similarity,
        region=region,
        region_text=req.word[region[0] : region[1]],
        target=target,
        selection=selection,
        final_losses=result.final_losses,
        seconds={k: round(v, 1) for k, v in timings.items()},
        config=asdict(req.morph),
    )
    (out / "run.json").write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2))
    (out / "losses.json").write_text(json.dumps(result.history))
    del pipe, ocr_encoder, result
    _free()
    return record
