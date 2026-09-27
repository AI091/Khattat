"""Command-line interface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Khattat: semantic typography with an OCR readability loss.",
)
console = Console()


def _default_out(word: str, concept: str) -> Path:
    import time

    slug = "".join(c if c.isalnum() else "-" for c in concept.lower()).strip("-")
    return Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"


def _resolve_concept(word: str, concept: str | None) -> str:
    from khattat.fonts.db import script_of

    if concept:
        return concept
    if script_of(word) != "latin":
        raise typer.BadParameter(
            "non-Latin words need --concept in English: the diffusion model's text "
            "encoder only understands English prompts."
        )
    return word.lower()


@app.command()
def run(
    word: str,
    concept: Annotated[
        str | None, typer.Option(help="English concept; defaults to the word.")
    ] = None,
    font: Annotated[Path | None, typer.Option(help="Skip FontCLIP and use this font.")] = None,
    region: Annotated[
        tuple[int, int] | None,
        typer.Option(help="Skip region selection: START END (characters, end exclusive)."),
    ] = None,
    target: Annotated[
        list[str] | None, typer.Option(help="Override the LLM's drawable targets (repeatable).")
    ] = None,
    iters: Annotated[int, typer.Option(help="Full-morph iterations.")] = 500,
    lite_iters: Annotated[
        int, typer.Option(help="Iterations per candidate during region selection.")
    ] = 100,
    max_region_len: Annotated[
        int | None, typer.Option(help="Longest substring considered.")
    ] = None,
    single_target: Annotated[
        bool, typer.Option(help="Only search the first target (3x faster selection).")
    ] = False,
    no_ocr: Annotated[bool, typer.Option(help="Disable the OCR loss (ablation).")] = False,
    seed: int = 0,
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
) -> None:
    """Full pipeline: prompt engine, FontCLIP, region selection, morph."""
    from khattat.engine.morph import MorphConfig
    from khattat.engine.pipeline import RunRequest
    from khattat.engine.pipeline import run as run_pipeline

    concept = _resolve_concept(word, concept)
    out = out or _default_out(word, concept)
    req = RunRequest(
        word=word,
        concept=concept,
        output_dir=out,
        font=font,
        region=tuple(region) if region else None,
        targets=target or None,
        lite_iters=lite_iters,
        max_region_len=max_region_len,
        select_target=not single_target,
        morph=MorphConfig(num_iter=iters, seed=seed, use_ocr=not no_ocr, save_every=25),
    )
    rec = run_pipeline(req, log=console.print)
    console.print(
        f"[green]done[/green] -> {out}/output.svg  "
        f"(region {rec.region_text!r}, target {rec.target!r}, font {Path(rec.font).name})"
    )


@app.command("morph")
def morph_cmd(
    word: str,
    concept: Annotated[str, typer.Option(help="Drawable target that drives SDS, e.g. 'wings'.")],
    font: Annotated[Path, typer.Option(exists=True)],
    region: Annotated[
        tuple[int, int], typer.Option(help="START END over characters, end exclusive.")
    ],
    iters: int = 500,
    no_ocr: bool = False,
    no_acap: bool = False,
    seed: int = 0,
    save_every: int = 25,
    out: Path | None = None,
) -> None:
    """Single morph with a fixed font, region and target."""
    from khattat.engine.morph import MorphConfig, morph
    from khattat.models.sd import load_sd_pipeline
    from khattat.models.surya import load_ocr_encoder

    out = out or _default_out(word, concept)
    pipe = load_sd_pipeline()
    enc = None if no_ocr else load_ocr_encoder()
    cfg = MorphConfig(
        num_iter=iters, use_ocr=not no_ocr, use_acap=not no_acap, seed=seed, save_every=save_every
    )
    res = morph(
        word=word,
        concept=concept,
        font_path=font,
        region=tuple(region),
        pipe=pipe,
        ocr_encoder=enc,
        config=cfg,
        output_dir=out,
    )
    import json
    from dataclasses import asdict

    (out / "run.json").write_text(
        json.dumps(
            {
                "word": word,
                "concept": concept,
                "font": str(font),
                "region": list(region),
                "region_text": word[region[0] : region[1]],
                "target": concept,
                "final_losses": res.final_losses,
                "config": asdict(cfg),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (out / "losses.json").write_text(json.dumps(res.history))
    console.print(f"[green]done[/green] -> {out}/output.svg  final {res.final_losses}")


@app.command()
def plan(
    concept: str,
    refresh: bool = False,
    backend: Annotated[
        str | None,
        typer.Option(help="ollama:<model>, hf:<repo>, gemini:<model>, anthropic:<model>, none"),
    ] = None,
) -> None:
    """Run the prompt engine only."""
    from khattat.engine.prompts import PromptEngine

    engine = PromptEngine(backend=backend)
    p = engine.plan(concept, refresh=refresh)
    console.print(f"source: {p.source}\ntargets: {p.targets}\nattributes: {p.attributes}")
    if engine.last_error:
        console.print(f"[yellow]backend error:[/yellow] {engine.last_error}")


@app.command()
def fonts(
    attributes: Annotated[list[str], typer.Option(help="Font attributes (repeatable).")],
    script: str = "latin",
    k: int = 10,
) -> None:
    """Rank installed fonts for a set of attributes with FontCLIP."""
    from khattat.engine.font_select import FontIndex
    from khattat.models.fontclip import FontCLIP, attribute_prompt

    fc = FontCLIP.load()
    index = FontIndex.build(fc, script=script)
    prompt = attribute_prompt(attributes)
    table = Table(title=f"{prompt!r}  ({len(index.fonts)} {script} families)")
    for col in ("rank", "sim", "family", "file"):
        table.add_column(col)
    for i, m in enumerate(index.rank(fc, prompt, k=k), 1):
        table.add_row(str(i), f"{m.similarity:+.4f}", m.font.family, m.font.path.name)
    console.print(table)


@app.command()
def colorize(
    image: Annotated[Path, typer.Argument(exists=True)],
    concept: Annotated[str, typer.Option()],
    strength: float = 0.7,
    seed: int = 0,
) -> None:
    """Colour an output with SD2 depth-to-image."""
    from khattat.engine.postprocess import colorize as run_colorize

    console.print(
        f"[green]wrote[/green] {run_colorize(image, concept, strength=strength, seed=seed)}"
    )


@app.command()
def evaluate(
    runs: Annotated[list[Path], typer.Argument(help="Run directories containing run.json.")],
    image: Annotated[str, typer.Option(help="Image file inside each run dir.")] = "output.png",
    out: Path | None = None,
) -> None:
    """OCR accuracy and CLIPScore over finished runs."""
    import json

    from khattat.eval.metrics import evaluate_runs, summarise

    summary = summarise(evaluate_runs(runs, image_name=image))
    table = Table(title=f"{summary['runs']} runs")
    for col in ("run", "word", "read as", "exact", "char acc", "ocr conf", "CLIP"):
        table.add_column(col)
    for s in summary["per_run"]:
        table.add_row(
            Path(s["run"]).name,
            s["word"],
            s["ocr_text"],
            "yes" if s["ocr_exact"] else "no",
            f"{s['char_accuracy']:.2f}",
            f"{s['ocr_confidence']:.2f}",
            f"{s['clip_score']:.3f}",
        )
    console.print(table)
    console.print(
        f"OCR accuracy {summary['ocr_accuracy']:.2f}   char accuracy "
        f"{summary['char_accuracy']:.2f}   confidence {summary['ocr_confidence']:.2f}   "
        f"CLIPScore {summary['clip_score']:.3f}"
    )
    if out:
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def doctor() -> None:
    """Check that every component loads."""
    ok = True

    def check(name: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            console.print(f"[green]ok[/green]    {name}  {detail or ''}")
        except Exception as exc:
            ok = False
            console.print(f"[red]FAIL[/red]  {name}  {type(exc).__name__}: {exc}")

    def _torch():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        free, total = torch.cuda.mem_get_info()
        return (
            f"torch {torch.__version__}, {torch.cuda.get_device_name()}, "
            f"{free / 1e9:.1f}/{total / 1e9:.1f} GB free"
        )

    def _diffvg():
        import pydiffvg
        import torch

        pydiffvg.set_use_gpu(True)
        pts = torch.tensor(
            [[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]], device="cuda", requires_grad=True
        )
        path = pydiffvg.Path(num_control_points=torch.tensor([0, 0, 0]), points=pts, is_closed=True)
        grp = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
        )
        args = pydiffvg.RenderFunction.serialize_scene(100, 100, [path], [grp])
        pydiffvg.RenderFunction.apply(100, 100, 2, 2, 0, None, *args)[:, :, 3].sum().backward()
        assert pts.grad is not None and pts.grad.abs().sum() > 0
        return "render + backward"

    def _harfbuzz():
        from khattat.fonts.db import discover_fonts
        from khattat.fonts.shaping import shape_word

        ar = discover_fonts(script="arabic")
        if not ar:
            return "no Arabic font installed (Latin only)"
        w = shape_word("حرية", ar[0].path)
        return f"Arabic shaping ok ({len(w.visible())} glyphs, {w.direction})"

    def _surya():
        from khattat.models.surya import load_ocr_encoder

        load_ocr_encoder()
        return "vikp/surya_rec encoder"

    def _sd():
        from khattat.models.sd import load_sd_pipeline

        pipe = load_sd_pipeline()
        return pipe.name_or_path

    def _fontclip():
        from khattat.models.fontclip import FontCLIP

        FontCLIP.load()
        return "checkpoint loaded, LoRA merged"

    def _llm():
        from khattat.engine.prompts import resolve_backend

        b = resolve_backend()
        if b is None:
            return "disabled (KHATTAT_LLM=none)"
        if not b.available():
            raise RuntimeError(f"{b.name} backend not available")
        out = b.complete("Reply with the single word: ok")
        b.close()
        return f"{b.name}: {getattr(b, 'model', None) or b.model_id} replied {out.strip()[:20]!r}"

    for name, fn in [
        ("torch/CUDA", _torch),
        ("diffvg", _diffvg),
        ("harfbuzz", _harfbuzz),
        ("surya encoder", _surya),
        ("stable diffusion", _sd),
        ("fontclip", _fontclip),
        ("prompt engine", _llm),
    ]:
        check(name, fn)
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
