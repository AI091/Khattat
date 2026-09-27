"""OCR accuracy and CLIPScore, as in the paper's Table 1."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

_LANG = {"latin": "en", "arabic": "ar"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return "".join(ch for ch in text if not ch.isspace() and unicodedata.category(ch)[0] != "P")


def char_accuracy(pred: str, truth: str) -> float:
    """1 - normalised Levenshtein distance, floored at 0."""
    a, b = normalise(pred), normalise(truth)
    if not b:
        return float(a == b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(b))


def crop_to_ink(img: Image.Image, *, pad_frac: float = 0.5, line_height: int = 64) -> Image.Image:
    """Crop to the ink, pad by `pad_frac` of the text height, scale to `line_height`.

    Surya misreads tightly cropped renders, even of undeformed words.
    """
    gray = ImageOps.invert(img.convert("L")).point(lambda v: 255 if v > 32 else 0)
    box = gray.getbbox()
    if not box:
        return img
    _, top, _, bottom = box
    ink = img.convert("RGB").crop(box)
    pad = int(pad_frac * (bottom - top))
    framed = Image.new("RGB", (ink.width + 2 * pad, ink.height + 2 * pad), "white")
    framed.paste(ink, (pad, pad))
    scale = line_height / framed.height
    return framed.resize(
        (max(1, round(framed.width * scale)), line_height), Image.Resampling.LANCZOS
    )


class OCRReader:
    def __init__(self) -> None:
        from khattat.models.surya import load_ocr_predictor

        self.model, self.processor = load_ocr_predictor()

    def read(self, images: list[Image.Image], scripts: list[str]) -> list[tuple[str, float]]:
        """(text, confidence) per image."""
        from surya.recognition import batch_recognition

        crops = [crop_to_ink(im) for im in images]
        langs = [[_LANG.get(s, "en")] for s in scripts]
        texts, confidences = batch_recognition(crops, langs, self.model, self.processor)
        return list(zip(texts, confidences, strict=True))


@dataclass
class RunScore:
    run: str
    word: str
    concept: str
    ocr_text: str
    ocr_confidence: float
    ocr_exact: bool
    char_accuracy: float
    clip_score: float
    init_readable: bool | None = None
    """Whether the undeformed word reads correctly."""


def evaluate_runs(run_dirs: list[Path], *, image_name: str = "output.png") -> list[RunScore]:
    """Score run directories containing `run.json` and the output image."""
    import torch
    from torchvision.transforms.functional import to_tensor

    from khattat.fonts.db import script_of
    from khattat.models.clipscore import CLIPScorer

    runs = []
    for d in map(Path, run_dirs):
        meta = json.loads((d / "run.json").read_text())
        runs.append((d, meta, Image.open(d / image_name).convert("RGB")))

    reader = OCRReader()
    scripts = [script_of(m["word"]) for _, m, _ in runs]
    texts = reader.read([im for _, _, im in runs], scripts)
    inits = [(d / "init.png") for d, _, _ in runs]
    init_reads: list[bool | None] = [None] * len(runs)
    have = [i for i, p in enumerate(inits) if p.exists()]
    if have:
        reads = reader.read(
            [Image.open(inits[i]).convert("RGB") for i in have], [scripts[i] for i in have]
        )
        for i, (text, _) in zip(have, reads, strict=True):
            init_reads[i] = normalise(text) == normalise(runs[i][1]["word"])
    del reader
    torch.cuda.empty_cache()

    scorer = CLIPScorer.load()
    out = []
    for (d, meta, im), (text, conf), init_ok in zip(runs, texts, init_reads, strict=True):
        clip = scorer.score(to_tensor(im).permute(1, 2, 0), f"a {meta['concept']}")
        out.append(
            RunScore(
                run=str(d),
                word=meta["word"],
                concept=meta["concept"],
                ocr_text=text,
                ocr_confidence=conf,
                ocr_exact=normalise(text) == normalise(meta["word"]),
                char_accuracy=char_accuracy(text, meta["word"]),
                clip_score=clip,
                init_readable=init_ok,
            )
        )
    return out


def summarise(scores: list[RunScore]) -> dict:
    def agg(xs: list[RunScore]) -> dict:
        n = len(xs) or 1
        return {
            "runs": len(xs),
            "ocr_accuracy": sum(s.ocr_exact for s in xs) / n,
            "char_accuracy": sum(s.char_accuracy for s in xs) / n,
            "ocr_confidence": sum(s.ocr_confidence for s in xs) / n,
            "clip_score": sum(s.clip_score for s in xs) / n,
        }

    readable = [s for s in scores if s.init_readable is not False]
    return {
        **agg(scores),
        "readable_subset": agg(readable),
        "excluded_unreadable_fonts": len(scores) - len(readable),
        "per_run": [asdict(s) for s in scores],
    }
