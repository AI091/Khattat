"""Region selection (paper §4.4): argmax over substrings of 0.5 * -L_OCR + 0.5 * CLIPScore.

Candidates are (region, target) pairs, so the search also picks the target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from khattat.engine.morph import MorphConfig, morph, save_png
from khattat.fonts.shaping import ShapedWord
from khattat.losses.ocr import OCRLoss
from khattat.losses.sds import build_caption, encode_captions
from khattat.render.scene import region_glyph_indices, to_nchw

LAMBDA = 0.5


@dataclass
class Candidate:
    region: tuple[int, int]
    target: str
    substring: str
    clip_score: float
    ocr_loss: float
    score: float


def candidate_regions(
    word: ShapedWord, *, min_len: int = 1, max_len: int | None = None
) -> list[tuple[int, int]]:
    """All contiguous character ranges that contain at least one visible glyph."""
    n = len(word.text)
    max_len = max_len or n
    out = []
    for length in range(min_len, min(max_len, n) + 1):
        for start in range(0, n - length + 1):
            region = (start, start + length)
            if word.text[start : start + length].strip() and region_glyph_indices(word, region):
                out.append(region)
    return out


def select_region(
    *,
    word: ShapedWord,
    concept: str,
    targets: list[str],
    pipe,
    ocr_encoder,
    clip_scorer,
    lite_iters: int = 100,
    lam: float = LAMBDA,
    max_len: int | None = None,
    base_config: MorphConfig | None = None,
    device: torch.device | str = "cuda",
    output_dir: Path | None = None,
    log=print,
) -> tuple[Candidate, list[Candidate]]:
    """Score every (region, target) pair with a short morph and return the best.

    CLIPScore is measured against the concept, so different targets are comparable.
    """
    base = base_config or MorphConfig()
    cfg = replace(base, num_iter=lite_iters, save_every=0)

    # The morphs below free the text encoder, so embed every caption first.
    encode_captions(pipe, [build_caption(t) for t in targets], device=device, free_encoder=True)

    regions = candidate_regions(word, max_len=max_len)
    total = len(regions) * len(targets)
    concept_prompt = f"a {concept}"
    results: list[Candidate] = []

    k = 0
    for target in targets:
        for region in regions:
            k += 1
            res = morph(
                word=word,
                concept=target,
                region=region,
                pipe=pipe,
                ocr_encoder=ocr_encoder,
                config=cfg,
                device=device,
                progress=False,
            )
            clip = clip_scorer.score(res.region_image, concept_prompt)
            with torch.no_grad():
                ocr_fn = OCRLoss(ocr_encoder, to_nchw(res.region_initial), device=device)
                ocr_val = float(ocr_fn(to_nchw(res.region_image)))
            score = lam * (-ocr_val) + (1.0 - lam) * clip
            cand = Candidate(region, target, word.text[region[0] : region[1]], clip, ocr_val, score)
            results.append(cand)
            log(
                f"[{k}/{total}] {cand.substring!r:>12} x {target!r:<16} "
                f"clip={clip:.4f} ocr={ocr_val:.4f} score={score:+.4f}"
            )
            if output_dir:
                save_png(
                    res.region_image,
                    Path(output_dir) / f"{region[0]}-{region[1]}_{_slug(target)}.png",
                )
            del res
            torch.cuda.empty_cache()

    best = max(results, key=lambda c: c.score)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "scores.json").write_text(
            json.dumps([asdict(c) for c in results], ensure_ascii=False, indent=2)
        )
    return best, results


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")
