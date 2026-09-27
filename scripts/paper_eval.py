"""Compare Khattat and Word-as-Image on a word list.

For each word the automatic pipeline picks the font, region and target, then
both methods run on that choice. `words.tsv` has one `word<TAB>concept` per
line. Finished runs are skipped. Scores go to `<out>/summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from khattat.engine.morph import MorphConfig, morph, word_as_image_config
from khattat.engine.pipeline import RunRequest, run

METHODS = {
    "khattat": lambda base: base,
    "word_as_image": lambda base: word_as_image_config(
        **{k: v for k, v in asdict(base).items() if k not in {"use_ocr", "use_tone", "sds"}},
        sds=base.sds,
    ),
}


def read_words(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, concept = line.partition("\t")
        out.append((word.strip(), (concept or word).strip().lower()))
    return out


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-") or "w"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--single-target", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lite-iters", type=int, default=100)
    ap.add_argument("--max-region-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N words.")
    args = ap.parse_args()

    words = read_words(args.words)[: args.limit]
    for i, (word, concept) in enumerate(words):
        base_dir = args.out / f"{i:02d}-{slug(concept)}"
        auto = base_dir / "khattat-seed0"

        if not (auto / "run.json").exists():
            run(
                RunRequest(
                    word=word,
                    concept=concept,
                    output_dir=auto,
                    select_target=not args.single_target,
                    lite_iters=args.lite_iters,
                    max_region_len=args.max_region_len,
                    morph=MorphConfig(num_iter=args.iters, seed=0),
                )
            )
        chosen = json.loads((auto / "run.json").read_text())
        region = tuple(chosen["region"])

        for method, make_cfg in METHODS.items():
            for seed in args.seeds:
                d = base_dir / f"{method}-seed{seed}"
                if (d / "run.json").exists() or (method == "khattat" and seed == 0):
                    continue  # khattat-seed0 is the automatic run itself
                cfg = replace(make_cfg(MorphConfig(num_iter=args.iters)), seed=seed)
                from khattat.models.sd import load_sd_pipeline
                from khattat.models.surya import load_ocr_encoder

                pipe = load_sd_pipeline()
                enc = load_ocr_encoder() if cfg.use_ocr else None
                res = morph(
                    word=word,
                    concept=chosen["target"],
                    font_path=chosen["font"],
                    region=region,
                    pipe=pipe,
                    ocr_encoder=enc,
                    config=cfg,
                    output_dir=d,
                    progress=False,
                )
                (d / "run.json").write_text(
                    json.dumps(
                        {
                            **chosen,
                            "method": method,
                            "seed": seed,
                            "final_losses": res.final_losses,
                            "config": asdict(cfg),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                del pipe, enc, res
                torch.cuda.empty_cache()

    from khattat.eval.metrics import evaluate_runs, summarise

    summary = {}
    for method in METHODS:
        dirs = sorted(args.out.glob(f"*/{method}-seed*"))
        dirs = [d for d in dirs if (d / "run.json").exists()]
        if dirs:
            summary[method] = summarise(evaluate_runs(dirs))
            s = summary[method]
            for label, v in (("all", s), ("readable-font", s["readable_subset"])):
                print(
                    f"{method:14s} {label:13s} n={v['runs']:3d}  OCR acc {v['ocr_accuracy']:.2f}  "
                    f"char acc {v['char_accuracy']:.2f}  conf {v['ocr_confidence']:.2f}  "
                    f"CLIP {v['clip_score']:.3f}"
                )
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
