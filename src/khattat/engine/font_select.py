"""Font selection with FontCLIP (paper §4.3, Algorithm 1).

FontCLIP's image-text cosines are near zero; only their ranking is meaningful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from khattat.fonts.db import FontEntry, discover_fonts, render_specimen
from khattat.models.fontclip import FontCLIP, attribute_prompt

DEFAULT_CACHE = Path.home() / ".cache" / "khattat" / "font_db"


@dataclass
class FontMatch:
    font: FontEntry
    similarity: float


def dedupe_families(fonts: list[FontEntry], *, prefer: str = "Regular") -> list[FontEntry]:
    """Keep one face per family, preferring the Regular weight."""
    by_family: dict[str, list[FontEntry]] = {}
    for f in fonts:
        by_family.setdefault(f.family, []).append(f)
    out = []
    for family in sorted(by_family):
        faces = by_family[family]
        exact = [f for f in faces if f.name.endswith(f"-{prefer}")]
        out.append(exact[0] if exact else min(faces, key=lambda f: len(f.name)))
    return out


class FontIndex:
    """Cached FontCLIP embeddings for a set of fonts in one script."""

    def __init__(self, fonts: list[FontEntry], embeddings: torch.Tensor, script: str) -> None:
        self.fonts = fonts
        self.embeddings = embeddings
        self.script = script

    @classmethod
    def build(
        cls,
        model: FontCLIP,
        *,
        script: str,
        font_dirs: list[Path] | None = None,
        cache_dir: Path = DEFAULT_CACHE,
        aug_num: int = 0,
        one_per_family: bool = True,
    ) -> FontIndex:
        """Load cached embeddings or embed every font.

        `aug_num=0` uses one centre crop per specimen instead of averaged augmentations.
        """
        fonts = discover_fonts(font_dirs, script=script)
        if one_per_family:
            fonts = dedupe_families(fonts)
        if not fonts:
            raise RuntimeError(f"no fonts found that support script {script!r}")

        key = hashlib.sha1(
            json.dumps([str(f.path) for f in fonts] + [script, aug_num]).encode()
        ).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{script}-{key}.pt"
        if cache_file.exists():
            blob = torch.load(cache_file, map_location="cpu", weights_only=False)
            return cls(fonts, blob["embeddings"].to(model.device), script)

        feats = []
        batch = 32
        for i in range(0, len(fonts), batch):
            imgs = [render_specimen(f, script=script) for f in fonts[i : i + batch]]
            if aug_num:
                feats.append(model.encode_specimens(imgs, aug_num=aug_num))
            else:
                feats.append(model.encode_images(imgs))
        embeddings = torch.cat(feats)
        torch.save(
            {"embeddings": embeddings.cpu(), "fonts": [str(f.path) for f in fonts]}, cache_file
        )
        return cls(fonts, embeddings, script)

    def rank(self, model: FontCLIP, prompt: str, k: int = 5) -> list[FontMatch]:
        query = model.encode_text([prompt])
        sims = (self.embeddings @ query.T).squeeze(1)
        top = sims.topk(min(k, len(self.fonts))).indices.tolist()
        return [FontMatch(self.fonts[i], float(sims[i])) for i in top]


def select_font(model: FontCLIP, index: FontIndex, attributes: list[str]) -> FontMatch:
    """Paper Algorithm 1."""
    return index.rank(model, attribute_prompt(attributes), k=1)[0]
