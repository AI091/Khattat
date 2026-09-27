"""FontCLIP (Tatsukawa et al. 2024) for font retrieval.

The released checkpoint stores its text-tower LoRA adapters unmerged
(q, k, v and output projections, rank 256, alpha 1024).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# Upstream `model.pt` on Google Drive (MIT).
CHECKPOINT_GDRIVE_ID = "1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1"
CHECKPOINT_FILENAME = "fontclip_model.pt"

MODEL_NAME = "ViT-B-32"


@dataclass
class LoRASpec:
    rank: int = 256
    alpha: float = 1024.0

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_signature(cls, signature: str) -> LoRASpec:
        """Parse `lora_t-qkvo_{rank}-{alpha}` from the checkpoint signature."""
        import re

        m = re.search(r"lora_(?:v-\w+_[\d.]+-[\d.]+_)?t-\w+_(\d+)-([\d.]+)", signature)
        if not m:
            return cls()
        return cls(rank=int(m.group(1)), alpha=float(m.group(2)))


def merge_lora(state_dict: dict[str, torch.Tensor], spec: LoRASpec) -> dict[str, torch.Tensor]:
    """Fold LoRA adapters into the base weights.

    q, k, v adapters act on the projection input:  W += s (A B)^T.
    The output adapter acts on the projection output:  W_o, b_o <- (I + s A B)^T (W_o, b_o).
    """
    sd = dict(state_dict)
    scale = spec.scale

    prefixes = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith("_lora_proj_weight_a")})
    for prefix in prefixes:
        in_proj_key = f"{prefix}.in_proj_weight"
        if in_proj_key in sd:
            w = sd[in_proj_key]
            dim = w.shape[1]
            for slot, (lo, hi) in enumerate([(0, dim), (dim, 2 * dim), (2 * dim, 3 * dim)]):
                name = ("q", "k", "v")[slot]
                a_key = f"{prefix}.{name}_lora_proj_weight_a"
                b_key = f"{prefix}.{name}_lora_proj_weight_b"
                if a_key not in sd:
                    continue
                a = sd[a_key].to(torch.float32)
                b = sd[b_key].to(torch.float32)
                delta = (a @ b) * scale  # (dim, dim), acts on the input
                w[lo:hi, :] = (w[lo:hi, :].to(torch.float32) + delta.T).to(w.dtype)
            sd[in_proj_key] = w

        a_key = f"{prefix}.out_lora_proj_weight_a"
        b_key = f"{prefix}.out_lora_proj_weight_b"
        out_w_key = f"{prefix}.out_proj.weight"
        out_b_key = f"{prefix}.out_proj.bias"
        if a_key in sd and out_w_key in sd:
            a = sd[a_key].to(torch.float32)
            b = sd[b_key].to(torch.float32)
            dim = a.shape[0]
            m = torch.eye(dim, dtype=torch.float32) + scale * (a @ b)
            w_out = sd[out_w_key]
            sd[out_w_key] = (m.T @ w_out.to(torch.float32)).to(w_out.dtype)
            if out_b_key in sd:
                b_out = sd[out_b_key]
                sd[out_b_key] = (m.T @ b_out.to(torch.float32)).to(b_out.dtype)

    for key in [k for k in sd if "_lora_proj_" in k]:
        del sd[key]
    return sd


def default_checkpoint_path() -> Path:
    return Path.home() / ".cache" / "khattat" / CHECKPOINT_FILENAME


def download_checkpoint(dest: Path | None = None) -> Path:
    """Download the FontCLIP checkpoint if missing."""
    dest = dest or default_checkpoint_path()
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "gdown is required to download the FontCLIP checkpoint: pip install gdown"
        ) from exc
    gdown.download(id=CHECKPOINT_GDRIVE_ID, output=str(dest), quiet=False)
    return dest


class FontCLIP:
    """CLIP ViT-B/32 with FontCLIP's LoRA merged in."""

    def __init__(self, model, preprocess, device: torch.device) -> None:
        self.model = model
        self.preprocess = preprocess
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint: str | Path | None = None,
        *,
        device: torch.device | str = "cuda",
    ) -> FontCLIP:
        import open_clip

        device = torch.device(device)
        path = Path(checkpoint) if checkpoint else default_checkpoint_path()
        if not path.exists():
            path = download_checkpoint(path)

        blob = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = blob.get("model_state_dict", blob)
        spec = LoRASpec.from_signature(str(blob.get("signature", "")))
        merged = merge_lora(state_dict, spec)

        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=None)
        missing, unexpected = model.load_state_dict(merged, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys after LoRA merge: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"missing keys in FontCLIP checkpoint: {missing[:5]}")

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, preprocess, device)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        import open_clip

        tokens = open_clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """One centre crop per image."""
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(batch).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_specimens(self, images: list, *, aug_num: int = 32) -> torch.Tensor:
        """Average embeddings over `aug_num` augmented crops, as FontCLIP's retrieval demo does."""
        transform = _augment_transform()
        out = []
        for img in images:
            small = _shrink_for_crop(img)
            batch = torch.stack([transform(small) for _ in range(aug_num)]).to(self.device)
            feats = self.model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            out.append(mean / mean.norm())
        return torch.stack(out)


def _shrink_for_crop(img, *, min_side: int = 320):
    """Downscale so the short side is `min_side`."""
    w, h = img.size
    short = min(w, h)
    if short <= min_side:
        return img
    scale = min_side / short
    from PIL import Image as _Image

    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), _Image.Resampling.BICUBIC)


def _augment_transform():
    """FontCLIP's `my_transform(lower_bound_of_scale=0.3)`."""
    from torchvision.transforms import (
        Compose,
        InterpolationMode,
        Normalize,
        RandomResizedCrop,
        RandomRotation,
        ToTensor,
    )

    bicubic = InterpolationMode.BICUBIC
    return Compose(
        [
            RandomRotation(180, fill=255, interpolation=bicubic),
            RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(1.0, 1.0), interpolation=bicubic),
            lambda im: im.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def attribute_prompt(attributes: list[str]) -> str:
    return f"This is a {', '.join(attributes)} font"
