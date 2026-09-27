"""CLIPScore with OpenAI CLIP ViT-B/32 (paper eq. 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPScorer:
    def __init__(self, model, tokenizer, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, *, device: torch.device | str = "cuda") -> CLIPScorer:
        import open_clip

        device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, open_clip.get_tokenizer("ViT-B-32"), device)

    @torch.no_grad()
    def text(self, prompts: list[str]) -> torch.Tensor:
        feats = self.model.encode_text(self.tokenizer(prompts).to(self.device)).float()
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(
            img.to(self.device).float(),
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        mean = x.new_tensor(_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_STD).view(1, 3, 1, 1)
        feats = self.model.encode_image((x - mean) / std).float()
        return F.normalize(feats, dim=-1)

    def score(self, img: torch.Tensor, prompt: str) -> float:
        return float((self.image(img) @ self.text([prompt]).T).squeeze())
