"""OCR readability loss (paper eq. 5): L_OCR = ||E(I_orig) - E(I_curr)||^2.

E is the last hidden layer of Surya's recognition encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Surya 0.4.15 recognition input size and normalisation.
OCR_HEIGHT = 196
OCR_WIDTH = 896
OCR_MEAN = 0.5
OCR_STD = 0.5
OCR_PAD_VALUE = 1.0  # white, i.e. 255 after rescaling by 1/255


def differentiable_preprocess(img: torch.Tensor) -> torch.Tensor:
    """Differentiable version of Surya's recognition preprocessing.

    Resize into 196x896 keeping aspect, pad with white, normalise to [-1, 1].
    Uses bicubic where Surya uses Lanczos.
    """
    if img.dim() != 4:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(img.shape)}")

    _, _, h, w = img.shape
    scale = min(OCR_WIDTH / w, OCR_HEIGHT / h)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    resized = F.interpolate(
        img, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True
    )
    resized = resized.clamp(0.0, 1.0)

    pad_h = OCR_HEIGHT - new_h
    pad_w = OCR_WIDTH - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    padded = F.pad(
        resized,
        (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
        mode="constant",
        value=OCR_PAD_VALUE,
    )
    return (padded - OCR_MEAN) / OCR_STD


class OCRLoss(nn.Module):
    """MSE between OCR encoder features of the original and current render."""

    def __init__(
        self,
        encoder: nn.Module,
        reference: torch.Tensor,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            self.reference_features = self._encode(reference).detach().float()

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        pixel_values = differentiable_preprocess(img).to(self.dtype)
        return self.encoder(pixel_values=pixel_values).last_hidden_state

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self._encode(img).float()
        return F.mse_loss(features, self.reference_features.expand_as(features))


def ocr_loss_weight(num_morphed_letters: int, base: float = 0.5) -> float:
    """`0.5 * n` for n morphed letters (paper §4.5)."""
    return base * max(1, num_morphed_letters)
