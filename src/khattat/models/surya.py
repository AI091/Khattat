"""Surya OCR loaders.

Pinned to surya-ocr 0.4.15, whose Donut-Swin encoder the OCR loss reads. Later
versions have no separable encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_CHECKPOINT = "vikp/surya_rec"


def load_ocr_encoder(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load the vision encoder of Surya's recognition model, dropping the decoder."""
    from surya.model.recognition.model import load_model

    model = load_model(checkpoint=checkpoint, device=str(device), dtype=dtype)
    encoder = model.encoder
    del model.decoder
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encoder = encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def load_ocr_predictor(*, device: torch.device | str = "cuda"):
    """Load the full recognition model and processor, for evaluation."""
    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor

    model = load_model(device=str(device), dtype=torch.float16)
    processor = load_processor()
    return model, processor
