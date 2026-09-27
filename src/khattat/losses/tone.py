"""Word-as-Image's tone loss, used only for the baseline.

L_tone = w(t) * ||blur(I_init) - blur(I_t)||^2, with w peaking at step 300.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


class ToneLoss(nn.Module):
    def __init__(
        self,
        initial_nchw: torch.Tensor,
        *,
        weight: float = 100.0,
        kernel: int = 201,
        sigma: float = 30.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.blur = GaussianBlur(kernel_size=(kernel, kernel), sigma=sigma)
        with torch.no_grad():
            self.target = self.blur(initial_nchw).detach()

    def schedule(self, step: int) -> float:
        return self.weight * math.exp(-(1 / 5) * ((step - 300) / 20) ** 2)

    def forward(self, img_nchw: torch.Tensor, step: int) -> torch.Tensor:
        return F.mse_loss(self.blur(img_nchw), self.target) * self.schedule(step)
