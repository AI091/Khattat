"""Stable Diffusion loaders."""

from __future__ import annotations

import torch

# Tried in order. Stability removed the original SD2 repositories.
MIRRORS = [
    "Manojb/stable-diffusion-2-1-base",
    "LanguageMachines/stable-diffusion-2-1-base",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
]

DEPTH_MIRRORS = [
    "sd2-community/stable-diffusion-2-depth",
    "radames/stable-diffusion-2-depth-img2img",
]


def load_sd_pipeline(
    model_id: str | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    low_memory: bool = True,
    free_decoder: bool = True,
):
    """Load Stable Diffusion for SDS. `free_decoder` drops the unused VAE decoder."""
    from diffusers import StableDiffusionPipeline

    candidates = [model_id] if model_id else MIRRORS
    errors: list[str] = []
    for repo in candidates:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                repo, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
            )
            break
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("could not load any Stable Diffusion checkpoint.\n" + "\n".join(errors))

    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    if low_memory:
        # Attention slicing would replace SDPA with a slower, larger processor.
        pipe.vae.enable_slicing()
    if free_decoder:
        pipe.vae.decoder = None
        pipe.vae.post_quant_conv = None
        torch.cuda.empty_cache()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_depth_pipeline(
    model_id: str | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load SD2 depth-to-image for optional colouring."""
    from diffusers import StableDiffusionDepth2ImgPipeline

    candidates = [model_id] if model_id else DEPTH_MIRRORS
    errors: list[str] = []
    for repo in candidates:
        try:
            pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(repo, torch_dtype=dtype)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load a depth-to-image checkpoint.\n" + "\n".join(errors))
