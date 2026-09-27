"""Score distillation sampling (paper eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import kornia.augmentation as K
import torch
import torch.nn as nn

# VectorFusion's prompt suffix, used by the paper.
DEFAULT_PROMPT_SUFFIX = "minimal flat 2d vector. lineal color. trending on artstation"

NEGATIVE_PROMPT = ""


def build_caption(concept: str, suffix: str = DEFAULT_PROMPT_SUFFIX) -> str:
    return f"a {concept}. {suffix}"


@torch.no_grad()
def encode_captions(
    pipe, captions: list[str], *, device: torch.device | str = "cuda", free_encoder: bool = False
) -> dict[str, torch.Tensor]:
    """Embed captions as `(2, 77, D)` [uncond, cond] pairs, cached on `pipe`.

    With `free_encoder=True` the text encoder is released afterwards.
    """
    cache: dict[str, torch.Tensor] = getattr(pipe, "_khattat_caption_cache", {})
    missing = [c for c in captions if c not in cache]
    if missing:
        if pipe.text_encoder is None:
            raise RuntimeError(
                f"text encoder was freed and {missing!r} is not cached; "
                "call encode_captions with every caption before freeing it"
            )
        tok, enc = pipe.tokenizer, pipe.text_encoder

        def embed(texts: list[str]) -> torch.Tensor:
            ids = tok(
                texts,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            return enc(ids)[0]

        uncond = embed([NEGATIVE_PROMPT])
        for c in missing:
            cache[c] = torch.cat([uncond, embed([c])])
        pipe._khattat_caption_cache = cache
    if free_encoder and pipe.text_encoder is not None:
        pipe.text_encoder = None
        torch.cuda.empty_cache()
    return {c: cache[c] for c in captions}


@dataclass
class SDSConfig:
    guidance_scale: float = 100.0
    t_min: int = 50
    t_max: int = 950
    batch_size: int = 1
    """Augmented crops per step."""
    cut_size: int = 512


class SDSLoss(nn.Module):
    """Score distillation sampling loss."""

    def __init__(
        self,
        pipe,
        caption: str,
        config: SDSConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = config or SDSConfig()
        self.device = torch.device(device)
        self.pipe = pipe

        scheduler = pipe.scheduler
        self.alphas = scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1.0 - scheduler.alphas_cumprod).to(self.device)
        self.num_train_timesteps = scheduler.config.num_train_timesteps

        self.text_embeddings = encode_captions(pipe, [caption], device=self.device)[caption]
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, dim=0)

        # kornia, not torchvision: torchvision's perspective fill writes in place
        # and breaks the backward pass.
        self.augment = nn.Sequential(
            K.RandomPerspective(distortion_scale=0.5, p=0.7),
            K.RandomCrop(
                size=(self.cfg.cut_size, self.cfg.cut_size),
                pad_if_needed=True,
                padding_mode="reflect",
                p=1.0,
            ),
        ).to(self.device)

    def augment_image(self, img_nchw: torch.Tensor) -> torch.Tensor:
        """Replicate to the batch size and apply random crops/perspective."""
        x = img_nchw.repeat(self.cfg.batch_size, 1, 1, 1)
        return self.augment(x)

    def forward(self, x_aug: torch.Tensor) -> torch.Tensor:
        """Surrogate loss whose gradient w.r.t. the latent is the SDS gradient."""
        x = x_aug * 2.0 - 1.0
        latent = self.pipe.vae.encode(x.to(self.pipe.vae.dtype)).latent_dist.sample()
        latent = latent * self.pipe.vae.config.scaling_factor

        with torch.no_grad():
            t = torch.randint(
                low=self.cfg.t_min,
                high=min(self.cfg.t_max, self.num_train_timesteps) - 1,
                size=(latent.shape[0],),
                device=self.device,
                dtype=torch.long,
            )
            eps = torch.randn_like(latent)
            noised = self.pipe.scheduler.add_noise(latent, eps, t)

            model_in = torch.cat([noised] * 2)
            t_in = torch.cat([t] * 2)
            pred = self.pipe.unet(model_in, t_in, encoder_hidden_states=self.text_embeddings).sample
            eps_uncond, eps_cond = pred.float().chunk(2)
            eps_hat = eps_uncond + self.cfg.guidance_scale * (eps_cond - eps_uncond)

            # w(t) = sqrt(alpha_t) * sigma_t
            w = (self.alphas[t] ** 0.5 * self.sigmas[t]).view(-1, 1, 1, 1)
            grad = w * (eps_hat - eps)
            grad = torch.nan_to_num(grad.detach().float(), 0.0, 0.0, 0.0)

        return (grad * latent.float()).sum(dim=1).mean()
