"""Optional colouring with SD2 depth-to-image (paper Appendix C).

Works best on `region_output.png`, where the letters fill the frame. The paper's
"negative prompt strength 0.7" is used as the img2img strength.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PROMPT = (
    "A vibrant, minimalist 2D vector illustration of {concept} using colors "
    "typically associated with {concept} objects"
)
NEGATIVE_PROMPT = "Deformities, ugliness, and incorrect anatomy"


def colorize(
    image_path: Path,
    concept: str,
    *,
    out_path: Path | None = None,
    strength: float = 0.7,
    steps: int = 50,
    seed: int = 0,
    device: str = "cuda",
) -> Path:
    import torch

    from khattat.models.sd import load_depth_pipeline

    pipe = load_depth_pipeline(device=device)
    img = Image.open(image_path).convert("RGB")
    gen = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=PROMPT.format(concept=concept),
        negative_prompt=NEGATIVE_PROMPT,
        image=img,
        strength=strength,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]
    out_path = out_path or image_path.with_name(image_path.stem + "_color.png")
    result.save(out_path)
    del pipe
    torch.cuda.empty_cache()
    return out_path
