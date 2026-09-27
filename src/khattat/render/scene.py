"""Build DiffVG scenes from shaped words.

A region scene holds only the glyphs being morphed, scaled to fill the canvas;
the losses see this one. A word scene holds the whole word and is used for the
initial and final renders.
"""

from __future__ import annotations

from dataclasses import dataclass

import pydiffvg
import torch

from khattat.fonts.bezier import subdivide_glyph
from khattat.fonts.shaping import Contour, Point, ShapedGlyph, ShapedWord


@dataclass(frozen=True)
class Affine:
    """Uniform scale then translate: p' = p * scale + offset."""

    scale: float
    dx: float
    dy: float

    def apply(self, p: Point) -> Point:
        return (p[0] * self.scale + self.dx, p[1] * self.scale + self.dy)

    def inverse(self) -> Affine:
        return Affine(1.0 / self.scale, -self.dx / self.scale, -self.dy / self.scale)

    def apply_tensor(self, pts: torch.Tensor) -> torch.Tensor:
        return pts * self.scale + pts.new_tensor([self.dx, self.dy])


IDENTITY = Affine(1.0, 0.0, 0.0)


def _map_contour(c: Contour, t: Affine) -> Contour:
    return Contour(
        start=t.apply(c.start),
        segments=[(t.apply(a), t.apply(b), t.apply(e)) for a, b, e in c.segments],
    )


def _map_glyph(g: ShapedGlyph, t: Affine) -> ShapedGlyph:
    return ShapedGlyph(
        glyph_id=g.glyph_id,
        cluster=g.cluster,
        contours=[_map_contour(c, t) for c in g.contours],
        advance=g.advance * t.scale,
        offset=g.offset,
    )


def region_glyph_indices(word: ShapedWord, region: tuple[int, int]) -> list[int]:
    """Visible glyphs whose source character lies in the half-open character range `region`.

    Selecting by character keeps Arabic dots, which are separate glyphs, with their letter.
    """
    start, end = region
    n_chars = len(word.text)
    if not (0 <= start < end <= n_chars):
        raise ValueError(f"region {region} out of range for a {n_chars}-character word")
    return [i for i, g in enumerate(word.visible()) if start <= g.cluster < end]


def fit_transform(glyphs: list[ShapedGlyph], canvas: int, margin: float = 0.1) -> Affine:
    """Scale and centre a set of glyphs to fill a square canvas."""
    pts = [
        p
        for g in glyphs
        for c in g.contours
        for p in [c.start, *(q for seg in c.segments for q in seg)]
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(max(xs) - min(xs), 1e-6)
    h = max(max(ys) - min(ys), 1e-6)
    scale = canvas * (1.0 - 2.0 * margin) / max(w, h)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return Affine(scale, canvas / 2.0 - cx * scale, canvas / 2.0 - cy * scale)


@dataclass
class GlyphScene:
    """A renderable scene, some of whose glyphs have trainable control points."""

    paths: list[pydiffvg.Path]
    groups: list[pydiffvg.ShapeGroup]
    width: int
    height: int
    trainable_points: list[torch.Tensor]
    """Point tensors carrying `requires_grad`; the optimizer's parameter list."""
    glyph_path_ids: list[list[int]]
    """`glyph_path_ids[i]` lists the path indices belonging to glyph i."""
    trainable_glyphs: list[int]
    """Indices into this scene's glyphs whose points are trainable, in order."""
    transform: Affine = IDENTITY
    """Maps word-canvas coordinates into this scene's coordinates."""

    def parameters(self) -> list[torch.Tensor]:
        return self.trainable_points

    def all_points(self) -> torch.Tensor:
        if not self.trainable_points:
            return torch.empty(0, 2)
        return torch.cat(self.trainable_points, dim=0)


def _build(
    glyphs: list[ShapedGlyph],
    trainable: set[int],
    *,
    size: int,
    control_points_per_glyph: int,
    device: torch.device,
    transform: Affine,
) -> GlyphScene:
    paths: list[pydiffvg.Path] = []
    groups: list[pydiffvg.ShapeGroup] = []
    points: list[torch.Tensor] = []
    glyph_path_ids: list[list[int]] = []

    for gi, glyph in enumerate(glyphs):
        is_trainable = gi in trainable
        g = subdivide_glyph(glyph, control_points_per_glyph) if is_trainable else glyph
        ids: list[int] = []
        for contour in g.contours:
            pts = torch.tensor(contour.points(), dtype=torch.float32, device=device)
            if is_trainable:
                pts.requires_grad_(True)
                points.append(pts)
            paths.append(
                pydiffvg.Path(
                    num_control_points=torch.full(
                        (len(contour),), 2, dtype=torch.int32, device=device
                    ),
                    points=pts,
                    is_closed=True,
                )
            )
            ids.append(len(paths) - 1)
        # Even-odd fill cuts counters out of the outer contour.
        groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor(ids, dtype=torch.int32, device=device),
                fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
                use_even_odd_rule=True,
            )
        )
        glyph_path_ids.append(ids)

    return GlyphScene(
        paths=paths,
        groups=groups,
        width=size,
        height=size,
        trainable_points=points,
        glyph_path_ids=glyph_path_ids,
        trainable_glyphs=sorted(trainable),
        transform=transform,
    )


def build_region_scene(
    word: ShapedWord,
    region: tuple[int, int],
    *,
    size: int = 600,
    margin: float = 0.1,
    control_points_per_glyph: int = 120,
    device: torch.device | str = "cuda",
) -> GlyphScene:
    """Scene of just the region's glyphs, all trainable, scaled to fill the canvas."""
    indices = region_glyph_indices(word, region)
    if not indices:
        raise ValueError(f"region {region} of {word.text!r} contains no visible glyphs")
    visible = word.visible()
    glyphs = [visible[i] for i in indices]
    t = fit_transform(glyphs, size, margin)
    return _build(
        [_map_glyph(g, t) for g in glyphs],
        trainable=set(range(len(glyphs))),
        size=size,
        control_points_per_glyph=control_points_per_glyph,
        device=torch.device(device),
        transform=t,
    )


def build_word_scene(
    word: ShapedWord,
    *,
    region: tuple[int, int] | None = None,
    region_scene: GlyphScene | None = None,
    device: torch.device | str = "cuda",
) -> GlyphScene:
    """Non-trainable scene of the whole word, with `region_scene` spliced back in if given."""
    device = torch.device(device)
    visible = word.visible()
    scene = _build(
        visible,
        trainable=set(),
        size=int(word.width),
        control_points_per_glyph=0,
        device=device,
        transform=IDENTITY,
    )
    if region is None or region_scene is None:
        return scene

    back = region_scene.transform.inverse()
    for local, gi in enumerate(region_glyph_indices(word, region)):
        src_ids = region_scene.glyph_path_ids[local]
        dst_ids = scene.glyph_path_ids[gi]
        new_ids: list[int] = []
        for sid in src_ids:
            src = region_scene.paths[sid]
            scene.paths.append(
                pydiffvg.Path(
                    num_control_points=src.num_control_points.clone(),
                    points=back.apply_tensor(src.points.detach()),
                    is_closed=True,
                )
            )
            new_ids.append(len(scene.paths) - 1)
        scene.groups[gi] = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor(new_ids, dtype=torch.int32, device=device),
            fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
            use_even_odd_rule=True,
        )
        scene.glyph_path_ids[gi] = new_ids
        del dst_ids  # old paths stay in the list but no group references them
    return scene


def render(
    scene: GlyphScene, *, size: int | None = None, seed: int = 0, samples: int = 2
) -> torch.Tensor:
    """Rasterize to an (H, W, 3) tensor in [0, 1] on white."""
    w = h = size or scene.width
    args = pydiffvg.RenderFunction.serialize_scene(w, h, scene.paths, scene.groups)
    img = pydiffvg.RenderFunction.apply(w, h, samples, samples, seed, None, *args)
    alpha = img[:, :, 3:4]
    return alpha * img[:, :, :3] + (1.0 - alpha)


def to_nchw(img: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) in [0,1] -> (1, 3, H, W), the layout every model here wants."""
    return img.permute(2, 0, 1).unsqueeze(0)
