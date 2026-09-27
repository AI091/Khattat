"""As-conformal-as-possible loss (paper eq. 6) on a constrained Delaunay triangulation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from khattat.render.scene import GlyphScene


class ACAPLoss:
    """Penalises change in triangle angles from the initial shape."""

    def __init__(self, scene: GlyphScene, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.points = scene.trainable_points
        self.faces = self._triangulate(scene)
        self.faces_rolled = [torch.roll(f, 1, dims=1) for f in self.faces]
        with torch.no_grad():
            self.target_angles = self._angles(self._all_points())

    def _all_points(self) -> torch.Tensor:
        return torch.cat(self.points, dim=0)

    def _triangulate(self, scene: GlyphScene) -> list[torch.Tensor]:
        """Delaunay triangles of each glyph whose centroid lies inside the glyph."""
        if not scene.trainable_points:
            return []

        faces: list[torch.Tensor] = []

        offsets: list[int] = []
        running = 0
        for tensor in self.points:
            offsets.append(running)
            running += tensor.shape[0]

        tensor_index = 0
        for gi in scene.trainable_glyphs:
            n_contours = len(scene.glyph_path_ids[gi])
            glyph_tensors = self.points[tensor_index : tensor_index + n_contours]
            glyph_offsets = offsets[tensor_index : tensor_index + n_contours]
            tensor_index += n_contours
            if not glyph_tensors:
                continue

            contours_np = [t.detach().cpu().numpy() for t in glyph_tensors]
            # The largest contour is the outline; the rest are holes.
            areas = [Polygon(c).buffer(0).area if len(c) >= 3 else 0.0 for c in contours_np]
            outer_idx = int(np.argmax(areas))
            shell = contours_np[outer_idx]
            holes = [c for i, c in enumerate(contours_np) if i != outer_idx and len(c) >= 3]

            try:
                poly = Polygon(shell, holes=holes).buffer(0)
            except Exception:
                poly = Polygon(shell).buffer(0)
            if poly.is_empty:
                continue

            all_pts = np.concatenate(contours_np, axis=0)
            if len(all_pts) < 4:
                continue
            try:
                tri = Delaunay(all_pts)
            except Exception:
                continue

            centroids = all_pts[tri.simplices].mean(axis=1)
            inside = np.array([poly.contains(ShapelyPoint(c)) for c in centroids], dtype=bool)
            kept = tri.simplices[inside]
            if len(kept) == 0:
                continue

            base = glyph_offsets[0]
            faces.append(torch.from_numpy(kept.astype(np.int64)).to(self.device) + base)

        return faces

    def _angles(self, points: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for face, rolled in zip(self.faces, self.faces_rolled, strict=True):
            tri = points[face]
            tri_rolled = points[rolled]
            edges = tri_rolled - tri
            # Coincident control points are common in fonts.
            lengths = edges.norm(dim=-1, keepdim=True)
            edges = edges / (lengths + 1e-1)
            edges_next = torch.roll(edges, 1, dims=1)
            cosine = (edges * edges_next).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            out.append(torch.arccos(cosine))
        return out

    def __call__(self) -> torch.Tensor:
        if not self.faces:
            return torch.zeros((), device=self.device)
        current = self._angles(self._all_points())
        total = torch.zeros((), device=self.device)
        for cur, target in zip(current, self.target_angles, strict=True):
            total = total + torch.nn.functional.mse_loss(cur, target)
        return total / len(current)

    @property
    def num_triangles(self) -> int:
        return sum(int(f.shape[0]) for f in self.faces)
