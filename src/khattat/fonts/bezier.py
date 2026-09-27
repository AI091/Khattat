"""Split the longest Bezier segment until a glyph reaches its control-point budget (paper §4.5)."""

from __future__ import annotations

import heapq

from khattat.fonts.shaping import Contour, Point, ShapedGlyph


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def split_cubic(
    p0: Point, p1: Point, p2: Point, p3: Point, t: float = 0.5
) -> tuple[tuple[Point, Point, Point], tuple[Point, Point, Point]]:
    """De Casteljau split at `t`; returns both halves as `(c1, c2, end)`."""
    a = _lerp(p0, p1, t)
    b = _lerp(p1, p2, t)
    c = _lerp(p2, p3, t)
    d = _lerp(a, b, t)
    e = _lerp(b, c, t)
    mid = _lerp(d, e, t)
    return ((a, d, mid), (e, c, p3))


def _control_polygon_length(p0: Point, seg: tuple[Point, Point, Point]) -> float:
    """Control-polygon length, used as a cheap upper bound on arc length."""
    pts = [p0, *seg]
    return sum(
        ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
        for i in range(len(pts) - 1)
    )


def _with_starts(contour: Contour) -> list[tuple[Point, tuple[Point, Point, Point]]]:
    """Pair each segment with its start point."""
    out = []
    pos = contour.start
    for seg in contour.segments:
        out.append((pos, seg))
        pos = seg[2]
    return out


def contour_length(contour: Contour) -> float:
    """Total control-polygon length of a contour."""
    return sum(_control_polygon_length(p0, seg) for p0, seg in _with_starts(contour))


def subdivide_contour(contour: Contour, target_segments: int) -> Contour:
    """Split the longest segment until the contour has `target_segments` segments."""
    if len(contour.segments) >= target_segments:
        return contour

    segments = _with_starts(contour)
    heap = [(-_control_polygon_length(p0, seg), i, p0, seg) for i, (p0, seg) in enumerate(segments)]
    heapq.heapify(heap)

    # A split keeps the first half in place and links the second half after it.
    slots: dict[int, tuple[Point, tuple[Point, Point, Point]]] = dict(enumerate(segments))
    successor: dict[int, int | None] = {}
    next_id = len(segments)

    while len(slots) < target_segments:
        _, idx, p0, (c1, c2, p3) = heapq.heappop(heap)
        first, second = split_cubic(p0, c1, c2, p3)
        mid = first[2]

        new_id = next_id
        next_id += 1
        slots[idx] = (p0, first)
        slots[new_id] = (mid, second)
        successor[new_id] = successor.get(idx)
        successor[idx] = new_id

        heapq.heappush(heap, (-_control_polygon_length(p0, first), idx, p0, first))
        heapq.heappush(heap, (-_control_polygon_length(mid, second), new_id, mid, second))

    ordered: list[tuple[Point, Point, Point]] = []
    for i in range(len(segments)):
        node: int | None = i
        while node is not None:
            ordered.append(slots[node][1])
            node = successor.get(node)
    return Contour(start=contour.start, segments=ordered)


def subdivide_glyph(glyph: ShapedGlyph, target_points: int) -> ShapedGlyph:
    """Subdivide a glyph to `target_points` control points.

    The budget is shared between contours by length, with at least 4 segments each.
    """
    if glyph.is_empty():
        return glyph

    target_segments = max(1, target_points // 3)
    current = glyph.num_segments
    if current >= target_segments:
        return glyph

    lengths = [contour_length(c) for c in glyph.contours]
    total = sum(lengths) or 1.0

    quotas: list[int] = []
    for contour, length in zip(glyph.contours, lengths, strict=True):
        share = round(target_segments * length / total)
        quotas.append(max(len(contour), min(share, target_segments), 4))

    while sum(quotas) > target_segments and max(quotas) > 4:
        quotas[quotas.index(max(quotas))] -= 1

    return ShapedGlyph(
        glyph_id=glyph.glyph_id,
        cluster=glyph.cluster,
        contours=[subdivide_contour(c, q) for c, q in zip(glyph.contours, quotas, strict=True)],
        advance=glyph.advance,
        offset=glyph.offset,
    )
