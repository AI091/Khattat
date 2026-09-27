"""Shape text with HarfBuzz and extract glyph outlines as cubic Beziers.

Shaping applies the font's substitution and positioning tables, so Arabic
letters get their contextual (joined) forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import uharfbuzz as hb

Point = tuple[float, float]


@dataclass
class Contour:
    """A closed loop of cubic segments `(c1, c2, end)`; the last `end` equals `start`."""

    start: Point
    segments: list[tuple[Point, Point, Point]] = field(default_factory=list)

    def points(self) -> list[Point]:
        """DiffVG layout `[v0, c, c, v1, c, c, ...]`, without the implied closing vertex."""
        out: list[Point] = [self.start]
        for c1, c2, end in self.segments[:-1]:
            out.extend((c1, c2, end))
        c1, c2, _ = self.segments[-1]
        out.extend((c1, c2))
        return out

    def __len__(self) -> int:
        return len(self.segments)


@dataclass
class ShapedGlyph:
    """A shaped glyph with its outline in canvas coordinates."""

    glyph_id: int
    cluster: int
    """Index of the source character in the shaped text."""
    contours: list[Contour]
    advance: float
    offset: Point

    @property
    def num_segments(self) -> int:
        return sum(len(c) for c in self.contours)

    def is_empty(self) -> bool:
        """True for glyphs with no outline, such as a space."""
        return not self.contours


@dataclass
class ShapedWord:
    """A shaped word laid out on a square canvas."""

    text: str
    glyphs: list[ShapedGlyph]
    width: float
    height: float
    font_path: Path
    direction: str

    def visible(self) -> list[ShapedGlyph]:
        return [g for g in self.glyphs if not g.is_empty()]

    def __len__(self) -> int:
        return len(self.glyphs)


class _OutlinePen:
    """Collects HarfBuzz draw callbacks into contours, converting everything to cubics."""

    def __init__(self) -> None:
        self.contours: list[Contour] = []
        self._current: Contour | None = None
        self._pos: Point = (0.0, 0.0)

    def move_to(self, x: float, y: float) -> None:
        self._flush()
        self._current = Contour(start=(x, y))
        self._pos = (x, y)

    def line_to(self, x: float, y: float) -> None:
        if self._current is None:
            self.move_to(x, y)
            return
        x0, y0 = self._pos
        c1 = (x0 + (x - x0) / 3.0, y0 + (y - y0) / 3.0)
        c2 = (x0 + 2.0 * (x - x0) / 3.0, y0 + 2.0 * (y - y0) / 3.0)
        self._current.segments.append((c1, c2, (x, y)))
        self._pos = (x, y)

    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> None:
        if self._current is None:
            self.move_to(x, y)
            return
        x0, y0 = self._pos
        c1 = (x0 + 2.0 / 3.0 * (cx - x0), y0 + 2.0 / 3.0 * (cy - y0))
        c2 = (x + 2.0 / 3.0 * (cx - x), y + 2.0 / 3.0 * (cy - y))
        self._current.segments.append((c1, c2, (x, y)))
        self._pos = (x, y)

    def cubic_to(self, c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> None:
        if self._current is None:
            self.move_to(x, y)
            return
        self._current.segments.append(((c1x, c1y), (c2x, c2y), (x, y)))
        self._pos = (x, y)

    def close_path(self) -> None:
        self._flush()

    def _flush(self) -> None:
        c = self._current
        self._current = None
        if c is None or not c.segments:
            return
        # Some fonts leave a gap between the last point and the start.
        if _dist(c.segments[-1][2], c.start) > 1e-6:
            self._pos = c.segments[-1][2]
            self._current = c
            self.line_to(*c.start)
            c = self._current
            self._current = None
        self.contours.append(c)


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _make_draw_funcs() -> hb.DrawFuncs:
    funcs = hb.DrawFuncs()
    funcs.set_move_to_func(lambda x, y, pen: pen.move_to(x, y))
    funcs.set_line_to_func(lambda x, y, pen: pen.line_to(x, y))
    funcs.set_quadratic_to_func(lambda cx, cy, x, y, pen: pen.quadratic_to(cx, cy, x, y))
    funcs.set_cubic_to_func(
        lambda c1x, c1y, c2x, c2y, x, y, pen: pen.cubic_to(c1x, c1y, c2x, c2y, x, y)
    )
    funcs.set_close_path_func(lambda pen: pen.close_path())
    return funcs


_DRAW_FUNCS = _make_draw_funcs()


def shape_word(
    text: str,
    font_path: str | Path,
    *,
    canvas_size: int = 600,
    margin: float = 0.12,
    direction: str | None = None,
    script: str | None = None,
    language: str | None = None,
) -> ShapedWord:
    """Shape `text` and return its outlines centred in a `canvas_size` square, y down.

    `direction`, `script` and `language` override HarfBuzz's detection.
    """
    font_path = Path(font_path)
    if not font_path.exists():
        raise FileNotFoundError(f"font not found: {font_path}")

    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    font = hb.Font(face)
    upem = face.upem
    font.scale = (upem, upem)

    buf = hb.Buffer()
    buf.add_str(text)
    if direction:
        buf.direction = direction
    if script:
        buf.script = script
    if language:
        buf.language = language
    buf.guess_segment_properties()
    resolved_direction = buf.direction

    hb.shape(font, buf)

    glyphs: list[ShapedGlyph] = []
    pen_x = 0.0
    pen_y = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        pen = _OutlinePen()
        font.draw_glyph(info.codepoint, _DRAW_FUNCS, pen)

        ox = pen_x + pos.x_offset
        oy = pen_y + pos.y_offset
        contours = [_translate(c, ox, oy) for c in pen.contours]

        glyphs.append(
            ShapedGlyph(
                glyph_id=info.codepoint,
                cluster=info.cluster,
                contours=contours,
                advance=float(pos.x_advance),
                offset=(ox, oy),
            )
        )
        pen_x += pos.x_advance
        pen_y += pos.y_advance

    return _fit_to_canvas(
        text=text,
        glyphs=glyphs,
        canvas_size=canvas_size,
        margin=margin,
        font_path=font_path,
        direction=resolved_direction,
    )


def _translate(contour: Contour, dx: float, dy: float) -> Contour:
    def t(p: Point) -> Point:
        return (p[0] + dx, p[1] + dy)

    return Contour(
        start=t(contour.start),
        segments=[(t(a), t(b), t(c)) for a, b, c in contour.segments],
    )


def _transform(contour: Contour, scale: float, dx: float, dy: float, flip_y: bool) -> Contour:
    sy = -scale if flip_y else scale

    def t(p: Point) -> Point:
        return (p[0] * scale + dx, p[1] * sy + dy)

    return Contour(
        start=t(contour.start),
        segments=[(t(a), t(b), t(c)) for a, b, c in contour.segments],
    )


def _fit_to_canvas(
    *,
    text: str,
    glyphs: list[ShapedGlyph],
    canvas_size: int,
    margin: float,
    font_path: Path,
    direction: str,
) -> ShapedWord:
    """Scale and centre the glyphs in a square canvas, flipping y."""
    pts = [
        p
        for g in glyphs
        for c in g.contours
        for p in [c.start, *(q for seg in c.segments for q in seg)]
    ]
    if not pts:
        return ShapedWord(
            text, glyphs, float(canvas_size), float(canvas_size), font_path, str(direction)
        )

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)

    usable = canvas_size * (1.0 - 2.0 * margin)
    scale = min(usable / w, usable / h)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    dx = canvas_size / 2.0 - cx * scale
    dy = canvas_size / 2.0 + cy * scale

    fitted = [
        ShapedGlyph(
            glyph_id=g.glyph_id,
            cluster=g.cluster,
            contours=[_transform(c, scale, dx, dy, flip_y=True) for c in g.contours],
            advance=g.advance * scale,
            offset=g.offset,
        )
        for g in glyphs
    ]
    return ShapedWord(
        text=text,
        glyphs=fitted,
        width=float(canvas_size),
        height=float(canvas_size),
        font_path=font_path,
        direction=str(direction),
    )
