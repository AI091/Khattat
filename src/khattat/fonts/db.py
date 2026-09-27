"""Font discovery and specimen rendering for FontCLIP."""

from __future__ import annotations

import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# FontCLIP's specimen format: a four-line pangram at 150px.
LATIN_SPECIMEN = "The quick\nbrown fox\njumps over\nthe lazy dog"

ARABIC_SPECIMEN = "نص حكيم له سر\nقاطع وذو شأن\nعظيم مكتوب على\nثوب أخضر"

CHAR_SIZE = 150

_ARABIC_PROBE = "ابجد"  # alef beh jeem dal
_LATIN_PROBE = "Hamburg"


@dataclass(frozen=True)
class FontEntry:
    path: Path
    family: str
    scripts: frozenset[str]

    def supports(self, script: str) -> bool:
        return script in self.scripts

    @property
    def name(self) -> str:
        return self.path.stem


def _covered(cmap: set[int], text: str) -> bool:
    return all(ord(ch) in cmap for ch in text if not ch.isspace())


# Excluded from the Arabic database: Nastaliq, which the OCR model cannot read,
# and fonts that only carry a few Arabic glyphs.
_ARABIC_EXCLUDE = ("nastaliq", "nerd font", "mono", "math")


def _excluded(entry: FontEntry, script: str | None) -> bool:
    if script != "arabic":
        return False
    name = f"{entry.family} {entry.path.name}".lower()
    return any(tag in name for tag in _ARABIC_EXCLUDE)


def inspect_font(path: Path) -> FontEntry | None:
    """Family name and supported scripts, or None if the font can't be read."""
    try:
        tt = TTFont(str(path), fontNumber=0, lazy=True)
        cmap: set[int] = set()
        for table in tt["cmap"].tables:
            cmap.update(table.cmap.keys())
        # Prefer the typographic family (ID 16), which is shared across weights.
        names: dict[int, str] = {}
        for record in tt["name"].names:
            if record.nameID in (1, 16) and record.nameID not in names:
                try:
                    names[record.nameID] = record.toUnicode()
                except Exception:
                    continue
        family = names.get(16) or names.get(1, "")
        tt.close()
    except Exception:
        return None

    scripts = set()
    if _covered(cmap, _LATIN_PROBE):
        scripts.add("latin")
    if _covered(cmap, _ARABIC_PROBE):
        scripts.add("arabic")
    if not scripts:
        return None
    return FontEntry(path=path, family=family or path.stem, scripts=frozenset(scripts))


def discover_fonts(
    directories: list[Path] | None = None, *, script: str | None = None
) -> list[FontEntry]:
    """Fonts in `directories`, or all fontconfig fonts."""
    paths: list[Path] = []
    if directories:
        for d in directories:
            paths.extend(p for p in Path(d).rglob("*") if p.suffix.lower() in {".ttf", ".otf"})
    else:
        try:
            out = subprocess.run(
                ["fc-list", ":", "file"], capture_output=True, text=True, check=True
            ).stdout
            paths = [
                Path(candidate)
                for line in out.splitlines()
                if (candidate := line.split(":")[0].strip())
                and candidate.lower().endswith((".ttf", ".otf"))
            ]
        except (subprocess.CalledProcessError, FileNotFoundError):
            paths = []

    seen: set[Path] = set()
    entries: list[FontEntry] = []
    for p in sorted(set(paths)):
        if p in seen or not p.exists():
            continue
        seen.add(p)
        entry = inspect_font(p)
        if entry and (script is None or entry.supports(script)) and not _excluded(entry, script):
            entries.append(entry)
    return entries


def specimen_text(script: str) -> str:
    return ARABIC_SPECIMEN if script == "arabic" else LATIN_SPECIMEN


def render_specimen(
    font: FontEntry, *, script: str = "latin", char_size: int = CHAR_SIZE
) -> Image.Image:
    """Render a specimen as FontCLIP's `generate_images_for_fonts` does."""
    text = specimen_text(script)
    lines = text.split("\n")
    line_num = len(lines)
    width = int(char_size * len(text) * 1.8 / line_num)
    height = int(char_size * 1.5) * line_num

    try:
        pil_font = ImageFont.truetype(str(font.path), char_size)
    except Exception:
        pil_font = ImageFont.load_default()

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y_text: float | None = None
    for line in lines:
        left, top, right, bottom = pil_font.getbbox(line)
        line_height = bottom - top
        line_width = right - left
        if y_text is None:
            spare = (height - line_height * line_num) / 2
            y_text = spare if spare > 0 else 0.0
        x_text = (width - line_width) / 2 if (width - line_width) / 2 > 0 else 0.0
        draw.text((x_text - left, y_text - top), line, font=pil_font, fill=(0, 0, 0))
        y_text += line_height
    return img


def script_of(text: str) -> str:
    """'arabic' or 'latin'."""
    for ch in text:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "ARABIC" in name:
            return "arabic"
    return "latin"
