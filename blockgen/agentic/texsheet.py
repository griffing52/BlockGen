"""Labelled texture contact sheets: what each block name LOOKS like, as an image.

The texture-palette context (``context.texture_sheet``) shows the model a grid of
real 16x16 block faces with the block name under each tile. The question it
exists to answer is the visual twin of the ontology experiment: does seeing the
materials help a model pick ones that go together, beyond knowing their names?
The shuffled control (labels permuted, tiles fixed) is what separates "the model
used the pictures" from "the model got a bigger prompt".

Pixels come from the same place the renderer draws from:
``renderer.textures.load_face_textures`` (tints and forced-opaque leaves already
applied, solid-colour fallback for anything unmapped), keyed by the legacy
``(id, data)`` pair the DSL name resolves to through ``blockstate``. So the tile
the model sees is exactly the texture its build will be rendered with.

Design points, each fixing a way the experiment could quietly go wrong:

* **Nearest-neighbour upscaling.** Smoothing 16x16 textures turns cobblestone
  into grey mush; the texture (flat / subtle / busy) is half of what a material
  looks like.
* **Paged at <= 1568 px per side.** That is the largest image Anthropic models
  take without server-side downscaling (OpenAI's limit is larger), so a tile
  stays legible on every vendor. A long block list becomes several pages, not one
  unreadable image.
* **Content-addressed disk cache** at ``outputs/cache/texsheets/<sha>.png``. The
  key covers the tile pixels themselves (not only the names), so re-fetching a
  texture pack invalidates it, and identical requests produce identical bytes:
  the PNG is part of the LLM request, and the response cache keys on it.
* **Graceful without textures.** A fresh clone has no ``data/textures/``; every
  tile then falls back to the renderer's solid colour and is listed in
  ``Sheet.missing``. The sheet still renders (and the gap is visible), rather
  than failing a whole battery over an optional asset.
* **Derangement for the control.** ``shuffled_labels`` uses Sattolo's algorithm,
  so NO tile keeps its own label. A plain shuffle leaves ~1 label correct on
  average, a small but real leak of the knowledge the control removes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: Default texture pack. Absolute (the renderer's own default is CWD-relative,
#: which silently gives solid colours when a run starts elsewhere).
PACK_DIR: Path = _REPO / "data" / "textures" / "vanilla"
#: Where finished pages are cached. Tests point this at a tmp dir.
CACHE_DIR: Path = Path(os.environ.get("BLOCKGEN_TEXSHEET_CACHE", "")
                       or _REPO / "outputs" / "cache" / "texsheets")
#: Largest page side, px (Anthropic's no-downscale limit).
MAX_PX = 1568
#: Bump when the layout changes, so stale cached pages are not reused.
LAYOUT_VERSION = 1

FACES = ("side", "top", "bottom")
_PAD = 6           # px around each tile inside its cell
_MARGIN = 8        # px around the whole page
_FONT_PX = 11
_LINE_H = 13
_MAX_LABEL_LINES = 3
_MIN_CELL_W = 84   # wide enough that a typical name wraps to <= 3 lines at 11 px
_BACKDROP = (170, 170, 170)   # behind tiles, so glass/cutouts stay visible


# ---------------------------------------------------------------------------
# Text helpers (shared with trace.py)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def load_font(size: int = _FONT_PX):
    """PIL's bundled font at ``size`` (no system font lookup: same bytes everywhere)."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                    # Pillow < 10.1: bitmap only
        return ImageFont.load_default()


def text_width(text: str, font) -> float:
    try:
        return float(font.getlength(text))
    except AttributeError:                               # bitmap font
        return float(font.getbbox(text)[2])


def wrap_label(name: str, width: float, font, max_lines: int = _MAX_LABEL_LINES) -> List[str]:
    """Wrap a block name at underscores to fit ``width`` px; hard-break long tokens.

    Underscores are kept at the end of the line they close, so the wrapped label
    still reads as the exact token the model must type.
    """
    tokens = [t + "_" for t in name.split("_")]
    tokens[-1] = tokens[-1][:-1]
    lines: List[str] = []
    cur = ""
    for tok in tokens:
        if cur and text_width(cur + tok, font) > width:
            lines.append(cur)
            cur = ""
        while text_width(tok, font) > width and len(tok) > 1:
            # A single token wider than the cell: break it by characters.
            cut = len(tok)
            while cut > 1 and text_width(tok[:cut], font) > width:
                cut -= 1
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(tok[:cut])
            tok = tok[cut:]
        cur += tok
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "~"   # visible truncation, never silent
    return lines


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    """One block's face texture. ``rgba`` is the raw (16x16-ish) RGBA array."""

    name: str
    block: Optional[Tuple[int, int]]
    rgba: Optional[np.ndarray]
    fallback: bool = False          # drawn with the renderer's solid colour


@lru_cache(maxsize=4)
def _library(pack_dir: str):
    from blockgen.renderer.textures import load_face_textures

    return load_face_textures(pack_dir)


def resolve_tiles(names: Sequence[str], face: str = "side",
                  pack_dir: Optional[Path | str] = None) -> List[Tile]:
    """Name -> legacy block -> face texture, for every name (order kept).

    A name that does not resolve keeps its slot with ``rgba=None`` (drawn as a
    hatched "?" tile), so a typo in a recipe's block list is visible on the sheet
    rather than shifting every later label by one.
    """
    if face not in FACES:
        raise ValueError(f"face must be one of {FACES}, got {face!r}")
    from blockgen.agentic.blockstate import try_resolve_block

    lib = _library(str(Path(pack_dir) if pack_dir else PACK_DIR))
    out: List[Tile] = []
    for name in names:
        pair = try_resolve_block(str(name))
        if pair is None or pair == (0, 0):
            out.append(Tile(str(name), None, None, fallback=True))
            continue
        faces = lib[pair]
        rgba = np.asarray(faces.get(face) if faces.get(face) is not None else faces["side"])
        out.append(Tile(str(name), pair, rgba, fallback=pair in lib.fallback_keys))
    return out


def shuffled_labels(names: Sequence[str], seed: int = 0) -> List[str]:
    """A derangement of ``names`` (Sattolo): no label stays on its own tile."""
    labels = list(names)
    rng = random.Random(seed)
    for i in range(len(labels) - 1, 0, -1):
        j = rng.randrange(i)                 # j < i: Sattolo -> one full cycle
        labels[i], labels[j] = labels[j], labels[i]
    return labels


def _tile_image(t: Tile, px: int) -> np.ndarray:
    """(px, px, 3) uint8: texture nearest-upscaled and composited on the backdrop."""
    if t.rgba is None:
        img = np.full((px, px, 3), 235, dtype=np.uint8)
        yy, xx = np.mgrid[:px, :px]
        img[((xx + yy) // max(2, px // 8)) % 2 == 0] = 205       # hatched = unknown name
        return img
    rgba = t.rgba
    if rgba.ndim == 2:
        rgba = np.stack([rgba] * 3 + [np.full_like(rgba, 255)], axis=-1)
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.full(rgba.shape[:2] + (1,), 255, rgba.dtype)], -1)
    h, w = rgba.shape[:2]
    yi = (np.arange(px) * h // px).clip(0, h - 1)
    xi = (np.arange(px) * w // px).clip(0, w - 1)
    up = rgba[yi][:, xi].astype(np.float32)
    a = up[..., 3:4] / 255.0
    rgb = up[..., :3] * a + np.asarray(_BACKDROP, np.float32) * (1 - a)
    return rgb.round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sheet
# ---------------------------------------------------------------------------
@dataclass
class Sheet:
    """A built (possibly multi-page) sheet plus everything needed to audit it."""

    pages: List[bytes]
    names: List[str]                        # tile order (what each tile IS)
    labels: List[str]                       # text under each tile (== names unless shuffled)
    cells: List[Tuple[int, int, int]]       # per tile: (page, x, y) of the tile's top-left
    tile: int
    face: str
    key: str                                # cache key (sha256 hex)
    missing: List[str] = field(default_factory=list)   # solid-colour fallback tiles
    unknown: List[str] = field(default_factory=list)   # names that did not resolve
    cached: bool = False

    @property
    def shuffled(self) -> bool:
        return self.labels != self.names


@dataclass(frozen=True)
class Layout:
    cols: int
    rows_per_page: int
    cell_w: int
    cell_h: int
    label_lines: int


def layout(n: int, *, tile: int, cols: int, labels: Sequence[str],
           max_px: int = MAX_PX) -> Layout:
    """Grid geometry so every page fits ``max_px`` on both sides."""
    if tile < 8:
        raise ValueError(f"tile must be >= 8 px, got {tile}")
    cell_w = max(tile, _MIN_CELL_W - 2 * _PAD) + 2 * _PAD
    if cell_w + 2 * _MARGIN > max_px:
        raise ValueError(f"tile {tile} px does not fit a {max_px} px page")
    font = load_font(_FONT_PX)
    lines = max([len(wrap_label(s, cell_w - 2, font)) for s in labels] or [1])
    cell_h = tile + 2 * _PAD + lines * _LINE_H
    if cell_h + 2 * _MARGIN > max_px:
        # rows_per_page would clamp to 1 and the page would silently exceed max_px
        # (and be downscaled server-side, the thing paging exists to avoid).
        raise ValueError(f"tile {tile} px plus labels does not fit a {max_px} px page")
    cols = max(1, min(int(cols), (max_px - 2 * _MARGIN) // cell_w, max(1, n)))
    rows = max(1, (max_px - 2 * _MARGIN) // cell_h)
    return Layout(cols, rows, cell_w, cell_h, lines)


def _key(tiles: Sequence[Tile], labels: Sequence[str], tile: int, cols: int, face: str,
         max_px: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"v": LAYOUT_VERSION, "labels": list(labels), "tile": tile,
                         "cols": cols, "face": face, "max_px": max_px,
                         "names": [t.name for t in tiles]}).encode())
    for t in tiles:   # the pixels themselves: a re-fetched pack invalidates the cache
        h.update(b"none" if t.rgba is None else np.ascontiguousarray(t.rgba).tobytes())
    return h.hexdigest()


def _cells(n: int, tile: int, lay: Layout) -> List[Tuple[int, int, int]]:
    """(page, x, y) of each tile's top-left corner; the single source of geometry."""
    per_page = lay.cols * lay.rows_per_page
    cells = []
    for i in range(n):
        page, j = divmod(i, per_page)
        r, c = divmod(j, lay.cols)
        cx, cy = _MARGIN + c * lay.cell_w, _MARGIN + r * lay.cell_h
        cells.append((page, cx + (lay.cell_w - tile) // 2, cy + _PAD))
    return cells


def _compose(tiles: Sequence[Tile], labels: Sequence[str], tile: int, lay: Layout
             ) -> List[bytes]:
    from PIL import Image, ImageDraw

    font = load_font(_FONT_PX)
    per_page = lay.cols * lay.rows_per_page
    cells = _cells(len(tiles), tile, lay)
    pages: List[bytes] = []
    for p0 in range(0, len(tiles), per_page):
        chunk = list(range(p0, min(len(tiles), p0 + per_page)))
        rows = (len(chunk) + lay.cols - 1) // lay.cols
        cols = min(lay.cols, len(chunk))
        w = 2 * _MARGIN + cols * lay.cell_w
        h = 2 * _MARGIN + rows * lay.cell_h
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i in chunk:
            _, tx, ty = cells[i]
            cx = tx - (lay.cell_w - tile) // 2
            img.paste(Image.fromarray(_tile_image(tiles[i], tile)), (tx, ty))
            draw.rectangle([tx - 1, ty - 1, tx + tile, ty + tile], outline=(90, 90, 90))
            for k, line in enumerate(wrap_label(labels[i], lay.cell_w - 2, font)):
                lw = text_width(line, font)
                draw.text((cx + (lay.cell_w - lw) / 2, ty + tile + 3 + k * _LINE_H), line,
                          fill=(0, 0, 0), font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pages.append(buf.getvalue())
    return pages


#: In-process sheets, LRU-bounded: ``used`` / ``dream_palette`` sheets differ per
#: build, so an unbounded dict grows by a few hundred KB per build over a battery.
_MEM: "OrderedDict[str, Sheet]" = OrderedDict()
_MEM_MAX = 32
_MEM_LOCK = threading.Lock()


def build_sheet(names: Sequence[str], *, labels: Optional[Sequence[str]] = None,
                tile: int = 64, cols: int = 10, face: str = "side",
                shuffled: bool = False, seed: int = 0,
                pack_dir: Optional[Path | str] = None,
                cache_dir: Optional[Path | str] = None, max_px: int = MAX_PX) -> Sheet:
    """Build (or load from cache) a labelled sheet for ``names``.

    ``shuffled=True`` replaces ``labels`` with a seeded derangement of the names:
    same tiles in the same cells, same label tokens, every label on the wrong tile.
    Duplicate names are kept once (first occurrence), since a repeated tile is
    just noise in the prompt.
    """
    names = list(dict.fromkeys(str(n) for n in names))
    if not names:
        raise ValueError("build_sheet: empty block list")
    if labels is not None and shuffled:
        raise ValueError("pass either labels= or shuffled=True, not both")
    if shuffled:
        labels = shuffled_labels(names, seed)
    labels = list(labels) if labels is not None else list(names)
    if len(labels) != len(names):
        raise ValueError(f"{len(labels)} labels for {len(names)} tiles")
    tiles = resolve_tiles(names, face=face, pack_dir=pack_dir)
    lay = layout(len(tiles), tile=tile, cols=cols, labels=labels, max_px=max_px)
    key = _key(tiles, labels, tile, cols, face, max_px)
    with _MEM_LOCK:
        hit = _MEM.get(key)
        if hit is not None:
            _MEM.move_to_end(key)
    if hit is not None:
        return hit

    root = Path(cache_dir) if cache_dir else CACHE_DIR
    per_page = lay.cols * lay.rows_per_page
    n_pages = (len(tiles) + per_page - 1) // per_page
    # One file per page, each named by its own sha (key + page index), so the
    # cache stays content-addressed whatever the page count.
    paths = [root / f"{hashlib.sha256(f'{key}:{p}'.encode()).hexdigest()[:24]}.png"
             for p in range(n_pages)]
    cells = _cells(len(tiles), tile, lay)
    cached = all(p.is_file() for p in paths)
    if cached:
        pages = [p.read_bytes() for p in paths]
    else:
        pages = _compose(tiles, labels, tile, lay)
        try:
            root.mkdir(parents=True, exist_ok=True)
            for p, png in zip(paths, pages):
                tmp = p.with_suffix(f".tmp{os.getpid()}")
                tmp.write_bytes(png)
                os.replace(tmp, p)                   # atomic: parallel arms share the cache
        except OSError as exc:                       # read-only checkout: still usable
            warnings.warn(f"texsheet: cache dir {root} not writable ({exc}); sheet not cached")
    sheet = Sheet(pages=pages, names=names, labels=labels, cells=cells, tile=tile,
                  face=face, key=key,
                  missing=[t.name for t in tiles if t.rgba is not None and t.fallback],
                  unknown=[t.name for t in tiles if t.rgba is None], cached=cached)
    with _MEM_LOCK:
        _MEM[key] = sheet
        while len(_MEM) > _MEM_MAX:
            _MEM.popitem(last=False)
    return sheet


def clear_cache() -> None:
    """Drop in-process sheets and the texture library (tests swap PACK_DIR)."""
    with _MEM_LOCK:
        _MEM.clear()
    _library.cache_clear()


__all__ = ["CACHE_DIR", "FACES", "Layout", "MAX_PX", "PACK_DIR", "Sheet", "Tile",
           "build_sheet", "clear_cache", "layout", "load_font", "resolve_tiles",
           "shuffled_labels", "text_width", "wrap_label"]
