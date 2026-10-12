"""Turnaround sheets: four views of one building in ONE generated image.

Why one image instead of four: per-view dreams are image edits of the iso dream,
and an edit model redraws its reference (same camera, sometimes a different
house), so the back and side targets drift from the front. A single 2x2 sheet is
drawn in one pass, so the four panels share one building by construction. The
cost is that the panels must be cut apart and assigned to cameras, which is what
this module does:

* :func:`split_grid` finds the gutters (the emptiest band near each nominal cut),
  crops, trims and pads each panel. Each tile carries the foreground fraction of
  the cut lines around it in a PNG text chunk (``blockgen.cut_fg``), so
  :func:`tiles_ok` can tell a real sheet (cuts through white gutters) from a
  single building that was sliced into quarters, without seeing the source.
* :func:`best_assignment` picks which panel is which camera by silhouette IoU
  against renders of the first build, over the 8 cyclic orders only (4 rotations
  x 2 directions): a turnaround is a rotation, so any other permutation would be
  a physically impossible sheet and only adds ways to overfit.

Pure numpy / PIL / scipy (via scoring); nothing here calls a model except
:func:`sheet_review`, which only builds the turn.
"""

from __future__ import annotations

import io
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.providers import Message, Part

#: Image prompt for the sheet (a DreamSpec template: {style} {desc} {W} {H} {D}).
SHEET_TEMPLATE = (
    "{style} A turnaround sheet of ONE Minecraft building: a 2 by 2 grid of four "
    "panels on a pure white background separated by wide white gutters. The SAME "
    "building in every panel, same scale, same isometric camera height (about 30 "
    "degrees up). Top-left: seen from its front-right corner. Each next panel "
    "(top-right, then bottom-left, then bottom-right) shows it rotated a further 90 "
    "degrees. {desc}. It fits in {W} x {D} blocks of footprint and {H} blocks of height.")

#: Literal tag in the sheet review turn (the mock:auto LLM keys its reply on it).
SHEET_REVIEW_TAG = "TURNAROUND SHEET REVIEW"

#: PNG text key carrying the cut-line foreground fraction of a split tile.
CUT_KEY = "blockgen.cut_fg"

#: Longest side the foreground mask is computed at (speed; cuts are found at
#: this scale and mapped back, which is exact enough for gutters >= 8 px).
_MASK_PX = 256

#: Panel names in reading order (2x2), for turns and docs.
PANEL_NAMES: Tuple[str, ...] = ("top-left", "top-right", "bottom-left", "bottom-right")


# ===========================================================================
# PNG helpers
# ===========================================================================
def _decode(png: bytes) -> np.ndarray:
    from blockgen.agentic.render import png_to_rgb
    return np.asarray(png_to_rgb(png))


def _encode(rgb: np.ndarray, meta: Optional[Mapping[str, Any]] = None) -> bytes:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = None
    if meta:
        info = PngInfo()
        for k, v in meta.items():
            info.add_text(str(k), json.dumps(v))
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def tile_meta(png: bytes) -> Dict[str, Any]:
    """The ``blockgen.*`` text chunks of a tile ({} for an ordinary PNG)."""
    from PIL import Image
    try:
        text = dict(getattr(Image.open(io.BytesIO(png)), "text", {}) or {})
    except Exception:  # noqa: BLE001 - not a PNG we wrote: no metadata
        return {}
    out: Dict[str, Any] = {}
    for k, v in text.items():
        if k.startswith("blockgen."):
            try:
                out[k] = json.loads(v)
            except ValueError:
                out[k] = v
    return out


def _small_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Foreground mask at <= _MASK_PX on the long side, and the scale used."""
    from PIL import Image

    from blockgen.agentic.scoring import foreground_mask
    h, w = rgb.shape[:2]
    s = min(1.0, _MASK_PX / float(max(h, w)))
    if s < 1.0:
        small = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize(
            (max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR))
    else:
        small = np.asarray(rgb, np.uint8)
    return foreground_mask(small), s


# ===========================================================================
# Split / compose
# ===========================================================================
def _find_cuts(profile: np.ndarray, n: int, search: float) -> List[Tuple[int, float]]:
    """``n - 1`` cut positions in a 1-D foreground profile, with the profile value
    at each (the fraction of that line that is foreground).

    Each cut is the centre of the longest run of minimal profile within
    ``+-search`` of its nominal position k/n (ties: the run nearest nominal), so a
    wide white gutter is cut in its middle and a panel that reaches into the
    gutter is not clipped. Cuts are kept strictly increasing.
    """
    size = len(profile)
    out: List[Tuple[int, float]] = []
    prev = 0
    for k in range(1, n):
        nominal = k * size / n
        lo = max(prev + 1, int(math.floor(nominal - search * size)))
        hi = min(size - 1, int(math.ceil(nominal + search * size)))
        if hi <= lo:
            c = min(size - 1, max(prev + 1, int(round(nominal))))
            out.append((c, float(profile[c]) if size else 0.0))
            prev = c
            continue
        win = profile[lo:hi + 1]
        m = float(win.min())
        good = win <= m + 1e-9
        runs: List[Tuple[int, int]] = []
        i = 0
        while i < len(good):
            if good[i]:
                j = i
                while j + 1 < len(good) and good[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        a, b = max(runs, key=lambda r: (r[1] - r[0], -abs(lo + (r[0] + r[1]) / 2 - nominal)))
        c = lo + (a + b) // 2
        out.append((c, m))
        prev = c
    return out


def _matte(crop: np.ndarray) -> np.ndarray:
    """Replace the crop's backdrop by pure white (the building is kept as drawn).

    Why: the crop is padded with pure white, and an image model's backdrop is
    rarely pure white (off-white, a faint gradient, noise). Left in place, the
    tile's border would be uniform white while the margin around the building is
    off-white, so ``scoring.foreground_mask`` (exact-colour regime for a uniform
    border) would take the whole off-white rectangle for the building: silhouette
    IoU against a render drops to about 0.5 and every tile passes ``tiles_ok``.
    A crop whose border is already exactly white (a render, a mock) is untouched.
    """
    from scipy import ndimage

    from blockgen.agentic.scoring import foreground_mask
    if crop.size == 0:
        return crop
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    if bool((border == 255).all()):
        return crop
    fg = ndimage.binary_dilation(foreground_mask(crop), iterations=1)
    return np.where(fg[..., None], crop, np.uint8(255)).astype(np.uint8)


def _trim_pad(cell: np.ndarray, cmask: Optional[np.ndarray], s: float, pad: float) -> np.ndarray:
    """Crop ``cell`` to its foreground bbox (``cmask`` is at scale ``s``), then pad
    to a centred square on white with a ``pad`` fraction margin."""
    h, w = cell.shape[:2]
    if cmask is None or not cmask.any():
        # No foreground: return the cell untouched. Padding it with white would
        # turn a uniform non-white cell (a "sheet" that is one flat picture) into
        # a coloured square on white, i.e. a plausible-looking panel.
        return np.ascontiguousarray(cell[..., :3])
    ys, xs = np.nonzero(cmask)
    m = int(math.ceil(1.0 / s)) + 1
    y0 = max(0, int(math.floor(ys.min() / s)) - m)
    y1 = min(h, int(math.ceil((ys.max() + 1) / s)) + m)
    x0 = max(0, int(math.floor(xs.min() / s)) - m)
    x1 = min(w, int(math.ceil((xs.max() + 1) / s)) + m)
    cell = _matte(np.ascontiguousarray(cell[y0:y1, x0:x1, :3]))
    h, w = cell.shape[:2]
    side = max(h, w)
    margin = int(round(pad * side))
    n = side + 2 * margin
    out = np.full((n, n, 3), 255, np.uint8)
    oy, ox = margin + (side - h) // 2, margin + (side - w) // 2
    out[oy:oy + h, ox:ox + w] = cell[..., :3]
    return out


def split_grid(png: bytes, rows: int = 2, cols: int = 2, *, search: float = 0.15,
               pad: float = 0.06) -> List[bytes]:
    """Cut a ``rows x cols`` sheet into tiles, reading order (row-major).

    Row cuts come from the whole image's row profile; column cuts are found per
    row band (an uneven sheet may have its gutters at different x in each row).
    With no foreground at all the cuts fall back to exact fractions. Every tile is
    trimmed to its foreground bbox and padded square on white (a cell with no
    foreground is returned as cut, so :func:`tiles_ok` sees it empty); its PNG carries
    ``blockgen.cut_fg`` = the largest foreground fraction of the cut lines around
    it (0 for a cut through a clean gutter).
    """
    rows, cols = max(1, int(rows)), max(1, int(cols))
    rgb = _decode(png)[..., :3]
    H, W = rgb.shape[:2]
    mask, s = _small_mask(rgb)
    h, w = mask.shape
    has_fg = bool(mask.any())

    def edges(n: int, size: int, prof: Optional[np.ndarray]) -> Tuple[List[int], List[float]]:
        if prof is None:
            cuts = [(int(round(k * size / n)), 0.0) for k in range(1, n)]
        else:
            cuts = _find_cuts(prof, n, search)
        return [0] + [c for c, _ in cuts] + [size], [0.0] + [f for _, f in cuts] + [0.0]

    ry, rf = edges(rows, h, mask.mean(axis=1) if has_fg else None)
    tiles: List[bytes] = []
    for r in range(rows):
        band = mask[ry[r]:ry[r + 1]]
        cx, cf = edges(cols, w, band.mean(axis=0) if has_fg and band.size else None)
        for c in range(cols):
            sub = band[:, cx[c]:cx[c + 1]]
            fy0, fy1 = int(round(ry[r] / s)), int(round(ry[r + 1] / s))
            fx0, fx1 = int(round(cx[c] / s)), int(round(cx[c + 1] / s))
            if r == rows - 1:
                fy1 = H
            if c == cols - 1:
                fx1 = W
            cell = rgb[fy0:max(fy1, fy0 + 1), fx0:max(fx1, fx0 + 1)]
            cut = max(rf[r], rf[r + 1], cf[c], cf[c + 1])
            tile = _trim_pad(cell, sub if sub.size else None, s, pad)
            tiles.append(_encode(tile, {CUT_KEY: round(float(cut), 4)}))
    return tiles


def compose_grid(pngs: Sequence[bytes], rows: int = 2, cols: int = 2, *,
                 gutter: int = 16, px: int = 1024) -> bytes:
    """Inverse of :func:`split_grid` (mock sheets, tests): each tile fitted into
    its cell (aspect kept, centred) on a white ``px`` square with ``gutter`` px
    of white around and between cells. Missing tiles leave their cell blank."""
    from PIL import Image
    rows, cols = max(1, int(rows)), max(1, int(cols))
    cw = max(1, (px - gutter * (cols + 1)) // cols)
    ch = max(1, (px - gutter * (rows + 1)) // rows)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    for i, png in enumerate(list(pngs)[:rows * cols]):
        im = Image.fromarray(np.asarray(_decode(png)[..., :3], np.uint8))
        k = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.BILINEAR)
        r, c = divmod(i, cols)
        x = gutter + c * (cw + gutter) + (cw - im.width) // 2
        y = gutter + r * (ch + gutter) + (ch - im.height) // 2
        canvas.paste(im, (x, y))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# Checks and assignment
# ===========================================================================
#: Largest foreground fraction a cut line may have: above it, a panel (or a single
#: building that was never a sheet) was sliced.
MAX_CUT_FG = 0.05
#: Share of a tile's foreground its largest component must hold ("one building").
DOMINANT = 0.6


def tile_check(png: bytes, *, lo: float = 0.03, hi: float = 0.9) -> Tuple[bool, str]:
    """(ok, reason) for one tile; the reason names the first failed rule."""
    from scipy import ndimage
    meta = tile_meta(png)
    cut = meta.get(CUT_KEY)
    if isinstance(cut, (int, float)) and cut > MAX_CUT_FG:
        return False, f"cut line {cut:.0%} foreground (panel sliced)"
    mask, _ = _small_mask(_decode(png)[..., :3])
    frac = float(mask.mean()) if mask.size else 0.0
    if not lo <= frac <= hi:
        return False, f"foreground {frac:.0%} outside [{lo:.0%}, {hi:.0%}]"
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n > 1:
        share = float(np.bincount(lab.ravel())[1:].max()) / float(mask.sum())
        if share < DOMINANT:
            return False, f"largest component holds {share:.0%} of the foreground"
    return True, "ok"


def tiles_ok(tiles: Sequence[bytes], *, lo: float = 0.03, hi: float = 0.9) -> bool:
    """Every tile shows one building of a sensible size and was not sliced.

    Rules per tile: cut-line foreground <= :data:`MAX_CUT_FG` (from the tile's
    ``blockgen.cut_fg``, when it has one), foreground fraction in [lo, hi], and the
    largest connected component >= :data:`DOMINANT` of the foreground.
    """
    return bool(tiles) and all(tile_check(t, lo=lo, hi=hi)[0] for t in tiles)


def cyclic_orders(order: Sequence[str]) -> List[Tuple[str, ...]]:
    """The candidate panel -> view assignments: 4 rotations x 2 directions.

    ``order[i]`` is the view of panel i. Rotation k shifts every panel by k views;
    the reversed direction walks the views the other way round (a sheet drawn
    turning left instead of right). The identity comes first, so a tie keeps the
    prompt's order. Duplicates (short or repeating orders) are dropped.
    """
    o = tuple(order)
    n = len(o)
    out: List[Tuple[str, ...]] = []
    for direction in (1, -1):
        for k in range(n):
            cand = tuple(o[(k + direction * i) % n] for i in range(n))
            if cand not in out:
                out.append(cand)
    return out


def _sim_table(tiles: Sequence[bytes], renders: Mapping[str, bytes]):
    from blockgen.agentic.scoring import silhouette_iou
    cache: Dict[Tuple[int, str], float] = {}

    def sim(i: int, v: str) -> Optional[float]:
        if v not in renders or i >= len(tiles):
            return None
        if (i, v) not in cache:
            cache[(i, v)] = float(silhouette_iou(tiles[i], renders[v]))
        return cache[(i, v)]
    return sim


def assignment_score(tiles: Sequence[bytes], renders: Mapping[str, bytes],
                     assignment: Sequence[str], _sim: Any = None) -> float:
    """Mean silhouette IoU of tile i vs the render at ``assignment[i]`` (panels
    whose view has no render are skipped; none at all scores 0)."""
    sim = _sim or _sim_table(tiles, renders)
    vals = [x for x in (sim(i, v) for i, v in enumerate(assignment)) if x is not None]
    return float(np.mean(vals)) if vals else 0.0


def best_assignment(tiles: Sequence[bytes], renders: Mapping[str, bytes],
                    order: Sequence[str]) -> Tuple[Tuple[str, ...], float]:
    """The cyclic order whose views best match the tiles, and its mean IoU.

    ``renders`` maps view name -> render PNG of a build (the first build). Score =
    :func:`assignment_score`, i.e. :func:`scoring.silhouette_iou` (texture-blind,
    so the dream's picture style does not decide which way the building faces).
    Ties keep the earlier candidate, so the prompt's order wins a tie.
    """
    sim = _sim_table(tiles, renders)
    best: Tuple[Tuple[str, ...], float] = (tuple(order), -1.0)
    for cand in cyclic_orders(order):
        val = assignment_score(tiles, renders, cand, sim)
        if val > best[1] + 1e-12:
            best = (cand, val)
    return best[0], max(0.0, best[1])


# ===========================================================================
# Review turn
# ===========================================================================
SHEET_RUBRIC = """\
{tag}
You review a generated TURNAROUND SHEET: a 2 by 2 grid of four panels that will be
cut apart and used as the TARGET views of ONE Minecraft build on a {W} x {H} x {D}
block canvas (x by y-up by z). Expected panels:
{panels}
Judge it on:
- consistent: the SAME building in all four panels (massing, roof, materials,
  colours), each panel rotated a further 90 degrees, same scale and camera height.
- layout: exactly four panels separated by clear white gutters; no panel cut off,
  no text, no extra panels.
- buildable: can it be built from full cube blocks at this size?
Give an overall score from 0 (useless) to 10 (ideal target).

Reply as JSON with exactly these keys:
{{"consistent": true/false, "score": 0-10, "issues": ["..."],
  "revised_prompt": "a rewritten description of the building only (no camera, layout, style or background words) that fixes the issues, or empty if none"}}"""


def sheet_review(ep: Any, png: bytes, order: Sequence[str]) -> Message:
    """Dream critic turn for a sheet (contains :data:`SHEET_REVIEW_TAG`, ends with
    ``turns.JSON_RULE``)."""
    from blockgen.agentic.turns import JSON_RULE
    w, h, d = tuple(ep.size)
    panels = "\n".join(f"  panel {i + 1} ({PANEL_NAMES[i] if i < len(PANEL_NAMES) else i}): "
                       f"view {v}" for i, v in enumerate(order))
    text = (f"Build request: {ep.prompt}\n\n"
            + SHEET_RUBRIC.format(tag=SHEET_REVIEW_TAG, W=w, H=h, D=d, panels=panels))
    return Message("user", [Part("text", text=text), Part("text", text="SHEET TO REVIEW:"),
                            Part("image", image_png=png), Part("text", text=JSON_RULE)])


def normalize_sheet_verdict(v: Mapping[str, Any]) -> Dict[str, Any]:
    """Typed fields with safe defaults (a missing 'consistent' is no complaint)."""
    c = v.get("consistent", True)
    if isinstance(c, str):
        c = c.strip().lower() in ("true", "yes", "y", "1")
    try:
        score = float(str(v.get("score", 0)).split("/")[0].strip())
    except ValueError:
        score = 0.0
    issues = v.get("issues") or []
    return {"consistent": bool(c), "score": score,
            "issues": [str(i) for i in issues] if isinstance(issues, list) else [str(issues)],
            "revised_prompt": str(v.get("revised_prompt") or "").strip()}


__all__ = ["CUT_KEY", "DOMINANT", "MAX_CUT_FG", "PANEL_NAMES", "SHEET_REVIEW_TAG",
           "SHEET_RUBRIC", "SHEET_TEMPLATE", "assignment_score", "best_assignment", "compose_grid",
           "cyclic_orders", "normalize_sheet_verdict", "sheet_review", "split_grid",
           "tile_check", "tile_meta", "tiles_ok"]
