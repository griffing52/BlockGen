"""Scorers and accept policies: how the loop decides whether a build got better.

A **scorer** is a registered function ``(ep, att, **params) -> ScoreTerm`` with a
value nominally in [0, 1]. The Episode combines terms with recipe weights and
gates (``core.Score``); this module only measures. Each scorer declares two meta
flags the Episode reads: ``needs_dream`` (its value depends on the dream images,
so the Score is stamped with ``dream_id`` and re-anchored after a re-dream) and
``needs_renders``.

=============  =========  ==========  ===================================================
name           dream?     renders?    what
=============  =========  ==========  ===================================================
embed          yes        yes         SigLIP-B (default) cosine dream vs matched render,
                                      calibrated against a random-pair baseline
silhouette     yes        yes         IoU of bbox-normalized foreground masks (texture-blind)
color_lab      yes        yes         Lab histogram intersection of the foregrounds
pixel          yes        yes         numpy downsample + colour histogram cosine (mock twin
                                      of embed / clip_text; no model weights)
clip_text      no         yes         CLIPScore render vs prompt (CLIP-B/16)
material       no         no          materials named in the prompt that the build uses
                                      (the bench's own metric: warns, see below)
validity       no         no          largest 26-connected component / all blocks
scale          no         no          player-scale heuristic: headroom over covered floor,
                                      1-wide openings a player fits (diagnostic)
vlm            yes        yes         LLM judge (role 'judge'): score/10 + a fix-hint note
=============  =========  ==========  ===================================================

Why the in-loop image backbone is SigLIP-B and not DINOv2/CLIP-L: the bench's
pillars live in those two feature spaces. Optimizing the loop in the space it is
later graded in is Goodhart's law with extra steps. ``material`` is the bench's
``faithfulness.material_agreement`` verbatim, so it warns on use and tags its note.

**Dream pairing.** For every scored :class:`~blockgen.agentic.core.DreamSpec`
(``view`` is not None) the latest dream version is compared with the render of
``ds.view`` and of every ``ds.view_search`` view, and the dream's value is the max
over them: an image model's "45 degree isometric" is approximate, and the max
absorbs which side it actually drew. Dream values are averaged with ``ds.weight``.

**Foregrounds.** Renders sit on pure white; dreams sit on "white" that is really
off-white, vignetted, gradient or shadowed. :func:`foreground_mask` therefore
segments the background as the border-connected region near a smooth backdrop
FIELD (a quadratic surface fitted to the backdrop, exact for a uniform render
border), with a tolerance adapted to the backdrop's own noise, then grows it into
soft same-hue shadows. White windows inside a house stay foreground (they do not
touch the border), and a vignette, a sky gradient or a contact shadow does not.

An **accept policy** is ``(prev_accepted, candidate) -> (bool, reason)``. It
decides which world the loop continues from; it does not decide what the build
returns (``select`` does).
"""

from __future__ import annotations

import functools
import hashlib
import math
import random
import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.core import (Attempt, DreamSpec, EmptyBuild, ScoreTerm, get_view,
                                   register)
from blockgen.agentic.render import png_to_rgb
from blockgen.utils.data import Structure

#: Backbones the bench grades with. Using them in the loop couples the optimizer to
#: the metric, so choosing one warns (it is allowed: ablations may want it).
BENCH_BACKBONES = frozenset({"dinov2b", "clipL"})

#: Views that dream-free image scorers (clip_text, pixel's no-dream proxy, vlm
#: rubric='prompt') read by default. FIXED on purpose (== episode.DEFAULT_SCORE_VIEWS):
#: ``att.renders`` grows as feedback turns render extra views (legacy_a/b, cutaway),
#: so "every render the attempt has" would score two attempts over different view
#: sets, and a dream-independent final_score would no longer compare like with like.
TEXT_VIEWS: Tuple[str, ...] = ("iso", "iso_back")

#: Working resolution for foreground segmentation. Masks are compared at 64^2, so
#: segmenting a 1024^2 dream at full size buys nothing but time and noise.
_SEG_PX = 256

#: Default ``thresh`` of the foreground scorers: a dream backdrop's residual
#: tolerance is at least ``255 - FG_THRESH`` (6). It was 245 (10) when the backdrop
#: was one median colour; with a fitted backdrop field the true-backdrop residual
#: of a denoised dream stays under ~6, and 10 ate light (quartz) roofs.
FG_THRESH = 249


# ===========================================================================
# Image utilities
# ===========================================================================
def _border(img: np.ndarray, w: int = 2) -> np.ndarray:
    return np.concatenate([img[:w].reshape(-1, img.shape[-1]), img[-w:].reshape(-1, img.shape[-1]),
                           img[:, :w].reshape(-1, img.shape[-1]),
                           img[:, -w:].reshape(-1, img.shape[-1])])


def _shrink(rgb: np.ndarray, px: int) -> np.ndarray:
    """Downscale so the long side is at most ``px`` (box filter, denoises dreams)."""
    h, w = rgb.shape[:2]
    if max(h, w) <= px:
        return rgb
    from PIL import Image
    s = px / float(max(h, w))
    im = Image.fromarray(np.ascontiguousarray(rgb)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.BOX)
    return np.asarray(im, np.uint8)


def _poly_terms(ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> np.ndarray:
    """Quadratic surface basis (1, x, y, x^2, xy, y^2) in [-1, 1] image coordinates."""
    x = xs / max(1.0, w - 1.0) * 2.0 - 1.0
    y = ys / max(1.0, h - 1.0) * 2.0 - 1.0
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)


def background_field(img: np.ndarray, sample: Optional[np.ndarray] = None, *,
                     iters: int = 4) -> Tuple[np.ndarray, float]:
    """Smooth backdrop estimate -> (field (H, W, 3), noise).

    A per-channel quadratic surface fitted to the ``sample`` pixels (default: the
    2-px border), trimmed so build or shadow pixels touching the border do not bend
    it. ``noise`` is the median residual of the inliers. A vignette or a sky
    gradient is a smooth field, not one colour: a single median colour turns the
    bright centre of a vignette into "foreground" and, through a tolerance widened
    to cover the gradient, swallows a light roof.
    """
    h, w = img.shape[:2]
    if sample is None:
        sample = np.zeros((h, w), bool)
        sample[:2], sample[-2:], sample[:, :2], sample[:, -2:] = True, True, True, True
    ys, xs = np.nonzero(sample)
    if len(ys) > 20000:                                   # deterministic subsample
        keep = np.linspace(0, len(ys) - 1, 20000).astype(np.int64)
        ys, xs = ys[keep], xs[keep]
    A = _poly_terms(ys.astype(np.float64), xs.astype(np.float64), h, w)
    v = img[ys, xs].astype(np.float64)
    use = np.ones(len(v), bool)
    coef = np.zeros((A.shape[1], 3))
    coef[0] = np.median(v, axis=0)
    res = np.abs(A @ coef - v).max(-1)
    for _ in range(max(1, iters)):
        if use.sum() < A.shape[1] * 2:
            break
        coef = np.linalg.lstsq(A[use], v[use], rcond=None)[0]
        res = np.abs(A @ coef - v).max(-1)
        cut = 3.0 * float(np.median(res[use])) + 3.0
        new = res <= cut
        if (new == use).all():
            break
        use = new
    noise = float(np.median(res[use])) if use.any() else float(np.median(res))
    yy, xx = np.mgrid[0:h, 0:w]
    field = _poly_terms(yy.astype(np.float64), xx.astype(np.float64), h, w) @ coef
    return field, noise


def _border_connected(near: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    lab, _ = ndimage.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    return np.isin(lab, edge)


#: Shadow growth: how much darker than the backdrop (Lab L) a shadow may get, how
#: much its hue may drift, how sharp it may be (L per px after a sigma-1.5 blur),
#: and from which fraction of the build's height down it may lie.
SHADOW_MAX_DL = 40.0
SHADOW_MAX_GRAD = 2.5
SHADOW_FROM = 0.5


def _grow_shadow(img: np.ndarray, field: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Grow the background into a soft contact or drop shadow under the build.

    A shadow on a neutral backdrop is a darker, same-hue, smooth region: and so is
    a quartz roof or a grey wall. Two things separate them. Smoothness: the
    shadow fades with no edge, while the build meets it (and the backdrop) along a
    hard edge, which stops the growth. Position: growth is limited to the lower
    part of the build's bbox (``SHADOW_FROM``), where contact shadows fall, so a
    light roof is never reachable.
    """
    from scipy import ndimage

    ys = np.nonzero((~bg).any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    lab = rgb_to_lab(np.clip(img, 0, 255))
    flab = rgb_to_lab(np.clip(field, 0, 255))
    dl = flab[..., 0] - lab[..., 0]                     # > 0: darker than the backdrop
    dchroma = np.hypot(lab[..., 1] - flab[..., 1], lab[..., 2] - flab[..., 2])
    sl = ndimage.gaussian_filter(lab[..., 0], 1.5)
    grad = np.hypot(ndimage.sobel(sl, 0), ndimage.sobel(sl, 1)) / 8.0
    shade = ((dl > -1.0) & (dl <= SHADOW_MAX_DL)
             & (dchroma <= 5.0 + 0.2 * np.maximum(dl, 0.0))
             & (grad <= SHADOW_MAX_GRAD))
    shade[:y0 + int(SHADOW_FROM * (y1 - y0))] = False
    return ndimage.binary_propagation(bg, mask=bg | shade)


def foreground_mask(rgb: np.ndarray, thresh: int = FG_THRESH, *, min_frac: float = 5e-4,
                    max_tol: int = 40, shadows: bool = True,
                    smooth: float = 1.0) -> np.ndarray:
    """Boolean foreground mask of a build (render) or an imagined build (dream).

    Background = pixels connected to the image border that sit within a tolerance
    of the BACKDROP FIELD there. Two regimes:

    * **Synthetic backdrop** (every border pixel the same colour: a render, or an
      oracle dream that is one): the backdrop is exactly that colour, so the
      tolerance is 1, and the backdrop is opened with a 3x3 square. Lit snow is
      within 10 of white (about half its texels ARE white), and the old tolerance
      of 10 merged a snow roof into the background.
    * **Dream backdrop**: the backdrop is a quadratic surface fitted to the border
      and then to the border-connected pixels near it (:func:`background_field`),
      so a vignette or a sky gradient is modelled rather than absorbed by a wide
      tolerance. The image is median-filtered (radius ``smooth`` px) first, so the
      tolerance tracks a denoised backdrop. The tolerance on the residual is ``255 - thresh`` (10 by default)
      or 3 x the border's median residual + 6 (its noise), capped at ``max_tol``.
      With ``shadows``, the background then grows into soft, darker, same-hue
      pixels in the lower half of the build's bbox (a contact shadow), see
      :func:`_grow_shadow`.

    Foreground specks smaller than ``min_frac`` of the image are dropped (dream
    noise, a stray grass tuft). Works at the input resolution.
    """
    from scipy import ndimage

    img = np.asarray(rgb)[..., :3].astype(np.float64)
    border = _border(img)
    if float(np.abs(border - border[0]).max()) <= 0.0:
        field = np.broadcast_to(border[0], img.shape)
        tol = 1.0
        # Opened with a 3x3 square: the real backdrop is wide, but ~46% of a snow
        # block's texels are pure white too, and without the opening the backdrop
        # leaks into a snow roof through runs of them along its edge.
        near = ndimage.binary_opening(np.abs(img - field).max(-1) <= tol,
                                      structure=np.ones((3, 3), bool))
        bg = _border_connected(near)
        shadows = False
    else:
        # Denoise first: the tolerance tracks the backdrop's noise, and on a raw
        # noisy dream it grows past the contrast of a light roof. A median filter
        # (not a blur): it cuts the noise without moving edges outward.
        if smooth > 0:
            n = 2 * int(round(smooth)) + 1
            img = ndimage.median_filter(img, size=(n, n, 1))
        field, noise = background_field(img)
        for _ in range(2):
            tol = float(np.clip(max(255 - int(thresh), 3.0 * noise + 3.0), 0, max_tol))
            bg = _border_connected(np.abs(img - field).max(-1) <= tol)
            if bg.sum() < 64:
                break
            field, noise = background_field(img, bg)  # refit on the whole backdrop
        tol = float(np.clip(max(255 - int(thresh), 3.0 * noise + 3.0), 0, max_tol))
        bg = _border_connected(np.abs(img - field).max(-1) <= tol)
    if shadows and (~bg).any():
        bg = _grow_shadow(img, field, bg)
    fg = ~bg
    flab, n = ndimage.label(fg)
    if n == 0:
        return fg
    sizes = np.bincount(flab.ravel())
    keep = sizes >= max(1.0, min_frac * fg.size)
    keep[0] = False
    return keep[flab]


@functools.lru_cache(maxsize=256)
def _mask_png(png: bytes, thresh: int) -> np.ndarray:
    m = foreground_mask(_shrink(png_to_rgb(png), _SEG_PX), thresh)
    m.setflags(write=False)
    return m


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _square(img: np.ndarray, fill: Any) -> np.ndarray:
    """Pad to a centred square (keeps aspect ratio, which is part of a silhouette)."""
    h, w = img.shape[:2]
    n = max(h, w)
    out = np.full((n, n) + img.shape[2:], fill, dtype=img.dtype)
    y0, x0 = (n - h) // 2, (n - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def normalized_mask(mask: np.ndarray, px: int = 64) -> np.ndarray:
    """Crop a mask to its bbox, pad square, resize to ``px``^2 (bool).

    Bbox normalization makes silhouettes translation- and scale-invariant: the
    renderer fits each build to its frame, the image model frames its own way.
    """
    from PIL import Image
    bb = _bbox(mask)
    if bb is None:
        return np.zeros((px, px), bool)
    y0, y1, x0, x1 = bb
    sq = _square(mask[y0:y1, x0:x1].astype(np.uint8) * 255, 0)
    im = Image.fromarray(sq).resize((px, px), Image.BILINEAR)
    return np.asarray(im) >= 128


def fg_crop(rgb: np.ndarray, mask: Optional[np.ndarray] = None, *, size: int = 224,
            pad: float = 0.04) -> np.ndarray:
    """Crop to the foreground bbox (+pad), centre on a white square, resize.

    ``mask`` may be at a lower resolution than ``rgb`` (it is rescaled). Without a
    foreground the whole image is used, never an empty crop.
    """
    from PIL import Image
    rgb = np.asarray(rgb)[..., :3]
    h, w = rgb.shape[:2]
    if mask is None:
        mask = foreground_mask(_shrink(rgb, _SEG_PX))
    bb = _bbox(mask)
    if bb is not None:
        sy, sx = h / mask.shape[0], w / mask.shape[1]
        y0, y1, x0, x1 = bb[0] * sy, bb[1] * sy, bb[2] * sx, bb[3] * sx
        p = pad * max(y1 - y0, x1 - x0)
        y0, y1 = int(max(0, y0 - p)), int(min(h, math.ceil(y1 + p)))
        x0, x1 = int(max(0, x0 - p)), int(min(w, math.ceil(x1 + p)))
        rgb = rgb[y0:y1, x0:x1]
    sq = _square(np.ascontiguousarray(rgb), 255)
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.BICUBIC), np.uint8)


@functools.lru_cache(maxsize=256)
def _prep_png(png: bytes, size: int, crop_fg: bool) -> np.ndarray:
    rgb = png_to_rgb(png)
    if crop_fg:
        out = fg_crop(rgb, _mask_png(png, FG_THRESH), size=size)
    else:
        from PIL import Image
        out = np.asarray(Image.fromarray(_square(np.ascontiguousarray(rgb), 255))
                         .resize((size, size), Image.BICUBIC), np.uint8)
    out.setflags(write=False)
    return out


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIE Lab (D65), float. L in [0, 100]."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab_hist(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalized 3-D Lab histogram of the masked pixels (zeros if none)."""
    px = np.asarray(rgb)[mask]
    if len(px) == 0:
        return np.zeros(bins ** 3)
    lab = rgb_to_lab(px)
    h, _ = np.histogramdd(lab, bins=(bins, bins, bins),
                          range=((0, 100), (-110, 110), (-110, 110)))
    return h.ravel() / max(1.0, h.sum())


# ===========================================================================
# Pairwise similarities (image a = dream, image b = render), PNG in
# ===========================================================================
def silhouette_iou(a: bytes, b: bytes, thresh: int = FG_THRESH, px: int = 64) -> float:
    ma = normalized_mask(_mask_png(a, thresh), px)
    mb = normalized_mask(_mask_png(b, thresh), px)
    union = float((ma | mb).sum())
    return float((ma & mb).sum()) / union if union else 0.0


def color_lab_sim(a: bytes, b: bytes, bins: int = 8, thresh: int = FG_THRESH) -> float:
    ha = lab_hist(_shrink(png_to_rgb(a), _SEG_PX), _mask_png(a, thresh), bins)
    hb = lab_hist(_shrink(png_to_rgb(b), _SEG_PX), _mask_png(b, thresh), bins)
    return float(np.minimum(ha, hb).sum())


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def pixel_sim(a: bytes, b: bytes, px: int = 32) -> float:
    """Mock image similarity: mean of a 32^2 "ink" cosine (255 - pixel, so the
    white background contributes nothing) and a 64-bin colour histogram cosine of
    the foregrounds. Both are non-negative, so the value is in [0, 1]."""
    ia = 255.0 - _prep_png(a, px, True).astype(np.float64)
    ib = 255.0 - _prep_png(b, px, True).astype(np.float64)
    shape_cos = _cos(ia.ravel(), ib.ravel())

    def hist(png: bytes) -> np.ndarray:
        rgb = _shrink(png_to_rgb(png), _SEG_PX)
        q = (rgb[_mask_png(png, FG_THRESH)] // 64).astype(np.int64)
        return np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(float)

    return float(np.clip(0.5 * shape_cos + 0.5 * _cos(hist(a), hist(b)), 0.0, 1.0))


# ===========================================================================
# Episode access (scorers touch only EpisodeLike + Attempt)
# ===========================================================================
def structure_of(att: Attempt) -> Optional[Structure]:
    """The attempt's world as a cropped Structure (None if it has no world)."""
    w = getattr(att, "world", None)
    return None if w is None else w.to_structure(crop=True)


def scored_dreams(ep: Any) -> List[Tuple[DreamSpec, Any]]:
    """(spec, latest DreamVersion) for every dream that is compared with a view."""
    out = []
    for ds in getattr(ep.kit, "dreams", None) or []:
        if ds.view is None:
            continue
        dv = ep.latest_dream(ds.key)
        if dv is not None:
            out.append((ds, dv))
    return out


def ensure_renders(ep: Any, att: Attempt, views: Sequence[str]) -> Dict[str, bytes]:
    """``att.renders`` restricted to ``views``, rendering (and storing) any missing
    one through ``ep.render``. An empty build has no renders."""
    missing = [v for v in dict.fromkeys(views) if v not in att.renders]
    if missing and att.blocks > 0 and getattr(ep.kit, "renderer", None) is not None:
        try:
            att.renders.update(ep.render(att, missing))
        except EmptyBuild:
            pass
    return {v: att.renders[v] for v in views if v in att.renders}


def paired_term(ep: Any, att: Attempt, sim, *, reduce: str = "paired", name: str = "") -> ScoreTerm:
    """Dream-vs-render term with the pairing rule in the module docstring.

    ``sim(dream_png, render_png) -> float``. ``reduce``: 'paired' = max over the
    dream's own view + view_search; 'mean' = mean over those; 'max' = max over
    the union of every scored dream's candidate views. ``per_view`` holds every
    candidate's value as ``'<dream key>@<view>'`` (the key turns.matched_view reads).
    """
    if reduce not in ("paired", "mean", "max"):
        raise ValueError(f"{name}: reduce must be paired|mean|max, got {reduce!r}")
    pairs = scored_dreams(ep)
    if not pairs:
        return ScoreTerm(0.0, note=f"{name}: no scored dream")
    if att.blocks <= 0:
        return ScoreTerm(0.0, note=f"{name}: empty build")
    # 'max' searches the union of every scored dream's candidate views: a FIXED set
    # per recipe. (Not "whatever att.renders holds": that grows as feedback turns
    # render more views, so two attempts would be scored over different sets.)
    union = tuple(dict.fromkeys(v for ds, _ in pairs
                                for v in (ds.view,) + tuple(ds.view_search)))
    num = den = 0.0
    per_view: Dict[str, float] = {}
    notes = []
    for ds, dv in pairs:
        cands = (union if reduce == "max"
                 else (ds.view,) + tuple(v for v in ds.view_search if v != ds.view))
        renders = ensure_renders(ep, att, cands)
        if not renders:
            continue
        vals = {v: float(np.clip(sim(dv.png, png), 0.0, 1.0)) for v, png in renders.items()}
        best_view = max(vals, key=lambda k: (vals[k], k == ds.view))
        val = float(np.mean(list(vals.values()))) if reduce == "mean" else vals[best_view]
        # Every candidate view, keyed '<dream>@<view>': turns.matched_view sums these
        # across scorers to pick the camera the feedback turn shows as YOURS.
        for v, x in vals.items():
            per_view[f"{ds.key}@{v}"] = round(x, 4)
        w = max(float(ds.weight), 0.0)
        num += w * val
        den += w
        notes.append(f"{ds.key} {val:.2f}" + (f" (best view {best_view})"
                                             if best_view != ds.view else ""))
    if den <= 0:
        return ScoreTerm(0.0, per_view=per_view, note=f"{name}: nothing rendered")
    return ScoreTerm(float(np.clip(num / den, 0.0, 1.0)), per_view=per_view,
                     note=f"{name}: " + ", ".join(notes))


# ===========================================================================
# Model-backed features (lazy, cached)
# ===========================================================================
_FEAT_CACHE: "OrderedDict[Tuple[str, ...], np.ndarray]" = OrderedDict()
_FEAT_LOCK = threading.Lock()
_FEAT_MAX = 4096
_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        warnings.warn(msg, RuntimeWarning)


#: Free GPU memory (bytes) below which the default device is the CPU. The in-loop
#: backbones need ~1-2 GB; a GPU already held by another job (a vLLM server, a
#: training run) must degrade to slower CPU scoring, not crash every build.
MIN_FREE_GPU_BYTES = 2 * 1024 ** 3


def resolve_device(device: Optional[str] = None) -> str:
    """``device`` if given, else cuda when it has room for a backbone, else cpu."""
    if device:
        return device
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - driver hiccup: be conservative
        return "cpu"
    if free < MIN_FREE_GPU_BYTES:
        warnings.warn(f"only {free / 1024 ** 3:.1f} GiB free on the GPU; in-loop scorers "
                      f"run on the CPU (pass device=cuda to force)", RuntimeWarning)
        return "cpu"
    return "cuda"


def backbone_device(backbone: str, device: Optional[str] = None) -> str:
    """The device to send ``backbone``'s inputs to.

    ``eval.bench.features._MODELS`` caches each backbone by NAME, on whichever
    device loaded it first; later calls get that same model whatever device they
    ask for. Sending inputs to the requested device instead crashes a battery
    whose arms mix ``embed?device=cpu`` with the default (cuda) with a tensor
    type mismatch. So once a backbone is loaded, its own device wins (with a
    one-time warning when that differs from the request).
    """
    want = resolve_device(device)
    try:
        from blockgen.eval.bench import features as ft
        entry = ft._MODELS.get(backbone)
    except Exception:                       # bench features unavailable: nothing loaded
        return want
    if entry is None:
        return want
    model = entry[0] if isinstance(entry, tuple) else entry
    try:
        have = str(next(model.parameters()).device)
    except Exception:
        return want
    if have.split(":")[0] != want.split(":")[0]:
        _warn_once(f"{backbone}: already loaded on {have}; using it instead of the "
                   f"requested {want!r} (the model cache is per backbone, not per device)")
    return have


def _cache_get(key: Tuple[str, ...]) -> Optional[np.ndarray]:
    with _FEAT_LOCK:
        v = _FEAT_CACHE.get(key)
        if v is not None:
            _FEAT_CACHE.move_to_end(key)
        return v


def _cache_put(key: Tuple[str, ...], v: np.ndarray) -> None:
    with _FEAT_LOCK:
        _FEAT_CACHE[key] = v
        while len(_FEAT_CACHE) > _FEAT_MAX:
            _FEAT_CACHE.popitem(last=False)


def image_features(pngs: Sequence[bytes], backbone: str = "siglipB", *, crop_fg: bool = True,
                   device: Optional[str] = None) -> np.ndarray:
    """L2-normalized ``[N, D]`` features via ``eval.bench.features.embed_images``.

    The model is loaded once per process (``features._MODELS``); features are cached
    per image content, so the dream is embedded once per build, not once per attempt.
    """
    keys = [("img", backbone, str(crop_fg), hashlib.sha256(p).hexdigest()) for p in pngs]
    out: List[Optional[np.ndarray]] = [_cache_get(k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        from blockgen.eval.bench.features import embed_images
        imgs = [np.array(_prep_png(pngs[i], 224, crop_fg)) for i in todo]
        feats = embed_images(imgs, backbone, backbone_device(backbone, device))
        for i, f in zip(todo, feats):
            out[i] = np.asarray(f, np.float32)
            _cache_put(keys[i], out[i])
    return np.stack(out) if out else np.zeros((0, 0), np.float32)  # type: ignore[arg-type]


def text_features(texts: Sequence[str], backbone: str = "clipB16", *,
                  device: Optional[str] = None) -> np.ndarray:
    keys = [("txt", backbone, hashlib.sha256(t.encode()).hexdigest()) for t in texts]
    out: List[Optional[np.ndarray]] = [_cache_get(k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        from blockgen.eval.bench.faithfulness import embed_texts
        feats = embed_texts([texts[i] for i in todo], backbone,
                            backbone_device(backbone, device))
        for i, f in zip(todo, feats):
            out[i] = np.asarray(f, np.float32)
            _cache_put(keys[i], out[i])
    return np.stack(out) if out else np.zeros((0, 0), np.float32)  # type: ignore[arg-type]


def embed_baseline(backbone: str) -> float:
    """Shipped random-pair cosine for ``backbone`` (calibrate.BASELINES), 0 if unknown."""
    from blockgen.agentic.calibrate import BASELINES
    if backbone not in BASELINES:
        _warn_once(f"embed: no calibrated baseline for backbone {backbone!r}; using raw "
                   f"cosine (run `python -m blockgen.agentic.calibrate --scorer "
                   f"embed?backbone={backbone}&baseline=0`)")
    return float(BASELINES.get(backbone, 0.0))


# ===========================================================================
# Scorers
# ===========================================================================
@register("scorer", "embed", needs_dream=True, needs_renders=True, gpu=True)
def embed(ep: Any, att: Attempt, *, backbone: str = "siglipB", reduce: str = "paired",
          crop_fg: bool = True, baseline: Optional[float] = None,
          device: Optional[str] = None) -> ScoreTerm:
    """Image-embedding similarity of dream and matched render, calibrated to [0, 1].

    value = clip((cos - baseline) / (1 - baseline), 0, 1), where ``baseline`` is the
    backbone's mean cosine between unrelated build renders and dreams (shipped in
    calibrate.BASELINES; pass ``baseline=0`` for the raw cosine). Without it every
    pair of houses scores ~0.8 and the loop has nothing to climb.
    """
    if backbone in BENCH_BACKBONES:
        _warn_once(f"embed: backbone {backbone!r} is a bench pillar backbone; optimizing "
                   f"in it couples the loop to the metric")
    base = embed_baseline(backbone) if baseline is None else float(baseline)

    def sim(a: bytes, b: bytes) -> float:
        fa, fb = image_features([a, b], backbone, crop_fg=crop_fg, device=device)
        cos = float(fa @ fb)
        return (cos - base) / max(1e-6, 1.0 - base)

    return paired_term(ep, att, sim, reduce=reduce, name=f"embed[{backbone}]")


@register("scorer", "silhouette", needs_dream=True, needs_renders=True)
def silhouette(ep: Any, att: Attempt, *, thresh: int = FG_THRESH, px: int = 64,
               reduce: str = "paired") -> ScoreTerm:
    """Texture-blind shape match: IoU of bbox-normalized foreground masks at 64^2.

    Blind to materials on purpose: the dream's textures will never match block
    textures, but its outline can, so this term survives the domain gap that
    hurts every appearance score.
    """
    return paired_term(ep, att, lambda a, b: silhouette_iou(a, b, thresh, px),
                       reduce=reduce, name="silhouette")


@register("scorer", "color_lab", needs_dream=True, needs_renders=True)
def color_lab(ep: Any, att: Attempt, *, bins: int = 8, thresh: int = FG_THRESH,
              reduce: str = "paired") -> ScoreTerm:
    """Palette match: Lab-histogram intersection of dream and render foregrounds."""
    return paired_term(ep, att, lambda a, b: color_lab_sim(a, b, bins, thresh),
                       reduce=reduce, name="color_lab")


@register("scorer", "pixel", needs_dream=True, needs_renders=True)
def pixel(ep: Any, att: Attempt, *, px: int = 32, reduce: str = "paired") -> ScoreTerm:
    """Offline mock of embed/clip_text: downsampled-ink + colour-histogram cosine.

    With no scored dream (a mocked ``clip_text``) it falls back to the mean
    foreground coverage of the renders: deterministic, model-free, and still
    larger for a bigger build, so mock loops have something to compare.
    """
    if scored_dreams(ep):
        return paired_term(ep, att, lambda a, b: pixel_sim(a, b, px), reduce=reduce,
                           name="pixel")
    if att.blocks <= 0:
        return ScoreTerm(0.0, note="pixel: empty build")
    renders = ensure_renders(ep, att, TEXT_VIEWS)
    if not renders:
        return ScoreTerm(0.0, note="pixel: no renders")
    cover = {v: float(_mask_png(p, FG_THRESH).mean()) for v, p in renders.items()}
    return ScoreTerm(float(np.mean(list(cover.values()))), per_view=cover,
                     note="pixel: no dream, foreground-coverage proxy")


@register("scorer", "clip_text", needs_dream=False, needs_renders=True, gpu=True)
def clip_text(ep: Any, att: Attempt, *, backbone: str = "clipB16", crop_fg: bool = True,
              views: Optional[Sequence[str]] = None, device: Optional[str] = None) -> ScoreTerm:
    """CLIPScore of the renders against the prompt: mean over views (default
    :data:`TEXT_VIEWS`) of clip(2.5 * cos, 0, 1) (Hessel et al. 2021's w = 2.5). Dream-independent, so it
    is the natural ``final_score`` across re-dreams and the no-dream control."""
    if backbone == "clipL":
        _warn_once("clip_text: clipL is the bench's faithfulness backbone; optimizing it "
                   "in the loop inflates the faithfulness pillar")
    if att.blocks <= 0:
        return ScoreTerm(0.0, note="clip_text: empty build")
    renders = ensure_renders(ep, att, list(views) if views else TEXT_VIEWS)
    if not renders:
        return ScoreTerm(0.0, note="clip_text: no renders")
    names = list(renders)
    img = image_features([renders[v] for v in names], backbone, crop_fg=crop_fg, device=device)
    txt = text_features([ep.prompt], backbone, device=device)[0]
    per = {v: float(np.clip(2.5 * float(f @ txt), 0.0, 1.0)) for v, f in zip(names, img)}
    return ScoreTerm(float(np.mean(list(per.values()))), per_view=per,
                     note=f"clip_text[{backbone}]")


@register("scorer", "material", needs_dream=False, needs_renders=False)
def material(ep: Any, att: Attempt, *, k: int = 5) -> ScoreTerm:
    """Fraction of materials named in the prompt among the build's ``k`` dominant
    ones (``faithfulness.material_agreement``). A prompt naming no material scores
    1.0: the term has nothing to check, and it must not penalize.

    This IS the bench's ``faithfulness.material_agreement`` metric, called
    verbatim: optimizing it in the loop inflates that reported number, so using it
    warns (like a bench backbone in ``embed``) and the term's note says so."""
    _warn_once("material: this is the bench's faithfulness.material_agreement metric; "
               "optimizing it in the loop inflates that reported number")
    s = structure_of(att)
    if s is None or att.blocks <= 0:
        return ScoreTerm(0.0, note="material: empty build")
    from blockgen.eval.bench.faithfulness import material_agreement
    res = material_agreement([s], [ep.prompt], k=k)
    if len(res["agreement"]) == 0:
        return ScoreTerm(1.0, note="material: prompt names no material "
                                   "[optimizes bench metric material_agreement]")
    return ScoreTerm(float(res["agreement"][0]),
                     note="material agreement [optimizes bench metric material_agreement]")


def largest_component_frac(s: Optional[Structure]) -> float:
    """Largest 26-connected component / all blocks (0 for an empty build).

    Why 26 and not 6: blocks touching at an edge or corner (a diagonal stair roof)
    stand in Minecraft, and the agent's DSL makes them routinely.
    """
    if s is None:
        return 0.0
    occ = s.occupied_mask
    n = int(occ.sum())
    if n == 0:
        return 0.0
    from scipy import ndimage
    lab, k = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    if k <= 1:
        return 1.0
    return float(np.bincount(lab.ravel())[1:].max()) / n


@register("scorer", "validity", needs_dream=False, needs_renders=False)
def validity(ep: Any, att: Attempt) -> ScoreTerm:
    """Connectivity: fraction of blocks in the largest 26-connected component.
    The standard gate against floating debris that games image scores."""
    v = largest_component_frac(structure_of(att)) if att.blocks > 0 else 0.0
    return ScoreTerm(v, note=f"validity: largest component {v:.0%} of blocks")


@register("scorer", "scale", needs_dream=False, needs_renders=False)
def scale(ep: Any, att: Attempt, *, ceiling: int = 3) -> ScoreTerm:
    """Player-scale heuristic (``scale.scale_stats``): the mean of the share of
    covered standing cells with ``ceiling`` (2 or 3) clear blocks above, and the
    share of 1-wide floor-level openings a player fits through. A covered build
    with no opening at all gets a door term of 0 (a sealed box is not a house a
    player can enter), so it cannot outscore one with a real door. 0 for an
    empty or roofless build. A diagnostic: no shipped recipe selects on it;
    ``MethodResult.metrics`` reports the same numbers for every build (there
    ``scale_door_ok`` stays None without openings)."""
    from blockgen.agentic.scale import scale_stats

    st = scale_stats(structure_of(att)) if att.blocks > 0 else {}
    head = st.get("scale_headroom3" if int(ceiling) >= 3 else "scale_headroom2")
    door = st.get("scale_door_ok")
    if head is not None and door is None:
        door = 0.0
    parts = [v for v in (head, door) if v is not None]
    v = float(sum(parts) / len(parts)) if parts else 0.0
    return ScoreTerm(v, note=(f"scale: headroom>={int(ceiling)} {head if head is not None else '-'}"
                              f", door_ok {st.get('scale_door_ok', '-')}, "
                              f"{st.get('scale_cells', 0)} covered cells"))


# --- vlm judge --------------------------------------------------------------
#: Same sentence as ``turns.JSON_RULE`` (the mock:auto LLM keys its JSON reply on
#: it); restated so this module imports without turns.py.
_FALLBACK_JSON_RULE = ("Reply with a single JSON object and nothing else: no prose, "
                       "no code fences.")

JUDGE_RUBRIC = (
    "Score how well YOURS (a render of the current voxel build) matches {what} on a "
    "0-10 scale: 10 = the same building (massing, footprint, roof shape, openings, "
    "materials and colours), 5 = right kind of building but clearly different "
    "proportions or roof, 0 = unrelated or broken. Judge the building, not the "
    "rendering style: block textures will never look like a painting.\n"
    'Return JSON: {{"score": <0-10>, "missing": ["..."], "wrong": ["..."], '
    '"fix_hint": "<one concrete instruction for the builder>"}}')


def _judge_turn(ep: Any, att: Attempt, rubric: str) -> Any:
    """The judge's user message: TARGET/YOURS image pairs (rubric 'match') or the
    renders alone (rubric 'prompt'), then the 0-10 rubric and the JSON rule.

    Not ``turns.judge``: that template asks for a list of differences (the VDM
    feedback text of ``feedback: vdm``), not for a score, so a real judge
    answering it would give every attempt 0.
    """
    from blockgen.agentic.providers import Message, Part
    try:
        from blockgen.agentic.turns import JSON_RULE as rule
    except ImportError:
        rule = _FALLBACK_JSON_RULE
    parts: List[Any] = [Part("text", text=f"Build request: {ep.prompt}")]
    pairs = scored_dreams(ep) if rubric == "match" else []
    for ds, dv in pairs:
        r = ensure_renders(ep, att, [ds.view])
        parts += [Part("text", text=f"TARGET {ds.key}:"), Part("image", image_png=dv.png)]
        if ds.view in r:
            parts += [Part("text", text=f"YOURS {ds.key} (view {ds.view}):"),
                      Part("image", image_png=r[ds.view])]
    if not pairs:
        renders = ensure_renders(ep, att, TEXT_VIEWS)
        for v, png in renders.items():
            parts += [Part("text", text=f"YOURS (view {v}):"), Part("image", image_png=png)]
    what = "the TARGET image(s)" if pairs else "the build request"
    parts.append(Part("text", text=JUDGE_RUBRIC.format(what=what) + "\n" + rule))
    return Message("user", parts)


def _listy(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v if str(x).strip())
    return str(v or "").strip()


@register("scorer", "vlm", needs_dream=True, needs_renders=True, llm=True)
def vlm(ep: Any, att: Attempt, *, rubric: str = "match") -> ScoreTerm:
    """LLM judge (role 'judge'): ``score``/10, with ``missing/wrong/fix_hint`` as the
    note, which ``feedback: vdm`` feeds back to the builder. ``rubric``: 'match'
    (renders vs dreams) or 'prompt' (renders vs the request text)."""
    if rubric not in ("match", "prompt"):
        raise ValueError(f"vlm: rubric must be match|prompt, got {rubric!r}")
    if att.blocks <= 0:
        return ScoreTerm(0.0, note="vlm: empty build")
    reply = ep.ask(_judge_turn(ep, att, rubric), role="judge", history=False,
                   json_reply=True, stage=f"judge{getattr(att, 'idx', '')}")
    if not isinstance(reply, dict) or "score" not in reply:
        return ScoreTerm(0.0, note="vlm: judge reply unparseable")
    try:
        score = float(reply.get("score"))
    except (TypeError, ValueError):
        return ScoreTerm(0.0, note=f"vlm: non-numeric score {reply.get('score')!r}")
    bits = [f"{k}: {_listy(reply.get(k))}"
            for k in ("missing", "wrong", "fix_hint", "differences") if _listy(reply.get(k))]
    return ScoreTerm(float(np.clip(score / 10.0, 0.0, 1.0)), note=" | ".join(bits))


# ===========================================================================
# Accept policies
# ===========================================================================
def _value(att: Optional[Attempt]) -> Optional[float]:
    return None if att is None or att.score is None else float(att.score.value)


@register("accept", "always")
def always(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
    """Accept every candidate (the last one always wins)."""
    return True, "always"


@register("accept", "nonempty_legacy")
def nonempty_legacy(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
    """BuildAgent's rule exactly (agent.py: ``new == 0 and old > 0``): reject only
    an empty build that would replace a non-empty one. Kept verbatim (its comment
    says "empty or emptier", the code checks only empty) for legacy parity."""
    if cand.blocks == 0 and prev is not None and prev.blocks > 0:
        return False, "empty build after a non-empty one"
    return True, "ok"


@register("accept", "nonempty")
def nonempty(prev: Optional[Attempt], cand: Attempt, *, min_frac: float = 0.5) -> Tuple[bool, str]:
    """Reject an empty build, and one that lost more than ``1 - min_frac`` of the
    current build's blocks (what the legacy comment meant by "emptier")."""
    if cand.blocks == 0:
        return False, "empty build"
    if prev is not None and prev.blocks > 0 and cand.blocks < min_frac * prev.blocks:
        return False, f"{cand.blocks} blocks < {min_frac:g} x {prev.blocks}"
    return True, "ok"


@register("accept", "improve")
def improve(prev: Optional[Attempt], cand: Attempt, *, eps: float = 0.0) -> Tuple[bool, str]:
    """Accept only a non-empty candidate whose score beats the current one by more
    than ``eps`` (the first non-empty candidate is always accepted). Unscored
    attempts fall back to the non-empty rule."""
    if cand.blocks == 0:
        return False, "empty build"
    if prev is None or prev.blocks == 0:
        return True, "first non-empty"
    a, b = _value(prev), _value(cand)
    if a is None or b is None:
        return True, "unscored"
    if b > a + eps:
        return True, f"improved {a:.3f} -> {b:.3f}"
    return False, f"no improvement {a:.3f} -> {b:.3f} (eps {eps:g})"


def _validity(att: Attempt) -> float:
    if att.score is not None and "validity" in att.score.parts:
        return float(att.score.parts["validity"])
    return largest_component_frac(structure_of(att))


@register("accept", "improve_valid")
def improve_valid(prev: Optional[Attempt], cand: Attempt, *, eps: float = 0.0,
                  max_validity_drop: float = 0.1) -> Tuple[bool, str]:
    """``improve`` plus: reject if connectivity fell by more than
    ``max_validity_drop``. Why: the cheapest way to raise an image score is
    floating decoration that matches the dream from one angle."""
    ok, why = improve(prev, cand, eps=eps)
    if not ok or prev is None or prev.blocks == 0:
        return ok, why
    va, vb = _validity(prev), _validity(cand)
    if vb < va - max_validity_drop:
        return False, f"validity dropped {va:.2f} -> {vb:.2f}"
    return ok, why


@register("accept", "anneal")
def anneal(prev: Optional[Attempt], cand: Attempt, *, t0: float = 0.05, decay: float = 0.8,
           seed: int = 0) -> Tuple[bool, str]:
    """Simulated annealing: always accept an improvement; accept a worse candidate
    with probability exp(-drop / T), T = t0 * decay**idx. Deterministic per
    (idx, seed) via ``random.Random(hash((idx, seed)))`` (int-tuple hashes are not
    salted), so reruns accept the same moves."""
    if cand.blocks == 0:
        return False, "empty build"
    if prev is None or prev.blocks == 0:
        return True, "first non-empty"
    a, b = _value(prev), _value(cand)
    if a is None or b is None:
        return True, "unscored"
    if b >= a:
        return True, f"improved {a:.3f} -> {b:.3f}"
    t = max(float(t0) * float(decay) ** max(int(cand.idx), 0), 1e-12)
    p = math.exp(-(a - b) / t)
    u = random.Random(hash((int(cand.idx), int(seed)))).random()
    if u < p:
        return True, f"anneal accept {a:.3f} -> {b:.3f} (p={p:.3f})"
    return False, f"anneal reject {a:.3f} -> {b:.3f} (p={p:.3f})"


__all__ = ["BENCH_BACKBONES", "FG_THRESH", "TEXT_VIEWS", "backbone_device", "background_field", "always", "anneal", "clip_text", "color_lab", "color_lab_sim",
           "embed", "embed_baseline", "ensure_renders", "fg_crop", "foreground_mask",
           "image_features", "improve", "improve_valid", "lab_hist", "largest_component_frac",
           "material", "nonempty", "nonempty_legacy", "normalized_mask", "paired_term",
           "pixel", "pixel_sim", "resolve_device", "rgb_to_lab", "scale", "scored_dreams",
           "silhouette", "silhouette_iou", "structure_of", "text_features", "validity", "vlm"]
