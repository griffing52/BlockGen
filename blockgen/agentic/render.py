"""Renderers for the agentic suite: ``textured`` (GL), ``flat`` (numpy/PIL), ``null``.

Every renderer implements :class:`core.Renderer`: ``render(structure, views) ->
[RGB uint8]``, one ``(px, px, 3)`` image per view on a white background. The
:class:`~blockgen.agentic.core.View` transform (cutaway, floor plan, section) is
applied to the *structure* before rendering, so no renderer knows about rooms or
roofs, and a new view is one line in ``core.VIEWS``.

Why three:

``textured``  the real thing (pyrender/EGL, real block textures). This is what the
              dream is compared against, and what the bench renders, so the loop
              optimizes the same pixels the evaluation sees.
``flat``      per-face painter's-algorithm projection with block colours, no GL.
              Same camera convention and framing as ``textured``, so silhouettes
              line up; it is what ``--mock`` and CI use.
``null``      a deterministic solid tile per view. For estimate mode and parity
              tests, where only the *shape* of the request matters.

Policy that differs from ``agent._render`` on purpose: nothing here swallows an
exception. An empty build raises :class:`~blockgen.agentic.core.EmptyBuild` (callers
score it 0 without rendering), and any GL failure raises
:class:`~blockgen.agentic.core.RenderError`. A renderer that silently returns
blanks turns a broken driver into "every build matches the dream perfectly"
(the bench learned this the hard way, see ``eval/bench/features.render_views``).
"""

from __future__ import annotations

import functools
import hashlib
import io
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blockgen.agentic.core import EmptyBuild, RenderError, View, get_view, register
from blockgen.utils.data import Structure

#: Repo root, so the texture pack resolves no matter where the process runs from.
#: (``renderer.textures.DEFAULT_PACK_DIR`` is CWD-relative: run elsewhere and every
#: block silently falls back to a solid colour.)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = REPO_ROOT / "data" / "textures" / "vanilla"

#: Above this fraction of voxels drawn with a solid-colour fallback texture, the
#: textured render no longer looks like the game; warn (once) rather than raise,
#: because a few exotic blocks in an otherwise fine build are normal.
MAX_FALLBACK_FRAC = 0.02

WHITE = (255, 255, 255)

ViewLike = Union[str, View]


# ===========================================================================
# Image helpers (shared with scoring.py / calibrate.py)
# ===========================================================================
def to_rgb(img: np.ndarray, bg: Tuple[int, int, int] = WHITE) -> np.ndarray:
    """(H, W, 3|4) array -> contiguous (H, W, 3) uint8, alpha composited onto ``bg``."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (255 * np.clip(arr, 0, 1)).astype(np.uint8) if arr.max() <= 1.0 else \
            np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        a = arr[..., 3:4].astype(np.float32) / 255.0
        arr = (arr[..., :3].astype(np.float32) * a
               + np.asarray(bg, np.float32) * (1.0 - a)).round().astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def png_bytes(img: np.ndarray) -> bytes:
    """RGB(A) array -> PNG bytes (alpha composited onto white)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(to_rgb(img)).save(buf, format="PNG")
    return buf.getvalue()


@functools.lru_cache(maxsize=32)
def _decode(png: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(png)) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    out = to_rgb(arr)
    out.setflags(write=False)       # shared through the cache: never mutate
    return out


def png_to_rgb(png: bytes) -> np.ndarray:
    """PNG bytes -> read-only (H, W, 3) uint8, alpha composited onto white.

    Cached by content: the same dream is decoded once per process, not once per
    scorer per attempt (a 1024^2 dream costs ~10 ms to decode).
    """
    return _decode(bytes(png))


def _views(views: Union[ViewLike, Sequence[ViewLike]]) -> List[View]:
    if isinstance(views, (str, View)):
        views = [views]
    return [get_view(v) for v in views]


def _shown(structure: Structure, view: View) -> Structure:
    """``view.apply`` plus the empty check every renderer must make."""
    if not bool(structure.occupied_mask.any()):
        raise EmptyBuild(f"empty structure (view {view.name})")
    s = view.apply(structure)
    if not bool(s.occupied_mask.any()):     # cannot happen with cut >= 1, but be explicit
        raise EmptyBuild(f"view {view.name} removes every block")
    return s


def render_pngs(renderer: "RendererLike", structure: Structure,
                views: Union[ViewLike, Sequence[ViewLike]]) -> Dict[str, bytes]:
    """``{view name: PNG}``: the form ``Attempt.renders`` stores."""
    vs = _views(views)
    imgs = renderer.render(structure, vs)
    return {v.name: png_bytes(im) for v, im in zip(vs, imgs)}


# ===========================================================================
# Camera (shared convention with renderer/textured.py)
# ===========================================================================
def camera_basis(azim_deg: float, elev_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, toward_camera) unit vectors, exactly as ``textured._look_at``
    builds them, so flat and textured renders of a build share a silhouette."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    forward = -d
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    return right, up, d


# ===========================================================================
# textured
# ===========================================================================
_GL_LOCK = threading.Lock()
_TEXLIBS: Dict[str, object] = {}
_WARNED_FALLBACK = False


def texture_library(pack_dir: Union[str, Path] = DEFAULT_PACK):
    """One shared ``FaceTextureLibrary`` per pack per process.

    Why shared: the library lazily caches decoded PNGs; ``agent._render`` built a
    new one per call, re-reading the pack for every critique round.
    """
    key = str(Path(pack_dir).resolve())
    with _GL_LOCK:
        lib = _TEXLIBS.get(key)
        if lib is None:
            from blockgen.renderer.textures import load_face_textures
            lib = load_face_textures(key)
            _TEXLIBS[key] = lib
    return lib


def _release_gl(px: int) -> None:
    """Unbind the cached EGL context from this thread after a render.

    Why: an EGL context can be current on only one thread, and pyrender's EGL
    ``make_uncurrent`` is a no-op, so the context stays bound to whichever thread
    rendered first and every other thread then fails with EGL_BAD_ACCESS (seen
    with the suite's parallel arms). Releasing after each render, under the lock,
    lets any thread render next. Best effort: a non-EGL platform has nothing to
    release.
    """
    try:
        from blockgen.renderer import textured as _tx
        plat = getattr(_tx._RENDERER_CACHE.get(px), "_platform", None)
        disp = getattr(plat, "_egl_display", None)
        if disp is None:
            return
        from OpenGL.EGL import EGL_NO_CONTEXT, EGL_NO_SURFACE, eglMakeCurrent
        eglMakeCurrent(disp, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
    except Exception:  # noqa: BLE001 - nothing bound, or not EGL
        pass


@register("renderer", "textured")
class TexturedRenderer:
    """Real-texture pyrender/EGL renders (orthographic, white background).

    ``px`` overrides every view's own ``px`` (None = use the view's). The pack is
    checked at construction, so a recipe on a machine without textures fails when
    the arm is built, not as solid-colour renders hours later.
    """

    name = "textured"

    def __init__(self, px: Optional[int] = None, *, pack_dir: Optional[str] = None,
                 ortho: bool = True, max_fallback_frac: float = MAX_FALLBACK_FRAC) -> None:
        self.px = None if px is None else int(px)
        self.pack_dir = Path(pack_dir) if pack_dir else DEFAULT_PACK
        if not self.pack_dir.is_absolute():
            self.pack_dir = REPO_ROOT / self.pack_dir
        if not self.pack_dir.is_dir() or not any(self.pack_dir.glob("*.png")):
            raise RenderError(
                f"texture pack not found at {self.pack_dir}: fetch it with "
                f"`python -m blockgen.renderer.textures --fetch` (or use renderer 'flat')")
        self.ortho = bool(ortho)
        self.max_fallback_frac = float(max_fallback_frac)
        self.fallback_frac = 0.0       # of the last rendered build, for diagnostics

    def render(self, structure: Structure, views: Sequence[ViewLike]) -> List[np.ndarray]:
        vs = _views(views)
        shown = [_shown(structure, v) for v in vs]          # EmptyBuild before any GL
        try:
            from blockgen.renderer.textured import render_structure
        except Exception as exc:                            # no pyrender / no EGL
            raise RenderError(f"textured renderer unavailable: {type(exc).__name__}: {exc}") from exc
        lib = texture_library(self.pack_dir)
        out: List[np.ndarray] = []
        # One lock around every GL call: pyrender's OffscreenRenderer cache is
        # per-px and not thread-safe, and the texture library mutates on a miss.
        with _GL_LOCK:
            for v, s in zip(vs, shown):
                px = int(self.px or v.px)
                try:
                    img = render_structure(s, px=px, azim_deg=v.azim,
                                           elev_deg=v.elev, ortho=self.ortho,
                                           bg=(1.0, 1.0, 1.0), face_textures=lib)
                except Exception as exc:
                    raise RenderError(f"textured render failed for view {v.name}: "
                                      f"{type(exc).__name__}: {exc}") from exc
                finally:
                    _release_gl(px)
                out.append(to_rgb(img))
            self._check_fallback(structure, lib)
        return out

    def _check_fallback(self, structure: Structure, lib) -> None:
        global _WARNED_FALLBACK
        fb = getattr(lib, "fallback_keys", set())
        occ = structure.occupied_mask
        n = int(occ.sum())
        if not fb or n == 0:
            self.fallback_frac = 0.0
            return
        keys = np.stack([structure.block_ids[occ], structure.block_data[occ]], 1)
        hit = np.zeros(len(keys), bool)
        for bid, bdata in fb:
            hit |= (keys[:, 0] == bid) & (keys[:, 1] == bdata)
        self.fallback_frac = float(hit.mean())
        if self.fallback_frac > self.max_fallback_frac and not _WARNED_FALLBACK:
            _WARNED_FALLBACK = True
            warnings.warn(f"textured renderer: {self.fallback_frac:.1%} of voxels use a "
                          f"solid-colour fallback texture (> {self.max_fallback_frac:.0%}); "
                          f"keys {sorted(fb)[:8]}", RuntimeWarning)


# ===========================================================================
# flat
# ===========================================================================
#: (normal, texture face, brightness): the same Minecraft-style shading as textured.
_FACES: Tuple[Tuple[Tuple[int, int, int], str, float], ...] = (
    ((0, 1, 0), "top", 1.0), ((0, -1, 0), "bottom", 0.55), ((0, 0, -1), "side", 0.85),
    ((0, 0, 1), "side", 0.85), ((-1, 0, 0), "side", 0.70), ((1, 0, 0), "side", 0.70),
)


def _face_corners(normal: Tuple[int, int, int]) -> np.ndarray:
    """4 corners (voxel-relative, in order around the face) of the face along ``normal``."""
    axis = int(np.flatnonzero(normal)[0])
    off = 1.0 if normal[axis] > 0 else 0.0
    a, b = [i for i in range(3) if i != axis]
    out = np.zeros((4, 3), np.float32)
    for k, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        out[k, axis], out[k, a], out[k, b] = off, u, v
    return out


_CORNERS = {n: _face_corners(n) for n, _, _ in _FACES}


_RGB_CACHE: Dict[Tuple[int, int, str, bool], Tuple[int, int, int]] = {}


def _mean_rgb(rgba: np.ndarray) -> Tuple[int, int, int]:
    """Mean colour of a texture tile as seen on white: alpha-weighted RGB, then the
    mean coverage composited onto white (glass reads pale, as it does on screen)."""
    t = np.asarray(rgba, np.float64).reshape(-1, rgba.shape[-1])
    a = t[:, 3] / 255.0 if t.shape[1] == 4 else np.ones(len(t))
    cover = max(float(a.mean()), 0.35)
    rgb = (t[:, :3] * a[:, None]).sum(0) / max(float(a.sum()), 1e-6)
    return tuple(int(round(c * cover + 255 * (1 - cover))) for c in rgb)  # type: ignore[return-value]


def block_rgb(block_id: int, block_data: int, face: str = "side",
              textures: bool = True) -> Tuple[int, int, int]:
    """Display colour of one block face ('top' | 'side' | 'bottom').

    With ``textures`` (and the pack present) it is the mean colour of the real
    texture, so a flat render has the palette of a textured one (grass tops are
    green, sides brown) and colour scorers behave alike offline. Without the pack
    it falls back to ``utils.data`` colours, which is also what the texture
    library does for unmapped blocks.
    """
    use_tex = bool(textures) and DEFAULT_PACK.is_dir()
    key = (int(block_id), int(block_data), face, use_tex)
    hit = _RGB_CACHE.get(key)
    if hit is not None:
        return hit
    if use_tex:
        lib = texture_library(DEFAULT_PACK)
        with _GL_LOCK:                              # the library mutates on a miss
            faces = lib[(int(block_id), int(block_data))]
        rgb = _mean_rgb(faces.get(face, faces["side"]))
    else:
        from blockgen.utils.data import _color_for, _resource_location_for
        r, g, b, a = _color_for(_resource_location_for(int(block_id), int(block_data)),
                                int(block_data))
        rgb = _mean_rgb(np.array([[[255 * r, 255 * g, 255 * b, 255 * a]]]))
    _RGB_CACHE[key] = rgb
    return rgb


@register("renderer", "flat")
class FlatRenderer:
    """No-GL renderer: exposed voxel faces projected orthographically and painted
    far-to-near (painter's algorithm) with each face's mean texture colour and face shading.

    Same camera, framing (bbox fit with a 5% margin) and white background as
    ``textured``, so silhouette and colour scorers behave alike offline; it lacks
    only the texture detail. A few ms per 32^3 view.
    """

    name = "flat"

    def __init__(self, px: Optional[int] = None, *, margin: float = 1.05,
                 textures: bool = True) -> None:
        self.px = None if px is None else int(px)
        self.margin = float(margin)
        self.textures = bool(textures)

    def render(self, structure: Structure, views: Sequence[ViewLike]) -> List[np.ndarray]:
        vs = _views(views)
        shown = [_shown(structure, v) for v in vs]
        return [self._one(s.crop_to_non_air(), v, int(self.px or v.px)) for v, s in zip(vs, shown)]

    def _one(self, s: Structure, view: View, px: int) -> np.ndarray:
        from PIL import Image, ImageDraw

        occ = s.occupied_mask
        dims = np.asarray(occ.shape, np.float64)
        center = dims / 2.0
        right, up, toward = camera_basis(view.azim, view.elev)
        corners = np.array([(x, y, z) for x in (0, dims[0]) for y in (0, dims[1])
                            for z in (0, dims[2])]) - center
        mag = max(float(np.abs(corners @ right).max()), float(np.abs(corners @ up).max()),
                  1e-3) * self.margin
        padded = np.pad(occ, 1)
        depths, quads, fills = [], [], []
        for normal, face, shade in _FACES:
            if float(np.dot(normal, toward)) <= 1e-6:       # back-facing: never visible
                continue
            dx, dy, dz = normal
            nb = padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dy:padded.shape[1] - 1 + dy,
                        1 + dz:padded.shape[2] - 1 + dz]
            coords = np.argwhere(occ & ~nb)
            if len(coords) == 0:
                continue
            pts = coords[:, None, :].astype(np.float64) + _CORNERS[normal][None] - center
            u = pts @ right
            v = pts @ up
            quads.append(np.stack([(u / mag + 1) * 0.5 * px, (1 - (v / mag + 1) * 0.5) * px], -1))
            depths.append(pts.mean(axis=1) @ toward)
            ids = s.block_ids[coords[:, 0], coords[:, 1], coords[:, 2]]
            data = s.block_data[coords[:, 0], coords[:, 1], coords[:, 2]]
            pairs, inv = np.unique(np.stack([ids, data], 1), axis=0, return_inverse=True)
            lut = np.array([block_rgb(i, d, face, self.textures) for i, d in pairs.tolist()],
                           np.float64)
            fills.append((lut[np.ravel(inv)] * shade).astype(np.int64))
        img = Image.new("RGB", (px, px), WHITE)
        if not quads:
            return np.asarray(img, np.uint8).copy()
        q = np.concatenate(quads)
        dep = np.concatenate(depths)
        col = np.concatenate(fills)
        draw = ImageDraw.Draw(img)
        for k in np.argsort(dep, kind="stable"):            # far first: near faces overwrite
            draw.polygon([tuple(p) for p in q[k].tolist()], fill=tuple(col[k].tolist()))
        return np.asarray(img, np.uint8).copy()


# ===========================================================================
# null
# ===========================================================================
def null_image(view: ViewLike, px: Optional[int] = None) -> np.ndarray:
    """The ``null`` renderer's tile for ``view``: a solid, never-white colour keyed
    by the view name only. Exposed so parity tests can feed BuildAgent the same
    images the native method sees."""
    v = get_view(view)
    h = hashlib.sha256(f"null|{v.name}".encode()).digest()
    rgb = (h[0] // 2 + 32, h[1] // 2 + 32, h[2] // 2 + 32)
    n = int(px or v.px)
    return np.full((n, n, 3), rgb, np.uint8)


@register("renderer", "null")
class NullRenderer:
    """Deterministic solid tile per view (no geometry). Estimate mode and parity
    tests: the request shape is real, the pixels are not."""

    name = "null"

    def __init__(self, px: Optional[int] = 64) -> None:
        self.px = None if px is None else int(px)

    def render(self, structure: Structure, views: Sequence[ViewLike]) -> List[np.ndarray]:
        vs = _views(views)
        for v in vs:
            _shown(structure, v)
        return [null_image(v, self.px) for v in vs]


RendererLike = Union[TexturedRenderer, FlatRenderer, NullRenderer]

__all__ = ["DEFAULT_PACK", "FlatRenderer", "MAX_FALLBACK_FRAC", "NullRenderer", "REPO_ROOT",
           "TexturedRenderer", "block_rgb", "camera_basis", "null_image", "png_bytes",
           "png_to_rgb", "render_pngs", "texture_library", "to_rgb"]
