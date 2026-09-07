"""Thumbnails: one render per build, ever.

Three constraints shape everything here, and all three are scars.

**One EGL context, one thread -- not one lock.** `blockgen.renderer.textured`
builds a single offscreen pyrender context and reuses it, and an EGL context is
*thread-affine*: `eglMakeCurrent` binds it to the calling thread and fails with
`EGL_BAD_ACCESS` while it is still current on another.

The first version of this module used a mutex, which is the obvious answer and
the wrong one. A lock serialises the *calls* but leaves each on a different
thread, and `ThreadingHTTPServer` hands every request to a fresh one -- so a
grid came back with roughly half its tiles as placeholders, intermittently, with
no pattern in which. Renders now go through `_on_render_thread`, a queue drained
by a single long-lived worker, so the context is made current once and never
migrates. The server stays threaded, which is the point: a slow render still
does not block the JSON endpoints.

**A blank image is not a render.** T25c: a PyOpenGL/Python version mismatch made
every `render_structure` call raise, every frame came back uniform white, and the
resulting features scored a near-perfect KID. Uniform output is the signature of
a broken renderer, so `looks_blank` exists and the test suite asserts against it
on a real build. A placeholder returned from here is *deliberately textured* for
the same reason -- so a broken renderer cannot masquerade as a plain build.

**The cache is content-addressed.** The key is the SHA-1 of the build's voxels,
not its `dataset:index`, following `features.structures_sha`. Two consequences,
both wanted: re-generating an arm to the same path invalidates only the builds
that actually changed, and the same build reached through `corpus:houses_32` and
through `split:houses_32:test` renders once and is shared. At 128 px a render is
80 ms, so the cache is what makes a 60-tile grid feel instant on the second look.

Views match `perceptual.DEFAULT_VIEWS` -- the same four orbit angles the feature
pipeline embeds -- so what a human sees in the lab is what the metrics saw.
"""

from __future__ import annotations

import hashlib
import io
import queue
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from blockgen.eval.perceptual import DEFAULT_VIEWS
from blockgen.utils.data import Structure
from tools.lab import THUMB_CACHE
from tools.lab import catalog

#: (azimuth, elevation) degrees. Index 0 is the thumbnail angle.
VIEWS: Tuple[Tuple[float, float], ...] = DEFAULT_VIEWS
ORTHO = True
JPEG_QUALITY = 82
#: White, matching `features.render_views`' composite, so lab pixels and metric
#: pixels are the same pixels.
BG = (255.0, 255.0, 255.0)

#: Every render runs on ONE dedicated thread, not merely one at a time.
#:
#: A mutex is not enough and the difference is not subtle. An EGL context is
#: *thread-affine*: `eglMakeCurrent` binds it to the calling thread and fails
#: with `EGL_BAD_ACCESS` if it is still current on another. `ThreadingHTTPServer`
#: hands every request to a fresh thread, so a lock serialises the calls while
#: leaving each one on a different thread -- which is exactly the condition the
#: error describes. Observed as roughly half a grid of tiles coming back as
#: placeholders, intermittently, with no pattern.
#:
#: So renders are submitted to a queue and executed by a single long-lived
#: worker. The context is made current once, on that thread, and stays there.
_RENDER_Q: "queue.Queue[tuple]" = queue.Queue()
_RENDER_THREAD: Optional[threading.Thread] = None
_RENDER_START = threading.Lock()


def _render_loop() -> None:
    while True:
        fn, box, done = _RENDER_Q.get()
        try:
            box.append(("ok", fn()))
        except BaseException as exc:                 # noqa: BLE001
            box.append(("err", exc))
        finally:
            done.set()
            _RENDER_Q.task_done()


def _on_render_thread(fn):
    """Run `fn` on the render thread and return its result (or re-raise).

    Started lazily so importing this module costs nothing, and daemonised so a
    stuck render can never hold the process open.
    """
    global _RENDER_THREAD
    if _RENDER_THREAD is None:
        with _RENDER_START:
            if _RENDER_THREAD is None:
                _RENDER_THREAD = threading.Thread(
                    target=_render_loop, name="lab-render", daemon=True)
                _RENDER_THREAD.start()
    # Already on the worker (a nested render): run inline or we would deadlock
    # waiting for a queue only this thread drains.
    if threading.current_thread() is _RENDER_THREAD:
        return fn()
    box: list = []
    done = threading.Event()
    _RENDER_Q.put((fn, box, done))
    done.wait()
    kind, value = box[0]
    if kind == "err":
        raise value
    return value
_TEXTURES = None
_TEXTURE_LOCK = threading.Lock()


def _textures():
    """The shared texture library. Loaded once, lazily: importing this module
    must not touch the GPU or the texture pack, because `catalog`-only callers
    (the datasets endpoint, the tests) have no business paying for it."""
    global _TEXTURES
    if _TEXTURES is None:
        with _TEXTURE_LOCK:
            if _TEXTURES is None:
                from blockgen.renderer.textures import load_face_textures
                _TEXTURES = load_face_textures()
    return _TEXTURES


# --- pixels ----------------------------------------------------------------
def _composite(img: np.ndarray) -> np.ndarray:
    """RGBA -> RGB over white, byte-identical to `bench.features.render_views`."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + np.array(BG, np.float32) * (1 - a)).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


#: Variance below which a uint8 RGB frame carries no build. Measured at 192 px
#: on this corpus rather than guessed:
#:
#:     real renders (5 builds, 3 kinds)   5168 - 10391
#:     a 6x6 sliver on white                     45
#:     one stray antialiased pixel              1.8
#:     all-white                                0.0
#:
#: 100 sits ~50x below the real minimum and ~2x above the worst near-blank case,
#: so it separates cleanly in both directions. An exact `== 0.0` test -- the
#: first version of this -- passes every row but the last, and the failure it
#: guards against is the most expensive one this project has had: T25c, a dead
#: renderer returning blank frames that scored *better than real* and would have
#: silently rebuilt the feature cache from blanks.
#:
#: Note the camera fits to the bounding box, so a small build fills the frame
#: just as a large one does; a low variance means no build, never a little one.
BLANK_VARIANCE = 100.0


def looks_blank(img: np.ndarray, threshold: float = BLANK_VARIANCE) -> bool:
    """True if the frame carries no build. The T25c canary; see `results.md`.

    Takes pixels, not encoded bytes -- JPEG bytes have high variance whatever
    they depict, so passing them here would make the check silently useless.
    """
    a = np.asarray(img)
    if a.dtype.kind not in "uif":
        raise TypeError(
            f"looks_blank expects an image array, got dtype {a.dtype!r}. "
            "Decode JPEG bytes to pixels first -- checking the variance of "
            "compressed bytes always passes and would defeat the canary.")
    return float(a.var()) < threshold


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8), "RGB").save(
        buf, format="JPEG", quality=quality)
    return buf.getvalue()


def placeholder(px: int = 256) -> bytes:
    """A visibly-not-a-build tile for a render that failed.

    Textured rather than flat on purpose: a uniform grey tile is precisely what a
    silently-broken renderer produces, and the whole point of T25c is that those
    two states must never look alike.
    """
    y, x = np.mgrid[0:px, 0:px]
    stripe = (((x + y) // max(px // 16, 1)) % 2).astype(np.uint8)
    img = np.where(stripe[..., None] == 1, np.uint8(232), np.uint8(198))
    return encode_jpeg(np.repeat(img, 3, axis=2))


# --- cache -----------------------------------------------------------------
def _content_sha(s: Structure) -> str:
    """Identity of the voxels, matching `features.structures_sha`'s recipe.

    Includes `block_data` because the renderer uses it for orientation-bearing
    blocks (stairs, logs): two builds with identical ids but different data draw
    differently, and collapsing them would serve the wrong thumbnail.
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(s.block_ids.astype(np.int32)).tobytes())
    h.update(np.ascontiguousarray(s.block_data.astype(np.int32)).tobytes())
    return h.hexdigest()


def cache_key(dataset_id: str, index: int, px: int, view: int = 0) -> str:
    """Content-addressed key. Falls back to the id when the build cannot load.

    The fallback keeps a missing dataset from turning into an exception in the
    middle of a grid request; it is still stable, just not shared across
    datasets, which only matters for builds that will render as placeholders
    anyway.
    """
    try:
        s = catalog.load_builds(dataset_id)[index]
        digest = _content_sha(s)[:20]
    except Exception:
        digest = "id" + hashlib.sha1(
            f"{dataset_id}:{index}".encode("utf-8")).hexdigest()[:18]
    return f"{digest}_p{int(px)}_v{int(view)}"


def cache_path(key: str) -> Path:
    # Two-character shard: a whole-corpus browse is 2661 builds x 4 views, and
    # one flat directory of 10k files is slow to stat on every request.
    return Path(THUMB_CACHE) / key[:2] / f"{key}.jpg"


# --- rendering -------------------------------------------------------------
def _render_one(s: Structure, px: int, view: int) -> bytes:
    azim, elev = VIEWS[view % len(VIEWS)]
    from blockgen.renderer.textured import render_structure

    tex = _textures()
    img = _on_render_thread(lambda: render_structure(
        s, px=px, azim_deg=azim, elev_deg=elev, ortho=ORTHO, face_textures=tex))
    return encode_jpeg(_composite(img))


def _cached_view(dataset_id: str, index: int, px: int, view: int) -> bytes:
    path = cache_path(cache_key(dataset_id, index, px, view))
    if path.is_file():
        try:
            return path.read_bytes()
        except OSError:
            pass                                   # torn write; fall through

    try:
        s = catalog.load_builds(dataset_id)[index]
        data = _render_one(s, px, view)
    except Exception as exc:
        # One pathological build must not take down a grid of sixty. The
        # placeholder is NOT written to the cache: a transient failure (a busy
        # GPU, a dataset mid-rewrite) should not be baked in forever.
        print(f"[lab.renders] {dataset_id}:{index} view {view} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return placeholder(px)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jpg.part")
        tmp.write_bytes(data)
        tmp.replace(path)                          # atomic: readers see whole files
    except OSError as exc:
        print(f"[lab.renders] cache write failed for {path}: {exc}", flush=True)
    return data


def thumb(dataset_id: str, index: int, px: int = 256) -> bytes:
    """JPEG bytes for the grid tile. Never raises."""
    return _cached_view(dataset_id, index, px, 0)


def views(dataset_id: str, index: int, px: int = 256) -> List[bytes]:
    """All four orbit views for the detail page. Never raises."""
    return [_cached_view(dataset_id, index, px, v) for v in range(len(VIEWS))]


def render_array(s: Structure, px: int = 256, view: int = 0) -> Optional[np.ndarray]:
    """RGB array for one structure, bypassing the cache. For tests and probes."""
    azim, elev = VIEWS[view % len(VIEWS)]
    from blockgen.renderer.textured import render_structure

    tex = _textures()
    img = _on_render_thread(lambda: render_structure(
        s, px=px, azim_deg=azim, elev_deg=elev, ortho=ORTHO, face_textures=tex))
    return _composite(img)
