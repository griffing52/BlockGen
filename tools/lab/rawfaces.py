"""The faces a facade model actually wrote, beside the build they lift to.

The lab's `facades` representation projects faces *from the finished build*. For a
consistent face set (Cross-Face decoding, real houses) that is exactly what the model
wrote; for unconstrained decoding it is not, because the lift settles every
disagreement by overwriting and the projection of the result hides the conflict.
Facade sampling runs save the raw face sets next to their build cache as
``faces.npz`` (keys ``<i>_<face>_<depth|ids|data>``, one entry per draw, empty draws
included). This module pairs a build with its raw face set and draws the six faces
with every cell that changes under lift-then-reproject tinted red.

Pairing. A sampling run drops draws whose lift is empty before writing the build
cache, so build ``k`` is the ``k``-th face set with at least one filled cell. The map
is used only when that count equals the number of builds; otherwise no picture is
offered, rather than a picture of the wrong build.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FACES = ("-z", "+x", "+z", "-x", "+y", "-y")
FACES_FILE = "faces.npz"
RED = np.array([220, 30, 30], np.float32)
DOC = ("the six faces as the model wrote them (faces.npz), red where a cell changes when the "
       "faces are lifted to 3D and projected back; red-free means a consistent face set")

_lock = threading.Lock()
_maps: Dict[str, Optional[List[int]]] = {}


def faces_path(build_npz: str | Path) -> Optional[Path]:
    """The raw face file saved beside a build cache, if there is one."""
    p = Path(build_npz).with_name(FACES_FILE)
    return p if p.is_file() else None


def _filled_sets(z) -> List[int]:
    ids = sorted({int(k.split("_", 1)[0]) for k in z.files})
    out = []
    for i in ids:
        if any((z[f"{i}_{f}_depth"] >= 0).any() for f in FACES if f"{i}_{f}_depth" in z.files):
            out.append(i)
    return out


def index_map(build_npz: str | Path, n_builds: int) -> Optional[List[int]]:
    """Build index -> face-set index, or None when the two files cannot be paired."""
    fp = faces_path(build_npz)
    if fp is None:
        return None
    key = f"{fp}:{fp.stat().st_mtime_ns}:{n_builds}"
    with _lock:
        if key in _maps:
            return _maps[key]
    try:
        with np.load(fp) as z:
            filled = _filled_sets(z)
        m = filled if len(filled) == n_builds else None
    except Exception:
        m = None
    with _lock:
        _maps[key] = m
    return m


def load_set(build_npz: str | Path, face_index: int):
    """(shape, {face: Facade}) of one raw face set."""
    from blockgen.corpus import facade as fc

    with np.load(faces_path(build_npz)) as z:
        X, Y = z[f"{face_index}_-z_depth"].shape
        Z = z[f"{face_index}_+x_depth"].shape[0]
        shape = (X, Y, Z)
        faces = {f: fc.Facade(f, z[f"{face_index}_{f}_depth"], z[f"{face_index}_{f}_ids"],
                              z[f"{face_index}_{f}_data"], shape["xyz".index(f[1])])
                 for f in FACES if f"{face_index}_{f}_depth" in z.files}
    return shape, faces


def disagreement(faces, shape) -> Tuple[Dict[str, np.ndarray], float]:
    """Per face, the (u, v) cells that change under project(lift(.)); and their overall share."""
    from blockgen.corpus import facade as fc

    s = fc.lift(list(faces.values()), shape)
    bad = {}
    for f, F in faces.items():
        B = fc.project(s, f)
        same = (F.mask == B.mask) & ((~F.mask) | ((F.depth == B.depth) & (F.ids == B.ids)))
        bad[f] = ~same
    share = float(np.mean(np.concatenate([b.ravel() for b in bad.values()]))) if bad else 0.0
    return bad, share


def sheet(faces, bad, px: int) -> np.ndarray:
    """The `facade.sheet` layout (sides in walking order, then roof and underside), red on `bad`."""
    from blockgen.corpus import facade as fc

    rows = [[f for f in faces if faces[f].axis != 1], [f for f in faces if faces[f].axis == 1]]
    rows = [r for r in rows if r]
    base = {}
    for f, F in faces.items():
        p = fc.face_picture(F).astype(np.float32)
        m = bad[f].T
        if F.axis != 1:
            m = m[::-1]
        if fc._FLIP_U[F.face]:
            m = m[:, ::-1]
        p[m] = 0.35 * p[m] + 0.65 * RED
        base[f] = p.astype(np.uint8)
    width_cells = max(sum(base[f].shape[1] for f in r) for r in rows)
    gap = 2
    scale = max(1, (px - gap * 6) // max(width_cells, 1))
    pics = {f: np.repeat(np.repeat(p, scale, 0), scale, 1) for f, p in base.items()}
    width = max(sum(pics[f].shape[1] for f in r) + gap * (len(r) + 1) for r in rows)
    height = sum(max(pics[f].shape[0] for f in r) for r in rows) + gap * (len(rows) + 1)
    out = np.full((height, width, 3), 255, np.uint8)
    y = gap
    for r in rows:
        x, h = gap, max(pics[f].shape[0] for f in r)
        for f in r:
            p = pics[f]
            out[y + h - p.shape[0]:y + h, x:x + p.shape[1]] = p
            x += p.shape[1] + gap
        y += h + gap
    return out


def picture(build_npz: str | Path, n_builds: int, index: int, px: int = 640) -> Optional[np.ndarray]:
    """RGB sheet of build `index`'s raw faces, or None when the build cannot be paired."""
    m = index_map(build_npz, n_builds)
    if m is None or not 0 <= index < len(m):
        return None
    shape, faces = load_set(build_npz, m[index])
    bad, _ = disagreement(faces, shape)
    return sheet(faces, bad, px)
