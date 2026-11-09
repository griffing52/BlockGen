"""Cross-face consistency for generated facades.

Six face images of one box describe one visible shell only if they agree: a voxel two
faces both see must carry one block, and a face's ray cannot report a first hit
behind a voxel another face has already filled. Real faces agree by construction
(they are projections of one structure); generated faces do not, and `lift`
resolves their quarrels by overwriting, which is where the lifted shells' extra
components come from (results T68).

`ShellState` remembers what the faces drawn so far committed, voxel by voxel: the
first hit of every cell as a *known* voxel, the cells in front of it as *empty*. For
a cell of a later face it returns which depths are still possible and which block,
if any, a depth is forced to carry. The sampler masks the model's logits with that
answer, so a generated face set is consistent by construction and the model keeps
every choice the earlier faces left open: it may still put a block in front of a
known voxel where no earlier ray passed, and it draws freely wherever nothing is
known.

`reprojection` measures the same thing after the fact on any face set: lift, project
back, and count the cells that come back different.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from . import facade as fc

FREE = -1


def ray_index(face: str, u: int, v: int, shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structure indices of the voxels a face cell looks through, in depth order (depth 0 first)."""
    axis, sign = fc._parse(face)
    ua, va = fc._uv_axes(axis)
    extent = int(shape[axis])
    d = np.arange(extent)
    along = d if sign > 0 else extent - 1 - d
    idx = [None, None, None]
    idx[axis] = along
    idx[ua] = np.full(extent, u)
    idx[va] = np.full(extent, v)
    return tuple(idx)  # type: ignore[return-value]


class ShellState:
    """What the faces committed so far say about every voxel of the box.

    `known[x, y, z]` is the key of the block a face put there (a token id, or any
    integer the caller chooses), `FREE` where no face has. `empty[x, y, z]` is True
    where some face's ray passed through on the way to its first hit, or where a
    see-through cell's whole column lies.
    """

    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(int(s) for s in shape)
        self.known = np.full(self.shape, FREE, np.int64)
        self.empty = np.zeros(self.shape, bool)

    def commit_cell(self, face: str, u: int, v: int, depth: int, key: int = 0) -> None:
        """Record one cell: see-through (`depth < 0`) empties the column, otherwise the
        voxels before `depth` are empty and the one at `depth` holds `key`."""
        idx = ray_index(face, u, v, self.shape)
        extent = len(idx[0])
        if depth < 0 or depth >= extent:
            self.empty[idx] = True
            return
        if depth:
            self.empty[tuple(i[:depth] for i in idx)] = True
        self.known[tuple(i[depth] for i in idx)] = int(key)

    def commit_face(self, f: fc.Facade, keys: np.ndarray) -> None:
        """Every cell of `f`; `keys` holds the block key per cell (same shape as `f.depth`)."""
        U, V = f.depth.shape
        for v in range(V):
            for u in range(U):
                self.commit_cell(f.face, u, v, int(f.depth[u, v]), int(keys[u, v]))

    def allowed(self, face: str, u: int, v: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """`(allowed, forced, allow_empty)` for one cell of a face not yet drawn.

        `allowed[d]` says the cell may report its first hit at depth `d`: the voxel is
        not known empty, and no known voxel lies in front of it. `forced[d]` is the key
        the block must carry at that depth (`FREE` when the voxel is unclaimed). The cell
        may be see-through only if no known voxel lies on its ray.
        """
        idx = ray_index(face, u, v, self.shape)
        known = self.known[idx]
        empty = self.empty[idx]
        allowed = ~empty
        has_known = bool((known != FREE).any())
        if has_known:
            first = int(np.argmax(known != FREE))
            allowed[first + 1:] = False
            allowed[first] = True
        return allowed, np.where(allowed, known, FREE), not has_known

    def determined(self, face: str, u: int, v: int) -> bool:
        """True when the cell has exactly one legal answer (the first open voxel is known)."""
        allowed, forced, allow_empty = self.allowed(face, u, v)
        return (not allow_empty) and int(allowed.sum()) == 1 and forced[np.argmax(allowed)] != FREE


def face_keys(f: fc.Facade, lut: np.ndarray) -> np.ndarray:
    """Block keys of a face's cells through a `(256, 16)` id/data lookup (the tokenizer's `block_lut`)."""
    return lut[np.clip(f.ids, 0, 255).astype(np.int64), np.clip(f.data, 0, 15).astype(np.int64)]


def consistent(faces: Mapping[str, fc.Facade] | Sequence[fc.Facade], shape: Sequence[int],
               order: Sequence[str] | None = None) -> bool:
    """Whether committing the faces one after another in `order` never meets a contradiction."""
    fd = {f.face: f for f in (faces.values() if isinstance(faces, Mapping) else faces)}
    order = list(order or fd)
    st = ShellState(shape)
    for name in order:
        f = fd[name]
        U, V = f.depth.shape
        for v in range(V):
            for u in range(U):
                allowed, forced, allow_empty = st.allowed(name, u, v)
                d = int(f.depth[u, v])
                key = int(f.ids[u, v]) * 16 + int(f.data[u, v])
                if d < 0:
                    if not allow_empty:
                        return False
                elif d >= len(allowed) or not allowed[d] or (forced[d] != FREE and forced[d] != key):
                    return False
                st.commit_cell(name, u, v, d, key)
    return True


def reprojection(faces: Mapping[str, fc.Facade] | Sequence[fc.Facade], shape: Sequence[int]) -> Dict[str, float]:
    """Lift the faces, project the shell back, and count what came back different.

    `cell_disagree`: share of all face cells whose reprojected (mask, depth, block)
    differs from what the face said; `hidden`: share of the faces' filled cells whose
    block is not the first hit of the lifted shell (another face's block sits in
    front of it, or the two faces disagree on that voxel); `filled_in`: share of
    see-through cells that the lifted shell fills. Means over faces; a real shell
    gives zeros.
    """
    fl = list(faces.values()) if isinstance(faces, Mapping) else list(faces)
    s = fc.lift(fl, tuple(int(x) for x in shape))
    dis, hid, fil = [], [], []
    for f in fl:
        back = fc.project(s, f.face)
        if back.depth.shape != f.depth.shape:
            continue
        m, mb = f.mask, back.mask
        same = (m == mb) & ((~m) | ((f.depth == back.depth) & (f.ids == back.ids) & (f.data == back.data)))
        dis.append(1.0 - same.mean())
        hid.append(float((m & ~same).sum() / max(m.sum(), 1)))
        fil.append(float((~m & mb).sum() / max((~m).sum(), 1)))
    if not dis:
        return dict(cell_disagree=float("nan"), hidden=float("nan"), filled_in=float("nan"))
    return dict(cell_disagree=float(np.mean(dis)), hidden=float(np.mean(hid)), filled_in=float(np.mean(fil)))
