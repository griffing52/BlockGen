"""Facades: a build as the block images its six faces present.

A *face image* looks at a build along one axis and records, for every column of
the view, the first block it hits and how far in it sits. Four side faces are the
building's facades; the top face is its roof plan. The representation is 2.5D:
it holds exactly what a camera at infinity sees along that axis, no interior and
nothing hidden behind an eave, and it lifts back to voxels exactly for that
visible shell. Its purpose is to make generation a 2D problem first: a facade is
a grid of (depth, block) tokens, which is the format a sequence model lays out
reliably, and a set of facades `lift`s into the 32^3 canvas the benchmark scores.

Conventions. Structures are XYZ with y up. A face is named by the axis it looks
along and the end it looks from: `-x` stands at x = 0 looking toward larger x,
`+x` stands at x = max looking back, and so on. Every face image is indexed
`(u, v)`: `v` is height `y` for the side faces and `z` for the top face, `u` is
the remaining axis, both in increasing coordinate order of the *structure*, so
two faces that share an edge agree on it and a D4 rotation of the structure is a
relabelling of faces plus a flip of `u`, nothing more. `depth` counts voxels from
the face plane to the first occupied cell, `-1` where the column is see-through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

#: The four facades, in the order a walk around the building visits them.
SIDES: Tuple[str, ...] = ("-z", "+x", "+z", "-x")
#: Facades plus the roof plan; what a generator of 2.5D shells should emit.
SIDES_TOP: Tuple[str, ...] = SIDES + ("+y",)
#: Every face, for a complete visible shell. `-y` is the underside and is almost
#: always a flat ground plate on this corpus.
ALL_FACES: Tuple[str, ...] = SIDES_TOP + ("-y",)

_AXIS = {"x": 0, "y": 1, "z": 2}
EMPTY = -1


@dataclass(frozen=True)
class Facade:
    """One face image. `depth`, `ids` and `data` share the `(u, v)` shape."""
    face: str
    depth: np.ndarray    # int16, EMPTY where the column holds no block
    ids: np.ndarray      # int16 block id of the first block, 0 where EMPTY
    data: np.ndarray     # int16 block data of that block, 0 where EMPTY
    extent: int          # length of the structure along the viewing axis

    @property
    def axis(self) -> int:
        return _AXIS[self.face[1]]

    @property
    def sign(self) -> int:
        return 1 if self.face[0] == "-" else -1

    @property
    def mask(self) -> np.ndarray:
        return self.depth >= 0

    @property
    def relief(self) -> np.ndarray:
        """`depth` as float with NaN where see-through, for image-style analysis."""
        return np.where(self.mask, self.depth, np.nan).astype(np.float32)


def _parse(face: str) -> Tuple[int, int]:
    if len(face) != 2 or face[0] not in "+-" or face[1] not in _AXIS:
        raise ValueError(f"face must be one of {ALL_FACES}, got {face!r}")
    return _AXIS[face[1]], (1 if face[0] == "-" else -1)


def _uv_axes(axis: int) -> Tuple[int, int]:
    """`(u, v)` structure axes of a face looking along `axis`: v is y for the
    sides and z for the top and bottom; u is whatever remains."""
    if axis == 1:
        return 0, 2
    return (2, 1) if axis == 0 else (0, 1)


def face_depth(occ: np.ndarray, face: str) -> np.ndarray:
    """Depth to the first occupied voxel along `face`; EMPTY where none."""
    axis, sign = _parse(face)
    u, v = _uv_axes(axis)
    a = np.moveaxis(np.asarray(occ, dtype=bool), (axis, u, v), (0, 1, 2))
    if sign < 0:
        a = a[::-1]
    hit = a.any(0)
    return np.where(hit, a.argmax(0), EMPTY).astype(np.int16)


def project(s: Structure, face: str) -> Facade:
    """The face image of `s` (not cropped here: callers crop once, upstream)."""
    axis, sign = _parse(face)
    depth = face_depth(s.occupied_mask, face)
    idx = _voxel_index(depth, axis, sign, s.shape)
    mask = depth >= 0
    ids = np.zeros(depth.shape, np.int16)
    data = np.zeros(depth.shape, np.int16)
    ids[mask] = s.block_ids[idx][mask]
    data[mask] = s.block_data[idx][mask]
    return Facade(face=face, depth=depth, ids=ids, data=data, extent=s.shape[axis])


def project_all(s: Structure, faces: Iterable[str] = SIDES_TOP) -> Dict[str, Facade]:
    return {f: project(s, f) for f in faces}


def _voxel_index(depth: np.ndarray, axis: int, sign: int,
                 shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structure indices `(x, y, z)` of every column's first block. Columns with
    no block index voxel 0 along the axis; callers mask them out."""
    u, v = _uv_axes(axis)
    d = np.clip(depth, 0, None).astype(np.int64)
    along = d if sign > 0 else (shape[axis] - 1 - d)
    uu, vv = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]),
                         indexing="ij")
    idx = [None, None, None]
    idx[axis], idx[u], idx[v] = along, uu, vv
    return tuple(idx)  # type: ignore[return-value]


def lift(facades: Mapping[str, Facade] | Iterable[Facade],
         shape: Tuple[int, int, int] | None = None) -> Structure:
    """Place every face's visible blocks into one canvas: the visible shell.

    Faces are written in the order given, later faces overwriting earlier ones
    where two faces name the same voxel. For faces projected from one structure
    the two readings agree, so order is immaterial; for generated faces that
    disagree, order is the tie-break and the caller chooses it. `shape` defaults
    to the smallest canvas consistent with the faces' own extents.
    """
    faces = list(facades.values()) if isinstance(facades, Mapping) else list(facades)
    if not faces:
        raise ValueError("lift needs at least one face")
    if shape is None:
        shape = _infer_shape(faces)
    ids = np.zeros(shape, np.int32)
    data = np.zeros(shape, np.int32)
    for f in faces:
        idx = _voxel_index(f.depth, f.axis, f.sign, shape)
        m = f.mask & (f.depth < shape[f.axis])
        ids[tuple(i[m] for i in idx)] = f.ids[m]
        data[tuple(i[m] for i in idx)] = f.data[m]
    return Structure(block_ids=ids, block_data=data)


def _infer_shape(faces: Sequence[Facade]) -> Tuple[int, int, int]:
    shape = [0, 0, 0]
    for f in faces:
        u, v = _uv_axes(f.axis)
        for ax, n in ((f.axis, f.extent), (u, f.depth.shape[0]), (v, f.depth.shape[1])):
            if shape[ax] and shape[ax] != n:
                raise ValueError(f"face {f.face} disagrees on axis {ax}: "
                                 f"{shape[ax]} vs {n}")
            shape[ax] = n
    if not all(shape):
        raise ValueError("faces do not determine every axis; pass `shape`")
    return tuple(shape)  # type: ignore[return-value]


def visible_mask(occ: np.ndarray, faces: Iterable[str] = SIDES_TOP) -> np.ndarray:
    """Voxels some face in `faces` sees: exactly the cells `lift(project_all)` fills."""
    occ = np.asarray(occ, dtype=bool)
    out = np.zeros(occ.shape, bool)
    for face in faces:
        axis, sign = _parse(face)
        depth = face_depth(occ, face)
        idx = _voxel_index(depth, axis, sign, occ.shape)
        m = depth >= 0
        out[tuple(i[m] for i in idx)] = True
    return out


def shell(s: Structure, faces: Iterable[str] = SIDES_TOP) -> Structure:
    """`s` reduced to what its `faces` show: the 2.5D oracle of a 3D build.

    Keeps the crop's bounding box so the renderer's camera fit and every
    silhouette the side faces define are unchanged. With `SIDES` alone a flat
    roof disappears, because no side ray reaches its upper surface; with
    `SIDES_TOP` the roof plan restores it. Either way nothing inside survives:
    this is the ceiling of any generator that emits facades and stops.
    """
    c = s.crop_to_non_air()
    lifted = lift(project_all(c, faces), c.shape)
    return Structure(block_ids=lifted.block_ids, block_data=lifted.block_data,
                     source_path=c.source_path, air_block_id=c.air_block_id,
                     metadata=dict(c.metadata))


# --- levels ------------------------------------------------------------------
#: Pooling rules for `pool`: how an f x f block of cells becomes one coarse cell.
POOL_RULES: Tuple[str, ...] = ("any", "majority", "center")


def _mode(values: np.ndarray) -> int:
    """Most common value; ties go to the smallest, so the result is deterministic."""
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts == counts.max()].min())


def pool(f: Facade, factor: int, rule: str = "any") -> Facade:
    """`f` at a coarser level: ceil(U/f) x ceil(V/f) cells, each summarising its
    f x f children under `rule`. Depths stay in voxels and `extent` is unchanged,
    so the coarse face still describes the same box.

    ``any``: filled if any child is; depth the nearest child (what a camera sees
    first); block the most common child block. Dilating, one-cell features
    survive. ``majority``: filled if at least half the in-bounds children are;
    depth and block the most common child values. Thin features vanish, as under
    T17's decimation. ``center``: the centre child of the in-bounds part of the
    block, strided decimation. Blocks at the far edge of a face that is not a
    multiple of `factor` count only the children inside the face.
    """
    if rule not in POOL_RULES:
        raise ValueError(f"rule must be one of {POOL_RULES}, got {rule!r}")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return f
    U, V = f.depth.shape
    Up, Vp = -(-U // factor) * factor, -(-V // factor) * factor
    nu, nv = Up // factor, Vp // factor
    depth = np.full((Up, Vp), EMPTY, np.int16)
    key = np.zeros((Up, Vp), np.int32)
    valid = np.zeros((Up, Vp), bool)
    depth[:U, :V] = f.depth
    key[:U, :V] = f.ids.astype(np.int32) * 16 + f.data.astype(np.int32)
    valid[:U, :V] = True

    def children(a: np.ndarray) -> np.ndarray:                 # (nu, nv, factor*factor)
        return a.reshape(nu, factor, nv, factor).transpose(0, 2, 1, 3).reshape(nu, nv, -1)

    d, k, ok = children(depth), children(key), children(valid)
    m = d >= 0
    out_d = np.full((nu, nv), EMPTY, np.int16)
    out_k = np.zeros((nu, nv), np.int32)
    if rule == "center":
        cu = np.minimum(factor // 2, np.maximum(U - np.arange(nu) * factor, 1) - 1)
        cv = np.minimum(factor // 2, np.maximum(V - np.arange(nv) * factor, 1) - 1)
        c = (cu[:, None] * factor + cv[None, :])[..., None]
        out_d = np.take_along_axis(d, c, -1)[..., 0].astype(np.int16)
        out_k = np.take_along_axis(k, c, -1)[..., 0]
    else:
        n_fill = m.sum(-1)
        filled = n_fill > 0 if rule == "any" else n_fill * 2 >= ok.sum(-1)
        for a, b in zip(*np.nonzero(filled)):
            mm = m[a, b]
            out_d[a, b] = d[a, b][mm].min() if rule == "any" else _mode(d[a, b][mm])
            out_k[a, b] = _mode(k[a, b][mm])
    mask = out_d >= 0
    return Facade(face=f.face, depth=out_d,
                  ids=np.where(mask, out_k // 16, 0).astype(np.int16),
                  data=np.where(mask, out_k % 16, 0).astype(np.int16), extent=f.extent)


def upsample(f: Facade, factor: int, shape: Tuple[int, int]) -> Facade:
    """A coarse face repeated back to `shape` cells, the naive decode of a level."""
    rep = lambda a: np.repeat(np.repeat(a, factor, 0), factor, 1)[:shape[0], :shape[1]]
    return Facade(face=f.face, depth=np.ascontiguousarray(rep(f.depth)),
                  ids=np.ascontiguousarray(rep(f.ids)), data=np.ascontiguousarray(rep(f.data)),
                  extent=f.extent)


# --- pictures ----------------------------------------------------------------
#: How each face is drawn: whether `u` runs right to left, so that walking the
#: sides in `SIDES` order shares each corner between neighbouring pictures, and
#: whether `v` is height (drawn upward) or a plan axis (drawn downward).
_FLIP_U = {"-z": False, "+x": False, "+z": True, "-x": True, "+y": False, "-y": True}
#: Depth darkens a column by this much at the far end of the extent.
_SHADE = 0.55


def face_picture(f: Facade) -> np.ndarray:
    """One face as an RGB uint8 image, one pixel per column: the block's base
    colour, darkened with depth, white where see-through."""
    u, v = f.depth.shape
    s = Structure(block_ids=f.ids.astype(np.int32)[:, :, None],
                  block_data=f.data.astype(np.int32)[:, :, None])
    rgb = s.facecolors()[:, :, 0, :3]                         # (u, v, 3) in 0..1
    shade = 1.0 - _SHADE * np.clip(f.depth, 0, None) / max(f.extent - 1, 1)
    rgb = rgb * shade[..., None]
    rgb = np.where(f.mask[..., None], rgb, 1.0)
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2)  # rows = v
    if f.axis != 1:
        img = img[::-1]                                        # height drawn upward
    if _FLIP_U[f.face]:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def sheet(facades: Mapping[str, Facade] | Iterable[Facade], scale: int = 8,
          gap: int = 1) -> np.ndarray:
    """Every face side by side: the sides in walking order on one row, the roof
    plan and underside (when present) on a second. `scale` pixels per block."""
    faces = list(facades.values()) if isinstance(facades, Mapping) else list(facades)
    rows = [[f for f in faces if f.axis != 1], [f for f in faces if f.axis == 1]]
    rows = [r for r in rows if r]
    pics = [[np.repeat(np.repeat(face_picture(f), scale, 0), scale, 1) for f in r]
            for r in rows]
    width = max(sum(p.shape[1] for p in r) + gap * (len(r) + 1) for r in pics)
    height = sum(max(p.shape[0] for p in r) for r in pics) + gap * (len(pics) + 1)
    out = np.full((height, width, 3), 255, np.uint8)
    y = gap
    for r in pics:
        x, h = gap, max(p.shape[0] for p in r)
        for p in r:
            out[y + h - p.shape[0]:y + h, x:x + p.shape[1]] = p   # bottom-aligned
            x += p.shape[1] + gap
        y += h + gap
    return out
