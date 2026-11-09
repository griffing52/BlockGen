"""Interior blindness on mesh data: the Fill/Strip probes outside Minecraft.

The paper's central measurement is that filling every sealed room of a voxel
build barely moves a render-and-embed metric while a voxel descriptor moves by
hundreds of noise floors. A reviewer's first objection is that this is a fact
about Minecraft. This module rebuilds the same experiment on ordinary textured
triangle meshes (Objaverse buildings) with the metrics mesh-generation papers
report, so the answer is a table rather than an argument.

Everything here is geometry on one normalised mesh; the statistics (KID, FD,
noise floors, MMD/COV/1-NNA) are imported from the voxel suite unchanged.

Frame
    Every mesh is centred on its bounding box and scaled so its longest side is
    1, y up (glTF convention, which is also the voxel suite's convention).

Visibility ("exterior")
    A face is *visible* if it is the first hit of at least one ray from an
    orthographic camera at infinity in one of `visibility_dirs()` directions:
    64 Fibonacci-sphere directions plus every evaluation camera, sampled on a
    `res x res` grid. This is "visible from some exterior viewpoint", the mesh
    analogue of the voxel suite's faces-bordering-exterior-air.

Hidden volume
    An empty cell of a `R^3` grid over the bounding box is *hidden* if a ray
    from its centre toward every visibility direction hits the mesh. These are
    the cells no exterior camera can see into: sealed rooms, plus room volume
    seen only through apertures narrower than the ray grid can find.

Probes (identical exterior by construction, verified by `exterior_delta`)
    strip   delete every non-visible face. The generator that ships a shell.
    fill    strip, plus a solid voxel mass over the hidden volume (eroded one
            cell so it cannot poke through the shell), coloured with the
            object's own dominant colour. The generator that ships a solid.
Positive controls (change the exterior)
    noise   Gaussian vertex jitter, sigma = 1% of the longest side.
    wall    delete every face whose centroid lies in the outer 30% slab on +x:
            a missing wall, which also exposes the interior to the cameras.
    decim   quadric decimation to 5% of the faces.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The paper's four-view protocol, (azimuth, elevation) in degrees.
VIEWS4 = ((45.0, 30.0), (135.0, 30.0), (225.0, 30.0), (315.0, 30.0))
VIEWS8 = tuple((float(a), 30.0) for a in range(0, 360, 45))
VIEWS16 = VIEWS8 + tuple((float(a) + 22.5, e) for a in range(0, 360, 90)
                         for e in (-10.0, 60.0))
HIDDEN_R = 48
VOXEL_R = 32


# --- the object ---------------------------------------------------------------
@dataclass
class MeshObj:
    uid: str
    vertices: np.ndarray          # (V, 3) float32, normalised frame
    faces: np.ndarray             # (F, 3) int32
    face_rgb: np.ndarray          # (F, 3) uint8
    #: Points known to be inside solid mass (the `fill` probe's core), used by
    #: the voxeliser, which otherwise only sees faces.
    solid_pts: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def tm(self):
        import trimesh
        return trimesh.Trimesh(self.vertices, self.faces, process=False)

    def submesh(self, keep: np.ndarray) -> "MeshObj":
        f = self.faces[keep]
        used, inv = np.unique(f.reshape(-1), return_inverse=True)
        return replace(self, vertices=self.vertices[used],
                       faces=inv.reshape(-1, 3).astype(np.int32),
                       face_rgb=self.face_rgb[keep], solid_pts=None)

    @property
    def face_areas(self) -> np.ndarray:
        t = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def _face_colours(g) -> np.ndarray:
    """Per-face RGB for one trimesh geometry: texture sampled at the vertices,
    else vertex/face colours, else the material's main colour, else grey."""
    n = len(g.faces)
    try:
        vis = g.visual
        if vis.kind == "texture":
            vis = vis.to_color()
        if vis.kind == "vertex":
            vc = np.asarray(vis.vertex_colors)[:, :3].astype(np.float32)
            return vc[g.faces].mean(1).astype(np.uint8)
        if vis.kind == "face":
            return np.asarray(vis.face_colors)[:, :3].astype(np.uint8)
        mat = getattr(g.visual, "material", None)
        if mat is not None and hasattr(mat, "main_color"):
            return np.tile(np.asarray(mat.main_color)[:3].astype(np.uint8), (n, 1))
    except Exception:
        pass
    return np.full((n, 3), 180, np.uint8)


def load_glb(path: str, uid: str = "", max_faces: int = 400_000) -> Optional[MeshObj]:
    """Flatten a GLB scene into one normalised, per-face-coloured mesh."""
    import trimesh
    scene = trimesh.load(path, force="scene", process=False)
    vs, fs, cs, off = [], [], [], 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry.get(gname)
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        vs.append(trimesh.transform_points(np.asarray(g.vertices, np.float64), T))
        fs.append(np.asarray(g.faces, np.int64) + off)
        cs.append(_face_colours(g))
        off += len(g.vertices)
        if sum(len(f) for f in fs) > max_faces:
            return None
    if not fs:
        return None
    v = np.concatenate(vs)
    f = np.concatenate(fs)
    c = np.concatenate(cs)
    # drop degenerate faces
    t = v[f]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    ok = area > 0
    f, c = f[ok], c[ok]
    lo, hi = v.min(0), v.max(0)
    ext = float((hi - lo).max())
    if not np.isfinite(ext) or ext <= 0:
        return None
    v = (v - (lo + hi) / 2.0) / ext
    used, inv = np.unique(f.reshape(-1), return_inverse=True)
    return MeshObj(uid=uid, vertices=v[used].astype(np.float32),
                   faces=inv.reshape(-1, 3).astype(np.int32), face_rgb=c)


# --- directions and rays --------------------------------------------------------
def view_dir(az: float, el: float) -> np.ndarray:
    """Unit vector from the object toward the camera (textured.py convention)."""
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def fibonacci_dirs(n: int = 64) -> np.ndarray:
    i = np.arange(n) + 0.5
    y = 1 - 2 * i / n
    r = np.sqrt(1 - y * y)
    phi = math.pi * (3 - math.sqrt(5)) * i
    return np.stack([r * np.cos(phi), y, r * np.sin(phi)], 1)


def visibility_dirs(n_fib: int = 64) -> np.ndarray:
    ev = [view_dir(a, e) for a, e in VIEWS4 + VIEWS16]
    return np.concatenate([fibonacci_dirs(n_fib), np.asarray(ev)], 0)


def _basis(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _intersector(m: MeshObj):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    return RayMeshIntersector(m.tm())


def visible_faces(m: MeshObj, dirs: Optional[np.ndarray] = None, res: int = 320
                  ) -> np.ndarray:
    """`(F,)` bool: face is the first hit of some exterior orthographic ray."""
    dirs = visibility_dirs() if dirs is None else dirs
    inter = _intersector(m)
    vis = np.zeros(len(m.faces), bool)
    s = np.linspace(-0.88, 0.88, res)
    S, T = np.meshgrid(s, s)
    S, T = S.reshape(-1, 1), T.reshape(-1, 1)
    for d in dirs:
        u, v = _basis(d)
        origins = S * u + T * v + 2.0 * d
        idx = inter.intersects_first(origins, np.tile(-d, (len(origins), 1)))
        vis[idx[idx >= 0]] = True
    return vis


@dataclass
class Grid:
    lo: np.ndarray      # (3,) world corner of cell (0,0,0)
    pitch: float
    shape: Tuple[int, int, int]

    def centres(self) -> np.ndarray:
        idx = np.indices(self.shape).reshape(3, -1).T
        return self.lo + (idx + 0.5) * self.pitch

    def cell_of(self, pts: np.ndarray) -> np.ndarray:
        ijk = np.floor((pts - self.lo) / self.pitch).astype(np.int64)
        return np.clip(ijk, 0, np.asarray(self.shape) - 1)


def grid_for(m: MeshObj, R: int) -> Grid:
    lo, hi = m.vertices.min(0).astype(np.float64), m.vertices.max(0).astype(np.float64)
    pitch = 1.0 / R
    shape = tuple(int(max(1, math.ceil((hi[k] - lo[k]) / pitch - 1e-6))) for k in range(3))
    centre = (lo + hi) / 2.0
    lo = centre - np.asarray(shape) * pitch / 2.0
    return Grid(lo=lo, pitch=pitch, shape=shape)


def surface_points(m: MeshObj, k: int, rng: np.random.Generator,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Area-weighted uniform samples on the faces (optionally a face subset)."""
    faces = m.faces if mask is None else m.faces[mask]
    if len(faces) == 0:
        return np.zeros((k, 3), np.float32)
    t = m.vertices[faces].astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if a.sum() <= 0:
        a = np.ones_like(a)
    fi = rng.choice(len(faces), size=k, p=a / a.sum())
    r1, r2 = rng.random(k), rng.random(k)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    tt = t[fi]
    p = tt[:, 0] + r1[:, None] * (tt[:, 1] - tt[:, 0]) + r2[:, None] * (tt[:, 2] - tt[:, 0])
    return p.astype(np.float32)


def cutaway_visible_faces(m: MeshObj, views: Sequence[Tuple[float, float]] = VIEWS4,
                          res: int = 320) -> np.ndarray:
    """`(F,)` bool: first hit of some ray of a cutaway camera, i.e. an
    orthographic ray that starts on the plane through the bounding-box centre
    normal to the view direction and travels away from the camera. This is
    what `render(cut=True)` shows (its near plane is that same plane)."""
    inter = _intersector(m)
    vis = np.zeros(len(m.faces), bool)
    lo, hi = m.vertices.min(0).astype(np.float64), m.vertices.max(0).astype(np.float64)
    centre = (lo + hi) / 2.0
    s = np.linspace(-0.88, 0.88, res)
    S, T = np.meshgrid(s, s)
    S, T = S.reshape(-1, 1), T.reshape(-1, 1)
    for az, el in views:
        d = view_dir(az, el)
        u, v = _basis(d)
        origins = centre + S * u + T * v
        idx = inter.intersects_first(origins, np.tile(-d, (len(origins), 1)))
        vis[idx[idx >= 0]] = True
    return vis


def surface_samples(m: MeshObj, k: int, rng: np.random.Generator,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """`(k, 6)` float32 xyz + rgb in [0, 1]: area-weighted uniform samples on
    the faces (or a face subset), each carrying its face colour. The sampler
    behind every 3D-native row; an empty subset gives k points at the origin
    coloured mid-grey, which every statistic then reads as maximally wrong."""
    faces_ix = np.arange(len(m.faces)) if mask is None else np.flatnonzero(mask)
    if len(faces_ix) == 0:
        return np.concatenate([np.zeros((k, 3), np.float32),
                               np.full((k, 3), 0.5, np.float32)], 1)
    t = m.vertices[m.faces[faces_ix]].astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if a.sum() <= 0:
        a = np.ones_like(a)
    fi = rng.choice(len(faces_ix), size=k, p=a / a.sum())
    r1, r2 = rng.random(k), rng.random(k)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    tt = t[fi]
    p = tt[:, 0] + r1[:, None] * (tt[:, 1] - tt[:, 0]) + r2[:, None] * (tt[:, 2] - tt[:, 0])
    rgb = m.face_rgb[faces_ix[fi]].astype(np.float32) / 255.0
    return np.concatenate([p.astype(np.float32), rgb], 1)


#: The three samplers of the 3D-native rows (notes §65).
SAMPLERS = ("all", "vis", "cut")


def sampler_masks(m: MeshObj) -> Dict[str, Optional[np.ndarray]]:
    """Face subsets per sampler: every face; faces visible from some exterior
    viewpoint (`visible_faces`, the set Strip keeps); faces visible in the four
    cutaway views (`cutaway_visible_faces`)."""
    return {"all": None, "vis": visible_faces(m), "cut": cutaway_visible_faces(m)}


def surface_occupancy(m: MeshObj, g: Grid, rng: np.random.Generator,
                      density: float = 40.0) -> np.ndarray:
    """Cells touched by the surface, via dense area-weighted sampling (about
    `density` samples per cell-face area) plus every vertex and face centroid."""
    occ = np.zeros(g.shape, bool)
    if len(m.faces) == 0:
        return occ
    area = float(m.face_areas.sum())
    k = int(min(2_000_000, max(20_000, density * area / g.pitch ** 2)))
    pts = np.concatenate([surface_points(m, k, rng), m.vertices,
                          m.vertices[m.faces].mean(1)])
    ijk = g.cell_of(pts)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occ


def hidden_cells(m: MeshObj, R: int = HIDDEN_R, dirs: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Grid]:
    """`(mask, grid)`: empty cells from which every visibility direction is
    blocked by the mesh. Candidate cells are pruned direction by direction."""
    dirs = visibility_dirs() if dirs is None else dirs
    rng = rng or np.random.default_rng(0)
    g = grid_for(m, R)
    occ = surface_occupancy(m, g, rng)
    cand = np.flatnonzero(~occ.reshape(-1))
    cen = g.centres()
    inter = _intersector(m)
    for d in dirs:
        if len(cand) == 0:
            break
        hit = inter.intersects_any(cen[cand], np.tile(d, (len(cand), 1)))
        cand = cand[hit]
    mask = np.zeros(int(np.prod(g.shape)), bool)
    mask[cand] = True
    return mask.reshape(g.shape), g


#: The stricter "multi-room, real interior" rule used for the paper-scale probe
#: (notes §65). An object passes iff all four hold.
STRICT_RULE = {"min_hidden_area": 0.30, "min_hidden_vol": 0.05, "min_rooms": 2,
               "min_deep_area": 0.03}


def interior_stats(m: MeshObj, vis: np.ndarray, hidden: np.ndarray,
                   R: int = HIDDEN_R) -> Dict[str, float]:
    """Statistics behind the strict interior rule, from the cached masks.

    rooms       6-connected components of the hidden volume holding at least
                max(27 cells, 0.2% of the box): separate sealed spaces.
    deep_area   share of the total face area on hidden faces whose centroid is
                at least 2 cells (2/48 of the longest side) from every visible
                face: interior walls, floors and furniture, as opposed to the
                inner skin of a thick exterior wall.
    """
    from scipy import ndimage
    g = grid_for(m, R)
    lab, n = ndimage.label(hidden, ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    rooms = int((sizes >= max(27, 0.002 * hidden.size)).sum())
    shell = surface_occupancy(m.submesh(vis), g, np.random.default_rng(0))
    dist = ndimage.distance_transform_edt(~shell)
    ijk = g.cell_of(m.vertices[m.faces].mean(1))
    d = dist[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    a = m.face_areas
    tot = max(float(a.sum()), 1e-12)
    return {"rooms": rooms,
            "deep_area": float((a * ((~vis) & (d >= 2))).sum() / tot),
            "hidden_area": float(a[~vis].sum() / tot),
            "hidden_vol": float(hidden.mean()),
            "largest_room_vol": float(sizes.max() / hidden.size) if len(sizes) else 0.0}


def passes_strict(st: Dict[str, float], rule: Dict[str, float] = STRICT_RULE) -> bool:
    return bool(st["hidden_area"] >= rule["min_hidden_area"]
                and st["hidden_vol"] >= rule["min_hidden_vol"]
                and st["rooms"] >= rule["min_rooms"]
                and st["deep_area"] >= rule["min_deep_area"])


# --- probes ---------------------------------------------------------------------
def dominant_colour(m: MeshObj, mask: Optional[np.ndarray] = None) -> np.ndarray:
    a = m.face_areas
    c = m.face_rgb.astype(np.int64)
    if mask is not None:
        a, c = a[mask], c[mask]
    q = (c // 32)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    w = np.bincount(key, weights=a, minlength=512)
    best = int(w.argmax())
    sel = key == best
    return np.average(c[sel], axis=0, weights=a[sel] + 1e-12).astype(np.uint8)


def voxel_surface(mask: np.ndarray, g: Grid, return_cells: bool = False):
    """Triangles of every cell face of `mask` that borders a non-mask cell.
    With `return_cells`, also the `(F, 3)` cell index each triangle belongs to."""
    verts, faces, owner = [], [], []
    p = np.pad(mask, 1)
    # (axis, sign) -> quad corner offsets in unit-cube coordinates, CCW outward.
    quads = {
        (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    off = 0
    for (ax, sg), corners in quads.items():
        nb = np.roll(p, -sg, axis=ax)[1:-1, 1:-1, 1:-1]
        cells = np.argwhere(mask & ~nb)
        if len(cells) == 0:
            continue
        c = np.asarray(corners, np.float64)
        q = cells[:, None, :] + c[None]                  # (n, 4, 3)
        verts.append(q.reshape(-1, 3))
        b = off + 4 * np.arange(len(cells))[:, None]
        faces.append(np.concatenate([b + [0, 1, 2], b + [0, 2, 3]], 0))
        owner.append(np.concatenate([cells, cells], 0))
        off += 4 * len(cells)
    if not verts:
        e = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        return e + (np.zeros((0, 3), np.int64),) if return_cells else e
    v = g.lo + np.concatenate(verts) * g.pitch
    out = (v.astype(np.float32), np.concatenate(faces).astype(np.int32))
    return out + (np.concatenate(owner),) if return_cells else out


def strip(m: MeshObj, vis: np.ndarray) -> MeshObj:
    return m.submesh(vis)


def fill(m: MeshObj, vis: np.ndarray, hidden: np.ndarray, g: Grid) -> MeshObj:
    from scipy import ndimage
    shell = m.submesh(vis)
    core = ndimage.binary_erosion(hidden, ndimage.generate_binary_structure(3, 1),
                                  border_value=0)
    v, f = voxel_surface(core, g)
    col = dominant_colour(m, vis)
    out = MeshObj(uid=m.uid,
                  vertices=np.concatenate([shell.vertices, v]).astype(np.float32),
                  faces=np.concatenate([shell.faces, f + len(shell.vertices)]).astype(np.int32),
                  face_rgb=np.concatenate([shell.face_rgb, np.tile(col, (len(f), 1))]),
                  meta=dict(m.meta))
    # Cell centres of the *un-eroded* hidden volume: the voxeliser fills these.
    out.solid_pts = g.centres()[hidden.reshape(-1)].astype(np.float32)
    out.meta["fill_faces"] = int(len(f))
    return out


def _welded(m: MeshObj) -> Tuple[np.ndarray, np.ndarray]:
    key = np.round(m.vertices.astype(np.float64) * 1e6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inv.reshape(-1)


def vertex_noise(m: MeshObj, sigma: float, rng: np.random.Generator) -> MeshObj:
    """Jitter welded vertex positions, so seams move together and do not tear."""
    first, inv = _welded(m)
    jit = rng.normal(0.0, sigma, size=(len(first), 3)).astype(np.float32)
    return replace(m, vertices=m.vertices + jit[inv], solid_pts=None)


def wall_cut(m: MeshObj, frac: float = 0.15) -> MeshObj:
    c = m.vertices[m.faces].mean(1)
    xmax, xmin = m.vertices[:, 0].max(), m.vertices[:, 0].min()
    keep = c[:, 0] < xmax - frac * (xmax - xmin)
    return m.submesh(keep)


def decimate(m: MeshObj, ratio: float = 0.05) -> MeshObj:
    import fast_simplification
    from scipy.spatial import cKDTree
    first, inv = _welded(m)
    v = m.vertices[first].astype(np.float32)
    f = inv[m.faces].astype(np.int32)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    v2, f2 = fast_simplification.simplify(v, f, target_reduction=1.0 - ratio)
    cen = m.vertices[m.faces].mean(1)
    _, nn = cKDTree(cen).query(v2[f2].mean(1))
    return MeshObj(uid=m.uid, vertices=v2.astype(np.float32), faces=f2.astype(np.int32),
                   face_rgb=m.face_rgb[nn], meta=dict(m.meta))


# --- rendering --------------------------------------------------------------------
_RENDERERS: Dict[int, object] = {}


def _renderer(px: int):
    import pyrender
    r = _RENDERERS.get(px)
    if r is None:
        r = pyrender.OffscreenRenderer(px, px)
        _RENDERERS[px] = r
    return r


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    if abs(fwd @ up) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    tu = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, tu, -fwd, eye
    return pose


def _pyrender_mesh(m: MeshObj):
    """Flat-shaded, per-face-coloured, opaque, double-sided primitive. Built
    directly because `pyrender.Mesh.from_trimesh` drops face colours whenever a
    material is supplied, and its default material is alpha-blended."""
    import pyrender
    tri = m.vertices[m.faces].astype(np.float32)                    # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    col = np.concatenate([m.face_rgb.astype(np.float32) / 255.0,
                          np.ones((len(m.faces), 1), np.float32)], 1)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                                             metallicFactor=0.0, roughnessFactor=1.0,
                                             alphaMode="OPAQUE", doubleSided=True)
    prim = pyrender.Primitive(positions=tri.reshape(-1, 3),
                              normals=np.repeat(n, 3, axis=0).astype(np.float32),
                              color_0=np.repeat(col, 3, axis=0), material=mat, mode=4)
    return pyrender.Mesh([prim])


def render(m: MeshObj, views: Sequence[Tuple[float, float]] = VIEWS4, px: int = 224,
           cut: bool = False, depth: bool = False):
    """Opaque, double-sided, lit renders on white, orthographic, framed on the
    mesh's own bounding box (as the voxel renderer frames its crop).

    `cut=True` sets the near plane through the bounding-box centre, so each
    view is a section that shows the inside of the far half.
    Returns a list of (px, px, 3) uint8, plus a list of world-space points
    back-projected from the depth buffer when `depth=True`.
    """
    import pyrender
    pm = _pyrender_mesh(m)
    lo, hi = m.vertices.min(0).astype(np.float64), m.vertices.max(0).astype(np.float64)
    centre = (lo + hi) / 2.0
    corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])]) - centre
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    dist = 2.5 * radius + 1.0
    imgs, clouds = [], []
    r = _renderer(px)
    for az, el in views:
        d = view_dir(az, el)
        pose = _look_at(centre + d * dist, centre)
        mag = max(float(np.abs(corners @ pose[:3, 0]).max()),
                  float(np.abs(corners @ pose[:3, 1]).max()), 1e-3) * 1.05
        znear = dist if cut else 0.01
        zfar = dist + 2.0 * radius
        scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45))
        scene.add(pm)
        cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=znear, zfar=zfar)
        scene.add(cam, pose=pose)
        # key light from the camera's upper left, fixed in camera space
        lp = _look_at(centre + view_dir(az - 35.0, min(el + 30.0, 85.0)) * dist, centre)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.6), pose=lp)
        color, dep = r.render(scene)
        imgs.append(np.ascontiguousarray(color[..., :3]))
        if depth:
            clouds.append(_unproject(dep, pose, mag, znear, zfar, px))
    return (imgs, clouds) if depth else imgs


def _unproject(dep: np.ndarray, pose: np.ndarray, mag: float, n: float, f: float,
               px: int) -> np.ndarray:
    """World points from pyrender's depth, undoing its perspective-only
    linearisation (it is wrong for orthographic cameras)."""
    ii, jj = np.nonzero(dep > 0)
    D = dep[ii, jj].astype(np.float64)
    x = (f + n - 2.0 * n * f / D) / (f - n)
    zd = (x + 1.0) / 2.0
    z = n + zd * (f - n)
    X = ((jj + 0.5) / px * 2.0 - 1.0) * mag
    Y = (1.0 - (ii + 0.5) / px * 2.0) * mag
    pc = np.stack([X, Y, -z, np.ones_like(z)], 1)
    return (pc @ pose.T)[:, :3].astype(np.float32)


def exterior_delta(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    """Mean absolute pixel difference (0-255) between two view lists."""
    return float(np.mean([np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()
                          for x, y in zip(a, b)]))


# --- voxel bridge ------------------------------------------------------------------
def voxelize(m: MeshObj, R: int = VOXEL_R, rng: Optional[np.random.Generator] = None):
    """A `Structure` (block id 1 = occupied) on an R-per-unit grid: surface
    cells plus any cell holding a `solid_pts` point. y is up, as in Minecraft."""
    from blockgen.utils.data import Structure
    rng = rng or np.random.default_rng(0)
    g = grid_for(m, R)
    occ = surface_occupancy(m, g, rng)
    if m.solid_pts is not None and len(m.solid_pts):
        ijk = g.cell_of(m.solid_pts)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    bi = occ.astype(np.int32)
    if not bi.any():
        bi[0, 0, 0] = 1
    return Structure(block_ids=bi, block_data=np.zeros_like(bi))


def structure_mesh(s, canvas: float = 32.0, uid: str = "") -> MeshObj:
    """A voxel build as a triangle mesh of its exposed block faces.

    Every face of an occupied voxel that borders air (exterior or sealed) is a
    quad coloured with the block's base colour (`Structure.facecolors`), so the
    mesh samplers see exactly the faces a cube renderer would draw. Frame: the
    voxel suite's (`pointcloud.points`): centred on the bounding box, divided
    by the canvas side, no per-shape rescaling; y up."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    shape = tuple(int(a) for a in occ.shape)
    pitch = 1.0 / canvas
    g = Grid(lo=-np.asarray(shape, np.float64) * pitch / 2.0, pitch=pitch, shape=shape)
    v, f, cells = voxel_surface(occ, g, return_cells=True)
    col = c.facecolors()[..., :3]
    rgb = (np.clip(col[cells[:, 0], cells[:, 1], cells[:, 2]], 0, 1) * 255).astype(np.uint8)
    return MeshObj(uid=uid, vertices=v, faces=f, face_rgb=rgb)


# --- cache I/O -----------------------------------------------------------------------
def save_npz(path: str, m: MeshObj, **extra) -> None:
    np.savez_compressed(path, vertices=m.vertices, faces=m.faces, face_rgb=m.face_rgb,
                        **extra)


def load_npz(path: str, uid: str = "") -> Tuple[MeshObj, Dict[str, np.ndarray]]:
    z = np.load(path)
    m = MeshObj(uid=uid, vertices=z["vertices"], faces=z["faces"], face_rgb=z["face_rgb"])
    return m, {k: z[k] for k in z.files if k not in ("vertices", "faces", "face_rgb")}
