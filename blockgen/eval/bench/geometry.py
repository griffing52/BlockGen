"""Structure-space metrics: the half of build quality a camera cannot see.

Why this module exists
----------------------
Every realism number in the suite so far is computed on *renders*. That was the
right call -- rendering is what makes a 16^3 arm and a 32^3 arm comparable at all
(T20) -- but it buys comparability by throwing away everything the camera cannot
reach. The cost is measurable, and it is large:

    filling every enclosed air cell of a real house -- a median **31% increase in
    block count** (mean 37%, affecting 91% of builds), which deletes every room
    in the corpus -- leaves MV-DINO-KID at **0.000 [-0.002, 0.004]** against a
    held-out-real floor of **-0.001 [-0.004, 0.003]**. The intervals overlap.
    Measured on the same 128 builds, `geom_kid` moves from -0.09 to **22.1**.

So a generator that emits solid blobs shaped like houses is, to the primary
realism metric, indistinguishable from a generator that emits houses. That is
not a hypothetical failure mode: T23d measured `native_oriented` at an
enclosed-air ratio of 0.005 against a real 0.114, i.e. essentially solid, while
scoring respectably on KID.

`solidify` is now a ladder rung (`probes.solidify`) so this blindness is a
recorded property of the metric rather than folklore.

What this module adds
---------------------
A per-structure **geometric descriptor** and a distribution distance over it,
`geom_kid`, built to be the structural counterpart of `mv_dino_kid`:

* it goes through the *identical* unbiased kernel estimator (`distances.kid`),
  so the two numbers are constructed the same way and read on the same scale;
* it goes through the *identical* validation ladder, so it has to earn the right
  to adjudicate exactly as the render metrics did;
* it needs no GPU, no renderer and no network, so it runs in CI and can gate a
  training run rather than being saved for the end.

**Every component is exactly invariant to the D4 yaw group** (four rotations
about the vertical axis, and the x-mirror) by construction, not by luck: local
patterns are canonicalized onto orbit representatives, and every scalar is built
from quantities that commute with the group. The ladder's invariance gate is
therefore satisfied by design, and any failure of it indicates an implementation
bug rather than a metric that measures pose.

**What it is deliberately blind to.** The descriptor reads occupancy only, so
material noise and material shuffle move it by exactly zero. That mirrors the
FAST palette tier, which is blind to arrangement by construction. Neither is a
defect; they are complementary halves, and the ladder scores each on the rungs
it is meant to see (`GEOMETRY_ORDERED_NOISE` replaces the material-noise gate
with `occupancy_noise`, its structural analogue).

Definitions worth reading before trusting a number
--------------------------------------------------
* **Thickness** is the taxicab distance transform of the solid, padded with air,
  so a one-block wall reads 1 everywhere and a solid cube reads up to half its
  width. It is an integer, which keeps the histogram bins honest.

* **Wall fraction** measures planarity: exposed faces are grouped into
  *coplanar* connected patches (connectivity restricted to the plane
  perpendicular to the face normal, done with one labelling pass per direction
  rather than one per slice), and the statistic is the share of faces sitting in
  a patch of at least `WALL_MIN_AREA` faces. Rubble scores low, walls score high.

* **Aperture-aware interior.** `topology.enclosed_air_ratio` counts air that is
  fully sealed, so a room with an open doorway counts as no room at all. Closing
  the solid seals openings up to two blocks wide; the air that becomes enclosed
  only after that closing is interior reachable through a door or window.
  `interior_ratio_open` is the sum, and it is the honest "does this build have
  rooms" number.

  This is not a marginal correction. On 200 held-out real houses, **9.0% have no
  sealed interior at all, and 94.4% of those do have a room** -- they simply have
  a doorway. Across the whole set, apertures account for **20.5% of all interior
  volume**. Every previous statement in this project about generated builds
  "having no interiors" was measured with the sealed-only counter and is
  understated for real data by that much.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from blockgen.eval.bench import stats
from blockgen.utils.data import Structure

# --- local occupancy patterns ----------------------------------------------
#: Side of the local pattern window. 2 gives 2^8 = 256 raw patterns, which
#: canonicalize to 55 D4 orbits -- small enough to estimate stably from the
#: n = 64 samples the suite actually has, where a 3^3 window (2^27 patterns)
#: would be pure noise.
PATCH = 2
STRUCT6 = ndimage.generate_binary_structure(3, 1)
WALL_MIN_AREA = 9          # a 3x3 patch: the smallest thing worth calling a wall
THICKNESS_BINS = (1, 2, 3, 4, 5)       # taxicab depth; last bin is "5 or more"
HEIGHT_BINS = 8


def _d4_orbit_table() -> Tuple[np.ndarray, int]:
    """Map each 2x2x2 occupancy pattern to its D4-orbit representative.

    The group is the four rotations about the vertical axis together with the
    x-mirror -- exactly the transformations `probes.rot90_*` and
    `probes.mirror_x` apply, and exactly the ones that leave a Minecraft build's
    quality unchanged. The vertical axis is *not* included: a build and its
    upside-down copy are not equally good, and collapsing them would throw away
    the roof/floor asymmetry that distinguishes a house from a pile.
    """
    codes = np.arange(1 << (PATCH ** 3))
    # bit (dx, dy, dz) -> index 4*dx + 2*dy + dz
    bits = ((codes[:, None] >> np.arange(8)[None, :]) & 1).astype(bool)
    grids = bits.reshape(-1, PATCH, PATCH, PATCH)

    def encode(g: np.ndarray) -> np.ndarray:
        flat = g.reshape(len(g), -1).astype(np.int64)
        return (flat << np.arange(8)).sum(1)

    orbit_min = codes.copy()
    for mirror in (False, True):
        base = grids[:, ::-1] if mirror else grids
        for k in range(4):
            orbit_min = np.minimum(orbit_min, encode(np.rot90(base, k, axes=(1, 3))))
    reps = np.unique(orbit_min)
    lookup = np.searchsorted(reps, orbit_min)
    return lookup.astype(np.int32), int(len(reps))


#: pattern code -> orbit index, and the number of orbits (55 for PATCH=2).
ORBIT_OF, N_ORBITS = _d4_orbit_table()

#: The all-empty orbit, dropped from every histogram: a padded volume is
#: overwhelmingly empty, so keeping it would make the histogram a very noisy
#: restatement of the block count and drown every structural signal.
EMPTY_ORBIT = int(ORBIT_OF[0])


def pattern_histogram(occ: np.ndarray) -> np.ndarray:
    """D4-canonical 2x2x2 occupancy-pattern distribution, empty orbit dropped.

    Windows are taken over the volume padded by one air cell on every side, so
    the patterns that describe the *surface* -- the ones that carry the
    architecture -- are counted rather than clipped away at the border.
    """
    occ = np.asarray(occ, dtype=bool)
    p = np.pad(occ, 1)
    code = np.zeros(tuple(d - 1 for d in p.shape), dtype=np.int64)
    for dx in range(PATCH):
        for dy in range(PATCH):
            for dz in range(PATCH):
                sub = p[dx:p.shape[0] - 1 + dx,
                        dy:p.shape[1] - 1 + dy,
                        dz:p.shape[2] - 1 + dz]
                code |= sub.astype(np.int64) << (4 * dx + 2 * dy + dz)
    hist = np.bincount(ORBIT_OF[code.ravel()], minlength=N_ORBITS).astype(float)
    hist[EMPTY_ORBIT] = 0.0
    total = hist.sum()
    return hist / total if total > 0 else hist


def thickness(occ: np.ndarray) -> np.ndarray:
    """Taxicab depth of every occupied voxel below the surface. Integer-valued.

    Padded with air, so "outside" counts as air and a free-standing one-block
    wall reads 1 rather than being measured against the array border.
    """
    occ = np.asarray(occ, dtype=bool)
    if not occ.any():
        return np.zeros(0, dtype=float)
    d = ndimage.distance_transform_cdt(np.pad(occ, 1), metric="taxicab")
    return d[1:-1, 1:-1, 1:-1][occ].astype(float)


def thickness_histogram(occ: np.ndarray) -> np.ndarray:
    t = thickness(occ)
    if t.size == 0:
        return np.zeros(len(THICKNESS_BINS))
    idx = np.clip(t.astype(int) - 1, 0, len(THICKNESS_BINS) - 1)
    return np.bincount(idx, minlength=len(THICKNESS_BINS)).astype(float) / t.size


def height_profile(occ: np.ndarray, bins: int = HEIGHT_BINS) -> np.ndarray:
    """Share of mass in each of `bins` equal slabs from floor to roof.

    Normalized by the build's own height, so it describes *shape* -- where the
    mass sits between ground and ridge -- and not size. A house puts mass in a
    dense base and a tapering roof; a blob is flat across the profile.
    """
    occ = np.asarray(occ, dtype=bool)
    per_y = occ.sum(axis=(0, 2)).astype(float)
    total = per_y.sum()
    if total <= 0:
        return np.zeros(bins)
    edges = np.linspace(0, len(per_y), bins + 1)
    idx = np.clip(np.digitize(np.arange(len(per_y)) + 0.5, edges[1:-1]), 0, bins - 1)
    return np.bincount(idx, weights=per_y, minlength=bins) / total


def _face_masks(occ: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Exposed faces per direction, as `(normal_axis, mask)` pairs."""
    out: List[Tuple[int, np.ndarray]] = []
    for axis in range(3):
        for shift in (1, -1):
            nb = np.roll(occ, shift, axis=axis)
            sl = [slice(None)] * 3
            sl[axis] = 0 if shift == 1 else -1
            nb[tuple(sl)] = False           # outside the array is air
            out.append((axis, occ & ~nb))
    return out


def _planar_structure(axis: int) -> np.ndarray:
    """Connectivity restricted to the plane perpendicular to `axis`.

    Labelling the 3-D face mask with this element is equivalent to labelling
    each 2-D slice independently, in one `ndimage.label` call instead of one per
    slice -- which is the difference between the tier running in seconds and in
    minutes over a corpus.
    """
    s = np.zeros((3, 3, 3), dtype=bool)
    s[1, 1, 1] = True
    for other in range(3):
        if other == axis:
            continue
        idx = [1, 1, 1]
        for d in (0, 2):
            idx[other] = d
            s[tuple(idx)] = True
    return s


_PLANAR = [_planar_structure(a) for a in range(3)]


def wall_fraction(occ: np.ndarray, min_area: int = WALL_MIN_AREA) -> float:
    """Share of exposed surface sitting in a coplanar patch of >= `min_area`.

    High for walls, roofs and floors; low for rubble, noise and lace. This is
    the statistic that separates "a building" from "a cloud with a building's
    silhouette", and the render metrics cannot see the difference once the
    silhouette matches.
    """
    occ = np.asarray(occ, dtype=bool)
    total = 0
    big = 0
    for axis, mask in _face_masks(occ):
        n_faces = int(mask.sum())
        if n_faces == 0:
            continue
        total += n_faces
        lab, n = ndimage.label(mask, structure=_PLANAR[axis])
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            big += int(sizes[sizes >= min_area].sum())
    return float(big) / total if total else 0.0


def surface_to_volume(occ: np.ndarray) -> float:
    """Exposed faces per occupied block. 6 for dust, ~0 for a large solid."""
    occ = np.asarray(occ, dtype=bool)
    n = int(occ.sum())
    if n == 0:
        return 0.0
    return float(sum(int(m.sum()) for _, m in _face_masks(occ))) / n


def yaw_symmetry(occ: np.ndarray) -> float:
    """Best mirror-symmetry IoU over the two horizontal axes.

    Buildings are close to bilaterally symmetric; sampled noise is not. Taking
    the max over x and z keeps the number invariant under the same yaw group the
    rest of the module respects.
    """
    occ = np.asarray(occ, dtype=bool)
    if not occ.any():
        return 0.0

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        u = int((a | b).sum())
        return float((a & b).sum()) / u if u else 0.0

    return max(iou(occ, occ[::-1]), iou(occ, occ[:, :, ::-1]))


#: Structuring element for the aperture-sealing closing. The 26-neighbourhood,
#: **not** the 6-neighbourhood used everywhere else for connectivity. Measured
#: on a hollow 9-cube: with the 6-connected cross, dilation cannot reach the cell
#: directly outside a one-block doorway, the erosion takes the plug straight back
#: out, and every aperture size scores zero -- the statistic was silently always
#: 0. The 26-connected cube seals openings up to two blocks wide and leaves a
#: 3x3 hole open, which is the distinction a door or window actually draws.
STRUCT26 = ndimage.generate_binary_structure(3, 3)


def interior_volumes(occ: np.ndarray) -> Tuple[int, int]:
    """`(sealed, reachable_through_apertures)` interior air cell counts.

    The first is what `topology.enclosed_air_ratio` already reports. The second
    is air that only becomes enclosed once the solid is closed -- that is, rooms
    whose door or window is at most two blocks wide. A house with an open
    doorway has a sealed interior of zero and an aperture interior of a whole
    room, which is why the sealed count alone understates real interiors.

    Calibrated on a hollow 9-cube with a 343-cell room:

        opening        sealed   through-aperture
        none              343                  0
        1x1 door            0                343
        1x2 door            0                343
        2x2 window          0                343
        3x3 hole            0                  0     <- no longer a room
        wall removed        0                  0
        solid cube          0                  0

    The 3x3 row is the intended cut-off, not a failure: past two blocks the
    opening stops being an aperture into a room and starts being one side of an
    open structure, and calling a pavilion's volume "interior" would make the
    statistic reward hollow shells with no walls.
    """
    occ = np.asarray(occ, dtype=bool)
    if not occ.any():
        return 0, 0
    filled = ndimage.binary_fill_holes(occ, structure=STRUCT6)
    sealed = filled & ~occ
    # Pad by more than the closing reaches, so the dilated solid never touches
    # the array border -- `binary_fill_holes` seeds from the border, and a solid
    # border would make it fill the entire volume.
    pad = 3
    closed = ndimage.binary_closing(np.pad(occ, pad), structure=STRUCT26)
    filled_c = ndimage.binary_fill_holes(closed, structure=STRUCT6)
    open_int = (filled_c & ~closed)[pad:-pad, pad:-pad, pad:-pad] & ~sealed & ~occ
    return int(sealed.sum()), int(open_int.sum())


# --- per-structure scalars --------------------------------------------------
#: Scalars compared against the real distribution, in report order. Each is
#: `distance_to_real`, never "higher is better" -- the reporting rule
#: `topology` establishes applies here for the same reason.
GEOMETRY_METRICS = (
    "thickness_mean",
    "thickness_p90",
    "surface_to_volume",
    "wall_frac",
    "yaw_symmetry",
    "interior_ratio_sealed",
    "interior_ratio_open",
    "height_entropy",
)


@dataclass
class GeometryStats:
    thickness_mean: float
    thickness_p90: float
    surface_to_volume: float
    wall_frac: float
    yaw_symmetry: float
    interior_ratio_sealed: float
    interior_ratio_open: float
    height_entropy: float
    pattern_hist: np.ndarray
    thickness_hist: np.ndarray
    height_hist: np.ndarray


def geometry_stats(s: Structure) -> GeometryStats:
    """Everything this module measures about one build, in a single pass."""
    occ = s.crop_to_non_air().occupied_mask
    vol = float(np.prod(occ.shape)) if occ.size else 1.0
    t = thickness(occ)
    sealed, open_int = interior_volumes(occ)
    hp = height_profile(occ)
    nz = hp[hp > 0]
    return GeometryStats(
        thickness_mean=float(t.mean()) if t.size else 0.0,
        thickness_p90=float(np.percentile(t, 90)) if t.size else 0.0,
        surface_to_volume=surface_to_volume(occ),
        wall_frac=wall_fraction(occ),
        yaw_symmetry=yaw_symmetry(occ),
        interior_ratio_sealed=sealed / vol,
        interior_ratio_open=(sealed + open_int) / vol,
        height_entropy=float(-(nz * np.log(nz)).sum()) if nz.size else 0.0,
        pattern_hist=pattern_histogram(occ),
        thickness_hist=thickness_histogram(occ),
        height_hist=hp,
    )


def geometry_table(structures: Sequence[Structure]) -> Dict[str, np.ndarray]:
    """Per-metric arrays over a set of structures (column-major `geometry_stats`)."""
    rows = [geometry_stats(s) for s in structures]
    return {k: np.array([getattr(r, k) for r in rows], dtype=float)
            for k in GEOMETRY_METRICS}


def geometry_report(generated: Sequence[Structure], real: Sequence[Structure], *,
                    n_boot: int = 1000, alpha: float = 0.05,
                    rng: np.random.Generator | None = None,
                    real_table: Dict[str, np.ndarray] | None = None
                    ) -> Dict[str, dict]:
    """Generated-vs-real geometry, in the same shape `topology.coherence_report`
    emits, so both render through the identical distance-to-real renderer."""
    rng = rng or np.random.default_rng(0)
    gen = geometry_table(generated)
    ref = real_table if real_table is not None else geometry_table(real)
    out: Dict[str, dict] = {}
    for key in GEOMETRY_METRICS:
        g, r = gen[key], ref[key]
        gp, glo, ghi = stats.bootstrap_ci(g, n_boot=n_boot, alpha=alpha, rng=rng)
        rp, rlo, rhi = stats.bootstrap_ci(r, n_boot=n_boot, alpha=alpha, rng=rng)
        d = stats.w1(g, r)
        spread = stats.robust_spread(r)
        out[key] = {"gen": {"mean": gp, "ci": [glo, ghi]},
                    "real": {"mean": rp, "ci": [rlo, rhi]},
                    "w1": d,
                    "w1_norm": (d / spread) if np.isfinite(spread) else float("nan"),
                    "real_spread": spread,
                    "direction": "distance_to_real"}
    return out


# --- the descriptor and its distribution distance ---------------------------
#: Descriptor block names and widths, in concatenation order.
def descriptor_layout() -> List[Tuple[str, int]]:
    return [("pattern", N_ORBITS), ("thickness", len(THICKNESS_BINS)),
            ("height", HEIGHT_BINS), ("scalars", len(GEOMETRY_METRICS) + 1)]


def descriptor(structures: Sequence[Structure]) -> np.ndarray:
    """`[N, D]` raw geometric descriptors, one row per build.

    Concatenates the three histograms with the scalar panel. `log1p(n_blocks)`
    is appended because every other component is deliberately size-free, and a
    benchmark that could not tell a cottage from a cathedral would be missing
    the most basic fact about a build.
    """
    rows: List[np.ndarray] = []
    for s in structures:
        g = geometry_stats(s)
        occ = s.crop_to_non_air().occupied_mask
        scal = [getattr(g, k) for k in GEOMETRY_METRICS]
        scal.append(float(np.log1p(int(occ.sum()))))
        rows.append(np.concatenate([g.pattern_hist, g.thickness_hist,
                                    g.height_hist, np.asarray(scal, float)]))
    if not rows:
        return np.zeros((0, sum(w for _, w in descriptor_layout())), dtype=float)
    return np.stack(rows).astype(np.float64)


@dataclass
class Standardizer:
    """Per-dimension whitening fitted **once on the reference set**.

    Fitting per arm would give every arm its own feature scale and make the
    kernel incomparable between them -- the same reason `distances.median_bandwidth`
    is frozen on the reference. `clip` bounds the influence of a single outlying
    build, which at n = 64 would otherwise dominate the statistic.
    """
    mean: np.ndarray
    scale: np.ndarray
    clip: float = 5.0

    @classmethod
    def fit(cls, x: np.ndarray, clip: float = 5.0) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        sd = x.std(axis=0)
        # A dimension that is constant on the reference carries no information;
        # dividing by its ~0 spread would turn float noise into a huge signal.
        sd = np.where(sd > 1e-8, sd, np.inf)
        return cls(mean=x.mean(axis=0), scale=sd, clip=clip)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = np.clip((np.asarray(x, np.float64) - self.mean) / self.scale,
                    -self.clip, self.clip)
        # Unit-norm rows, matching the DINO path, so `kid`'s default gamma = 1/D
        # lands in the same informative range for both metrics.
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def geom_features(structures: Sequence[Structure], std: Standardizer) -> np.ndarray:
    """Whitened, unit-norm geometric features ready for a kernel statistic."""
    d = descriptor(structures)
    return std(d) if len(d) else d


def geom_kid(gen: np.ndarray, ref: np.ndarray, **kw) -> float:
    """Unbiased polynomial-kernel MMD^2 on geometric features.

    Deliberately `distances.kid` itself rather than a reimplementation: the
    structural and render metrics must be the same estimator applied to
    different features, so that a difference between them is a fact about what
    the features see and never about how the distance was computed.
    """
    from blockgen.eval.bench import distances as D
    return D.kid(gen, ref, **kw)


def pattern_jsd(gen: Sequence[Structure], real: Sequence[Structure]) -> float:
    """Jensen-Shannon divergence between pooled local-pattern distributions.

    A readable companion to `geom_kid`: bounded in [0, 1], interpretable without
    a reference to the estimator, and computed on the corpus-level histogram
    rather than per build.
    """
    def pooled(xs: Sequence[Structure]) -> np.ndarray:
        h = np.zeros(N_ORBITS)
        for s in xs:
            h += pattern_histogram(s.crop_to_non_air().occupied_mask)
        return h

    return stats.jsd(pooled(gen), pooled(real))
