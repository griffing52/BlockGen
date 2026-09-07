"""Non-learned generators: the floor a benchmark needs, and one attack on it.

A leaderboard with only trained models on it cannot be read. "MV-DINO-KID 0.188"
means nothing until you know what a hand-written box scores, and a competition
without a trivial floor rewards whoever tunes hardest against a scale nobody has
calibrated. Everything here is procedural, seeded, and fitted **only to the train
split** -- a baseline that saw val or test would be quietly competing with an
advantage no submitted model has.

The five arms, in intended quality order:

    uniform_random     random voxels at real density. The floor.
    shell_box          a hollow box with a floor and a flat roof.
    gabled_house       shell box + pitched roof + door + windows.
    patchwork          8^3 patches from *different* real builds, tiled.
    train_copy_noise   a real training build with p of its blocks displaced.

The ordering is a *prediction*, and the benchmark earns its keep by reproducing
it. `scripts/bench_baselines.py` checks exactly that.

Why `patchwork` is here
-----------------------
It is an adversarial probe, not a baseline anyone would ship. Every 8x8x8 patch
is lifted verbatim from a real house, so the *local* statistics -- the 2x2x2
pattern histogram that does most of the work in `geom_kid` -- are close to
perfect, while the global structure is nonsense: walls that stop mid-air, four
roofs, no interior. A local-pattern metric is exactly the kind of metric that
should be fooled by this, and reporting how far it is fooled is the honest way
to state the descriptor's limitation. Compare its `geom_kid` against its
`mv_dino_kid`: the render tier sees a global silhouette and should not be fooled,
which makes the pair a second demonstration that the two tiers are complementary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

#: Legacy `(id, data)` pairs used when a generator needs a material and has no
#: reason to prefer one. Overridden by `Palette` wherever a real one is fitted.
FALLBACK = [(4, 0), (5, 0), (17, 0), (98, 0), (20, 0)]


@dataclass
class Palette:
    """Material choices and size distribution, fitted on the train split only.

    A procedural baseline that guessed at these would be penalised for the wrong
    reason -- the benchmark would be measuring "did the author pick oak" rather
    than "is this a building". Fitting them removes palette and scale from the
    comparison so what remains is structure.
    """
    materials: List[Tuple[int, int]]
    weights: np.ndarray
    dims: np.ndarray                       # [N, 3] real bounding boxes
    density: np.ndarray                    # [N] occupied / bbox volume

    @classmethod
    def fit(cls, train: Sequence[Structure], max_materials: int = 24) -> "Palette":
        counts: Dict[Tuple[int, int], int] = {}
        dims, dens = [], []
        for s in train:
            c = s.crop_to_non_air()
            occ = c.occupied_mask
            n = int(occ.sum())
            if n == 0:
                continue
            dims.append(c.shape)
            dens.append(n / float(np.prod(c.shape)))
            ids, data = c.block_ids[occ], c.block_data[occ]
            for key, k in zip(*np.unique(np.stack([ids, data], 1), axis=0,
                                         return_counts=True)):
                counts[(int(key[0]), int(key[1]))] = \
                    counts.get((int(key[0]), int(key[1])), 0) + int(k)
        if not counts:
            return cls(FALLBACK, np.ones(len(FALLBACK)) / len(FALLBACK),
                       np.array([[12, 10, 12]]), np.array([0.3]))
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:max_materials]
        mats = [k for k, _ in top]
        w = np.array([v for _, v in top], dtype=float)
        return cls(mats, w / w.sum(), np.array(dims, dtype=int),
                   np.array(dens, dtype=float))

    def sample_dims(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.dims[rng.integers(len(self.dims))])

    def pick(self, rng: np.random.Generator, n: int = 1) -> List[Tuple[int, int]]:
        idx = rng.choice(len(self.materials), size=n, p=self.weights)
        return [self.materials[i] for i in np.atleast_1d(idx)]


def _to_structure(ids: np.ndarray, data: np.ndarray) -> Structure:
    return Structure(block_ids=ids.astype(np.int32), block_data=data.astype(np.int32))


def _paint(occ: np.ndarray, mat: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.where(occ, mat[0], 0).astype(np.int32)
    data = np.where(occ, mat[1], 0).astype(np.int32)
    return ids, data


# --- the arms ---------------------------------------------------------------
def uniform_random(pal: Palette, rng: np.random.Generator) -> Structure:
    """Random voxels in a real-sized box at a real density. The floor.

    Matched on size, density and palette so that everything it loses by is
    structure. Any metric that cannot separate this from real is not a metric.
    """
    X, Y, Z = pal.sample_dims(rng)
    p = float(pal.density[rng.integers(len(pal.density))])
    occ = rng.random((X, Y, Z)) < p
    ids = np.zeros((X, Y, Z), np.int32)
    data = np.zeros((X, Y, Z), np.int32)
    n = int(occ.sum())
    if n:
        mats = pal.pick(rng, n)
        ids[occ] = [m[0] for m in mats]
        data[occ] = [m[1] for m in mats]
    return _to_structure(ids, data)


def shell_box(pal: Palette, rng: np.random.Generator) -> Structure:
    """A hollow rectangular box: floor, four walls, flat roof.

    The simplest thing anyone would call a building. It has a real interior, real
    planar walls and a real thickness of 1, so it should sit far above
    `uniform_random` on every structural measure while remaining obviously not a
    house.
    """
    X, Y, Z = pal.sample_dims(rng)
    X, Y, Z = max(X, 4), max(Y, 4), max(Z, 4)
    occ = np.zeros((X, Y, Z), bool)
    occ[:, 0, :] = True                      # floor
    occ[:, -1, :] = True                     # roof
    occ[0, :, :] = occ[-1, :, :] = True      # walls
    occ[:, :, 0] = occ[:, :, -1] = True
    wall, floor = pal.pick(rng, 2)
    ids, data = _paint(occ, wall)
    ids[:, 0, :] = np.where(occ[:, 0, :], floor[0], 0)
    data[:, 0, :] = np.where(occ[:, 0, :], floor[1], 0)
    return _to_structure(ids, data)


def gabled_house(pal: Palette, rng: np.random.Generator) -> Structure:
    """Shell box, pitched roof, a door and windows. A plausible bad house.

    The most important baseline in the set: it is what a competent afternoon of
    hand-written rules produces. A benchmark on which a learned model cannot beat
    this is not yet measuring anything a learned model is good for, so this arm
    is the bar the leaderboard actually has to clear.
    """
    X, Y, Z = pal.sample_dims(rng)
    X, Z = max(X, 6), max(Z, 6)
    Y = max(Y, 6)
    wall_h = max(3, int(round(Y * 0.6)))
    occ = np.zeros((X, Y, Z), bool)
    occ[:, 0, :] = True
    occ[0, :wall_h, :] = occ[-1, :wall_h, :] = True
    occ[:, :wall_h, 0] = occ[:, :wall_h, -1] = True

    # Pitched roof: each course steps in from both z edges as it rises, and the
    # two gable ends are filled in. Without the gable ends the roof is a pair of
    # free-standing slopes, the interior vents straight out of the triangles,
    # and `interior_ratio_open` reads 0.000 -- which is what the first version
    # of this baseline did, and is a good illustration of why the aperture-aware
    # interior measure is worth having.
    courses = max(1, min(Z // 2, Y - wall_h))
    for k in range(courses):
        y = wall_h + k
        occ[:, y, k] = True
        occ[:, y, Z - 1 - k] = True
        occ[0, y, k:Z - k] = True                     # gable end
        occ[-1, y, k:Z - k] = True                    # gable end
        if k == courses - 1:                          # close the ridge
            occ[:, y, k:Z - k] = True

    ids, data = _paint(occ, pal.pick(rng)[0])
    roof = pal.pick(rng)[0]
    ids[:, wall_h:, :] = np.where(occ[:, wall_h:, :], roof[0], 0)
    data[:, wall_h:, :] = np.where(occ[:, wall_h:, :], roof[1], 0)

    # A door and a row of windows -- so the interior is aperture-reachable
    # rather than sealed, which is what real houses look like (T25d).
    dz = Z // 2
    ids[0, 1:min(3, wall_h), dz] = 0
    data[0, 1:min(3, wall_h), dz] = 0
    wy = max(1, wall_h // 2)
    for z in range(2, Z - 2, 3):
        ids[0, wy, z] = ids[-1, wy, z] = 0
        data[0, wy, z] = data[-1, wy, z] = 0
    return _to_structure(ids, data)


def patchwork(train: Sequence[Structure], rng: np.random.Generator,
              patch: int = 8, pal: Palette | None = None) -> Structure:
    """Tile 8^3 patches lifted from *different* real builds. Adversarial.

    Locally real, globally incoherent. See the module docstring: this exists to
    measure how far a local-pattern metric can be fooled, not because anyone
    would generate this way. Patches are taken from distinct source builds so no
    two adjacent cells share a build's global layout.
    """
    pal = pal or Palette.fit(train)
    X, Y, Z = pal.sample_dims(rng)
    X, Y, Z = max(X, patch), max(Y, patch), max(Z, patch)
    ids = np.zeros((X, Y, Z), np.int32)
    data = np.zeros((X, Y, Z), np.int32)
    crops = [s.crop_to_non_air() for s in train]

    for ox in range(0, X, patch):
        for oy in range(0, Y, patch):
            for oz in range(0, Z, patch):
                src = crops[rng.integers(len(crops))]
                sx, sy, sz = src.shape
                if min(sx, sy, sz) < 2:
                    continue
                ax = int(rng.integers(0, max(1, sx - patch + 1)))
                ay = int(rng.integers(0, max(1, sy - patch + 1)))
                az = int(rng.integers(0, max(1, sz - patch + 1)))
                sub_i = src.block_ids[ax:ax + patch, ay:ay + patch, az:az + patch]
                sub_d = src.block_data[ax:ax + patch, ay:ay + patch, az:az + patch]
                ex, ey, ez = (min(ox + sub_i.shape[0], X), min(oy + sub_i.shape[1], Y),
                              min(oz + sub_i.shape[2], Z))
                ids[ox:ex, oy:ey, oz:ez] = sub_i[:ex - ox, :ey - oy, :ez - oz]
                data[ox:ex, oy:ey, oz:ez] = sub_d[:ex - ox, :ey - oy, :ez - oz]
    return _to_structure(ids, data)


def train_copy_noise(train: Sequence[Structure], rng: np.random.Generator,
                     p: float = 0.10) -> Structure:
    """A real training build with a fraction `p` of its blocks displaced.

    Interpolates between `train_verbatim` (p=0, disqualified for memorization)
    and something genuinely novel. Useful for checking that the memorization gate
    releases as p grows rather than firing on anything that resembles training
    data at all.
    """
    from blockgen.eval.bench import probes

    src = train[rng.integers(len(train))]
    return probes.occupancy_noise(src, p, rng)


#: name -> builder, in a *prior* quality order (worst first). This is a
#: prediction the benchmark is asked to reproduce, not a measurement -- and the
#: places where it disagrees are the interesting ones, `patchwork` above all,
#: whose position is genuinely uncertain and metric-dependent by design.
ORDER = ("uniform_random", "shell_box", "patchwork", "gabled_house",
         "train_copy_noise")

#: One line per baseline: what it is and what it is *for*, in a sentence, for a
#: reader looking at a leaderboard row rather than at this file. Baselines are
#: built in-process, so they have no npz and no manifest to link to -- this is
#: the whole of their provenance, and it ships as `meta.provenance.recipe`.
#: The module docstring's table names them; these sentences say why each one is
#: in the set, which is the part a row on a page cannot infer.
#:
#: Keyed by `ORDER`, and a name added to one belongs in the other.
RECIPES: Dict[str, str] = {
    "uniform_random":
        "random voxels in a real-sized box at a real density, matched to the "
        "train split on size, density and palette so that everything it loses "
        "by is structure; a metric that cannot separate this from real is not a "
        "metric",
    "shell_box":
        "a hollow rectangular box -- floor, four walls, flat roof -- the "
        "simplest thing anyone would call a building, with a real interior and "
        "real planar walls, so it should sit far above uniform_random on every "
        "structural measure while remaining obviously not a house",
    "patchwork":
        "8^3 patches lifted verbatim from *different* real builds and tiled: "
        "locally real, globally nonsense. An adversarial probe, not a baseline "
        "anyone would ship -- it measures how far a local-pattern metric like "
        "geom_kid can be fooled, and the render tier should not be fooled by it",
    "gabled_house":
        "shell box plus a pitched roof, a door and a row of windows -- what a "
        "competent afternoon of hand-written rules produces, and therefore the "
        "bar a learned model actually has to clear before the leaderboard is "
        "measuring anything a learned model is good for",
    "train_copy_noise":
        "a real training build with 10% of its blocks displaced; it interpolates "
        "between train_verbatim (disqualified for memorization) and something "
        "genuinely novel, so it shows whether the memorization gate releases as "
        "the copy degrades rather than firing on any resemblance at all",
}


def build(name: str, n: int, train: Sequence[Structure], seed: int = 0,
          **kw) -> List[Structure]:
    """Generate `n` structures from one named baseline."""
    rng = np.random.default_rng(seed)
    pal = Palette.fit(train)
    makers: Dict[str, Callable[[], Structure]] = {
        "uniform_random": lambda: uniform_random(pal, rng),
        "shell_box": lambda: shell_box(pal, rng),
        "gabled_house": lambda: gabled_house(pal, rng),
        "patchwork": lambda: patchwork(train, rng, pal=pal, **kw),
        "train_copy_noise": lambda: train_copy_noise(train, rng, **kw),
    }
    if name not in makers:
        raise ValueError(f"unknown baseline {name!r}; have {sorted(makers)}")
    return [makers[name]() for _ in range(n)]


def arms(train: Sequence[Structure], n: int, seed: int = 0) -> Dict[str, List[Structure]]:
    """Every baseline, as `{name: structures}`, ready to wrap in `ArmSpec`s."""
    return {name: build(name, n, train, seed=seed + i)
            for i, name in enumerate(ORDER)}
