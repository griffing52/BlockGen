"""Structural coherence: connectivity, floating mass, enclosed interior.

**The reporting rule this module enforces.** A coherence number is meaningless
without the real-data value beside it. Only ~43% of *real* curated houses are a
single 6-connected component at canon-16, so an arm scoring `lcc_ratio = 1.0` is
as far from the data as one scoring 0.2 -- it is just wrong in the other
direction. Consequently `coherence_report` emits, for every metric, the
generated distribution, the real distribution, and the Wasserstein distance
between them, tagged `direction="distance_to_real"`. There is deliberately no
function here that returns a bare coherence rate.

Two definitions worth reading before trusting the numbers:

* **Floating**, not "ground contact". Every loader in the repo calls
  `crop_to_non_air()`, so "does it touch y=0" is trivially true for every
  structure and carries zero information. A component is *grounded* here iff its
  minimum y equals the cropped structure's minimum y; anything else is floating
  mass. That is bbox-invariant and actually discriminates.

* **Enclosed air** delegates to `curation.houses.enclosed_air_count` verbatim
  (6-connected flood fill inward from all six faces), so the interior signal in
  the benchmark is the same one the corpus was curated with -- 675 builds were
  dropped for `no_interior`. Reported as a ratio of bbox volume so 16^3 and 32^3
  builds are comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from blockgen.curation.houses import enclosed_air_count
from blockgen.eval.bench import stats
from blockgen.utils.data import Structure

# 6-connectivity, matching novelty._STRUCT6 and validity._NEIGHBORS.
STRUCT6 = ndimage.generate_binary_structure(3, 1)


def label6(occ: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label 6-connected components of a boolean occupancy volume."""
    lab, n = ndimage.label(np.asarray(occ, dtype=bool), structure=STRUCT6)
    return lab, int(n)


def component_sizes(occ: np.ndarray) -> np.ndarray:
    """Sizes of 6-connected components, descending. Empty volume -> empty array."""
    lab, n = label6(occ)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    sizes = np.bincount(lab.ravel())[1:]
    return np.sort(sizes)[::-1]


@dataclass
class Coherence:
    n_blocks: int
    dims: Tuple[int, int, int]
    bbox_fill: float                  # occupied / bbox volume
    n_components: int
    lcc_ratio: float                  # largest component / total occupied
    disconnected_block_ratio: float   # 1 - lcc_ratio
    floating_component_frac: float    # components not reaching the base plane
    floating_block_frac: float        # blocks in those components
    enclosed_air: int
    enclosed_air_ratio: float         # enclosed air / bbox volume


#: Metrics `coherence_report` compares against real data, in report order.
COHERENCE_METRICS = (
    "lcc_ratio",
    "n_components",
    "disconnected_block_ratio",
    "floating_component_frac",
    "floating_block_frac",
    "enclosed_air_ratio",
    "bbox_fill",
    "n_blocks",
)


def coherence(s: Structure) -> Coherence:
    """Per-structure coherence. Operates on the cropped structure."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    total = int(occ.sum())
    dims = tuple(int(d) for d in c.shape)
    volume = int(np.prod(dims)) if total else 0

    if total == 0:
        return Coherence(0, dims, 0.0, 0, 0.0, 1.0, 1.0, 1.0, 0, 0.0)

    lab, n = label6(occ)
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())

    # Grounded == reaches the base plane of the cropped volume (y is axis 1).
    base_labels = set(np.unique(lab[:, 0, :])) - {0}
    grounded = np.array([(i + 1) in base_labels for i in range(n)], dtype=bool)
    floating_blocks = int(sizes[~grounded].sum())

    ea = enclosed_air_count(c.block_ids)

    return Coherence(
        n_blocks=total,
        dims=dims,
        bbox_fill=total / volume,
        n_components=n,
        lcc_ratio=largest / total,
        disconnected_block_ratio=1.0 - largest / total,
        floating_component_frac=float((~grounded).sum()) / n,
        floating_block_frac=floating_blocks / total,
        enclosed_air=ea,
        enclosed_air_ratio=ea / volume,
    )


def coherence_table(structures: Sequence[Structure]) -> Dict[str, np.ndarray]:
    """Per-metric arrays over a set of structures (column-major `coherence`)."""
    rows = [asdict(coherence(s)) for s in structures]
    out: Dict[str, np.ndarray] = {}
    for key in COHERENCE_METRICS:
        out[key] = np.array([float(r[key]) for r in rows], dtype=float)
    out["max_dim"] = np.array([float(max(r["dims"])) for r in rows], dtype=float)
    return out


def coherence_report(
    generated: Sequence[Structure],
    real: Sequence[Structure],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    real_table: Dict[str, np.ndarray] | None = None,
) -> Dict[str, dict]:
    """Compare generated coherence to the real distribution, metric by metric.

    Every entry carries the generated mean, the real mean, both CIs, and the
    Wasserstein distance between the full distributions. `w1_norm` divides by the
    real IQR so metrics on different scales are readable side by side.

    Pass `real_table` to reuse the reference statistics across arms -- they must
    be identical across arms for the numbers to be comparable.
    """
    rng = rng or np.random.default_rng(0)
    gen_table = coherence_table(generated)
    real_table = real_table if real_table is not None else coherence_table(real)

    report: Dict[str, dict] = {}
    for key in list(COHERENCE_METRICS) + ["max_dim"]:
        g, r = gen_table[key], real_table[key]
        gp, glo, ghi = stats.bootstrap_ci(g, n_boot=n_boot, alpha=alpha, rng=rng)
        rp, rlo, rhi = stats.bootstrap_ci(r, n_boot=n_boot, alpha=alpha, rng=rng)
        d = stats.w1(g, r)
        spread = stats.robust_spread(r)
        report[key] = {
            "gen": {"mean": gp, "ci": [glo, ghi]},
            "real": {"mean": rp, "ci": [rlo, rhi]},
            "w1": d,
            "w1_norm": (d / spread) if np.isfinite(spread) else float("nan"),
            "real_spread": spread,
            "direction": "distance_to_real",
        }
    return report


def n_components_fast(s: Structure) -> int:
    """scipy-backed equivalent of `blockgen.eval.validity.n_components`.

    Pinned to the pure-python DFS by `tests/eval/test_topology.py`. Kept separate
    rather than replacing `validity.py`, which stays byte-identical so every
    previously published number remains reproducible.
    """
    return label6(s.crop_to_non_air().occupied_mask)[1]
