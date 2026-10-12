"""Record what the aperture-aware interior counter reads on real houses.

Appendix~\\ref{app:interior} quotes four numbers about real data: how many
held-out houses have no *sealed* interior, how many of those nonetheless have a
room reachable through a doorway, the pooled share of interior volume that sits
behind an aperture, and the per-build sealed share. All four decide how much the
sealed-only counter understates real builds, so they belong in an artifact a
claim can be registered against rather than in a docstring.

The draw is `rng(0).permutation(test)[:200]`, the same 200 builds the probe
statistics in \\S4.1 are measured on, so the two sets of numbers are comparable
by construction: a build is altered by `solidify` if and only if it has at least
one sealed air cell, so `altered + no_sealed` is exactly the sample size.

    .venv/bin/python scripts/figs/interior_counts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "figstage" / "interior_counts.json"
N_DRAW = 200
SEED = 0
#: The OMR audit reports its median damage over models with at least this many
#: sealed cells (the `lego.py` gate), so the house median is recorded on the
#: same basis as well as unconditionally.
GATE_CELLS = 8


def measure(structures) -> dict:
    from blockgen.eval.bench import probes as pb, geometry as geo

    altered = no_sealed = room = 0
    sealed_tot = aperture_tot = 0
    deltas, shares, gated = [], [], []
    for s in structures:
        occ = s.crop_to_non_air().occupied_mask
        n0 = int(occ.sum())
        n1 = int(pb.solidify(s).occupied_mask.sum())
        frac = (n1 - n0) / max(n0, 1)
        deltas.append(frac)
        if n1 != n0:
            altered += 1
        sealed, aperture = geo.interior_volumes(occ)
        if sealed == 0:
            no_sealed += 1
            if aperture > 0:
                room += 1
        if sealed >= GATE_CELLS:
            gated.append(frac)
        sealed_tot += int(sealed)
        aperture_tot += int(aperture)
        total = sealed + aperture
        shares.append(1.0 if total == 0 else sealed / total)

    d, sh = np.array(deltas), np.array(shares)
    n = len(structures)
    interior = sealed_tot + aperture_tot
    return {
        "n": n,
        "altered": altered,
        "altered_frac": altered / n,
        "no_sealed": no_sealed,
        "no_sealed_frac": no_sealed / n,
        "no_sealed_with_aperture_room": room,
        "block_delta_median": float(np.median(d)),
        "block_delta_mean": float(d.mean()),
        "block_delta_median_gated": float(np.median(gated)) if gated else None,
        "n_gated": len(gated),
        "sealed_share_per_build_median": float(np.median(sh)),
        "sealed_share_per_build_mean": float(sh.mean()),
        "aperture_share_pooled": aperture_tot / interior,
        "sealed_share_pooled": sealed_tot / interior,
    }


def main() -> int:
    from blockgen.eval.bench import splits

    sp = splits.load_split("houses_32", SEED)
    test = splits.split_structures(sp, "test")
    draw = [test[i] for i in np.random.default_rng(SEED).permutation(len(test))[:N_DRAW]]

    rows = {"draw_200": measure(draw), "full_test": measure(test)}
    for name, r in rows.items():
        print(f"[{name}] n={r['n']}")
        print(f"  altered {r['altered']}/{r['n']} = {r['altered_frac']:.2%}"
              f"   no sealed {r['no_sealed']}/{r['n']} = {r['no_sealed_frac']:.2%}"
              f"   (sum {r['altered'] + r['no_sealed']})")
        print(f"  of those, have an aperture room: "
              f"{r['no_sealed_with_aperture_room']}/{r['no_sealed']}")
        print(f"  block delta: median {r['block_delta_median']:.2%}"
              f"  mean {r['block_delta_mean']:.2%}"
              f"  median over the {r['n_gated']} gated builds "
              f"{r['block_delta_median_gated']:.2%}")
        print(f"  sealed share of interior: per-build median "
              f"{r['sealed_share_per_build_median']:.2%}"
              f"  mean {r['sealed_share_per_build_mean']:.2%}"
              f"  pooled {r['sealed_share_pooled']:.2%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "gate_cells": GATE_CELLS,
                               "rows": rows}, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
