"""What a contiguous test-set draw does to the noise floor.

Our harness once took the first n test builds as the "fresh real draw" that
calibrates the noise floor, instead of a shuffled sample. Corpora are stored in
source and family order, so a contiguous block is not an exchangeable sample of
the corpus: it over-represents one source and one family, and the resulting
floor is both inflated and off-centre.

The magnitude was once quoted in this project as "roughly 45%" from a run that
no longer exists. This measures it from the cached features of the pinned split,
so the number in the paper is recomputable.

Run: `.venv/bin/python scripts/figs/contiguous_draw_check.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATS = ("data/minecraft/features/houses_32/%s"
         "/v4_az45-135-225-315_el30-30-30-30_px224_ortho/dinov2b.npz")
N, REPS, SEED = 128, 48, 991
OUT = ROOT / "outputs" / "figstage" / "contiguous_draw_check.json"


def main() -> int:
    from blockgen.eval.bench import features as ft, distances as D

    def load(split: str) -> np.ndarray:
        z = np.load(ROOT / (FEATS % split))
        a = z[list(z.files)[0]]
        return ft.pooled(a) if a.ndim == 3 else a

    test, ref = load("test"), load("val")

    rng = np.random.default_rng(SEED)
    shuffled = np.array([D.kid(test[rng.permutation(len(test))[:N]], ref)
                         for _ in range(REPS)])
    rng = np.random.default_rng(SEED)
    contiguous = np.array([
        D.kid(test[(np.arange(N) + rng.integers(len(test))) % len(test)], ref)
        for _ in range(REPS)])

    out = {
        "meta": {"n": N, "reps": REPS, "seed": SEED, "metric": "mv_dino_kid",
                 "n_test": int(len(test)), "n_ref": int(len(ref))},
        "shuffled":   {"sd": float(shuffled.std()), "mean": float(shuffled.mean())},
        "contiguous": {"sd": float(contiguous.std()), "mean": float(contiguous.mean())},
        "sd_inflation": float(contiguous.std() / shuffled.std()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"shuffled   sd={out['shuffled']['sd']:.6f} mean={out['shuffled']['mean']:+.6f}")
    print(f"contiguous sd={out['contiguous']['sd']:.6f} mean={out['contiguous']['mean']:+.6f}")
    print(f"inflation  {out['sd_inflation']:.2f}x")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
