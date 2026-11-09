"""BrickGPT's stability analysis, solved with SCIP instead of Gurobi.

BrickGPT's physical-stability check (`brickgpt.stability_analysis`, the paper's
Eqn. 3) is written against gurobipy, and the pip gurobipy licence is size-limited:
it refuses every StableText2Brick structure we tried, down to 20 bricks. This
module runs the *same source file* against SCIP (pyscipopt, Apache-2.0) by
loading it with three mechanical rewrites and a tiny `gp` shim:

* `x.X` (solution value)                 -> `_VAL(x)`  (SCIP `getVal`)
* `addConstr(y == gp.abs_(x))`           -> `y >= x, y >= -x`
* `addConstr(m == gp.max_(f for ...))`   -> `m >= f` for each f

The two relaxations are exact at the optimum because `y` and `m` enter the
minimised objective with positive weight (`eq_obj` and `alpha *
sum_brick_max_f_down`) and nowhere else in a binding way, so the solver drives
them down onto |x| and max(f). The bilinear complementarity constraints
(`n_down * f_up == 0`) are kept as they are; SCIP solves them as nonconvex
quadratic constraints, like Gurobi does.

Agreement with Gurobi is *measured*, not assumed: StableText2Brick ships the
per-voxel Gurobi stability heat for every structure, and `--validate N`
compares this solver's heat with it (see notes.md).

Run in libs/BrickGPT/.venv:
    python scripts/brickgpt_scip_stability.py --validate 40
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pyscipopt

_CUR = {"model": None}
TIME_LIMIT = 300.0


def _VAL(v):
    return _CUR["model"].getVal(v)


class _Params:
    pass


class _Model:
    def __init__(self, name=""):
        self.m = pyscipopt.Model(name)
        self.m.hideOutput()
        self.m.setParam("limits/time", TIME_LIMIT)
        self.m.setParam("limits/gap", 1e-4)
        self.Params = _Params()
        self.modelSense = None
        self._obj = None
        _CUR["model"] = self.m

    def setParam(self, *_a, **_k):
        pass

    def addVar(self, lb=0.0, vtype=None, name=""):
        return self.m.addVar(lb=lb, ub=None, vtype="C")

    def addVars(self, n, lb=0.0, vtype=None, name=""):
        return {i: self.m.addVar(lb=lb, ub=None, vtype="C") for i in range(n)}

    def addConstr(self, cons):
        return self.m.addCons(cons)

    def addAbs(self, y, x):
        self.m.addCons(y >= x)
        self.m.addCons(y >= -x)

    def addMax(self, m, xs):
        for x in xs:
            self.m.addCons(m >= x)

    def setObjective(self, expr):
        self._obj = expr

    def update(self):
        pass

    def optimize(self):
        self.m.setObjective(self._obj, "minimize")
        self.m.optimize()

    @property
    def Status(self):
        return self.m.getStatus()

    @property
    def NumVars(self):
        return self.m.getNVars()

    @property
    def NumConstrs(self):
        return self.m.getNConss()

    @property
    def objVal(self):
        return self.m.getObjVal()

    def close(self):
        pass


def _quicksum(it):
    return pyscipopt.quicksum(list(it))


_GRB = types.SimpleNamespace(CONTINUOUS="C", MINIMIZE="minimize",
                             Status=types.SimpleNamespace(OPTIMAL="optimal"))
_gp = types.SimpleNamespace(Model=_Model, GRB=_GRB, quicksum=_quicksum)


def load_module():
    """BrickGPT's stability_analysis.py, rewritten to run on the shim."""
    import brickgpt.stability_analysis.stability_analysis as orig
    src = Path(orig.__file__).read_text()
    src = src.replace("import gurobipy as gp\nfrom gurobipy import GRB\n", "")
    src = src.replace("from .utils import *", "from brickgpt.stability_analysis.utils import *")
    n0 = src.count(".X")
    src = re.sub(r"([A-Za-z_][A-Za-z_0-9]*(?:\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])+)\.X\b",
                 r"_VAL(\1)", src)
    assert ".X " not in src and ".X\n" not in src, "unrewritten .X"
    src, n_abs = re.subn(r"model\.addConstr\((.+?) == gp\.abs_\((.+)\)\)\n",
                         r"model.addAbs(\1, \2)\n", src)
    src, n_max = re.subn(r"model\.addConstr\(\s*(\S+ - 1\]) == gp\.max_\((.+?) for k in "
                         r"range\(len\(brick_f_down_list\)\)\)\)\n",
                         r"model.addMax(\1, [\2 for k in range(len(brick_f_down_list))])\n",
                         src, flags=re.S)
    assert n_abs == 5 and n_max == 1, (n_abs, n_max)
    mod = types.ModuleType("brickgpt_stability_scip")
    mod.__dict__.update(gp=_gp, GRB=_GRB, _VAL=_VAL)
    sys.modules[mod.__name__] = mod      # dataclasses look the module up
    exec(compile(src, orig.__file__ + "[scip]", "exec"), mod.__dict__)
    return mod, n0


def stability_heat(bs, mod=None):
    """Per-voxel heat exactly as `BrickStructure.stability_scores()` returns it,
    or None if SCIP did not reach a proven optimum in TIME_LIMIT."""
    from brickgpt.data.brick_library import brick_library
    mod = mod or load_module()[0]
    heat, nv, nc, total_t, _ = mod.stability_score(
        bs.to_json(), brick_library,
        mod.StabilityConfig(world_dimension=(bs.world_dim,) * 3))
    status = _CUR["model"].getStatus()
    return (heat if status == "optimal" else None), status, total_t


def per_brick_scores(bs, heat):
    return np.array([1.0 - float(np.max(heat[b.slice])) for b in bs.bricks])


ROOT = Path(__file__).resolve().parents[1] / "data/lego/stabletext2brick/data"
_TABLE = {}


def _validate_one(i: int) -> dict:
    import time
    import pyarrow.parquet as pq
    from brickgpt.data import BrickStructure
    if "t" not in _TABLE:
        _TABLE["t"] = pq.read_table(ROOT / "test-00000-of-00001.parquet",
                                    columns=["structure_id", "bricks", "stability_scores"])
        _TABLE["mod"] = load_module()[0]
    r = _TABLE["t"].slice(int(i), 1).to_pylist()[0]
    bs = BrickStructure.from_txt(r["bricks"])
    t0 = time.time()
    heat, status, _ = stability_heat(bs, _TABLE["mod"])
    dt = time.time() - t0
    # The dataset stores the per-voxel SCORE s (1 = stable), i.e. 1 - heat.
    sref = per_brick_scores(bs, 1.0 - np.asarray(r["stability_scores"], dtype=float))
    row = {"row": int(i), "n_bricks": len(bs), "status": status, "seconds": dt,
           "ref_stable": bool(np.all(sref > 0)), "ref_mean": float(sref.mean()),
           "ref_min": float(sref.min())}
    if heat is not None:
        s = per_brick_scores(bs, heat)
        row.update(stable=bool(np.all(s > 0) and not bs.has_floating_bricks()),
                   mean=float(s.mean()), min=float(s.min()),
                   max_abs_score_diff=float(np.abs(s - sref).max()))
    return row


def main() -> int:
    import argparse
    import json
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    n_rows = pq.ParquetFile(ROOT / "test-00000-of-00001.parquet").metadata.num_rows
    idx = np.random.default_rng(0).permutation(n_rows)[:args.validate]
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_validate_one, [int(i) for i in idx]):
            rows.append(row)
            print(json.dumps(row), flush=True)
    solved = [r for r in rows if "stable" in r]
    summ = {"n": len(rows), "solved": len(solved),
            "ref_stable_frac": float(np.mean([r["ref_stable"] for r in rows])),
            "scip_stable_frac_of_solved": float(np.mean([r["stable"] for r in solved])) if solved else None,
            "stable_agree": float(np.mean([r["stable"] == r["ref_stable"] for r in solved])) if solved else None,
            "median_max_abs_score_diff": float(np.median([r["max_abs_score_diff"] for r in solved])) if solved else None,
            "mean_abs_mean_score_diff": float(np.mean([abs(r["mean"] - r["ref_mean"]) for r in solved])) if solved else None,
            "mean_abs_min_score_diff": float(np.mean([abs(r["min"] - r["ref_min"]) for r in solved])) if solved else None,
            "median_seconds": float(np.median([r["seconds"] for r in rows]))}
    print(json.dumps(summ, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
