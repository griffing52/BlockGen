"""BrickGPT's own validity metrics (paper Table 1), computed with BrickGPT's code.

Runs in libs/BrickGPT/.venv. For every generated structure in `<run>/gen/*.json`
and for the held-out StableText2Brick structure its prompt was drawn from:

* `valid`: no out-of-library, out-of-bounds or colliding brick, non-empty;
* `connected`: every brick reaches the ground through stud connections
  (`BrickStructure.is_connected`, the repo's no-Gurobi fallback check);
* `stable`, `mean_stability`, `min_stability`: BrickGPT's force-equilibrium
  programme (paper Eqn. 3), solved with SCIP through
  `scripts/brickgpt_scip_stability.py` because the pip Gurobi licence refuses
  every structure. Stable iff every brick scores > 0 and none floats, i.e.
  `BrickStructure.is_stable()`. A solve that does not reach proven optimality
  in the time limit is `null` and reported as such, never as unstable.
  For the reference, the dataset's own Gurobi scores are recorded as well.

Stability, like the paper, is aggregated over valid structures only.
Writes `<run>/validity.jsonl` and `<run>/validity_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brickgpt_scip_stability as scip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data/lego/stabletext2brick/data"
_MOD = {}


def metrics(txt: str, use_stability: bool = True) -> dict:
    from brickgpt.data import BrickStructure
    bs = BrickStructure.from_txt(txt)
    out = {"n_bricks": len(bs)}
    try:
        lib_ok = all(b.brick_id is not None for b in bs.bricks)
    except ValueError:
        lib_ok = False
    out["valid"] = bool(lib_ok and len(bs) > 0 and not bs.has_out_of_bounds_bricks()
                        and not bs.has_collisions())
    out["stable"] = out["mean_stability"] = out["min_stability"] = None
    if not out["valid"]:
        out["connected"] = False
        return out
    out["connected"] = bool(bs.is_connected())
    if use_stability:
        if "mod" not in _MOD:
            _MOD["mod"] = scip.load_module()[0]
        try:
            heat, status, secs = scip.stability_heat(bs, _MOD["mod"])
        except Exception as exc:   # SCIP "error in LP solver" on a few inputs
            heat, status, secs = None, f"error: {exc}"[:80], None
        out["scip_status"], out["scip_seconds"] = status, secs
        if heat is not None:
            s = scip.per_brick_scores(bs, heat)
            out["stable"] = bool(np.all(s > 0) and not bs.has_floating_bricks())
            out["mean_stability"], out["min_stability"] = float(s.mean()), float(s.min())
    return out


def one(job: dict) -> dict:
    from brickgpt.data import BrickStructure
    rec = {"idx": job["idx"], "category": job["category"],
           "gen": metrics(job["gen"]), "ref": metrics(job["ref"])}
    bs = BrickStructure.from_txt(job["ref"])
    s = scip.per_brick_scores(bs, 1.0 - np.asarray(job["ref_scores"], dtype=float))
    rec["ref"].update(dataset_stable=bool(np.all(s > 0)),
                      dataset_mean_stability=float(s.mean()),
                      dataset_min_stability=float(s.min()))
    return rec


def agg(recs, side):
    rows = [r[side] for r in recs]
    valid = [x for x in rows if x["valid"]]
    st = [x for x in valid if x["stable"] is not None]
    out = {"n": len(rows), "valid": float(np.mean([x["valid"] for x in rows])),
           "connected_of_valid": float(np.mean([x["connected"] for x in valid])) if valid else None,
           "median_bricks": float(np.median([x["n_bricks"] for x in rows])),
           "stability_solved": len(st), "stability_unsolved": len(valid) - len(st)}
    if st:
        out["stable_of_solved"] = float(np.mean([x["stable"] for x in st]))
        out["mean_stability"] = float(np.mean([x["mean_stability"] for x in st]))
        out["min_stability"] = float(np.mean([x["min_stability"] for x in st]))
    if side == "ref":
        for k in ("dataset_stable", "dataset_mean_stability", "dataset_min_stability"):
            out[k] = float(np.mean([x[k] for x in rows]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    run = Path(args.run)
    ref_rows = {r["structure_id"]: r for r in pq.read_table(
        ROOT / "test-00000-of-00001.parquet",
        columns=["structure_id", "bricks", "stability_scores"]).to_pylist()}
    done = {}
    vpath = run / "validity.jsonl"
    if vpath.exists():                       # resumable
        for line in vpath.read_text().splitlines():
            r = json.loads(line)
            done[r["idx"]] = r
    jobs = []
    for f in sorted((run / "gen").glob("*.json")):
        g = json.loads(f.read_text())
        if g["idx"] in done:
            continue
        r = ref_rows[g["structure_id"]]
        jobs.append({"idx": g["idx"], "category": g["category"], "gen": g["bricks"],
                     "ref": r["bricks"], "ref_scores": r["stability_scores"]})
    with Pool(args.workers) as pool, open(vpath, "a") as fh:
        for rec in pool.imap_unordered(one, jobs):
            done[rec["idx"]] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(rec["idx"], json.dumps(rec["gen"]), flush=True)
    recs = [done[k] for k in sorted(done)]
    summary = {"gen": agg(recs, "gen"), "ref": agg(recs, "ref")}
    (run / "validity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
