"""Near-twin gate audit: its false-positive rate on real held-out draws and on the 40
generators of the rank-flip study, its sensitivity to the IoU threshold, and the
leaderboard with and without it.

Twin rate = share of builds whose nearest other build in the same set reaches occupancy
IoU >= t (best of 8 yaw/mirror variants and every translation; `twins.within_set_twins`,
exact for IoU >= 0.8, so rates at 0.9 and 0.95 come from the same pass).
Gate (composite.GATES): fires when rate > real_control + 3 * max(spread, 0.02); on the
leaderboard of record the bar is 0.0078 + 0.06 = 0.068.

Real draws: random subsets of the held-out pool (houses_32 test + val, 798 builds) at
n = 128 and 256, 12 draws each (test-only and val-only included).

    .venv/bin/python scripts/twin_gate_audit.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blockgen.eval.bench import features as ftr  # noqa: E402
from blockgen.eval.bench import scorecard as sc  # noqa: E402
from blockgen.eval.bench import splits  # noqa: E402
from blockgen.eval.bench import twins as tw  # noqa: E402
from blockgen.utils.runs import new_run_dir  # noqa: E402

RF = Path("outputs/run_20261106_193513_render_blind_rank_flips")
LB = Path("outputs/run_20261103_090315_bench_n_256_leaderboard_of_record")
RESCORE = Path("outputs/run_20261104_214146_bench_twin_gate_rescore")
TH = (0.8, 0.9, 0.95)
BAR = 0.0078125 + 3 * 0.02
EXTRA = ("dreamcubed", "gabled_house", "shell_box", "vanilla_village_houses",
         "vanilla_village_all")


def twins_lowmem(structs, threshold=0.8):
    """`twins.within_set_twins` with a per-pair FFT grid instead of one global grid.

    Same statistic (best IoU over the 8 yaw/mirror variants and every translation, size
    bound pruning, exact repeats keyed); the global grid of the library is sized by the
    largest build in the set, which for the LLM arms (crops up to 253 wide) needs tens of GB.
    """
    from scipy import fft as sfft
    occs = [tw._occ(s) for s in structs]
    n = len(occs)
    sizes = np.array([int(o.sum()) for o in occs])
    nn = np.zeros(n)
    keys = [min(v.tobytes() + bytes(str(v.shape), "ascii") for v in tw.d4(o)) if sz else None
            for o, sz in zip(occs, sizes)]
    groups = {}
    for i, k in enumerate(keys):
        if k is not None:
            groups.setdefault(k, []).append(i)
    for m in groups.values():
        if len(m) > 1:
            nn[m] = 1.0
    for i in range(n):
        if not sizes[i]:
            continue
        lo, hi = np.minimum(sizes, sizes[i]), np.maximum(sizes, sizes[i])
        bound = np.where(hi > 0, lo / np.maximum(hi, 1), 0.0)
        cand = [j for j in range(i + 1, n) if sizes[j] and bound[j] >= threshold
                and not (nn[i] >= 1.0 and nn[j] >= 1.0) and keys[j] != keys[i]]
        for j in cand:
            best = 0.0
            for v in tw.d4(occs[i]):
                shape = tuple(sfft.next_fast_len(a + b) for a, b in zip(v.shape, occs[j].shape))
                A = sfft.rfftn(v.astype(np.float32), s=shape)
                B = sfft.rfftn(occs[j].astype(np.float32), s=shape)
                best = max(best, float(np.round(sfft.irfftn(A * np.conj(B), s=shape).max())))
            iou = best / (sizes[i] + sizes[j] - best)
            nn[i], nn[j] = max(nn[i], iou), max(nn[j], iou)
    return nn


def rates(structs):
    big = max(max(tw._occ(s).shape) for s in structs)
    if big > 64:
        nnv = twins_lowmem(structs)
    else:
        nnv = tw.within_set_twins(structs, threshold=0.8, workers=8).nn_iou
    return {str(t): float((nnv >= t).mean()) for t in TH}, float(np.median(nnv))


def load_npz(npz, sha=None):
    ss = [s.crop_to_non_air() for s in sc.ArmSpec("x", npz=npz).load()]
    ss = [s for s in ss if s.occupied_mask.any()]
    if sha and ftr.structures_sha(ss) != sha:
        raise SystemExit(f"{npz}: sha mismatch")
    return ss


def main():
    resume = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else None
    run = Path(resume) if resume else new_run_dir("twin_gate_audit")
    shutil.copy(__file__, run / Path(__file__).name)
    log = lambda s: print(s, flush=True)
    res = {"bar_n256": BAR, "thresholds": TH, "real_draws": [], "generators": {}, "extra": {}}
    if resume:
        res = json.loads((run / "results.json").read_text())

    split = splits.load_split("houses_32", 0)
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    val = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]
    pool = test + val
    rng = np.random.default_rng(11)
    for n in ((128, 256) if not res["real_draws"] else ()):
        for k in range(12):
            src = "test" if k < 3 else ("val" if k < 6 else "test+val")
            P = test if src == "test" else (val if src == "val" else pool)
            ix = rng.choice(len(P), n, replace=False)
            r, med = rates([P[i] for i in ix])
            res["real_draws"].append({"n": n, "pool": src, "rates": r, "nn_median": med})
            log(f"real n={n} {src}: {r}")

    arms = json.loads((RF / "arms.json").read_text())
    rfres = json.loads((RF / "results.json").read_text())
    gens = rfres["sets"]["generators"]["arms"]
    lbcard = json.loads((LB / "scorecard.json").read_text())

    def npz_of(name):
        if name in arms:
            return arms[name]["npz"], arms[name].get("sha")
        meta = lbcard["arms"][name]["meta"]
        return (meta.get("npz") or (meta.get("provenance") or {}).get("npz"),
                meta.get("structures_sha"))

    for name in list(gens) + [e for e in EXTRA]:
        if name in res["generators"] or name in res["extra"]:
            continue
        t0 = time.time()
        try:
            npz, sha = npz_of(name)
            ss = load_npz(npz, sha)
        except Exception as e:  # baselines are rebuilt in the rescore run; read its twins.json
            ss = None
            log(f"{name}: no npz ({e}); taking the rescore's twins.json value")
        if ss is None:
            continue
        r, med = rates(ss)
        rec = {"n": len(ss), "rates": r, "nn_median": med, "family":
               arms.get(name, {}).get("family", "leaderboard"),
               "gated": r["0.8"] > BAR}
        (res["generators"] if name in gens else res["extra"])[name] = rec
        log(f"{name:55s} n={len(ss)} {r} {time.time() - t0:.0f}s")
        (run / "results.json").write_text(json.dumps(res, indent=2) + "\n")

    tj = json.loads((RESCORE / "twins.json").read_text())
    for a in ("gabled_house", "shell_box", "dreamcubed"):
        if a not in res["extra"]:
            res["extra"][a] = {"rates": {"0.8": tj["per_arm"][a]["twin_rate"]},
                               "source": "rescore twins.json"}
    (run / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    log(f"-> {run}")


if __name__ == "__main__":
    main()
