"""geom_kid of the facade arms against two references: the real val houses (the card's) and
the same houses reduced to their six-face shell (results T122 addendum). Prints a table.
Usage: PYTHONPATH=. python scripts/facade_shell_reference.py"""
import glob, json, sys, time
import numpy as np
from blockgen.eval.bench import geometry as geom, splits, scorecard as sc
from blockgen.corpus import facade as fc
from blockgen.curation.houses import load_structures_from_cache

ctx = sc.BenchContext(corpus="houses_32", seed=0, grid=24, min_n=16, n_boot=1000, n_ref=512, text_backbone="clipL")
split = splits.load_split("houses_32", 0)
val = splits.split_structures(split, "val")
rng = ctx.rng()
if len(val) > 512:
    val = [val[i] for i in rng.permutation(len(val))[:512]]
t0 = time.time()
d_real = geom.descriptor(val)
shells = [fc.shell(s, fc.ALL_FACES) for s in val]
d_shell = geom.descriptor(shells)
print(f"reference descriptors in {time.time()-t0:.0f}s", flush=True)
refs = {}
for name, d in [("real", d_real), ("shell", d_shell)]:
    std = geom.Standardizer.fit(d)
    refs[name] = (std, std(d))

R = "outputs/run_20261028_125240_houses32_facades_joint_d4_20m"
FR = "outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07"
def npz_in(d):
    return [p for p in glob.glob(d + "/*.npz") if "manifest" not in p and "faces" not in p][0]
arms = [
    ("real_test (full)", None),
    ("real@facades6 (ceiling)", "outputs/bench_arms/real_facades6_32.npz"),
    ("facade_patchwork", "outputs/bench_arms/facade_patchwork_32.npz"),
    ("grammar T0.7", npz_in(f"{R}/constrained_none_t0.7_n399")),
    ("constrained T0.7", npz_in(f"{R}/constrained_constrain_t0.7_n399")),
    ("best-of-4 T0.7", npz_in(f"{R}/t0.7/bon_face_solidity_k4_n399")),
    ("constrained best-of-4", npz_in(f"{R}/constrained_bon_face_solidity_k4_t0.7_n399")),
    ("RFT T0.7", npz_in(f"{FR}/samples399")),
    ("constrained RFT", npz_in(f"{FR}/constrained_constrain_t0.7_n399")),
    ("constrained best-of-4 on RFT", npz_in(f"{FR}/constrained_bon_face_solidity_k4_t0.7_n399")),
]
test = splits.split_structures(split, "test")
rows = []
for name, path in arms:
    xs = test if path is None else load_structures_from_cache(path)[0]
    xs = [s for s in xs if s.occupied_mask.any()]
    t0 = time.time()
    d = geom.descriptor(xs)
    out = {}
    for rn, (std, gref) in refs.items():
        out[rn] = geom.geom_kid(std(d), gref)
    rows.append((name, len(xs), out["real"], out["shell"]))
    print(f"{name:32s} n={len(xs):3d}  vs real {out['real']:6.2f}  vs shell {out['shell']:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
print()
print("| arm | n | geom_kid vs real houses | geom_kid vs their six-face shells |")
print("|---|---|---|---|")
for name, n, a, b in rows:
    print(f"| {name} | {n} | {a:.2f} | {b:.2f} |")
