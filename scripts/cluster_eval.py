"""Score arms against one style cluster's held-out builds (scripts/cluster_tiers.py).

    python scripts/cluster_eval.py --name cottage --arms base:<npz> cot:<npz> ... [--n 256] [--boot 200]

The reference is the cluster's houses_32 val and test builds
(``outputs/bench_arms/clusters/<name>_ref_32.npz``), so the numbers are on a
cluster scale and comparable only inside one table. Per arm, at the first ``n``
non-empty builds: ``geom_kid`` (v2 descriptor whitened on the reference) and
``mv_dino_kid`` (four-view DINOv2, view mean), both the benchmark's KID
(``distances.kid``, x1000) with a 95% interval from resampling the arm; and the
style hit rate, the share of builds the cluster's k-means places in the cluster
within its acceptance radius (``cluster_assign.fit_view``). Two real controls
are added: the cluster's own train houses (the in-style real anchor) and
houses_32 test builds of every style (what a perfect all-houses model would
score). Writes ``outputs/run_<stamp>_cluster_eval_<name>/{scores.json,summary.md}``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/bench_arms/clusters")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label:npz")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)

    import cluster_assign as ca
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.eval.bench import distances as D
    from blockgen.eval.bench import features as F
    from blockgen.eval.bench import geometry as geom
    from blockgen.utils.runs import new_run_dir

    meta = json.loads((OUT / f"{a.name}.json").read_text())
    ref, _ = load_structures_from_cache(meta["reference"])
    structs, houses, embed = ca.house_space()
    key = meta["view"]
    view, km, thr = ca.fit_view(meta["clusters"], key, houses)
    labels = json.loads((Path(meta["clusters"]) / "clusters.json").read_text())[key]["labels"]
    rng = np.random.default_rng(0)
    in_train = [structs["train"][i] for i, c in enumerate(labels["train"]) if c == meta["cluster"]]
    test_all = list(structs["test"])
    arms = {"real cluster train": [in_train[i] for i in rng.permutation(len(in_train))[:a.n]],
            "real houses test (all styles)": [test_all[i] for i in rng.permutation(len(test_all))[:a.n]]}
    for spec in a.arms:
        label, path = spec.split(":", 1)
        ss = [s for s in load_structures_from_cache(path)[0] if (s.block_ids != s.air_block_id).any()]
        arms[label] = ss[:a.n]

    std = geom.Standardizer.fit(geom.descriptor(ref))
    g_ref = geom.geom_features(ref, std)
    d_ref = F.pooled(F.embed_views(ref, verbose=False))
    out = new_run_dir(f"cluster_eval_{a.name}")
    scores = {"name": a.name, "meta": {k: meta[k] for k in ("clusters", "assign", "view", "cluster", "reference")},
              "n_ref": len(ref), "arms": {}}
    lines = [f"# {a.name}: arms against the cluster's {len(ref)} held-out builds", "",
             "| arm | n | geom_kid | mv_dino_kid | style hit rate |", "|---|--:|--:|--:|--:|"]
    for label, ss in arms.items():
        g_raw = geom.descriptor(ss)
        g = std(g_raw)
        views = F.embed_views(ss, verbose=False)
        d = F.pooled(views)
        e = embed(g_raw, views.mean(axis=1))[view]
        dist = np.linalg.norm(e[:, None] - km.cluster_centers_[None], axis=2)
        lab = dist.argmin(1)
        hit = float(((lab == meta["cluster"]) & (dist[np.arange(len(lab)), lab] <= thr[lab])).mean())
        row = {"n": len(ss), "style_hit": round(hit, 3)}
        for metric, x, r in (("geom_kid", g, g_ref), ("mv_dino_kid", d, d_ref)):
            point = D.kid(x, r)
            bs = [D.kid(x[idx], r) for idx in (rng.integers(0, len(x), len(x)) for _ in range(a.boot))]
            row[metric] = [round(point, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        scores["arms"][label] = row
        lines.append(f"| {label} | {row['n']} | {row['geom_kid'][0]} [{row['geom_kid'][1]}, {row['geom_kid'][2]}] | "
                     f"{row['mv_dino_kid'][0]} [{row['mv_dino_kid'][1]}, {row['mv_dino_kid'][2]}] | {row['style_hit']} |")
        print(lines[-1], flush=True)
    (out / "scores.json").write_text(json.dumps(scores, indent=1) + "\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[cluster_eval] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
