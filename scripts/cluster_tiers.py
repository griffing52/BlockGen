"""Write the SFT and eval tiers of one style cluster (scripts/cluster_houses.py + cluster_assign.py).

    python scripts/cluster_tiers.py --clusters <cluster_houses run> --assign <cluster_assign run>
                                    --view both_k6 --cluster 1 --name cottage [--tokenizer 60684456e598]

``sft-<name>``: the cluster's houses_32 train builds plus every pool build the
assign run accepted into it. ``eval-<name>``: the cluster's houses_32 val and
test builds (the pinned split's holdout, never in a pretraining tier). Both are
derived tiers written by ``shards.from_structures`` with its default tags
(``src:curated q:4 win:whole``), so a model samples the style under the tags it
always samples with. ``<name>_ref_32.npz`` (the eval builds as a structure cache,
the reference ``scripts/cluster_eval.py`` scores against) and ``<name>.json``
(the ids and counts) go to ``outputs/bench_arms/clusters/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blockgen.corpus.shards import from_structures, resolve_tokenizer
from blockgen.curation.houses import save_house_cache
from blockgen.eval.bench import splits

OUT = Path("outputs/bench_arms/clusters")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--view", default="both_k6")
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="60684456e598")
    a = ap.parse_args(argv)
    import cluster_assign as ca

    labels = json.loads((Path(a.clusters) / "clusters.json").read_text())[a.view]["labels"]
    assigned = json.loads((Path(a.assign) / "assign.json").read_text())
    pool_ids = assigned["views"][a.view][str(a.cluster)]["ids"]
    split = splits.load_split("houses_32", 0)
    houses = {s: splits.split_structures(split, s) for s in ("train", "val", "test")}
    pick = {s: [houses[s][i] for i, c in enumerate(labels[s]) if c == a.cluster] for s in houses}

    rows = ca.pool_table(assigned["tier"], assigned["max_dim"])
    pool = ca.load_structs(rows.loc[pool_ids, "path"].tolist(), {})
    tok = resolve_tokenizer(a.tokenizer)
    sft = from_structures(pick["train"] + pool, tier=f"sft-{a.name}", tok=tok,
                          source=f"{a.clusters}:{a.view}:{a.cluster} + {a.assign}")
    ev = from_structures(pick["val"] + pick["test"], tier=f"eval-{a.name}", tok=tok,
                         source=f"{a.clusters}:{a.view}:{a.cluster} val+test")
    OUT.mkdir(parents=True, exist_ok=True)
    ref = save_house_cache(pick["val"] + pick["test"], 32, cache_dir=str(OUT), name=f"{a.name}_ref")
    (OUT / f"{a.name}.json").write_text(json.dumps({
        "clusters": a.clusters, "assign": a.assign, "view": a.view, "cluster": a.cluster,
        "n_train_houses": len(pick["train"]), "n_pool": len(pool), "n_eval": len(pick["val"]) + len(pick["test"]),
        "pool_ids": pool_ids, "sft_shard": sft, "eval_shard": ev, "reference": str(ref)}, indent=1) + "\n")
    print(f"[tiers] sft-{a.name}: {len(pick['train'])} houses + {len(pool)} pool -> {sft}")
    print(f"[tiers] eval-{a.name}: {len(pick['val']) + len(pick['test'])} -> {ev}; reference {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
