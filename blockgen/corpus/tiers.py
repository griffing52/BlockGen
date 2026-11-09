"""Tiers: a named filter over the manifest, materialised to an id list with a hash.

A tier config (``configs/corpus/<name>.yaml``)::

    description: everything, deduplicated
    sources: [worldcrop, grabcraft, 3dcraft, text2mc, text2mc_schem]   # manifest `source` values
    filter: "n_blocks >= 20"            # pandas query over manifest columns, plus
                                        # dedup (is_rep, cluster, cluster_worlds) and
                                        # kinds (kind, mine_share) sidecar columns
    dedup: true                         # keep one representative per duplicate cluster
    exclude: [eval-holdout]             # drop rows whose world_id/cluster appear in these tiers
    aux: [gc_small_48]                  # also read these auxiliary manifests (manifest_aux/)

The labels sidecar (``corpus/labels.parquet``, ``blockgen.corpus.labels``) is merged
like dedup and kinds, so a filter can use ``cat`` (a class of
``labels.CLASSES``, ``none`` when unlabeled), ``cat_split`` (train / eval /
holdout / extra / dup / drop / wc_train, empty when unlabeled) and ``cat_src``.
Auxiliary manifests are read only by tiers that list them, so adding one never
changes what an existing tier resolves to.

Materialising writes ``data/minecraft/corpus/tiers/<name>.json`` (name, filter,
count, sha256 of the sorted ids) and ``<name>.ids.txt``. The sha256 is the tier
hash recorded in every checkpoint trained on it (docs/pipeline.md §3.3).

A tier pinned to a split (``holdout_split``: eval-holdout, sft-houses32) that
resolves to zero rows raises ``EmptyTierError`` instead of writing an empty tier:
either the cache the split indexes is missing from the manifest or the tier's own
filters removed every row, and a run on an empty SFT or holdout tier would train
on nothing and score nothing. The error says which, so ``tiers build`` stops there.

CLI::

    python -m blockgen.corpus.tiers build [names ...]      # all configs when no names
    python -m blockgen.corpus.tiers show <name>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import yaml

from .manifest import CORPUS_DIR, ROOT, load

CONFIG_DIR = os.path.join(ROOT, "configs", "corpus")
TIER_DIR = os.path.join(CORPUS_DIR, "tiers")
DEDUP = os.path.join(CORPUS_DIR, "dedup.parquet")
KINDS = os.path.join(CORPUS_DIR, "kinds.parquet")
LABELS = os.path.join(CORPUS_DIR, "labels.parquet")


def _configs() -> Dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))):
        out[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(open(p)) or {}
    return out


def _frame(aux: Optional[List[str]] = None, labels: bool = True):
    import pandas as pd
    df = load(aux=list(aux or []))
    if os.path.exists(DEDUP):
        d = pd.read_parquet(DEDUP)
        df = df.merge(d, on="id", how="left")
        df["is_rep"] = df["is_rep"].fillna(True).astype(bool)
        df["cluster"] = df["cluster"].fillna(-1).astype(int)
        df["cluster_worlds"] = df["cluster_worlds"].fillna(1).astype(int)
    else:
        df["is_rep"] = True
        df["cluster"] = -1
        df["cluster_worlds"] = 1
    if os.path.exists(KINDS):
        k = pd.read_parquet(KINDS)
        df = df.merge(k, on="id", how="left")
        df["kind"] = df["kind"].fillna("build")
        df["mine_share"] = df["mine_share"].fillna(0.0)
    else:
        df["kind"] = "build"
        df["mine_share"] = 0.0
    if labels and os.path.exists(LABELS):
        lab = pd.read_parquet(LABELS, columns=["id", "cat", "cat_split", "cat_src"])
        df = df.merge(lab, on="id", how="left")
        df["cat"] = df["cat"].fillna("none")
        df["cat_split"] = df["cat_split"].fillna("")
        df["cat_src"] = df["cat_src"].fillna("")
    else:
        df["cat"] = "none"; df["cat_split"] = ""; df["cat_src"] = ""
    df["max_dim"] = df[["dx", "dy", "dz"]].max(axis=1)
    return df


def _ids_hash(ids: List[str]) -> str:
    h = hashlib.sha256()
    for i in sorted(ids):
        h.update(i.encode()); h.update(b"\n")
    return h.hexdigest()


class EmptyTierError(RuntimeError):
    """A split-backed tier resolved to no manifest rows: the cache it indexes is missing from the
    manifest, or the tier's own filters removed every row of the split."""


def materialise(name: str, cfg: dict, df, built: Dict[str, dict]) -> dict:
    sub = df
    split = None
    if cfg.get("holdout_split"):
        split = json.load(open(os.path.join(ROOT, cfg["holdout_split"])))
        parts = cfg.get("holdout_parts", ["val", "test"])
        idx = [i for part in parts for i in split[part]]
        ids = {f"cache:{split['corpus']}:{i}" for i in idx}
        sub = sub[sub.id.isin(ids)]
    if cfg.get("sources"):
        sub = sub[sub.source.isin(cfg["sources"])]
    if cfg.get("filter"):
        sub = sub.query(cfg["filter"])
    if cfg.get("dedup", True):
        sub = sub[sub.is_rep]
    for ex in cfg.get("exclude", []) or []:
        other = built.get(ex) or json.load(open(os.path.join(TIER_DIR, f"{ex}.json")))
        ex_ids = set(open(os.path.join(TIER_DIR, f"{ex}.ids.txt")).read().split())
        rows = df[df.id.isin(ex_ids)]
        sub = sub[~sub.world_id.isin(set(rows.world_id)) & ~sub.cluster.isin(set(rows.cluster[rows.cluster >= 0]))]
    ids = sub.id.tolist()
    if not ids and split is not None:
        corpus = split["corpus"]
        in_manifest = int(df.id.str.startswith(f"cache:{corpus}:").sum())
        head = (f"tier {name} resolved to 0 rows: it is defined over cache:{corpus} ids from "
                f"{cfg['holdout_split']} and the manifest holds {in_manifest} rows of that cache; ")
        if in_manifest == 0:
            advice = f"run python -m blockgen.corpus.manifest curated --caches {corpus} and rebuild the tiers"
        else:
            advice = (f"the cache is present, so the tier's sources/filter/dedup/exclude settings in "
                      f"configs/corpus/{name}.yaml removed every row of the split's parts {parts}")
        raise EmptyTierError(head + advice)
    os.makedirs(TIER_DIR, exist_ok=True)
    rec = {"name": name, "description": cfg.get("description", ""), "filter": cfg.get("filter", ""),
           "sources": cfg.get("sources"), "dedup": cfg.get("dedup", True), "exclude": cfg.get("exclude", []),
           "n": len(ids), "n_blocks_total": int(sub.n_blocks.sum()), "sha256": _ids_hash(ids),
           "fits32": int((sub.max_dim <= 32).sum()), "fits48": int((sub.max_dim <= 48).sum()),
           "by_source": sub.groupby("source").size().to_dict(), "built": time.strftime("%Y-%m-%d %H:%M")}
    if cfg.get("aux"):
        rec["aux"] = list(cfg["aux"])
        rec["by_cat"] = sub.groupby("cat").size().to_dict()
    json.dump(rec, open(os.path.join(TIER_DIR, f"{name}.json"), "w"), indent=1)
    open(os.path.join(TIER_DIR, f"{name}.ids.txt"), "w").write("\n".join(ids))
    return rec


def build(names: Optional[List[str]] = None) -> None:
    cfgs = _configs()
    names = names or list(cfgs)
    # tiers that others exclude must be built first
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for ex in (cfgs[n].get("exclude") or []):
            visit(ex)
        order.append(n)

    for n in names:
        visit(n)
    frames: Dict[tuple, object] = {}
    built = {}
    for n in order:
        aux = tuple(cfgs[n].get("aux") or ())
        if aux not in frames:
            frames[aux] = _frame(list(aux))
        rec = materialise(n, cfgs[n], frames[aux], built)
        built[n] = rec
        print(f"{n}: {rec['n']} builds, {rec['n_blocks_total'] / 1e6:.1f}M blocks, "
              f"fits32 {rec['fits32']}, sha {rec['sha256'][:12]}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("names", nargs="*")
    s = sub.add_parser("show"); s.add_argument("name")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.names or None)
    else:
        print(json.dumps(json.load(open(os.path.join(TIER_DIR, f"{a.name}.json"))), indent=1))


if __name__ == "__main__":
    main()
