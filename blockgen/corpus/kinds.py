"""Kind of every manifest row: human build, or generated content the detector let through.

The world-crop detector masks vanilla structures from the ``data/*.dat`` bounding
boxes, but 58% of worlds carry no such files and pieces leak past the boxes in the
rest. Two generators are caught here, each by the signature its output cannot hide:

* ``village``   recurrence: the cluster (dedup.parquet) occurs in >= 5 distinct worlds.
* ``mineshaft`` composition: >= 90% of member voxels are planks, fences, rails,
  cobwebs, torches or chests, and at least one rail or cobweb is present
  (mineshafts are randomised, so recurrence misses them; hand-checked at 70/72).
* ``mixed``     50-90% mineshaft material with a rail or cobweb: a player build that
  grew into a shaft, or a shaft a player dug into. Kept and tagged, never dropped.
* ``build``     everything else.

Output ``data/minecraft/corpus/kinds.parquet`` with ``id, kind, mine_share``; tiers
merge it like dedup (``tiers._frame``) so configs can filter on ``kind``.

CLI::

    python -m blockgen.corpus.kinds
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np

from .manifest import CORPUS_DIR, MANIFEST_DIR

OUT = os.path.join(CORPUS_DIR, "kinds.parquet")
DEDUP = os.path.join(CORPUS_DIR, "dedup.parquet")

# legacy ids a mineshaft is made of: planks, fence, rail, powered/detector rail,
# cobweb, torch, chest, monster egg (silverfish stone, stronghold leftovers)
MINE_IDS = (5, 85, 66, 27, 28, 30, 50, 54, 97)
TELL_IDS = (66, 30)              # rail, cobweb: the generator's signature
MINE_SHARE = 0.9
MIXED_SHARE = 0.5
VILLAGE_WORLDS = 5


def classify(mine_share: np.ndarray, has_tell: np.ndarray, cluster_worlds: np.ndarray) -> np.ndarray:
    """Kind per row from the three signals (see module docstring)."""
    kind = np.full(len(mine_share), "build", dtype=object)
    kind[(mine_share >= MIXED_SHARE) & has_tell] = "mixed"
    kind[(mine_share >= MINE_SHARE) & has_tell] = "mineshaft"
    kind[(cluster_worlds >= VILLAGE_WORLDS) & (kind != "mineshaft")] = "village"
    return kind


def run() -> None:
    import pandas as pd
    import pyarrow.parquet as pq
    t0 = time.time()
    ids, share, tell = [], [], []
    for p in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet"))):
        if p.endswith("_sample.parquet"):
            continue
        t = pq.read_table(p, columns=["id"]).to_pandas()
        f = np.load(p.replace(".parquet", "_feats.npy"), mmap_mode="r")
        ids.append(t.id.to_numpy())
        share.append(np.asarray(f[:, list(MINE_IDS)], np.float32).sum(1))
        tell.append(np.asarray(f[:, list(TELL_IDS)], np.float32).sum(1))
    df = pd.DataFrame({"id": np.concatenate(ids), "mine_share": np.concatenate(share)})
    cw = np.ones(len(df))
    if os.path.exists(DEDUP):
        d = pd.read_parquet(DEDUP, columns=["id", "cluster_worlds"])
        cw = df[["id"]].merge(d, on="id", how="left").cluster_worlds.fillna(1).to_numpy()
    df["kind"] = classify(df.mine_share.to_numpy(), np.concatenate(tell) > 0, cw)
    df["id"] = df["id"].astype(str)
    df.to_parquet(OUT, index=False)
    counts = df.kind.value_counts()
    print(f"{len(df)} rows -> " + ", ".join(f"{k} {v} ({v / len(df):.1%})" for k, v in counts.items())
          + f"; {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    run()
