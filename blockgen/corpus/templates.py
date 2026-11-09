"""The game's own structure templates as a corpus source.

Since 1.14 the client jar ships its jigsaw structure pieces as NBT templates under
``data/minecraft/structure/`` (1.21.1: 1,180 files; village 483, trial chambers 170,
bastion 167, trail ruins 84, woodland mansion rooms 73, ancient city 58, ocean
ruins 48, shipwreck 20, end city 20, ruined portals 13, pillager outpost 11, igloo
3, fossils). They are complete, labelled, known-good pieces at every scale from a
lamp post to a 46-block mansion room. The code-generated structures (strongholds,
monuments, fortresses, temples, mineshafts) are not templates and need a world.

Each template becomes ``data/minecraft/templates/<family>/<name>.npz`` with legacy
``ids``/``data`` in XYZ (modern block states mapped through
``blockgen.utils.block_remap``, jigsaw blocks replaced by their ``final_state``,
structure blocks and voids by air), and a manifest table ``templates.parquet`` with
ids ``tpl:<family>/<name>`` and ``source == "templates"``. Mojang assets: an
internal training source only, never part of a data release or a benchmark
reference (docs/pipeline.md §3.3).

CLI::

    python -m blockgen.corpus.templates [--jar PATH] [--workers 6]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from .manifest import ROOT, _clf, _write, describe

TEMPLATE_DIR = os.path.join(ROOT, "data", "minecraft", "templates")
JAR_GLOB = os.path.expanduser("~/.gradle/caches/fabric-loom/*/minecraft-client.jar")
PREFIX = "data/minecraft/structure/"

# states block_remap does not know; (id, data) in legacy numbering
_OVERRIDE: Dict[str, Tuple[int, int]] = {
    "minecraft:structure_block": (0, 0), "minecraft:structure_void": (0, 0),
    "minecraft:redstone_wall_torch": (76, 0), "minecraft:redstone_lamp": (123, 0),
    "minecraft:trial_spawner": (52, 0), "minecraft:vault": (52, 0),
    "minecraft:decorated_pot": (140, 0), "minecraft:soul_fire": (51, 0),
    "minecraft:sea_pickle": (0, 0), "minecraft:powder_snow": (80, 0),
}


def find_jar() -> Optional[str]:
    jars = sorted(glob.glob(JAR_GLOB))
    return jars[-1] if jars else None


def _state_key(name: str, props: Dict[str, str]) -> str:
    return name + ("[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]" if props else "")


def _legacy(name: str, props: Dict[str, str], unknown: collections.Counter) -> Tuple[int, int]:
    from blockgen.utils.block_remap import remap_name
    if name in _OVERRIDE:
        return _OVERRIDE[name]
    if name.startswith("minecraft:potted_"):
        return (140, 0)
    r = remap_name(_state_key(name, props))
    if r is None:
        unknown[name] += 1
        return (1, 0)
    return int(r[0]), int(r[1])


def _parse_state(s: str) -> Tuple[str, Dict[str, str]]:
    """``minecraft:oak_fence[east=true,west=false]`` -> (name, props)."""
    name, _, rest = s.partition("[")
    props = {}
    if rest:
        for kv in rest.rstrip("]").split(","):
            k, _, v = kv.partition("=")
            props[k.strip()] = v.strip()
    return name, props


def convert(nbt_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, collections.Counter]:
    """Template NBT -> (ids, data) XYZ uint16/uint8 plus the unknown-state counter."""
    import nbtlib
    raw = gzip.decompress(nbt_bytes) if nbt_bytes[:2] == b"\x1f\x8b" else nbt_bytes
    t = nbtlib.File.parse(io.BytesIO(raw))
    if "size" not in t and len(t) == 1:          # nbtlib wraps the root compound by name
        t = t[list(t.keys())[0]]
    sx, sy, sz = (int(v) for v in t["size"])
    palette = t["palette"] if "palette" in t else t["palettes"][0]
    unknown: collections.Counter = collections.Counter()
    pal = []
    for p in palette:
        name = str(p["Name"])
        props = {str(k): str(v) for k, v in p["Properties"].items()} if "Properties" in p else {}
        pal.append((name, props))
    lut_id = np.zeros(len(pal), np.uint16); lut_dat = np.zeros(len(pal), np.uint8)
    for i, (name, props) in enumerate(pal):
        if name == "minecraft:jigsaw":
            continue
        bid, bdat = _legacy(name, props, unknown)
        lut_id[i], lut_dat[i] = bid, bdat
    ids = np.zeros((sx, sy, sz), np.uint16); data = np.zeros((sx, sy, sz), np.uint8)
    for b in t["blocks"]:
        x, y, z = (int(v) for v in b["pos"])
        s = int(b["state"])
        if pal[s][0] == "minecraft:jigsaw":
            final = str(b["nbt"]["final_state"]) if "nbt" in b and "final_state" in b["nbt"] else "minecraft:air"
            bid, bdat = _legacy(*_parse_state(final), unknown) if final != "minecraft:air" else (0, 0)
        else:
            bid, bdat = int(lut_id[s]), int(lut_dat[s])
        ids[x, y, z] = bid; data[x, y, z] = bdat
    return ids, data, unknown


def extract(jar: str) -> List[dict]:
    """Write every template as npz; return index records."""
    recs = []
    unknown: collections.Counter = collections.Counter()
    with zipfile.ZipFile(jar) as z:
        names = [n for n in z.namelist() if n.startswith(PREFIX) and n.endswith(".nbt")]
        for n in sorted(names):
            rel = n[len(PREFIX):-4]
            family = rel.split("/", 1)[0]
            ids, data, unk = convert(z.read(n))
            unknown.update(unk)
            out = os.path.join(TEMPLATE_DIR, rel + ".npz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, ids=ids, data=data)
            recs.append({"name": rel, "family": family, "path": os.path.relpath(out, ROOT),
                         "size": list(ids.shape), "n_blocks": int(np.count_nonzero(ids))})
    with open(os.path.join(TEMPLATE_DIR, "templates.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    if unknown:
        print("unmapped states (-> stone):", dict(unknown.most_common(20)), flush=True)
    return recs


def _worker(recs: List[dict]):
    rows, feats = [], []
    for r in recs:
        z = np.load(os.path.join(ROOT, r["path"]))
        row, f = describe(z["ids"].astype(np.uint16), None)
        row.update({"id": f"tpl:{r['name']}", "source": "templates", "path": r["path"], "month": "",
                    "world_id": f"tpl:{r['family']}", "title": r["name"].rsplit("/", 1)[-1], "author": "Mojang",
                    "url": "", "p_built": np.nan, "cut": False, "oversize": False, "route": r["family"]})
        rows.append(row); feats.append(f)
    return rows, np.stack(feats).astype(np.float16)


def build_manifest(recs: List[dict], workers: int) -> int:
    recs = [r for r in recs if r["n_blocks"] > 0]
    jobs = [recs[i:i + 50] for i in range(0, len(recs), 50)]
    rows, feats = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, F in ex.map(_worker, jobs):
            rows.extend(r); feats.append(F)
    F = np.concatenate(feats)
    for row, p in zip(rows, _clf().predict_proba(F.astype(np.float32))[:, 1]):
        row["p_built"] = float(p)
    _write("templates", rows, F)
    return len(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    jar = a.jar or find_jar()
    if not jar:
        raise SystemExit(f"no client jar found under {JAR_GLOB}; pass --jar")
    t0 = time.time()
    recs = extract(jar)
    fam = collections.Counter(r["family"] for r in recs)
    print(f"{len(recs)} templates from {jar} in {time.time() - t0:.0f}s: {dict(fam.most_common())}", flush=True)
    n = build_manifest(recs, a.workers)
    print(f"manifest templates.parquet: {n} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
