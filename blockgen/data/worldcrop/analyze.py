"""Summarise a world-crop run and render sample sheets.

    python -m blockgen.data.worldcrop.analyze data/minecraft/worldcrop/201407 [--n 64]

Writes ``<run>/analysis.json``, ``<run>/sheet_masked.png`` (component only) and
``<run>/sheet_context.png`` (full bbox), sampling crops uniformly at random.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

import numpy as np


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def summarize(run_dir: str) -> dict:
    worlds = load_jsonl(os.path.join(run_dir, "worlds.jsonl"))
    crops = load_jsonl(os.path.join(run_dir, "crops.jsonl"))
    status = collections.Counter(w["status"] for w in worlds)
    sub = [sw for w in worlds for sw in w.get("worlds", []) if "scan" in sw]
    routes = collections.Counter(sw["scan"].get("route") for sw in sub)
    flat_rej = sum(1 for sw in sub if sw["scan"].get("flat_rejected"))
    modded = sum(1 for sw in sub if sw["level"].get("modded"))
    chunks = sum(sw["scan"]["chunks"] for sw in sub)
    secs = sum(w.get("seconds", 0) for w in worlds)
    per_world = [sw["n_crops"] for sw in sub]
    dims = np.array([c["shape"] for c in crops]) if crops else np.zeros((0, 3))
    maxdim = dims.max(axis=1) if len(dims) else np.zeros(0)
    seed_ids = collections.Counter()
    for c in crops:
        for k, v in c.get("top_ids", {}).items():
            seed_ids[int(k)] += v
    try:
        from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
        name = lambda i: STANDARD_VOCAB.get(str(i), str(i)).split("(")[0].strip()
    except Exception:
        name = str
    out = {
        "worlds": len(worlds), "status": dict(status), "world_dirs": len(sub), "routes": dict(routes),
        "flat_rejected": flat_rej, "modded": modded, "chunks": chunks, "seconds": round(secs, 1),
        "chunks_per_sec_aggregate": round(chunks / secs, 1) if secs else None,
        "crops": len(crops), "crops_saved": sum(1 for c in crops if c.get("file")),
        "oversize": sum(1 for c in crops if c.get("oversize")),
        "worlds_with_crops": sum(1 for n in per_world if n > 0),
        "crops_per_world_pct": {p: float(np.percentile(per_world, p)) for p in (50, 75, 90, 99)} if per_world else {},
        "maxdim_pct": {p: float(np.percentile(maxdim, p)) for p in (10, 50, 90, 99)} if len(maxdim) else {},
        "n_component_pct": {p: float(np.percentile([c["n_component"] for c in crops], p)) for p in (10, 50, 90)} if crops else {},
        "fits_32": int((maxdim <= 32).sum()), "fits_48": int((maxdim <= 48).sum()), "fits_64": int((maxdim <= 64).sum()),
        "fits_128": int((maxdim <= 128).sum()),
        "top_block_ids": [(name(i), n) for i, n in seed_ids.most_common(25)],
        "sites": dict(collections.Counter((w.get("entry") or {}).get("site") for w in worlds)),
        "errors": collections.Counter((w.get("error") or "")[:60] for w in worlds if w.get("error")).most_common(8),
    }
    return out


def render_sheets(run_dir: str, n: int = 64, seed: int = 0, cols: int = 8, max_dim: int = 96):
    from blockgen.renderer.grid import save_grid
    from blockgen.utils.data import Structure
    crops = [c for c in load_jsonl(os.path.join(run_dir, "crops.jsonl")) if c.get("file") and max(c["shape"]) <= max_dim]
    random.Random(seed).shuffle(crops)
    sel = crops[:n]
    masked, context = [], []
    for c in sel:
        z = np.load(os.path.join(run_dir, "crops", c["file"]))
        ids, dat, m = z["ids"].astype(np.int32), z["data"].astype(np.int32), z["mask"]
        masked.append(Structure(block_ids=ids * m, block_data=dat * m, metadata={"corpus": "worldcrop"}))
        context.append(Structure(block_ids=ids, block_data=dat, metadata={"corpus": "worldcrop"}))
    if sel:
        save_grid(masked, os.path.join(run_dir, "sheet_masked.png"), cols=cols)
        save_grid(context, os.path.join(run_dir, "sheet_context.png"), cols=cols)
    return len(sel)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--max-dim", type=int, default=96)
    args = ap.parse_args(argv)
    s = summarize(args.run_dir)
    with open(os.path.join(args.run_dir, "analysis.json"), "w") as f:
        json.dump(s, f, indent=2, default=str)
    print(json.dumps(s, indent=1, default=str))
    if not args.no_render:
        print("rendered", render_sheets(args.run_dir, n=args.n, max_dim=args.max_dim))


if __name__ == "__main__":
    main()
