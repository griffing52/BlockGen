"""Select Objaverse buildings that actually have interiors, for the mesh probe.

Pipeline: keyword-filter the Objaverse-1.0 annotations to building-like names,
download candidates in batches, flatten each GLB to one normalised coloured mesh
(`mesh_probe.load_glb`), ray-cast visibility and the hidden volume, and keep an
object iff a meaningful part of it is invisible from every exterior viewpoint:

    hidden_area_frac >= --min-hidden-area   (share of face area no ray reaches)
    hidden_vol_frac  >= --min-hidden-vol    (hidden empty cells / bbox cells)

Accepted objects are cached as npz (mesh + visibility + hidden mask) under
data/external/objaverse_interior/meshes/; every GLB is deleted after it is
processed, accepted or not, so disk use stays at the cache size. Every candidate
seen is logged to manifest.jsonl with its licence and the statistics, so the
acceptance rule is auditable.

    .venv/bin/python scripts/fetch_objaverse_interior.py --target 256

Paper scale (notes §65): `--strict-target 512` switches the stop rule to the
stricter multi-room rule (`mesh_probe.STRICT_RULE`: hidden area >= 30%, hidden
volume >= 5%, >= 2 sealed spaces, >= 3% of the area on hidden faces at least
two cells from the shell). Candidates are then the interior-worded tier
(`PRIORITY2` over name, tags and description) ordered by GLB size ascending and
capped at `--max-mb`, because small files return several times more accepted
objects per downloaded GB (pilot: 124/GB under 1 MB, 12/GB at 10-20 MB). The
rule's statistics for every cached object live in interior_stats.jsonl.

    .venv/bin/python scripts/fetch_objaverse_interior.py --strict-target 512 --max-gb 8
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("data/external/objaverse_interior")
BUILDING = re.compile(r"\b(house|houses|building|cabin|cottage|villa|mansion|apartment|"
                      r"castle|church|barn|tavern|inn|farmhouse|bungalow|chalet|cathedral|"
                      r"warehouse|shop|store|home|residence|hotel|school|office|hut|"
                      r"shack|lodge|bakery|pub|cafe|restaurant|library|station|temple|"
                      r"chapel|manor|mill|cabana|dollhouse|interior|rooms?|floor ?plan)\b", re.I)
EXCLUDE = re.compile(r"\b(table|chair|sofa|couch|bed|lamp|dresser|pillow|vase|cup|plant|"
                     r"shelf|cabinet|desk|stool|bench|clock|door|window|key|car|truck|"
                     r"character|girl|boy|man|woman|scan|photogrammetry|3d ?scan|tree|"
                     r"rock|sword|gun|weapon|bottle|book|kit|pack|asset ?pack|props?|"
                     r"modular|tile|texture|material|wall|floor|ceiling|stairs?)\b", re.I)
PRIORITY = re.compile(r"interior|furnished|rooms|bedroom|floor ?plan|inside|"
                      r"two[- ]stor|multi|apartment", re.I)
#: Paper-scale tier: words that name an interior or a multi-room layout.
PRIORITY2 = re.compile(r"interior|furnished|rooms|bedroom|living ?room|kitchen|bathroom|"
                       r"floor ?plan|inside|two[- ]stor|multi[- ]?stor|apartment|flat\b|loft|"
                       r"condo|dorm|classroom|office|lobby|hallway|corridor|archviz|arch ?viz|"
                       r"real ?estate|first floor|ground floor|second floor|storey|cutaway|"
                       r"dollhouse", re.I)
BUILDING2 = re.compile(BUILDING.pattern[:-3] + r"|bedroom|kitchen|bathroom|living ?room|loft|"
                       r"condo|dorm|classroom|lobby|hallway|corridor|archviz|studio)\b", re.I)


def candidates(seed: int = 0):
    ann = json.load(gzip.open(ROOT / "annotations_slim.json.gz", "rt"))
    hi, lo = [], []
    for uid, a in ann.items():
        name = a.get("name") or ""
        f = a.get("faces") or 0
        if not (2_000 <= f <= 300_000) or (a.get("anim") or 0) > 0:
            continue
        if not BUILDING.search(name) or EXCLUDE.search(name):
            continue
        text = " ".join([name] + a["tags"] + [a.get("desc") or ""])
        (hi if PRIORITY.search(text) else lo).append(uid)
    rng = random.Random(seed)
    rng.shuffle(hi)
    rng.shuffle(lo)
    return hi + lo, ann, len(hi)


def candidates_strict(max_mb: float, seed: int = 0):
    """Interior-worded candidates by GLB size ascending (sizes from the HF tree,
    cached in glb_sizes.json.gz), then the rest of the building tier under 1.5 MB."""
    ann = json.load(gzip.open(ROOT / "annotations_slim.json.gz", "rt"))
    sizes = json.load(gzip.open(ROOT / "glb_sizes.json.gz", "rt"))
    hi, lo = [], []
    for uid, a in ann.items():
        name = a.get("name") or ""
        f = a.get("faces") or 0
        if not (2_000 <= f <= 300_000) or (a.get("anim") or 0) > 0 or uid not in sizes:
            continue
        text = " ".join([name] + a["tags"] + [a.get("desc") or ""])
        building = BUILDING2.search(name) or (BUILDING.search(" ".join(a["tags"]))
                                              and PRIORITY2.search(text))
        if not building or EXCLUDE.search(name):
            continue
        mb = sizes[uid] / 1e6
        if PRIORITY2.search(text) and mb <= max_mb:
            hi.append((mb, uid))
        elif mb <= 1.5:
            lo.append((mb, uid))
    rng = random.Random(seed)
    rng.shuffle(hi)
    rng.shuffle(lo)
    hi.sort(key=lambda t: round(t[0], 1))
    lo.sort(key=lambda t: round(t[0], 1))
    return [u for _, u in hi] + [u for _, u in lo], ann, len(hi)


def _stats_job(uid):
    from blockgen.eval.bench import mesh_probe as mp
    m, ex = mp.load_npz(str(ROOT / "meshes" / f"{uid}.npz"), uid)
    st = mp.interior_stats(m, ex["visible"], ex["hidden"])
    return {"uid": uid, **st, "strict": mp.passes_strict(st)}


def backfill_stats(pool) -> dict:
    """interior_stats.jsonl for every cached object that lacks it."""
    path = ROOT / "interior_stats.jsonl"
    have = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            have[r["uid"]] = r
    todo = [p.stem for p in (ROOT / "meshes").glob("*.npz") if p.stem not in have]
    if todo:
        with path.open("a") as fh:
            for r in pool.map(_stats_job, todo):
                have[r["uid"]] = r
                fh.write(json.dumps(r) + "\n")
    return have


def process(args):
    uid, path, min_area, min_vol = args
    import numpy as np
    from blockgen.eval.bench import mesh_probe as mp
    t0 = time.time()
    rec = {"uid": uid}
    try:
        m = mp.load_glb(path, uid)
        if m is None:
            rec["skip"] = "load"
            return rec, None
        vis = mp.visible_faces(m)
        area = m.face_areas
        rec["faces"] = int(len(m.faces))
        rec["hidden_area_frac"] = float(area[~vis].sum() / max(area.sum(), 1e-12))
        rec["hidden_face_frac"] = float((~vis).mean())
        hid, g = mp.hidden_cells(m)
        rec["hidden_vol_frac"] = float(hid.mean())
        rec["grid"] = list(g.shape)
        rec["accepted"] = bool(rec["hidden_area_frac"] >= min_area
                               and rec["hidden_vol_frac"] >= min_vol)
        rec["sec"] = round(time.time() - t0, 1)
        if rec["accepted"]:
            out = ROOT / "meshes" / f"{uid}.npz"
            mp.save_npz(str(out), m, visible=vis, hidden=hid)
            st = mp.interior_stats(m, vis, hid)
            rec["interior"] = st
            rec["strict"] = mp.passes_strict(st)
        return rec, None
    except Exception as exc:          # a broken GLB is a skip, never a crash
        rec["skip"] = f"{type(exc).__name__}: {exc}"[:200]
        return rec, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-hidden-area", type=float, default=0.25)
    ap.add_argument("--min-hidden-vol", type=float, default=0.03)
    ap.add_argument("--max-gb", type=float, default=8.0, help="cumulative download cap")
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--strict-target", type=int, default=None,
                    help="stop at this many objects passing mesh_probe.STRICT_RULE")
    ap.add_argument("--min-free-gb", type=float, default=15.0,
                    help="stop before a batch if free disk space is below this")
    ap.add_argument("--max-mb", type=float, default=8.0,
                    help="GLB size cap for the strict candidate tier")
    args = ap.parse_args()

    import objaverse
    objaverse.BASE_PATH = str(ROOT)
    objaverse._VERSIONED_PATH = str(ROOT / "hf-objaverse-v1")
    (ROOT / "meshes").mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "manifest.jsonl"
    seen = set()
    accepted = 0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            seen.add(r["uid"])
            accepted += bool(r.get("accepted"))
    strict = args.strict_target is not None
    if strict:
        cands, ann, n_hi = candidates_strict(args.max_mb)
        cands = [u for u in cands if u not in seen]
    else:
        cands, ann, n_hi = candidates()
        cands = [u for u in cands[:args.max_candidates] if u not in seen]
    print(f"[fetch] {len(cands)} candidates ({n_hi} priority), {accepted} already accepted",
          flush=True)
    downloaded = 0.0
    glb_dir = ROOT / "hf-objaverse-v1" / "glbs"
    with ProcessPoolExecutor(args.workers) as pool:
        n_strict = 0
        if strict:
            n_strict = sum(r["strict"] for r in backfill_stats(pool).values())
            print(f"[fetch] {n_strict} cached objects pass the strict rule", flush=True)
        stats_fh = (ROOT / "interior_stats.jsonl").open("a") if strict else None
        for b in range(0, len(cands), args.batch):
            if (n_strict >= args.strict_target if strict else accepted >= args.target) \
                    or downloaded / 1e9 >= args.max_gb:
                break
            free = shutil.disk_usage(str(ROOT)).free / 1e9
            if free < args.min_free_gb + 0.5:
                print(f"[fetch] stop: {free:.1f} GB free < {args.min_free_gb} GB floor", flush=True)
                break
            batch = cands[b:b + args.batch]
            paths = objaverse.load_objects(batch, download_processes=8)
            sizes = {u: os.path.getsize(p) for u, p in paths.items() if os.path.exists(p)}
            downloaded += sum(sizes.values())
            jobs = [(u, p, args.min_hidden_area, args.min_hidden_vol) for u, p in paths.items()]
            with manifest.open("a") as fh:
                for rec, _ in pool.map(process, jobs):
                    a = ann[rec["uid"]]
                    rec.update(name=a["name"], license=a["license"], user=a["user"],
                               uri=a["uri"], glb_bytes=sizes.get(rec["uid"]))
                    accepted += bool(rec.get("accepted"))
                    fh.write(json.dumps(rec) + "\n")
                    if stats_fh is not None and "interior" in rec:
                        n_strict += bool(rec["strict"])
                        stats_fh.write(json.dumps({"uid": rec["uid"], **rec["interior"],
                                                   "strict": rec["strict"]}) + "\n")
                        stats_fh.flush()
            shutil.rmtree(glb_dir, ignore_errors=True)
            print(f"[fetch] batch {b // args.batch}: accepted {accepted}, strict {n_strict}, "
                  f"downloaded {downloaded / 1e9:.2f} GB", flush=True)
    print(f"[fetch] done: {accepted} accepted, {n_strict} strict", flush=True)


if __name__ == "__main__":
    main()
