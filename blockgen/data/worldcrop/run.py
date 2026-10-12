"""Orchestrate world-crop extraction over local directories or archive.org months.

    # local worlds (any dir tree with level.dat files or world archives)
    python -m blockgen.data.worldcrop.run --local /path/to/worlds --out data/minecraft/worldcrop/local

    # stream a Minecraft Archive Project month (never downloads the 10-60 GB zip)
    python -m blockgen.data.worldcrop.run --month 201407 --limit 100 --workers 8

Outputs under ``<out>/<month>/``:
  worlds.jsonl  one record per world archive (provenance, level.dat facts, scan summary, status)
  crops.jsonl   one record per saved crop
  crops/<world_id>_<k>.npz  ids (X,Y,Z) uint16, data uint8, mask bool (compressed)
Resumable: worlds already in worlds.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from . import anvil, detect, remote

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                           "data", "minecraft", "worldcrop")
SCRATCH = os.environ.get("BLOCKGEN_SCRATCH", os.path.join(tempfile.gettempdir(), "blockgen_worldcrop"))

_archive_cache: Dict[str, remote.MonthArchive] = {}


def _archive(month: str, url: str) -> remote.MonthArchive:
    a = _archive_cache.get(month)
    if a is None:
        a = remote.MonthArchive(month, url=url)
        _archive_cache[month] = a
    return a


def _which(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


_UNRAR = os.environ.get("BLOCKGEN_UNRAR") or _which("unrar")
_BSDTAR = _which("bsdtar")
_7Z = _which("7z", "7za", "7zz")


def extract_archive(path: str, dest: str) -> bool:
    """Extract zip/rar/7z. Debian's 7zip lacks the RAR codec, so .rar goes through
    unrar (conda-forge / BLOCKGEN_UNRAR), then libarchive's bsdtar, then 7z.
    Returns True if at least one region file or level.dat came out."""
    os.makedirs(dest, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    cmds = []
    if ext == ".rar":
        if _UNRAR:
            cmds.append([_UNRAR, "x", "-y", "-idq", "-o+", path, dest + os.sep])
        if _BSDTAR:
            cmds.append([_BSDTAR, "-xf", path, "-C", dest])
    if _7Z:
        cmds.append([_7Z, "x", "-y", "-bd", "-bso0", "-bsp0", f"-o{dest}", path])
    if ext != ".rar" and _BSDTAR:
        cmds.append([_BSDTAR, "-xf", path, "-C", dest])
    for cmd in cmds:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception:
            continue
        if _has_world(dest):
            return True
    return _has_world(dest)


def _has_world(dest: str) -> bool:
    for dp, _, fns in os.walk(dest):
        if "level.dat" in fns or any(f.endswith((".mca", ".mcr", ".schematic")) for f in fns):
            return True
    return False


def extract_nested(dest: str, depth: int = 1) -> int:
    """Creators often zip a zip. Extract inner archives in place (one level), return count."""
    n = 0
    for _ in range(depth):
        inner = [os.path.join(dp, f) for dp, _, fns in os.walk(dest) for f in fns
                 if f.lower().endswith(remote.WORLD_EXTS) and "__MACOSX" not in dp]
        if not inner:
            break
        for p in inner:
            extract_archive(p, p + "_x")
            os.remove(p)
            n += 1
    return n


def find_schematics(root: str):
    return sorted(os.path.join(dp, f) for dp, _, fns in os.walk(root) for f in fns
                  if f.lower().endswith(".schematic") and "__MACOSX" not in dp)


def import_schematics(paths, world_id: str, out_dir: str, max_dim: int):
    """PlanetMinecraft 'download schematic' zips hold finished builds: import as whole crops."""
    from blockgen.utils.data import Structure
    crop_dir = os.path.join(out_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    recs = []
    for k, p in enumerate(paths[:64]):
        try:
            st = Structure.from_schematic_path(p).crop_to_non_air()
        except Exception:
            continue
        ids = (st.block_ids & 0xFFF).astype(np.uint16)
        data = (st.block_data & 0xF).astype(np.uint8)
        if ids.size == 0 or (ids != 0).sum() < 8:
            continue
        shape = list(ids.shape)
        rec = {"world_id": world_id, "k": k, "route": "schematic", "shape": shape, "oversize": max(shape) > max_dim,
               "n_nonair": int((ids != 0).sum()), "n_component": int((ids != 0).sum()), "n_seed": -1, "n_weak": -1,
               "distinct_ids": int(len(np.unique(ids))) - 1, "inhabited_time_mean": -1.0, "bbox": None,
               "source_file": os.path.basename(p), "top_ids": {}, "top_seed_ids": {}}
        vals, counts = np.unique(ids[ids != 0], return_counts=True)
        order = np.argsort(-counts)[:12]
        rec["top_ids"] = {str(int(vals[i])): int(counts[i]) for i in order}
        if not rec["oversize"]:
            fn = f"{world_id}_s{k}.npz"
            np.savez_compressed(os.path.join(crop_dir, fn), ids=ids, data=data, mask=ids != 0)
            rec["file"] = fn
        else:
            rec["file"] = None
        recs.append(rec)
    return recs


def process_world_dir(world_dir: str, world_id: str, out_dir: str, cfg: detect.DetectConfig,
                      max_crops: int = 2000) -> dict:
    level, summary, crops = detect.detect_world(world_dir, cfg)
    crop_dir = os.path.join(out_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    recs = []
    n_saved = n_over = 0
    for k, c in enumerate(crops):
        rec = c.record()
        rec.update({"world_id": world_id, "k": k})
        if c.oversize:
            n_over += 1
            rec["file"] = None
        elif n_saved < max_crops:
            fn = f"{world_id}_{k}.npz"
            np.savez_compressed(os.path.join(crop_dir, fn), ids=c.ids, data=c.data, mask=c.mask)
            rec["file"] = fn
            n_saved += 1
        recs.append(rec)
    world_rec = {
        "world_id": world_id, "level": asdict(level), "scan": summary,
        "n_crops": len(recs), "n_saved": n_saved, "n_oversize": n_over,
    }
    return {"world": world_rec, "crops": recs}


def _work(job: dict) -> dict:
    """Runs in a worker process. job: {kind: local|remote, ...}"""
    t0 = time.time()
    cfg = detect.DetectConfig(**job["cfg"])
    out_dir = job["out_dir"]
    world_id = job["world_id"]
    tmp = os.path.join(SCRATCH, world_id)
    result = {"world_id": world_id, "status": "ok", "worlds": [], "crops": [], "entry": job.get("entry")}
    try:
        if job["kind"] == "remote":
            os.makedirs(tmp, exist_ok=True)
            arc = _archive(job["month"], job["url"])
            entry = remote.WorldEntry(**job["entry"])
            local = arc.fetch(entry, os.path.join(tmp, "archive" + os.path.splitext(entry.member)[1]))
            if not extract_archive(local, os.path.join(tmp, "x")):
                result["status"] = "extract_failed"
            os.remove(local)
            roots = [os.path.join(tmp, "x")]
        elif job["kind"] == "local_archive":
            os.makedirs(tmp, exist_ok=True)
            if not extract_archive(job["path"], os.path.join(tmp, "x")):
                result["status"] = "extract_failed"
            roots = [os.path.join(tmp, "x")]
        else:
            roots = [job["path"]]
        if job["kind"] != "local_dir":
            extract_nested(roots[0])
        dirs = []
        for r in roots:
            dirs += anvil.find_world_dirs(r)
        schems = [] if job["kind"] == "local_dir" else find_schematics(roots[0])
        if schems:
            recs = import_schematics(schems, world_id, out_dir, cfg.max_dim)
            result["crops"] += recs
            result["worlds"].append({"world_id": world_id, "kind": "schematics", "n_files": len(schems),
                                     "n_crops": len(recs), "n_saved": sum(1 for r in recs if r.get("file")),
                                     "n_oversize": sum(1 for r in recs if r.get("oversize"))})
        if (dirs or schems) and result["status"] == "extract_failed":
            result["status"] = "ok"  # the outer extractor saw nothing, a nested archive had the world
        if not dirs and not schems and result["status"] == "ok":
            result["status"] = "no_level_dat"
        for i, wd in enumerate(dirs[:8]):  # an archive may bundle several worlds
            wid = world_id if i == 0 else f"{world_id}_w{i}"
            if os.path.isdir(os.path.join(wd, "db")):  # Bedrock / Pocket Edition (LevelDB) world
                result["worlds"].append({"world_id": wid, "kind": "bedrock"})
                if result["status"] == "ok" and len(dirs) == 1 and not schems:
                    result["status"] = "bedrock_world"
                continue
            try:
                r = process_world_dir(wd, wid, out_dir, cfg)
                r["world"]["subdir"] = os.path.relpath(wd, roots[0]) if job["kind"] != "local_dir" else wd
                result["worlds"].append(r["world"])
                result["crops"] += r["crops"]
            except Exception as e:
                result["worlds"].append({"world_id": wid, "error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc()[-1500:]})
                result["status"] = "partial_error"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-1500:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result["seconds"] = round(time.time() - t0, 2)
    return result


def _done_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["world_id"])
                except Exception:
                    pass
    return ids


def _limit_memory(mem_gb: float):
    """Worker initializer: cap the address space so one giant world raises MemoryError
    inside _work (caught -> status 'error') instead of getting the worker OOM-killed."""
    if mem_gb and mem_gb > 0:
        lim = int(mem_gb * 2 ** 30)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ValueError, OSError):
            pass


def run_jobs(jobs: List[dict], out_dir: str, workers: int, mem_gb: float = 5.0, max_pool_breaks: int = 20) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    wpath, cpath = os.path.join(out_dir, "worlds.jsonl"), os.path.join(out_dir, "crops.jsonl")
    done = _done_ids(wpath)
    jobs = [j for j in jobs if j["world_id"] not in done]
    tot = {"jobs": len(jobs), "ok": 0, "failed": 0, "crops": 0, "saved": 0, "oversize": 0, "seconds": 0.0,
           "pool_breaks": 0}
    t0 = time.time()
    i = 0
    attempts: Dict[str, int] = {}
    pending = list(jobs)
    with open(wpath, "a") as wf, open(cpath, "a") as cf:
        while pending:
            ex = ProcessPoolExecutor(max_workers=workers, initializer=_limit_memory, initargs=(mem_gb,))
            futs = {ex.submit(_work, j): j for j in pending}
            try:
                for fut in as_completed(futs):
                    r = fut.result()
                    pending.remove(futs[fut])
                    i += 1
                    _record(r, wf, cf, tot, t0, i, len(jobs))
                ex.shutdown(wait=True)
                break
            except BrokenProcessPool:
                # A worker died (OOM killer / segfault). Nobody knows which job killed it, so re-run
                # every unfinished job; one that has sat in a broken pool twice is written off.
                tot["pool_breaks"] += 1
                ex.shutdown(wait=False, cancel_futures=True)
                finished = {futs[f]["world_id"] for f in futs
                            if f.done() and not f.cancelled() and f.exception() is None}
                survivors = []
                for j in pending:
                    if j["world_id"] in finished:
                        continue
                    attempts[j["world_id"]] = attempts.get(j["world_id"], 0) + 1
                    if attempts[j["world_id"]] >= 2 or tot["pool_breaks"] > max_pool_breaks:
                        i += 1
                        _record({"world_id": j["world_id"], "status": "worker_killed", "entry": j.get("entry"),
                                 "seconds": 0.0, "worlds": [], "crops": [],
                                 "error": "worker process terminated (likely OOM)"}, wf, cf, tot, t0, i, len(jobs))
                    else:
                        survivors.append(j)
                pending = survivors
                print(f"[pool break #{tot['pool_breaks']}] re-queued {len(pending)} unfinished jobs", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(tot, f, indent=2)
    return tot


def _record(r: dict, wf, cf, tot: dict, t0: float, i: int, n: int):
    rec = {"world_id": r["world_id"], "status": r["status"], "entry": r.get("entry"),
           "seconds": r["seconds"], "worlds": r["worlds"], "error": r.get("error"), "trace": r.get("trace")}
    wf.write(json.dumps(rec) + "\n"); wf.flush()
    for c in r["crops"]:
        cf.write(json.dumps(c) + "\n")
    cf.flush()
    tot["ok" if r["status"] == "ok" else "failed"] += 1
    tot["crops"] += len(r["crops"])
    tot["saved"] += sum(1 for c in r["crops"] if c.get("file"))
    tot["oversize"] += sum(1 for c in r["crops"] if c.get("oversize"))
    tot["seconds"] = round(time.time() - t0, 1)
    if i % 10 == 0 or i == n:
        print(f"[{i}/{n}] ok={tot['ok']} failed={tot['failed']} crops={tot['crops']} "
              f"saved={tot['saved']} oversize={tot['oversize']} {tot['seconds']}s", flush=True)


def jobs_for_month(month: str, out_dir: str, cfg: detect.DetectConfig, limit: Optional[int] = None,
                   sites: Optional[List[str]] = None, min_size: int = 20_000, max_size: int = 400_000_000) -> List[dict]:
    arc = remote.MonthArchive(month)
    entries = arc.worlds()
    entries = [e for e in entries if min_size <= e.size <= max_size and (not sites or e.site in sites)]
    if limit:
        # deterministic spread over the month rather than the first N
        step = max(1, len(entries) // limit)
        entries = entries[::step][:limit]
    cfgd = asdict(cfg)
    return [{"kind": "remote", "month": month, "url": arc.url, "entry": asdict(e), "world_id": e.world_id,
             "out_dir": out_dir, "cfg": cfgd} for e in entries]


def jobs_for_local(root: str, out_dir: str, cfg: detect.DetectConfig) -> List[dict]:
    import hashlib
    cfgd = asdict(cfg)
    jobs = []
    for wd in anvil.find_world_dirs(root):
        wid = "local_" + hashlib.sha1(wd.encode()).hexdigest()[:12]
        jobs.append({"kind": "local_dir", "path": wd, "world_id": wid, "out_dir": out_dir, "cfg": cfgd,
                     "entry": {"member": wd, "title": os.path.basename(wd)}})
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(remote.WORLD_EXTS):
                p = os.path.join(dp, fn)
                wid = "local_" + hashlib.sha1(p.encode()).hexdigest()[:12]
                jobs.append({"kind": "local_archive", "path": p, "world_id": wid, "out_dir": out_dir,
                             "cfg": cfgd, "entry": {"member": p, "title": fn}})
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", help="archive.org Worlds_YYYYMM item to stream")
    ap.add_argument("--local", help="local directory of worlds / archives")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dimension", default="overworld")
    ap.add_argument("--min-seed", type=int, default=12)
    ap.add_argument("--tile-chunks", type=int, default=24)
    ap.add_argument("--max-dim", type=int, default=320)
    ap.add_argument("--mem-gb", type=float, default=5.0,
                    help="address-space cap per worker (0 = none); oversize worlds fail cleanly instead of OOM-killing")
    args = ap.parse_args(argv)
    cfg = detect.DetectConfig(dimension=args.dimension, min_seed=args.min_seed, tile_chunks=args.tile_chunks,
                              max_dim=args.max_dim)
    os.makedirs(SCRATCH, exist_ok=True)
    if args.month:
        out_dir = os.path.join(args.out, args.month)
        jobs = jobs_for_month(args.month, out_dir, cfg, limit=args.limit, sites=args.sites)
    elif args.local:
        out_dir = os.path.join(args.out, "local")
        jobs = jobs_for_local(args.local, out_dir, cfg)
    else:
        ap.error("--month or --local required")
    print(f"{len(jobs)} jobs -> {out_dir} (workers={args.workers})", flush=True)
    tot = run_jobs(jobs, out_dir, args.workers, mem_gb=args.mem_gb)
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
