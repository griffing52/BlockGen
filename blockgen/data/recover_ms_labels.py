"""Recover minecraft-schematics.com labels for the local `data/minecraft/raw` corpus.

Why this exists. `data/minecraft/raw/*.schematic` was long documented as unlabeled
("filenames != metadata", notes.md §8, data_sources.md). That conclusion came from
joining the files against the *PlanetMinecraft* tfrecord metadata by index offset,
which correctly failed -- because these files are not from PlanetMinecraft. They came
from `data/download.py`, which walks minecraft-schematics.com ids 1824..19000, and the
files are named `<id>.schematic` (observed range 1..18905). The filename IS the
m-s.com schematic id, so every file re-links to its own page.

Route. The live site sits behind Cloudflare and 403s plain HTTP regardless of
user-agent, so we read the Internet Archive instead: no circumvention, and ~96% of
the local corpus has a 200-status snapshot. Fetches use the `id_` raw-content mode so
we parse the original page rather than the Wayback toolbar wrapper.

Output is a QUARANTINED sidecar (`data/minecraft/ms_labels/labels.jsonl`). It is
deliberately not merged into the curated house caches: `houses_32` is the reference
set for the T11/T12 val_nn 0.405 baseline, and changing the data underneath it would
make the pending native-resolution comparison uninterpretable. Merge after that runs.

Note the schematics themselves remain copyright their submitters; this recovers
factual metadata (category/theme/author/date) for research indexing only.

Usage::

    python -m blockgen.data.recover_ms_labels --build-index   # CDX -> snapshot index
    python -m blockgen.data.recover_ms_labels --limit 50      # smoke test
    python -m blockgen.data.recover_ms_labels                 # full, resumable
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

RAW_DIR = Path("data/minecraft/raw")
OUT_DIR = Path("data/minecraft/ms_labels")
INDEX_PATH = OUT_DIR / "snapshots.json"
LABELS_PATH = OUT_DIR / "labels.jsonl"

CDX = ("http://web.archive.org/cdx/search/cdx?url=minecraft-schematics.com/schematic/*"
       "&output=json&fl=original,timestamp,statuscode&filter=statuscode:200"
       "&collapse=urlkey&limit=60000")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/122.0.0.0 Safari/537.36")

_print_lock = threading.Lock()


def local_ids() -> set:
    """Schematic ids we hold on disk (any extension)."""
    out = set()
    if not RAW_DIR.exists():
        return out
    for p in RAW_DIR.iterdir():
        stem = p.name.split(".")[0]
        if stem.isdigit():
            out.add(int(stem))
    return out


def _get(url: str, timeout: int = 45, retries: int = 4) -> Optional[bytes]:
    """GET with backoff on the archive's throttling responses."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 504, 502):
                time.sleep(2 ** attempt + random.random() * 2)
                continue
            return None
        except Exception:
            time.sleep(1 + attempt)
    return None


def build_index() -> Dict[str, Tuple[str, str]]:
    """Latest 200-status snapshot per schematic id, from the CDX API."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("querying CDX (this takes ~30s)...", flush=True)
    raw = _get(CDX, timeout=180)
    if not raw:
        sys.exit("CDX query failed")
    rows = json.loads(raw)[1:]
    best: Dict[int, Tuple[str, str]] = {}
    for orig, ts, _sc in rows:
        m = re.search(r"/schematic/(\d+)/?$", orig)
        if not m:
            continue
        i = int(m.group(1))
        if i not in best or ts > best[i][0]:
            best[i] = (ts, orig)
    index = {str(k): v for k, v in best.items()}
    INDEX_PATH.write_text(json.dumps(index))
    print(f"snapshot index: {len(index)} ids -> {INDEX_PATH}")
    return index


_TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WANTED = {"category", "theme", "size", "file format", "submitted by",
           "posted on", "download(s)", "rating"}


def parse_page(raw: bytes, sid: int) -> Optional[dict]:
    """Pull the metadata table + title out of an archived schematic page."""
    txt = raw.decode("utf-8", errors="ignore")
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)

    rec: dict = {"id": sid}
    m = _TITLE.search(txt)
    if m:
        t = html.unescape(_TAG.sub("", m.group(1))).strip()
        # pages title as "<name>, creation #<id>"
        rec["title"] = re.sub(r",\s*creation\s*#\d+\s*$", "", t, flags=re.I).strip()

    for r in _ROW.findall(txt):
        cells = [html.unescape(_TAG.sub("", c)).strip() for c in _CELL.findall(r)]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        key = cells[0].rstrip(":").strip().lower()
        if key in _WANTED:
            rec[key.replace("(s)", "s").replace(" ", "_")] = cells[1]

    if "downloads" in rec:
        d = re.search(r"([\d,]+)\s+times", rec["downloads"])
        rec["downloads"] = int(d.group(1).replace(",", "")) if d else None
    if not rec.get("category") and not rec.get("title"):
        return None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-index", action="store_true", help="refresh the CDX snapshot index")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent fetches (be gentle)")
    ap.add_argument("--all-ids", action="store_true",
                    help="also fetch ids we do NOT hold locally (metadata-only discovery)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.build_index or not INDEX_PATH.exists():
        index = build_index()
    else:
        index = json.loads(INDEX_PATH.read_text())

    have = local_ids()
    print(f"local corpus: {len(have)} ids")

    done = set()
    if LABELS_PATH.exists():
        for line in LABELS_PATH.read_text().splitlines():
            try:
                done.add(int(json.loads(line)["id"]))
            except Exception:
                pass
        print(f"resuming: {len(done)} already recovered")

    targets = sorted(int(k) for k in index)
    if not args.all_ids:
        targets = [i for i in targets if i in have]
    targets = [i for i in targets if i not in done]
    if args.limit:
        targets = targets[: args.limit]
    print(f"to fetch: {len(targets)}\n", flush=True)
    if not targets:
        return

    out_lock = threading.Lock()
    fh = LABELS_PATH.open("a")
    stats = {"ok": 0, "fail": 0}

    def work(sid: int) -> None:
        ts, orig = index[str(sid)]
        # `id_` = raw archived bytes, no Wayback toolbar injected
        url = f"http://web.archive.org/web/{ts}id_/{orig}"
        raw = _get(url)
        rec = parse_page(raw, sid) if raw else None
        with out_lock:
            if rec:
                rec["snapshot"] = ts
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                stats["ok"] += 1
            else:
                stats["fail"] += 1
            n = stats["ok"] + stats["fail"]
            if n % 100 == 0:
                print(f"  {n}/{len(targets)}  ok={stats['ok']} fail={stats['fail']}",
                      flush=True)
        time.sleep(0.15 + random.random() * 0.2)  # politeness

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, targets))
    fh.close()
    print(f"\ndone: ok={stats['ok']} fail={stats['fail']} -> {LABELS_PATH}")


if __name__ == "__main__":
    main()
