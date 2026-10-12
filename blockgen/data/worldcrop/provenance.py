"""Provenance for streamed worlds: which forum thread each archive came from.

Every ``Worlds_YYYYMM`` zip carries a static finding aid under ``threads/detail/*.html``
(one page per thread: title, author, original URL, creation date, description, and a
"Mirrored resources" table whose links are the zip members). Some months also carry
``threads.ndjson`` / ``resources.ndjson``, but the download resources there almost never
record a referer, so the HTML table is the only reliable archive -> thread join.

The pages are small (~8 MB/month) and stored contiguously, so they are fetched with ONE
range request and inflated by hand instead of 6,000 per-member reads.

    python -m blockgen.data.worldcrop.provenance --rejoin        # fix entries in existing worlds.jsonl
    python -m blockgen.data.worldcrop.provenance --month 201508  # print the join rate
"""
from __future__ import annotations

import argparse
import glob
import html as htmlmod
import json
import os
import re
import zlib
from typing import Dict, List, Optional
from urllib.parse import unquote

from . import remote

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return htmlmod.unescape(_TAG.sub(" ", fragment)).replace("\xa0", " ").strip()


def _bulk_members(zf, infos) -> Dict[str, bytes]:
    """Read many small, contiguous members with one range request."""
    if not infos:
        return {}
    lo = min(i.header_offset for i in infos)
    hi = max(i.header_offset + 30 + len(i.filename.encode("utf8")) + 2 ** 16 + i.compress_size for i in infos)
    f = zf.fp
    f.seek(lo)
    blob = f.read(hi - lo)
    out = {}
    for i in infos:
        off = i.header_offset - lo
        if blob[off:off + 4] != b"PK\x03\x04":
            continue
        nlen = int.from_bytes(blob[off + 26:off + 28], "little")
        elen = int.from_bytes(blob[off + 28:off + 30], "little")
        start = off + 30 + nlen + elen
        raw = blob[start:start + i.compress_size]
        try:
            out[i.filename] = zlib.decompress(raw, -15) if i.compress_type == 8 else raw
        except zlib.error:
            continue
    return out


def parse_thread_page(page: str) -> dict:
    body = _WS.sub(" ", page)
    t = {"title": "", "author": "", "url": "", "created": "", "description": "", "members": [], "originals": []}
    m = re.search(r"<h1>(.*?)</h1>", body)
    if m:
        t["title"] = _text(m.group(1))
    m = re.search(r"<h2>\s*by\s+(.*?)</h2>", body)
    if m:
        t["author"] = _text(m.group(1))
    sec = re.search(r"<section class=\"description\">(.*?)</section>", body)
    if sec:
        paras = re.findall(r"<p>(.*?)</p>", sec.group(1))
        if paras:
            head = paras[0]
            m = re.search(r"href=\"([^\"]+)\"", head)
            if m:
                t["url"] = htmlmod.unescape(m.group(1))
            m = re.search(r"created ([0-9]{4}-[0-9]{2}-[0-9]{2}[ 0-9:]*)", head)
            if m:
                t["created"] = m.group(1).strip()
            t["description"] = "\n".join(_text(p) for p in paras[1:] if _text(p))[:4000]
    res = re.search(r"<section class=\"resources\">(.*?)</section>", body)
    if res:
        for row in re.findall(r"<tr>(.*?)</tr>", res.group(1)):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
            if len(cells) < 2:
                continue
            m = re.search(r"href=\"([^\"]+)\"", cells[1])
            if not m:
                continue
            href = htmlmod.unescape(m.group(1))
            rel = unquote(href[6:] if href.startswith("../../") else href)
            orig = re.search(r"href=\"([^\"]+)\"", cells[2]) if len(cells) > 2 else None
            t["members"].append(rel)
            t["originals"].append(htmlmod.unescape(orig.group(1)) if orig else "")
    return t


def month_threads(archive: remote.MonthArchive) -> Dict[str, dict]:
    """member path (relative to the month prefix) -> thread record."""
    infos = [i for i in archive.zf.infolist()
             if i.filename.startswith(archive.prefix + "threads/detail/") and i.filename.endswith(".html")]
    pages = _bulk_members(archive.zf, infos)
    by_member: Dict[str, dict] = {}
    for name, raw in pages.items():
        t = parse_thread_page(raw.decode("utf8", "replace"))
        t["page"] = name[len(archive.prefix):]
        for rel, orig in zip(t["members"], t["originals"]):
            if rel not in by_member:
                rec = {k: t[k] for k in ("title", "author", "url", "created", "description", "page")}
                rec["original"] = orig
                by_member[rel] = rec
    return by_member


def apply_to_entries(entries: List[remote.WorldEntry], threads: Dict[str, dict]) -> int:
    n = 0
    for e in entries:
        rel = e.member[len("Minecraft - Worlds/") + len(e.month) + 1:] if e.member.startswith("Minecraft - Worlds/") else e.member
        t = threads.get(rel)
        if t:
            e.title, e.description, e.thread = t["title"], t["description"], t["url"]
            e.url = e.url or t["original"]
            e.meta = {**(e.meta or {}), "author": t["author"], "created": t["created"], "page": t["page"]}
            n += 1
    return n


def rejoin_month(month_dir: str, dry: bool = False) -> dict:
    """Rewrite entry.title/description/thread/url/meta in an existing worlds.jsonl."""
    month = os.path.basename(month_dir.rstrip("/"))
    path = os.path.join(month_dir, "worlds.jsonl")
    if not os.path.exists(path):
        return {"month": month, "skipped": "no worlds.jsonl"}
    threads = month_threads(remote.MonthArchive(month))
    rows = [json.loads(l) for l in open(path) if l.strip()]
    hit = 0
    for r in rows:
        e = r.get("entry")
        if not e:
            continue
        member = e.get("member", "")
        rel = member[len("Minecraft - Worlds/") + len(month) + 1:] if member.startswith("Minecraft - Worlds/") else member
        t = threads.get(rel)
        if t:
            e["title"], e["description"], e["thread"] = t["title"], t["description"], t["url"]
            e["url"] = e.get("url") or t["original"]
            e["meta"] = {**(e.get("meta") or {}), "author": t["author"], "created": t["created"], "page": t["page"]}
            hit += 1
    if not dry:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    return {"month": month, "worlds": len(rows), "joined": hit, "threads": len(threads)}


def _month_in_progress(root: str) -> Optional[str]:
    log = os.path.join(root, "logs", "ingest.log")
    if not os.path.exists(log):
        return None
    started, done = [], set()
    for line in open(log):
        m = re.search(r"\] (start|done)\s+(\S+)", line)
        if m:
            (started.append if m.group(1) == "start" else done.add)(m.group(2))
    for m in reversed(started):
        if m not in done:
            return m
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--month")
    ap.add_argument("--rejoin", action="store_true", help="fix every finished month under --root")
    from .run import DEFAULT_OUT
    ap.add_argument("--root", default=DEFAULT_OUT)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    if args.month and not args.rejoin:
        a = remote.MonthArchive(args.month)
        th = month_threads(a)
        ents = a.worlds()
        print(json.dumps({"month": args.month, "threads": len(th), "worlds": len(ents), "joined": apply_to_entries(ents, th)}))
        for e in [x for x in ents if x.title][:3]:
            print(" ", e.title, "|", e.meta.get("author"), "|", e.description[:80].replace("\n", " "))
        return
    root = os.path.normpath(args.root)
    busy = _month_in_progress(root)
    months = [args.month] if args.month else sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "2*")) if os.path.isdir(d))
    for m in months:
        if m == busy:
            print(json.dumps({"month": m, "skipped": "ingest in progress"}), flush=True)
            continue
        try:
            print(json.dumps(rejoin_month(os.path.join(root, m), dry=args.dry)), flush=True)
        except Exception as ex:
            print(json.dumps({"month": m, "error": f"{type(ex).__name__}: {ex}"}), flush=True)


if __name__ == "__main__":
    main()
