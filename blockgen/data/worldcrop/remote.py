"""Stream world archives out of the Minecraft Archive Project monthly zips on archive.org.

Each ``Minecraft_Archive_Project_Worlds_YYYYMM`` item is one 10-65 GB zip. We never
download it whole: :class:`HttpFile` exposes the remote file as a seekable object and
``zipfile`` reads the central directory (a few range requests), after which single
members (the per-world .zip/.rar/.7z archives, typically 1-10 MB) are fetched on demand.

The month also carries ``threads.ndjson`` (the original forum/PMC posts: permalink,
title, description, links) and ``resources.ndjson`` (every mirrored file -> its
``referer`` thread). We join the two so each world keeps its provenance.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

IA_ITEM = "Minecraft_Archive_Project_Worlds_{month}"
WORLD_EXTS = (".zip", ".rar", ".7z")
_UA = "BlockGen-worldcrop/0.1 (academic research; contact via project repo)"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP Range requests."""

    def __init__(self, url: str, retries: int = 6):
        self.pos = 0
        self.retries = retries
        self.nreq = 0
        self.nbytes = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _UA})
        last = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.size = int(r.headers["Content-Length"])
                    self.url = r.url  # follow archive.org's redirect to the storage node once
                break
            except Exception as e:  # archive.org answers 500/503 sporadically
                last = e
                time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise last

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n == -1:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        hdr = {"Range": f"bytes={self.pos}-{self.pos + n - 1}", "User-Agent": _UA}
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=hdr), timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:  # transient IA hiccups are common
                last = e
                time.sleep(min(60, 2 ** attempt))
        else:
            raise last
        self.pos += len(data)
        self.nreq += 1
        self.nbytes += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def ia_file_url(month: str) -> str:
    item = IA_ITEM.format(month=month)
    meta = json.load(urllib.request.urlopen(
        urllib.request.Request(f"https://archive.org/metadata/{item}", headers={"User-Agent": _UA}), timeout=60))
    zips = [f for f in meta["files"] if f["name"].lower().endswith(".zip") and int(f.get("size", 0)) > 1e8]
    if not zips:
        raise FileNotFoundError(f"no main zip in {item}")
    name = max(zips, key=lambda f: int(f["size"]))["name"]
    return f"https://archive.org/download/{item}/{urllib.parse.quote(name)}"


@dataclass
class WorldEntry:
    member: str            # path inside the monthly zip
    size: int
    month: str
    site: str              # planetminecraft.com / minecraftforum.net / ...
    url: str = ""          # original download URL (resources._id)
    thread: str = ""       # permalink of the post that linked it
    title: str = ""
    description: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def world_id(self) -> str:
        """Stable id: month + sha1 of the member path."""
        import hashlib
        return f"{self.month}_{hashlib.sha1(self.member.encode()).hexdigest()[:12]}"


class MonthArchive:
    def __init__(self, month: str, url: Optional[str] = None):
        self.month = month
        self.url = url or ia_file_url(month)
        self.http = HttpFile(self.url)
        self.zf = zipfile.ZipFile(io.BufferedReader(self.http, buffer_size=1 << 20))
        self.prefix = self._prefix()

    def _prefix(self) -> str:
        for i in self.zf.infolist():
            if i.filename.endswith("/resources.ndjson"):
                return i.filename[: -len("resources.ndjson")]
        return ""

    def _ndjson(self, name: str) -> List[dict]:
        try:
            raw = self.zf.read(self.prefix + name)
        except KeyError:
            return []
        out = []
        for line in raw.decode("utf8", "replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def worlds(self) -> List[WorldEntry]:
        resources = self._ndjson("resources.ndjson")
        threads = {t.get("permalink"): t for t in self._ndjson("threads.ndjson")}
        by_path: Dict[str, dict] = {}
        for r in resources:
            dp = r.get("data_path")
            if dp:
                by_path[dp] = r
        out = []
        for info in self.zf.infolist():
            fn = info.filename
            if not fn.lower().endswith(WORLD_EXTS) or info.file_size == 0:
                continue
            rel = fn[len(self.prefix):] if fn.startswith(self.prefix) else fn
            parts = rel.split("/")
            site = parts[1] if len(parts) > 2 else "?"
            e = WorldEntry(member=fn, size=info.file_size, month=self.month, site=site)
            r = by_path.get(rel)
            if r:
                e.url = r.get("_id", "")
                e.thread = r.get("referer", "") or ""
                t = threads.get(e.thread)
                if t:
                    e.title = str(t.get("title", "") or "")
                    e.description = _strip_html(str(t.get("description", "") or ""))
                    e.meta = {k: t[k] for k in ("tags", "category", "author", "date", "views", "downloads", "diamonds")
                              if k in t}
            out.append(e)
        # the finding-aid pages are the reliable archive -> thread join (see provenance.py)
        try:
            from . import provenance
            provenance.apply_to_entries(out, provenance.month_threads(self))
        except Exception:
            pass
        return out

    def fetch(self, entry: WorldEntry, dest_path: str) -> str:
        data = self.zf.read(entry.member)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path


_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return re.sub(r"\s+\n", "\n", _TAG.sub("", s).replace("<br/>", "\n")).strip()


def list_months() -> List[str]:
    """All Worlds_YYYYMM items on archive.org (via the advanced search API)."""
    q = urllib.parse.quote("identifier:Minecraft_Archive_Project_Worlds_*")
    url = f"https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&fl[]=item_size&rows=500&output=json"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=60))
    months = []
    for doc in d["response"]["docs"]:
        m = doc["identifier"].rsplit("_", 1)[-1]
        months.append((m, int(doc.get("item_size", 0))))
    return [m for m, _ in sorted(months)]
