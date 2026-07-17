"""Fetch the LDraw Official Model Repository (OMR) into data/lego/omr/.

The OMR is the license-clean core of the LEGO corpus (roadmap.md): official
LEGO sets as LDraw MPD files (full part+pose per line), each required to carry
``0 !LICENSE Licensed under CC BY 4.0``.

There is no bulk download, so this scrapes the set index politely:
paginate ``/omr/sets?page=N`` -> collect ``/omr/sets/<id>`` pages -> extract
``/library/omr/*.mpd`` links -> download. Resumable: already-downloaded files
are skipped and the manifest is rewritten every page.

    python -m blockgen.data.fetch_omr            # full fetch (~45 min polite)
    python -m blockgen.data.fetch_omr --max-pages 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://library.ldraw.org"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "lego" / "omr"
UA = {"User-Agent": "BlockGen-research/0.1 (academic use; contact: ggalimi@math.ucla.edu)"}

_SET_HREF = re.compile(r"href=\"https://library\.ldraw\.org/omr/sets/(\d+)\"")
_MPD_HREF = re.compile(r"href=\"(https://library\.ldraw\.org/library/omr/[^\"]+\.mpd)\"")
_TITLE = re.compile(r"<title>([^<]*)</title>")


def _get(url: str, delay: float, retries: int = 4) -> str | bytes:
    time.sleep(delay)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            break
        except Exception:  # noqa: BLE001  (502s and timeouts are transient here)
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * 2 ** attempt)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data


def fetch_omr(out_dir: Path = DEFAULT_OUT, delay: float = 0.25,
              max_pages: int | None = None) -> dict:
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # 1. paginate the set index until a page adds nothing new
    set_ids: list[str] = []
    seen: set = set(manifest)
    page = 1
    while True:
        html = _get(f"{BASE}/omr/sets?page={page}", delay)
        ids = list(dict.fromkeys(_SET_HREF.findall(html)))
        new = [i for i in ids if i not in set_ids]
        if not new:
            break
        set_ids += new
        print(f"page {page}: +{len(new)} sets (total {len(set_ids)})", flush=True)
        page += 1
        if max_pages and page > max_pages:
            break

    # 2. per set: find mpd links, download
    n_files = n_skipped = n_err = 0
    for k, sid in enumerate(set_ids):
        if sid in manifest and all((files_dir / f).exists()
                                   for f in manifest[sid].get("files", [])):
            n_skipped += 1
            continue
        try:
            html = _get(f"{BASE}/omr/sets/{sid}", delay)
            urls = list(dict.fromkeys(_MPD_HREF.findall(html)))
            title = (_TITLE.search(html) or [None, ""])[1]
            fnames = []
            for u in urls:
                fname = u.rsplit("/", 1)[-1]
                dest = files_dir / fname
                if not dest.exists():
                    data = _get(u, delay)
                    dest.write_bytes(data if isinstance(data, bytes) else data.encode())
                    n_files += 1
                fnames.append(fname)
            manifest[sid] = {"title": title.strip(), "files": fnames}
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  set {sid}: ERROR {e}", flush=True)
        if (k + 1) % 50 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"[{k + 1}/{len(set_ids)}] {n_files} new files, "
                  f"{n_skipped} skipped, {n_err} errors", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    total = len(list(files_dir.glob("*.mpd")))
    report = {"sets_indexed": len(set_ids), "sets_in_manifest": len(manifest),
              "mpd_files": total, "new": n_files, "errors": n_err}
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch the LDraw OMR (official LEGO sets).")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--delay", type=float, default=0.25)
    ap.add_argument("--max-pages", type=int, default=None)
    a = ap.parse_args()
    fetch_omr(Path(a.out), delay=a.delay, max_pages=a.max_pages)


if __name__ == "__main__":
    main()
