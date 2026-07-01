"""Scrape GrabCraft builds into resumable, per-build raw JSON artifacts.

GrabCraft doesn't expose a schematic download; instead every build page references a
``myRenderObject_<id>.js`` file served from ``/js/RenderObject/`` whose body is::

    var myRenderObject = {"<x>": {"<y>": {"<n>": {"x":.., "y":.., "z":.., "name":..,
                                                   "texture": "<id>_<data>.png", ...}}}}

The nested dict is one entry per *occupied* block. Crucially each entry carries a
``texture`` field of the form ``"<legacy_id>_<data>.png"`` — i.e. the classic
pre-flattening Minecraft numeric id + data value — so we can recover an exact
``(block_id, block_data)`` without any fuzzy name matching (see
:mod:`blockgen.data.grabcraft_dataset`).

This module handles only the *network* half: crawl a category's paginated listing,
resolve each build's render object + page metadata (title, dimensions, tags, views),
and write one JSON file per build under ``data/grabcraft/<category>/<slug>.json``.
Scraping is slow and rate-limited, so artifacts are cached on disk and re-runs skip
builds already fetched (resume). Turning those artifacts into a Structure cache is a
separate, fast, offline step in :mod:`blockgen.data.grabcraft_dataset`.

Run as a script::

    # scrape the whole Houses category (highest priority), politely
    python -m blockgen.data.grabcraft_scraper --category houses

    # a quick smoke test: 1 listing page only
    python -m blockgen.data.grabcraft_scraper --category houses --max-pages 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = str(_REPO_ROOT / "data" / "grabcraft" / "raw")

BASE_URL = "https://www.grabcraft.com"
RENDER_OBJECT_BASE = BASE_URL + "/js/RenderObject/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Two-segment build links: /minecraft/<slug>/<subcategory>. Sidebar/category links
# are single-segment (/minecraft/houses) and pagination is /minecraft/houses/pg/N,
# so requiring exactly two segments (and rejecting 'pg') isolates real builds.
_BUILD_HREF = re.compile(r'href="(/minecraft/[a-z0-9][a-z0-9\-]*/[a-z0-9][a-z0-9\-]*)"')
_RENDER_OBJECT = re.compile(r'(myRenderObject_[0-9a-zA-Z]+\.js)')
_TITLE = re.compile(r'content-title[^>]*>\s*([^<]+?)\s*<')
_DIM_X = re.compile(r'dimension-x[^>]*>\s*(\d+)\s*<')
_DIM_Y = re.compile(r'dimension-y[^>]*>\s*(\d+)\s*<')
_DIM_Z = re.compile(r'dimension-z[^>]*>\s*(\d+)\s*<')
_TAGS = re.compile(r'class="value tags"[^>]*>\s*([^<]*?)\s*<')
_VIEWS = re.compile(r'Views:\s*(?:&nbsp;|\s)*([\d,]+)')


# --- HTTP ------------------------------------------------------------------
def _fetch(url: str, retries: int = 3, backoff: float = 2.0) -> Optional[str]:
    """GET a URL as text, retrying transient failures. Returns None on hard failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # genuinely missing — don't retry
            last = e
        except Exception as e:  # timeouts, connection resets, etc.
            last = e
        time.sleep(backoff * (attempt + 1))
    print(f"    ! giving up on {url}: {last}")
    return None


# --- listing enumeration ---------------------------------------------------
def iter_build_urls(
    category: str, max_pages: Optional[int] = None, delay: float = 1.0,
) -> Iterator[str]:
    """Yield absolute build URLs across a category's paginated listing.

    Stops when a page 404s, yields no new builds, or ``max_pages`` is reached.
    """
    seen: set = set()
    page = 1
    while max_pages is None or page <= max_pages:
        listing = f"{BASE_URL}/minecraft/{category}/pg/{page}"
        html = _fetch(listing)
        if html is None:
            break
        hrefs = [h for h in dict.fromkeys(_BUILD_HREF.findall(html)) if "/pg/" not in h]
        fresh = [h for h in hrefs if h not in seen]
        if not fresh:
            break  # pagination exhausted (site repeats the last page or empties out)
        for href in fresh:
            seen.add(href)
            yield BASE_URL + href
        page += 1
        time.sleep(delay)


# --- per-build extraction --------------------------------------------------
@dataclass
class BuildRecord:
    url: str
    slug: str
    category: str          # subcategory from the build's own url (e.g. "medieval-houses")
    title: str
    width: int
    height: int
    depth: int
    tags: List[str]
    views: int
    blocks: Dict           # the parsed renderObject dict (x -> y -> n -> block)

    def to_json(self) -> dict:
        return {
            "url": self.url, "slug": self.slug, "category": self.category,
            "title": self.title, "width": self.width, "height": self.height,
            "depth": self.depth, "tags": self.tags, "views": self.views,
            "blocks": self.blocks,
        }


def _parse_render_object(js: str) -> Optional[Dict]:
    """Extract the ``{...}`` JSON assigned to ``var myRenderObject`` in the JS body."""
    start = js.find("{")
    if start < 0:
        return None
    body = js[start:].strip().rstrip(";").strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Some payloads have a trailing ``;`` mid-string or stray tail; retry on the
        # substring up to the last closing brace.
        end = body.rfind("}")
        if end > 0:
            try:
                return json.loads(body[:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _slug_and_category(url: str) -> Tuple[str, str]:
    parts = url.rstrip("/").split("/")
    return parts[-2], parts[-1]  # /minecraft/<slug>/<category>


def fetch_build(url: str, delay: float = 1.0) -> Optional[BuildRecord]:
    """Fetch a build page + its render object, returning a fully populated record."""
    page = _fetch(url)
    if page is None:
        return None
    m = _RENDER_OBJECT.search(page)
    if not m:
        print(f"    ! no renderObject on {url}")
        return None
    ro_js = _fetch(RENDER_OBJECT_BASE + m.group(1))
    if ro_js is None:
        return None
    blocks = _parse_render_object(ro_js)
    if not blocks:
        print(f"    ! unparseable renderObject for {url}")
        return None

    slug, category = _slug_and_category(url)
    title_m = _TITLE.search(page)
    tags_m = _TAGS.search(page)
    views_m = _VIEWS.search(page)
    tags = [t.strip() for t in tags_m.group(1).split(",")] if tags_m else []

    def _dim(rx):
        d = rx.search(page)
        return int(d.group(1)) if d else 0

    time.sleep(delay)
    return BuildRecord(
        url=url, slug=slug, category=category,
        title=title_m.group(1).strip() if title_m else slug,
        width=_dim(_DIM_X), height=_dim(_DIM_Y), depth=_dim(_DIM_Z),
        tags=[t for t in tags if t],
        views=int(views_m.group(1).replace(",", "")) if views_m else 0,
        blocks=blocks,
    )


# --- orchestration ---------------------------------------------------------
def _artifact_path(raw_dir: str, category: str, slug: str) -> str:
    return os.path.join(raw_dir, category, f"{slug}.json")


def scrape_category(
    category: str,
    raw_dir: str = DEFAULT_RAW_DIR,
    max_pages: Optional[int] = None,
    max_builds: Optional[int] = None,
    delay: float = 1.0,
    overwrite: bool = False,
) -> dict:
    """Crawl one category, saving one JSON artifact per build. Resumes by skipping
    builds already on disk (unless ``overwrite``). Returns a run summary."""
    n_seen = n_saved = n_skipped = n_failed = 0
    for url in iter_build_urls(category, max_pages=max_pages, delay=delay):
        if max_builds is not None and n_seen >= max_builds:
            break
        n_seen += 1
        slug, subcat = _slug_and_category(url)
        out = _artifact_path(raw_dir, subcat, slug)
        if not overwrite and os.path.exists(out):
            n_skipped += 1
            continue
        rec = fetch_build(url, delay=delay)
        if rec is None:
            n_failed += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec.to_json(), f)
        n_saved += 1
        print(f"  [{n_saved:5d}] {rec.title[:48]:48s} "
              f"{rec.width}x{rec.height}x{rec.depth}  ({subcat})")

    summary = {
        "category": category, "seen": n_seen, "saved": n_saved,
        "skipped_existing": n_skipped, "failed": n_failed, "raw_dir": raw_dir,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape GrabCraft builds to raw JSON.")
    parser.add_argument("--category", default="houses",
                        help="category/subcategory slug, e.g. houses, medieval-houses, castles")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="limit listing pages crawled (quick tests)")
    parser.add_argument("--max-builds", type=int, default=None,
                        help="stop after this many builds seen")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between requests (be polite)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-fetch builds even if an artifact already exists")
    args = parser.parse_args()
    scrape_category(
        args.category, raw_dir=args.raw_dir, max_pages=args.max_pages,
        max_builds=args.max_builds, delay=args.delay, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
