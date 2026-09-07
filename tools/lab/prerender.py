"""Fill the thumbnail cache ahead of time, in parallel.

Why this is a separate process pool and not more threads
--------------------------------------------------------
`renders` funnels every render through one thread on purpose: an EGL context is
thread-affine and a second thread touching it fails with `EGL_BAD_ACCESS`. That
makes the in-server renderer correct and also caps it at one build at a time --
about 223 ms each, so the full corpus is roughly five hours.

Separate *processes* do not share that constraint: each gets its own EGL context
and its own render thread, and they coordinate only through the content-addressed
cache on disk, which is already written atomically (`.part` then `replace`). So
this is `multiprocessing`, and the speedup is close to linear in workers.

`spawn`, not `fork`: forking a process that has already initialised EGL gives the
child a context it does not own, which is the same class of bug one level up.

What it costs, measured on this corpus at 252 px
------------------------------------------------
    11.2 KB per thumbnail, 223 ms per render (one worker)

    everything, 1 view  (the grid)     0.81 GB    4.7 h serial
    everything, 4 views (detail)       3.25 GB   18.8 h serial

Storage is not the constraint -- the numbers above are against 275 GB free, two
orders of magnitude of headroom. Time is, which is what the pool is for. Render
the single grid view for everything; let the other three views stay lazy, since
they are only ever needed for a build someone actually opens.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import List, Optional, Sequence, Tuple

DEFAULT_PX = 252


def _render_chunk(args: Tuple[str, Sequence[int], int, int]) -> Tuple[int, int, int]:
    """Render a slice of one dataset in this worker. `(done, skipped, failed)`."""
    dataset_id, indices, px, n_views = args
    from tools.lab import renders

    done = skipped = failed = 0
    place = renders.placeholder(px)
    for i in indices:
        try:
            keys = [renders.cache_key(dataset_id, i, px, v) for v in range(n_views)]
            if all(renders.cache_path(k).is_file() for k in keys):
                skipped += n_views
                continue
        except Exception:
            pass          # keying needs the build; if that fails, rendering will too
        try:
            frames = ([renders.thumb(dataset_id, i, px=px)] if n_views <= 1
                      else renders.views(dataset_id, i, px=px)[:n_views])
        except Exception:
            failed += n_views
            continue
        for f in frames:
            # A placeholder means the build could not be rendered. `renders`
            # deliberately does not cache it, so counting it as done would
            # overstate coverage on corpora that are a quarter broken files.
            if f == place:
                failed += 1
            else:
                done += 1
    return done, skipped, failed


def prerender(dataset_ids: Sequence[str], px: int = DEFAULT_PX, views: int = 1,
              workers: int = 8, chunk: int = 64,
              limit: Optional[int] = None) -> dict:
    from tools.lab import catalog

    jobs: List[Tuple[str, List[int], int, int]] = []
    totals = {}
    for ds_id in dataset_ids:
        ds = catalog.get_dataset(ds_id)
        if ds is None:
            print(f"[prerender] unknown dataset {ds_id!r}, skipping", flush=True)
            continue
        n = min(ds.n, limit) if limit else ds.n
        totals[ds_id] = n
        for start in range(0, n, chunk):
            jobs.append((ds_id, list(range(start, min(start + chunk, n))), px, views))

    if not jobs:
        return {"datasets": {}, "done": 0, "skipped": 0, "failed": 0, "elapsed_s": 0.0}

    total_builds = sum(totals.values())
    print(f"[prerender] {total_builds:,} builds x {views} view(s) at {px}px "
          f"across {len(jobs)} chunks, {workers} workers", flush=True)

    t0 = time.time()
    done = skipped = failed = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for k, (d, s, f) in enumerate(pool.imap_unordered(_render_chunk, jobs), 1):
            done += d
            skipped += s
            failed += f
            if k % 20 == 0 or k == len(jobs):
                el = time.time() - t0
                rate = (done + skipped) / max(el, 1e-9)
                left = (total_builds * views - done - skipped) / max(rate, 1e-9)
                print(f"[prerender] {k}/{len(jobs)} chunks · {done:,} rendered · "
                      f"{skipped:,} cached · {failed:,} unrenderable · "
                      f"{rate:.0f}/s · ~{left/60:.0f} min left", flush=True)

    return {"datasets": totals, "done": done, "skipped": skipped,
            "failed": failed, "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    from tools.lab import catalog

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="dataset ids; default is every corpus")
    ap.add_argument("--kinds", nargs="*", default=["corpus"],
                    help="discover by kind when --datasets is empty "
                         "(corpus, raw, split, arm)")
    ap.add_argument("--px", type=int, default=DEFAULT_PX)
    ap.add_argument("--views", type=int, default=1,
                    help="1 fills the grid; 4 also fills the detail drawer")
    ap.add_argument("--workers", type=int, default=max(2, (mp.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=None, help="first N per dataset")
    args = ap.parse_args()

    ids = args.datasets or [d.id for d in catalog.list_datasets()
                            if d.kind in set(args.kinds)]
    report = prerender(ids, px=args.px, views=args.views,
                       workers=args.workers, limit=args.limit)
    print(f"\n[prerender] {report['done']:,} rendered, {report['skipped']:,} "
          f"already cached, {report['failed']:,} unrenderable "
          f"in {report['elapsed_s']/60:.1f} min")


if __name__ == "__main__":
    main()
