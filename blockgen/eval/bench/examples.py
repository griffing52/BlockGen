"""A handful of real builds from every arm, written where the lab can already see them.

A scorecard is a wall of numbers. The one question a reader asks first --
*what did `real@solidify` actually look like?* -- has never had an answer on
screen, and 13 of the 16 rows on the user's largest card are arms that exist
only in process memory and were never written anywhere at all. This module is
the fix: after an arm is scored, `k` of its builds are copied into one
`examples_<max_dim>.npz` in the run directory, and the scorecard records which
rows belong to which arm.

**Why a voxel cache and not a PNG.** The obvious answer is to render a contact
sheet next to `scorecard.json`, and it is the wrong one: `tools/lab/api.py` has
*no route that serves a file out of a run directory*. `_static`
(api.py:788-795) resolves only under `tools/lab/static/`, with a traversal
prefix check; the only image routes are `/api/thumb/<build_id>` and
`/api/view/<build_id>/<k>`, both keyed by build id and drained through
`renders.py`. So a PNG dropped in the run dir is a file nothing can fetch.
Adding a route to reach it would open a traversal surface over a directory of
checkpoints and bypass the content-addressed render cache, the `looks_blank`
guard and the single-thread EGL discipline -- to buy a thumbnail the render
pipeline already produces. An example therefore has to be expressible as a
**build id**, not as a path.

**Why exactly an npz + manifest pair.** `catalog._openable` (catalog.py:150-153)
admits a dataset iff `<x>.npz` and `<x>_manifest.json` both exist, and
`catalog._arm_paths` (157-168) already rglobs `outputs/run_*/**/*.npz`. Writing
that pair -- with `curation.houses.save_house_cache`, so there is no new IO code
here -- makes the run's examples a first-class browsable dataset, thumbnailable,
openable in Curate, grouped under its own run node in the hub tree, with **zero**
discovery changes on the lab side. The eval writer records a *path*; the lab
derives the id in `catalog._arm_id` alone, so the two sides cannot drift.

**The aliasing scar, by name.** `ArmSpec.load()` returns `self.structures`
*uncropped and unwrapped* for in-memory arms: the caller gets the very objects
the control builder made. The `real@single_mode` control (fast.py:330-331) is
`[pool[0]] * len(pool)` -- one `Structure` repeated `n` times -- and `pool[0]`
is simultaneously row 0 of the `real_test` arm and a member of the shared `test`
split that the lab serves to Curate. A single in-place `s.metadata["category"] =
arm` here would therefore rewrite one object seen from three datasets, and would
keep doing so for the life of the process, because `catalog` memoizes loaded
builds. Hence `_relabel` returns a `dataclasses.replace` copy with a *new*
metadata dict, always, and `test_write_does_not_mutate_the_inputs` exists to
keep it that way.

**The metadata-less-probe scar.** `save_house_cache` persists only
`corpus/category/title/url` per row (houses.py:385-390), and the probe and
baseline builders construct `Structure(block_ids=..., block_data=...)` with no
metadata at all (probes.py:51,64,81,112,193,240,281,304; baselines.py:97) -- as
does `sample_to_npz`, whose `native_oriented_32_manifest.json` item 0 is
`{"title":"","category":"","corpus":""}`. Captions are not free; if this module
does not set them, most of a card's rows come back blank-captioned and the strip
is 128 unlabelled thumbnails. So the copy gets `category = <arm name>` and a
title, falling back to `"<arm> #<source_row>"` where the source carried none.
Real-split and agentic arms *do* carry a title (the GrabCraft name, the full
prompt sentence) and it is preserved.

**Nothing is rendered here.** One thread-affine EGL context lives in the lab and
is drained by a single worker (`renders.py`); the eval side writes identities and
lets the lab render them, lazily, cached on the voxel SHA-1 -- so a build that
appears both in `bench_arms/*.npz` and here renders exactly once. That also keeps
`blockgen/eval/bench` free of any renderer import on the fast tier, which is the
half that has to run on a reviewer's laptop.

Selection is a seeded permutation (`ctx.rng(9)`, salt 9 being unused by the
runner and by `score_fast`), sorted ascending so the strip reads in source
order. It is uniform, and the page says "a sample, not a best-of" rather than
pretending otherwise: a stratified policy is a body change to `pick` alone.
"""

from __future__ import annotations

import dataclasses
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from blockgen.curation.houses import save_house_cache
from blockgen.utils.data import Structure

#: The `name` handed to `save_house_cache`, so the pair on disk is
#: `examples_<max_dim>.npz` + `examples_<max_dim>_manifest.json`.
CACHE_NAME = "examples"

#: What the sweep (D20) is allowed to delete: exactly the shape
#: `houses.house_cache_path` produces for `CACHE_NAME`, never an arbitrary
#: `examples*` file a human put in the run dir.
_STALE = re.compile(r"^examples_\d+$")


def pick(n: int, k: int, rng: np.random.Generator) -> List[int]:
    """Choose up to `k` of `n` rows: a seeded permutation, sorted ascending.

    A permutation rather than the head of the list, because generation order is
    not random for every writer: `dump_samples.py` keeps source order whenever
    it is not subsampling, so `[:k]` would show the same corner of the corpus
    every time. Sorted on the way out so the strip reads in source order and a
    human can find the neighbours upstream. This is the same idiom
    `dump_samples.py:110-111` already uses to choose its own subset.
    """
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted(int(i) for i in rng.permutation(n)[:k])


def pick_slate(n: int, k: int, rng: np.random.Generator, *,
               prompts: Sequence[str] = (), slate: Sequence[str] = ()) -> List[int]:
    """Rows for the leaderboard strip: the protocol's slate where it applies.

    A protocol may pin a slate of prompts so that column *i* of the leaderboard
    is the same request on every row. That claim is only honourable for arms that
    actually have prompts, so this degrades in one step: with both a slate and
    prompts, return the first row matching each slate entry, in SLATE order (not
    row order -- the columns must line up across models, and that is the whole
    point); otherwise fall back to `pick`'s seeded permutation.

    Matching is exact after case-folding and whitespace collapse, because a slate
    is authored beside the prompts it names. A slate entry with no matching build
    is SKIPPED rather than filled with a substitute: a hole in the strip says "this
    model produced nothing for this prompt", which is true and worth seeing, while
    a silent substitution would put a different build under that column's label.
    """
    if not slate or not prompts:
        return pick(n, k, rng)

    def norm(text: str) -> str:
        return " ".join(str(text or "").split()).casefold()

    first: Dict[str, int] = {}
    for row, prompt in enumerate(prompts[:n]):
        first.setdefault(norm(prompt), row)

    rows: List[int] = []
    for entry in slate:
        row = first.get(norm(entry))
        if row is not None and row not in rows:
            rows.append(row)
        if k > 0 and len(rows) >= k:
            break
    return rows if rows else pick(n, k, rng)


def _relabel(s: Structure, arm: str, src_row: int, corpus: str) -> Structure:
    """A COPY of `s` carrying the fields `save_house_cache` persists.

    Of the four it keeps per row -- `corpus`, `category`, `title`, `url` -- this
    sets the first three and leaves `url` to whatever the source carried.

    Never mutates in place. See the aliasing scar in the module docstring:
    `real@single_mode` is one object repeated, and that object is also
    `real_test` row 0 and a member of the shared test split the lab serves.
    """
    meta = dict(s.metadata or {})
    return dataclasses.replace(s, metadata={
        **meta,
        "category": arm,
        # An existing title is a real caption (GrabCraft name, or the agentic
        # arm's full prompt); the fallback is only for the metadata-less
        # probe/baseline/sampler builds, which are most of a card.
        "title": meta.get("title") or f"{arm} #{src_row}",
        "corpus": meta.get("corpus") or corpus,
    })


def write_run_examples(
    run_dir: Union[str, Path],
    picked: "OrderedDict[str, Tuple[List[Structure], List[int]]]",
    *,
    corpus: str,
    k: int,
    seed: int,
    run_name: str = "",
) -> Dict[str, Any]:
    """Write one examples cache for a whole run; return what the card records.

    `picked` maps arm name -> (the already-chosen structures, their indices in
    that arm's own scored list), in card order. One file per *run*, not per arm:
    `catalog.list_datasets()` is rebuilt on every call and opens a zipfile per
    entry, so 16 per-arm caches per run would be a real cost for no gain.

    Returns ``{"run": {...}, "arms": {name: {"rows": [...], "source_rows":
    [...]}}}`` -- `run` goes to `card.run["examples"]`, and each `arms` entry to
    that arm's `meta["examples"]`. `rows` index the run's npz; `source_rows`
    index the arm's own list, so a human can find the same build upstream.
    Nothing to write returns ``{}`` and touches no file.
    """
    run_dir = Path(run_dir)

    rows: List[Structure] = []
    index: "OrderedDict[str, Dict[str, List[int]]]" = OrderedDict()
    for arm, (structs, src_rows) in picked.items():
        if not structs:
            continue          # an arm that produced nothing gets no `examples` key at all
        start = len(rows)
        rows.extend(_relabel(s, arm, int(src), corpus)
                    for s, src in zip(structs, src_rows))
        index[arm] = {"rows": list(range(start, len(rows))),
                      "source_rows": [int(i) for i in src_rows[:len(structs)]]}
    if not rows:
        # Return before the sweep, deliberately: the sweep exists to stop *this*
        # write leaving an orphan under a different `max_dim`, and a run that
        # writes nothing has no claim on a pair it did not produce. The cost is
        # a stale examples dataset surviving a rerun into the same --out dir
        # with --examples 0; deleting a file we are not replacing is worse.
        return {}

    # D20: the filename carries the true max dimension of the chosen builds, so
    # a rerun into the same --out dir with different builds writes a *different*
    # name and would otherwise leave the previous pair behind as an orphan
    # dataset the lab still lists. A fixed fake `max_dim` would be a small lie
    # in a file the lab reads, so sweep instead.
    for suffix in (".npz", "_manifest.json"):
        for stale in sorted(run_dir.glob(f"examples_*{suffix}")):
            if _STALE.match(stale.name[: -len(suffix)]):
                stale.unlink(missing_ok=True)

    max_dim = max(max(s.shape) for s in rows)
    report = {
        "kind": "bench_examples",
        "run": run_dir.name,
        "run_name": run_name,
        "k": int(k),
        "seed": int(seed),
        "arms": {arm: entry["rows"] for arm, entry in index.items()},
    }
    path = Path(save_house_cache(rows, max_dim=int(max_dim),
                                 cache_dir=str(run_dir), report=report,
                                 name=CACHE_NAME))
    return {
        "run": {
            # Run-dir-RELATIVE and POSIX, never absolute: a run directory that is
            # copied, tarred or moved must still resolve, and `run.dir` itself is
            # already an absolute scratchpad path on two cards on disk.
            "npz": path.name,
            "manifest": path.name.replace(".npz", "_manifest.json"),
            "k": int(k),
            "count": len(rows),
            "seed": int(seed),
        },
        "arms": dict(index),
    }
