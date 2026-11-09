"""StableText2Brick as a Minecraft-encoded corpus: a **second domain** for the
interior probe.

Why this module exists
----------------------
`geometry` and `probes.solidify` record one of this project's load-bearing
findings: filling every enclosed air cell of a real build leaves the render
metric where it was (MV-DINO-KID 0.000 [-0.002, 0.004] against a real floor of
-0.001 [-0.004, 0.003]) while `geom_kid` moves ~200 noise floors. Every number
behind that claim was measured on one corpus -- 32^3 GrabCraft houses -- and the
obvious objection is that it is a fact about Minecraft houses rather than about
render-and-embed metrics. A house is an unusually favourable case: its interior
is a large connected void behind flat opaque walls, so of course the camera
misses it.

So the claim needs a domain that shares none of that. `data/lego/stabletext2brick`
(BrickGPT's corpus, MIT, 47,389 structures already on disk) is the strongest one
available:

* the objects are ShapeNet furniture and vehicles -- cars, sofas, bottles,
  cameras, towers -- not architecture, so "interior" here means the hollow of a
  car body or a bottle, not a room;
* the grid is 20^3, not 32^3, and structures are ~95 bricks of tiled cuboids
  rather than block-by-block walls, so the local pattern statistics are a
  different distribution entirely;
* 48.7% of them enclose a void at all, which is what makes the probe meaningful
  rather than a no-op -- `build_corpus` keeps exactly those.

Nothing here trains anything. That is why it lives under `eval/bench/` and not
in `utils/corpora.py`, whose remit is corpora the *models* pretrain on: this is
an evaluation corpus, built by the benchmark, for the benchmark.

Encoding decisions, and why each one is the way it is
-----------------------------------------------------
**The brick list.** Each structure is text, one brick per line, `hxw (x,y,z)`:
a 1-unit-tall cuboid spanning `h` along x, `w` along y, sitting on layer `z`.
Verified against 1,000 structures: every brick fits inside 20^3 and **no two
bricks of a structure overlap**, which is the check that pins the axis
convention -- read `h`/`w` onto the wrong pair of axes and overlaps appear
immediately. `bricks_to_structure` therefore treats an overlap as a parse error
rather than as last-write-wins, because silently absorbing one would let a
future format change pass as data.

**z is up, y is not.** The corpus stacks 1-unit-tall bricks along z; `Structure`
is XYZ with **y** vertical (`utils/data.py`). Every brick would be a vertical
plate instead of a flat tile if this were transposed wrongly, so the loader maps
`(x, y, z)_lego -> (x, z, y)_mc` and `probes.jitter_columns`, `geometry.height_profile`
and the renderer's camera all then mean what they say.

**Bricks become wool, keyed by brick footprint.** The eight brick types are the
only per-brick attribute the corpus carries (it has no colours), and throwing
them away would make every render a single flat mass in which the brick tiling
-- the thing that makes a LEGO model look like a LEGO model -- is invisible.
Mapping each footprint to a distinct wool colour keeps that tiling on the
rendered surface, which *helps* the render metric: the camera gets strictly more
information than a silhouette, so a blindness result measured this way cannot be
dismissed as "the renderer was shown nothing". Wool specifically, rather than a
spread across stone/ore/glass, because wool is the one matte untextured family
in the vocabulary -- it reads as plastic, and it does not add high-frequency
texture detail that would confound the feature distance with material noise.

The cost is honest and worth stating: every block in this corpus is one palette
*family*, so `palette_jsd_family` is degenerate here and only the `exact` level
discriminates. This corpus is for the structural probe; it is not a palette
benchmark.

**Split grouping.** 47k structures cover only ~28k ShapeNet objects, so the same
mesh appears brick-ified several times -- near-duplicates that must not straddle
a split, or the held-out-real floor is measured against twins of itself.
`splits.group_key` builds its family key from `category::clean_title`, so the
loader puts the object id in `title`. The `-obj` suffix is load-bearing:
`splits._TRAILING_NUM` strips trailing digits (it exists to merge "Brick House 1/2/3"),
and a bare hex object id ending in digits would be truncated into collisions.

**Where the cache lands.** `splits.load_split` reads
`curation.houses.DEFAULT_CACHE_DIR` (`data/minecraft/cache/`) and the bench
runner gives no way to point it elsewhere, so that is where the `.npz` goes. It
is not as wrong as it looks: what is cached is the *Minecraft-encoded* corpus,
which belongs beside the other Minecraft-encoded caches. The LEGO source stays
in `data/lego/`.

Usage
-----
    python -m blockgen.eval.bench.lego --limit 3000            # -> lego_void_20.npz
    python -m blockgen.eval.bench.lego --name lego_all --no-void-filter
    python -m blockgen.eval.bench --corpus lego_void_20 --tier both --n 256
    # BrickGPT's own held-out split, unfiltered: the reference for scoring BrickGPT
    python -m blockgen.eval.bench.lego --name lego_st2b_test --splits test \
        --no-void-filter --min-blocks 1 --limit 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.curation.houses import (DEFAULT_CACHE_DIR, enclosed_air_count,
                                     save_house_cache)
from blockgen.utils.data import Structure

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "data" / "lego" / "stabletext2brick" / "data"

#: The 20x20x20 world grid every StableText2Brick structure is placed in.
GRID = 20

#: ShapeNet synset id -> the category name the dataset card lists. All 21 of
#: them; `category` is what `splits.group_key` pairs with the object id, and it
#: is also the axis a reviewer will ask to slice this result along ("does it
#: replicate on cars but not bottles?"), so the names are spelled out rather
#: than left as synsets.
CATEGORY_NAMES: Dict[str, str] = {
    "02801938": "basket",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02942699": "camera",
    "02958343": "car",
    "03001627": "chair",
    "03467517": "guitar",
    "03593526": "jar",
    "03797390": "mug",
    "03928116": "piano",
    "03991062": "pot",
    "04256520": "sofa",
    "04379243": "table",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
}

#: Categories whose objects are hollow shells often enough to be worth naming.
#: Used only by `--categories`; the default corpus filters on measured voids
#: instead, which is the stronger criterion.
VOID_RICH_CATEGORIES = ("camera", "car", "bottle", "bus", "tower", "sofa")

WOOL_ID = 35

#: Brick footprint (sorted, so 1x2 and 2x1 are one type) -> wool data value.
#: Eight types, eight colours, chosen to be far apart in hue so the tiling is
#: legible in a 224px render rather than merely present.
BRICK_WOOL: Dict[Tuple[int, int], int] = {
    (1, 1): 14,   # red
    (1, 2): 11,   # blue
    (2, 2): 4,    # yellow
    (1, 4): 5,    # lime
    (2, 4): 1,    # orange
    (1, 6): 3,    # light blue
    (2, 6): 0,    # white
    (1, 8): 15,   # black
}

#: `hxw (x,y,z)`. Anchored, so a line carrying anything else -- a comment, a
#: colour field a future version adds -- is a parse error and not a silent drop.
BRICK_RE = re.compile(r"^(\d+)x(\d+)\s*\((\d+),\s*(\d+),\s*(\d+)\)$")

#: `(h, w, x, y, z)`, in the corpus's own axis convention.
Brick = Tuple[int, int, int, int, int]


def parse_bricks(text: str) -> List[Brick]:
    """Brick-list text -> `(h, w, x, y, z)` tuples, in file order.

    Raises on a line that does not match `BRICK_RE`. The corpus is clean (0 bad
    lines in 1,000 structures), so a malformed line means the format moved, and
    a loader that skipped it would report a smaller structure rather than a
    problem.
    """
    out: List[Brick] = []
    for lineno, raw in enumerate(text.strip().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = BRICK_RE.match(line)
        if m is None:
            raise ValueError(f"line {lineno}: not a brick spec: {raw!r}")
        h, w, x, y, z = (int(g) for g in m.groups())
        out.append((h, w, x, y, z))
    return out


def brick_wool(h: int, w: int) -> int:
    """Wool data value for a footprint, orientation-independent.

    An unseen footprint gets a colour by falling back on its area rather than
    raising: a new brick type in a future release should cost this corpus one
    ambiguous colour, not the ability to load.
    """
    key = (min(h, w), max(h, w))
    if key in BRICK_WOOL:
        return BRICK_WOOL[key]
    return int((h * w) % 16)


def bricks_to_structure(bricks: Sequence[Brick], grid: int = GRID,
                        source_path: str = "",
                        metadata: Optional[Dict[str, str]] = None) -> Structure:
    """Voxelize a brick list into a `Structure` (XYZ, y vertical, wool blocks).

    Overlapping bricks raise. The corpus contains none, and the check is the
    cheapest available proof that `h`/`w`/`z` have been read onto the right axes
    -- a transposed convention produces overlaps in the first few structures.
    """
    ids = np.zeros((grid, grid, grid), dtype=np.int32)
    data = np.zeros((grid, grid, grid), dtype=np.int32)
    for h, w, x, y, z in bricks:
        if not (0 <= x and x + h <= grid and 0 <= y and y + w <= grid
                and 0 <= z < grid):
            raise ValueError(f"brick {h}x{w} at ({x},{y},{z}) leaves the "
                             f"{grid}^3 grid")
        # LEGO (x, y, z) with z up -> Structure (x, y=z, z=y) with y up.
        cell = np.s_[x:x + h, z, y:y + w]
        if ids[cell].any():
            raise ValueError(f"brick {h}x{w} at ({x},{y},{z}) overlaps another")
        ids[cell] = WOOL_ID
        data[cell] = brick_wool(h, w)
    return Structure(block_ids=ids, block_data=data, source_path=source_path,
                     metadata=dict(metadata or {}))


def enclosed_void_blocks(s: Structure) -> int:
    """Air cells sealed inside the build: what `probes.solidify` would fill.

    Delegates to `curation.houses.enclosed_air_count`, the same 6-connected
    flood fill the house corpus was curated with and `topology` reports, so
    "has an interior" means one thing across the whole suite.
    """
    return enclosed_air_count(s.crop_to_non_air().block_ids)


def iter_parquet(root: Path | str = DEFAULT_ROOT,
                 split: str = "train") -> Iterator[Dict[str, object]]:
    """Yield `{structure_id, object_id, category_id, bricks}` rows, row group at
    a time.

    Only those four columns are read. `stability_scores` is a 20x20x20 float
    array per row -- 3.2 GB decompressed against 43 MB on disk -- so touching it
    would turn a two-minute corpus build into a memory incident for a field this
    module never uses.
    """
    import pyarrow.parquet as pq

    files = sorted(Path(root).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet under {root}")
    cols = ["structure_id", "object_id", "category_id", "bricks"]
    for path in files:
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=cols).to_pylist():
                yield row


def row_structure(row: Dict[str, object], split: str,
                  grid: int = GRID) -> Structure:
    """One parquet row -> `Structure`, carrying the metadata the split needs."""
    cat_id = str(row.get("category_id") or "")
    obj_id = str(row.get("object_id") or "")
    sid = str(row.get("structure_id") or "")
    return bricks_to_structure(
        parse_bricks(str(row["bricks"])), grid=grid,
        source_path=f"stabletext2brick:{split}:{sid}",
        metadata={
            "corpus": "stabletext2brick",
            "category": CATEGORY_NAMES.get(cat_id, cat_id),
            # The split's family key. `-obj` defends the id from the
            # trailing-digit stripper; see the module docstring.
            "title": f"{obj_id}-obj",
            "url": "hf://AvaLovelace/StableText2Brick",
        })


def build_corpus(root: Path | str = DEFAULT_ROOT,
                 splits: Sequence[str] = ("train", "test"),
                 min_void: int = 8,
                 min_blocks: int = 40,
                 categories: Sequence[str] = (),
                 limit: Optional[int] = None,
                 seed: int = 0,
                 grid: int = GRID,
                 verbose: bool = True) -> Tuple[List[Structure], dict]:
    """Load, filter to structures with interiors, and report what was dropped.

    `min_void = 8` rather than 1 on purpose. The probe must have something to
    destroy: a single sealed cell is a modelling artifact, not an interior, and
    a corpus of those would measure both metrics against damage that is barely
    there and would make a null result uninterpretable. Eight cells is a 2x2x2
    pocket, the smallest thing worth calling hollow.

    The dataset's own train/test split is pooled, because the benchmark imposes
    its own group-aware split (`bench.splits`) and honouring BrickGPT's would
    leave the pool smaller for no gain -- nothing here is trained, so there is
    no leakage to protect against.

    **`limit` subsamples with a reservoir, and that matters.** The parquet is
    written in category order, so truncating the scan at the first `limit`
    survivors -- which is what the first version did -- yields a corpus of
    baskets, beds and benches and no cars at all. A reservoir keeps the sample
    uniform over the whole corpus at O(limit) memory, and the whole scan costs
    about 20 seconds, so there is nothing to buy by stopping early. Statistics
    in the report cover every *eligible* structure, not only the sampled ones.
    """
    keep: List[Structure] = []
    report = {"seen": 0, "parse_errors": 0, "too_small": 0, "no_void": 0,
              "off_category": 0, "voids": [], "blocks": [],
              "by_category": {}}
    wanted = {c.lower() for c in categories}
    rng = np.random.default_rng(seed)
    n_eligible = 0
    for split in splits:
        for row in iter_parquet(root, split):
            report["seen"] += 1
            try:
                s = row_structure(row, split, grid=grid)
            except ValueError:
                report["parse_errors"] += 1
                continue
            if wanted and s.metadata.get("category", "") not in wanted:
                report["off_category"] += 1
                continue
            n_blocks = int(s.occupied_mask.sum())
            if n_blocks < min_blocks:
                report["too_small"] += 1
                continue
            void = enclosed_void_blocks(s)
            if void < min_void:
                report["no_void"] += 1
                continue
            report["voids"].append(void)
            report["blocks"].append(n_blocks)
            cat = s.metadata.get("category", "?")
            report["by_category"][cat] = report["by_category"].get(cat, 0) + 1
            cropped = s.crop_to_non_air()
            if limit is None or len(keep) < limit:      # reservoir: fill
                keep.append(cropped)
            else:                                       # reservoir: replace
                j = int(rng.integers(0, n_eligible + 1))
                if j < limit:
                    keep[j] = cropped
            n_eligible += 1
            if verbose and n_eligible % 2000 == 0:
                print(f"[lego] {n_eligible} eligible of {report['seen']} seen",
                      flush=True)

    voids = np.asarray(report.pop("voids"), dtype=float)
    blocks = np.asarray(report.pop("blocks"), dtype=float)
    report["eligible"] = n_eligible
    report["kept"] = len(keep)
    report["sample_seed"] = seed
    report["min_void"] = min_void
    report["min_blocks"] = min_blocks
    report["categories"] = list(categories)
    if len(keep):
        # The size of the damage `probes.solidify` will do, in the same unit the
        # house result is quoted in ("a median 31% more blocks"), so the two
        # domains can be compared on how hard the probe hits before looking at
        # what either metric did about it.
        report["void_frac_median"] = float(np.median(voids / blocks))
        report["void_frac_mean"] = float(np.mean(voids / blocks))
        report["n_blocks_median"] = float(np.median(blocks))
    if verbose:
        print(f"[lego] {len(keep)} kept (sampled from {n_eligible} eligible of "
              f"{report['seen']} seen; dropped: {report['no_void']} no_void, "
              f"{report['too_small']} too_small, {report['parse_errors']} "
              f"parse_errors, {report['off_category']} off_category)", flush=True)
        if len(keep):
            print(f"[lego] solidify would add a median "
                  f"{report['void_frac_median']:.1%} blocks", flush=True)
    return keep, report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m blockgen.eval.bench.lego",
        description="Build a Minecraft-encoded StableText2Brick corpus cache.")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--name", default="lego_void",
                    help="cache name; the corpus is <name>_<grid>")
    ap.add_argument("--limit", type=int, default=3000,
                    help="uniformly subsample this many structures (0 = all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the subsample reservoir")
    ap.add_argument("--min-void", type=int, default=8)
    ap.add_argument("--min-blocks", type=int, default=40)
    ap.add_argument("--categories", nargs="*", default=[],
                    help=f"restrict to these categories "
                         f"(void-rich: {' '.join(VOID_RICH_CATEGORIES)})")
    ap.add_argument("--no-void-filter", action="store_true",
                    help="keep structures with no interior too (the control "
                         "corpus: the probe has nothing to destroy there)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    help="dataset splits to pool. `--splits test` builds a "
                         "corpus BrickGPT never trained on, which is the only "
                         "honest reference for scoring BrickGPT itself")
    args = ap.parse_args(argv)

    structures, report = build_corpus(
        root=args.root, splits=tuple(args.splits),
        min_void=0 if args.no_void_filter else args.min_void,
        min_blocks=args.min_blocks, categories=args.categories,
        limit=args.limit or None, seed=args.seed)
    if not structures:
        print("[lego] nothing kept; refusing to write an empty cache")
        return 1
    path = save_house_cache(structures, max_dim=GRID, cache_dir=args.cache_dir,
                            report=report, name=args.name)
    print(f"-> {path}  (corpus {Path(path).stem})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
