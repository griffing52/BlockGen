"""What the lab can look at: arms, corpora, splits, and past scorecards.

This module is the read side of BlockLab. It never writes to `outputs/` or
`data/` -- the batch pipeline owns those -- and every entry point degrades to an
empty result rather than raising, because an empty `outputs/` is the normal state
on a fresh clone and the pages must render an empty state, not a stack trace.

**Discovery does not load structures.** Walking `outputs/` finds ~20 caches; the
biggest of them (`houses_32`, 2661 builds) takes 90 ms to load and tens of MB to
hold, so opening all of them to answer "what is there" would make the first page
paint cost seconds and hundreds of MB. `n` instead comes from the `.npy` header
inside the zip, which is a few hundred bytes off disk. Structures are loaded on
first *use* and memoized after.

**A cache without a `_manifest.json` sibling is not a dataset.**
`load_structures_from_cache` reads row `i`'s metadata out of that manifest and
raises without it, so the manifest's presence is the cheapest honest test of
"can this be opened". It also happens to filter exactly the right things here:
the seven `outputs/run_*/*/samples.npz` files from the T-series attachment runs
are bare arrays with no manifest and would otherwise appear as broken rows.

**Scorecards are normalised on the way in, and the result is shared.**
`_read_scorecard` puts every parsed card through `cards.migrate` *before* the
mtime-keyed memo insert, so the shape-guessing that `bench/1`'s four on-disk
shapes force is paid once per file per mtime and every reader downstream sees
the same normalised blob. The consequence to hold on to: what comes back -- and
in particular its `arms` block -- *is* the object in the process-wide cache.
Read-time enrichment goes beside `arms`, never inside it (see `load_scorecard`).

Ambiguity worth recording: the design contract says corpora live in
`data/cache/*.npz`, but `blockgen.data.build_cache.DEFAULT_CACHE_DIR` -- what
every loader in the repo actually uses -- is `data/minecraft/cache`. Both are
scanned so the contract's path keeps working if it is ever created.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from blockgen.utils.data import Structure
from tools.lab import cards

OUTPUTS_ROOT = Path("outputs")
#: Where generated arms land. `bench_arms/` is the curated benchmark set; the
#: `run_*/` dirs are every training/sampling run ever done (see MEMORY: run dirs
#: are named `run_<YYYYMMDD_HHMMSS>_<name>` so they sort newest-last by name).
ARM_ROOTS = ("bench_arms", "run_*")
CORPUS_DIRS = (Path("data/minecraft/cache"), Path("data/cache"))
CANONICAL_CORPUS = "houses_32"
SPLIT_SIDES = ("train", "val", "test")

#: Cropped structure lists, keyed by dataset id. Small on purpose: `houses_32`
#: alone is ~50 MB resident, and the lab is a browsing tool where two or three
#: datasets are open at a time. Evicting the least-recently-used one costs a
#: 90 ms reload; holding all twenty would cost most of a gigabyte.
_MAX_CACHED_DATASETS = 4
_BUILDS: "OrderedDict[str, List[Structure]]" = OrderedDict()

_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Dataset:
    """One browsable collection of builds.

    `kind` decides how the pages treat it: an "arm" is something a model made and
    is therefore a candidate for labelling, a "corpus" is ground truth, a "split"
    is a named slice of a corpus that the benchmark scores against.
    """

    id: str
    name: str
    kind: str
    n: int
    source: str
    note: str = ""

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "n": self.n,
                "source": self.source, "note": self.note}


# --- ids -------------------------------------------------------------------
def make_build_id(dataset_id: str, index: int) -> str:
    return f"{dataset_id}:{index}"


def parse_build_id(build_id: str) -> Tuple[str, int]:
    """Split `"<dataset_id>:<index>"`.

    Splits from the *right*, because dataset ids legitimately contain colons
    (`split:houses_32:test`). A malformed id raises rather than silently becoming
    index 0 -- these ids arrive from URLs and from the SQLite store, and a
    mislabelled build is worse than a 400.
    """
    text = str(build_id)
    head, sep, tail = text.rpartition(":")
    if not sep or not head:
        raise ValueError(f"malformed build_id {build_id!r}: expected '<dataset>:<index>'")
    try:
        index = int(tail)
    except ValueError:
        raise ValueError(f"malformed build_id {build_id!r}: {tail!r} is not an index")
    if index < 0:
        raise ValueError(f"malformed build_id {build_id!r}: negative index")
    return head, index


def _slug(text: str) -> str:
    return _SLUG.sub("_", text).strip("_")


def _arm_id(path: Path) -> str:
    """Stable, filesystem- and URL-safe id derived from the path under outputs/.

    Path-derived rather than counter-derived so a label survives a restart, a
    `git pull`, and a new run appearing beside this one. `/` becomes `__` because
    the id is a URL path segment in `/api/thumb/<build_id>`.
    """
    try:
        rel = path.resolve().relative_to(OUTPUTS_ROOT.resolve())
    except ValueError:
        rel = Path(path.name)
    return "arm:" + _slug(str(rel.with_suffix("")).replace("/", "__"))


# --- cheap probes ----------------------------------------------------------
def _npz_rows(path: Path) -> Optional[int]:
    """Number of rows in the cache, from the `.npy` header alone.

    The arrays are object-dtype (ragged volumes), so `np.load` would have to
    unpickle every structure to tell us a length that is written in a 100-byte
    header. Reads the header, never the data.
    """
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("block_ids.npy") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                else:
                    return None
        return int(shape[0]) if shape else 0
    except Exception:
        return None


def _manifest_path(npz: Path) -> Path:
    return npz.with_name(npz.name[: -len(".npz")] + "_manifest.json")


def _openable(npz: Path) -> bool:
    return npz.is_file() and _manifest_path(npz).is_file()


def _arm_paths() -> List[Path]:
    seen: Dict[Path, None] = {}
    for root in ARM_ROOTS:
        for base in sorted(OUTPUTS_ROOT.glob(root)):
            if not base.is_dir():
                continue
            for npz in base.rglob("*.npz"):
                if _openable(npz):
                    seen.setdefault(npz.resolve(), None)
    return list(seen)


# --- discovery -------------------------------------------------------------
def _arm_datasets() -> List[Dataset]:
    out: List[Dataset] = []
    for path in _arm_paths():
        n = _npz_rows(path)
        if n is None:
            continue
        try:
            rel = path.relative_to(OUTPUTS_ROOT.resolve())
        except ValueError:
            rel = Path(path.name)
        parent = rel.parent.name or "outputs"
        note = "benchmark arm" if parent == "bench_arms" else parent
        if path.stem.startswith("examples") and cards.RUN_DIR_RE.match(parent):
            # The handful of builds a bench run kept out of every arm, written
            # beside its scorecard. Named rather than left as the bare run dir
            # for two reasons: the hub should say what the node is, and
            # `_flag_smoke` has to tell a k=8 examples cache apart from an
            # eight-build smoke run, which by size alone it cannot.
            note = f"{EXAMPLES_NOTE} · {parent}"
        out.append(Dataset(
            id=_arm_id(path), name=path.stem, kind="arm", n=n,
            # `OUTPUTS_ROOT`, not a literal "outputs": `_load_uncached` opens
            # `Dataset.source` as a path, so a hard-coded prefix would make every
            # arm *discoverable* but *unopenable* whenever the root is repointed
            # (a test's `tmp_path`, or a future deployment that configures it).
            source=str(OUTPUTS_ROOT / rel),
            note=note))
    # bench_arms first (they are the ones the scorecards actually reference),
    # then the run dirs newest-first -- run names sort chronologically by design.
    curated = [d for d in out if d.note == "benchmark arm"]
    runs = [d for d in out if d.note != "benchmark arm"]
    curated.sort(key=lambda d: d.name)
    runs.sort(key=lambda d: (d.note, d.name), reverse=True)
    return curated + runs


def _corpus_datasets() -> List[Dataset]:
    out: List[Dataset] = []
    seen: set = set()
    for directory in CORPUS_DIRS:
        if not directory.is_dir():
            continue
        for npz in sorted(directory.glob("*.npz")):
            if not _openable(npz) or npz.stem in seen:
                continue
            n = _npz_rows(npz)
            if n is None:
                continue
            seen.add(npz.stem)
            out.append(Dataset(id=f"corpus:{_slug(npz.stem)}", name=npz.stem,
                               kind="corpus", n=n, source=str(npz),
                               note="real builds"))
    return out


def _split_datasets() -> List[Dataset]:
    """Every split in `data/minecraft/splits/`, read straight out of its JSON.

    Globbed rather than hard-wired to `CANONICAL_CORPUS`, which is what this used
    to do. Two corpora are split on disk today, and one card in `outputs/`
    (`run_20260906_190029_bench`) is scored on `houses_48`: its `real_test` row
    asks the drill-down for `split:houses_48:test`, the catalog had never heard
    of it, and the thumbnails silently did not appear. The constant stays --
    other callers still mean "the corpus this project is about" by it.

    Only the seed-0, 70-15-15 split of each corpus is listed, because that is the
    only one a `split:<corpus>:<side>` id can address: `_load_uncached` resolves
    such an id with `splits.load_split(corpus, 0)` and default fracs. Listing a
    second seed under the same id would hand back builds from the first.

    Deliberately does *not* call `splits.load_split`: that function rebuilds and
    **writes** the split file when it is missing or stale, and discovery must not
    have side effects on the data directory. It also loads the whole corpus to
    recompute `source_sha`, which is exactly the cost this module avoids. Sizes
    are in the JSON already.
    """
    try:
        from blockgen.eval.bench import splits
    except Exception:
        return []
    root = Path(getattr(splits, "SPLIT_ROOT", "data/minecraft/splits"))
    if not root.is_dir():
        return []

    out: List[Dataset] = []
    seen: set = set()
    for path in sorted(root.glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        key = str(blob.get("corpus") or "")
        # `corpus` plus a canonical filename is the cheapest honest test of "is
        # this one of ours": the repo root also carries an unrelated
        # `splits.json` of 153 `C###` ids (see `splits`' own docstring).
        if not key or path.name != splits.split_path(key, 0, (0.70, 0.15, 0.15)).name:
            continue
        sizes = blob.get("sizes") or {}
        for side in SPLIT_SIDES:
            dataset_id = f"split:{key}:{side}"
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            n = int(sizes.get(side) or len(blob.get(side) or []))
            out.append(Dataset(
                id=dataset_id, name=f"{key} {side}", kind="split", n=n,
                source=f"{key}.s{blob.get('seed', 0)}.70-15-15."
                       f"{blob.get('version', 'v1')}:{side}",
                note="held-out real" if side != "train" else "training real"))
    return out


#: Ordering weight per kind. Corpora first because they are what a session
#: usually starts from -- the discovery order (arms, corpora, splits) buried the
#: nine real corpora under seventeen one-to-eight-build smoke runs from
#: `outputs/run_*/`, which made the picker look like it had found the wrong
#: thing entirely. Kind first, then size, so the substantial datasets of each
#: kind lead and the smoke runs sink without being hidden.
_KIND_ORDER = {"corpus": 0, "raw": 1, "subset": 2, "split": 3, "arm": 4}

#: Arms below this are single-build smoke tests, not results. Still listed --
#: hiding data a researcher put on disk is worse than ranking it low -- but
#: flagged so the picker can say why they are at the bottom.
SMOKE_N = 8


def _raw_datasets() -> List[Dataset]:
    """The uncurated source corpora, indexed lazily. See `rawcorpora`.

    Registered as their own kind because they are a different thing from the
    caches: unfiltered, overlapping nothing, and 3.5x larger. A label applied
    here is a judgement about the corpus; a label applied to `houses_32` is a
    judgement about the pipeline's output.
    """
    from tools.lab import rawcorpora as rc

    out: List[Dataset] = []
    for spec in rc.specs():
        if not spec.available():
            continue
        n = len(rc.build_index(spec))
        if n:
            out.append(Dataset(id=f"raw:{spec.id}", name=spec.name, kind="raw",
                               n=n, source=str(spec.root), note=spec.note))
    if out:
        out.append(Dataset(
            id=f"raw:{rc.UNION_ID}", name="Everything (all raw corpora)",
            kind="raw", n=sum(d.n for d in out),
            source=str(rc.MC),
            note="every source corpus concatenated; not deduped -- see docs"))
    return out


def _subset_datasets(known: Dict[str, Dataset]) -> List[Dataset]:
    """Saved branches (see `subsets`), registered so every page treats one
    exactly like a cache on disk -- curate it, thumbnail it, score it.

    `known` is the id -> Dataset map built from the on-disk datasets, and it is
    passed down rather than looked up: resolving a branch needs its parent's
    size, and asking the catalog for that from inside the catalog's own listing
    recurses until the stack dies. It was written that way first and did exactly
    that.

    Resolution is dependency-ordered, not file-ordered, because a branch may
    hang off another branch and the files are read by creation time. Each pass
    resolves whatever now has a known parent and adds it to the map; when a pass
    resolves nothing, whatever is left is genuinely unreachable -- a deleted
    parent or a loop -- and is listed at n=0 with the reason, because a branch
    that silently vanishes looks like data loss.
    """
    from tools.lab import subsets as sb

    pending = list(sb.load_all())
    out: List[Dataset] = []
    while pending:
        ready = [s for s in pending if s.parent in known]
        if not ready:
            break
        for sub in ready:
            try:
                n, note = len(sb.resolve_indices(sub, lookup=known)), sub.note
            except Exception as exc:                  # noqa: BLE001
                n, note = 0, f"unresolved: {type(exc).__name__}: {exc}"
            dataset = Dataset(id=sub.dataset_id, name=sub.name, kind="subset", n=n,
                              source=f"{sub.parent} · {sub.describe()}", note=note)
            known[dataset.id] = dataset
            out.append(dataset)
        pending = [s for s in pending if s not in ready]

    for sub in pending:
        out.append(Dataset(id=sub.dataset_id, name=sub.name, kind="subset", n=0,
                           source=f"{sub.parent} · {sub.describe()}",
                           note=f"parent {sub.parent} not found"))
    return out


def list_datasets() -> List[Dataset]:
    """Everything browsable right now. Cheap: headers and one small JSON."""
    out: List[Dataset] = []
    for discover in (_arm_datasets, _corpus_datasets, _split_datasets,
                     _raw_datasets):
        try:
            out.extend(discover())
        except Exception as exc:                      # never break the hub page
            print(f"[lab.catalog] {discover.__name__} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    try:
        out.extend(_subset_datasets({d.id: d for d in out}))
    except Exception as exc:                          # never break the hub page
        print(f"[lab.catalog] _subset_datasets failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
    out = [_flag_smoke(d) for d in out]
    out.sort(key=lambda d: (_KIND_ORDER.get(d.kind, 9), -d.n, d.id))
    return out


#: Appended to a tiny arm's note. Matched exactly rather than by substring:
#: several runs are *named* "..._live_smoke", so a `"smoke" in note` test
#: silently skipped flagging them.
SMOKE_NOTE = "smoke run"

#: Prefix of the note on a bench run's example builds (`examples_<max_dim>.npz`
#: beside its scorecard). It is a prefix rather than a whole note because the run
#: id follows it, and it is matched here so `_flag_smoke` can leave those caches
#: alone: the bench keeps `k = 8` builds per arm and `SMOKE_N` is 8, so a
#: single-arm run's examples would otherwise be labelled "smoke run (8 builds)",
#: which is a lie about a benchmark artifact rather than a hint about a scratch one.
EXAMPLES_NOTE = "bench examples"


def _flag_smoke(d: Dataset) -> Dataset:
    if d.note.startswith(EXAMPLES_NOTE):
        return d
    if d.kind == "arm" and d.n <= SMOKE_N and SMOKE_NOTE not in d.note:
        note = (d.note + " · " if d.note else "") + f"{SMOKE_NOTE} ({d.n} builds)"
        return Dataset(id=d.id, name=d.name, kind=d.kind, n=d.n,
                       source=d.source, note=note.strip())
    return d


def get_dataset(dataset_id: str) -> Optional[Dataset]:
    for d in list_datasets():
        if d.id == dataset_id:
            return d
    return None


#: How long the `source -> dataset id` map may be reused. `list_datasets()` walks
#: `outputs/`, opens a zip header per cache and indexes the raw corpora; the
#: leaderboard asks this question once per arm, sixteen times for one card, and
#: rebuilding discovery sixteen times to answer one request is the whole reason
#: this memo exists. Seconds rather than minutes so an arm generated while the
#: server is up still links from the next page load, and keyed on `OUTPUTS_ROOT`
#: so a test that repoints it does not read a map built from the real one.
_SOURCE_INDEX_TTL = 2.0
_SOURCE_INDEX: Optional[Tuple[str, float, Dict[str, str]]] = None


def _source_index() -> Dict[str, str]:
    global _SOURCE_INDEX
    now = time.monotonic()
    root = str(OUTPUTS_ROOT)
    if _SOURCE_INDEX is not None:
        cached_root, stamp, index = _SOURCE_INDEX
        if cached_root == root and now - stamp < _SOURCE_INDEX_TTL:
            return index
    index = {}
    # First writer wins, matching the page's `datasets.find(...)`: the listing is
    # sorted by kind, so a corpus beats an arm on the rare shared source string.
    for d in list_datasets():
        if d.source:
            index.setdefault(d.source, d.id)
    _SOURCE_INDEX = (root, now, index)
    return index


def source_dataset_id(source: str) -> Optional[str]:
    """The browsable dataset an arm's `meta.source` points at, or None.

    This is the join `leaderboard.html` used to do in JS by exact string equality
    against every `Dataset.source`, moved here so the page is handed an id rather
    than a search. Exact equality is kept on purpose: `meta.source` is written as
    `arm.npz or "in-memory"` and is byte-identical to the `source` discovery
    builds for the same file, so the match is either right or absent -- there is
    no fuzzy path resolution here that could quietly link a row to the wrong
    builds.

    Its measured coverage is the point of the fallbacks around it: 13 of 16 rows
    on one card and 8 of 12 on another are the literal `"in-memory"` and can
    never match, because nothing was ever written for them. That is a different
    answer from "the file moved", and `cards.arm_origin` is what tells them apart.
    """
    src = str(source or "").strip()
    if not src or src == "in-memory":
        return None
    return _source_index().get(src)


# --- loading ---------------------------------------------------------------
def _load_uncached(dataset_id: str) -> List[Structure]:
    from blockgen.curation.houses import load_structures_from_cache

    if dataset_id.startswith("split:"):
        from blockgen.eval.bench import splits
        _, corpus, side = dataset_id.split(":", 2)
        if side not in SPLIT_SIDES:
            raise ValueError(f"unknown split side {side!r}")
        # `split_structures` already crops; it is also the one place that knows
        # how index -> structure works for a split, so it is not reimplemented.
        return splits.split_structures(splits.load_split(corpus, 0), side)

    if dataset_id.startswith("subset:"):
        from tools.lab import subsets as sb
        sub = sb.get(dataset_id)
        if sub is None:
            raise KeyError(dataset_id)
        # Lazy over the parent: a branch costs its own length, not its parent's.
        return sb.SubsetBuilds(sub)

    if dataset_id.startswith("raw:"):
        from tools.lab import rawcorpora as rc
        # Lazy: a raw corpus is up to 28,235 builds and must never be
        # materialised to answer a question about one of them.
        return rc.LazyBuilds(dataset_id.split(":", 1)[1])

    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(dataset_id)
    structs = _read_npz_structures(Path(dataset.source))
    # Cropped, matching `ArmSpec.load` and `split_structures` -- so a build's
    # dims here are the dims the benchmark scored and the renderer draws.
    return [s.crop_to_non_air() for s in structs]


#: The two cache schemas in `data/minecraft/cache`, told apart by their keys.
#: This distinction is not cosmetic -- getting it wrong made five of the nine
#: corpora render as placeholder tiles, which looked like a broken renderer and
#: was actually a `KeyError: 'sources'` swallowed by the render guard.
#:
#:   house cache   block_ids, block_data, corpus, sources    houses_*, all_*
#:   build cache   block_ids, block_data, shapes, urls|paths gc_small_*, small_*,
#:                                                           tf_small_*
#:
#: `houses.load_structures_from_cache` reads only the first. The second is what
#: `data.build_cache` writes, and its loader is hardcoded to `small_{dim}.npz`,
#: so it cannot open `gc_small_32.npz` either -- neither existing reader covers
#: the whole cache directory, which is why this one dispatches.
_HOUSE_KEYS = frozenset({"sources"})
_BUILD_SOURCE_KEYS = ("urls", "paths")


def _sidecar_meta(npz: Path) -> Dict[str, dict]:
    """Per-build titles/tags from a `*_meta.json`, keyed by source url.

    The build-cache schema keeps no titles in the npz; the scrape metadata sits
    in a sibling file keyed by the same url the npz stores. Without this the
    older corpora show blank captions, which is the difference between a grid
    you can navigate and 5,866 anonymous tiles.
    """
    path = npz.with_name(npz.name[: -len(".npz")] + "_meta.json")
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return {}
    return blob if isinstance(blob, dict) else {}


def _read_npz_structures(npz: Path) -> List[Structure]:
    """Structures from either cache schema. See `_HOUSE_KEYS`."""
    with np.load(npz, allow_pickle=True) as blob:
        keys = set(blob.files)
        if _HOUSE_KEYS & keys:
            pass                              # handled below, outside the `with`
        else:
            src_key = next((k for k in _BUILD_SOURCE_KEYS if k in keys), None)
            ids, data = blob["block_ids"], blob["block_data"]
            sources = blob[src_key] if src_key else [""] * len(ids)
            meta = _sidecar_meta(npz)
            out: List[Structure] = []
            for i in range(len(ids)):
                src = str(sources[i]) if i < len(sources) else ""
                row = meta.get(src) or {}
                out.append(Structure(
                    block_ids=ids[i], block_data=data[i], source_path=src,
                    metadata={"title": str(row.get("title") or ""),
                              "category": str(row.get("subtitle") or ""),
                              "url": src}))
            return out
    # Imported here rather than at module scope: `curation.houses` pulls in the
    # corpora loaders, and paying that import cost on `import catalog` would
    # slow dataset discovery, which the contract requires to stay cheap.
    from blockgen.curation.houses import load_structures_from_cache

    structs, _ = load_structures_from_cache(str(npz))
    return structs


def load_builds(dataset_id: str) -> List[Structure]:
    """Cropped structures for a dataset, memoized in-process (LRU, small)."""
    if dataset_id in _BUILDS:
        _BUILDS.move_to_end(dataset_id)
        return _BUILDS[dataset_id]
    builds = _load_uncached(dataset_id)
    _BUILDS[dataset_id] = builds
    while len(_BUILDS) > _MAX_CACHED_DATASETS:
        _BUILDS.popitem(last=False)
    return builds


def forget(dataset_id: Optional[str] = None) -> None:
    """Drop memoized structures. For tests and for a rebuilt cache on disk."""
    if dataset_id is None:
        _BUILDS.clear()
    else:
        _BUILDS.pop(dataset_id, None)


def forget_scorecards() -> None:
    """Drop memoized scorecards (`_SCORECARDS`, declared with the read path).

    Named beside `forget` because it is the same gesture for the other memo, and
    tests need it: the scorecard cache is keyed on `(path, mtime)`, and a test
    that writes two different cards to the same tmp path inside one filesystem
    timestamp tick would otherwise read the first one twice.
    """
    _SCORECARDS.clear()


def build_row(dataset_id: str, index: int) -> dict:
    """The grid row for one build. Raises IndexError past the end."""
    builds = load_builds(dataset_id)
    if not 0 <= index < len(builds):
        raise IndexError(f"{dataset_id} has {len(builds)} builds, asked for {index}")
    s = builds[index]
    meta = s.metadata or {}
    return {
        "build_id": make_build_id(dataset_id, index),
        "dataset": dataset_id,
        "index": index,
        "dims": [int(v) for v in s.shape],
        "n_blocks": int(s.occupied_mask.sum()),
        "title": str(meta.get("title") or ""),
        "category": str(meta.get("category") or ""),
    }


# --- scorecards ------------------------------------------------------------
_SCORECARDS: Dict[Path, Tuple[float, dict]] = {}


def _read_scorecard(path: Path) -> dict:
    """Parse with a mtime-keyed memo: these run 90-390 KB and are re-read often.

    Normalisation (`cards.migrate`) happens *before* the memo insert, so it costs
    one pass per file per mtime and every reader in the process gets the same
    already-normalised blob -- there is no second, un-normalised shape anywhere
    for a caller to accidentally hold. That is the whole reason this is the
    chokepoint: `list_scorecards`, `load_scorecard` and the API all come through
    here, and none of them has to know which of `bench/1`'s four shapes it got.

    `migrate` returns a new top-level dict whose `arms` is the input's `arms` by
    identity, so what lands in the memo is normalised *and* still the parsed
    file's own metric blocks -- no copy of a 390 KB card, and no path by which a
    reader's enrichment can reach the parsed data.

    The try/except is not defensive dressing. A checkout of this file older than
    the card it is reading, or a `cards` module that a future schema confuses,
    must still open the run: falling back to the raw blob costs the `compat`
    block and the run-level defaults, which every consumer here already treats as
    optional, and it prints once so the degradation is visible rather than felt.
    """
    stamp = path.stat().st_mtime
    hit = _SCORECARDS.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    blob = json.loads(path.read_text())
    try:
        blob = cards.migrate(blob, path.parent.name, stamp)
    except Exception as exc:                          # never break the page
        print(f"[lab.catalog] could not normalise {path}: "
              f"{type(exc).__name__}: {exc}; reading it raw", flush=True)
    _SCORECARDS[path] = (stamp, blob)
    return blob


def _scorecard_paths() -> List[Path]:
    """Every `outputs/*/scorecard.json`, newest run first.

    Ordered by the `run_<YYYYMMDD_HHMMSS>_` stamp in the directory name, not by
    `st_mtime`, which is what this sorted by first. File modification time is not
    a property of the run: a `git checkout`, an rsync, a `touch`, a second writer
    dropping a `table.md` into a run dir -- any of them silently reordered the run
    picker, so the "newest" entry was whichever card the filesystem had been
    poked at last. All eleven directories on disk parse, so the ordering costs no
    JSON reads at all; `st_mtime` remains the fallback for a directory that does
    not follow the convention, which is the one case where it is genuinely the
    best guess available.

    Note the glob: `outputs/*/scorecard.json`, every directory, not
    `outputs/run_*_bench/`. Two of the cards on disk do not match the narrower
    pattern and are read anyway, which is the intended behaviour -- a run
    directory is whatever `--out` said it was.
    """
    if not OUTPUTS_ROOT.is_dir():
        return []
    return sorted((p for p in OUTPUTS_ROOT.glob("*/scorecard.json") if p.is_file()),
                  key=lambda p: cards.dir_stamp(p.parent.name) or _when(p),
                  reverse=True)


def _when(path: Path) -> str:
    from datetime import datetime, timezone
    return (datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def _arm_kind_counts(arms: dict) -> Dict[str, int]:
    """`{"submission": n, "control": n, "baseline": n}` over a card's arms.

    Exists because `n_arms` is a number no human recognises: the card everyone
    points at has sixteen arms and three results, the other thirteen rows being
    calibration controls and procedural baselines. The rule for which is which
    lives in `cards.arm_kind` and is not re-derived here.
    """
    counts = {kind: 0 for kind in cards.KINDS}
    if isinstance(arms, dict):
        for arm in arms.values():
            meta = arm.get("meta") if isinstance(arm, dict) else None
            counts[cards.arm_kind(meta or {})] += 1
    return counts


def list_scorecards() -> List[dict]:
    """Past benchmark runs, newest first. Empty list if none were ever run.

    `run`, `path`, `when`, `n_arms` and `corpus` are the original contract and
    keep their exact names and meanings -- `run` is still the directory name and
    still the only identity, doing all three of its jobs (picker value, `?run=`
    parameter, `/api/scorecard/<run>` path segment). Everything after them is
    additive and derived; every one of them falls back for the eleven cards that
    predate the field, so no row in this list is ever missing a key.

    `label` is display only and is deliberately not unique (`cards.run_label`).
    `git_dirty` is `None`, not `False`, when the card does not carry the flag: a
    run recorded before the flag existed does not know whether the tree was
    clean, and "unknown" must not render as "clean" next to a commit sha.
    """
    out: List[dict] = []
    for path in _scorecard_paths():
        try:
            blob = _read_scorecard(path)
        except Exception as exc:
            print(f"[lab.catalog] unreadable scorecard {path}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        ctx = blob.get("context") or {}
        run = blob.get("run") or {}
        arms = blob.get("arms") or {}
        counts = _arm_kind_counts(arms)
        run_id = path.parent.name
        out.append({"run": run_id, "path": str(path), "when": _when(path),
                    "n_arms": len(arms),
                    "corpus": ctx.get("corpus", ""),
                    "label": cards.run_label(blob, run_id),
                    "note": str(run.get("note") or ""),
                    "started": (run.get("started_at")
                                or cards.dir_stamp(run_id) or _when(path)),
                    "tier": str(run.get("tier") or ""),
                    "n_submissions": counts["submission"],
                    "n_controls": counts["control"],
                    "n_baselines": counts["baseline"],
                    "git_dirty": run.get("git_dirty"),
                    "has_examples": bool(run.get("examples"))})
    return out


def _metric_names(arms: dict) -> List[str]:
    """`section.key` for every leaf metric, in first-seen order.

    A metric is a dict carrying `direction` -- that is the invariant
    `scorecard.Metric` enforces (no bare floats), so it is a reliable way to tell
    a metric apart from the `meta` block or a coherence gen/real pair.
    """
    names: List[str] = []
    for arm in arms.values():
        if not isinstance(arm, dict):
            continue
        for section, block in arm.items():
            if not isinstance(block, dict):
                continue
            for key, leaf in block.items():
                if isinstance(leaf, dict) and "direction" in leaf:
                    name = f"{section}.{key}"
                    if name not in names:
                        names.append(name)
    return names


def load_scorecard(run: str) -> dict:
    """A parsed scorecard plus the keys the leaderboard page needs.

    Contract note: `/api/scorecard/<run>` is specified as "the parsed scorecard,
    plus {leaderboard, arms, metrics, head_to_head}", but the scorecard already
    has an `arms` key holding the per-arm metric blocks. Overwriting it with a
    list of names would throw away the page's actual payload, so `arms` is left
    as-is and the names are added as `arm_names`.

    `leaderboard` and `head_to_head` are lifted out of `run` where the bench
    runner writes them (`runner.py` -> `card.run["blockscore"]`), so the page does
    not have to know that.

    Everything added here is a NEW TOP-LEVEL KEY, and that is a rule rather than
    a coincidence -- see the comment at the shallow copy below. `label`, `note`
    and `started` are the run's identity as a human would state it; the three
    counts split `n_arms` into results and calibration; `compat` says which of
    `bench/1`'s shapes this card is and what it therefore cannot show. Per-arm
    derived data belongs in `arms_index`, built per request in `api`, for exactly
    the same reason.
    """
    candidates = [Path(run), Path(run) / "scorecard.json",
                  OUTPUTS_ROOT / run / "scorecard.json"]
    path = next((p for p in candidates if p.is_file() and p.name == "scorecard.json"),
                None)
    if path is None:
        raise FileNotFoundError(f"no scorecard for run {run!r}")

    # `blob["arms"]` is the object in the process-wide `_SCORECARDS` memo.
    # Enrich BESIDE `arms` (top-level keys, or `arms_index`), never inside it.
    # This is a shallow copy: only the top-level dict is ours to write into. One
    # `arms[name]["meta"]["kind"] = ...` here would be written into the cache
    # itself, permanently and cumulatively, and the next request would be served
    # a card that the file on disk does not contain.
    blob = dict(_read_scorecard(path))
    arms = blob.get("arms") or {}
    meta = blob.get("run") or {}
    compat = blob.get("compat")
    counts = _arm_kind_counts(arms)
    run_id = path.parent.name
    blob.update({
        "run_id": run_id,
        "path": str(path),
        "when": _when(path),
        "leaderboard": meta.get("blockscore") or [],
        "head_to_head": meta.get("head_to_head") or [],
        "arm_names": list(arms),
        "metrics": _metric_names(arms),
        "label": cards.run_label(blob, run_id),
        "note": str(meta.get("note") or ""),
        "started": (meta.get("started_at")
                    or cards.dir_stamp(run_id) or _when(path)),
        "n_submissions": counts["submission"],
        "n_controls": counts["control"],
        "n_baselines": counts["baseline"],
        # Normally written by `cards.migrate`; `{}` only if normalisation was
        # skipped, and the page must render a card without it either way.
        "compat": compat if isinstance(compat, dict) else {},
    })
    return blob


def arm_recipe(name: str, meta: dict) -> str:
    """One sentence saying how this arm's builds were made, or `""`.

    Prefers what the card recorded for itself: `bench/2` writes
    `meta.provenance.recipe` at scoring time, where how the arm was constructed
    is actually known. Falls back to the static table in `cards.RECIPES` for the
    cards that predate it -- which is every card on disk today, and is why the
    fallback exists at all: 13 of 16 rows on the newest of them are generated
    in-process and the page has nothing to say about them beyond "nothing on disk
    to show", while the sentence explaining what `real@solidify` *is* has been
    sitting in a docstring in the eval package the whole time.

    The preference order lives here rather than in `cards`, deliberately.
    `cards.RECIPES` is a lookup table that must not reach into a scorecard, so
    exactly one function in the lab knows that a written recipe beats a
    remembered one -- and when the recipes drift, the card wins.
    """
    meta = meta if isinstance(meta, dict) else {}
    provenance = meta.get("provenance")
    if isinstance(provenance, dict):
        recipe = provenance.get("recipe")
        if isinstance(recipe, str) and recipe.strip():
            return recipe
    return cards.RECIPES().get(str(name), "")


def example_build_ids(run_dir: Path, blob: dict) -> dict[str, list[str]]:
    """`{arm_name: [build_id, ...]}` for a run's example builds, else `{}`.

    A bench run writes the handful of builds it kept from each arm as one
    `examples_<max_dim>.npz` + `_manifest.json` pair in its own directory, and
    that pair is *already* a browsable dataset: `_arm_paths` walks every npz with
    a manifest sibling under `outputs/run_*/`, so the file is discovered with no
    change to discovery, and the ids minted here are the ones
    `/api/thumb/<build_id>` has always served. No new route, no image file in the
    run directory -- `api._static` resolves only under `tools/lab/static/`, so a
    PNG written beside a scorecard is not fetchable by the page at all.

    **Id derivation stays in `_arm_id`.** The bench records a run-dir-relative
    *path* and never a lab id, and the lab turns paths into ids in exactly one
    place, so the two sides cannot drift into two different slugging rules and
    orphan every label and 2AFC comparison keyed on the old one.

    Returns `{}` rather than ids that resolve to nothing when the npz is gone or
    has lost its manifest, and -- the case worth stating -- when the run
    directory is outside `outputs/` or outside the `ARM_ROOTS` globs discovery
    actually walks. `--out` may point anywhere; two cards on disk record an
    absolute scratchpad path. Discovery would never register such a file, so
    every id minted from it would 404 on the thumbnail route. Saying "no
    examples" is the honest answer, and the page still has its other fallbacks.
    """
    run = blob.get("run") if isinstance(blob, dict) else None
    examples = (run or {}).get("examples") if isinstance(run, dict) else None
    name = (examples or {}).get("npz") if isinstance(examples, dict) else None
    if not name:
        return {}

    path = Path(run_dir) / str(name)
    if not _openable(path):
        return {}
    try:
        rel = path.resolve().relative_to(OUTPUTS_ROOT.resolve())
    except ValueError:
        return {}
    # `rel.parts[0]` is the directory `_arm_paths` globs; anything deeper it
    # reaches by rglob. A file directly in `outputs/` is not reachable either.
    if len(rel.parts) < 2 or not any(Path(rel.parts[0]).match(g) for g in ARM_ROOTS):
        return {}

    dataset_id = _arm_id(path)
    total = _npz_rows(path)          # from the npy header; opens no structures
    out: dict[str, list[str]] = {}
    for arm, block in (blob.get("arms") or {}).items():
        meta = block.get("meta") if isinstance(block, dict) else None
        rows = ((meta or {}).get("examples") or {}).get("rows") or []
        ids: list[str] = []
        for row in rows:
            try:
                index = int(row)
            except (TypeError, ValueError):
                continue
            # A row past the end of the file is a card describing an npz that a
            # later run overwrote with fewer builds. Dropping it costs one
            # thumbnail; keeping it costs a broken image on every page load.
            if index < 0 or (total is not None and index >= total):
                continue
            ids.append(make_build_id(dataset_id, index))
        if ids:
            out[str(arm)] = ids
    return out
