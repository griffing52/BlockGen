"""The raw corpora, browsable without loading 59,000 builds into memory.

Why this is separate from `catalog`'s cache discovery
-----------------------------------------------------
`data/minecraft/cache/*.npz` holds *curated* sets — each one already filtered,
cropped and deduped by the batch pipeline, and each one overlapping the others.
Measured across the nine caches plus the three splits: **39,016 rows, 16,734
distinct builds**. `split:houses_32:*` is exactly `houses_32` re-partitioned,
`all_32` contains `houses_32`, and `gc_small_32` is the GrabCraft slice of both.
Browsing them tells you about the pipeline's output, not about what was thrown
away — and "I see a lot of bad builds" is a question about the input.

So this module indexes the **sources**: 6,560 GrabCraft artifacts, 2,537
3D-Craft houses, 11,092 text2mc h5 grids, 28,235 unconverted text2mc `.schem`,
and 10,963 legacy `.schematic` files. Nothing is curated, nothing is filtered,
and the junk is all still in there — which is the point.

Everything is lazy
------------------
The existing loaders (`corpora.load_3dcraft`, `load_text2mc`,
`load_grabcraft_structures`) return a fully-materialised list. That is right for
a training run and wrong for a browser: 59,000 builds will not fit in memory and
nobody wants to wait for all of them to look at twelve. Here a corpus is a
**list of paths**, walked once and cached to `outputs/lab/index/`, and a build
is read from disk only when someone asks for it. Discovery reads the cached
index length; it never touches a build.

The per-item readers are the repo's own, never reimplemented:
`grabcraft_dataset.structure_from_artifact`, `schem.schem_to_legacy`,
`Structure.from_schematic_path`, and `block_remap.remap_token_array`.

Two corpora need a note
-----------------------
**text2mc h5 stores its own token ids, not legacy `(id, data)`.** Its grids
index a 3,717-entry block-state vocabulary, so handing them to the renderer
unremapped paints every build in whatever legacy blocks those integers happen to
collide with — plausible-looking and completely wrong. `remap_token_array` with
`build_token_lut()` is applied on load, the same path `curation.houses` uses.

**`data/minecraft/more` is empty on this machine.** `data_sources.md` documents
a 36,290-record tfrecord crawl there; the directory exists and holds nothing, so
that corpus is simply absent rather than broken, and it is not registered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from blockgen.utils.data import Structure
from tools.lab import LAB_ROOT

#: Relative, matching `catalog.CORPUS_DIRS` and `LAB_ROOT`. The whole tool
#: already assumes it is run from the repo root (`python -m tools.lab`), and
#: keeping every root relative is what lets a test `chdir` into an empty
#: directory and see the fresh-clone world instead of this machine's data.
MC = Path("data") / "minecraft"
INDEX_DIR = LAB_ROOT / "index"

#: Cheap prefilter. A build with no blocks is a broken file, not a bad build,
#: and there is nothing to look at; text2mc ships 2,700+ of them (its own index
#: records `occ: 0`). Anything above this is shown however ugly -- filtering on
#: quality here would defeat the purpose of browsing the raw pool.
MIN_BLOCKS = 1


@dataclass(frozen=True)
class RawSpec:
    """One raw corpus: where it lives, how to enumerate it, how to read one."""
    id: str
    name: str
    root: Path
    pattern: str
    note: str
    reader: str                       # dispatch key for `_READERS`
    dirs: bool = False                # entries are directories, not files

    def available(self) -> bool:
        return self.root.is_dir()


SPECS: Tuple[RawSpec, ...] = (
    RawSpec("grabcraft", "GrabCraft (raw artifacts)", MC / "grabcraft" / "raw",
            "**/*.json", "category-labeled; exact (id,data)", "grabcraft"),
    RawSpec("3dcraft", "3D-Craft (raw houses)", MC / "3d_craft" / "houses",
            "**/schematic.npy", "human build traces; single class", "3dcraft",
            dirs=True),
    RawSpec("text2mc_h5", "text2mc (processed h5)", MC / "text2mc" / "processed_builds",
            "**/*.h5", "token grids, remapped to legacy on load", "text2mc_h5"),
    RawSpec("text2mc_schem", "text2mc (unconverted .schem)", MC / "text2mc",
            "**/*.schem", "the 28k the dataset author never converted",
            "text2mc_schem"),
    RawSpec("legacy_raw", "Legacy schematics", MC / "raw", "**/*.schematic",
            "drifted crawl; filenames are not metadata", "legacy"),
)

BY_ID: Dict[str, RawSpec] = {s.id: s for s in SPECS}


def specs() -> Tuple[RawSpec, ...]:
    """The registered corpora. A function, not the constant, so callers resolve
    `SPECS` at call time -- roots are relative, so what exists depends on cwd."""
    return SPECS

#: The union. Kept as a virtual corpus over the others' indices rather than an
#: index of its own, so it can never drift out of sync with its parts.
UNION_ID = "all"


# --- indexing ---------------------------------------------------------------
def _index_path(spec_id: str) -> Path:
    return INDEX_DIR / f"{spec_id}.json"


def build_index(spec: RawSpec, force: bool = False) -> List[str]:
    """Paths for one corpus, relative to its root. Walked once, then cached.

    Walking 28,235 `.schem` files takes seconds, which is fine once and not
    fine on every page load, so the result is cached to disk. Sorted, so a
    build's index is stable across machines and restarts -- an unstable index
    would silently reattach every label in the database to a different build,
    which is worse than losing them because the labels would still look valid.

    The sort is over `Path` objects, so it is component-wise:
    `cartoon-characters/ash.json` precedes `cartoon-characters-183/baymax.json`,
    where a plain string sort would reverse them ('-' sorts before '/'). Either
    is fine and neither is more correct; what matters is that it is the same
    every time, and that the hash index built alongside it shares this order.
    """
    path = _index_path(spec.id)
    if path.is_file() and not force:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass                       # unreadable cache: rebuild rather than die
    if not spec.available():
        return []
    hits = sorted(spec.root.glob(spec.pattern))
    if spec.dirs:
        rel = sorted({str(p.parent.relative_to(spec.root)) for p in hits})
    else:
        rel = [str(p.relative_to(spec.root)) for p in hits]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rel))
    return rel


def index(spec_id: str) -> List[str]:
    """Cached index for one corpus, or the concatenation for the union."""
    if spec_id == UNION_ID:
        out: List[str] = []
        for spec in SPECS:
            out.extend(f"{spec.id}/{rel}" for rel in index(spec.id))
        return out
    spec = BY_ID.get(spec_id)
    return build_index(spec) if spec else []


def counts() -> Dict[str, int]:
    """`{corpus_id: n}` for everything present, plus the union. Cheap after the
    first walk; the union is a sum, never a second index."""
    per = {s.id: len(build_index(s)) for s in SPECS if s.available()}
    if per:
        per[UNION_ID] = sum(per.values())
    return per


# --- per-item readers -------------------------------------------------------
_TOKEN_LUT: Optional[np.ndarray] = None


def _token_lut() -> np.ndarray:
    global _TOKEN_LUT
    if _TOKEN_LUT is None:
        from blockgen.utils.block_remap import build_token_lut
        _TOKEN_LUT = build_token_lut()
    return _TOKEN_LUT


def _read_grabcraft(path: Path) -> Optional[Structure]:
    from blockgen.data.grabcraft_dataset import (_artifact_metadata,
                                                 structure_from_artifact)
    art = json.loads(path.read_text())
    s = structure_from_artifact(art)
    if s is None:
        return None
    meta = dict(_artifact_metadata(art) or {})
    meta.setdefault("corpus", "grabcraft")
    return Structure(block_ids=s.block_ids, block_data=s.block_data,
                     source_path=str(path), metadata=meta)


def _read_3dcraft(path: Path) -> Optional[Structure]:
    sch = np.load(path / "schematic.npy")                 # (Y, Z, X, 2)
    if sch.ndim != 4 or sch.shape[-1] != 2:
        return None
    ids = sch[..., 0].transpose(2, 0, 1).astype(np.int32)  # -> (X, Y, Z)
    data = sch[..., 1].transpose(2, 0, 1).astype(np.int32)
    return Structure(block_ids=ids, block_data=data, source_path=str(path),
                     metadata={"corpus": "3dcraft", "category": "house",
                               "title": path.name})


def _read_text2mc_h5(path: Path) -> Optional[Structure]:
    import h5py
    from blockgen.utils.block_remap import remap_token_array
    with h5py.File(path, "r") as h:
        arr = np.asarray(h[list(h.keys())[0]])
    if arr.ndim != 3:
        return None
    ids, data = remap_token_array(arr, _token_lut())
    return Structure(block_ids=ids, block_data=data, source_path=str(path),
                     metadata={"corpus": "text2mc", "title": path.stem})


def _read_text2mc_schem(path: Path) -> Optional[Structure]:
    from blockgen.utils.schem import schem_to_legacy
    got = schem_to_legacy(path)
    if got is None:
        return None
    ids, data = got if isinstance(got, tuple) else (got, np.zeros_like(got))
    return Structure(block_ids=np.asarray(ids, np.int32),
                     block_data=np.asarray(data, np.int32),
                     source_path=str(path),
                     metadata={"corpus": "text2mc_schem", "title": path.stem})


def _read_legacy(path: Path) -> Optional[Structure]:
    return Structure.from_schematic_path(str(path))


_READERS: Dict[str, Callable[[Path], Optional[Structure]]] = {
    "grabcraft": _read_grabcraft,
    "3dcraft": _read_3dcraft,
    "text2mc_h5": _read_text2mc_h5,
    "text2mc_schem": _read_text2mc_schem,
    "legacy": _read_legacy,
}


def load_one(spec_id: str, i: int) -> Optional[Structure]:
    """Read build `i` of a raw corpus. `None` when the file is unreadable.

    Returns `None` rather than raising: these are unfiltered scrapes and a few
    thousand entries are genuinely broken (text2mc's own index records 2,700+
    with zero blocks). A broken file is a fact about the corpus that the grid
    should show as an empty tile, not an exception that kills the page.
    """
    if spec_id == UNION_ID:
        rel = index(UNION_ID)
        if not 0 <= i < len(rel):
            return None
        sub, _, tail = rel[i].partition("/")
        return _load_rel(BY_ID[sub], tail) if sub in BY_ID else None
    spec = BY_ID.get(spec_id)
    if spec is None:
        return None
    rel = build_index(spec)
    if not 0 <= i < len(rel):
        return None
    return _load_rel(spec, rel[i])


def _load_rel(spec: RawSpec, rel: str) -> Optional[Structure]:
    try:
        s = _READERS[spec.reader](spec.root / rel)
    except Exception:
        return None
    if s is None:
        return None
    try:
        s = s.crop_to_non_air()
    except Exception:
        return None
    if int(s.occupied_mask.sum()) < MIN_BLOCKS:
        return None
    return s


class LazyBuilds:
    """A read-only sequence of builds that reads each one on access.

    The rest of the lab talks to a dataset as `list[Structure]` -- `build_row`
    indexes it, `renders` indexes it, `api` slices it. Materialising 59,387 raw
    builds to satisfy that shape is not an option, so this presents the same
    surface and loads on `__getitem__` instead. Downstream code needs no change,
    which is the whole reason it is a sequence and not a new API.

    A small LRU matters more than it looks: a grid tile fetches the thumbnail and
    the row separately, so the same index is read at least twice in a row, and
    a `.schem` read is 30 ms.

    Slicing returns a plain list, because a slice is always something the caller
    is about to iterate, and iterating is exactly when laziness stops helping.
    """

    __slots__ = ("spec_id", "_n", "_cache", "_order", "_max")

    def __init__(self, spec_id: str, max_cached: int = 64) -> None:
        self.spec_id = spec_id
        self._n = len(index(spec_id))
        self._cache: Dict[int, Optional[Structure]] = {}
        self._order: List[int] = []
        self._max = max_cached

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(f"{self.spec_id} has {self._n} builds, asked for {i}")
        if i in self._cache:
            return self._cache[i]
        s = load_one(self.spec_id, i)
        # An unreadable entry caches as an empty structure rather than as None:
        # callers index this like a list and `None.shape` would crash a grid
        # over a corpus that is 24% broken files.
        if s is None:
            s = Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                          block_data=np.zeros((1, 1, 1), np.int32),
                          source_path="", metadata={"corpus": self.spec_id,
                                                    "unreadable": "1"})
        self._cache[i] = s
        self._order.append(i)
        while len(self._order) > self._max:
            self._cache.pop(self._order.pop(0), None)
        return s

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


# --- deduplication ----------------------------------------------------------
def content_hash(s: Structure) -> str:
    """Identity of a build's *content*, independent of where it came from.

    Blocks and their data, cropped, in XYZ order -- so the same build reached
    through two corpora hashes the same, and a build that merely sits at a
    different offset in its source file does not hash differently. Ignores
    metadata on purpose: two records of one build with different scraped titles
    are still one build.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(np.ascontiguousarray(s.block_ids, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(s.block_data, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def hash_index_path(spec_id: str) -> Path:
    return INDEX_DIR / f"{spec_id}.hashes.json"


def build_hash_index(spec_id: str, force: bool = False,
                     progress_every: int = 2000) -> List[str]:
    """Content hash per build, cached. `""` marks an unreadable entry.

    This is the expensive one -- every build in the corpus is decoded once -- so
    it is a separate, explicit pass rather than something a page load can
    trigger. `.schem` decoding dominates at ~30 ms/build.
    """
    path = hash_index_path(spec_id)
    if path.is_file() and not force:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    rel = index(spec_id)
    out: List[str] = []
    for i in range(len(rel)):
        s = load_one(spec_id, i)
        out.append(content_hash(s) if s is not None else "")
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[rawcorpora] {spec_id}: hashed {i + 1}/{len(rel)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def overlap_report(spec_ids: Optional[List[str]] = None) -> dict:
    """Distinct builds per corpus and pairwise sharing. Needs the hash index."""
    ids = spec_ids or [s.id for s in SPECS if s.available()]
    sets = {}
    rows = {}
    for sid in ids:
        hs = [h for h in build_hash_index(sid) if h]
        sets[sid] = set(hs)
        rows[sid] = {"listed": len(index(sid)), "readable": len(hs),
                     "distinct": len(sets[sid])}
    shared = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            n = len(sets[a] & sets[b])
            if n:
                shared[f"{a}|{b}"] = n
    union = set().union(*sets.values()) if sets else set()
    return {"per_corpus": rows, "shared": shared,
            "union_distinct": len(union),
            "total_listed": sum(r["listed"] for r in rows.values())}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Index and dedupe the raw corpora.")
    ap.add_argument("--hash", action="store_true", help="build the content-hash index")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="print the overlap report")
    args = ap.parse_args()

    print(json.dumps(counts(), indent=2))
    if args.hash:
        for spec in SPECS:
            if spec.available():
                build_hash_index(spec.id, force=args.force)
    if args.report:
        print(json.dumps(overlap_report(), indent=2))


if __name__ == "__main__":
    main()
