"""Derived datasets: a named branch off any dataset the lab can see.

A subset is *not* a copy. It is `{parent, rule}` on disk in a few hundred bytes,
resolved to a list of parent indices on read. Materialising instead would mean a
new `.npz` per experiment -- `houses_32` is 50 MB, and the whole point of a
branch is that you make a dozen of them while you are deciding what the corpus
should be. It also means a subset stays honest when its parent is rebuilt: the
rule re-runs, rather than silently describing builds that no longer exist.

Two rules, because they answer different questions:

- **filter** is live. "Everything I labelled good", "grabcraft only", "a random
  256". Re-resolving after another labelling session picks up the new labels,
  which is what you want while curating.
- **indices** is frozen. An explicit list, captured once. This is what an
  experiment cites, because a live rule is not a reproducible sample -- an eval
  set that quietly grows between two runs makes those runs incomparable.

Nesting is allowed and cheap: a subset's parent may be another subset, so
`houses_32 -> good -> random 256` composes index maps without loading twice.

**Why JSON files and not the SQLite store.** `store.py` holds *observations* --
labels, notes, pairwise decisions -- which are append-only and only ever grow. A
subset is a *definition*: something you want to read, diff, hand-edit, copy to a
config, and commit next to the results it produced. It belongs with the export
files as a seam into the batch pipeline, not inside a database the lab owns.

Resolution cost is bounded by never loading structures when a cheaper source
exists. Label rules read the store; field rules read the corpus manifest, which
carries `corpus`/`category`/`n_blocks` per row without opening the `.npz`. Only
a field rule on a dataset with no manifest falls back to loading builds, and
that path refuses datasets past `_SCAN_LIMIT` rather than quietly spending
minutes walking a raw corpus off disk.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from tools.lab import LAB_ROOT

SUBSET_DIR = LAB_ROOT / "subsets"
ID_PREFIX = "subset:"

#: Loading builds purely to read their metadata is the slow path. Above this
#: many rows a field rule is refused with an explanation instead of running for
#: minutes -- `raw:all` is 59,387 builds, each a separate file read.
_SCAN_LIMIT = 20_000

_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

#: Every recognised filter key. Anything else in a rule is a typo, and a typo
#: that silently widens a selection is the failure mode worth spending an error
#: message on -- you would get a plausible number back and never look again.
FILTER_KEYS = ("labels", "exclude_labels", "corpus", "category", "title_contains",
               "min_blocks", "max_blocks", "max_dim", "limit", "seed")

LABEL_VALUES = ("good", "bad", "unsure", "unlabelled")


class SubsetError(ValueError):
    """A subset definition that cannot be stored or resolved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Subset:
    id: str
    name: str
    parent: str
    mode: str = "filter"
    rule: Dict[str, Any] = field(default_factory=dict)
    note: str = ""
    created_at: str = ""

    @property
    def dataset_id(self) -> str:
        return ID_PREFIX + self.id

    def to_json(self) -> dict:
        return {"id": self.id, "dataset_id": self.dataset_id, "name": self.name,
                "parent": self.parent, "mode": self.mode, "rule": dict(self.rule),
                "note": self.note, "created_at": self.created_at}

    def describe(self) -> str:
        """One line saying what this branch keeps. Shown under the tree node."""
        if self.mode == "indices":
            return f"frozen list of {len(self.rule.get('indices', []))}"
        bits: List[str] = []
        r = self.rule
        if r.get("labels"):
            bits.append("labelled " + "/".join(r["labels"]))
        if r.get("exclude_labels"):
            bits.append("not " + "/".join(r["exclude_labels"]))
        for key, label in (("corpus", "from"), ("category", "category")):
            if r.get(key):
                bits.append(f"{label} " + "/".join(r[key]))
        if r.get("title_contains"):
            bits.append(f"title ~ {r['title_contains']!r}")
        if r.get("min_blocks") is not None:
            bits.append(f"≥{r['min_blocks']} blocks")
        if r.get("max_blocks") is not None:
            bits.append(f"≤{r['max_blocks']} blocks")
        if r.get("max_dim") is not None:
            bits.append(f"fits {r['max_dim']}³")
        if r.get("limit") is not None:
            bits.append(f"random {r['limit']} (seed {r.get('seed', 0)})")
        return ", ".join(bits) if bits else "everything in the parent"


# --- store -----------------------------------------------------------------
def _path(subset_id: str) -> Path:
    return SUBSET_DIR / f"{subset_id}.json"


def load_all() -> List[Subset]:
    """Every subset on disk, oldest first. A corrupt file is skipped, not fatal:
    one bad hand-edit must not take the whole datasets page down."""
    if not SUBSET_DIR.is_dir():
        return []
    out: List[Subset] = []
    for path in sorted(SUBSET_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text())
            out.append(Subset(id=str(blob["id"]), name=str(blob.get("name") or blob["id"]),
                              parent=str(blob["parent"]), mode=str(blob.get("mode", "filter")),
                              rule=dict(blob.get("rule") or {}), note=str(blob.get("note") or ""),
                              created_at=str(blob.get("created_at") or "")))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    out.sort(key=lambda s: (s.created_at, s.id))
    return out


def get(subset_id: str) -> Optional[Subset]:
    """By bare id or by `subset:` dataset id."""
    want = subset_id[len(ID_PREFIX):] if subset_id.startswith(ID_PREFIX) else subset_id
    for s in load_all():
        if s.id == want:
            return s
    return None


def _validate(subset: Subset, known: Sequence[str]) -> None:
    if not _SLUG_OK.match(subset.id):
        raise SubsetError(f"id {subset.id!r} must be lowercase a-z0-9_- and start alphanumeric")
    if subset.mode not in ("filter", "indices"):
        raise SubsetError(f"unknown mode {subset.mode!r}")
    if subset.parent == subset.dataset_id:
        raise SubsetError("a subset cannot be its own parent")
    if known and subset.parent not in known:
        raise SubsetError(f"unknown parent dataset {subset.parent!r}")
    if subset.mode == "filter":
        bad = sorted(set(subset.rule) - set(FILTER_KEYS))
        if bad:
            raise SubsetError(f"unknown filter key(s): {', '.join(bad)}")
        for key in ("labels", "exclude_labels"):
            for value in subset.rule.get(key) or []:
                if value not in LABEL_VALUES:
                    raise SubsetError(f"{key}: {value!r} is not one of {', '.join(LABEL_VALUES)}")
        limit = subset.rule.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise SubsetError("limit must be a positive integer")
    else:
        idx = subset.rule.get("indices")
        if not isinstance(idx, list) or not idx:
            raise SubsetError("mode 'indices' needs a non-empty 'indices' list")
        if any(not isinstance(i, int) or i < 0 for i in idx):
            raise SubsetError("indices must be non-negative integers")


def _cycles(subset: Subset) -> None:
    """Walk parents to the root. A branch of a branch is normal; a loop is not,
    and a loop would hang resolution rather than fail it."""
    seen = {subset.dataset_id}
    parent = subset.parent
    while parent.startswith(ID_PREFIX):
        if parent in seen:
            raise SubsetError(f"parent chain loops at {parent}")
        seen.add(parent)
        nxt = get(parent)
        if nxt is None:
            raise SubsetError(f"parent {parent} does not exist")
        parent = nxt.parent


def save(subset: Subset, known: Sequence[str] = (), overwrite: bool = False) -> Subset:
    """Write a subset. `known` is the list of valid parent dataset ids; pass it
    from the catalog so a typo in a parent id fails here and not at browse time."""
    _validate(subset, known)
    _cycles(subset)
    path = _path(subset.id)
    if path.exists() and not overwrite:
        raise SubsetError(f"subset {subset.id!r} already exists")
    subset.created_at = subset.created_at or _now()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(subset.to_json(), indent=2) + "\n")
    tmp.replace(path)          # atomic: a half-written rule is a silent filter
    return subset


def delete(subset_id: str) -> bool:
    """Remove a subset. Children are left in place and will report a missing
    parent -- deleting them too would throw away rules you may want to re-point."""
    want = subset_id[len(ID_PREFIX):] if subset_id.startswith(ID_PREFIX) else subset_id
    path = _path(want)
    if not path.exists():
        return False
    path.unlink()
    return True


# --- resolution ------------------------------------------------------------
def _lookup(dataset_id: str, lookup: Optional[Dict[str, Any]]) -> Any:
    """The parent Dataset, from a caller-supplied map when there is one.

    The map matters: `catalog.list_datasets()` resolves subsets as part of
    building its own answer, so a subset that asked the catalog for its parent
    would re-enter that call and recurse until the stack died. Callers that are
    mid-listing pass what they already know instead.
    """
    if lookup is not None and dataset_id in lookup:
        return lookup[dataset_id]
    from tools.lab import catalog

    return catalog.get_dataset(dataset_id)


def _manifest_rows(dataset_id: str, lookup: Optional[Dict[str, Any]] = None) -> Optional[List[dict]]:
    """Per-build metadata without opening the `.npz`, when a manifest has it."""
    dataset = _lookup(dataset_id, lookup)
    if dataset is None or getattr(dataset, "kind", None) != "corpus":
        return None
    manifest = Path(dataset.source).with_name(Path(dataset.source).stem + "_manifest.json")
    if not manifest.is_file():
        return None
    try:
        items = json.loads(manifest.read_text()).get("items")
    except (OSError, ValueError):
        return None
    return items if isinstance(items, list) and items else None


def _scan_rows(dataset_id: str, n: int) -> List[dict]:
    """Fallback: read metadata off the structures themselves."""
    from tools.lab import catalog

    if n > _SCAN_LIMIT:
        raise SubsetError(
            f"{dataset_id} has {n:,} builds and no manifest, so a field filter would "
            f"have to open every one. Filter it by label, or branch a smaller parent.")
    rows: List[dict] = []
    for s in catalog.load_builds(dataset_id):
        meta = s.metadata or {}
        rows.append({"corpus": meta.get("corpus"), "category": meta.get("category"),
                     "title": meta.get("title"), "n_blocks": int(s.occupied_mask.sum()),
                     "dims": [int(v) for v in s.shape]})
    return rows


def _needs_rows(rule: Dict[str, Any]) -> bool:
    return any(rule.get(k) is not None for k in
               ("corpus", "category", "title_contains", "min_blocks", "max_blocks", "max_dim"))


def _in(value: Any, allowed: Optional[Sequence[str]]) -> bool:
    return not allowed or str(value or "") in set(allowed)


def resolve_indices(subset: Subset, parent_n: Optional[int] = None,
                    lookup: Optional[Dict[str, Any]] = None) -> List[int]:
    """Indices into `subset.parent` that this branch keeps, ascending.

    `lookup` is an id -> Dataset map for callers already holding one; without it
    the catalog is consulted, which is fine everywhere except from inside the
    catalog's own listing.
    """
    if parent_n is None:
        parent = _lookup(subset.parent, lookup)
        if parent is None:
            raise SubsetError(f"parent {subset.parent} is gone")
        parent_n = parent.n

    if subset.mode == "indices":
        return sorted(i for i in subset.rule.get("indices", []) if 0 <= i < parent_n)

    rule = subset.rule
    keep = list(range(parent_n))

    labels_in = rule.get("labels") or []
    labels_out = rule.get("exclude_labels") or []
    if labels_in or labels_out:
        from tools.lab.store import Store
        with Store() as store:
            marks = store.labels(subset.parent)
        def label_of(i: int) -> str:
            return marks.get(f"{subset.parent}:{i}", "unlabelled")
        if labels_in:
            keep = [i for i in keep if label_of(i) in set(labels_in)]
        if labels_out:
            keep = [i for i in keep if label_of(i) not in set(labels_out)]

    if _needs_rows(rule):
        rows = _manifest_rows(subset.parent, lookup)
        if rows is None:
            rows = _scan_rows(subset.parent, parent_n)
        lo, hi = rule.get("min_blocks"), rule.get("max_blocks")
        dim, sub = rule.get("max_dim"), rule.get("title_contains")
        picked: List[int] = []
        for i in keep:
            if i >= len(rows):
                continue
            row = rows[i]
            if not _in(row.get("corpus"), rule.get("corpus")):
                continue
            if not _in(row.get("category"), rule.get("category")):
                continue
            if sub and sub.lower() not in str(row.get("title") or "").lower():
                continue
            blocks = row.get("n_blocks")
            if lo is not None and (blocks is None or blocks < lo):
                continue
            if hi is not None and (blocks is None or blocks > hi):
                continue
            if dim is not None and max(row.get("dims") or [0]) > dim:
                continue
            picked.append(i)
        keep = picked

    limit = rule.get("limit")
    if limit is not None and len(keep) > limit:
        # Seeded and sorted, so the same rule yields the same sample on every
        # machine -- a random eval set that shifts per process is not a sample.
        keep = sorted(random.Random(int(rule.get("seed", 0))).sample(keep, limit))
    return keep


class SubsetBuilds:
    """List surface over the parent's builds, by index. Lazy for the same reason
    `rawcorpora.LazyBuilds` is: a branch of a 59k corpus must cost its own size,
    not its parent's."""

    def __init__(self, subset: Subset) -> None:
        from tools.lab import catalog

        self._subset = subset
        self._parent = catalog.load_builds(subset.parent)
        self._idx = resolve_indices(subset, len(self._parent))

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [self._parent[j] for j in self._idx[i]]
        return self._parent[self._idx[i]]

    def __iter__(self) -> Iterator[Any]:
        for j in self._idx:
            yield self._parent[j]

    @property
    def parent_indices(self) -> List[int]:
        """Where each row came from -- what you cite when exporting a branch."""
        return list(self._idx)
