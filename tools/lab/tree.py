"""The datasets forest: which collection came from which.

The flat list this replaces was not merely harder to scan, it was wrong. It
showed `raw:all` beside the five raw corpora it *is*, and the three
`split:houses_32:*` beside the corpus they partition, so its "total builds"
counted ~70k things as ~178k. A tree fixes that by construction: a node's builds
are counted once, at the node, and a parent's total is its own.

**Every edge here is recorded on disk, never inferred from a name.** That is the
line that decides what belongs in the tree:

- `raw:all` -> the five raw corpora: `rawcorpora.SPECS` defines the union.
- corpus -> its splits: `splits.load_split` assigns the rows.
- run directory -> its arms: the arm's `.npz` is inside the run dir.
- subset -> its parent: written in the subset's own rule.

The one relationship that is *not* an edge is corpus provenance. `houses_32`
draws 1,360 builds from GrabCraft, 1,267 from 3D-Craft and 34 from text2mc --
recorded per row in its manifest, so it is known exactly, but it makes the
corpus a child of three parents at once. Drawing it as a tree edge would force a
choice between lying about the other two and duplicating the node. It is
attached as a `sources` annotation on the node instead, which says the true
thing in the space available.

Curated corpora also record what curation *dropped* (`houses_32`: 675 builds
with no interior, 52 single-material blobs, ...). That is carried as `drops`,
because "2,661 kept" only means something next to "3,493 pooled".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.lab import catalog, subsets

#: Order of the roots. Raw sources first because they are upstream of
#: everything; runs last because there are many and they are the most churn.
_ROOT_ORDER = {"raw": 0, "corpus": 1, "run": 2, "subset": 3}


@dataclass
class Node:
    id: str
    name: str
    kind: str
    n: int
    source: str = ""
    note: str = ""
    rule: str = ""
    sources: Dict[str, int] = field(default_factory=dict)
    drops: Dict[str, int] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)

    def to_json(self) -> dict:
        out = {"id": self.id, "name": self.name, "kind": self.kind, "n": self.n,
               "source": self.source, "note": self.note, "rule": self.rule,
               "children": [c.to_json() for c in self.children]}
        if self.sources:
            out["sources"] = self.sources
        if self.drops:
            out["drops"] = self.drops
        return out

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def _manifest(source: str) -> dict:
    """`<cache>.npz` -> `<cache>_manifest.json`, or `{}`."""
    path = Path(source)
    if path.suffix != ".npz":
        return {}
    sibling = path.with_name(path.stem + "_manifest.json")
    if not sibling.is_file():
        return {}
    try:
        blob = json.loads(sibling.read_text())
    except (OSError, ValueError):
        return {}
    return blob if isinstance(blob, dict) else {}


def _provenance(source: str) -> tuple[Dict[str, int], Dict[str, int]]:
    """(kept-per-source, dropped-per-reason) for a curated corpus.

    `report.by_corpus` counts the *pooled* input, before quality gates, so it
    overstates what is actually in the cache. The per-row `items` are the kept
    builds, so they are counted instead when present.
    """
    blob = _manifest(source)
    if not blob:
        return {}, {}
    drops = {str(k): int(v) for k, v in
             (blob.get("report", {}).get("quality_drops") or {}).items()}
    items = blob.get("items")
    if isinstance(items, list) and items:
        kept: Dict[str, int] = {}
        for row in items:
            key = str((row or {}).get("corpus") or "unknown")
            kept[key] = kept.get(key, 0) + 1
        return dict(sorted(kept.items(), key=lambda kv: -kv[1])), drops
    return {}, drops


def _run_group(dataset_id: str, source: str) -> Optional[tuple[str, str]]:
    """(group id, display name) for the run directory an arm was written into.

    `outputs/bench_arms/x.npz` and `outputs/run_<stamp>_<name>/x.npz` are the two
    shapes; anything else keeps its own root rather than being forced into one.
    """
    parent = Path(source).parent
    if parent.name and parent.parent.name == "outputs":
        return f"run:{parent.name}", parent.name
    return None


def build() -> List[Node]:
    """The forest, roots in display order, each node counted exactly once."""
    datasets = {d.id: d for d in catalog.list_datasets()}
    nodes: Dict[str, Node] = {}
    for d in datasets.values():
        if d.kind == "subset":
            continue          # `_attach_subsets` owns these; it has the rule text
        node = Node(id=d.id, name=d.name, kind=d.kind, n=d.n,
                    source=d.source, note=d.note)
        if d.kind == "corpus":
            node.sources, node.drops = _provenance(d.source)
        nodes[d.id] = node

    roots: List[Node] = []
    groups: Dict[str, Node] = {}

    for d in datasets.values():
        node = nodes.get(d.id)
        if node is None:
            continue                                   # a subset; handled below
        parent_id: Optional[str] = None

        if d.kind == "raw" and d.id != "raw:all" and "raw:all" in nodes:
            parent_id = "raw:all"
        elif d.kind == "split":
            _, corpus, _side = d.id.split(":", 2)
            parent_id = f"corpus:{corpus}" if f"corpus:{corpus}" in nodes else None
        elif d.kind == "arm":
            group = _run_group(d.id, d.source)
            if group is not None:
                gid, gname = group
                if gid not in groups:
                    groups[gid] = Node(id=gid, name=gname, kind="run", n=0,
                                       source=str(Path(d.source).parent))
                groups[gid].children.append(node)
                continue

        if parent_id is not None:
            nodes[parent_id].children.append(node)
        else:
            roots.append(node)

    for group in groups.values():
        # A run's total is the arms inside it -- the dir itself holds no builds.
        group.n = sum(child.n for child in group.children)
        group.children.sort(key=lambda c: (-c.n, c.id))
        roots.append(group)

    _attach_subsets(nodes, roots, datasets)

    roots.sort(key=lambda node: (_ROOT_ORDER.get(node.kind, 9), -node.n, node.id))
    for node in roots:
        node.children.sort(key=lambda c: (_ROOT_ORDER.get(c.kind, 9), -c.n, c.id))
    return roots


def _attach_subsets(nodes: Dict[str, Node], roots: List[Node],
                    datasets: Dict[str, Any]) -> None:
    """Hang saved branches under their parents.

    Counts come from `datasets`, which the catalog has already resolved --
    re-resolving here would double the work and, worse, could disagree with the
    number the rest of the app shows for the same branch.

    Two passes, because a subset may be the parent of another subset and the
    files are read in creation order, not tree order. An orphan -- its parent
    deleted -- becomes a root carrying the reason rather than vanishing: a rule
    you can see is a rule you can re-point.
    """
    pending: List[tuple[Node, str]] = []
    for sub in subsets.load_all():
        known = datasets.get(sub.dataset_id)
        node = Node(id=sub.dataset_id, name=sub.name, kind="subset",
                    n=getattr(known, "n", 0), source=str(subsets._path(sub.id)),
                    note=getattr(known, "note", "") or sub.note, rule=sub.describe())
        nodes[node.id] = node
        pending.append((node, sub.parent))

    for node, parent_id in pending:
        target = nodes.get(parent_id)
        if target is None:
            node.note = node.note or f"parent {parent_id} not found"
            roots.append(node)
        else:
            target.children.append(node)


def totals(roots: List[Node]) -> dict:
    """Headline counts for the hub.

    `builds` deliberately excludes any node contained in another -- splits
    partition their corpus, `raw:all` is its five children, a subset is part of
    its parent -- so the number is builds on disk, not rows in a table.
    """
    contained = {"split", "subset"}
    total = 0
    n_nodes = 0
    for root in roots:
        for node in root.walk():
            n_nodes += 1
        total += root.n if root.kind != "run" else sum(c.n for c in root.children)
    leaves = sum(1 for root in roots for node in root.walk()
                 if node.kind not in contained and node.kind != "run")
    return {"datasets": n_nodes, "roots": len(roots), "builds": total,
            "collections": leaves}
