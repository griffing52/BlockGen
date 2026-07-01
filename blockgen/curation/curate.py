"""Dataset curation tool: sort, filter, group, dedupe, and flag structures.

The scraped dataset is noisy — duplicates, tiny fragments, single-material blobs,
oversized builds. This module helps you *curate* it: compute per-structure features,
slice the set by any of them, group similar structures (by shape IoU or by feature
clusters), surface near-duplicates, and record keep/remove decisions that persist to
JSON so the choices survive across sessions.

Typical notebook workflow::

    from blockgen.curation import Curator
    cur = Curator.from_cache(max_dim=24)
    cur.summary()                                  # aggregate stats
    cur.table(sort_by="n_blocks", limit=20)        # peek at the biggest

    dups = cur.find_duplicates(iou_threshold=0.95) # near-identical groups
    cur.show_group(dups[0])                         # render one group
    cur.mark_remove(dups[0][1:], reason="duplicate")

    tiny = cur.filter(max_blocks=12).indices
    cur.mark_remove(tiny, reason="too small")

    cur.save_decisions()                            # persist
    kept = cur.apply()                              # structures minus removals

Everything is numpy + matplotlib only (no sklearn/scipy/pandas/widgets).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from blockgen.data.build_cache import DEFAULT_CACHE_DIR, load_cached_structures
from blockgen.utils.data import Structure, _resource_location_for
from blockgen.utils.serialize import BlockVocab, build_block_vocab

# Metadata feature keys attached to every row (defaults when a structure has none).
_META_DEFAULTS = {
    "title": "", "category": "", "tags": (), "description": "", "user": "",
    "views": 0, "downloads": 0, "diamonds": 0, "favorites": 0, "comments": 0,
    "has_metadata": False,
}


_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

DEFAULT_DECISIONS_PATH = str(Path(DEFAULT_CACHE_DIR) / "curation_decisions.json")


# --- connectivity ----------------------------------------------------------
def _connected_components(occ: np.ndarray) -> tuple:
    """Return (n_components, largest_fraction) for a 3D boolean occupancy array."""
    coords = np.argwhere(occ)
    if coords.shape[0] == 0:
        return 0, 0.0
    occ_set = {tuple(c) for c in coords.tolist()}
    seen: set = set()
    sizes: List[int] = []
    for start in occ_set:
        if start in seen:
            continue
        comp = 0
        stack = [start]
        seen.add(start)
        while stack:
            x, y, z = stack.pop()
            comp += 1
            for dx, dy, dz in _NEIGHBORS:
                p = (x + dx, y + dy, z + dz)
                if p in occ_set and p not in seen:
                    seen.add(p)
                    stack.append(p)
        sizes.append(comp)
    return len(sizes), max(sizes) / float(len(occ_set))


# --- per-structure features ------------------------------------------------
def compute_features(structures: Sequence[Structure]) -> List[dict]:
    """Compute a feature dict per structure (operates on the cropped bounding box)."""
    rows: List[dict] = []
    for i, s in enumerate(structures):
        c = s.crop_to_non_air()
        occ = c.occupied_mask
        n_blocks = int(occ.sum())
        sx, sy, sz = c.shape
        bbox_vol = sx * sy * sz
        ids = c.block_ids[occ]
        datas = c.block_data[occ]

        # Block-type composition (by resource location name).
        names: Dict[str, int] = {}
        for bid, bdata in zip(ids.tolist(), datas.tolist()):
            name = _resource_location_for(int(bid), int(bdata))
            names[name] = names.get(name, 0) + 1
        dominant_name, dominant_count = (
            max(names.items(), key=lambda kv: kv[1]) if names else ("minecraft:air", 0)
        )
        n_components, largest_frac = _connected_components(occ)
        # Exact (id, data) palette: distinguishes material/color variations of the
        # same shape (e.g. oak vs spruce planks share a resource name but differ here).
        palette_sig = frozenset(
            (int(bid), int(bdata)) for bid, bdata in zip(ids.tolist(), datas.tolist())
        )

        row = {
            "index": i,
            "path": os.path.basename(s.source_path) if s.source_path else f"#{i}",
            "source_path": s.source_path or "",
            "sx": sx, "sy": sy, "sz": sz,
            "max_dim": max(sx, sy, sz),
            "n_blocks": n_blocks,
            "bbox_volume": bbox_vol,
            "density": round(n_blocks / bbox_vol, 4) if bbox_vol else 0.0,
            "footprint": sx * sz,
            "height": sy,
            "n_block_types": len(names),
            "dominant_block": dominant_name,
            "dominant_frac": round(dominant_count / n_blocks, 3) if n_blocks else 0.0,
            "n_components": n_components,
            "largest_component_frac": round(largest_frac, 3),
            "palette_sig": palette_sig,
        }
        row.update(_META_DEFAULTS)
        rows.append(row)
    return rows


def attach_metadata(features: List[dict], meta_by_url: Dict[str, dict]) -> int:
    """Fill metadata fields on feature rows from a url->record map (keyed by source_path).

    Returns the number of rows that got metadata. ``subtitle`` becomes ``category``.
    """
    n = 0
    for r in features:
        m = meta_by_url.get(r["source_path"])
        if not m:
            continue
        n += 1
        r["title"] = m.get("title", "") or ""
        r["category"] = m.get("subtitle", "") or ""
        r["description"] = m.get("description", "") or ""
        r["user"] = m.get("user", "") or ""
        r["tags"] = tuple(m.get("tags", []) or ())
        r["views"] = int(m.get("views", 0) or 0)
        r["downloads"] = int(m.get("downloads", 0) or 0)
        r["diamonds"] = int(m.get("diamondCount", 0) or 0)
        r["favorites"] = int(m.get("favorites", 0) or 0)
        r["comments"] = int(m.get("comments", 0) or 0)
        r["has_metadata"] = True
    return n


# --- union-find for grouping -----------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            buckets.setdefault(self.find(i), []).append(i)
        return list(buckets.values())


@dataclass
class Curator:
    """Holds structures + features and supports filtering, grouping, and flagging.

    A filtered ``Curator`` is a *view*: ``self.indices`` maps its rows back to the
    positions in the original ``structures`` list it was created from, so decisions
    always reference the same underlying structures.
    """

    structures: List[Structure]
    features: List[dict]
    indices: List[int] = field(default_factory=list)
    decisions: Dict[str, dict] = field(default_factory=dict)
    vocab: Optional[BlockVocab] = None
    metadata: Dict[str, dict] = field(default_factory=dict)
    _grid_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.indices:
            self.indices = list(range(len(self.structures)))

    # --- construction ------------------------------------------------------
    @classmethod
    def from_cache(cls, max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR) -> "Curator":
        structures = load_cached_structures(max_dim=max_dim, cache_dir=cache_dir)
        vocab = build_block_vocab(structures, max_dim=max_dim)
        return cls(structures=structures, features=compute_features(structures), vocab=vocab)

    @classmethod
    def from_labeled_cache(cls, max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR) -> "Curator":
        """Load the metadata-labeled tfrecord cache (titles, categories, tags, popularity)."""
        from blockgen.data.tfrecord_dataset import load_labeled_structures
        structures, meta = load_labeled_structures(max_dim=max_dim, cache_dir=cache_dir)
        vocab = build_block_vocab(structures, max_dim=max_dim)
        features = compute_features(structures)
        n = attach_metadata(features, meta)
        print(f"loaded {len(structures)} structures, {n} with metadata "
              f"({100 * n / max(1, len(structures)):.0f}%)")
        return cls(structures=structures, features=features, vocab=vocab, metadata=meta)

    @classmethod
    def from_grabcraft_cache(cls, max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR) -> "Curator":
        """Load the scraped-GrabCraft cache (titles, subcategory, tags, views)."""
        from blockgen.data.grabcraft_dataset import load_grabcraft_structures
        structures, meta = load_grabcraft_structures(max_dim=max_dim, cache_dir=cache_dir)
        vocab = build_block_vocab(structures, max_dim=max_dim)
        features = compute_features(structures)
        n = attach_metadata(features, meta)
        print(f"loaded {len(structures)} structures, {n} with metadata "
              f"({100 * n / max(1, len(structures)):.0f}%)")
        return cls(structures=structures, features=features, vocab=vocab, metadata=meta)

    @classmethod
    def from_structures(cls, structures: Sequence[Structure], max_dim: int = 24,
                        metadata: Optional[Dict[str, dict]] = None) -> "Curator":
        structs = list(structures)
        vocab = build_block_vocab(structs, max_dim=max_dim)
        features = compute_features(structs)
        if metadata:
            attach_metadata(features, metadata)
        return cls(structures=structs, features=features, vocab=vocab, metadata=metadata or {})

    def _view(self, indices: Sequence[int]) -> "Curator":
        """A child view sharing the same structures, vocab, and decisions dict."""
        child = Curator(
            structures=self.structures,
            features=self.features,
            indices=list(indices),
            decisions=self.decisions,  # shared by reference so flags persist
            vocab=self.vocab,
            metadata=self.metadata,
            _grid_cache=self._grid_cache,
        )
        return child

    def __len__(self) -> int:
        return len(self.indices)

    def rows(self) -> List[dict]:
        return [self.features[i] for i in self.indices]

    # --- reporting ---------------------------------------------------------
    def summary(self) -> None:
        rows = self.rows()
        n = len(rows)
        print(f"structures: {n}")
        if n == 0:
            return
        def col(k):
            return np.array([r[k] for r in rows], dtype=float)
        for k in ["n_blocks", "max_dim", "density", "n_block_types", "n_components", "largest_component_frac"]:
            v = col(k)
            print(f"  {k:24s} min {v.min():7.2f}  median {np.median(v):7.2f}  "
                  f"p90 {np.percentile(v,90):7.2f}  max {v.max():7.2f}")
        # most common dominant materials
        from collections import Counter
        dom = Counter(r["dominant_block"] for r in rows).most_common(8)
        print("  top dominant materials:", ", ".join(f"{name.split(':')[-1]}({c})" for name, c in dom))
        # metadata, if present
        n_meta = sum(1 for r in rows if r["has_metadata"])
        if n_meta:
            print(f"  with metadata: {n_meta}/{n}")
            cats = Counter(r["category"] for r in rows if r["category"]).most_common(8)
            print("  top categories:", ", ".join(f"{c}({n})" for c, n in cats))
        n_dec = sum(1 for r in rows if r["source_path"] in self.decisions)
        print(f"  flagged so far: {n_dec}  (remove={len(self.remove_list())}, keep={len(self.keep_list())})")

    def table(
        self,
        indices: Optional[Sequence[int]] = None,
        sort_by: Optional[str] = None,
        reverse: bool = True,
        limit: int = 30,
        columns: Sequence[str] = ("index", "path", "sx", "sy", "sz", "n_blocks",
                                  "density", "n_block_types", "n_components", "dominant_block"),
    ) -> None:
        idx = list(indices) if indices is not None else list(self.indices)
        rows = [self.features[i] for i in idx]
        if sort_by:
            rows.sort(key=lambda r: r[sort_by], reverse=reverse)
        rows = rows[:limit]
        cols = list(columns)
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else (v.split(":")[-1] if isinstance(v, str) and ":" in v else str(v))
        widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) if rows else len(c) for c in cols}
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-+-".join("-" * widths[c] for c in cols))
        for r in rows:
            print(" | ".join(fmt(r[c]).ljust(widths[c]) for c in cols))

    # --- filtering / sorting ----------------------------------------------
    def filter(
        self,
        *,
        min_blocks: Optional[int] = None,
        max_blocks: Optional[int] = None,
        min_dim: Optional[int] = None,
        max_dim: Optional[int] = None,
        min_density: Optional[float] = None,
        max_density: Optional[float] = None,
        min_block_types: Optional[int] = None,
        max_block_types: Optional[int] = None,
        max_components: Optional[int] = None,
        min_largest_component_frac: Optional[float] = None,
        dominant_in: Optional[Sequence[str]] = None,     # keep if dominant material matches any substring
        contains_block: Optional[str] = None,            # substring match on dominant material
        # --- metadata criteria (require from_labeled_cache) ---
        category_in: Optional[Sequence[str]] = None,     # exact category (subtitle) match, any of
        title_contains: Optional[str] = None,            # case-insensitive substring of the title
        tag_contains: Optional[str] = None,              # case-insensitive substring of any tag
        min_downloads: Optional[int] = None,
        min_diamonds: Optional[int] = None,
        min_views: Optional[int] = None,
        has_metadata: Optional[bool] = None,
        predicate=None,                                  # custom: fn(feature_row) -> bool
    ) -> "Curator":
        """Return a filtered view. All criteria are ANDed; None means 'no constraint'."""
        tl = title_contains.lower() if title_contains else None
        gl = tag_contains.lower() if tag_contains else None
        cats = set(category_in) if category_in else None
        keep: List[int] = []
        for i in self.indices:
            r = self.features[i]
            if min_blocks is not None and r["n_blocks"] < min_blocks: continue
            if max_blocks is not None and r["n_blocks"] > max_blocks: continue
            if min_dim is not None and r["max_dim"] < min_dim: continue
            if max_dim is not None and r["max_dim"] > max_dim: continue
            if min_density is not None and r["density"] < min_density: continue
            if max_density is not None and r["density"] > max_density: continue
            if min_block_types is not None and r["n_block_types"] < min_block_types: continue
            if max_block_types is not None and r["n_block_types"] > max_block_types: continue
            if max_components is not None and r["n_components"] > max_components: continue
            if min_largest_component_frac is not None and r["largest_component_frac"] < min_largest_component_frac: continue
            if dominant_in is not None and not any(d in r["dominant_block"] for d in dominant_in): continue
            if contains_block is not None and contains_block not in r["dominant_block"]: continue
            if cats is not None and r["category"] not in cats: continue
            if tl is not None and tl not in r["title"].lower(): continue
            if gl is not None and not any(gl in t.lower() for t in r["tags"]): continue
            if min_downloads is not None and r["downloads"] < min_downloads: continue
            if min_diamonds is not None and r["diamonds"] < min_diamonds: continue
            if min_views is not None and r["views"] < min_views: continue
            if has_metadata is not None and r["has_metadata"] != has_metadata: continue
            if predicate is not None and not predicate(r): continue
            keep.append(i)
        return self._view(keep)

    def search(self, text: str) -> "Curator":
        """Full-text view: case-insensitive match across title, description, and tags."""
        q = text.lower()
        keep = [
            i for i in self.indices
            if q in self.features[i]["title"].lower()
            or q in self.features[i]["description"].lower()
            or any(q in t.lower() for t in self.features[i]["tags"])
        ]
        return self._view(keep)

    def sort(self, key: str, reverse: bool = True) -> "Curator":
        ordered = sorted(self.indices, key=lambda i: self.features[i][key], reverse=reverse)
        return self._view(ordered)

    # --- shape voxelization (cached) --------------------------------------
    def _occupancy(self, grid: int = 24) -> np.ndarray:
        """Flattened occupancy grids for ALL structures (cached, translation-tolerant)."""
        from blockgen.eval.novelty import voxelize_occupancy
        if self._grid_cache is None or self._grid_cache.shape[1] != grid ** 3:
            occ, _ = voxelize_occupancy(self.structures, grid, self.vocab)
            self._grid_cache = occ
        return self._grid_cache

    # --- grouping ----------------------------------------------------------
    @staticmethod
    def _iou_matrix_fast(occ: np.ndarray) -> np.ndarray:
        """Pairwise occupancy IoU. Uses the GPU (torch) when available, else numpy."""
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.from_numpy(occ.astype(np.float32)).to(dev)
            inter = a @ a.T
            s = a.sum(1, keepdim=True)
            union = s + s.T - inter
            iou = torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(inter))
            return iou.cpu().numpy()
        except Exception:
            from blockgen.eval.novelty import _iou_matrix
            return _iou_matrix(occ, occ)

    def group_by_similarity(self, iou_threshold: float = 0.6, grid: int = 24) -> List[List[int]]:
        """Group structures whose occupancy IoU exceeds a threshold (union-find).

        Returns groups as lists of ORIGINAL indices, largest group first.
        """
        idx = self.indices
        occ = self._occupancy(grid)[idx]
        iou = self._iou_matrix_fast(occ)
        uf = _UnionFind(len(idx))
        # Vectorized: take only upper-triangle pairs above threshold.
        pairs = np.argwhere(np.triu(iou >= iou_threshold, k=1))
        for a, b in pairs:
            uf.union(int(a), int(b))
        groups = [[idx[j] for j in g] for g in uf.groups()]
        groups.sort(key=len, reverse=True)
        return groups

    def find_duplicates(self, iou_threshold: float = 0.95, grid: int = 24) -> List[List[int]]:
        """Near-duplicate groups (high IoU). Returns only groups with >1 member."""
        return [g for g in self.group_by_similarity(iou_threshold, grid) if len(g) > 1]

    def _split_by_palette(self, group: Sequence[int]) -> List[List[int]]:
        """Split a shape group into sub-groups that share an identical (id,data) palette."""
        buckets: Dict[frozenset, List[int]] = {}
        for i in group:
            buckets.setdefault(self.features[i]["palette_sig"], []).append(i)
        return list(buckets.values())

    def find_exact_duplicates(self, iou_threshold: float = 0.95, grid: int = 24) -> List[List[int]]:
        """Same shape AND identical material palette — true copies, safe to drop extras.

        Returns sub-groups (size > 1) whose members share both occupancy (IoU >=
        threshold) and the exact same set of (block_id, data) materials.
        """
        out: List[List[int]] = []
        for g in self.group_by_similarity(iou_threshold, grid):
            if len(g) < 2:
                continue
            for sub in self._split_by_palette(g):
                if len(sub) > 1:
                    out.append(sub)
        out.sort(key=len, reverse=True)
        return out

    def find_variant_groups(self, iou_threshold: float = 0.9, grid: int = 24) -> List[List[int]]:
        """Same shape, DIFFERENT materials — color/wood variations worth KEEPING.

        Returns shape groups that contain at least two distinct material palettes
        (e.g. the same build in oak, spruce, and birch, or recolored wool).
        """
        out: List[List[int]] = []
        for g in self.group_by_similarity(iou_threshold, grid):
            if len(g) < 2:
                continue
            if len({self.features[i]["palette_sig"] for i in g}) > 1:
                out.append(g)
        out.sort(key=len, reverse=True)
        return out

    def dedupe_keep_variants(self, iou_threshold: float = 0.95, grid: int = 24,
                             reason: str = "exact-duplicate") -> int:
        """Flag exact-duplicate extras for removal while preserving material variants.

        Within each shape group, keeps one structure per distinct palette (so every
        color/wood variation survives) and marks the remaining true copies 'remove'.
        Returns the number flagged.
        """
        to_remove: List[int] = []
        for sub in self.find_exact_duplicates(iou_threshold, grid):
            to_remove.extend(sub[1:])  # keep the first of each identical-palette copy
        self.mark_remove(to_remove, reason=reason)
        return len(to_remove)

    def cluster_features(self, k: int = 8, seed: int = 0, iters: int = 50) -> List[List[int]]:
        """K-means over standardized features. Returns k groups of ORIGINAL indices."""
        idx = self.indices
        keys = ["n_blocks", "max_dim", "density", "footprint", "height",
                "n_block_types", "largest_component_frac", "dominant_frac"]
        X = np.array([[self.features[i][key] for key in keys] for i in idx], dtype=float)
        X[:, 0] = np.log1p(X[:, 0])  # block count is heavy-tailed
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X = (X - mu) / sd
        rng = np.random.default_rng(seed)
        k = min(k, len(idx))
        centers = X[rng.choice(len(idx), k, replace=False)]
        labels = np.zeros(len(idx), dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for c in range(k):
                pts = X[labels == c]
                if len(pts):
                    centers[c] = pts.mean(0)
        return [[idx[j] for j in np.where(labels == c)[0]] for c in range(k)]

    # --- decisions / flagging ---------------------------------------------
    def mark(self, indices: Sequence[int], decision: str, reason: str = "",
             tags: Optional[Sequence[str]] = None) -> None:
        for i in indices:
            path = self.structures[i].source_path or f"#{i}"
            self.decisions[path] = {"decision": decision, "reason": reason,
                                    "tags": list(tags or [])}

    def auto_mark_reliable(self, min_diamonds: int = 5, min_downloads: int = 50,
                           require_both: bool = False, reason: str = "popular") -> int:
        """Mark popular builds 'keep' as a likely-good seed subset.

        Popularity (diamonds / downloads) is a crowd-sourced quality signal: highly
        favorited builds are usually clean, complete, and well-formed. Marks the
        matching structures 'keep' and returns the count. With ``require_both`` the
        structure must clear *both* thresholds; otherwise *either* qualifies.
        """
        hits: List[int] = []
        for i in self.indices:
            r = self.features[i]
            d_ok = r["diamonds"] >= min_diamonds
            dl_ok = r["downloads"] >= min_downloads
            if (d_ok and dl_ok) if require_both else (d_ok or dl_ok):
                hits.append(i)
        self.mark_keep(hits, reason=reason)
        return len(hits)

    def categories(self) -> List[tuple]:
        """(category, count) over the current view, most common first."""
        from collections import Counter
        c = Counter(self.features[i]["category"] for i in self.indices if self.features[i]["category"])
        return c.most_common()

    def mark_remove(self, indices: Sequence[int], reason: str = "") -> None:
        self.mark(indices, "remove", reason)

    def mark_keep(self, indices: Sequence[int], reason: str = "") -> None:
        self.mark(indices, "keep", reason)

    def clear_marks(self, indices: Sequence[int]) -> None:
        for i in indices:
            self.decisions.pop(self.structures[i].source_path or f"#{i}", None)

    def _paths_with(self, decision: str) -> List[str]:
        return [p for p, d in self.decisions.items() if d.get("decision") == decision]

    def remove_list(self) -> List[str]:
        return self._paths_with("remove")

    def keep_list(self) -> List[str]:
        return self._paths_with("keep")

    def save_decisions(self, path: str = DEFAULT_DECISIONS_PATH) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.decisions, f, indent=2)
        return path

    def load_decisions(self, path: str = DEFAULT_DECISIONS_PATH) -> int:
        if os.path.exists(path):
            with open(path) as f:
                self.decisions.update(json.load(f))
        return len(self.decisions)

    def apply(self) -> List[Structure]:
        """Return structures minus those marked 'remove'.

        If any structures are explicitly marked 'keep', returns ONLY those (an
        allow-list workflow); otherwise returns everything not removed.
        """
        removed = set(self.remove_list())
        kept = set(self.keep_list())
        out = []
        for s in self.structures:
            p = s.source_path or ""
            if p in removed:
                continue
            if kept and p not in kept:
                continue
            out.append(s)
        return out

    def export_keep_list(self, path: str) -> str:
        """Write the surviving structures' source paths (newline-separated)."""
        with open(path, "w") as f:
            f.write("\n".join(s.source_path or "" for s in self.apply()))
        return path

    # --- rendering ---------------------------------------------------------
    def contact_sheet(self, indices: Sequence[int], cols: int = 6, max_n: int = 24,
                      label_keys: Sequence[str] = ("path", "n_blocks"), figscale: float = 2.4):
        """Render a grid of thumbnails for visual inspection. Returns the Figure."""
        import matplotlib.pyplot as plt
        from blockgen.renderer.render import render_schem

        idx = list(indices)[:max_n]
        rows = (len(idx) + cols - 1) // cols
        fig = plt.figure(figsize=(figscale * cols, figscale * rows))
        for n, i in enumerate(idx):
            ax = fig.add_subplot(rows, cols, n + 1, projection="3d")
            render_schem(self.structures[i], ax=ax, show=False)
            r = self.features[i]
            decision = self.decisions.get(self.structures[i].source_path or f"#{i}", {}).get("decision", "")
            label = " ".join(str(r[k]) for k in label_keys)
            if r["title"]:
                title = r["title"][:28] + ("…" if len(r["title"]) > 28 else "")
                label = f"{title}\n{label}"
            if decision:
                label += f" [{decision}]"
            ax.set_title(label, fontsize=8)
        fig.tight_layout()
        return fig

    def show_group(self, group: Sequence[int], **kwargs):
        return self.contact_sheet(group, **kwargs)


def _suggest_removals(cur: Curator) -> None:
    """Print rule-of-thumb removal candidates for a quick first pass."""
    exact = cur.find_exact_duplicates(0.95)
    n_exact_extra = sum(len(g) - 1 for g in exact)
    variants = cur.find_variant_groups(0.9)
    tiny = len(cur.filter(max_blocks=8).indices)
    monotype = len(cur.filter(max_block_types=1).indices)
    fragmented = len(cur.filter(predicate=lambda r: r["n_components"] >= 5 and r["largest_component_frac"] < 0.4).indices)
    print("removal candidates (rules of thumb):")
    print(f"  exact-duplicate extras (same shape+palette): {n_exact_extra}  ({len(exact)} groups)")
    print(f"  material/color VARIANT groups (KEEP these) : {len(variants)}  ({sum(len(g) for g in variants)} structures)")
    print(f"  tiny (<=8 blocks)                          : {tiny}")
    print(f"  single material type                       : {monotype}")
    print(f"  fragmented (>=5 comps, largest<40%)        : {fragmented}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Curate the cached structure dataset.")
    parser.add_argument("--max-dim", type=int, default=24)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--labeled", action="store_true",
                        help="use the metadata-labeled tfrecord cache (tf_small_<dim>.npz)")
    args = parser.parse_args()
    if args.labeled:
        cur = Curator.from_labeled_cache(max_dim=args.max_dim, cache_dir=args.cache_dir)
    else:
        cur = Curator.from_cache(max_dim=args.max_dim, cache_dir=args.cache_dir)
    cur.summary()
    print()
    _suggest_removals(cur)


if __name__ == "__main__":
    main()
