"""The canonical train/val/test split. One definition, used by everything.

Before this module there were two incompatible ad-hoc splits: a plain random
permutation in `experiments_transfer._split_houses` (2-way, no test set) and a
filtered disjoint pool in `scripts/validate_perceptual.py`. Arms split one way
were being compared against references split the other.

**Splits are group-aware, and that is the point.** GrabCraft ships sibling
builds -- "American Middle Class House 10 / 22 / 9", "Seashore Brick House 1 / 2
/ 3" -- that are near-duplicates of each other. A random *row* split scatters
siblings across train and val, which corrupts the two numbers the suite cares
about most in opposite directions: the held-out-real floor is inflated (val
looks easy because train contains its twin) and every memorization metric is
deflated (a model reciting train scores well against val). Grouping by
normalized title keeps families whole.

`splits.json` at the repo root is *not* this. It contains 153 `C###` ids, is
referenced nowhere in the codebase, and is unrelated to the house corpus. It is
left alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.curation.houses import DEFAULT_CACHE_DIR, load_structures_from_cache
from blockgen.labeling.templates import clean_title
from blockgen.utils.data import Structure

SPLIT_VERSION = "v1"
SPLIT_ROOT = Path("data/minecraft/splits")

_TRAILING_NUM = re.compile(r"[\s\-_#]*\d+\s*$")


def corpus_path(corpus: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{corpus}.npz"


def load_corpus(corpus: str = "houses_32",
                cache_dir: str | Path = DEFAULT_CACHE_DIR
                ) -> Tuple[List[Structure], dict]:
    """Load a cached corpus and its manifest (`load_structures_from_cache`)."""
    return load_structures_from_cache(str(corpus_path(corpus, cache_dir)))


def group_key(item: dict) -> str:
    """Family key for one manifest row: siblings must collide, strangers must not.

    "American Middle Class House 10" and "... 22" both reduce to
    `brick-houses::american middle class house`. Falls back to the source path so
    an untitled row is its own group rather than joining a giant "" bucket.
    """
    title = clean_title(str(item.get("title") or ""))
    title = _TRAILING_NUM.sub("", title).strip().lower()
    category = str(item.get("category") or "").strip().lower()
    if not title:
        return f"path::{item.get('source_path') or id(item)}"
    return f"{category}::{title}"


def source_sha(manifest: dict) -> str:
    """Fingerprint of the corpus, so a rebuilt cache invalidates the split.

    Covers every field the split actually depends on -- path *and* the title and
    category that `group_key` reads. Hashing paths alone would let a re-captioned
    corpus silently reuse a split whose families no longer match.
    """
    rows = ["\x1f".join((str(it.get("source_path", "")), str(it.get("title", "")),
                         str(it.get("category", ""))))
            for it in manifest.get("items", [])]
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()[:16]


@dataclass
class Split:
    corpus: str
    seed: int
    fracs: Tuple[float, float, float]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    source_sha: str
    n_groups: int = 0
    version: str = SPLIT_VERSION
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    notes: Dict[str, str] = field(default_factory=dict)

    def key(self) -> str:
        f = "-".join(str(int(round(x * 100))) for x in self.fracs)
        return f"{self.corpus}.s{self.seed}.{f}.{self.version}"

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def indices(self, which: str) -> np.ndarray:
        return {"train": self.train, "val": self.val, "test": self.test}[which]

    def to_json(self) -> dict:
        return {
            "version": self.version, "corpus": self.corpus, "seed": self.seed,
            "fracs": list(self.fracs), "source_sha": self.source_sha,
            "n_groups": self.n_groups, "cache_dir": self.cache_dir,
            "sizes": self.sizes, "notes": self.notes,
            "train": [int(i) for i in self.train],
            "val": [int(i) for i in self.val],
            "test": [int(i) for i in self.test],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Split":
        return cls(
            corpus=d["corpus"], seed=d["seed"], fracs=tuple(d["fracs"]),
            train=np.array(d["train"], dtype=int), val=np.array(d["val"], dtype=int),
            test=np.array(d["test"], dtype=int), source_sha=d["source_sha"],
            n_groups=d.get("n_groups", 0), version=d.get("version", SPLIT_VERSION),
            cache_dir=d.get("cache_dir", str(DEFAULT_CACHE_DIR)),
            notes=d.get("notes", {}),
        )


def make_split(
    corpus: str = "houses_32",
    seed: int = 0,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    manifest: dict | None = None,
) -> Split:
    """Build a deterministic group-aware split.

    Groups are shuffled under `default_rng(seed)` then assigned greedily to
    whichever split is furthest below its target share, so family integrity is
    preserved exactly and the realized sizes land close to `fracs`.
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0, got {fracs}")
    if manifest is None:
        _, manifest = load_corpus(corpus, cache_dir)
    items = manifest.get("items", [])
    if not items:
        raise ValueError(f"corpus {corpus!r} manifest has no items")

    groups: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault(group_key(it), []).append(i)

    names = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(names))
    # Largest groups first within the shuffled order keeps the greedy assignment
    # from overshooting on the last few placements.
    names = sorted((names[i] for i in order), key=lambda g: -len(groups[g]))

    n_total = len(items)
    targets = [f * n_total for f in fracs]
    buckets: List[List[int]] = [[], [], []]
    for g in names:
        deficits = [targets[k] - len(buckets[k]) for k in range(3)]
        buckets[int(np.argmax(deficits))].extend(groups[g])

    return Split(
        corpus=corpus, seed=seed, fracs=fracs,
        train=np.sort(np.array(buckets[0], dtype=int)),
        val=np.sort(np.array(buckets[1], dtype=int)),
        test=np.sort(np.array(buckets[2], dtype=int)),
        source_sha=source_sha(manifest), n_groups=len(groups),
        cache_dir=str(cache_dir),
        notes={"grouping": "clean_title minus trailing digits, + category"},
    )


def split_path(corpus: str, seed: int, fracs: Tuple[float, float, float],
               root: Path | str = SPLIT_ROOT) -> Path:
    f = "-".join(str(int(round(x * 100))) for x in fracs)
    return Path(root) / f"{corpus}.s{seed}.{f}.{SPLIT_VERSION}.json"


def load_split(
    corpus: str = "houses_32",
    seed: int = 0,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    root: Path | str = SPLIT_ROOT,
    rebuild: bool = False,
) -> Split:
    """Read the split from disk, rebuilding if absent or if the corpus changed."""
    path = split_path(corpus, seed, fracs, root)
    _, manifest = load_corpus(corpus, cache_dir)
    sha = source_sha(manifest)

    if path.exists() and not rebuild:
        cached = Split.from_json(json.loads(path.read_text()))
        if cached.source_sha == sha:
            return cached
        print(f"[splits] {path.name}: corpus changed "
              f"({cached.source_sha} -> {sha}); rebuilding")

    split = make_split(corpus, seed, fracs, cache_dir, manifest=manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_json(), indent=2) + "\n")
    return split


def split_structures(
    split: Split,
    which: str,
    cache_dir: str | Path | None = None,
    crop: bool = True,
    predicate: Callable[[Structure], bool] | None = None,
) -> List[Structure]:
    """Materialize one side of a split as cropped `Structure`s.

    `predicate` selects a *named subset* (e.g. the ladder's decimator-sensitive
    pool) without creating a second split -- the underlying partition is
    unchanged, so subset numbers stay comparable to full-split numbers.
    """
    structs, _ = load_corpus(split.corpus, cache_dir or split.cache_dir)
    out = [structs[i] for i in split.indices(which)]
    if crop:
        out = [s.crop_to_non_air() for s in out]
    if predicate is not None:
        out = [s for s in out if predicate(s)]
    return out


def group_leakage(split: Split, manifest: dict) -> Dict[str, int]:
    """Groups appearing in more than one side. Must be all-zero by construction."""
    where: Dict[str, set] = {}
    items = manifest["items"]
    for name, idxs in (("train", split.train), ("val", split.val), ("test", split.test)):
        for i in idxs:
            where.setdefault(group_key(items[i]), set()).add(name)
    shared = [g for g, s in where.items() if len(s) > 1]
    return {"n_groups": len(where), "n_leaked": len(shared)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or rebuild a canonical split.")
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fracs = tuple(args.fracs)
    split = load_split(args.corpus, args.seed, fracs, rebuild=args.rebuild)
    _, manifest = load_corpus(args.corpus, split.cache_dir)

    print(f"split      {split.key()}")
    print(f"source_sha {split.source_sha}")
    print(f"groups     {split.n_groups} over {sum(split.sizes.values())} structures")
    for name, n in split.sizes.items():
        print(f"  {name:<6} {n:>5}  ({n / sum(split.sizes.values()):.1%})")
    print(f"leakage    {group_leakage(split, manifest)}")
    print(f"-> {split_path(args.corpus, args.seed, fracs)}")

    if args.show:
        items = manifest["items"]
        for name in ("train", "val", "test"):
            print(f"\n[{name}] first 5:")
            for i in split.indices(name)[:5]:
                print(f"  {i:>5} {group_key(items[i])}")


if __name__ == "__main__":
    main()
