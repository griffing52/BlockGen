"""The phone queue: what to show next to someone labelling in spare minutes.

Why a server-side queue
-----------------------
The desktop pages hand the person a dataset and a grid and let them choose. That
is the wrong shape for a phone in a waiting room: the point there is to open the
page and have the next thing already decided, do it in one tap, and come back an
hour later to the thing after that. So the decisions about *what* to show live
here, driven by `configs/lab/phone.json`, and the page only asks "next, please".

Three task kinds, in order of value to the project (notes.md section 49):

* `compare` -- a forced-choice pair. Serves the human study (metric-vs-human
  agreement) and preference tuning (pipeline stage 7.4) at once. Pairs are drawn
  to spread appearances across builds and never repeat an identical pair under
  the same tag; sides are randomized here, not in the page.
* `curate` -- good / bad / unsure on one build. Walks each dataset in a fixed
  seeded shuffle and skips anything already labelled, so progress is simply
  "what is in the labels table" and a session resumes by itself.
* `caption` -- one line of text for a build, preferring builds already marked
  good. Not in the default mix because typing is the slow part on a phone.

Everything here is a *proposal* in the sense `store.py` means: nothing is
written to a corpus. The planner reads the store for history and the catalog
for pools; `api.py` does the writes through the existing label/compare routes
plus the new caption route.

Pools are resolved against the catalog at call time and memoized for a short
while, so a bench run that lands while the server is up joins the rotation
without a restart, and a dataset that was deleted drops out instead of 404ing
on its thumbnails.
"""

from __future__ import annotations

import fnmatch
import json
import random
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CONFIG_PATH = Path("configs/lab/phone.json")
KINDS = ("compare", "curate", "caption")
MODES = ("mix",) + KINDS

#: How many candidate builds to look at when picking the less-seen one for a
#: trial. Four is enough to flatten appearance counts without turning sampling
#: into a scan of the pool.
_BALANCE_DRAWS = 4
#: Pool memo lifetime. Long enough that a batch of six tasks resolves the
#: catalog once; short enough that a new arm shows up within a minute.
_POOL_TTL_S = 45.0

DEFAULT_CONFIG: Dict[str, Any] = {
    "mix": {"compare": 0.6, "curate": 0.4},
    "compare": [
        {"tag": "more_real",
         "question": "Which looks more like a <b>real Minecraft house</b>?",
         "weight": 1.0, "a": ["split:*:test"], "b": ["arm:*"], "pairing": "across"},
    ],
    "curate": [
        {"question": "Keep this as a <b>training house</b>?", "weight": 1.0,
         "datasets": ["corpus:houses_*"]},
    ],
    "caption": [
        {"prompt": "Describe it in one line.", "datasets": ["corpus:houses_*"],
         "prefer_label": "good", "skip_label": "bad"},
    ],
}


# --- config ------------------------------------------------------------------
def load_config(path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """The committed config, or the built-in default when it is missing or
    broken. A typo in the JSON should degrade to "the page still works" and a
    stderr line, not to a 500 on every tap."""
    p = Path(path)
    if not p.is_file():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        cfg = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        print(f"[lab.phone] {p}: {exc}; using the built-in default", flush=True)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(cfg, dict):
        return json.loads(json.dumps(DEFAULT_CONFIG))
    out = json.loads(json.dumps(DEFAULT_CONFIG))
    out.update({k: v for k, v in cfg.items() if not k.startswith("_")})
    return out


# --- pools -------------------------------------------------------------------
class _Pools:
    """Glob -> [dataset] resolution with a TTL memo and a seeded order per dataset."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._datasets: List[Any] = []
        self._at = 0.0
        self._orders: Dict[str, List[int]] = {}

    def datasets(self) -> List[Any]:
        from tools.lab import catalog
        with self._lock:
            if time.monotonic() - self._at > _POOL_TTL_S:
                try:
                    self._datasets = [d for d in catalog.list_datasets() if (d.n or 0) > 0]
                except Exception as exc:                       # noqa: BLE001
                    print(f"[lab.phone] catalog unavailable: {exc}", flush=True)
                    self._datasets = []
                self._at = time.monotonic()
            return list(self._datasets)

    def resolve(self, patterns: Sequence[str]) -> List[Any]:
        """Datasets matching the patterns, in *pattern* order, each once.

        Pattern order rather than catalog order because the curate walk finishes
        one dataset before starting the next, and the config author put the
        dataset they care about first for that reason."""
        seen: set = set()
        out: List[Any] = []
        datasets = self.datasets()
        for pat in patterns:
            for d in datasets:
                if d.id not in seen and fnmatch.fnmatchcase(d.id, pat):
                    seen.add(d.id)
                    out.append(d)
        return out

    def order(self, dataset_id: str, n: int) -> List[int]:
        """A fixed shuffle of `range(n)`, seeded by the dataset id.

        Fixed so that the walk through a dataset is the same on every visit:
        progress is "everything before here is labelled", and two sessions a
        week apart continue rather than restart. Shuffled rather than in index
        order because corpora are often sorted by source, and labelling the
        first 300 in file order would label one source and nothing else.
        """
        with self._lock:
            got = self._orders.get(dataset_id)
            if got is not None and len(got) == n:
                return got
            rng = random.Random(zlib.crc32(dataset_id.encode("utf-8")))
            order = list(range(n))
            rng.shuffle(order)
            self._orders[dataset_id] = order
            return order

    def forget(self) -> None:
        with self._lock:
            self._at = 0.0
            self._orders.clear()


_POOLS = _Pools()


def _weighted_choice(rng: random.Random, items: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    weights = [max(0.0, float(it.get("weight", 1.0) or 0.0)) for it in items]
    if not items or sum(weights) <= 0:
        return None
    return rng.choices(list(items), weights=weights, k=1)[0]


def _bid(dataset_id: str, index: int) -> str:
    from tools.lab.catalog import make_build_id
    return make_build_id(dataset_id, index)


def _row(dataset_id: str, index: int) -> Dict[str, Any]:
    """The build row the page needs, tolerant of a dataset that will not open."""
    from tools.lab import catalog
    try:
        row = dict(catalog.build_row(dataset_id, index))
    except Exception:                                          # noqa: BLE001
        row = {"build_id": _bid(dataset_id, index), "dataset": dataset_id,
               "index": index, "dims": None, "n_blocks": None, "title": "",
               "category": ""}
    return row


# --- the planner -------------------------------------------------------------
class Planner:
    """Builds the next few tasks for the phone page.

    `store` is any object with the read methods of `store.Store`; it is passed
    in rather than opened here so `api.py` can hold its own session lock around
    the call, and so tests can hand in a store on a temp file.
    """

    def __init__(self, store: Any, config: Optional[Dict[str, Any]] = None,
                 pools: Optional[_Pools] = None, seed: Optional[int] = None) -> None:
        self.store = store
        self.config = config if config is not None else load_config()
        self.pools = pools or _POOLS
        self.rng = random.Random(seed)

    # -- public --
    def next_tasks(self, mode: str = "mix", n: int = 6,
                   exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
        """Up to `n` tasks. `exclude` is build ids the page already holds (queued
        or pending a write) so two batches never overlap on a curate/caption
        item -- the server cannot see a label that is still in the page's undo
        window."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        excluded = set(exclude)
        out: List[Dict[str, Any]] = []
        misses = 0
        while len(out) < n and misses < 3 * max(1, n):
            kind = self._draw_kind(mode)
            task = None
            if kind == "compare":
                task = self._compare(excluded)
            elif kind == "curate":
                task = self._curate(excluded)
            elif kind == "caption":
                task = self._caption(excluded)
            if task is None:
                misses += 1
                # In mix mode a dry kind should not starve the others: if curate
                # ran out, keep serving compares rather than stopping the page.
                if mode != "mix":
                    break
                continue
            # Curate and caption items are hard-excluded once handed out; a
            # compare build is only *discouraged* (see `_compare`), because a
            # small real pool would otherwise cap a batch at its size.
            if kind != "compare":
                for bid in task.get("build_ids", ()):
                    excluded.add(bid)
            out.append(task)
        return out

    def summary(self) -> Dict[str, Any]:
        """What the queue has to offer and how far along each pool is. Drives
        the page's mode chips (a mode with nothing left is shown as done)."""
        jobs: Dict[str, Any] = {k: [] for k in KINDS}
        for job in self.config.get("compare", []):
            a = self.pools.resolve(job.get("a", []))
            b = self.pools.resolve(job.get("b", []))
            jobs["compare"].append({
                "tag": job.get("tag", ""), "question": job.get("question", ""),
                "a": [d.id for d in a], "b": [d.id for d in b],
                "judged": sum(1 for _ in self.store.compared_pair_keys(job.get("tag") or None)),
            })
        labelled = self.store.labels() or {}
        for job in self.config.get("curate", []):
            ds = self.pools.resolve(job.get("datasets", []))
            total = sum(int(d.n) for d in ds)
            done = sum(1 for bid in labelled if any(bid.startswith(d.id + ":") for d in ds))
            jobs["curate"].append({"question": job.get("question", ""),
                                   "datasets": [d.id for d in ds],
                                   "total": total, "done": done})
        captioned = self.store.captions() or {}
        for job in self.config.get("caption", []):
            ds = self.pools.resolve(job.get("datasets", []))
            total = sum(int(d.n) for d in ds)
            done = sum(1 for bid in captioned if any(bid.startswith(d.id + ":") for d in ds))
            jobs["caption"].append({"prompt": job.get("prompt", ""),
                                    "datasets": [d.id for d in ds],
                                    "total": total, "done": done})
        return {"mix": self.config.get("mix", {}), "jobs": jobs}

    # -- kinds --
    def _draw_kind(self, mode: str) -> str:
        if mode != "mix":
            return mode
        mix = self.config.get("mix") or {}
        kinds = [k for k in KINDS if float(mix.get(k, 0) or 0) > 0]
        if not kinds:
            return "compare"
        return self.rng.choices(kinds, weights=[float(mix[k]) for k in kinds], k=1)[0]

    def _compare(self, excluded: set) -> Optional[Dict[str, Any]]:
        jobs = [j for j in self.config.get("compare", []) if isinstance(j, dict)]
        self.rng.shuffle(jobs)
        for job in sorted(jobs, key=lambda j: -float(j.get("weight", 1.0) or 0)):
            pool_a = self.pools.resolve(job.get("a", []))
            pool_b = self.pools.resolve(job.get("b", []))
            if not pool_a or not pool_b:
                continue
            tag = str(job.get("tag") or "")
            pairing = str(job.get("pairing") or "any")
            seen_pairs = self.store.compared_pair_keys(tag or None)
            counts = dict(self.store.compared_pairs(tag or None))
            # Builds the page is already holding count as seen, so the balance
            # draw steers away from them without forbidding them outright.
            for bid in excluded:
                counts[bid] = counts.get(bid, 0) + 1
            for _ in range(12):
                da = self.rng.choice(pool_a)
                if pairing == "within":
                    db = da if da in pool_b else self.rng.choice(pool_b)
                elif pairing == "across":
                    others = [d for d in pool_b if d.id != da.id]
                    if not others:
                        break
                    db = self.rng.choice(others)
                else:
                    db = self.rng.choice(pool_b)
                a = self._less_seen(da, counts, set())
                b = self._less_seen(db, counts, set(), avoid=a)
                if a is None or b is None or a == b:
                    continue
                key = (a, b) if a <= b else (b, a)
                if key in seen_pairs:
                    continue
                for bid in (a, b):
                    counts[bid] = counts.get(bid, 0) + 1
                flip = self.rng.random() < 0.5
                left, right = (b, a) if flip else (a, b)
                return {
                    "kind": "compare", "tag": tag,
                    "question": str(job.get("question") or "Which is better?"),
                    "a": a, "b": b, "left": left, "right": right,
                    "build_ids": [a, b],
                }
        return None

    def _less_seen(self, dataset: Any, counts: Dict[str, int], excluded: set,
                   avoid: Optional[str] = None) -> Optional[str]:
        """A build from `dataset`, biased toward ones with few trials so far."""
        n = int(dataset.n or 0)
        if n <= 0:
            return None
        best, best_c = None, None
        for _ in range(_BALANCE_DRAWS):
            bid = _bid(dataset.id, self.rng.randrange(n))
            if bid in excluded or bid == avoid:
                continue
            c = counts.get(bid, 0)
            if best_c is None or c < best_c:
                best, best_c = bid, c
        return best

    def _curate(self, excluded: set) -> Optional[Dict[str, Any]]:
        jobs = [j for j in self.config.get("curate", []) if isinstance(j, dict)]
        job = _weighted_choice(self.rng, jobs)
        tried: set = set()
        while job is not None:
            tried.add(id(job))
            ds = self.pools.resolve(job.get("datasets", []))
            # Walk the datasets in order: finish one before starting the next,
            # so a half-done pass is a complete label set over a prefix of the
            # fixed shuffle rather than a thin scatter over everything.
            for d in ds:
                labelled = self.store.labels(dataset=d.id) or {}
                nxt = self._next_unseen(d, labelled, excluded)
                if nxt is None:
                    continue
                row = _row(d.id, nxt)
                return {
                    "kind": "curate",
                    "question": str(job.get("question") or "Keep this build?"),
                    "build_id": row["build_id"], "dataset": d.id, "index": nxt,
                    "title": row.get("title") or "", "category": row.get("category") or "",
                    "dims": row.get("dims"), "n_blocks": row.get("n_blocks"),
                    "remaining": int(d.n) - len(labelled),
                    "build_ids": [row["build_id"]],
                }
            rest = [j for j in jobs if id(j) not in tried]
            job = _weighted_choice(self.rng, rest)
        return None

    def _caption(self, excluded: set) -> Optional[Dict[str, Any]]:
        jobs = [j for j in self.config.get("caption", []) if isinstance(j, dict)]
        job = _weighted_choice(self.rng, jobs)
        tried: set = set()
        while job is not None:
            tried.add(id(job))
            ds = self.pools.resolve(job.get("datasets", []))
            prefer = job.get("prefer_label")
            skip = job.get("skip_label")
            for d in ds:
                captioned = self.store.captions(dataset=d.id) or {}
                labelled = self.store.labels(dataset=d.id) or {}
                order = self.pools.order(d.id, int(d.n))
                # Two passes: builds already judged good first (they are the
                # ones that will be in the SFT set), then everything not bad.
                passes = []
                if prefer:
                    passes.append(lambda bid: labelled.get(bid) == prefer)
                passes.append(lambda bid: labelled.get(bid) != skip if skip else True)
                for ok in passes:
                    for i in order:
                        bid = _bid(d.id, i)
                        if bid in captioned or bid in excluded or not ok(bid):
                            continue
                        row = _row(d.id, i)
                        return {
                            "kind": "caption",
                            "prompt": str(job.get("prompt") or "Describe it in one line."),
                            "build_id": bid, "dataset": d.id, "index": i,
                            "title": row.get("title") or "",
                            "category": row.get("category") or "",
                            "dims": row.get("dims"), "n_blocks": row.get("n_blocks"),
                            "label": labelled.get(bid),
                            "remaining": int(d.n) - len(captioned),
                            "build_ids": [bid],
                        }
            rest = [j for j in jobs if id(j) not in tried]
            job = _weighted_choice(self.rng, rest)
        return None

    def _next_unseen(self, dataset: Any, labelled: Dict[str, str],
                     excluded: set) -> Optional[int]:
        n = int(dataset.n or 0)
        for i in self.pools.order(dataset.id, n):
            bid = _bid(dataset.id, i)
            if bid not in labelled and bid not in excluded:
                return i
        return None


def forget() -> None:
    """Drop the pool memo (tests, or after a dataset is added)."""
    _POOLS.forget()
