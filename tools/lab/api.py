"""HTTP surface for BlockLab: routing, JSON shaping, and static serving.

Implements the contract frozen in `tools/lab/__init__.py`. Three properties of
this file are load-bearing and worth stating before the code:

**Sibling modules are imported late, per request, and failure is not cached.**
`catalog`, `renders`, `store` and `gates` each own an artifact this server only
reads. If one of them is absent or raises on import, the server still starts and
every *other* route still answers; the affected route returns a 503 that names
the missing file. Retrying the import on each request (rather than memoizing the
exception) means a module that appears while the server is running starts
working without a restart -- which is the normal state of affairs while this
tool is being built.

**Every response is sanitized before `json.dumps`.** The metrics upstream are
numpy scalars and can legitimately be NaN or +/-inf (a bootstrap CI on a
degenerate arm, a ratio over an empty build). `json.dumps` would happily emit
bare `NaN`, which is not JSON and which `JSON.parse` rejects -- the page would
show an empty table with no clue why. `_clean` maps those to `null` so the
front-end's em-dash formatter can do its job.

**Nothing here writes to a corpus.** Labels and notes go to SQLite via `store`;
renders go to the disk cache. The batch pipeline still owns every artifact under
`outputs/` that is not `outputs/lab/`.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from tools.lab import DEFAULT_PORT

STATIC = Path(__file__).resolve().parent / "static"

#: Pages that live in `static/` and are reachable at a bare path. Three of these
#: are written by other hands; a missing one must render a note, not a traceback.
PAGES = {
    "/": "index.html",
    "/curate": "curate.html",
    "/leaderboard": "leaderboard.html",
    "/runs": "runs.html",
    "/compare": "compare.html",
    "/curation": "curation.html",
    "/ontology": "ontology.html",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json",
}

LABELS = ("good", "bad", "unsure")

# Every SQLite touch is serialized through this. The lab is a single-user tool
# with a handful of writes per second, so a global lock costs nothing measurable
# and removes the whole class of "database is locked" flakes that a
# thread-per-request server otherwise invites.
_DB_LOCK = threading.Lock()


class LabError(Exception):
    """An error with an HTTP status attached, rendered as `{error, detail}`."""

    def __init__(self, status: int, error: str, detail: str = "") -> None:
        super().__init__(error)
        self.status = status
        self.error = error
        self.detail = detail


def lab_module(name: str) -> Any:
    """Import `tools.lab.<name>`, or raise a 503 that names the missing file.

    Successes are cached by `importlib`; failures deliberately are not, so a
    module that lands on disk after the server booted is picked up live.
    """
    try:
        return importlib.import_module(f"tools.lab.{name}")
    except Exception as exc:  # ImportError, but also anything raised at import
        raise LabError(
            503,
            f"tools/lab/{name}.py is not available",
            f"{type(exc).__name__}: {exc}",
        ) from exc


class _StoreHandle:
    """A `store.Store` opened per call and closed after, under `_DB_LOCK`.

    Per-call open (rather than one long-lived connection) sidesteps sqlite3's
    `check_same_thread` guard entirely: `ThreadingHTTPServer` runs each request
    on a new thread, and a connection created on thread A raises if touched from
    thread B regardless of any lock we hold.
    """

    def __enter__(self) -> Any:
        _DB_LOCK.acquire()
        try:
            self.store = lab_module("store").Store()
        except Exception:
            _DB_LOCK.release()
            raise
        return self.store

    def __exit__(self, *exc: Any) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        finally:
            _DB_LOCK.release()


def store_session() -> _StoreHandle:
    return _StoreHandle()


# --- JSON hygiene ----------------------------------------------------------
def _clean(obj: Any) -> Any:
    """Make anything the bench produces safe for `json.dumps` and `JSON.parse`."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    # numpy scalars/arrays without importing numpy at module scope.
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "shape", None) == ():
        return _clean(item())
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _clean(tolist())
    if hasattr(obj, "__dict__"):
        return _clean(vars(obj))
    return str(obj)


def _one(query: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = query.get(key)
    return values[0] if values else default


def _int(query: dict[str, list[str]], key: str, default: int,
         lo: int | None = None, hi: int | None = None) -> int:
    raw = _one(query, key)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(float(raw))
        except ValueError:
            raise LabError(400, f"{key} must be an integer", f"got {raw!r}")
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _coerce(raw: str) -> Any:
    """Query strings are untyped; gate params are not. Guess conservatively."""
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _build_id(rest: str) -> tuple[str, int]:
    catalog = lab_module("catalog")
    try:
        return catalog.parse_build_id(urllib.parse.unquote(rest))
    except ValueError as exc:
        raise LabError(400, "malformed build_id", str(exc)) from exc


# --- per-build detail ------------------------------------------------------
def _structure(dataset_id: str, index: int) -> Any:
    catalog = lab_module("catalog")
    if catalog.get_dataset(dataset_id) is None:
        raise LabError(404, "unknown dataset", dataset_id)
    try:
        builds = catalog.load_builds(dataset_id)
    except Exception as exc:  # a truncated or half-written .npz is a 404, not a crash
        raise LabError(404, "dataset could not be loaded",
                       f"{dataset_id}: {type(exc).__name__}: {exc}") from exc
    if not builds:
        raise LabError(404, "dataset has no builds", dataset_id)
    if not 0 <= index < len(builds):
        raise LabError(404, "index out of range",
                       f"{dataset_id} has {len(builds)} builds; asked for {index}")
    return builds[index]


def _features(structure: Any) -> dict[str, Any]:
    """Cheap per-build descriptors: coherence plus the palette that made it.

    Deliberately *not* the DINO features the benchmark uses -- those need a GPU
    backbone and a reference set, and neither belongs behind a click on a tile.
    """
    out: dict[str, Any] = {}
    try:
        from blockgen.eval.bench import topology

        coh = topology.coherence(structure)
        out.update({k: getattr(coh, k) for k in
                    ("n_blocks", "bbox_fill", "n_components", "lcc_ratio",
                     "disconnected_block_ratio", "floating_component_frac",
                     "floating_block_frac", "enclosed_air", "enclosed_air_ratio")})
    except Exception as exc:
        out["coherence_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from blockgen.eval.bench import palette as pal

        hist = pal.palette_hist(structure, level="exact")
        total = sum(hist.values()) or 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:12]
        out["palette_size"] = len(hist)
        out["palette_top"] = [{"block": str(k), "n": int(v), "frac": v / total}
                              for k, v in top]
    except Exception as exc:
        out["palette_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _geometry(structure: Any) -> dict[str, Any]:
    """The scalar half of `bench.geometry.geometry_stats`; histograms are dropped
    because a tile drawer has no room for a 64-bin pattern histogram."""
    try:
        from blockgen.eval.bench import geometry as geo

        stats = geo.geometry_stats(structure)
        return {k: getattr(stats, k) for k in
                ("thickness_mean", "thickness_p90", "surface_to_volume",
                 "wall_frac", "yaw_symmetry", "interior_ratio_sealed",
                 "interior_ratio_open", "height_entropy")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- scorecard derivation --------------------------------------------------
#: The version-controlled model registry `deploy/inference/models.json`. It is
#: the richest identity record in the system -- prose `description` and `notes`
#: written by hand for every checkpoint the server can serve -- and until now
#: nothing downstream read those two fields. Resolved from `__file__` rather
#: than the cwd because it is a committed file at a fixed place in the tree,
#: unlike everything under `outputs/`.
MODELS_JSON = Path(__file__).resolve().parents[2] / "deploy" / "inference" / "models.json"

#: (mtime, cards) or None. See `_model_cards`.
_MODEL_CARDS: tuple[float, dict[str, dict[str, Any]]] | None = None


def _model_cards() -> dict[str, dict[str, Any]]:
    """The prose half of `models.json`, keyed by model name, memoized on mtime.

    Joined to an arm on `meta.provenance.model`, falling back to the arm's own
    name -- the name `sample_to_npz.py` writes into the arm manifest is the same
    string that keys `models["models"]`, so the registry entry a human curated
    for a checkpoint can finally appear beside that checkpoint's scores. Only
    `kind`, `description` and `notes` are
    lifted: the rest of an entry is serving configuration (paths, `d_model`,
    `nhead`, vocab files) that says nothing about what the model *is*, and one
    of those paths is a checkpoint location this page has no business
    advertising in a JSON payload.

    Fail-soft to `{}`: a leaderboard must render for someone whose checkout has
    no `deploy/` tree, and a hand-edited registry with a trailing comma must
    cost the reader a missing paragraph, not a 500. The empty result is
    memoized like any other, which is safe precisely because fixing the JSON
    changes the mtime and invalidates it.
    """
    global _MODEL_CARDS
    try:
        stamp = MODELS_JSON.stat().st_mtime
    except OSError:
        return {}
    if _MODEL_CARDS is not None and _MODEL_CARDS[0] == stamp:
        return _MODEL_CARDS[1]
    cards: dict[str, dict[str, Any]] = {}
    try:
        blob = json.loads(MODELS_JSON.read_text())
        for name, entry in (blob.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            cards[str(name)] = {"kind": str(entry.get("kind") or ""),
                                "description": str(entry.get("description") or ""),
                                "notes": str(entry.get("notes") or "")}
    except Exception as exc:  # noqa: BLE001 -- prose is never worth a failed route
        print(f"[lab.api] unreadable model registry {MODELS_JSON}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        cards = {}
    _MODEL_CARDS = (stamp, cards)
    return cards


def _scorecard_payload(card: dict[str, Any]) -> dict[str, Any]:
    """Fill in the contract's derived keys, without clobbering catalog's.

    `catalog.load_scorecard` already lifts `leaderboard`, `head_to_head`,
    `arm_names`, a `metrics` list of `"section.key"` strings and the run-level
    identity keys out of the raw card. Overwriting any of those here would
    silently change the shape the leaderboard page reads, so this only *adds*
    `arms_index`: one row per arm, joining its `meta` to its BlockScore row, to
    its recipe, to the dataset its builds live in, to its rendered examples and
    to its `models.json` entry. Five joins the page used to do by hand in JS
    against three separately-fetched documents.

    **Derived per-arm data goes in `arms_index`, never into `arms`.** `card` is
    `catalog.load_scorecard`'s *shallow* copy, so `card["arms"]` is the object
    in the process-wide `_SCORECARDS` memo (the comment at catalog.py:598
    carries the same rule): enrich BESIDE `arms` -- top-level keys, or
    `arms_index` -- never inside it. Writing into `arms[name]["meta"]` here
    would poison every later reader in the process, cumulatively, and the
    poisoning would be invisible until someone diffed a payload against the
    file on disk. `arms_index` is a fresh list of fresh dicts, rebuilt per
    request, and nothing in it is ever read back. (D27/D29.)

    The `if not` guards cover a card read by some other path that skipped
    catalog's enrichment.
    """
    catalog = lab_module("catalog")
    cards = lab_module("cards")
    run = card.get("run") or {}
    arms = card.get("arms") or {}
    payload = dict(card)
    if not payload.get("leaderboard"):
        payload["leaderboard"] = run.get("blockscore") or []
    if not payload.get("head_to_head"):
        payload["head_to_head"] = run.get("head_to_head") or []
    if not payload.get("arm_names"):
        payload["arm_names"] = list(arms)
    leaderboard = payload["leaderboard"]

    # Example builds are addressed as build_ids the existing /api/thumb route
    # already serves, so no new route -- and therefore no path traversal
    # surface -- is opened over a run directory full of checkpoints (D4). The
    # npz lives on disk and can be truncated, absent, or outside `outputs/`, so
    # a failure here costs the strip and nothing else.
    examples: dict[str, list[str]] = {}
    path = card.get("path")
    if path:
        try:
            examples = catalog.example_build_ids(Path(path).parent, card) or {}
        except Exception as exc:  # noqa: BLE001
            print(f"[lab.api] example builds unavailable for {path}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    models = _model_cards()
    # `source_dataset_id` scans the dataset list, which `catalog` rebuilds on
    # every call, and most arms share a source string -- 13 of 16 rows on a
    # real card are the literal "in-memory". Resolve each distinct source once.
    dataset_of: dict[str, str | None] = {}
    rows = []
    for name, blocks in arms.items():
        meta = (blocks or {}).get("meta") or {}
        score = next((r for r in leaderboard if r.get("arm") == name), {})
        prov = meta.get("provenance")
        prov = prov if isinstance(prov, dict) else {}
        source = meta.get("source") or ""
        if source not in dataset_of:
            dataset_of[source] = catalog.source_dataset_id(source)
        rows.append({"name": name, "track": meta.get("track"),
                     "n": meta.get("n"), "n_empty": meta.get("n_empty"),
                     "source": meta.get("source"),
                     "has_prompts": meta.get("has_prompts"),
                     "score": score.get("score"), "status": score.get("status"),
                     "disqualified": score.get("disqualified"),
                     # `kind`/`origin` are read through `cards` so that the
                     # written field wins and the fallback from `track`/`source`
                     # has exactly one definition, shared with `list_scorecards`.
                     "kind": cards.arm_kind(meta),
                     "origin": cards.arm_origin(meta),
                     "provenance": prov,
                     # A legacy card has no provenance; the static recipe map is
                     # what stops 81% of its rows saying only "in-memory" (D15).
                     "recipe": catalog.arm_recipe(name, meta),
                     "source_run_id": meta.get("source_run_id"),
                     "structures_sha": meta.get("structures_sha"),
                     "examples": examples.get(name) or [],
                     "dataset": dataset_of[source],
                     # Fall back to the ARM NAME as the join key: an arm is named
                     # after the model it ran by convention -- `native_oriented`
                     # and `agentic` are both arm names and `models.json` keys --
                     # and `provenance` exists only on a `bench/2` card, so
                     # without this every one of the eleven legacy runs on disk
                     # shows no prose at all, including for arms whose names ARE
                     # registry keys. Still null when neither key resolves.
                     "model_card": (models.get(str(prov.get("model") or ""))
                                    or models.get(str(name)))})
    payload["arms_index"] = sorted(rows, key=lambda r: r["name"])
    return payload


# --- routes ----------------------------------------------------------------
Query = dict[str, list[str]]


def api_datasets(query: Query) -> Any:
    catalog = lab_module("catalog")
    return [d.to_json() if hasattr(d, "to_json") else _clean(d)
            for d in catalog.list_datasets()]


def api_tree(query: Query) -> Any:
    """The datasets forest plus its headline counts. See `tree`."""
    tree = lab_module("tree")
    roots = tree.build()
    return {"roots": [r.to_json() for r in roots], "totals": tree.totals(roots)}


def api_subsets(query: Query) -> Any:
    subsets = lab_module("subsets")
    return [s.to_json() | {"describe": s.describe()} for s in subsets.load_all()]


def _subset_from_body(body: dict[str, Any]) -> Any:
    subsets = lab_module("subsets")
    parent = str(body.get("parent") or "").strip()
    if not parent:
        raise LabError(400, "parent is required")
    rule = body.get("rule") or {}
    if not isinstance(rule, dict):
        raise LabError(400, "rule must be an object")
    sid = str(body.get("id") or "").strip().lower().replace(" ", "-")
    return subsets.Subset(
        id=sid, name=str(body.get("name") or sid or parent),
        parent=parent, mode=str(body.get("mode") or "filter"),
        rule=rule, note=str(body.get("note") or ""))


def post_subset_preview(body: dict[str, Any]) -> Any:
    """Resolve a rule without saving it.

    The whole point of a branch is deciding where to cut, and a rule you cannot
    see the size of before committing is a guess. Returns the count, the parent
    count, and the first few rows so the shape is visible too.
    """
    subsets, catalog = lab_module("subsets"), lab_module("catalog")
    draft = _subset_from_body(body)
    draft.id = draft.id or "preview"
    parent = catalog.get_dataset(draft.parent)
    if parent is None:
        raise LabError(404, "no such parent dataset", draft.parent)
    try:
        idx = subsets.resolve_indices(draft, parent.n)
    except subsets.SubsetError as exc:
        raise LabError(400, "rule cannot be resolved", str(exc)) from exc
    sample = []
    for i in idx[:12]:
        try:
            sample.append(catalog.build_row(draft.parent, i))
        except Exception:                              # noqa: BLE001
            break                                      # preview is best-effort
    # The full index list, not a page of it: freezing a branch means sending
    # these back as an `indices` rule, and a truncated list would freeze a
    # silently smaller set than the one previewed. 59k ints is ~400 KB over
    # loopback, which is cheaper than that class of bug.
    return {"n": len(idx), "parent_n": parent.n, "describe": draft.describe(),
            "indices": idx, "sample": sample}


def post_subset(body: dict[str, Any]) -> Any:
    """Create (or, with `overwrite`, redefine) a branch."""
    subsets, catalog = lab_module("subsets"), lab_module("catalog")
    draft = _subset_from_body(body)
    known = [d.id for d in catalog.list_datasets()]
    try:
        saved = subsets.save(draft, known=known, overwrite=bool(body.get("overwrite")))
    except subsets.SubsetError as exc:
        raise LabError(400, "invalid subset", str(exc)) from exc
    catalog.forget(saved.dataset_id)
    return saved.to_json() | {"describe": saved.describe(),
                              "n": len(subsets.resolve_indices(saved))}


def post_subset_delete(body: dict[str, Any]) -> Any:
    subsets, catalog = lab_module("subsets"), lab_module("catalog")
    sid = str(body.get("id") or "")
    if not sid:
        raise LabError(400, "id is required")
    removed = subsets.delete(sid)
    if not removed:
        raise LabError(404, "no such subset", sid)
    catalog.forget()
    return {"deleted": sid}


def api_builds(query: Query) -> Any:
    catalog = lab_module("catalog")
    datasets = catalog.list_datasets()
    if not datasets:
        return {"total": 0, "items": [], "dataset": None}
    wanted = _one(query, "dataset")
    if wanted:
        dataset = catalog.get_dataset(wanted)
        if dataset is None:
            raise LabError(404, "unknown dataset", wanted)
    else:
        dataset = datasets[0]  # a bare /api/builds is useful when poking by hand
    dataset_id = dataset.id

    n = int(getattr(dataset, "n", 0) or 0)
    if n <= 0:
        n = len(catalog.load_builds(dataset_id))

    label_filter = (_one(query, "label", "any") or "any").lower()
    if label_filter in ("", "all"):
        label_filter = "any"
    if label_filter not in ("any", "unlabeled", *LABELS):
        raise LabError(400, "bad label filter", label_filter)

    labels: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    try:
        with store_session() as store:
            labels = store.labels(dataset=dataset_id) or {}
            notes = store.notes(dataset=dataset_id) or {}
    except LabError:
        pass  # no store module yet: show the builds, just without decisions

    indices = range(n)
    if label_filter != "any":
        def keep(i: int) -> bool:
            lab = labels.get(catalog.make_build_id(dataset_id, i))
            return lab is None if label_filter == "unlabeled" else lab == label_filter
        indices = [i for i in indices if keep(i)]

    total = len(indices)
    offset = _int(query, "offset", 0, lo=0)
    limit = _int(query, "limit", 60, lo=1, hi=500)
    page = list(indices)[offset:offset + limit]

    items = []
    for i in page:
        row = dict(catalog.build_row(dataset_id, i))
        bid = row.get("build_id") or catalog.make_build_id(dataset_id, i)
        row["build_id"] = bid
        row["label"] = labels.get(bid)
        row["has_note"] = bool((notes.get(bid) or "").strip())
        items.append(row)
    return {"total": total, "items": items, "dataset": dataset_id,
            "offset": offset, "limit": limit, "n_all": n}


def api_build(build_id: str) -> Any:
    dataset_id, index = _build_id(build_id)   # 400, not a 500, on a bad id
    catalog = lab_module("catalog")
    structure = _structure(dataset_id, index)
    row = dict(catalog.build_row(dataset_id, index))
    bid = row.get("build_id") or catalog.make_build_id(dataset_id, index)

    label = note = None
    try:
        with store_session() as store:
            label = store.get_label(bid)
            note = store.get_note(bid)
    except LabError:
        pass

    meta = dict(getattr(structure, "metadata", {}) or {})
    if getattr(structure, "source_path", None):
        meta.setdefault("source_path", structure.source_path)

    quoted = urllib.parse.quote(bid, safe="")
    out = dict(row)
    out.update({
        "build_id": bid, "dataset": dataset_id, "index": index,
        "meta": meta,
        "features": _features(structure),
        "geometry": _geometry(structure),
        "label": label, "note": note or "",
        "thumb": f"/api/thumb/{quoted}",
        "views": [f"/api/view/{quoted}/{k}" for k in range(4)],
    })
    return out


def api_labels(query: Query) -> Any:
    dataset_id = _one(query, "dataset")
    counts = {"good": 0, "bad": 0, "unsure": 0, "unlabeled": 0}
    by_build: dict[str, Any] = {}
    try:
        with store_session() as store:
            counts.update({k: int(v) for k, v in
                           (store.counts(dataset=dataset_id) or {}).items()})
            by_build = store.labels(dataset=dataset_id) or {}
    except LabError:
        return {"counts": counts, "by_build": {}, "store": "unavailable"}

    # `unlabeled` is a property of the dataset, not of the label table, so the
    # store cannot know it unless it was told the size. Recompute when we can.
    try:
        catalog = lab_module("catalog")
        if dataset_id:
            dataset = catalog.get_dataset(dataset_id)
            total = int(getattr(dataset, "n", 0) or 0) if dataset else 0
        else:
            total = sum(int(getattr(d, "n", 0) or 0) for d in catalog.list_datasets())
        if total:
            counts["unlabeled"] = max(0, total - sum(
                v for k, v in counts.items() if k in LABELS))
    except LabError:
        pass
    return {"counts": counts, "by_build": by_build}


def api_export(query: Query) -> Any:
    what = (_one(query, "what", "labels") or "labels").lower()
    if what not in ("labels", "notes", "compares"):
        raise LabError(400, "bad export", f"what must be labels|notes|compares, got {what!r}")
    with store_session() as store:
        return {"what": what, "exported_at": time.time(), "rows": store.export(what)}


def api_standings(query: Query) -> Any:
    """The cross-run board for one protocol.

    Kept out of `/api/scorecards` deliberately: that route answers "what runs
    exist" and is read by three pages, while this one opens and collapses every
    card that matches a protocol. Folding the second into the first would put an
    O(runs) read behind the hub's first paint.
    """
    return lab_module("standings").board(_one(query, "protocol"))


def api_protocols(query: Query) -> Any:
    """The pinned protocols themselves, so a page can explain what it filtered on."""
    proto = lab_module("standings").proto
    return {"default": proto.DEFAULT_ID,
            "protocols": [p.to_json() for p in proto.OFFICIAL]}


def api_scorecards(query: Query) -> Any:
    return lab_module("catalog").list_scorecards()


def api_scorecard(run: str) -> Any:
    catalog = lab_module("catalog")
    try:
        card = catalog.load_scorecard(urllib.parse.unquote(run))
    except FileNotFoundError as exc:
        raise LabError(404, "no such scorecard", str(exc)) from exc
    if not card:
        raise LabError(404, "no such scorecard", run)
    return _scorecard_payload(card)


# --- ontology --------------------------------------------------------------
def _ontology_args(query: Query) -> tuple[str | None, str, int]:
    variant = (_one(query, "variant", "mined") or "mined").lower()
    ontology = lab_module("ontology")
    if variant not in ontology.VARIANTS:
        raise LabError(400, "bad ontology variant",
                       f"expected one of {ontology.VARIANTS}, got {variant!r}")
    return _one(query, "catalog"), variant, _int(query, "seed", 0, lo=0, hi=10_000)


def api_ontology_catalogs(query: Query) -> Any:
    return lab_module("ontology").list_catalogs()


def api_ontology(query: Query) -> Any:
    ontology = lab_module("ontology")
    catalog, variant, seed = _ontology_args(query)
    try:
        return ontology.overview(catalog, variant, seed)
    except FileNotFoundError as exc:
        # A repo that has never run the miner is the normal state, not an error
        # worth a traceback -- the page renders the build command instead.
        raise LabError(404, "no ontology built yet", str(exc)) from exc


def api_ontology_prompt(query: Query) -> Any:
    ontology = lab_module("ontology")
    catalog, variant, seed = _ontology_args(query)
    raw = _one(query, "fields", "") or ""
    fields = [f for f in (part.strip() for part in raw.split(",")) if f]
    try:
        return ontology.prompt_view(catalog, variant, fields, seed)
    except FileNotFoundError as exc:
        raise LabError(404, "no ontology built yet", str(exc)) from exc


def api_ontology_part(rest: str, query: Query) -> Any:
    ontology = lab_module("ontology")
    catalog, variant, seed = _ontology_args(query)
    part_id = urllib.parse.unquote(rest).strip("/")
    try:
        return ontology.part_detail(part_id, catalog, variant, seed)
    except KeyError as exc:
        raise LabError(404, "no such part", part_id) from exc
    except FileNotFoundError as exc:
        raise LabError(404, "no ontology built yet", str(exc)) from exc


def api_curation_preview(query: Query) -> Any:
    """Delegated wholesale to `gates.preview`, which owns the gate semantics."""
    gates = lab_module("gates")
    params = {k: _coerce(v[0]) for k, v in query.items() if v}
    try:
        return gates.preview(**params)
    except TypeError as exc:
        raise LabError(400, "bad gate parameters", str(exc)) from exc


def post_label(body: dict[str, Any]) -> Any:
    bid = str(body.get("build_id") or "")
    if not bid:
        raise LabError(400, "build_id required")
    label = body.get("label")
    if label in ("", "none", "null"):
        label = None
    if label is not None and label not in LABELS:
        raise LabError(400, "bad label", f"expected one of {LABELS} or null, got {label!r}")
    _build_id(bid)  # reject ids the catalog cannot parse before writing them
    with store_session() as store:
        store.set_label(bid, label)
    return {"ok": True, "label": label}


def post_note(body: dict[str, Any]) -> Any:
    bid = str(body.get("build_id") or "")
    if not bid:
        raise LabError(400, "build_id required")
    _build_id(bid)
    with store_session() as store:
        store.set_note(bid, str(body.get("text") or ""))
    return {"ok": True}


def post_compare(body: dict[str, Any]) -> Any:
    a, b = str(body.get("a") or ""), str(body.get("b") or "")
    if not a or not b:
        raise LabError(400, "a and b are required")
    winner = body.get("winner")
    if winner not in (None, "", a, b):
        raise LabError(400, "winner must be a, b, or null", str(winner))
    try:
        ms = int(body.get("ms") or 0)
    except (TypeError, ValueError):
        raise LabError(400, "ms must be a number", repr(body.get("ms")))
    with store_session() as store:
        store.add_compare(a, b, winner or None, ms, str(body.get("tag") or ""))
    return {"ok": True}


# --- image routes ----------------------------------------------------------
_VIEW_CACHE: dict[tuple[str, int, int], list[bytes]] = {}
_VIEW_LOCK = threading.Lock()


def _views(dataset_id: str, index: int, px: int) -> list[bytes]:
    """One `renders.views` call serves all four view requests from a detail panel.

    The browser fires them in parallel the moment the drawer opens; without this
    the same four-view render would run four times behind the renderer's lock.
    """
    key = (dataset_id, index, px)
    with _VIEW_LOCK:
        hit = _VIEW_CACHE.get(key)
    if hit is not None:
        return hit
    out = lab_module("renders").views(dataset_id, index, px=px)
    with _VIEW_LOCK:
        if len(_VIEW_CACHE) > 64:      # a drawer's worth of history, no more
            _VIEW_CACHE.clear()
        _VIEW_CACHE[key] = out
    return out


def _placeholder(px: int) -> bytes:
    """The renderer's "nothing to show" tile, for telling it apart from a build."""
    try:
        return lab_module("renders").placeholder(px)
    except Exception:
        return b""


def image_thumb(rest: str, query: Query) -> bytes:
    dataset_id, index = _build_id(rest)
    px = _int(query, "px", 256, lo=32, hi=1024)
    return lab_module("renders").thumb(dataset_id, index, px=px)


def image_view(rest: str, query: Query) -> bytes:
    build_part, _, k_part = rest.rpartition("/")
    if not build_part:
        raise LabError(400, "expected /api/view/<build_id>/<k>", rest)
    dataset_id, index = _build_id(build_part)
    try:
        k = int(k_part)
    except ValueError:
        raise LabError(400, "view index must be an integer", k_part)
    px = _int(query, "px", 320, lo=64, hi=1024)
    views = _views(dataset_id, index, px)
    if not views:
        raise LabError(404, "no views rendered", f"{dataset_id}:{index}")
    return views[k % len(views)]


# --- handler ---------------------------------------------------------------
NOT_FOUND_PAGE = """<!doctype html><meta charset="utf-8">
<title>Not here — BlockLab</title>
<link rel="stylesheet" href="/static/lab.css">
<main class="wrap"><div class="empty">
  <p><strong>{title}</strong></p>
  <p class="hint">{hint}</p>
  <p class="hint"><a href="/">Back to the hub</a></p>
</div></main>"""


class LabHandler(BaseHTTPRequestHandler):
    server_version = "BlockLab"
    sys_version = ""
    protocol_version = "HTTP/1.1"   # keep-alive; a grid pulls ~60 thumbs at once

    # -- plumbing --
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        pass  # replaced by the one-line log in `_finish`

    def _finish(self, status: int, started: float) -> None:
        ms = (time.perf_counter() - started) * 1000
        print(f"[lab] {self.command} {self.path} -> {status} {ms:.0f}ms",
              file=sys.stderr, flush=True)

    def _send(self, status: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any,
              extra: dict[str, str] | None = None) -> None:
        body = json.dumps(_clean(payload), allow_nan=False).encode()
        headers = {"Cache-Control": "no-store"}
        headers.update(extra or {})
        self._send(status, body, "application/json", headers)

    def _error(self, exc: BaseException) -> int:
        if isinstance(exc, LabError):
            self._json(exc.status, {"error": exc.error, "detail": exc.detail})
            return exc.status
        traceback.print_exc(file=sys.stderr)
        self._json(500, {"error": type(exc).__name__, "detail": str(exc)})
        return 500

    # -- static --
    def _static(self, rel: str) -> int:
        path = (STATIC / rel).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.is_file():
            return self._not_found(rel)
        ctype = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        # Sources change constantly while the lab is being built; never cache them.
        self._send(200, path.read_bytes(), ctype, {"Cache-Control": "no-cache"})
        return 200

    def _not_found(self, rel: str) -> int:
        owned_elsewhere = rel in ("leaderboard.html", "compare.html", "curation.html")
        hint = (f"<code>tools/lab/static/{rel}</code> has not been written yet."
                if owned_elsewhere else
                f"No route for <code>{rel}</code>.")
        body = NOT_FOUND_PAGE.format(title="Nothing here yet", hint=hint).encode()
        self._send(404, body, "text/html; charset=utf-8")
        return 404

    # -- dispatch --
    def do_GET(self) -> None:  # noqa: N802
        started = time.perf_counter()
        status = 500
        try:
            parsed = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(parsed.path)
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            status = self._get(path, query)
        except BrokenPipeError:
            status = 499  # the grid cancels in-flight thumbs when you scroll fast
        except BaseException as exc:  # noqa: BLE001
            status = self._error(exc)
        self._finish(status, started)

    def _get(self, path: str, query: Query) -> int:
        if path in PAGES:
            return self._static(PAGES[path])
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return 204

        if path.startswith("/api/thumb/"):
            px = _int(query, "px", 256, lo=32, hi=1024)
            return self._image(image_thumb(path[len("/api/thumb/"):], query), px)
        if path.startswith("/api/view/"):
            px = _int(query, "px", 320, lo=64, hi=1024)
            return self._image(image_view(path[len("/api/view/"):], query), px)

        if path.startswith("/api/ontology/swatch/"):
            return self._swatch(path[len("/api/ontology/swatch/"):])

        simple: dict[str, Callable[[Query], Any]] = {
            "/api/datasets": api_datasets,
            "/api/tree": api_tree,
            "/api/subsets": api_subsets,
            "/api/builds": api_builds,
            "/api/labels": api_labels,
            "/api/scorecards": api_scorecards,
            "/api/standings": api_standings,
            "/api/protocols": api_protocols,
            "/api/curation/preview": api_curation_preview,
            "/api/ontology": api_ontology,
            "/api/ontology/catalogs": api_ontology_catalogs,
            "/api/ontology/prompt": api_ontology_prompt,
        }
        if path in simple:
            self._json(200, simple[path](query))
            return 200
        if path == "/api/export":
            payload = api_export(query)   # validates `what` before it reaches a header
            self._json(200, payload, {
                "Content-Disposition":
                f'attachment; filename="lab_{payload["what"]}.json"'})
            return 200
        if path.startswith("/api/build/"):
            self._json(200, api_build(urllib.parse.unquote(path[len("/api/build/"):])))
            return 200
        if path.startswith("/api/ontology/part/"):
            self._json(200, api_ontology_part(path[len("/api/ontology/part/"):], query))
            return 200
        if path.startswith("/api/scorecard/"):
            self._json(200, api_scorecard(path[len("/api/scorecard/"):]))
            return 200
        if path.startswith("/api/"):
            raise LabError(404, "no such endpoint", path)
        return self._not_found(path.lstrip("/") or "index.html")

    def _swatch(self, rest: str) -> int:
        """A block's texture tile. 16x16 PNG, so no resizing and no render lock."""
        part_id = urllib.parse.unquote(rest).strip("/")
        try:
            body, ctype = lab_module("ontology").swatch(part_id)
        except FileNotFoundError as exc:
            raise LabError(404, "no texture for that block", str(exc)) from exc
        except Exception as exc:                              # noqa: BLE001
            raise LabError(400, "bad block name", f"{part_id!r}: {exc}") from exc
        # Textures change only when someone swaps the resource pack, which is a
        # restart-scale event; an hour of caching keeps a 70-swatch table cheap.
        self._send(200, body, ctype, {"Cache-Control": "public, max-age=3600"})
        return 200

    def _image(self, jpeg: bytes, px: int) -> int:
        if not jpeg:
            raise LabError(500, "renderer returned no bytes")
        # Content-addressed by (build, px): real pixels can never change for the
        # same request, so cache them for a year. The renderer's placeholder is
        # the exception -- it means "no EGL right now", a condition that gets
        # fixed, and a year-long cache of a grey square would outlive the fix.
        immutable = jpeg != _placeholder(px)
        self._send(200, jpeg, "image/jpeg", {
            "Cache-Control": "public, max-age=31536000, immutable" if immutable
            else "no-store"})
        return 200

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        started = time.perf_counter()
        status = 500
        try:
            path = urllib.parse.urlsplit(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise LabError(400, "body must be JSON", str(exc)) from exc
            if not isinstance(body, dict):
                raise LabError(400, "body must be a JSON object")
            routes: dict[str, Callable[[dict[str, Any]], Any]] = {
                "/api/label": post_label,
                "/api/note": post_note,
                "/api/compare": post_compare,
                "/api/subset": post_subset,
                "/api/subset/preview": post_subset_preview,
                "/api/subset/delete": post_subset_delete,
            }
            if path not in routes:
                raise LabError(404, "no such endpoint", path)
            self._json(200, routes[path](body))
            status = 200
        except BrokenPipeError:
            status = 499
        except BaseException as exc:  # noqa: BLE001
            status = self._error(exc)
        self._finish(status, started)


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind and return the server. 127.0.0.1 only -- this is a workstation tool
    with no auth, and the labels in it are unpublished research judgements."""
    server = ThreadingHTTPServer((host, port), LabHandler)
    server.daemon_threads = True
    return server
