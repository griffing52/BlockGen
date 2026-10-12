"""Core contract of the agentic methods suite: types, protocols, registry, ledger.

Every other suite module codes against THIS file, so it is deliberately free of
heavy imports (no torch, no pyrender, no diffusers, no anthropic) and of logic
that belongs to an implementation. Why one file: the suite is built by several
people in parallel, and a single, small, fully-specified contract is what lets
them do that without stepping on each other.

Concepts (see docs/agentic-suite.md for the user guide):

    MethodSpec   a recipe: one arm of an experiment, loaded from YAML
    Kit          the services a recipe resolves to (LLMs per role, image gen, ...)
    EpisodeLike  what a method function talks to: ask / try_ / render / dream / score / best
    Attempt      one executed candidate (subclass of the legacy Round)
    MethodResult a BuildResult subclass, so report.write_run / bench / deploy accept it

Extension = one decorated function or class:

    @register("scorer", "my_metric")
    def my_metric(ep, att, *, k: int = 3) -> ScoreTerm: ...
"""

from __future__ import annotations

import copy
import difflib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import threading
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Literal, Mapping, Optional,
                    Protocol, Sequence, Tuple, Union, runtime_checkable)
from urllib.parse import parse_qsl

import numpy as np

from blockgen.agentic.agent import BuildResult, Round
from blockgen.agentic.canvas import Canvas, structure_to_canvas
from blockgen.agentic.dsl import ExecutionReport, Issue, run_program
from blockgen.agentic.providers import LLMProvider, LLMResponse, Message, Part
from blockgen.utils.data import Structure

# ===========================================================================
# Registry
# ===========================================================================
#: Component kinds. Function kinds are bound with functools.partial(fn, **params);
#: class kinds are instantiated as cls(arg, **params) / cls(**params).
KINDS: Tuple[str, ...] = ("method", "actions", "image", "renderer",
                          "context", "scorer", "accept")
FUNCTION_KINDS = frozenset({"method", "context", "scorer", "accept"})

#: Modules whose import registers the built-in components. Each must be import-light
#: (heavy deps imported lazily inside functions).
BUILTIN_MODULES: Tuple[str, ...] = (
    "blockgen.agentic.actions", "blockgen.agentic.pyapi", "blockgen.agentic.imagegen",
    "blockgen.agentic.render", "blockgen.agentic.scoring", "blockgen.agentic.context",
    "blockgen.agentic.recipes",
)

Spec = Union[str, Mapping[str, Any], None]


@dataclass
class Entry:
    """One registered component. ``target`` may be a lazy 'pkg.mod:attr' string."""

    kind: str
    name: str
    target: Union[Callable[..., Any], str]
    meta: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Callable[..., Any]:
        if isinstance(self.target, str):
            mod, _, attr = self.target.partition(":")
            self.target = getattr(importlib.import_module(mod), attr)
        return self.target

    def doc(self) -> str:
        """One-line summary: ``summary=`` / ``doc=`` meta, else the docstring's first
        PARAGRAPH (a first line alone often ends mid-sentence), whitespace collapsed."""
        for key in ("summary", "doc"):
            if self.meta.get(key):
                return " ".join(str(self.meta[key]).split())
        if isinstance(self.target, str):
            return f"(lazy) {self.target}"
        para = (inspect.getdoc(self.target) or "").strip().split("\n\n", 1)[0]
        return " ".join(para.split())


_REG: Dict[str, Dict[str, Entry]] = {k: {} for k in KINDS}
_ALIASES: Dict[str, Dict[str, str]] = {k: {} for k in KINDS}
_BUILTINS_LOADED = False
_BUILTINS_LOADING = False
#: Built-in modules skipped by load_builtins (module -> why). Reported by lookup's
#: KeyError, so a missing component fails with its cause rather than a bare name.
_BUILTINS_SKIPPED: Dict[str, str] = {}
_LOCK = threading.Lock()
# Re-entrant: a built-in module that calls lookup() while being imported must not
# deadlock on (or re-trigger) the load it is part of.
_BUILTINS_LOCK = threading.RLock()


def _check_kind(kind: str) -> None:
    if kind not in _REG:
        raise ValueError(f"unknown component kind '{kind}'; kinds: {KINDS}")


def register(kind: str, name: str, *aliases: str, **meta: Any) -> Callable[[Any], Any]:
    """Decorator: make a function/class reachable from YAML as ``name``.

    Re-registering the same object (module re-import) is a no-op; registering a
    DIFFERENT object under a taken name raises, so two packages cannot silently
    shadow each other.
    """
    _check_kind(kind)

    def deco(obj: Any) -> Any:
        with _LOCK:
            old = _REG[kind].get(name)
            if old is not None and old.target is not obj and not isinstance(old.target, str):
                same = (getattr(old.target, "__qualname__", None) == getattr(obj, "__qualname__", None)
                        and getattr(old.target, "__module__", None) == getattr(obj, "__module__", None))
                if not same:
                    raise ValueError(f"{kind} '{name}' is already registered to {old.target!r}")
            _REG[kind][name] = Entry(kind, name, obj, dict(meta))
            for a in aliases:
                _ALIASES[kind][a] = name
        return obj
    return deco


def register_lazy(kind: str, name: str, target: str, *aliases: str, **meta: Any) -> None:
    """Register ``'pkg.mod:attr'`` without importing it (optional heavy deps)."""
    _check_kind(kind)
    with _LOCK:
        _REG[kind][name] = Entry(kind, name, target, dict(meta))
        for a in aliases:
            _ALIASES[kind][a] = name


def load_builtins() -> None:
    """Import BUILTIN_MODULES once.

    A built-in module that is itself absent (e.g. stripped from a release) is
    skipped with a warning and recorded in ``_BUILTINS_SKIPPED``. Any other import
    error, including a present module importing a missing sibling or third-party
    dependency eagerly, is raised: that is a bug.
    """
    global _BUILTINS_LOADED, _BUILTINS_LOADING
    if _BUILTINS_LOADED:
        return
    with _BUILTINS_LOCK:
        if _BUILTINS_LOADED or _BUILTINS_LOADING:
            return
        _BUILTINS_LOADING = True
        try:
            for mod in BUILTIN_MODULES:
                try:
                    importlib.import_module(mod)
                except ModuleNotFoundError as exc:
                    # Only a built-in module that is itself absent is skipped. A
                    # present module whose own import fails (a missing sibling or
                    # dependency) raises here, instead of turning every component
                    # it registers into a confusing 'unknown <kind>' later.
                    if (exc.name or "") != mod:
                        raise
                    missing = mod
                    _BUILTINS_SKIPPED[mod] = f"module {missing} not found"
                    warnings.warn(f"agentic suite: built-in module {mod} skipped "
                                  f"({missing} not present)")
            _BUILTINS_LOADED = True
        finally:
            _BUILTINS_LOADING = False
    # Out-of-tree components named by the environment (set by the CLI's --plugin),
    # so worker processes and the Python API see them too.
    if os.environ.get(PLUGINS_ENV, "").strip():
        load_plugins()


#: Extra component modules to import after the built-ins: ``os.pathsep``-separated
#: dotted module names or ``.py`` file paths. The CLI's ``--plugin`` sets it, so a
#: battery's child processes and ``experiments_agentic`` see the same components.
PLUGINS_ENV = "BLOCKGEN_AGENTIC_PLUGINS"
#: Plugin specs already imported (spec -> module name), so a repeat is a no-op.
_PLUGINS_LOADED: Dict[str, str] = {}


def load_plugins(specs: Union[str, Sequence[str], None] = None) -> List[str]:
    """Import out-of-tree component modules (a method, scorer, ... in a scratch dir).

    Why: without this, a new ``@register`` component only becomes reachable from
    YAML by editing ``BUILTIN_MODULES`` in the repo, so trying a variant meant
    touching shared code. A spec is a dotted module name or a path to a ``.py``
    file; ``None`` reads ``PLUGINS_ENV``. A missing file or module raises
    ConfigError naming the spec (a typo must not become a later "unknown method").
    Returns the imported module names.
    """
    if specs is None:
        specs = os.environ.get(PLUGINS_ENV, "")
    if isinstance(specs, str):
        specs = [s for s in specs.split(os.pathsep) if s.strip()]
    load_builtins()
    out: List[str] = []
    for raw in specs:
        spec = raw.strip()
        if spec.endswith(".py") or os.sep in spec or "/" in spec:
            path = Path(spec).expanduser().resolve()
            key = str(path)
            if key in _PLUGINS_LOADED:
                out.append(_PLUGINS_LOADED[key])
                continue
            if not path.is_file():
                raise ConfigError(f"plugin file not found: {spec}")
            mod_name = "blockgen_plugin_" + hashlib.sha1(key.encode()).hexdigest()[:8] + "_" + path.stem
            mspec = importlib.util.spec_from_file_location(mod_name, path)
            if mspec is None or mspec.loader is None:
                raise ConfigError(f"plugin {spec}: not an importable Python file")
            module = importlib.util.module_from_spec(mspec)
            sys.modules[mod_name] = module
            try:
                mspec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(mod_name, None)
                raise
        else:
            key, mod_name = spec, spec
            if key in _PLUGINS_LOADED:
                out.append(mod_name)
                continue
            try:
                importlib.import_module(spec)
            except ModuleNotFoundError as exc:
                if (exc.name or "") != spec and not spec.startswith((exc.name or "") + "."):
                    raise
                raise ConfigError(f"plugin module not found: {spec}") from exc
        _PLUGINS_LOADED[key] = mod_name
        out.append(mod_name)
    return out


def _suggest(word: str, options: Sequence[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=3, cutoff=0.5)
    return f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""


def _coerce(v: str) -> Any:
    low = v.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return v


def parse_spec(spec: Spec) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Component spec -> (name, arg, params).

    Accepted forms (identical in YAML, CLI and code)::

        "dsl"                                  -> ("dsl", None, {})
        "openai:gpt-image-1-mini?quality=low"  -> ("openai", "gpt-image-1-mini", {"quality": "low"})
        "embed?backbone=siglipB&reduce=max"    -> ("embed", None, {...})
        {"type": "texture_sheet", "tile": 64}  -> ("texture_sheet", None, {"tile": 64})
        {"texture_sheet": {"tile": 64}}        -> same
    """
    if spec is None:
        raise ValueError("empty component spec")
    if isinstance(spec, Mapping):
        d = dict(spec)
        if "type" in d:
            name = str(d.pop("type"))
            arg = d.pop("arg", None)
            return name, (None if arg is None else str(arg)), d
        if len(d) == 1:
            name, params = next(iter(d.items()))
            if params is not None and not isinstance(params, Mapping):
                raise ValueError(f"spec {{{name}: ...}} needs a mapping of params, "
                                 f"got {params!r}")
            return str(name), None, dict(params or {})
        raise ValueError(f"mapping spec needs 'type' or exactly one key: {spec!r}")
    head, _, query = str(spec).strip().partition("?")
    name, sep, arg = head.partition(":")
    params = {k: _coerce(v) for k, v in parse_qsl(query, keep_blank_values=True)}
    return name.strip(), (arg.strip() if sep else None), params


def spec_str(spec: Spec) -> str:
    """Canonical one-line string for a spec (provenance / logs)."""
    name, arg, params = parse_spec(spec)
    out = name + (f":{arg}" if arg else "")
    if params:
        out += "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return out


def lookup(kind: str, name: str) -> Entry:
    _check_kind(kind)
    load_builtins()
    name = _ALIASES[kind].get(name, name)
    if name not in _REG[kind]:
        known = names(kind)
        skipped = (f"; built-in modules not loaded: {_BUILTINS_SKIPPED}"
                   if _BUILTINS_SKIPPED else "")
        hint = (f"; defined outside blockgen? pass --plugin <file.py|module> "
                f"(or set {PLUGINS_ENV})")
        raise KeyError(f"unknown {kind} '{name}'{_suggest(name, known + list(_ALIASES[kind]))}; "
                       f"registered: {known}{skipped}{hint}")
    return _REG[kind][name]


def validate_params(fn: Callable[..., Any], params: Mapping[str, Any], *, where: str) -> None:
    """Raise TypeError naming the valid keys when ``params`` has a typo.

    Why: an unknown YAML key silently running the default is the classic way a
    sweep "shows no effect".
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    valid = {n for n, p in sig.parameters.items()
             if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)}
    bad = sorted(set(params) - valid)
    if bad:
        hints = "".join(_suggest(b, sorted(valid)) for b in bad)
        raise TypeError(f"{where}: unknown parameter(s) {bad}{hints}; valid: {sorted(valid)}")


def make(kind: str, spec: Spec, **inject: Any) -> Any:
    """Build a component from a spec.

    Function kinds return ``functools.partial(fn, **params)`` (with ``.spec_name``
    set). Class kinds return an instance; ``inject`` kwargs are passed only when
    the constructor accepts them (so e.g. ``size=`` reaches only what wants it).
    """
    name, arg, params = parse_spec(spec)
    entry = lookup(kind, name)
    target = entry.load()
    validate_params(target, params, where=f"{kind} '{name}'")
    if kind in FUNCTION_KINDS:
        if arg is not None:
            raise ValueError(f"{kind} '{name}' takes no ':arg' (got {arg!r})")
        bound = functools.partial(target, **params)
        bound.spec_name = name  # type: ignore[attr-defined]
        return bound
    try:
        accepted = inspect.signature(target).parameters
        kw_ok = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
        extra = {k: v for k, v in inject.items() if kw_ok or k in accepted}
    except (TypeError, ValueError):
        extra = {}
    kwargs = {**extra, **params}
    return target(arg, **kwargs) if arg is not None else target(**kwargs)


def names(kind: str) -> List[str]:
    _check_kind(kind)
    load_builtins()
    return sorted(_REG[kind])


def describe(kind: str) -> List[Tuple[str, str]]:
    """(name, first docstring line) for ``python -m blockgen.agentic list``."""
    return [(n, _REG[kind][n].doc()) for n in names(kind)]


def meta(kind: str, name: str) -> Dict[str, Any]:
    return dict(lookup(kind, name).meta)


# ===========================================================================
# Spend gate, cache modes, budget
# ===========================================================================
SPEND_ENV = "BLOCKGEN_ALLOW_SPEND"
CacheMode = Literal["off", "read_write", "read_only"]
CACHE_MODES: Tuple[str, ...] = ("off", "read_write", "read_only")


class SpendNotAllowed(RuntimeError):
    """A paying/quota backend was about to be called without permission."""


class CacheMiss(RuntimeError):
    """cache='read_only' and the request is not in the cache."""


class ConfigError(ValueError):
    """A recipe / battery / CLI configuration error (typo, unknown name, bad param).

    Raised before any call is made; the CLI prints it as one line and exits 2.
    """


class BudgetExceeded(RuntimeError):
    """A ledger cap would be crossed by the next call. Raised BEFORE the call."""


def spend_allowed() -> bool:
    return os.environ.get(SPEND_ENV, "") == "1"


def require_spend(what: str) -> None:
    """Called by the caching wrappers on a cache MISS of a backend with ``spends=True``."""
    if not spend_allowed():
        raise SpendNotAllowed(
            f"{what}: cache miss on a paying/quota backend. Set {SPEND_ENV}=1 or pass "
            f"--allow-spend (tests never do; use mock backends).")


def billing_of(obj: Any) -> str:
    """'api' | 'subscription' | 'local' | 'mock' for a (possibly cache-wrapped) backend."""
    inner = getattr(obj, "inner", obj)
    return str(getattr(inner, "billing", getattr(obj, "billing", "api")))


@dataclass
class Charge:
    """One metered backend call. ``usd`` is what was PAID (0 on a cache hit);
    ``would_cost_usd`` is the full price, the honest figure for compute comparisons."""

    role: str
    kind: Literal["llm", "image"]
    model: str
    usd: float = 0.0
    would_cost_usd: float = 0.0
    priced: bool = True
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    images: int = 0
    latency_s: float = 0.0
    billing: str = "api"

    @classmethod
    def from_llm(cls, role: str, resp: LLMResponse, billing: str = "api") -> "Charge":
        full = float(resp.cost_usd)
        return cls(role=role, kind="llm", model=resp.model,
                   usd=0.0 if resp.cached else full, would_cost_usd=full,
                   priced=bool(resp.priced), cached=bool(resp.cached),
                   prompt_tokens=int(resp.prompt_tokens),
                   completion_tokens=int(resp.completion_tokens),
                   latency_s=float(resp.latency_s), billing=billing)

    @classmethod
    def from_image(cls, role: str, res: "ImageResult", billing: str = "api") -> "Charge":
        return cls(role=role, kind="image", model=res.model,
                   usd=0.0 if res.cached else float(res.cost_usd),
                   would_cost_usd=float(res.cost_usd), priced=bool(res.priced),
                   cached=bool(res.cached), images=len(res.images),
                   latency_s=float(res.latency_s), billing=billing)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    """Caps for one ledger level. None = uncapped. Calls/images count cache MISSES only."""

    usd: Optional[float] = None
    max_llm_calls: Optional[int] = None
    max_images: Optional[int] = None


class Ledger:
    """Thread-safe, hierarchical (build -> arm -> battery) cost meter."""

    def __init__(self, budget: Optional[Budget] = None, parent: Optional["Ledger"] = None,
                 name: str = "") -> None:
        self.budget = budget or Budget()
        self.parent = parent
        self.name = name
        self.charges: List[Charge] = []
        self._lock = threading.Lock()

    def child(self, budget: Optional[Budget] = None, name: str = "") -> "Ledger":
        return Ledger(budget, parent=self, name=name)

    @property
    def spent_usd(self) -> float:
        return sum(c.usd for c in self.charges)

    @property
    def would_cost_usd(self) -> float:
        return sum(c.would_cost_usd for c in self.charges)

    @property
    def n_llm_calls(self) -> int:
        return sum(1 for c in self.charges if c.kind == "llm" and not c.cached)

    @property
    def n_images(self) -> int:
        return sum(c.images for c in self.charges if c.kind == "image" and not c.cached)

    @property
    def cost_known(self) -> bool:
        return bool(self.charges) and all(c.priced for c in self.charges)

    def check(self, kind: Literal["llm", "image"], est_usd: Optional[float] = 0.0) -> None:
        """Raise BudgetExceeded if the NEXT call would cross a cap at any level."""
        b = self.budget
        if b.usd is not None and self.spent_usd + float(est_usd or 0.0) > b.usd + 1e-12:
            raise BudgetExceeded(f"ledger '{self.name}': ${self.spent_usd:.4f} spent + "
                                 f"${float(est_usd or 0):.4f} next > cap ${b.usd:.4f}")
        if kind == "llm" and b.max_llm_calls is not None and self.n_llm_calls >= b.max_llm_calls:
            raise BudgetExceeded(f"ledger '{self.name}': LLM call cap {b.max_llm_calls} reached")
        if kind == "image" and b.max_images is not None and self.n_images >= b.max_images:
            raise BudgetExceeded(f"ledger '{self.name}': image cap {b.max_images} reached")
        if self.parent is not None:
            self.parent.check(kind, est_usd)

    def charge(self, c: Charge) -> None:
        with self._lock:
            self.charges.append(c)
        if self.parent is not None:
            self.parent.charge(c)

    def summary(self) -> Dict[str, Any]:
        """Per-role and total paid / full-price cost. cost_known=False if any call unpriced."""
        roles: Dict[str, Dict[str, Any]] = {}
        for c in self.charges:
            r = roles.setdefault(c.role, {"usd": 0.0, "would_cost_usd": 0.0, "calls": 0,
                                          "images": 0, "cached": 0, "priced": True,
                                          "billing": set()})
            r["usd"] += c.usd
            r["would_cost_usd"] += c.would_cost_usd
            r["calls"] += 1
            r["images"] += c.images
            r["cached"] += int(c.cached)
            r["priced"] = r["priced"] and c.priced
            r["billing"].add(c.billing)
        for r in roles.values():
            r["cached_frac"] = round(r.pop("cached") / max(1, r["calls"]), 4)
            r["cost_known"] = r.pop("priced")
            r["billing"] = sorted(r["billing"])
            r["usd"] = round(r["usd"], 6)
            r["would_cost_usd"] = round(r["would_cost_usd"], 6)
        return {"usd": round(self.spent_usd, 6), "would_cost_usd": round(self.would_cost_usd, 6),
                "cost_known": self.cost_known, "llm_calls": self.n_llm_calls,
                "images": self.n_images, "roles": roles}


# ===========================================================================
# Views and dreams (data)
# ===========================================================================
@dataclass(frozen=True)
class View:
    """A camera plus an optional structure transform applied BEFORE rendering.

    ``keep_below``: keep the bottom fraction of the (cropped) build's height, so
    0.6 is an interior cutaway with the roof removed and 0.3 with a top-down camera
    is a ground-floor plan. ``section``: ('x'|'z', f) keeps voxels in the first
    fraction f of that axis (a vertical cut). Transforms mean no renderer changes.
    """

    name: str = "iso"
    azim: float = 45.0
    elev: float = 30.0
    px: int = 384
    keep_below: Optional[float] = None
    section: Optional[Tuple[str, float]] = None

    def apply(self, s: Structure) -> Structure:
        if self.keep_below is None and self.section is None:
            return s
        s = s.crop_to_non_air()
        ids, data = s.block_ids.copy(), s.block_data.copy()
        if self.keep_below is not None:
            cut = max(1, int(math.ceil(ids.shape[1] * float(self.keep_below))))
            ids[:, cut:, :] = s.air_block_id
            data[:, cut:, :] = 0
        if self.section is not None:
            axis = {"x": 0, "z": 2}[self.section[0]]
            cut = max(1, int(math.ceil(ids.shape[axis] * float(self.section[1]))))
            sl = [slice(None)] * 3
            sl[axis] = slice(cut, None)
            ids[tuple(sl)] = s.air_block_id
            data[tuple(sl)] = 0
        return Structure(ids, data, air_block_id=s.air_block_id, metadata=dict(s.metadata))


#: Named views. Add one line to add a view everywhere (dreams, scorers, critique).
VIEWS: Dict[str, View] = {
    "iso":       View("iso", 45, 30),
    "iso_back":  View("iso_back", 225, 30),
    "iso_left":  View("iso_left", 135, 30),
    "iso_right": View("iso_right", 315, 30),
    "front":     View("front", 0, 10),
    "side":      View("side", 90, 10),
    "top":       View("top", 0, 89),
    "cutaway":   View("cutaway", 45, 55, keep_below=0.6),
    "plan":      View("plan", 0, 89, keep_below=0.3),
    "section_x": View("section_x", 90, 20, section=("x", 0.5)),
    # BuildAgent's fixed critique views (agent.CRITIQUE_VIEWS), for classic parity.
    "legacy_a":  View("legacy_a", 45, 25),
    "legacy_b":  View("legacy_b", 225, 25),
    # Unit cutaways (staged methods). The structure is already masked to one room
    # or storey (region.region_structure), so these carry only a camera: steep
    # enough to see over walls cut at half height, and straight down.
    "room":      View("room", 45, 55),
    "room_top":  View("room_top", 0, 89),
}


def get_view(v: Union[str, View]) -> View:
    if isinstance(v, View):
        return v
    if v not in VIEWS:
        raise KeyError(f"unknown view '{v}'; known: {sorted(VIEWS)}")
    return VIEWS[v]


@dataclass(frozen=True)
class DreamSpec:
    """What to imagine, and how it is compared to the build.

    ``template`` fields: {desc} {style} {W} {H} {D}. ``view`` names the VIEWS entry
    the dream is scored against (None = shown to the builder, never scored).
    ``condition_on`` lists other dream keys passed as reference images, which keeps
    multi-view dreams consistent. ``view_search`` adds candidate views; that dream's
    term is the max over (view, *view_search), absorbing the image model's
    approximate camera.
    """

    key: str
    template: str
    view: Optional[str] = "iso"
    weight: float = 1.0
    condition_on: Tuple[str, ...] = ()
    show_builder: bool = True
    view_search: Tuple[str, ...] = ()


@dataclass
class DreamVersion:
    """One generated dream image. Re-generations append versions (never overwrite)."""

    key: str
    version: int
    png: bytes
    prompt: str
    generator: str
    seed: int = 0
    critique: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.png).hexdigest()[:16]


# ===========================================================================
# Image generation
# ===========================================================================
@dataclass(frozen=True)
class ImageRequest:
    """``seed`` is a cache-key / sample index: API backends (gpt-image) ignore it,
    so reproducibility for them comes from the cache, not from regeneration."""

    prompt: str
    refs: Tuple[bytes, ...] = ()
    n: int = 1
    size: str = "1024x1024"
    quality: str = "low"
    seed: int = 0
    view: str = ""                 # VIEWS key the dream targets (mock backends render it)
    background: str = "white"

    def key(self, backend: str, model: str) -> str:
        payload = {"backend": backend, "model": model, "prompt": self.prompt,
                   "refs": [hashlib.sha256(r).hexdigest()[:32] for r in self.refs],
                   "n": self.n, "size": self.size, "quality": self.quality,
                   "seed": self.seed, "view": self.view, "background": self.background}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class ImageResult:
    images: List[bytes]
    model: str
    cost_usd: float = 0.0          # full price of this request (0 for local/mock)
    priced: bool = True
    cached: bool = False
    latency_s: float = 0.0
    revised_prompt: str = ""


@runtime_checkable
class ImageGen(Protocol):
    """Implementation-hiding image generator: 'openai:...', 'diffusers:...', 'mock:...'."""

    name: str
    model: str
    spends: bool          # True for paid APIs: cache misses need the spend gate
    billing: str          # 'api' | 'local' | 'mock'

    def generate(self, req: ImageRequest) -> ImageResult: ...
    def estimate_usd(self, req: ImageRequest) -> Optional[float]: ...   # None = unpriced
    def supports_refs(self) -> bool: ...
    def describe(self) -> str: ...


# ===========================================================================
# Action spaces and worlds
# ===========================================================================
@dataclass
class PromptCtx:
    """What an ActionSpace needs to write its grammar/instructions."""

    size: Tuple[int, int, int]
    palette: Sequence[str]
    prompt: str = ""
    target_blocks: Optional[int] = None


@dataclass
class Lowered:
    """Model text lowered to the single IR: canonical DSL, one command per line."""

    source: str
    program_text: str
    issues: List[Issue] = field(default_factory=list)   # lowering errors; line_no = SOURCE line
    n_source_lines: int = 0


@runtime_checkable
class ActionSpace(Protocol):
    """How the model expresses a build. Everything lowers to DSL text."""

    name: str
    patchable: bool       # whether a reply may be applied on top of the current world

    def instructions(self, ctx: PromptCtx) -> str: ...      # grammar + rules (system prompt)
    def output_rule(self) -> str: ...                       # one stable sentence ending each build turn
    def examples(self, query: str, k: int) -> List[Tuple[str, str]]: ...  # (request, answer IN THIS FORMAT)
    def lower(self, text: str) -> Lowered: ...              # never raises on bad model text
    def mock_reply(self) -> str: ...                        # a valid small house, for mock:auto


@runtime_checkable
class World(Protocol):
    size: Tuple[int, int, int]

    def run(self, lowered: Lowered) -> ExecutionReport: ...   # mutate in place
    def clone(self) -> "World": ...
    def to_structure(self, crop: bool = True) -> Structure: ...
    def block_count(self) -> int: ...


@dataclass
class MinecraftWorld:
    """Legacy-ID Minecraft world over :class:`Canvas` + the DSL executor.

    ``clone`` copies the clipboard (legacy ``Canvas.clone`` does not, see
    docs/agentic-suite.md#legacy); this fixes patch-mode ``paste`` for the new
    core only, leaving published legacy behaviour untouched.
    """

    canvas: Canvas

    @property
    def size(self) -> Tuple[int, int, int]:
        return tuple(self.canvas.size)  # type: ignore[return-value]

    @classmethod
    def empty(cls, size: Sequence[int]) -> "MinecraftWorld":
        return cls(Canvas(size))

    @classmethod
    def from_structure(cls, s: Structure, size: Optional[Sequence[int]] = None) -> "MinecraftWorld":
        return cls(structure_to_canvas(s, size))

    def run(self, lowered: Lowered) -> ExecutionReport:
        _, report, _ = run_program(lowered.program_text, canvas=self.canvas, size=self.size)
        report.issues = list(lowered.issues) + list(report.issues)
        return report

    def clone(self) -> "MinecraftWorld":
        c = self.canvas.clone()
        c.clipboard = copy.deepcopy(self.canvas.clipboard)
        return MinecraftWorld(c)

    def to_structure(self, crop: bool = True) -> Structure:
        return self.canvas.to_structure(crop=crop)

    def block_count(self) -> int:
        return int(self.canvas.block_count())


# ===========================================================================
# Rendering
# ===========================================================================
class EmptyBuild(ValueError):
    """Rendering an empty structure. Callers score empties as 0 without rendering."""


class RenderError(RuntimeError):
    """Rendering failed. Never swallowed into 'no critique' (cf. agent._render)."""


@runtime_checkable
class Renderer(Protocol):
    name: str

    def render(self, structure: Structure, views: Sequence[View]) -> List[np.ndarray]: ...
    # -> one (px, px, 3) uint8 RGB per view, white background; View.apply already
    #    handled by the renderer; raises EmptyBuild / RenderError.


# ===========================================================================
# Scores, attempts, contexts
# ===========================================================================
@dataclass
class ScoreTerm:
    """One scorer's output. ``value`` nominally in [0, 1]; ``note`` may be fed back."""

    value: float
    per_view: Dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class Score:
    """Weighted composite of terms after gates. ``dream_id`` = the dream set it was
    measured against ('' = dream-independent); scores with different dream_ids are
    NOT comparable, which is why Episode.rescore_all exists."""

    value: float
    parts: Dict[str, float] = field(default_factory=dict)
    per_view: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    dream_id: str = ""
    gated: str = ""                # name of the gate that zeroed the score, if any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Attempt(Round):
    """One executed candidate. Subclasses the legacy Round (stage, program_text,
    report, response, blocks) so every Round reader accepts it.

    ``program_text`` = CUMULATIVE canonical DSL that rebuilds ``world`` from empty
    (deploy replay); ``delta_text`` = this attempt's lowered DSL only;
    ``source_text`` = the raw model reply.
    """

    idx: int = 0
    source_text: str = ""
    delta_text: str = ""
    patch: bool = False
    world: Optional[Any] = None                 # World snapshot after this attempt
    renders: Dict[str, bytes] = field(default_factory=dict)   # view name -> PNG
    score: Optional[Score] = None
    accepted: bool = False
    reason: str = ""
    #: The value :meth:`Episode.best` ranked this attempt by (final_score when the
    #: recipe has one), set by ``Episode.result``. Kept separate from ``score``
    #: because the contact sheet must show what selection actually used.
    selection: Optional[float] = None
    #: Method annotations, e.g. ``{"unit": "kitchen", "unit_score": 0.41,
    #: "reverted": 12, "region": "u_kitchen"}``. JSON-able values only.
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {"idx": self.idx, "stage": self.stage, "blocks": self.blocks,
                "patch": self.patch, "accepted": self.accepted, "reason": self.reason,
                "score": self.score.to_dict() if self.score else None,
                "selection": self.selection, "extra": dict(self.extra),
                "n_commands": self.report.n_commands, "n_failed": self.report.n_failed}


@dataclass
class ContextParts:
    """Context for the model: text goes in the system prompt; images go in the
    first build turn (Anthropic / Claude Code system prompts cannot hold images)."""

    system_text: str = ""
    user_parts: List[Part] = field(default_factory=list)


#: Function-kind signatures (params bound from YAML via functools.partial).
ScoreFn = Callable[["EpisodeLike", Attempt], ScoreTerm]
AcceptFn = Callable[[Optional[Attempt], Attempt], Tuple[bool, str]]
ContextFn = Callable[["EpisodeLike"], ContextParts]
MethodFn = Callable[["EpisodeLike"], None]


# ===========================================================================
# Trace
# ===========================================================================
@runtime_checkable
class TraceLike(Protocol):
    root: Optional[Path]

    def asset(self, png: bytes) -> str: ...          # content-addressed relative path
    def event(self, kind: str, **data: Any) -> None: ...
    def request(self, role: str, messages: Sequence[Message], response: LLMResponse,
                stage: str) -> None: ...
    def dream(self, d: DreamVersion) -> None: ...
    def attempt(self, a: Attempt) -> None: ...
    def close(self, result: "MethodResult") -> None: ...  # writes contact.png, result.json


class NullTrace:
    """Trace that writes nothing (unit tests, estimate mode)."""

    root: Optional[Path] = None

    def asset(self, png: bytes) -> str:
        return ""

    def event(self, kind: str, **data: Any) -> None:
        pass

    def request(self, role: str, messages: Sequence[Message], response: LLMResponse,
                stage: str) -> None:
        pass

    def dream(self, d: DreamVersion) -> None:
        pass

    def attempt(self, a: Attempt) -> None:
        pass

    def close(self, result: "MethodResult") -> None:
        pass


# ===========================================================================
# Recipe (MethodSpec) and Kit
# ===========================================================================
ROLES: Tuple[str, ...] = ("builder", "planner", "critic", "judge", "dream_critic")


def _default_gen() -> Dict[str, Any]:
    # Same three keys as AgentConfig.provider_params(), so CachingProvider keys
    # for legacy-equivalent recipes are identical and old cache entries hit.
    return {"max_tokens": 16000, "temperature": None, "reasoning_effort": "medium"}


#: How a build's result is picked from its attempts. ``last``: the last accepted
#: attempt. ``best``: argmax of the selection value over every non-empty attempt,
#: rejected ones included (keep-best is separate from acceptance). ``best_accepted``:
#: the same argmax over accepted attempts only. Staged methods use it: their final
#: exterior pass can damage a finished build in ways the dream-anchored accept
#: policy catches but a dream-independent selector (clip_text) rewards.
SELECT_MODES = ("last", "best", "best_accepted")


@dataclass
class MethodSpec:
    """One arm. Loaded from configs/agentic/recipes/<name>.yaml by recipe.load_recipe."""

    name: str
    method: str = "classic"
    description: str = ""
    llm: Union[str, Dict[str, str]] = "openai:gpt-5-mini"   # str = all roles; dict = per role
    image: Optional[str] = None
    actions: Any = "dsl"
    contexts: List[Any] = field(default_factory=lambda: ["palette"])
    renderer: str = "textured"
    dreams: List[Any] = field(default_factory=list)          # DREAMS keys or inline DreamSpec dicts
    dream_style: Optional[str] = None                        # None = dreams.DEFAULT_STYLE
    score: Dict[str, Any] = field(default_factory=dict)      # scorer spec -> weight | {weight, **params}
    gates: Dict[str, float] = field(default_factory=dict)    # scorer -> min term value
    final_score: Dict[str, Any] = field(default_factory=dict)  # dream-independent selector ({} = score)
    accept: Any = "nonempty_legacy"
    select: Literal["last", "best", "best_accepted"] = "last"
    canvas: Tuple[int, int, int] = (32, 32, 32)
    gen: Dict[str, Any] = field(default_factory=_default_gen)
    params: Dict[str, Any] = field(default_factory=dict)     # -> method function kwargs
    budget: Dict[str, Optional[float]] = field(default_factory=dict)  # usd_per_build, max_llm_calls, max_images
    cache: str = "read_write"
    seed: int = 0
    keep_history: bool = True
    target_blocks: Optional[int] = None
    tags: Dict[str, Any] = field(default_factory=dict)       # provenance overrides (e.g. train_split)
    #: Sampling salt. Non-empty = this arm draws fresh image and LLM samples
    #: (image seeds shift, every system message gains a '[sample <salt>]' line),
    #: so it no longer shares cached samples with its unsalted twin. Use it for
    #: variance estimates; leave it empty for arms meant to be compared paired.
    salt: str = ""

    def roles(self) -> Dict[str, str]:
        """Role -> backend spec, with fallbacks to 'builder' resolved."""
        if isinstance(self.llm, str):
            return {r: self.llm for r in ROLES}
        if not self.llm:
            raise ValueError(f"recipe '{self.name}': llm mapping is empty")
        builder = self.llm.get("builder") or next(iter(self.llm.values()))
        return {r: self.llm.get(r, builder) for r in ROLES}

    def resolved(self) -> Dict[str, Any]:
        """Canonical JSON-able dict (defaults filled). Basis of recipe_hash."""
        d = asdict(self)
        d["canvas"] = list(self.canvas)
        d["gen"] = {**_default_gen(), **self.gen}
        d["roles"] = self.roles()
        if not d.get("salt"):
            # Absent when empty, so every recipe hash from before the field existed
            # is unchanged (cache keys and provenance stay comparable).
            d.pop("salt", None)
        return json.loads(json.dumps(d, sort_keys=True, default=str))

    def hash(self) -> str:
        """sha256 of the resolved spec minus name/description/tags (cosmetic)."""
        d = {k: v for k, v in self.resolved().items() if k not in ("name", "description", "tags")}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def provenance(self) -> Dict[str, Any]:
        """Fields promoted into the npz manifest 'report' and metrics.json."""
        ctx_names = [parse_spec(c)[0] for c in self.contexts]
        uses_corpus = any(n == "examples" and parse_spec(c)[2].get("pool", "builtin") != "builtin"
                          for n, c in zip(ctx_names, self.contexts))
        out = {
            "method": self.method,
            "recipe": self.name,
            "recipe_hash": self.hash(),
            "action_repr": spec_str(self.actions),
            "backend": self.roles(),
            "image_generator": self.image or "none",
            "renderer": self.renderer,
            "contexts": [spec_str(c) for c in self.contexts],
            "dreams": [d if isinstance(d, str) else dict(d).get("key", "?") for d in self.dreams],
            "loop": {"accept": spec_str(self.accept), "select": self.select,
                     "score": self.score, "gates": self.gates, **self.params},
            "train_split": "train_only" if uses_corpus else "none",
        }
        if self.salt:
            out["salt"] = self.salt
        out.update(self.tags)
        return out


@dataclass
class Kit:
    """Services resolved from a MethodSpec, cached across the prompts of one arm."""

    spec: MethodSpec
    llms: Dict[str, LLMProvider]                       # role -> provider (fallbacks resolved)
    image: Optional[ImageGen]
    actions: ActionSpace
    renderer: Optional[Renderer]
    contexts: List[Tuple[str, ContextFn]]
    scorers: Dict[str, Tuple[ScoreFn, float]]          # name -> (fn, weight)
    final_scorers: Dict[str, Tuple[ScoreFn, float]]
    accept: AcceptFn
    method: MethodFn
    dreams: List[DreamSpec]
    ledger: Ledger                                     # arm-level; episodes take children
    world_factory: Callable[[Tuple[int, int, int]], World] = MinecraftWorld.empty
    mock: bool = False
    #: Cost-estimate mode: every optional paid step runs (e.g. all repair rounds),
    #: so ``suite.estimate`` is an upper bound. Never set for a real run.
    worst_case: bool = False

    def llm(self, role: str) -> LLMProvider:
        return self.llms.get(role) or self.llms["builder"]


# ===========================================================================
# Episode protocol (implemented by blockgen.agentic.episode.Episode)
# ===========================================================================
class EpisodeLike(Protocol):
    """Everything a method may touch (the shipped recipes use nothing else).
    Helpers live in recipes/steps.py, not here, so methods do not couple to
    internals. Method params arrive as keyword arguments, not via the episode."""

    prompt: str
    spec: MethodSpec
    kit: Kit
    seed: int                                          # effective seed (spec.seed if none given)
    variation: Optional[int]                           # the EXPLICIT seed, None if none was given
    size: Tuple[int, int, int]                         # canvas (== spec.canvas)
    select: str                                        # 'last' | 'best'; best_of_n overrides
    dream_desc: Dict[str, str]                         # current description per dream key
    world: World                                       # current ACCEPTED world
    dreams: Dict[str, List[DreamVersion]]              # key -> versions (latest last)
    reference_images: List[bytes]                      # user-supplied conditioning images
    attempts: List[Attempt]
    history: List[Message]                             # accepted dialogue (excl. system)
    plan: str
    notes: Dict[str, Any]
    ledger: Ledger                                     # build-level child ledger
    trace: TraceLike

    def system(self) -> Message: ...
    def ask(self, turn: Union[str, Message], *, images: Sequence[bytes] = (),
            role: str = "builder", history: bool = True, json_reply: bool = False,
            stage: str = "") -> Any: ...
    #   -> str, or dict when json_reply (tolerant JSON extraction; {} on failure).
    #   Metered (ledger.check before, charge after), traced. Context user_parts are
    #   prepended to the FIRST builder turn automatically.
    def try_(self, stage: str, text: str, *, patch: bool = False,
             region: Optional[Any] = None, offset: Optional[Tuple[int, int, int]] = None,
             ir: bool = False, accept: Optional[AcceptFn] = None) -> Attempt: ...
    #   lower -> run on fresh world (or clone of current if patch) -> render score
    #   views -> score -> accept(prev, cand) -> if accepted: world = cand.world and the
    #   last (turn, reply) pair joins history (if spec.keep_history). Never raises on
    #   bad model text. ``region`` (region.Region): writes outside it are reverted to
    #   the current world and reported as warnings; ``offset``: the text is in
    #   region-local coordinates; ``ir``: the text is canonical DSL (no action-space
    #   lowering, never commits an LLM pair); ``accept`` overrides the recipe policy.
    def render(self, target: Union[World, Attempt, Structure],
               views: Union[str, Sequence[str]] = "score") -> Dict[str, bytes]: ...
    #   'score' = union of scored dream views (+ view_search) or ['iso','iso_back'] when no dreams.
    def dream(self, key: str, *, prompt: Optional[str] = None, refs: Sequence[bytes] = (),
              seed: Optional[int] = None, store: bool = True,
              view_hint: Optional[str] = None) -> DreamVersion: ...
    def register_dream(self, ds: DreamSpec, *, trace_view: Optional[str] = None
                       ) -> DreamSpec: ...
    #   runtime dream (traced, metered) that is never scored, never in dream_id and
    #   never in the builder's TARGET prefix (per-room dreams, a turnaround sheet).
    def put_dream(self, key: str, png: bytes, *, prompt: str = "",
                  generator: str = "derived",
                  critique: Optional[Dict[str, Any]] = None) -> DreamVersion: ...
    #   append a version without calling the generator (no charge).
    def dream_spec(self, key: str) -> DreamSpec: ...    # KeyError naming the recipe's dreams
    def latest_dream(self, key: str) -> Optional[DreamVersion]: ...
    def note_target_changed(self, desc: str = "") -> None: ...  # after a re-dream (steps.redream)
    def dream_id(self) -> str: ...                     # digest over latest versions of scored dreams
    def score(self, att: Attempt) -> Score: ...
    def rescore_all(self) -> None: ...                 # re-anchor every attempt to the current dream_id
    def current(self) -> Optional[Attempt]: ...        # last accepted
    def best(self) -> Optional[Attempt]: ...           # per spec.select (+ final_score)
    select_from: int                                   # best() ranks attempts[select_from:]
    salt: str                                          # spec.salt (see MethodSpec.salt)
    def new_branch(self) -> None: ...                  # fresh world + dialogue, attempts kept
    def result(self) -> "MethodResult": ...

    # Legacy-parity hooks: used only by the classic / legacy methods to send
    # BuildAgent's exact request bytes. A new method should not need them.
    inject_first_turn: bool                            # False = no context/TARGET prefix
    config: Any                                        # the AgentConfig (legacy method)
    transcript: List[Dict[str, Any]]
    def set_system(self, msg: Message) -> None: ...
    def adopt(self, att: Attempt, *, accepted: bool, reason: str = "") -> Attempt: ...


# ===========================================================================
# Result
# ===========================================================================
@dataclass
class MethodResult(BuildResult):
    """BuildResult + suite extras. ``rounds`` holds the Attempts. Cost comes from
    ``charges`` (every LLM role AND image call), not only builder rounds."""

    spec_name: str = ""
    recipe_hash: str = ""
    dreams: Dict[str, List[DreamVersion]] = field(default_factory=dict)
    charges: List[Charge] = field(default_factory=list)
    best_idx: Optional[int] = None
    truncated: str = ""                 # '' | 'budget' | 'cache_miss' | 'error'
    trace_dir: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def attempts(self) -> List[Attempt]:
        return [r for r in self.rounds if isinstance(r, Attempt)]

    @property
    def cost_usd(self) -> float:
        """PAID usd (0 for a cache hit). NB metrics()['cost_usd'] is the full price,
        like legacy BuildResult; metrics()['paid_usd'] is this value."""
        return sum(c.usd for c in self.charges)

    @property
    def would_cost_usd(self) -> float:
        return sum(c.would_cost_usd for c in self.charges)

    @property
    def image_cost_usd(self) -> float:
        return sum(c.would_cost_usd for c in self.charges if c.kind == "image")

    @property
    def tokens(self) -> Tuple[int, int]:
        return (sum(c.prompt_tokens for c in self.charges),
                sum(c.completion_tokens for c in self.charges))

    @property
    def cached(self) -> bool:
        return bool(self.charges) and all(c.cached for c in self.charges)

    @property
    def cost_known(self) -> bool:
        return bool(self.charges) and all(c.priced for c in self.charges)

    def method_metrics(self) -> Dict[str, Any]:
        atts = self.attempts
        scored = [a for a in atts if a.score is not None]
        first = scored[0].score.value if scored else None  # type: ignore[union-attr]
        best = (atts[self.best_idx].score.value  # type: ignore[union-attr]
                if self.best_idx is not None and 0 <= self.best_idx < len(atts)
                and atts[self.best_idx].score else None)
        return {
            "would_cost_usd": round(self.would_cost_usd, 6),
            "image_cost_usd": round(self.image_cost_usd, 6),
            "n_llm_calls": sum(1 for c in self.charges if c.kind == "llm"),
            "n_images": sum(c.images for c in self.charges if c.kind == "image"),
            "n_attempts": len(atts),
            "accept_rate": round(sum(a.accepted for a in atts) / max(1, len(atts)), 4),
            "best_idx": self.best_idx,
            "dream_score_first": first,
            "dream_score_best": best,
            "dream_gain": (round(best - first, 4) if best is not None and first is not None else None),
            "n_dream_versions": sum(len(v) for v in self.dreams.values()),
            "truncated": self.truncated,
            **self.scale_metrics(),
        }

    def scale_metrics(self) -> Dict[str, Any]:
        """Player-scale diagnostic of the final build (``scale.scale_stats``, a
        heuristic: headroom over covered standing cells, door-like openings by
        height). Reported, never selected on; {} if it cannot be computed."""
        s = getattr(self, "structure", None)
        if s is None:
            return {}
        try:
            from blockgen.agentic.scale import scale_stats
            return scale_stats(s)
        except Exception:  # noqa: BLE001 - a diagnostic must never fail a result
            return {}

    def metrics(self, connectivity: bool = True) -> Dict[str, Any]:
        """BuildResult metrics + method metrics.

        ``cost_usd`` keeps the legacy meaning, the FULL price (BuildResult counts a
        cached round at full price too), so the shared key is comparable between
        experiments_agentic and suite runs; what was actually paid is ``paid_usd``.
        """
        out = super().metrics(connectivity)
        out.update(self.method_metrics())
        out["cost_usd"] = round(self.would_cost_usd, 6)
        out["paid_usd"] = round(self.cost_usd, 6)
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({"spec": self.spec_name, "recipe_hash": self.recipe_hash,
                  "provenance": self.provenance, "trace_dir": self.trace_dir,
                  "attempts": [a.summary() for a in self.attempts],
                  "dreams": {k: [{"version": v.version, "prompt": v.prompt,
                                  "generator": v.generator, "digest": v.digest,
                                  "critique": v.critique} for v in vs]
                             for k, vs in self.dreams.items()},
                  "charges": [c.to_dict() for c in self.charges]})
        return d


__all__ = [
    "AcceptFn", "ActionSpace", "Attempt", "Budget", "BudgetExceeded", "CACHE_MODES",
    "CacheMiss", "CacheMode", "Charge", "ConfigError", "ContextFn", "ContextParts", "DreamSpec",
    "DreamVersion", "EmptyBuild", "EpisodeLike", "ImageGen", "ImageRequest", "ImageResult",
    "KINDS", "Kit", "Ledger", "PLUGINS_ENV", "Lowered", "MethodFn", "MethodResult", "MethodSpec",
    "MinecraftWorld", "NullTrace", "PromptCtx", "ROLES", "RenderError", "Renderer",
    "SPEND_ENV", "Score", "ScoreFn", "ScoreTerm", "SpendNotAllowed", "TraceLike", "VIEWS",
    "View", "World", "billing_of", "describe", "get_view", "load_builtins", "load_plugins",
    "lookup",
    "make", "meta", "names", "parse_spec", "register", "register_lazy", "require_spend",
    "spec_str", "spend_allowed", "validate_params",
]
