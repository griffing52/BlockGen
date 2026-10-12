"""Resolve a recipe into services: ``build_kit(spec) -> Kit``.

A :class:`~blockgen.agentic.core.MethodSpec` only NAMES things ("dsl", "textured",
"openai:gpt-image-1-mini?quality=low"). This module turns those names into live
objects once per arm, so heavy models (SigLIP, diffusers, the texture library)
load once per arm instead of once per prompt.

Every component goes through the registry (:func:`core.make`), never a direct
import, so a recipe can name anything a module registered.

``mock=True`` swaps every backend for its offline twin (table :data:`MOCK_SUBS`),
so any recipe runs in CI with no network and no GPU. The swap is recorded in the
kit's spec (``tags.mock = True``), which also changes its ``recipe_hash``: a mock
run is never comparable with a real one.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic.core import (CACHE_MODES, Budget, DreamSpec, Kit, Ledger,
                                   MethodSpec, MinecraftWorld, _default_gen, _suggest,
                                   get_view, lookup, make, parse_spec, validate_params)
from blockgen.agentic.providers import LLMProvider

#: What ``mock=True`` (CLI ``--mock``) substitutes. Scorers not listed run as-is
#: (they are cheap and CPU-only); ``vlm`` keeps its name but its judge is the mock LLM.
#: The renderer is NOT swapped: the textured renderer is local, offline and ~30 ms a
#: view, so mock runs show the same pictures a real run would. ``renderer`` here is
#: only the fallback for a machine without the texture pack or EGL.
MOCK_SUBS: Dict[str, Any] = {
    "llm": "mock:auto",
    "image": "mock:render",
    "renderer": "flat",
    "scorer": {"embed": "pixel", "clip_text": "pixel"},
}


@functools.lru_cache(maxsize=1)
def textured_available() -> bool:
    """True if the textured renderer can draw here (texture pack on disk + EGL)."""
    from blockgen.agentic.render import DEFAULT_PACK
    if not DEFAULT_PACK.is_dir() or not any(DEFAULT_PACK.glob("*.png")):
        return False
    try:
        import blockgen.renderer.textured  # noqa: F401  (sets PYOPENGL_PLATFORM=egl)
    except Exception:  # noqa: BLE001 - no pyrender / no EGL
        return False
    return True


def _mock_renderer(renderer: Any) -> Any:
    """Keep the recipe's renderer under --mock unless it is textured and unusable."""
    if renderer in (None, "", "none"):
        return renderer
    if parse_spec(renderer)[0] == "textured" and not textured_available():
        warnings.warn("textured renderer unavailable (no texture pack or EGL); "
                      "--mock falls back to the flat renderer")
        return MOCK_SUBS["renderer"]
    return renderer



# ===========================================================================
# Helpers
# ===========================================================================
def _key(spec: Any) -> str:
    """Hashable, order-independent cache key for a component spec."""
    return json.dumps(spec, sort_keys=True, default=str)


def episode_budget(spec: MethodSpec) -> Budget:
    """Per-build caps from ``spec.budget`` (what run_episode gives each build)."""
    b = spec.budget or {}
    return Budget(usd=b.get("usd_per_build"),
                  max_llm_calls=None if b.get("max_llm_calls") is None else int(b["max_llm_calls"]),
                  max_images=None if b.get("max_images") is None else int(b["max_images"]))


def mock_spec(spec: MethodSpec) -> MethodSpec:
    """``spec`` with every backend swapped for its offline twin (see MOCK_SUBS)."""
    subs = MOCK_SUBS["scorer"]

    def swap_scores(score: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in score.items():
            name = parse_spec(k)[0]
            if name in subs:
                # Keep the term NAME (gates and score parts refer to it) but run the
                # mock function with its own defaults: the real scorer's params
                # (backbone=...) mean nothing to it.
                w = v.get("weight", 1.0) if isinstance(v, Mapping) else v
                out[name] = {"type": subs[name], "weight": w}
            else:
                out[k] = v
        return out

    return dataclasses.replace(
        spec,
        llm=MOCK_SUBS["llm"],
        image=MOCK_SUBS["image"] if spec.image else None,
        renderer=_mock_renderer(spec.renderer),
        score=swap_scores(spec.score),
        final_score=swap_scores(spec.final_score),
        tags={**spec.tags, "mock": True},
    )


# ===========================================================================
# Heavy, cached components
# ===========================================================================
@functools.lru_cache(maxsize=32)
def _cached_component(kind: str, key: str) -> Any:
    return make(kind, json.loads(key))


@functools.lru_cache(maxsize=32)
def _cached_image(key: str, mode: str, spend_gate: bool) -> Any:
    return _resolve_image(json.loads(key), mode, spend_gate)


def clear_cache() -> None:
    """Drop cached renderers / image generators (tests; freeing VRAM between arms)."""
    _cached_component.cache_clear()
    _cached_image.cache_clear()


def _resolve_image(spec: Any, mode: str, spend_gate: bool) -> Any:
    """Image generator, wrapped in the content-addressed cache (and spend gate)."""
    from blockgen.agentic.imagegen import get_image_gen
    return get_image_gen(spec, cache=mode, spend_gate=spend_gate)


# ===========================================================================
# LLMs
# ===========================================================================
def _resolve_llms(spec: MethodSpec, mode: str, spend_gate: bool) -> Dict[str, LLMProvider]:
    """One provider per DISTINCT spec string; roles sharing a spec share it.

    Construction mirrors ``experiments_agentic.run_arm`` (``get_provider(spec,
    cache=..., **gen)``) so a legacy-equivalent recipe hits the same cache keys
    (``cache_mode`` / ``spend_gate`` are named arguments, never part of the key).
    """
    from blockgen.agentic import providers

    gen = {**_default_gen(), **(spec.gen or {})}
    built: Dict[str, LLMProvider] = {}
    out: Dict[str, LLMProvider] = {}
    for role, s in spec.roles().items():
        if s not in built:
            built[s] = providers.get_provider(s, cache=mode != "off", cache_mode=mode,
                                              spend_gate=spend_gate, **gen)
        out[role] = built[s]
    return out


# ===========================================================================
# Scorers, dreams
# ===========================================================================
def _resolve_scorers(score: Mapping[str, Any], where: str) -> Dict[str, Tuple[Any, float]]:
    """``{spec: weight | {weight, type?, **params}}`` -> ``{name: (fn, weight)}``."""
    out: Dict[str, Tuple[Any, float]] = {}
    for k, v in (score or {}).items():
        name, arg, params = parse_spec(k)
        weight = 1.0
        target = name
        if isinstance(v, Mapping):
            v = dict(v)
            weight = float(v.pop("weight", 1.0))
            target = str(v.pop("type", name))
            params.update(v)
        elif v is not None:
            weight = float(v)
        if name in out:
            raise ValueError(f"{where}: scorer '{name}' listed twice ({k!r}); one term per name")
        spec_: Dict[str, Any] = {"type": target, **params}
        if arg is not None:
            spec_["arg"] = arg
        out[name] = (make("scorer", spec_), weight)
    return out


def _as_dreamspec(d: Mapping[str, Any]) -> DreamSpec:
    d = dict(d)
    for k in ("condition_on", "view_search"):
        if k in d and d[k] is not None:
            d[k] = (d[k],) if isinstance(d[k], str) else tuple(d[k])
    valid = {f.name for f in dataclasses.fields(DreamSpec)}
    bad = sorted(set(d) - valid)
    if bad:
        raise TypeError(f"dream {d.get('key', '?')!r}: unknown field(s) {bad}; valid: {sorted(valid)}")
    return DreamSpec(**d)


def resolve_dreams(entries: List[Any]) -> List[DreamSpec]:
    """DREAMS preset names or inline dicts -> DreamSpecs.

    An inline dict whose ``key`` is a preset overrides that preset's fields
    (``{key: iso, weight: 0.5}``); any other inline dict must be complete.
    """
    presets: Dict[str, DreamSpec] = dict(dreams_mod.DREAMS)
    out: List[DreamSpec] = []
    expansions = dict(dreams_mod.EXPANSIONS)
    # Expansion names ('turnaround' -> iso, back, left, right) become their presets.
    # Not dreams.expand(): it drops repeated keys, and a repeat must still raise below.
    entries = [x for e in (entries or [])
               for x in (expansions[e] if isinstance(e, str) and e in expansions else (e,))]
    for e in entries or []:
        if isinstance(e, DreamSpec):
            ds = e
        elif isinstance(e, str) or (isinstance(e, Mapping) and "template" not in e):
            key = e if isinstance(e, str) else e.get("key")
            if key not in presets:
                raise KeyError(f"unknown dream preset {key!r}{_suggest(str(key), sorted(presets))}; "
                               f"presets: {sorted(presets)}")
            ds = presets[key]
            if isinstance(e, Mapping):
                ds = _as_dreamspec({**dataclasses.asdict(ds), **e})
        elif isinstance(e, Mapping):
            ds = _as_dreamspec(e)
        else:
            raise TypeError(f"dream entry must be a preset name or a mapping, got {e!r}")
        out.append(ds)
    keys = [d.key for d in out]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate dream key(s) {dup}")
    for d in out:
        for v in ((d.view,) if d.view else ()) + tuple(d.view_search):
            get_view(v)
        missing = [c for c in d.condition_on if c not in keys]
        if missing:
            raise ValueError(f"dream {d.key!r} conditions on unknown key(s) {missing}")
    return out


# ===========================================================================
# build_kit
# ===========================================================================
def resolve_method(spec: MethodSpec) -> Any:
    """The method function with ``spec.params`` bound and validated.

    Binding here means a typo in ``params:`` fails when the arm is built, not
    after the first paid call of its first prompt.
    """
    name, arg, inline = parse_spec(spec.method)
    if arg is not None:
        raise ValueError(f"method '{name}' takes no ':arg' (got {arg!r})")
    fn = lookup("method", name).load()
    params = {**inline, **(spec.params or {})}
    validate_params(fn, params, where=f"method '{name}' (recipe '{spec.name}')")
    bound = functools.partial(fn, **params)
    bound.spec_name = name  # type: ignore[attr-defined]
    return bound


def _check_component(kind: str, spec: Any, where: str) -> None:
    """Name exists and params match the signature, WITHOUT constructing it.

    A component whose module needs a missing optional dependency (ImportError on
    load) passes: that is a per-arm runtime condition, recorded when the kit is
    built, not a typo.
    """
    name, _, params = parse_spec(spec)
    entry = lookup(kind, name)
    try:
        target = entry.load()
    except ImportError:
        return
    validate_params(target, params, where=f"{where}: {kind} '{name}'")


def validate_spec(spec: MethodSpec) -> None:
    """Raise on configuration errors in ``spec`` without loading any model.

    Checks the method and its params, every named component (actions, accept,
    contexts, renderer, image, scorers) with its params, dreams, gates and the
    cache mode. Used by the suite to fail a whole run up front, before the first
    arm makes a single call, and by ``show``.
    """
    where = f"recipe '{spec.name}'"
    if spec.cache not in CACHE_MODES:
        raise ValueError(f"{where}: cache mode must be one of {CACHE_MODES}, got {spec.cache!r}")
    resolve_method(spec)
    resolve_dreams(spec.dreams)
    _check_component("actions", spec.actions, where)
    _check_component("accept", spec.accept, where)
    for c in spec.contexts or []:
        _check_component("context", c, where)
    if spec.renderer not in (None, "", "none"):
        _check_component("renderer", spec.renderer, where)
    if spec.image:
        _check_component("image", spec.image, where)
    names_: List[str] = []
    for label, score in (("score", spec.score), ("final_score", spec.final_score)):
        for k, v in (score or {}).items():
            name, arg, params = parse_spec(k)
            target = name
            if isinstance(v, Mapping):
                v = dict(v)
                v.pop("weight", None)
                target = str(v.pop("type", name))
                params.update(v)
            sub = {"type": target, **params}
            if arg is not None:
                sub["arg"] = arg
            _check_component("scorer", sub, f"{where} {label}")
            if label == "score":
                names_.append(name)
    bad_gates = sorted(set(spec.gates or {}) - set(names_))
    if bad_gates:
        raise ValueError(f"{where}: gate(s) {bad_gates} name no scorer in score: {sorted(names_)}")


def build_kit(spec: MethodSpec, *, mock: bool = False, cache_mode: Optional[str] = None,
              ledger: Optional[Ledger] = None, spend_gate: bool = True) -> Kit:
    """Resolve every axis of ``spec`` once.

    ``cache_mode`` overrides ``spec.cache``. ``ledger`` is the PARENT (battery)
    ledger: the kit gets an arm-level child, capped by ``budget.usd_per_arm`` if
    set, and each episode takes a per-build child of that. ``spend_gate`` makes
    cache misses on paying backends require BLOCKGEN_ALLOW_SPEND=1.
    """
    if mock:
        spec = mock_spec(spec)
    mode = cache_mode or spec.cache
    if mode not in CACHE_MODES:
        raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")

    # size= reaches only action spaces that declare it (python / mixed), so their
    # SIZE constant matches the canvas the grammar describes.
    actions = make("actions", spec.actions, size=tuple(spec.canvas))
    llms = _resolve_llms(spec, mode, spend_gate)

    image = _cached_image(_key(spec.image), mode, spend_gate) if spec.image else None
    renderer = (None if spec.renderer in (None, "", "none")
                else _cached_component("renderer", _key(spec.renderer)))
    contexts = [(parse_spec(c)[0], make("context", c)) for c in spec.contexts or []]
    scorers = _resolve_scorers(spec.score, f"recipe '{spec.name}' score")
    final_scorers = _resolve_scorers(spec.final_score, f"recipe '{spec.name}' final_score")
    bad_gates = sorted(set(spec.gates or {}) - set(scorers))
    if bad_gates:
        raise ValueError(f"recipe '{spec.name}': gate(s) {bad_gates} name no scorer in "
                         f"score: {sorted(scorers)}")
    accept = make("accept", spec.accept)
    method = resolve_method(spec)
    dreams = resolve_dreams(spec.dreams)
    if dreams and image is None:
        warnings.warn(f"recipe '{spec.name}' has dreams but no image generator")

    arm_budget = Budget(usd=(spec.budget or {}).get("usd_per_arm"))
    arm_ledger = (ledger.child(arm_budget, name=spec.name) if ledger is not None
                  else Ledger(arm_budget, name=spec.name))
    return Kit(spec=spec, llms=llms, image=image, actions=actions, renderer=renderer,
               contexts=contexts, scorers=scorers, final_scorers=final_scorers,
               accept=accept, method=method, dreams=dreams, ledger=arm_ledger,
               world_factory=MinecraftWorld.empty, mock=mock)


__all__ = ["MOCK_SUBS", "build_kit", "clear_cache", "episode_budget", "mock_spec",
           "resolve_dreams", "resolve_method", "validate_spec"]
