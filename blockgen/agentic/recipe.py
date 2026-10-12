"""Recipes and batteries: YAML -> :class:`~blockgen.agentic.core.MethodSpec`.

A *recipe* is one arm of an experiment, one file in ``configs/agentic/recipes/``
(the file name is the recipe name). A *battery* is a list of recipes plus a prompt
set, a budget and an optional sweep. Both are data, so "so many options" stays a
YAML diff instead of a code path.

The loader is strict on purpose. Every key must be a ``MethodSpec`` field and every
dotted override must land on one: a typo that silently ran the default is the
classic way a sweep "shows no effect". Method params are checked later, against
the method function's signature, by :func:`blockgen.agentic.kit.build_kit`.

Mechanics::

    extends: <name> | [<name>, ...]   parents deep-merged first (mappings merge,
                                      lists and scalars replace); cycles raise
    overrides {"params.max_iters": 5} applied last, after inheritance
    legacy-<arm>                      virtual recipe for each experiments_agentic.ARMS
                                      entry, via from_agent_config (no YAML needed)
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from blockgen.agentic.core import (SELECT_MODES, CACHE_MODES, ROLES, Budget, MethodSpec, _suggest,
                                   _default_gen)

_REPO = Path(__file__).resolve().parents[2]
#: Searched in order; the first ``<name>.yaml`` found wins.
RECIPE_DIRS: List[Path] = [_REPO / "configs" / "agentic" / "recipes"]
BATTERY_DIRS: List[Path] = [_REPO / "configs" / "agentic" / "batteries"]

#: Prefix of the virtual recipes built from ``experiments_agentic.ARMS``.
LEGACY_PREFIX = "legacy-"

SPEC_FIELDS: Tuple[str, ...] = tuple(f.name for f in fields(MethodSpec))

#: Keys a recipe's ``budget:`` mapping may hold. ``usd_per_build``/``max_llm_calls``/
#: ``max_images`` cap each build (the episode ledger); ``usd_per_arm`` caps the arm.
BUDGET_KEYS: Tuple[str, ...] = ("usd_per_build", "max_llm_calls", "max_images", "usd_per_arm")

#: AgentConfig fields that from_agent_config lifts OUT of the legacy ``config`` param,
#: because the spec carries them itself (provider -> llm, canvas, gen) or the suite
#: owns them (cache mode). Leaving ``cache`` in would let a stale AgentConfig flag
#: override ``--cache off`` for legacy arms.
_LIFTED_CONFIG_KEYS = ("provider", "canvas_size", "max_tokens", "temperature",
                       "reasoning_effort", "cache")


# ===========================================================================
# Dict plumbing
# ===========================================================================
def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive merge: mappings merge key by key, everything else is replaced.

    Lists replace rather than append, because "contexts: [palette]" in a child
    means exactly that list, not the parent's plus palette.
    """
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(d: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Set ``d["a"]["b"]["c"] = value`` for ``key="a.b.c"``, creating mappings.

    Raises TypeError when an intermediate is a non-mapping (``params.x.y`` where
    ``params.x`` is a number): silently replacing it would hide the mistake.
    Returns ``d`` for chaining.
    """
    parts = key.split(".")
    if not all(parts):
        raise ValueError(f"bad dotted key {key!r}")
    cur = d
    for i, p in enumerate(parts[:-1]):
        nxt = cur.get(p)
        if nxt is None:
            nxt = cur[p] = {}
        elif not isinstance(nxt, dict):
            if isinstance(nxt, Mapping):
                nxt = cur[p] = dict(nxt)
            else:
                raise TypeError(f"override {key!r}: '{'.'.join(parts[:i + 1])}' is "
                                f"{type(nxt).__name__} {nxt!r}, not a mapping")
        cur = nxt
    cur[parts[-1]] = value
    return d


def parse_sets(items: Sequence[str]) -> Dict[str, Any]:
    """CLI ``--set k=v`` strings -> overrides dict. Values are YAML-parsed, so
    ``5`` is an int, ``[iso, plan]`` a list and ``null`` None."""
    out: Dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def apply_overrides(d: Dict[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply dotted overrides to a raw recipe dict (validated against SPEC_FIELDS)."""
    for key, value in (overrides or {}).items():
        top = key.split(".", 1)[0]
        if top not in SPEC_FIELDS:
            raise TypeError(f"override {key!r}: unknown recipe field '{top}'; "
                            f"valid: {sorted(SPEC_FIELDS)}")
        # `llm: openai:x` means "every role"; `llm.judge=...` on top of it must keep
        # the other roles on x, which is exactly the builder fallback of roles().
        if top == "llm" and "." in key and isinstance(d.get("llm"), str):
            d["llm"] = {"builder": d["llm"]}
        set_dotted(d, key, copy.deepcopy(value))
    return d


def _slug(v: Any) -> str:
    """Value -> short arm-name token: [iso, plan] -> 'iso+plan'."""
    if isinstance(v, Mapping):
        if "key" in v:
            return _slug(v["key"])
        return "+".join(f"{_slug(k)}" for k in v)
    if isinstance(v, (list, tuple)):
        return "+".join(_slug(x) for x in v) or "none"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(v)).strip("-") or "none"


# ===========================================================================
# Recipes
# ===========================================================================
def _find(name_or_path: Union[str, Path], dirs: Sequence[Path], what: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p.resolve()
    for d in dirs:
        for ext in (".yaml", ".yml"):
            cand = d / f"{name_or_path}{ext}"
            if cand.is_file():
                return cand
    known = _list_dir(dirs)
    if what == "recipe":
        known += [f"{LEGACY_PREFIX}{a}" for a in _legacy_arms()
                  if f"{LEGACY_PREFIX}{a}" not in known]
    raise FileNotFoundError(f"no {what} '{name_or_path}'{_suggest(str(name_or_path), known)} "
                            f"in {[str(d) for d in dirs]}; known: {known}")


def _list_dir(dirs: Sequence[Path], include_private: bool = False) -> List[str]:
    out = set()
    for d in dirs:
        if d.is_dir():
            for f in list(d.glob("*.yaml")) + list(d.glob("*.yml")):
                if include_private or not f.stem.startswith("_"):
                    out.add(f.stem)
    return sorted(out)


#: Keys ``MethodSpec.resolved()`` / config.json add on top of the recipe fields.
DERIVED_KEYS = ("roles", "recipe_hash", "provenance")


def _legacy_arms() -> Dict[str, dict]:
    # Imported lazily: experiments_agentic is the legacy battery driver, and only
    # the virtual legacy-<arm> names need its ARMS table.
    from blockgen.experiments_agentic import ARMS
    return ARMS


#: experiments_agentic's CLI defaults (NOT AgentConfig's dataclass defaults, whose
#: canvas is 64): a virtual legacy-<arm> recipe must send the requests the legacy
#: driver sends at its defaults, else existing response-cache entries never hit.
LEGACY_CLI_DEFAULTS: Dict[str, Any] = {
    "provider": "openai:gpt-5-mini", "canvas_size": (48, 48, 48), "max_tokens": 16000,
    "reasoning_effort": "medium", "temperature": None, "critique_mode": "rewrite",
}


def _legacy_description(arm: str, fields: Mapping[str, Any]) -> str:
    knobs = ", ".join(f"{k}={v}" for k, v in fields.items())
    return f"legacy BuildAgent arm '{arm}' ({knobs})" if knobs else f"legacy BuildAgent arm '{arm}'"


def _virtual_dict(name: str) -> Optional[Dict[str, Any]]:
    """Raw dict for a virtual ``legacy-<arm>`` recipe, or None if ``name`` is not one.

    Built on experiments_agentic's CLI defaults (:data:`LEGACY_CLI_DEFAULTS`), so
    its requests and cache keys equal ``run_arm``'s at default flags.
    """
    if not name.startswith(LEGACY_PREFIX):
        return None
    arm = name[len(LEGACY_PREFIX):]
    arms = _legacy_arms()
    if arm not in arms:
        return None
    from blockgen.agentic.agent import AgentConfig
    spec = from_agent_config(AgentConfig(**{**LEGACY_CLI_DEFAULTS, **arms[arm]}), name=name)
    d = {f: copy.deepcopy(getattr(spec, f)) for f in SPEC_FIELDS}
    d["description"] = _legacy_description(arm, arms[arm])
    d["canvas"] = list(spec.canvas)
    d["params"]["arm"] = arm  # provenance: which ARMS entry this is
    return d


def _load_raw(name_or_path: Union[str, Path], stack: Tuple[str, ...]) -> Dict[str, Any]:
    """Recipe dict with its ``extends`` chain resolved (no overrides, no coercion)."""
    key = str(name_or_path)
    if key in stack:
        raise ValueError(f"recipe extends cycle: {' -> '.join(stack + (key,))}")
    virtual = _virtual_dict(key) if isinstance(name_or_path, str) else None
    if virtual is not None:
        try:
            _find(key, RECIPE_DIRS, "recipe")
        except FileNotFoundError:
            return virtual   # a real YAML file of the same name wins
    path = _find(name_or_path, RECIPE_DIRS, "recipe")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"recipe {path} must be a mapping, got {type(raw).__name__}")
    # Derived fields of a stored recipe (``show`` / config.json 'recipes'): ignored,
    # so a run's resolved recipe loads back as a recipe.
    raw = {k: v for k, v in raw.items() if k not in DERIVED_KEYS}
    parents = raw.pop("extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: Dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, _load_raw(parent, stack + (key,)))
    return deep_merge(merged, raw)


def _coerce_spec(d: Dict[str, Any], name: str) -> MethodSpec:
    """Validate a merged recipe dict and build the MethodSpec."""
    unknown = sorted(set(d) - set(SPEC_FIELDS))
    if unknown:
        raise TypeError(f"recipe '{name}': unknown field(s) {unknown}; valid: {sorted(SPEC_FIELDS)}")
    d = dict(d)
    d["name"] = name
    if "canvas" in d:
        c = d["canvas"]
        c = [c, c, c] if isinstance(c, int) else list(c)
        if len(c) != 3 or not all(isinstance(v, int) and v > 0 for v in c):
            raise ValueError(f"recipe '{name}': canvas must be 3 positive ints, got {d['canvas']!r}")
        d["canvas"] = tuple(c)
    if d.get("select", "last") not in SELECT_MODES:
        raise ValueError(f"recipe '{name}': select must be one of {SELECT_MODES}, got {d['select']!r}")
    if d.get("cache", "read_write") not in CACHE_MODES:
        raise ValueError(f"recipe '{name}': cache must be one of {CACHE_MODES}, got {d['cache']!r}")
    llm = d.get("llm")
    if isinstance(llm, Mapping):
        bad = sorted(set(llm) - set(ROLES))
        if bad:
            raise ValueError(f"recipe '{name}': unknown llm role(s) {bad}; roles: {list(ROLES)}")
        d["llm"] = dict(llm)
    budget = d.get("budget") or {}
    bad = sorted(set(budget) - set(BUDGET_KEYS))
    if bad:
        raise ValueError(f"recipe '{name}': unknown budget key(s) {bad}; valid: {list(BUDGET_KEYS)}")
    for key in ("contexts", "dreams"):
        if key in d and d[key] is not None and not isinstance(d[key], list):
            d[key] = [d[key]] if not isinstance(d[key], tuple) else list(d[key])
    for key in ("score", "gates", "final_score", "params", "budget", "tags", "gen"):
        if key in d and d[key] is None:
            d[key] = {}
    if "gen" in d:
        d["gen"] = {**_default_gen(), **d["gen"]}
    return MethodSpec(**d)


def load_recipe(name_or_path: Union[str, Path],
                overrides: Optional[Mapping[str, Any]] = None) -> MethodSpec:
    """Load ``configs/agentic/recipes/<name>.yaml`` (or a path, or ``legacy-<arm>``).

    The spec's ``name`` is the requested name (file stem for a path), never an
    inherited one, so ``dream-refine`` extending ``dream`` is still 'dream-refine'.
    An explicit ``name`` override (a battery's ``as:``) renames the arm.
    """
    name = Path(name_or_path).stem if isinstance(name_or_path, Path) or \
        str(name_or_path).endswith((".yaml", ".yml")) else str(name_or_path)
    d = _load_raw(name_or_path, ())
    d.pop("name", None)      # inherited names are never meaningful
    d = apply_overrides(d, overrides)
    final_name = str(d.pop("name", None) or name)
    return _coerce_spec(d, final_name)


def list_recipes(*, virtual: bool = True) -> List[str]:
    """Recipe names (files not starting with '_'), plus ``legacy-<arm>`` names."""
    out = _list_dir(RECIPE_DIRS)
    if virtual:
        out += [f"{LEGACY_PREFIX}{a}" for a in _legacy_arms() if f"{LEGACY_PREFIX}{a}" not in out]
    return out


def from_agent_config(cfg: Any, name: str = "legacy", *, native: bool = False) -> MethodSpec:
    """An :class:`~blockgen.agentic.agent.AgentConfig` as a recipe.

    ``native=False`` (default): method ``legacy``, which runs BuildAgent itself, so
    the request bytes (and response-cache keys) are exactly the legacy ones. The
    AgentConfig travels as ``params.config`` minus the fields the spec carries
    (provider -> llm, canvas, gen, cache).

    ``native=True``: method ``classic``, the Episode re-implementation, with the
    loop knobs mapped to its params and the ontology/examples as context modules.
    """
    contexts: List[Any] = ["palette"]
    if cfg.ontology not in ("none", "", None):
        contexts.append({"ontology": {"variant": cfg.ontology, "path": cfg.ontology_path,
                                      "seed": cfg.ontology_seed}})
    if cfg.n_examples:
        contexts.append({"examples": {"k": cfg.n_examples}})
    common = dict(
        name=name, llm=cfg.provider, contexts=contexts, canvas=tuple(int(v) for v in cfg.canvas_size),
        gen=dict(cfg.provider_params()), keep_history=bool(cfg.keep_history),
        target_blocks=cfg.target_blocks, cache="read_write" if cfg.cache else "off",
        accept="nonempty_legacy", select="last",
    )
    if not native:
        config = {k: v for k, v in cfg.to_dict().items() if k not in _LIFTED_CONFIG_KEYS}
        return MethodSpec(method="legacy", description="legacy BuildAgent",
                          params={"config": config}, **common)
    params = {"plan": bool(cfg.plan), "n_examples": int(cfg.n_examples),
              "repair_rounds": int(cfg.repair_rounds),
              "critique_rounds": int(cfg.critique_rounds),
              "critique_mode": cfg.critique_mode}
    return MethodSpec(method="classic", description="native classic (from AgentConfig)",
                      params=params, **common)


# ===========================================================================
# Batteries
# ===========================================================================
@dataclass
class Battery:
    """A list of arms run over one prompt set, under one budget, into one run dir."""

    name: str
    prompts: Dict[str, Any] = field(default_factory=dict)   # tasks.load_prompts kwargs
    recipes: List[MethodSpec] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    cache: str = "read_write"

    def arm_names(self) -> List[str]:
        return [r.name for r in self.recipes]


def sweep_arm_name(base: str, combo: Mapping[str, Any]) -> str:
    """``dream@max_iters=1,dreams=iso`` (keys shortened to their last dotted part)."""
    return base + "@" + ",".join(f"{k.rsplit('.', 1)[-1]}={_slug(v)}" for k, v in combo.items())


def expand_sweep(sweep: Mapping[str, Any],
                 overrides: Optional[Mapping[str, Any]] = None) -> List[MethodSpec]:
    """Cartesian product over ``sweep.grid`` applied to ``sweep.base``.

    More combinations than ``max_arms`` (default 8) raises instead of truncating:
    a silently dropped arm is worse than a loud error, and the cap exists to stop
    an accidental 3-axis sweep from spending the battery budget.
    """
    base = sweep.get("base")
    grid = dict(sweep.get("grid") or {})
    if not base or not grid:
        raise ValueError(f"sweep needs 'base' and a non-empty 'grid': {dict(sweep)!r}")
    unknown = sorted(set(sweep) - {"base", "grid", "max_arms"})
    if unknown:
        raise TypeError(f"sweep: unknown key(s) {unknown}; valid: ['base', 'grid', 'max_arms']")
    max_arms = int(sweep.get("max_arms", 8))
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    combos = list(itertools.product(*values))
    if len(combos) > max_arms:
        raise ValueError(f"sweep over {keys} gives {len(combos)} arms > max_arms={max_arms}")
    out = []
    for combo in combos:
        c = dict(zip(keys, combo))
        name = sweep_arm_name(str(base), c)
        out.append(load_recipe(str(base), {**(overrides or {}), **c, "name": name}))
    return out


def _recipe_entry(entry: Any, overrides: Optional[Mapping[str, Any]]) -> MethodSpec:
    """A battery ``recipes:`` item: 'name' or {name: {as: arm, dotted.key: value}}."""
    if isinstance(entry, str):
        return load_recipe(entry, overrides)
    if isinstance(entry, Mapping) and len(entry) == 1:
        rname, opts = next(iter(entry.items()))
        opts = dict(opts or {})
        arm = opts.pop("as", None)
        merged = {**(overrides or {}), **opts}
        if arm:
            merged["name"] = arm
        return load_recipe(str(rname), merged)
    raise ValueError(f"battery recipe entry must be a name or {{name: {{...}}}}, got {entry!r}")


def load_battery(path: Union[str, Path],
                 overrides: Optional[Mapping[str, Any]] = None) -> Battery:
    """Load a battery YAML (a path, or a name under ``configs/agentic/batteries``).

    ``overrides`` are dotted recipe overrides applied to EVERY arm (the CLI's
    ``--set``). A battery-level ``cache:`` is applied to every arm, so one battery
    runs under one cache mode.
    """
    p = _find(path, BATTERY_DIRS, "battery")
    raw = yaml.safe_load(p.read_text()) or {}
    valid = {"name", "prompts", "budget", "cache", "recipes", "sweep", "description"}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise TypeError(f"battery {p}: unknown key(s) {unknown}; valid: {sorted(valid)}")
    cache = raw.get("cache", "read_write")
    if cache not in CACHE_MODES:
        raise ValueError(f"battery {p}: cache must be one of {CACHE_MODES}, got {cache!r}")
    ov = dict(overrides or {})
    if "cache" in raw:
        ov.setdefault("cache", cache)
    recipes = [_recipe_entry(e, ov) for e in raw.get("recipes") or []]
    if raw.get("sweep"):
        recipes += expand_sweep(raw["sweep"], ov)
    seen: Dict[str, int] = {}
    for r in recipes:
        seen[r.name] = seen.get(r.name, 0) + 1
    dup = sorted(n for n, c in seen.items() if c > 1)
    if dup:
        raise ValueError(f"battery {p}: duplicate arm name(s) {dup}; use 'as:' to rename")
    b = raw.get("budget") or {}
    bad = sorted(set(b) - {"usd", "max_llm_calls", "max_images"})
    if bad:
        raise ValueError(f"battery {p}: unknown budget key(s) {bad}")
    return Battery(name=str(raw.get("name") or p.stem), prompts=dict(raw.get("prompts") or {}),
                   recipes=recipes, budget=Budget(**b), cache=cache)


__all__ = ["BATTERY_DIRS", "BUDGET_KEYS", "Battery", "LEGACY_PREFIX", "RECIPE_DIRS",
           "SPEC_FIELDS", "apply_overrides", "deep_merge", "expand_sweep",
           "from_agent_config", "list_recipes", "load_battery", "load_recipe",
           "parse_sets", "set_dotted", "sweep_arm_name"]
