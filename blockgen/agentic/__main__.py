"""CLI of the agentic methods suite: ``python -m blockgen.agentic <command>``.

    list [kind]          the catalogue: method actions image renderer context scorer
                         accept recipes views dreams llm (no kind = everything)
    show <recipe>        resolved recipe as YAML + recipe_hash + provenance
    run --recipe A[,B]   arms x prompts -> outputs/run_<stamp>_<name>[_mock]/
    battery FILE         a battery YAML (recipes, prompts, budget, sweep)
    estimate             upper-bound $ for a run, on free mocks, zero network calls
    calibrate            does a scorer rank a build's own render above distractors?
    rescore RUN_DIR      re-run scorers offline from a run's traces

Every command takes ``--plugin FILE.py|MODULE`` (repeatable) to import an
out-of-tree module whose ``@register`` components (a scratch method, scorer, ...)
recipes may then name, without editing BUILTIN_MODULES.

Spend safety: any cache miss on a paying backend (OpenAI, Gemini, Anthropic,
Claude Code quota) raises SpendNotAllowed unless ``--allow-spend`` (or
BLOCKGEN_ALLOW_SPEND=1) is given. ``--mock`` runs everything offline:

    .venv/bin/python -m blockgen.agentic run --recipe dream-multiview --prompts short:2 --mock
    .venv/bin/python -m blockgen.agentic estimate --recipe dream-refine --prompts short:12
    .venv/bin/python -m blockgen.agentic run --recipe dream --prompts short:4 \\
        --allow-spend --budget-usd 2

Full guide: docs/agentic-suite.md.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LIST_KINDS = ("method", "actions", "image", "renderer", "context", "scorer", "accept",
              "recipes", "views", "dreams", "llm")
#: Other spellings ``list`` accepts (singular or plural of every kind).
KIND_ALIASES = {"methods": "method", "action": "actions", "images": "image",
                "renderers": "renderer", "contexts": "context", "scorers": "scorer",
                "accepts": "accept", "recipe": "recipes", "view": "views",
                "dream": "dreams", "llms": "llm"}


# ===========================================================================
# list / show
# ===========================================================================
def _signature(target: Any) -> str:
    """'(a=1, b=2)' of a component, without the episode/attempt positional args."""
    try:
        fn = target.__init__ if inspect.isclass(target) else target
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return ""
    skip = {"self", "ep", "att", "prev", "cand", "arg"}
    parts = []
    for p in params:
        if p.name in skip or (p.kind is p.POSITIONAL_ONLY):
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.kind is p.VAR_POSITIONAL:
            continue
        elif p.default is inspect.Parameter.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={p.default!r}")
    return "(" + ", ".join(parts) + ")"


def list_kind(kind: str) -> List[str]:
    """Lines describing one catalogue kind (also used by the docs and tests)."""
    from blockgen.agentic import core
    core.load_builtins()
    lines: List[str] = []
    if kind in core.KINDS:
        aliases = {}
        for a, n in core._ALIASES[kind].items():
            aliases.setdefault(n, []).append(a)
        for name, doc in core.describe(kind):
            entry = core._REG[kind][name]
            try:
                sig = _signature(entry.load())   # a lazy entry is imported to show params
            except ImportError:
                sig = "  (module not importable: optional dependency missing)"
            meta = {k: v for k, v in entry.meta.items() if k != "doc"}
            extra = (f"  [aliases: {', '.join(sorted(aliases[name]))}]" if name in aliases else "")
            extra += f"  {meta}" if meta else ""
            lines.append(f"  {name}{sig}{extra}\n      {doc}")
        if core._BUILTINS_SKIPPED:
            lines.append(f"  (skipped built-in modules: {core._BUILTINS_SKIPPED})")
    elif kind == "recipes":
        from blockgen.agentic.recipe import list_recipes, load_recipe
        for name in list_recipes():
            try:
                spec = load_recipe(name)
                lines.append(f"  {name:<20} [{spec.method}] {spec.description}  #{spec.hash()}")
            except Exception as exc:  # noqa: BLE001 - one bad file must not hide the list
                lines.append(f"  {name:<20} ERROR {type(exc).__name__}: {exc}")
    elif kind == "views":
        for name, v in core.VIEWS.items():
            lines.append(f"  {name:<10} {v}")
    elif kind == "dreams":
        from blockgen.agentic import dreams
        for name, ds in dreams.DREAMS.items():
            lines.append(f"  {name:<10} view={ds.view} weight={ds.weight} "
                         f"condition_on={list(ds.condition_on)} view_search={list(ds.view_search)}")
        for name, keys in dreams.EXPANSIONS.items():
            lines.append(f"  {name:<10} expands to {list(keys)}")
    elif kind == "llm":
        from blockgen.agentic import providers
        for vendor in sorted(providers._PROVIDERS):
            default = providers.DEFAULT_MODELS.get(vendor, "")
            lines.append(f"  {vendor:<12} default model: {default or '-'}")
        lines.append("  mock:auto    offline AutoMock (answers every suite turn)")
        lines.append("  priced models (USD per 1M tokens in/out): " + ", ".join(
            f"{m} {p[0]}/{p[1]}" for m, p in providers.PRICES.items()))
    else:
        if kind in KIND_ALIASES:
            return list_kind(KIND_ALIASES[kind])
        raise SystemExit(f"unknown kind '{kind}'; choose from {LIST_KINDS}")
    return lines


def cmd_list(args: argparse.Namespace) -> None:
    kinds = [KIND_ALIASES.get(args.kind, args.kind)] if args.kind else LIST_KINDS
    for kind in kinds:
        print(f"{kind}:")
        print("\n".join(list_kind(kind)) or "  (none)")


def _config(fn: Any, *a: Any, **kw: Any) -> Any:
    """Call a recipe/battery loader; user-input errors become one ConfigError."""
    from blockgen.agentic.core import ConfigError
    try:
        return fn(*a, **kw)
    except ConfigError:
        raise
    except (TypeError, KeyError, ValueError, FileNotFoundError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        raise ConfigError(f"{type(exc).__name__}: {msg}") from exc


def cmd_show(args: argparse.Namespace) -> None:
    """The resolved recipe as LOADABLE YAML (derived fields as trailing comments),
    validated like a run would validate it."""
    import yaml
    from blockgen.agentic.kit import validate_spec
    from blockgen.agentic.recipe import DERIVED_KEYS, load_recipe
    spec = _config(load_recipe, args.recipe, _overrides(args))
    _config(validate_spec, spec)
    d = {k: v for k, v in spec.resolved().items() if k not in DERIVED_KEYS}
    print(yaml.safe_dump(d, sort_keys=False, default_flow_style=None).rstrip())
    print(f"\n# roles: {json.dumps(spec.roles())}")
    print(f"# recipe_hash: {spec.hash()}")
    print("# provenance: " + json.dumps(spec.provenance(), default=str))


# ===========================================================================
# overrides shared by run / battery / estimate
# ===========================================================================
def _role_overrides(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for item in items or []:
        role, sep, spec = item.partition("=")
        if not sep:
            raise SystemExit(f"--llm-role expects ROLE=SPEC, got {item!r}")
        out[f"llm.{role.strip()}"] = spec.strip()
    return out


def _backend_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """--llm / --llm-role / --imagegen / --renderer as dotted recipe overrides."""
    out: Dict[str, Any] = {}
    if getattr(args, "llm", None):
        out["llm"] = args.llm
    out.update(_role_overrides(getattr(args, "llm_role", None) or []))
    if getattr(args, "imagegen", None):
        out["image"] = args.imagegen
    if getattr(args, "renderer", None):
        out["renderer"] = args.renderer
    return out


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Recipe overrides applied when loading. With --mock, backend flags are kept
    for AFTER the mock substitution instead (see suite.prepare_spec)."""
    from blockgen.agentic.recipe import parse_sets
    out = parse_sets(getattr(args, "set", None) or [])
    if not getattr(args, "mock", False):
        out.update(_backend_overrides(args))
    return out


def _post_mock(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    return _backend_overrides(args) or None if getattr(args, "mock", False) else None


def _specs(names: str, overrides: Dict[str, Any]) -> List[Any]:
    from blockgen.agentic.recipe import load_recipe
    return [_config(load_recipe, n.strip(), overrides) for n in names.split(",") if n.strip()]


def _budget(args: argparse.Namespace) -> Any:
    from blockgen.agentic.core import Budget
    return Budget(usd=args.budget_usd) if getattr(args, "budget_usd", None) is not None else None


def _prompts(args: argparse.Namespace) -> Optional[List[str]]:
    from blockgen.agentic.suite import parse_prompts
    if not getattr(args, "prompts", None):
        return None
    prompts = parse_prompts(args.prompts, n=args.n, seed=args.seed)
    if not prompts:
        raise SystemExit(f"prompt source '{args.prompts}' yielded nothing")
    return prompts


def _images(args: argparse.Namespace) -> Tuple[bytes, ...]:
    return tuple(Path(p).read_bytes() for p in (getattr(args, "image", None) or []))


def _run_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(mock=args.mock, post_mock=_post_mock(args), cache_mode=args.cache,
                images=_images(args), dreams_first=args.dreams_first,
                render=not args.no_render, base=args.base, seed=args.variation)


# ===========================================================================
# commands
# ===========================================================================
def _run_status(run_dir: Path) -> int:
    """1 if some arm could not be built (recorded as failed builds), else 0."""
    try:
        errors = json.loads((Path(run_dir) / "config.json").read_text()).get("kit_errors") or {}
    except (OSError, ValueError):
        return 0
    for arm, err in errors.items():
        print(f"ERROR: arm '{arm}' did not run: {err}", file=sys.stderr)
    return 1 if errors else 0


def cmd_run(args: argparse.Namespace) -> int:
    from blockgen.agentic.suite import run_recipes
    specs = _specs(args.recipe, _overrides(args))
    if not args.prompts:
        args.prompts = "short:4"
    prompts = _prompts(args) or []
    name = args.name or "suite_" + "+".join(s.name for s in specs)[:60]
    return _run_status(run_recipes(specs, prompts, name=name, budget=_budget(args),
                                   **_run_kwargs(args)))


def cmd_battery(args: argparse.Namespace) -> int:
    from blockgen.agentic.recipe import load_battery
    from blockgen.agentic.suite import run_battery
    b = _config(load_battery, args.file, _overrides(args))
    only = [a.strip() for a in args.only.split(",")] if args.only else None
    return _run_status(run_battery(b, prompts=_prompts(args), only=only, name=args.name,
                                   budget=_budget(args), **_run_kwargs(args)))


def cmd_estimate(args: argparse.Namespace) -> None:
    from blockgen.agentic.recipe import load_battery
    from blockgen.agentic.suite import estimate, format_estimate, parse_prompts
    overrides = _overrides(args)
    if args.battery:
        b = _config(load_battery, args.battery, overrides)
        specs = b.recipes
        prompts = _prompts(args)
        if prompts is None:
            p = dict(b.prompts or {})
            prompts = parse_prompts(str(p.pop("source", "detailed")), n=p.pop("n", None),
                                    seed=int(p.pop("seed", 0)))
    elif args.recipe:
        specs = _specs(args.recipe, overrides)
        prompts = _prompts(args) or parse_prompts("short:4")
    else:
        raise SystemExit("estimate needs --recipe or --battery")
    rows = estimate(specs, prompts, completion_frac=args.completion_frac)
    print(f"estimate: {len(prompts)} prompt(s), completion_frac={args.completion_frac}, "
          "upper bound for suite methods (loops run to max_iters, every repair round "
          "fires, every dream critic says regenerate; legacy-* arms do not force their "
          "repair rounds); no network calls were made")
    print(format_estimate(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


def cmd_calibrate(args: argparse.Namespace, rest: Sequence[str]) -> None:
    from blockgen.agentic import calibrate
    calibrate.main(list(rest))


def cmd_rescore(args: argparse.Namespace) -> None:
    from blockgen.agentic.suite import rescore
    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    path = rescore(args.run_dir, scorers, arms=arms, mock=args.mock, renderer=args.renderer,
                   cache_mode=args.cache or "read_write")
    print(f"rescore -> {path}")
    blob = json.loads(Path(path).read_text())
    for arm, row in blob["arms"].items():
        print(f"  {arm:<28} mean {row['mean_value']}")


# ===========================================================================
# argparse
# ===========================================================================
def _add_common(p: argparse.ArgumentParser, *, run: bool = True) -> None:
    p.add_argument("--prompts", default=None,
                   help="short:4 | detailed | large | captions:<i> | path (.txt/.json)")
    p.add_argument("--n", type=int, default=None, help="prompt count (overrides ':n')")
    p.add_argument("--seed", type=int, default=0, help="prompt sampling seed")
    p.add_argument("--llm", default=None, help="LLM spec for every role, e.g. openai:gpt-5-mini")
    p.add_argument("--llm-role", action="append", default=[], metavar="ROLE=SPEC",
                   help="one role's LLM, e.g. judge=claude-code:sonnet (repeatable)")
    p.add_argument("--imagegen", default=None, help="image generator spec")
    p.add_argument("--renderer", default=None, help="textured | flat | null")
    p.add_argument("--mock", action="store_true",
                   help="fully offline: mock:auto LLM, mock:render dreams, recipe renderer kept (flat only if no EGL), "
                        "pixel scorers (explicit --llm/--imagegen/--renderer still win)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted recipe override, e.g. params.max_iters=5 (repeatable)")
    p.add_argument("--allow-spend", action="store_true",
                   help="allow cache misses on paying backends (sets BLOCKGEN_ALLOW_SPEND=1)")
    _add_plugin(p)
    if run:
        p.add_argument("--cache", default=None, choices=["read_write", "read_only", "off"])
        p.add_argument("--budget-usd", type=float, default=None, help="run-level $ cap")
        p.add_argument("--name", default=None, help="run-directory suffix")
        p.add_argument("--image", action="append", default=[],
                       help="reference image for every prompt (repeatable)")
        p.add_argument("--dreams-first", action="store_true",
                       help="generate all dreams first, then unload the image model")
        p.add_argument("--no-render", action="store_true", help="skip samples.png")
        p.add_argument("--variation", type=int, default=None,
                       help="explicit design-variation seed per build (changes requests)")
        p.add_argument("--base", default="outputs", help="parent of the run directory")


def _add_plugin(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plugin", action="append", default=[], metavar="FILE.py|MODULE",
                   help="import an out-of-tree module that @register's components, e.g. "
                        "a scratch method (repeatable; sets BLOCKGEN_AGENTIC_PLUGINS)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m blockgen.agentic", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="the component catalogue")
    p.add_argument("kind", nargs="?", choices=list(LIST_KINDS) + sorted(KIND_ALIASES))
    _add_plugin(p)

    p = sub.add_parser("show", help="a resolved recipe")
    p.add_argument("recipe")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    _add_plugin(p)

    p = sub.add_parser("run", help="run recipes over prompts")
    p.add_argument("--recipe", required=True, help="comma list of recipe names or paths")
    _add_common(p)

    p = sub.add_parser("battery", help="run a battery YAML")
    p.add_argument("file")
    p.add_argument("--only", default=None, help="comma list of arm names")
    _add_common(p)

    p = sub.add_parser("estimate", help="upper-bound cost, zero network calls")
    p.add_argument("--recipe", default=None)
    p.add_argument("--battery", default=None)
    p.add_argument("--completion-frac", type=float, default=0.3,
                   help="completion tokens = gen.max_tokens x this")
    p.add_argument("--json", default=None, help="also write the rows as JSON")
    _add_common(p, run=False)

    sub.add_parser("calibrate", help="scorer calibration (args go to blockgen.agentic.calibrate)",
                   add_help=False)

    p = sub.add_parser("rescore", help="re-run scorers offline over a run's traces")
    p.add_argument("run_dir")
    p.add_argument("--scorers", required=True, help="comma list, e.g. clip_text,vlm")
    p.add_argument("--arms", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--renderer", default=None)
    p.add_argument("--cache", default=None, choices=["read_write", "read_only", "off"])
    p.add_argument("--allow-spend", action="store_true")
    _add_plugin(p)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    from blockgen.agentic.core import SPEND_ENV, ConfigError, SpendNotAllowed
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["calibrate"]:
        cmd_calibrate(argparse.Namespace(), argv[1:])
        return 0
    args = build_parser().parse_args(argv)
    if getattr(args, "allow_spend", False):
        os.environ[SPEND_ENV] = "1"
    plugins = list(getattr(args, "plugin", None) or [])
    if plugins:
        from blockgen.agentic.core import PLUGINS_ENV, load_plugins
        # Absolute paths in the env, so a child process with another cwd finds them.
        plugins = [str(Path(p).expanduser().resolve()) if p.endswith(".py") else p
                   for p in plugins]
        try:
            load_plugins(plugins)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        # Exported only once loaded, so a bad path never poisons later imports.
        prev = [x for x in os.environ.get(PLUGINS_ENV, "").split(os.pathsep) if x]
        os.environ[PLUGINS_ENV] = os.pathsep.join(dict.fromkeys(prev + plugins))
    handlers = {"list": cmd_list, "show": cmd_show, "run": cmd_run, "battery": cmd_battery,
                "estimate": cmd_estimate, "rescore": cmd_rescore}
    try:
        status = handlers[args.cmd](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except SpendNotAllowed as exc:
        print(f"\nSpendNotAllowed: {exc}\nNothing was charged. Re-run with --allow-spend "
              f"to permit paid calls, or --mock / --cache read_only to stay free.",
              file=sys.stderr)
        return 2
    return int(status or 0)


if __name__ == "__main__":
    sys.exit(main())
