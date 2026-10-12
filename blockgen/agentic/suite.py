"""Run recipes and batteries into one run directory; estimate and rescore them.

This is the integration layer of the methods suite. Everything below it (recipes,
kits, episodes, traces, scorers) works on one arm or one build; this module turns a
list of arms and a prompt set into the standard agentic run directory that
``report.write_run`` defines, so a suite run is benchmarkable with
``python -m blockgen.eval.bench`` exactly like a legacy battery::

    outputs/run_<stamp>_<name>/
      config.json        every resolved recipe + recipe_hash + git commit/dirty + versions
      metrics.json       per-build metrics, per-arm aggregates, per-arm ledger summary
      summary.md         tables (incl. method metrics and paid / would_cost / cost_known)
      samples.png        textured render of every build (repo rule: every run renders)
      <arm>_32.npz       builds in the structure-cache format, provenance in the manifest
      contact_<arm>.png  stacked per-build iteration strips
      variants.png       rows = prompts, cols = arms, cells 'dream | final'
      builds/<arm>/NNN_*.txt|json + NNN/ (the per-build Trace)

Four entry points, all used by the CLI (``python -m blockgen.agentic``):

    run_recipes   arms x prompts -> run dir (a failed build is recorded, never fatal)
    run_battery   a Battery (YAML) -> run_recipes with its prompts, budget and cache
    estimate      the same loop on free mocks, each call priced as the REAL model
                  would be: an upper-bound dollar figure with zero network calls
    rescore       re-run scorers offline over a run's traces (structures + dreams)

Why arms run sequentially in one process: the textured renderer holds one EGL
context and diffusers holds the GPU; parallel arms would contend for both, and
the ledger tree (build -> arm -> battery) is simplest to reason about in order.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from blockgen.agentic.core import (Budget, Charge, ConfigError, ImageRequest, ImageResult,
                                   Kit, Ledger, MethodResult, MethodSpec, NullTrace,
                                   SpendNotAllowed, load_builtins, lookup, parse_spec)
from blockgen.agentic.providers import LLMProvider, LLMResponse, Message

_REPO = Path(__file__).resolve().parents[2]

#: Built-in prompt sets of ``tasks.load_prompts`` that take ``:n`` as a count.
_COUNT_SOURCES = ("short", "detailed", "large")


# ===========================================================================
# Small helpers
# ===========================================================================
def parse_prompts(source: str, n: Optional[int] = None, seed: int = 0) -> List[str]:
    """CLI prompt source -> prompts.

    ``short:4`` means the built-in ``short`` set with n=4 (the suite's shorthand).
    ``captions:2`` keeps ``tasks.load_prompts``' meaning (caption detail index 2),
    with the count from ``n``. A plain name or a .txt/.json path also takes ``n``.
    An explicit ``n`` overrides the ``:n`` shorthand.
    """
    from blockgen.agentic.tasks import load_prompts
    name, sep, arg = source.partition(":")
    if sep and name in _COUNT_SOURCES and arg.isdigit():
        return load_prompts(name, n=int(n if n is not None else arg), seed=seed)
    return load_prompts(source, n=8 if n is None else int(n), seed=seed)


def git_info() -> Dict[str, Any]:
    """Commit and dirty flag of the repo, so a run says what code produced it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:  # noqa: BLE001 - not a git checkout (release tarball): still run
        return {"git_commit": "", "git_dirty": None, "git_branch": ""}
    return {"git_commit": commit, "git_dirty": dirty, "git_branch": branch}


def _jsonable(obj: Any) -> Any:
    """Round-trip through JSON (str fallback): manifests are written without default=."""
    return json.loads(json.dumps(obj, default=str))


def _versions(kits: Sequence[Kit]) -> Dict[str, Any]:
    """Interpreter + backend versions that change outputs (e.g. the Claude Code CLI)."""
    out: Dict[str, Any] = {"python": platform.python_version()}
    for kit in kits:
        for p in kit.llms.values():
            inner = getattr(p, "inner", p)
            v = getattr(inner, "params", {}).get("cli_version") or getattr(inner, "cli_version", None)
            if v:
                out[f"cli:{getattr(inner, 'name', 'cli')}"] = str(v)
    return out


def spec_from_resolved(d: Mapping[str, Any]) -> MethodSpec:
    """A MethodSpec back from ``spec.resolved()`` (as stored in config.json)."""
    names = {f.name for f in dataclasses.fields(MethodSpec)}
    kw = {k: v for k, v in d.items() if k in names}
    kw["canvas"] = tuple(kw.get("canvas") or (32, 32, 32))
    return MethodSpec(**kw)


def prepare_spec(spec: MethodSpec, *, mock: bool = False,
                 post_mock: Optional[Mapping[str, Any]] = None) -> MethodSpec:
    """The spec an arm actually runs: mock-substituted, then ``post_mock`` applied.

    ``post_mock`` holds explicit CLI choices that must win over the mock table, e.g.
    ``--mock --renderer textured`` (offline, but real-looking renders) or ``--mock
    --imagegen mock:solid`` (no corpus needed). Overrides use the recipe's dotted
    syntax, and the spec keeps its name, so the arm name is unchanged.
    """
    from blockgen.agentic import recipe as recipe_mod
    from blockgen.agentic.kit import mock_spec
    if mock:
        spec = mock_spec(spec)
    if post_mock:
        d = {f.name: getattr(spec, f.name) for f in dataclasses.fields(MethodSpec)}
        d["canvas"] = list(spec.canvas)
        d = recipe_mod.apply_overrides(d, post_mock)
        name = d.pop("name")
        spec = recipe_mod._coerce_spec(d, name)
    if spec.image and parse_spec(spec.image)[:2] == ("mock", "render"):
        # mock:render dreams ARE train-split houses: such an arm is an oracle, never
        # ranked, whether a recipe asked for it or --mock substituted it.
        spec = dataclasses.replace(spec, tags={**spec.tags, "train_split": "oracle_train",
                                               "leaderboard": False})
    return spec


def _empty_result(spec: MethodSpec, prompt: str, error: str) -> MethodResult:
    """A recorded failure: an empty build that keeps its prompt, recipe and error."""
    from blockgen.agentic.canvas import Canvas
    from blockgen.agentic.dsl import ExecutionReport
    canvas = Canvas(tuple(int(v) for v in spec.canvas))
    return MethodResult(description=prompt, structure=canvas.to_structure(crop=False),
                        canvas=canvas, program_text="", report=ExecutionReport(),
                        error=error, spec_name=spec.name, recipe_hash=spec.hash(),
                        truncated="error", provenance=spec.provenance())


def _arm_dirname(arm: str) -> str:
    """Arm names may contain sweep punctuation ('dream@max_iters=1'); keep them as-is
    on disk (Linux-safe) except path separators."""
    return arm.replace("/", "_")


def _print_ledger(arm: str, summary: Mapping[str, Any]) -> None:
    known = "" if summary.get("cost_known") else "  (cost_known=False: some calls unpriced)"
    print(f"  {arm:<28} paid ${summary['usd']:.4f}  would_cost ${summary['would_cost_usd']:.4f}"
          f"  llm_calls {summary['llm_calls']}  images {summary['images']}{known}", flush=True)


#: Views rendered for the trace of a build whose recipe has no scorer (so the loop
#: never rendered it): enough for contact sheets and variants.png to show it.
TRACE_VIEWS: Tuple[str, ...] = ("iso", "iso_back")


def _ensure_trace_renders(kit: Kit, res: MethodResult, trace: Any) -> None:
    """Render unrendered non-empty attempts, then refresh the trace.

    Why: arms without scorers (classic, legacy) never render inside the loop, and
    their contact sheets / variants cells would be blank. Rendering happens AFTER
    the episode, so nothing the model saw (or paid for) changes.
    """
    if kit.renderer is None:
        return
    todo = [a for a in res.attempts if not a.renders and a.blocks > 0 and a.world is not None]
    if not todo:
        return
    from blockgen.agentic.render import render_pngs
    for a in todo:
        try:
            a.renders.update(render_pngs(kit.renderer, a.world.to_structure(),
                                         list(TRACE_VIEWS)))
        except Exception as exc:  # noqa: BLE001 - a missing picture is not a failed build
            trace.event("trace_render_error", idx=a.idx, error=repr(exc))
    trace.close(res)            # rewrites renders + contact.png


def validate_specs(specs: Sequence[MethodSpec]) -> None:
    """Validate every arm (:func:`kit.validate_spec`) before any arm makes a call.

    Raises :class:`ConfigError` naming the arm. Why all up front: kits are built
    lazily per arm, so a typo in the last arm of a battery would otherwise surface
    only after every earlier arm had run (and paid).
    """
    from blockgen.agentic.kit import validate_spec
    load_builtins()
    for s in specs:
        try:
            validate_spec(s)
        except (TypeError, KeyError, ValueError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ConfigError(f"arm '{s.name}': {type(exc).__name__}: {msg}") from exc


# ===========================================================================
# dreams-first
# ===========================================================================
class _PrefetchLedger(Ledger):
    """Ledger of the dreams-first pre-pass.

    Its charges still reach (and are capped by) the arm and battery ledgers, but
    as role ``<role>_prefetch`` with ``would_cost_usd = 0``: the build phase reads
    the same images back as cache hits and records their full price there, so the
    compute figure counts each image once. Paid usd is counted here, once.
    """

    def charge(self, c: Charge) -> None:
        super().charge(dataclasses.replace(c, role=f"{c.role}_prefetch", would_cost_usd=0.0))


def _prefetches_dreams(kit: Kit) -> bool:
    """False for a method registered with ``prefetch_dreams=False``.

    Why: staged methods (blueprint, floors, turnaround) make their dreams themselves,
    from images that only exist mid-episode (a massing render, a split sheet). A
    per-key prefetch would buy images the method never asks for, e.g. four separate
    per-view images where turnaround wants one sheet.
    """
    from blockgen.agentic.core import lookup
    try:
        return bool(lookup("method", kit.spec.method).meta.get("prefetch_dreams", True))
    except Exception:  # noqa: BLE001 - an unknown method fails later, with a clear error
        return True


def _dreams_first(kit: Kit, prompts: Sequence[str], images: Sequence[bytes],
                  seed: Optional[int] = None) -> None:
    """Generate every prompt's (unrefined) dreams before any build, then unload.

    Why: a local diffusers model and the textured renderer / scorer models compete
    for one GPU. Generating all dreams first (they land in the content-addressed
    dream cache) and unloading the pipeline means the build phase reads dreams from
    the cache instead of keeping the image model resident. Requests are built by the
    real Episode with the build phase's ``seed``, so the cache keys match exactly
    what the build phase will ask for. The pre-pass is paid once (see
    :class:`_PrefetchLedger`); per-build results then see cache hits (paid $0,
    would_cost full price), so would_cost counts each image once.
    """
    if not kit.dreams or kit.image is None:
        return
    if not _prefetches_dreams(kit):
        return
    from blockgen.agentic import describe as describe_mod
    from blockgen.agentic.episode import Episode
    from blockgen.agentic.recipes.dream import make_dreams
    for prompt in prompts:
        ledger = _PrefetchLedger(parent=kit.ledger, name=f"dreams:{prompt[:32]}")
        ep = Episode(prompt, kit, ledger=ledger, seed=seed, images=images, trace=NullTrace())
        try:
            # A describe stage that expands into the image changes the dream
            # requests: write the spec first (the build phase hits its LLM cache).
            dcfg = describe_mod.config_from_params(kit.spec.params or {})
            if dcfg is not None and dcfg.expands and dcfg.expand_into_image:
                describe_mod.setup(ep, dcfg)
            make_dreams(ep, refine=0)
        except SpendNotAllowed:
            raise
        except Exception as exc:  # noqa: BLE001 - the build phase will retry and record it
            print(f"[dreams-first] {prompt[:60]}: {type(exc).__name__}: {exc}", flush=True)
    unload = getattr(kit.image, "unload", None)
    if callable(unload):
        unload()


# ===========================================================================
# run_recipes / run_battery
# ===========================================================================
def run_recipes(specs: Sequence[MethodSpec], prompts: Sequence[str], *, name: str,
                mock: bool = False, post_mock: Optional[Mapping[str, Any]] = None,
                cache_mode: Optional[str] = None, budget: Optional[Budget] = None,
                images: Sequence[bytes] = (), dreams_first: bool = False,
                render: bool = True, base: Union[str, Path] = "outputs",
                battery: Optional[Mapping[str, Any]] = None, seed: Optional[int] = None,
                max_dim: int = 32, keep_empty: bool = False,
                extra_config: Optional[Mapping[str, Any]] = None) -> Path:
    """Run every arm over every prompt into ``<base>/run_<stamp>_<name>[_mock]``.

    Every arm's recipe is validated first (:func:`kit.validate_spec`): a config
    error (typo in a param, scorer, dream, ...) raises :class:`ConfigError` before
    ANY arm makes a call. Per arm, ``build_kit`` then runs once (heavy models load
    once per arm), and each prompt gets ``run_episode`` with a Trace at
    ``builds/<arm>/NNN``. A build that fails, or an arm whose kit cannot be built
    at runtime (missing optional dependency), is recorded as an error result
    (``config.json`` ``kit_errors``) and the run goes on. ``SpendNotAllowed`` is the one
    exception that propagates: it means "this run would spend money and you did not
    say so", which must stop the run before any further call.

    ``seed=None`` keeps BuildAgent-identical requests (legacy cache keys); an explicit
    seed asks every build for a distinct design variation. ``max_dim=32`` writes
    ``<arm>_32.npz``, the houses32-v1 comparability protocol. ``keep_empty`` stays
    False until the bench reads ``n_requested`` (see report.save_structures).
    """
    from blockgen.agentic.episode import run_episode
    from blockgen.agentic.kit import build_kit
    from blockgen.agentic.report import write_run
    from blockgen.agentic.trace import Trace, arm_contact_sheet, variants_sheet
    from blockgen.utils.runs import new_run_dir

    load_builtins()
    prompts = list(prompts)
    if not prompts:
        raise ValueError("run_recipes: no prompts")
    names = [s.name for s in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate arm name(s) {dup}; rename with a battery 'as:'")
    prepared = {s.name: prepare_spec(s, mock=mock, post_mock=post_mock) for s in specs}
    validate_specs(prepared.values())
    run_name = name + ("_mock" if mock and not name.endswith("_mock") else "")
    run_dir = new_run_dir(run_name, base=str(base))
    git = git_info()
    battery_ledger = Ledger(budget or Budget(), name=str((battery or {}).get("name", run_name)))
    print(f"=== agentic suite -> {run_dir} ===", flush=True)
    print(f"arms={names} prompts={len(prompts)} mock={mock} cache={cache_mode or 'recipe'}",
          flush=True)
    t0 = time.time()

    arms: Dict[str, List[MethodResult]] = {}
    kits: Dict[str, Kit] = {}
    run_specs: Dict[str, MethodSpec] = {}
    provenance: Dict[str, Dict[str, Any]] = {}
    kit_errors: Dict[str, str] = {}
    for spec in specs:
        arm = spec.name
        spec_run = prepared[arm]
        run_specs[arm] = spec_run
        results: List[MethodResult] = []
        try:
            kit = build_kit(spec_run, mock=False, cache_mode=cache_mode, ledger=battery_ledger)
            kit.mock = mock
            kits[arm] = kit
        except SpendNotAllowed:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. diffusers not installed: skip arm
            traceback.print_exc()
            err = f"kit: {type(exc).__name__}: {exc}"
            kit_errors[arm] = err
            print(f"[{arm}] cannot build kit, recording {len(prompts)} failed builds", flush=True)
            arms[arm] = [_empty_result(spec_run, p, err) for p in prompts]
            provenance[arm] = _jsonable({**spec_run.provenance(), "n_requested": len(prompts),
                                         "git_commit": git["git_commit"],
                                         "git_dirty": git["git_dirty"], "mock": mock,
                                         "kit_error": err})
            continue
        if dreams_first:
            _dreams_first(kit, prompts, images, seed=seed)
        for i, prompt in enumerate(prompts):
            print(f"[{arm}] {i + 1}/{len(prompts)}: {prompt[:80]}", flush=True)
            root = run_dir / "builds" / _arm_dirname(arm) / f"{i:03d}"
            try:
                trace = Trace.for_kit(root, kit)
                res = run_episode(kit, prompt, images=images, seed=seed, trace=trace)
                _ensure_trace_renders(kit, res, trace)
            except SpendNotAllowed:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad build must not kill a battery
                traceback.print_exc()
                res = _empty_result(spec_run, prompt, f"{type(exc).__name__}: {exc}")
                res.trace_dir = str(root)
            if res.truncated:
                print(f"[{arm}]   truncated: {res.truncated} ({res.error[:120]})", flush=True)
            results.append(res)
        arms[arm] = results
        provenance[arm] = _jsonable({**spec_run.provenance(), "n_requested": len(prompts),
                                     "git_commit": git["git_commit"],
                                     "git_dirty": git["git_dirty"], "mock": mock})

    config = {
        "suite": True,
        "name": run_name,
        "battery": dict(battery) if battery else None,
        "prompts": prompts,
        "mock": mock,
        "post_mock": dict(post_mock or {}),
        "cache_mode": cache_mode,
        "seed": seed,
        "budget": dataclasses.asdict(budget) if budget else None,
        "recipes": {arm: {**s.resolved(), "recipe_hash": s.hash()} for arm, s in run_specs.items()},
        "source_recipes": {s.name: {**s.resolved(), "recipe_hash": s.hash()} for s in specs},
        **git,
        "versions": _versions(list(kits.values())),
        "kit_errors": kit_errors,
        "argv": list(sys.argv),
        **dict(extra_config or {}),
    }
    title = f"agentic suite: {run_name}"
    metrics = write_run(run_dir, arms=arms, config=_jsonable(config), title=title,
                        max_dim=max_dim, render=render, provenance=provenance)
    if keep_empty:
        from blockgen.agentic.report import save_structures
        for arm, results in arms.items():
            save_structures(results, run_dir, arm, max_dim=max_dim, keep_empty=True,
                            extra_report=provenance.get(arm))

    # Ledgers: what each arm (and the whole run) paid vs. what it would have cost.
    metrics["ledger"] = battery_ledger.summary()
    for arm, kit in kits.items():
        metrics["arms"][arm]["ledger"] = kit.ledger.summary()
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=1))

    # Visual artifacts drawn from the traces (they read files only, never live objects).
    for arm in arms:
        roots = [Path(r.trace_dir) for r in arms[arm]
                 if r.trace_dir and Path(r.trace_dir).is_dir()]
        if roots:
            try:
                arm_contact_sheet(roots, run_dir / f"contact_{_arm_dirname(arm)}.png", k=8)
            except Exception as exc:  # noqa: BLE001 - a sheet is a convenience, not a result
                print(f"[contact] {arm}: {type(exc).__name__}: {exc}", flush=True)
    try:
        variants_sheet(run_dir, [_arm_dirname(a) for a in arms], k=12)
    except Exception as exc:  # noqa: BLE001
        print(f"[variants] {type(exc).__name__}: {exc}", flush=True)

    with (run_dir / "summary.md").open("a") as f:
        f.write("\n## Ledger\n\n```json\n" + json.dumps(_jsonable(metrics["ledger"]), indent=1)
                + "\n```\n")

    print(f"\n=== done in {time.time() - t0:.0f}s -> {run_dir} ===", flush=True)
    for arm, kit in kits.items():
        _print_ledger(arm, kit.ledger.summary())
    _print_ledger("TOTAL", metrics["ledger"])
    return run_dir


def run_battery(battery: Any, *, prompts: Optional[Sequence[str]] = None,
                only: Optional[Sequence[str]] = None, name: Optional[str] = None,
                cache_mode: Optional[str] = None, budget: Optional[Budget] = None,
                **kw: Any) -> Path:
    """Run a :class:`~blockgen.agentic.recipe.Battery` (or a path / name of one).

    The battery's prompt spec (``tasks.load_prompts`` kwargs), budget and cache mode
    are used unless the caller overrides them; ``only`` keeps a subset of arms.
    """
    from blockgen.agentic.recipe import Battery, load_battery
    b = battery if isinstance(battery, Battery) else load_battery(battery)
    specs = list(b.recipes)
    if only:
        missing = sorted(set(only) - {s.name for s in specs})
        if missing:
            raise ConfigError(f"--only: arm(s) {missing} not in battery {b.name}: {b.arm_names()}")
        specs = [s for s in specs if s.name in set(only)]
    if prompts is None:
        p = dict(b.prompts or {})
        source = str(p.pop("source", "detailed"))
        prompts = parse_prompts(source, n=p.pop("n", None), seed=int(p.pop("seed", 0)))
    battery_meta = {"name": b.name, "prompts": b.prompts, "cache": b.cache,
                    "budget": dataclasses.asdict(b.budget), "arms": [s.name for s in specs]}
    return run_recipes(specs, prompts, name=name or b.name,
                       cache_mode=cache_mode or b.cache, budget=budget or b.budget,
                       battery=battery_meta, **kw)


# ===========================================================================
# estimate
# ===========================================================================
#: Worst-case verdicts for the estimate's mock critic: every dream is rejected (refine
#: runs all K regenerations) and every stalled build is called unreachable (the
#: re-dream fires). Together with ``Kit.worst_case`` (every repair round runs) and
#: :func:`_upper_bound_params` (loops run to max_iters) the figure is an upper bound
#: for suite methods. Exception: ``legacy-*`` arms run BuildAgent, whose repair
#: rounds still depend on the (always clean) mock replies, so they are undercounted.
ESTIMATE_JSON = {"buildable": False, "unreachable": True}


class EstimatingLLM(LLMProvider):
    """mock:auto replies, priced as the REAL configured model would charge.

    Prompt tokens come from AutoMock's estimate (chars/4 + 800 per image);
    completion tokens are ``gen.max_tokens * completion_frac``. A request already in
    the real response cache (``peek``) is marked cached, i.e. $0 paid. No network.
    """

    name = "estimate"
    spends = False

    def __init__(self, mock: LLMProvider, real: Optional[LLMProvider], model: str,
                 completion_tokens: int, billing: str = "api") -> None:
        super().__init__(model)
        self.mock, self.real = mock, real
        self.completion_tokens = int(completion_tokens)
        self.billing = billing
        self.n_peek_hits = 0

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        r = self.mock.complete(messages, **overrides)
        hit = False
        peek = getattr(self.real, "peek", None)
        if callable(peek):
            try:
                hit = bool(peek(messages, **overrides))
            except Exception:  # noqa: BLE001 - a peek failure only loses a $0 discount
                hit = False
        self.n_peek_hits += int(hit)
        return LLMResponse(text=r.text, model=self.model, prompt_tokens=r.prompt_tokens,
                           completion_tokens=self.completion_tokens, cached=hit)

    def describe(self) -> str:
        return f"estimate({self.model})"


class EstimatingImageGen:
    """A free placeholder image, priced and cache-checked as the real generator.

    Size / quality / ref support come from the real generator, so the Episode builds
    the same ImageRequest it would in a real run (same price row, same cache key).
    """

    spends = False

    def __init__(self, placeholder: Any, real: Optional[Any], spec: str) -> None:
        self.placeholder, self.real, self.spec = placeholder, real, spec
        src = real if real is not None else placeholder
        self.name = str(getattr(src, "name", "image"))
        self.model = str(getattr(real, "model", spec) if real is not None else spec)
        self.billing = str(getattr(real, "billing", "api")) if real is not None else "api"
        self.size = str(getattr(real, "size", "1024x1024")) if real is not None else "1024x1024"
        self.quality = str(getattr(real, "quality", "low")) if real is not None else "low"

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        return self.real.estimate_usd(req) if self.real is not None else None

    def supports_refs(self) -> bool:
        return bool(self.real.supports_refs()) if self.real is not None else True

    def describe(self) -> str:
        return f"estimate({self.spec})"

    def generate(self, req: ImageRequest) -> ImageResult:
        res = self.placeholder.generate(req)
        est = self.estimate_usd(req)
        hit = False
        peek = getattr(self.real, "peek", None)
        if callable(peek):
            try:
                hit = bool(peek(req))
            except Exception:  # noqa: BLE001
                hit = False
        return ImageResult(images=list(res.images), model=self.model,
                           cost_usd=float(est or 0.0), priced=est is not None, cached=hit)

    def unload(self) -> None:
        return None


def _upper_bound_params(spec: MethodSpec) -> Dict[str, Any]:
    """Method params that make loops run to their maximum (stop_at=inf, no patience)."""
    name = parse_spec(spec.method)[0]
    try:
        fn = lookup("method", name).load()
        sig = inspect.signature(fn).parameters
    except Exception:  # noqa: BLE001
        return {}
    takes = (lambda k: k in sig or name == "dream_vote")   # dream_vote forwards **loop
    out: Dict[str, Any] = {}
    if takes("stop_at"):
        out["stop_at"] = math.inf
    if takes("patience"):
        out["patience"] = 10 ** 6
    return out


def estimate_kit(spec: MethodSpec, *, completion_frac: float = 0.3,
                 ledger: Optional[Ledger] = None) -> Kit:
    """A kit that runs ``spec`` on free mocks but prices every call for real."""
    from blockgen.agentic import providers
    from blockgen.agentic.core import make
    from blockgen.agentic.kit import build_kit, mock_spec
    from blockgen.agentic.providers_cli import AutoMockProvider

    real_roles = spec.roles()
    gen = {**spec.gen}
    est = mock_spec(spec)
    est = dataclasses.replace(
        est, image="mock:solid" if spec.image else None,
        renderer="null" if spec.renderer not in (None, "", "none") else spec.renderer,
        params={**(spec.params or {}), **_upper_bound_params(spec)})
    kit = build_kit(est, mock=False, cache_mode="off", ledger=ledger)
    kit.mock = True
    kit.worst_case = True       # every repair round fires (mock replies never fail)
    completion = int(float(gen.get("max_tokens") or 16000) * float(completion_frac))
    built: Dict[str, LLMProvider] = {}
    for role, s in real_roles.items():
        if s not in built:
            try:
                real = providers.get_provider(s, cache=True, cache_mode="read_only",
                                              spend_gate=True, **gen)
                model, billing = real.model, str(getattr(real, "billing", "api"))
            except Exception:  # noqa: BLE001 - e.g. no claude binary: unpriced, no peek
                real, model, billing = None, s, "api"
            mock = AutoMockProvider(json_overrides=ESTIMATE_JSON)
            built[s] = EstimatingLLM(mock, real, model, completion, billing)
        kit.llms[role] = built[s]
    if spec.image:
        try:
            from blockgen.agentic.imagegen import get_image_gen
            real_img = get_image_gen(spec.image, cache="read_only", spend_gate=True)
        except Exception:  # noqa: BLE001 - e.g. diffusers not installed: unpriced
            real_img = None
        kit.image = EstimatingImageGen(make("image", "mock:solid"), real_img,
                                       str(spec.image))
    return kit


def estimate(specs: Sequence[MethodSpec], prompts: Sequence[str], *,
             completion_frac: float = 0.3) -> List[Dict[str, Any]]:
    """Upper-bound cost of running ``specs`` over ``prompts``. Zero network calls.

    Returns one row per arm (plus a TOTAL row): builds, LLM calls, images, estimated
    paid USD (cache hits free) and estimated full-price USD. ``cost_known`` is False
    when some backend is unpriced (Claude Code, local), whose dollars are then 0.
    Requests after the first turn of a build depend on model replies, so only the
    first turns can be found in the real cache: the paid figure is conservative.
    """
    from blockgen.agentic.episode import run_episode

    load_builtins()
    validate_specs(specs)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        ledger = Ledger(name=spec.name)
        kit = estimate_kit(spec, completion_frac=completion_frac, ledger=ledger)
        errors = 0
        for prompt in prompts:
            res = run_episode(kit, prompt, trace=NullTrace())
            errors += int(res.truncated == "error")
        s = ledger.summary()
        n = max(1, len(prompts))
        rows.append({"arm": spec.name, "builds": len(prompts), "llm_calls": s["llm_calls"],
                     "images": s["images"], "est_paid_usd": round(s["usd"], 4),
                     "est_full_usd": round(s["would_cost_usd"], 4),
                     "est_full_usd_per_build": round(s["would_cost_usd"] / n, 4),
                     "cost_known": bool(s["cost_known"]), "errors": errors,
                     "budget_usd_per_build": (spec.budget or {}).get("usd_per_build")})
    rows.append({"arm": "TOTAL", "builds": sum(r["builds"] for r in rows),
                 "llm_calls": sum(r["llm_calls"] for r in rows),
                 "images": sum(r["images"] for r in rows),
                 "est_paid_usd": round(sum(r["est_paid_usd"] for r in rows), 4),
                 "est_full_usd": round(sum(r["est_full_usd"] for r in rows), 4),
                 "est_full_usd_per_build": None,
                 "cost_known": all(r["cost_known"] for r in rows),
                 "errors": sum(r["errors"] for r in rows), "budget_usd_per_build": None})
    return rows


def format_estimate(rows: Sequence[Mapping[str, Any]]) -> str:
    """Plain-text table for the CLI."""
    cols = ("arm", "builds", "llm_calls", "images", "est_paid_usd", "est_full_usd",
            "est_full_usd_per_build", "budget_usd_per_build", "cost_known", "errors")
    cells = [[("-" if r.get(c) is None else
               (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r[c]))) for c in cols]
             for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(c) for c in cells])


# ===========================================================================
# rescore
# ===========================================================================
def _load_trace_dreams(root: Path) -> Dict[str, list]:
    """``dreams/<key>_v<k>.{png,json}`` -> {key: [DreamVersion...]} in version order."""
    from blockgen.agentic.core import DreamVersion
    out: Dict[str, list] = {}
    for meta in sorted((root / "dreams").glob("*.json")):
        png = meta.with_suffix(".png")
        if not png.exists():
            continue
        m = json.loads(meta.read_text())
        out.setdefault(m["key"], []).append(DreamVersion(
            key=m["key"], version=int(m.get("version", 0)), png=png.read_bytes(),
            prompt=m.get("prompt", ""), generator=m.get("generator", ""),
            seed=int(m.get("seed", 0)), critique=m.get("critique") or {},
            cost_usd=float(m.get("cost_usd", 0.0)), cached=bool(m.get("cached", False))))
    for vs in out.values():
        vs.sort(key=lambda d: d.version)
    return out


def rescore(run_dir: Union[str, Path], scorers: Sequence[str], *,
            arms: Optional[Sequence[str]] = None, mock: bool = False,
            renderer: Optional[str] = None, cache_mode: str = "read_write") -> Path:
    """Re-run ``scorers`` over every attempt of a run, offline from its traces.

    Each attempt's world is rebuilt from ``iters/*/structure.npz`` and scored against
    the build's FINAL dream versions (the same anchor ``rescore_all`` uses), with the
    renders saved in the trace reused and missing views rendered. Nothing in the run
    dir is modified; results go to ``rescore_<stamp>.json``. Scorers that call an LLM
    (``vlm``) go through the normal cache and spend gate.
    """
    from blockgen.agentic.core import Attempt, MinecraftWorld
    from blockgen.agentic.dsl import ExecutionReport
    from blockgen.agentic.episode import Episode
    from blockgen.agentic.kit import build_kit
    from blockgen.agentic.trace import load_structure
    from blockgen.utils.runs import run_stamp

    load_builtins()
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    recipes = config.get("recipes") or {}
    if not recipes:
        raise ValueError(f"{run_dir}/config.json has no 'recipes': not a suite run")
    chosen = list(arms) if arms else list(recipes)
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scorers": list(scorers), "arms": {}}
    for arm in chosen:
        spec = spec_from_resolved(recipes[arm])
        over: Dict[str, Any] = {"score": {s: 1.0 for s in scorers}, "gates": {},
                                "final_score": {}}
        if renderer:
            over["renderer"] = renderer
        spec = dataclasses.replace(spec, **over)
        kit = build_kit(spec, mock=mock, cache_mode=cache_mode)
        arm_rows: Dict[str, Any] = {}
        for root in sorted((run_dir / "builds" / _arm_dirname(arm)).glob("[0-9][0-9][0-9]")):
            if not root.is_dir():
                continue
            result = json.loads((root / "result.json").read_text()) \
                if (root / "result.json").exists() else {}
            prompt = result.get("description", "")
            ep = Episode(prompt, kit, trace=NullTrace())
            ep.dreams = _load_trace_dreams(root)
            rows = {}
            for d in sorted((root / "iters").glob("*")):
                if not (d / "structure.npz").exists():
                    continue
                s = load_structure(d / "structure.npz")
                world = MinecraftWorld.from_structure(s, spec.canvas)
                att = Attempt(d.name, "", ExecutionReport(), LLMResponse("", "rescore"),
                              world.block_count(), idx=len(rows), world=world)
                for png in d.glob("render_*.png"):
                    att.renders[png.stem[len("render_"):]] = png.read_bytes()
                sc = ep.score(att)
                rows[d.name] = {"value": sc.value, "parts": sc.parts, "notes": sc.notes,
                                "dream_id": sc.dream_id}
            arm_rows[root.name] = {"prompt": prompt, "best_idx": result.get("best_idx"),
                                   "attempts": rows}
        vals = [r["value"] for b in arm_rows.values() for r in b["attempts"].values()]
        out["arms"][arm] = {"builds": arm_rows,
                            "mean_value": round(sum(vals) / len(vals), 4) if vals else None,
                            "ledger": kit.ledger.summary()}
    path = run_dir / f"rescore_{run_stamp()}.json"
    path.write_text(json.dumps(_jsonable(out), indent=1))
    return path


__all__ = ["ESTIMATE_JSON", "EstimatingImageGen", "EstimatingLLM", "estimate",
           "estimate_kit", "format_estimate", "git_info", "parse_prompts", "prepare_spec",
           "rescore", "run_battery", "run_recipes", "spec_from_resolved", "validate_specs"]
