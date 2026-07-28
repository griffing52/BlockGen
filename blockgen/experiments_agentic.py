"""Agentic battery (track E): ablate the loop, the prompt and the model.

One driver, one run directory, N *arms* — where an arm is an
:class:`~blockgen.agentic.agent.AgentConfig` and every arm sees the *same*
prompts, so the comparison isolates the loop rather than the request. The arms
answer the questions the track exists to test:

  ``zeroshot``      program-only, no scaffolding — the honest baseline
  ``oneshot``       + one in-context example program (format + grounding)
  ``plan``          + a natural-language design pass first
  ``repair``        + symbolic feedback from the executor (errors, no-ops)
  ``critique``      + visual feedback: the model sees a render of its own build
  ``full``          plan + example + repair + critique, the everything arm

Metrics per build come from :meth:`BuildResult.metrics` — block count, connected
components and largest-component fraction (the same connectivity notion
``blockgen/eval/validity.py`` applies to the neural tracks), command success rate,
tokens and dollars. Every arm writes a textured sample sheet and a structure cache
in the standard format, so agentic builds go straight into the existing eval and
render tooling.

Cost control: responses are cached on disk by request hash, so re-running a
battery to add a metric or re-render costs nothing. Start with ``--quick``.

    .venv/bin/python -m blockgen.experiments_agentic --quick --provider mock
    .venv/bin/python -m blockgen.experiments_agentic --arms zeroshot,plan,full \\
        --prompts detailed --n 8 --provider openai:gpt-5-mini
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Dict, List

from blockgen.agentic.agent import AgentConfig, BuildAgent, BuildResult
from blockgen.agentic.providers import get_provider
from blockgen.agentic.report import write_run
from blockgen.agentic.tasks import describe_sources, load_prompts
from blockgen.utils.runs import new_run_dir

# Arm definitions. Each is a partial AgentConfig: fields set here override the
# battery-wide defaults (provider, canvas, token budget) so a new ablation is one
# dict entry, not a new code path.
ARMS: Dict[str, dict] = {
    "zeroshot":  dict(plan=False, n_examples=0, repair_rounds=0, critique_rounds=0),
    "oneshot":   dict(plan=False, n_examples=1, repair_rounds=0, critique_rounds=0),
    "plan":      dict(plan=True,  n_examples=0, repair_rounds=0, critique_rounds=0),
    "repair":    dict(plan=False, n_examples=0, repair_rounds=2, critique_rounds=0),
    "critique":  dict(plan=False, n_examples=0, repair_rounds=1, critique_rounds=1),
    "full":      dict(plan=True,  n_examples=1, repair_rounds=2, critique_rounds=1),
}

DEFAULT_ARMS = "zeroshot,oneshot,plan,repair,full"


def run_arm(name: str, overrides: dict, prompts: List[str], args) -> List[BuildResult]:
    """Run one arm over every prompt. A failed build is recorded, never fatal."""
    cfg = AgentConfig(
        provider=args.provider,
        canvas_size=(args.canvas, args.canvas_height or args.canvas, args.canvas),
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        target_blocks=args.target_blocks,
        critique_mode=args.critique_mode,
        cache=not args.no_cache,
        verbose=not args.quiet,
        **overrides,
    )
    provider = get_provider(cfg.provider, cache=cfg.cache, **cfg.provider_params())
    agent = BuildAgent(provider, cfg)
    results: List[BuildResult] = []
    for i, prompt in enumerate(prompts):
        print(f"[{name}] {i + 1}/{len(prompts)}: {prompt[:80]}", flush=True)
        try:
            results.append(agent.build(prompt))
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill a sweep
            traceback.print_exc()
            from blockgen.agentic.canvas import Canvas
            from blockgen.agentic.dsl import ExecutionReport
            empty = Canvas(cfg.canvas_size)
            results.append(BuildResult(
                description=prompt, structure=empty.to_structure(crop=False),
                canvas=empty, program_text="", report=ExecutionReport(),
                config=cfg, error=f"{type(exc).__name__}: {exc}"))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config (path or name under configs/); its values become "
                         "defaults, explicit CLI flags still override")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help=f"comma list of: {', '.join(ARMS)}")
    ap.add_argument("--prompts", default="detailed",
                    help="prompt source:\n" + describe_sources())
    ap.add_argument("--n", type=int, default=8, help="prompts per arm")
    ap.add_argument("--seed", type=int, default=0, help="prompt sampling seed")
    ap.add_argument("--provider", default="openai:gpt-5-mini",
                    help="'<vendor>:<model>' — openai, gemini, anthropic, or mock")
    ap.add_argument("--canvas", type=int, default=48, help="canvas x/z extent")
    ap.add_argument("--canvas-height", type=int, default=None,
                    help="canvas y extent (defaults to --canvas)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning-effort", default="medium",
                    help="reasoning models only; 'none' to omit")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--target-blocks", type=int, default=None,
                    help="nudge the model towards a build of roughly this mass")
    ap.add_argument("--critique-mode", default="rewrite", choices=["rewrite", "patch"])
    ap.add_argument("--name", default="agentic", help="run-directory suffix")
    ap.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    ap.add_argument("--no-render", action="store_true", help="skip sample sheets")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="2 prompts, zeroshot+full only — the smoke test")
    args = ap.parse_args()
    if args.config:
        from blockgen.config import apply_to_parser, load_config
        apply_to_parser(ap, load_config(args.config))
        args = ap.parse_args()  # re-parse so CLI flags still win over the config
    if args.reasoning_effort in ("none", "None", ""):
        args.reasoning_effort = None
    if args.quick:
        args.n = 2
        args.arms = "zeroshot,full"

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")

    prompts = load_prompts(args.prompts, n=args.n, seed=args.seed)
    if not prompts:
        raise SystemExit(f"prompt source '{args.prompts}' yielded nothing")

    run = new_run_dir(args.name)
    print(f"=== agentic battery -> {run} ===", flush=True)
    print(f"provider={args.provider} arms={arm_names} prompts={len(prompts)} "
          f"canvas={args.canvas}", flush=True)
    t0 = time.time()

    arms: Dict[str, List[BuildResult]] = {}
    for name in arm_names:
        arms[name] = run_arm(name, ARMS[name], prompts, args)

    config = {"args": vars(args), "arms": {a: ARMS[a] for a in arm_names},
              "prompts": prompts}
    metrics = write_run(run, arms=arms, config=config,
                        title=f"agentic battery ({args.provider}, {args.prompts})",
                        max_dim=max(args.canvas, args.canvas_height or args.canvas),
                        render=not args.no_render)

    print(f"\n=== done in {time.time() - t0:.0f}s -> {run} ===", flush=True)
    for name, agg in metrics["arms"].items():
        print(f"  {name:<10} blocks {agg['blocks']}  components {agg['n_components']}  "
              f"coherence {agg['coherence_rate']}  cmd_ok {agg['command_success_rate']}  "
              f"${agg['total_cost_usd']}", flush=True)
    print((run / "summary.md").read_text().split("## Config")[0], flush=True)


if __name__ == "__main__":
    main()
