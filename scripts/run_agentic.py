"""Single-prompt entry point for the agentic track (one build, one run dir).

The battery (``python -m blockgen.experiments_agentic``) is for comparing arms;
this is for looking at one build closely — iterating on prompts, on the DSL, or
on a provider — and for the two offline modes that need no API key at all.

    # build one thing
    .venv/bin/python scripts/run_agentic.py "a small oak cottage with a stone chimney"

    # planning pass + one example + a visual critique round, on Gemini
    .venv/bin/python scripts/run_agentic.py "a red brick village church" \\
        --provider gemini:gemini-2.5-pro --plan --examples 1 --critique-rounds 1

    # image-conditioned
    .venv/bin/python scripts/run_agentic.py "build this house" --image ref.png

    # offline: run a hand-written program through the executor and render it
    .venv/bin/python scripts/run_agentic.py --program my_build.txt

    # offline: the whole agent loop against the scripted provider (no key needed)
    .venv/bin/python scripts/run_agentic.py "a cottage" --provider mock

    # what commands exist?
    .venv/bin/python scripts/run_agentic.py --list-commands
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blockgen.agentic.agent import AgentConfig, BuildAgent, BuildResult
from blockgen.agentic.dsl import command_reference, run_program
from blockgen.agentic.providers import get_provider
from blockgen.agentic.report import write_run
from blockgen.utils.runs import new_run_dir


def _program_only(args) -> None:
    """Execute a program file with no LLM in the loop — the DSL's own smoke test."""
    text = Path(args.program).read_text()
    canvas, report, program = run_program(
        text, size=(args.canvas, args.canvas_height or args.canvas, args.canvas))
    print(report.summary(), flush=True)
    run = new_run_dir(args.name)
    result = BuildResult(description=args.prompt or Path(args.program).stem,
                         structure=canvas.to_structure(), canvas=canvas,
                         program_text=text, report=report)
    write_run(run, arms={"program": [result]},
              config={"program_file": str(args.program), "args": vars(args)},
              title=f"program: {Path(args.program).name}",
              max_dim=max(args.canvas, args.canvas_height or args.canvas),
              render=not args.no_render)
    print(f"[done] {run}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=None, help="what to build")
    ap.add_argument("--provider", default="openai:gpt-5-mini",
                    help="'<vendor>:<model>' — openai, gemini, anthropic, mock")
    ap.add_argument("--canvas", type=int, default=48, help="canvas x/z extent")
    ap.add_argument("--canvas-height", type=int, default=None, help="canvas y extent")
    ap.add_argument("--plan", action="store_true", help="natural-language plan first")
    ap.add_argument("--examples", type=int, default=0, dest="n_examples",
                    help="in-context example programs (0-4)")
    ap.add_argument("--repair-rounds", type=int, default=1,
                    help="rounds of executor-error feedback")
    ap.add_argument("--critique-rounds", type=int, default=0,
                    help="rounds of visual (rendered) self-critique")
    ap.add_argument("--critique-mode", default="rewrite", choices=["rewrite", "patch"])
    ap.add_argument("--ontology", default="none",
                    help="block-ontology variant in the system prompt: "
                         "none | mined | shuffled | stats (build it with "
                         "`python -m blockgen.ontology`)")
    ap.add_argument("--ontology-path", default=None, help="ontology JSON override")
    ap.add_argument("--image", action="append", default=[],
                    help="reference image (repeatable) — image-conditioned build")
    ap.add_argument("--target-blocks", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning-effort", default="medium", help="'none' to omit")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--n", type=int, default=1, help="samples for this prompt")
    ap.add_argument("--name", default="agentic_one", help="run-directory suffix")
    ap.add_argument("--program", default=None,
                    help="execute this program file instead of calling an LLM")
    ap.add_argument("--list-commands", action="store_true",
                    help="print the DSL reference (exactly what the model is told)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list_commands:
        print(command_reference())
        return
    if args.program:
        _program_only(args)
        return
    if not args.prompt:
        ap.error("give a prompt, or --program FILE, or --list-commands")
    if args.reasoning_effort in ("none", "None", ""):
        args.reasoning_effort = None

    cfg = AgentConfig(
        provider=args.provider,
        canvas_size=(args.canvas, args.canvas_height or args.canvas, args.canvas),
        plan=args.plan, n_examples=args.n_examples,
        repair_rounds=args.repair_rounds, critique_rounds=args.critique_rounds,
        critique_mode=args.critique_mode, target_blocks=args.target_blocks,
        max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
        temperature=args.temperature, ontology=args.ontology,
        ontology_path=args.ontology_path,
        cache=not args.no_cache, verbose=not args.quiet)

    images = [Path(p).read_bytes() for p in args.image]
    provider = get_provider(cfg.provider, cache=cfg.cache, **cfg.provider_params())
    agent = BuildAgent(provider, cfg)

    results = []
    for i in range(max(1, args.n)):
        if args.n > 1:
            print(f"--- sample {i + 1}/{args.n} ---", flush=True)
        # Pass the sample index as the variation seed: the response cache keys on the
        # request, so without it every sample of one prompt is the same build.
        results.append(agent.build(args.prompt, images=images,
                                   seed=i if args.n > 1 else None))

    run = new_run_dir(args.name)
    metrics = write_run(run, arms={"build": results},
                        config={"config": cfg.to_dict(), "args": vars(args)},
                        title=args.prompt,
                        max_dim=max(args.canvas, args.canvas_height or args.canvas),
                        render=not args.no_render)
    print(json.dumps(metrics["arms"]["build"], indent=1), flush=True)
    print(f"\n[program]\n{results[0].program_text}", flush=True)
    print(f"[done] {run}", flush=True)
    if all(r.blocks == 0 for r in results):
        sys.exit(1)  # nothing built: surface it to the shell / a sweep script


if __name__ == "__main__":
    main()
