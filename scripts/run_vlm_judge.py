"""Run the VLM-judge arm (blockgen/eval/bench/vlm_judge.py) end to end.

    # smoke test: <= --max-calls API calls, writes a cost estimate for the full plan
    BLOCKGEN_ALLOW_SPEND=1 python scripts/run_vlm_judge.py --smoke

    # full pre-registered battery + system-level ranking (paid; see the estimate)
    BLOCKGEN_ALLOW_SPEND=1 python scripts/run_vlm_judge.py --max-calls 4000

    # re-analyse (free, cache/records only)
    python scripts/run_vlm_judge.py --run-dir outputs/run_..._vlm_judge --analyze-only

Order of operations, which is the point: build arms (sha-checked against the
leaderboard run of record) -> make the trial plan -> write prereg.json and
plan.json -> render stimuli + samples.png -> judge -> analyse. prereg.json is on
disk before the first API call; a resumed run refuses a plan whose sha differs
from the pre-registered one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blockgen.eval.bench import vlm_judge as vj  # noqa: E402
from blockgen.utils.runs import new_run_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge", default="openai:gpt-4o-2024-11-20",
                    help="provider spec (blockgen.agentic.providers.get_provider); e.g. "
                         "openai:gpt-5.5-2026-04-23, gemini:gemini-2.5-pro")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="ignored by reasoning models (the provider drops it)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 300, or 4000 for reasoning models")
    ap.add_argument("--reasoning-effort", default="low",
                    help="reasoning models only (gpt-5*, o*)")
    ap.add_argument("--run-dir", default=None, help="resume / re-analyse this run")
    ap.add_argument("--name", default="vlm_judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies condition n")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset by name")
    ap.add_argument("--system-k", type=int, default=8,
                    help="build pairs per arm pair in the system run (0 = skip)")
    ap.add_argument("--system-protocol", default="ext4")
    ap.add_argument("--system-pool", type=int, default=64,
                    help="system builds are drawn from the first N of each arm")
    ap.add_argument("--exclude-arms", nargs="*", default=list(vj.DEFAULT_EXCLUDE),
                    help="system arms left out (default: the LLM-FT arms, whose samples "
                         "are truncated sticks; user decision 2026-11-02)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=50,
                    help="refuse to start if more uncached calls than this are planned")
    ap.add_argument("--smoke", action="store_true",
                    help="1 pair per key condition + 2 system arm pairs (20 calls)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    return ap


def smoke_plan(seed: int):
    conds = {c.name: c for c in vj.default_conditions()}
    picks = [("interior_ext", 2), ("interior_cut", 1), ("catch", 1),
             ("gabled_patchwork", 1), ("interior_cutq", 1), ("hollow_cut", 1)]
    out = []
    for name, n in picks:
        c = conds[name]
        c.n_pairs = n
        out.append(c)
    return out, ("real_test", "uniform_random", "real@solidify")  # system arms


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(
        args.name + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as fh:
            fh.write(msg + "\n")

    if args.smoke:
        conditions, sys_arms = smoke_plan(args.seed)
        system_k = 1
    else:
        conditions = vj.default_conditions(args.scale)
        if args.conditions:
            conditions = [c for c in conditions if c.name in set(args.conditions)]
        sys_arms = tuple(a for a in vj.ALL_SYSTEM_ARMS if a not in set(args.exclude_arms))
        system_k = args.system_k

    log(f"[vlm] run dir {run_dir}")
    t0 = time.time()
    arms = vj.leaderboard_arms()
    log(f"[vlm] {len(arms)} arms rebuilt and sha-verified against "
        f"{vj.RUN_OF_RECORD.name} ({time.time() - t0:.0f}s)")
    # Licence rule: only 3D-Craft content reaches the API.
    arms, restriction = vj.restrict_to_3dcraft(arms, log=log)
    (run_dir / "restriction.json").write_text(json.dumps(restriction, indent=1))
    trials = vj.make_trials(conditions, {k: len(v) for k, v in arms.items()},
                            seed=args.seed, system_arms=sys_arms, system_k=system_k,
                            system_protocol=args.system_protocol,
                            system_pool=args.system_pool)
    sha = vj.plan_sha(trials)

    reasoning = any(m in args.judge for m in (":gpt-5", ":o1", ":o3", ":o4"))
    max_tokens = args.max_tokens or (4000 if reasoning else 300)
    params = {"max_tokens": max_tokens, "temperature": args.temperature}
    if args.judge.startswith("openai:"):
        params["reasoning_effort"] = args.reasoning_effort if reasoning else None

    pre_path = run_dir / "prereg.json"
    if pre_path.is_file():
        pre = json.loads(pre_path.read_text())
        if pre["plan_sha"] != sha:
            raise SystemExit(f"plan sha {sha} != pre-registered {pre['plan_sha']}; "
                             "refusing to run a different plan in this run dir")
        log(f"[vlm] pre-registration on file ({pre['written_at']}), plan {sha}")
    else:
        pre = vj.prereg(conditions, trials, args.judge, extra={
            "judge_params": params, "seed": args.seed, "smoke": args.smoke,
            "system_arms": list(sys_arms), "system_k": system_k,
            "excluded_arms": {a: "user decision: LLM-FT arms are truncated sticks, "
                                 "not representative" for a in args.exclude_arms},
            "system_protocol": args.system_protocol, "run_of_record": vj.RUN_OF_RECORD.name,
            "argv": sys.argv,
            "restriction": {k: ({kk: vv for kk, vv in v.items() if kk != "leaderboard_rows"}
                                if isinstance(v, dict) else v)
                            for k, v in restriction.items()},
            "door_diagnostic": "per real3dc house: cells solidify fills that are reachable "
                               "from outside through air or transparent blocks; "
                               "interior_* preference split by any (>0) vs none"})
        pre_path.write_text(json.dumps(pre, indent=1))
        (run_dir / "plan.json").write_text(json.dumps([t.to_dict() for t in trials]))
        log(f"[vlm] pre-registered {len(trials)} trials, plan {sha} -> {pre_path}")

    vis_path = run_dir / "visible_fill.json"
    if not vis_path.is_file():
        vis = [vj.visible_fill(s) for s in arms[vj.REAL3]]
        vis_path.write_text(json.dumps(vis))
    vis = json.loads(vis_path.read_text())
    log(f"[vlm] door diagnostic: {sum(v['visible_added'] > 0 for v in vis)}/{len(vis)} reachable, {sum(v['behind_window'] > 0 for v in vis)} behind a window; of "
        f"real3dc houses have exterior-visible filled cells")

    store = vj.StimulusStore(run_dir / "stimuli", arms)
    builds = [b for t in trials for b in (t.a, t.b)]
    t0 = time.time()
    made = store.prepare(builds, log=log)
    log(f"[vlm] {made} stimuli rendered ({time.time() - t0:.0f}s)")
    vj.samples_png(store, trials, run_dir / "samples.png")

    records_path = run_dir / "records.jsonl"
    if not args.analyze_only:
        from blockgen.agentic.providers import get_provider
        provider = get_provider(args.judge, cache_dir=vj.CACHE_DIR, spend_gate=True, **params)
        uncached = sum(not provider.peek(vj.judge_messages(t, store)) for t in trials)
        log(f"[vlm] {len(trials)} trials, {uncached} not in cache (cap {args.max_calls})")
        if uncached > args.max_calls:
            raise SystemExit(f"{uncached} uncached calls planned > --max-calls "
                             f"{args.max_calls}; raise it to approve the spend")
        if uncached and os.environ.get("BLOCKGEN_ALLOW_SPEND") != "1":
            raise SystemExit("set BLOCKGEN_ALLOW_SPEND=1 to make paid calls")
        vj.run_trials(provider, trials, store, records_path, workers=args.workers, log=log)

    recs = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()] \
        if records_path.is_file() else []
    keep = {t.id for t in trials}
    latest = {}
    for r in recs:
        if r["id"] in keep:
            latest[r["id"]] = r
    recs = list(latest.values())
    an = vj.analyze(recs, conditions, metrics=vj.leaderboard_metrics(),
                    n_boot=args.n_boot, seed=args.seed, system_arms=sys_arms)
    an["excluded_arms"] = list(args.exclude_arms)
    an["door_diagnostic"] = vj.door_diagnostic(recs, vis)
    an["restriction"] = {k: ({kk: vv for kk, vv in v.items() if kk != "leaderboard_rows"}
                             if isinstance(v, dict) else v) for k, v in restriction.items()}
    an["plan_sha"] = sha
    an["judge"] = args.judge
    an["judge_params"] = params
    an["estimate"] = estimate(recs, run_dir)
    (run_dir / "analysis.json").write_text(json.dumps(an, indent=1, default=str))
    (run_dir / "analysis.md").write_text(vj.report_md(an))
    log(vj.report_md(an))
    log(json.dumps(an["estimate"], indent=1))
    return 0


def estimate(recs, run_dir: Path):
    """Cost of the FULL default plan, from the measured tokens per call by protocol."""
    import numpy as np
    per = {}
    for r in recs:
        if r.get("prompt_tokens"):
            proto = r["a"].split("@")[-1]
            per.setdefault(proto, []).append((r["prompt_tokens"], r["completion_tokens"],
                                              r.get("model")))
    if not per:
        return None
    model = next(iter(per.values()))[0][2]
    price = vj.price_of(model)
    full = vj.make_trials(vj.default_conditions(),
                          {a: 128 for a in vj.SYSTEM_ARMS + ("real@hollow", vj.REAL3, vj.REAL3 + "@solidify", vj.REAL3 + "@hollow")},
                          system_arms=vj.SYSTEM_ARMS, system_k=8)
    rows, total_calls, total_usd = {}, 0, 0.0
    allp = [x for v in per.values() for x in v]
    for t in full:
        key = t.condition.split(":")[0]
        proto = t.a.protocol
        samp = per.get(proto) or per.get("ext4") or allp
        pin = float(np.mean([s[0] for s in samp]))
        pout = float(np.mean([s[1] for s in samp]))
        usd = None if price is None else (pin * price[0] + pout * price[1]) / 1e6
        row = rows.setdefault(key, {"calls": 0, "usd": 0.0, "protocol": proto})
        row["calls"] += 1
        row["usd"] += usd or 0.0
        total_calls += 1
        total_usd += usd or 0.0
    return {"model": model, "price_per_M": price,
            "tokens_per_call": {p: [float(np.mean([s[0] for s in v])),
                                    float(np.mean([s[1] for s in v]))] for p, v in per.items()},
            "per_condition": {k: {**v, "usd": round(v["usd"], 3)} for k, v in rows.items()},
            "total_calls": total_calls, "total_usd": round(total_usd, 2)}


if __name__ == "__main__":
    raise SystemExit(main())
