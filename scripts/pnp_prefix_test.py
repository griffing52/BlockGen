"""Prefix test: can the model read a real partial build, or is it blind?

This is T21's diagnostic, run against the pick-and-place model so the two are
directly comparable. Teacher-force the first K nodes of a *real* held-out build,
then let the model continue freely.

The logic. Under mere *drift*, a real prefix is a gift: it holds the model
on-distribution, so a longer prefix should produce a better continuation. Under
*blindness* the opposite happens — the model cannot interpret what it is being
shown, and hands it more real structure and it shuts down faster. T21 measured
exactly that: close-rate climbed 0.693 → 0.838 → 0.896 across prefix fractions
while the ground-truth continuation rate stayed flat at ~0.606, which ruled out
the obvious "late ops are naturally CLOSE-heavy" confound.

So the headline is `cont_ratio` = how far the model continues, as a fraction of
how far the real build actually goes from that same point. The ground-truth
column is 1.0 by construction; a blind model falls, and falls further as the
prefix grows.

**The confound this script must not fall into.** If every build in the pool is
truncated to the same `max_nodes`, then `true_remaining` is `cap - K` for all of
them, and a model that always emits a fixed number of nodes scores a flat, high
`cont_ratio` without reading anything at all. Measured on the first attempt:
every build hit the 192 cap, the model emitted ~188 every time, and `cont_ratio`
came out 0.979/0.978/0.972/0.976 — a clean-looking "not blind" that was pure
arithmetic. So the pool is restricted to builds shorter than the cap (real length
variance), and `length_corr` — the correlation between what the model continues
and what the build actually had left — is reported as the real signal. The script
refuses to issue a verdict when the pool lacks variance.

    .venv/bin/python scripts/pnp_prefix_test.py --run outputs/run_..._pnp_fixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.eval.bench import splits
from blockgen.eval.bench.topology import coherence
from blockgen.models.pick_n_place import (PickAndPlace, PickAndPlaceConfig, Rollout,
                                          generate, rollout_to_structure)
from blockgen.training.train_pick_n_place import PieceCodec
from blockgen.utils.growth_order import structure_to_growth


def load_run(run: Path, device: str):
    cfg_json = json.loads((run / "cmd.txt").read_text())
    codec = PieceCodec.from_json(json.loads((run / "palette.json").read_text()))
    cfg = PickAndPlaceConfig(n_pieces=codec.n_pieces, d_model=cfg_json["d_model"],
                             nhead=cfg_json["nhead"], num_layers=cfg_json["layers"],
                             rel_clamp=cfg_json["rel_clamp"],
                             max_nodes=cfg_json["max_nodes"])
    model = PickAndPlace(cfg)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device,
                                     weights_only=True))
    return model.to(device).eval(), codec, cfg_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-builds", type=int, default=24)
    ap.add_argument("--fractions", type=float, nargs="*",
                    default=[0.0, 0.10, 0.25, 0.50])
    ap.add_argument("--min-nodes", type=int, default=32)
    ap.add_argument("--untruncated-only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="only builds the cap did not truncate (needed for a "
                         "meaningful cont_ratio)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run = Path(args.run)
    model, codec, cfg_json = load_run(run, args.device)
    max_nodes = cfg_json["max_nodes"]

    split = splits.load_split(cfg_json["corpus"], cfg_json["seed"])
    pool = splits.split_structures(split, "test")
    seqs = []
    for s in pool:
        seq = structure_to_growth(s, ordering=cfg_json["ordering"],
                                  oriented=cfg_json["oriented"],
                                  max_nodes=max_nodes)
        if seq is None or seq.n_nodes < args.min_nodes:
            continue
        # Only builds the cap did NOT truncate. A truncated build has
        # true_remaining == cap - K for everyone, which makes cont_ratio
        # measure arithmetic instead of the model (see module docstring).
        if args.untruncated_only and seq.n_nodes >= max_nodes:
            continue
        if any(codec.encode(t) is None for t in seq.pieces.tolist()):
            continue
        seqs.append(seq)
        if len(seqs) >= args.n_builds:
            break
    if not seqs:
        raise SystemExit(
            "no usable held-out builds. With --untruncated-only the pool is "
            f"builds shorter than max_nodes={max_nodes}; train with a larger "
            "--max-nodes, or pass --no-untruncated-only and read length_corr "
            "rather than cont_ratio.")

    lengths = np.array([s.n_nodes for s in seqs], dtype=float)
    spread = float(lengths.std())
    if spread < 1.0:
        print(f"[warn] build lengths have std {spread:.2f} -- cont_ratio is "
              f"arithmetic here, not evidence. Read nothing into the verdict.")

    ref_blocks = float(np.median([s.n_nodes for s in seqs]))
    ref_coh = [coherence(rollout_to_structure(
        Rollout(np.array([codec.encode(t) for t in s.pieces.tolist()]),
                s.coords, s.parent, s.direction, True), codec.decode))
        for s in seqs]
    print(f"[pool] {len(seqs)} held-out builds, median {ref_blocks:.0f} nodes "
          f"(cap {max_nodes}, length std {spread:.1f}), real lcc "
          f"{np.mean([c.lcc_ratio for c in ref_coh]):.3f}")
    print()
    print(f"{'prefix':>7} {'K':>5} {'continued':>10} {'true rem':>9} "
          f"{'cont_ratio':>11} {'len_corr':>11} {'STOP':>6} {'blocks':>7}")
    print("-" * 72)

    rows = []
    for frac in args.fractions:
        cont, true_rem, stops, blocks = [], [], [], []
        for seq in seqs:
            k = int(round(frac * seq.n_nodes))
            pref = None
            if k >= 1:
                pref = Rollout(
                    pieces=np.array([codec.encode(int(t)) for t in seq.pieces[:k]]),
                    coords=seq.coords[:k].astype(np.int64),
                    parent=seq.parent[:k], direction=seq.direction[:k], stopped=False)
            r = generate(model, max_nodes=max_nodes, temperature=args.temperature,
                         top_k=args.top_k, device=args.device, prefix=pref)
            produced = len(r.pieces) - k
            cont.append(max(produced, 0))
            true_rem.append(seq.n_nodes - k)
            stops.append(float(r.stopped))
            blocks.append(len(r.pieces))
        ratio = float(np.mean(cont)) / max(float(np.mean(true_rem)), 1e-9)
        # Does the model continue FURTHER for builds that genuinely had further
        # to go? This is the part a fixed-output model cannot fake.
        corr = (float(np.corrcoef(cont, true_rem)[0, 1])
                if np.std(cont) > 1e-9 and np.std(true_rem) > 1e-9 else float("nan"))
        rows.append({"prefix": frac, "k_median": float(np.median(
            [int(round(frac * s.n_nodes)) for s in seqs])),
            "continued": float(np.mean(cont)), "true_remaining": float(np.mean(true_rem)),
            "cont_ratio": ratio, "length_corr": corr,
            "stop_rate": float(np.mean(stops)), "blocks": float(np.mean(blocks))})
        print(f"{frac:>7.2f} {rows[-1]['k_median']:>5.0f} "
              f"{rows[-1]['continued']:>10.1f} {rows[-1]['true_remaining']:>9.1f} "
              f"{ratio:>11.3f} {corr:>11.3f} {rows[-1]['stop_rate']:>6.2f} "
              f"{rows[-1]['blocks']:>7.1f}")

    ratios = [r["cont_ratio"] for r in rows]
    corrs = [r["length_corr"] for r in rows if np.isfinite(r["length_corr"])]
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")

    # Judge distance from correct, |ratio - 1|, not the raw direction of change.
    # A model that free-runs too LONG gets closer to correct by continuing less,
    # and a rule that reads any decline as blindness calls that a failure:
    # measured 1.213 -> 0.897, which is overshoot-by-21% improving to
    # undershoot-by-10%, i.e. the prefix pulling it toward the right length.
    # T21's blindness was shutting down *below* target, so it is the error that
    # has to grow, not the ratio that has to fall.
    err = [abs(r - 1.0) for r in ratios]
    if spread < 1.0:
        verdict = "INCONCLUSIVE -- no length variance in the pool"
    elif len(err) > 1 and err[-1] > err[0] + 0.05:
        verdict = (f"BLIND (length error grows with more real prefix: "
                   f"{err[0]:.2f} -> {err[-1]:.2f})")
    elif len(err) > 1 and err[-1] < err[0] - 0.05:
        verdict = (f"USES THE PREFIX (length error shrinks: "
                   f"{err[0]:.2f} -> {err[-1]:.2f})")
    elif np.isfinite(mean_corr) and mean_corr > 0.3:
        verdict = (f"READS THE PREFIX (continuation tracks true remaining length, "
                   f"r={mean_corr:.2f})")
    else:
        verdict = (f"NOT BLIND BUT NOT READING LENGTH (flat continuation, "
                   f"r={mean_corr:.2f}) -- it emits a roughly fixed budget")
    print()
    print(f"cont_ratio  across prefixes: {[round(x, 3) for x in ratios]}")
    print(f"length_corr across prefixes: {[round(x, 3) for x in corrs]}")
    print(f"VERDICT: {verdict}")

    out = run / "prefix_test.json"
    out.write_text(json.dumps({"rows": rows, "verdict": verdict,
                               "n_builds": len(seqs), "length_std": spread,
                               "mean_length_corr": mean_corr,
                               "median_true_nodes": ref_blocks}, indent=2) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
