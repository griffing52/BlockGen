"""Rank intervals and pairwise flip probabilities for a bench run's BlockScore board.

The scorecard stores each metric's interval but not its replicates, so a rank
interval cannot be read off it. This script recomputes the replicates from the
per-build features a run saves with `--save-features` (`features.npz`), using
the scheme the scorecard's own intervals use (`stats.subsample_ci`: m-out-of-n,
m = 0.8 n, without replacement), with one reference subsample per replicate
shared by every arm, so arms are compared against the same reference draw.

Per replicate, the appearance (`mv_dino_kid`) and geometry (`geom_kid`) pillars
are recomputed and put in the run's own units, `(value - real_test) / noise_sd`
with the floor and the measured noise sd taken from the card. The palette,
coherence and geometry-scalar pillars are held at their point values: on every
card to date none of them is within a factor of three of the worst pillar of a
ranked arm, and the script checks that per arm (`small_pillar_margin`).

    python scripts/bench_rank_intervals.py outputs/run_<stamp>_bench_<slug>

Writes `rank_intervals.json` and `rank_intervals.md` into the run directory.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from blockgen.eval.bench import distances as D

#: Pillars recomputed per replicate: pillar -> (feature key prefix, metric path).
RESAMPLED = {"appearance": ("dino", "realism.mv_dino_kid"),
             "geometry": ("geom", "realism.geom_kid")}
#: A pair is a tie when the order flips in at least this share of replicates.
TIE_AT = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    run = Path(args.run_dir)
    card = json.loads((run / "scorecard.json").read_text())
    feats = np.load(run / "features.npz")
    sd = card["context"]["noise_floor_sd"]
    rows = [r for r in card["run"]["blockscore"] if r["status"] == "ok"]
    arms = [r["arm"] for r in rows]
    point = {r["arm"]: r["score"] for r in rows}
    floor = {p: card["arms"]["real_test"]["realism"][m.split(".")[1]]["value"]
             for p, (_, m) in RESAMPLED.items()}

    margin = {}
    for r in rows:
        big = max(r["pillars"].get(p, -np.inf) for p in RESAMPLED)
        small = max([v for p, v in r["pillars"].items() if p not in RESAMPLED] or [0.0])
        margin[r["arm"]] = float(big / small) if small > 0 else float("inf")

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    scores = np.zeros((B, len(arms)))
    refs = {k: feats[f"ref::{k}"] for k, _ in RESAMPLED.values()}
    for b in range(B):
        ref_idx = {k: rng.choice(len(v), int(round(args.frac * len(v))), replace=False)
                   for k, v in refs.items()}
        for j, r in enumerate(rows):
            pil = dict(r["pillars"])
            for p, (k, m) in RESAMPLED.items():
                if p not in pil:
                    continue
                x = feats[f"{k}::{r['arm']}"]
                idx = rng.choice(len(x), int(round(args.frac * len(x))), replace=False)
                v = D.kid(x[idx], refs[k][ref_idx[k]])
                pil[p] = (v - floor[p]) / sd[m]
            scores[b, j] = max(pil.values())

    # Ranks per replicate (1 = best), among the ranked arms of the card.
    ranks = scores.argsort(axis=1).argsort(axis=1) + 1
    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    out_rows = []
    for j, a in enumerate(arms):
        out_rows.append({
            "arm": a, "rank": j + 1, "blockscore": point[a],
            "blockscore_ci": [float(np.percentile(scores[:, j], lo_q)),
                              float(np.percentile(scores[:, j], hi_q))],
            "rank_ci": [int(np.percentile(ranks[:, j], lo_q, method="lower")),
                        int(np.percentile(ranks[:, j], hi_q, method="higher"))],
            "small_pillar_margin": margin[a],
        })
    pairs = []
    for i, j in combinations(range(len(arms)), 2):
        # point order is i before j (rows are sorted best first)
        flip = float((scores[:, j] <= scores[:, i]).mean())
        pairs.append({"a": arms[i], "b": arms[j], "p_flip": flip,
                      "tie": flip >= TIE_AT})

    res = {"run": run.name, "n_boot": B, "frac": args.frac, "seed": args.seed,
           "alpha": args.alpha, "tie_at": TIE_AT,
           "method": "m-out-of-n subsample of each arm and of the reference "
                     "(shared across arms per replicate); appearance and geometry "
                     "pillars recomputed, other pillars held at point value",
           "rows": out_rows, "pairs": pairs}
    (run / "rank_intervals.json").write_text(json.dumps(res, indent=1))

    md = [f"## rank intervals — BlockScore, {B} replicates, m = {args.frac:g} n",
          "", "| # | arm | BlockScore | 95% interval | rank 95% interval | tied with |",
          "|---|---|---|---|---|---|"]
    for r in out_rows:
        tied = [p["b"] if p["a"] == r["arm"] else p["a"] for p in pairs
                if p["tie"] and r["arm"] in (p["a"], p["b"])]
        lo, hi = r["blockscore_ci"]
        md.append(f"| {r['rank']} | {r['arm']} | {r['blockscore']:.2f} | "
                  f"[{lo:.2f}, {hi:.2f}] | {r['rank_ci'][0]}–{r['rank_ci'][1]} | "
                  f"{', '.join(tied) or '—'} |")
    md += ["", "## adjacent-pair flip probabilities", "",
           "| a | b | P(order flips) | tie |", "|---|---|---|---|"]
    for i in range(len(arms) - 1):
        p = next(q for q in pairs if q["a"] == arms[i] and q["b"] == arms[i + 1])
        md.append(f"| {p['a']} | {p['b']} | {p['p_flip']:.3f} | "
                  f"{'**tie**' if p['tie'] else ''} |")
    md += ["", f"A pair is marked a tie when its order flips in at least "
               f"{TIE_AT:.0%} of replicates. Full pair table in rank_intervals.json.", ""]
    (run / "rank_intervals.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
