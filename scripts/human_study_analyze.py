"""Turn 2AFC judgements into the number a metric paper has to report.

Three steps, in this order, and the order matters because the first one must not
be able to see the third.

**1. Exclude inattentive raters.** Anyone who does not reliably pick the real
house over uniform noise is not doing the task. The rule is fixed here in
advance -- fewer than `--catch-floor` of the catch trials correct and the whole
session is dropped -- so it cannot be tuned after seeing which exclusion helps a
metric look better.

**2. Fit an arm-level human quality score.** Raters compare individual builds,
but the metrics score distributions, so the comparisons are pooled to the arm
pair and a Bradley-Terry model turns the pairwise win rates into one latent
score per arm. Bradley-Terry rather than raw win rate because win rate depends
on which opponents an arm happened to be drawn against; the latent score does
not, which matters when coverage is uneven -- and with a random subset shown per
session, coverage is always uneven.

**3. Score the metrics against it.** Two numbers per metric:

* **Pairwise agreement** -- of the arm pairs where humans have a clear
  preference, how often does the metric order them the same way. Directly
  interpretable, and the quantity a leaderboard's credibility rests on.
* **Spearman rho** against the Bradley-Terry scores -- uses the whole ordering
  rather than thresholded pairs, and is what the generative-metric literature
  reports.

Pairs where humans are near chance are reported but excluded from agreement:
counting them would score a metric on coin flips, and with enough of them any
metric converges to 50%.

    python scripts/human_study_analyze.py --key <run>/answer_key.json \\
        --responses responses.json --scorecard <run>/scorecard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Study arm name -> scorecard arm name. The study calls the held-out real
#: builds `real` because that is what a rater is being asked about; the
#: scorecard calls the same set `real_test`. Without the alias the two tables
#: silently fail to join and every metric reports "arm names do not overlap".
ALIASES = {"real": "real_test"}

#: Metrics read from the scorecard, with the direction that means "better".
METRICS = {
    "geom_kid": ("realism", "lower_better"),
    "mv_dino_kid": ("realism", "lower_better"),
    "palette_jsd_exact": ("dataset_stats", "lower_better"),
    "voxel_nn_iou_mean": ("novelty", "lower_better"),
}


def wins_from_lab(db_path: str) -> Dict[Tuple[str, str], int]:
    """Pairwise wins from a BlockLab comparison log, keyed by *dataset*.

    The batch study and the lab tool collect the same judgement through two
    different doors. The study serves a fixed trial list and records a side
    (`left`/`right`), so recovering which arm won needs the answer key. The lab
    records the winning `build_id` directly, and a build id carries its dataset,
    so the arm is already in the datum -- no key, no join, no chance of the two
    drifting apart.

    That makes the lab log the better instrument for casual, continuous
    comparison, and this function exists so those judgements land in the same
    Bradley-Terry fit rather than in a second, incompatible analysis. `winner`
    of `None` is "can't tell": recorded by the tool, and dropped here, because
    Bradley-Terry has no representation for a draw and silently scoring it as
    half a win for each side would invent information the rater declined to give.
    """
    from collections import defaultdict

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.lab.store import Store

    def dataset_of(build_id: str) -> str:
        return build_id.rsplit(":", 1)[0]

    wins: Dict[Tuple[str, str], int] = defaultdict(int)
    store = Store(db_path)
    try:
        for row in store.compares():
            if not row.get("winner"):
                continue
            a, b, w = row["a"], row["b"], row["winner"]
            loser = b if w == a else a
            wins[(dataset_of(w), dataset_of(loser))] += 1
    finally:
        store.close()
    return dict(wins)


def load_responses(paths: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for p in paths:
        raw = json.loads(Path(p).read_text())
        out.extend(raw if isinstance(raw, list) else [raw])
    return out


def bradley_terry(wins: Dict[Tuple[str, str], int], arms: Sequence[str],
                  iters: int = 500, prior: float = 0.5) -> Dict[str, float]:
    """Minorization-maximization fit of Bradley-Terry strengths (log scale).

    `prior` adds a half-win each way to every observed pair, which keeps an arm
    that lost every one of its comparisons from taking an infinite negative
    score -- `uniform_random` does exactly that against most opponents, and
    without the prior the whole ordering becomes unreportable.
    """
    idx = {a: i for i, a in enumerate(arms)}
    n = len(arms)
    w = np.zeros((n, n))
    for (a, b), k in wins.items():
        if a in idx and b in idx:
            w[idx[a], idx[b]] += k
    seen = (w + w.T) > 0
    w = w + prior * seen                        # symmetric smoothing
    total = w + w.T
    wins_i = w.sum(1)

    p = np.ones(n)
    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if total[i, j] > 0:
                    denom[i] += total[i, j] / (p[i] + p[j])
        new = np.where(denom > 0, wins_i / np.maximum(denom, 1e-12), p)
        new = np.where(new > 0, new, 1e-12)
        new /= np.exp(np.mean(np.log(new)))     # fix the scale (geometric mean 1)
        if np.max(np.abs(np.log(new) - np.log(p))) < 1e-10:
            p = new
            break
        p = new
    return {a: float(np.log(p[idx[a]])) for a in arms}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson interval -- correct at the extremes, where a Wald interval is not.

    Several arm pairs land at or near 0/1 preference (nobody prefers uniform
    noise to a real house), and a Wald interval there is either zero-width or
    runs outside [0, 1).
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report_from_wins(wins: Dict[Tuple[str, str], int], args, ap) -> int:
    """Bradley-Terry ranking from a win table alone, for a lab-only log."""
    if not wins:
        ap.error("no decided comparisons in that lab database")
    arms = sorted({a for pair in wins for a in pair})
    bt = bradley_terry(wins, arms)
    order = sorted(arms, key=lambda a: -bt[a])
    print()
    print("| rank | dataset | Bradley-Terry | comparisons |")
    print("|---|---|---|---|")
    for i, a in enumerate(order, 1):
        n = sum(v for (x, y), v in wins.items() if a in (x, y))
        print(f"| {i} | `{a}` | {bt[a]:+.3f} | {n} |")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"source": "lab", "bradley_terry": bt, "order": order,
             "wins": {f"{a}|{b}": k for (a, b), k in wins.items()}}, indent=2) + "\n")
        print(f"\n-> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", help="answer_key.json from the export")
    ap.add_argument("--responses", nargs="*", default=[])
    ap.add_argument("--lab-db", help="BlockLab lab.db; use its comparison log "
                                     "instead of (or as well as) a study export")
    ap.add_argument("--scorecard", help="scorecard.json to score metrics against")
    ap.add_argument("--catch-floor", type=float, default=0.75,
                    help="minimum fraction of catch trials correct to keep a session")
    ap.add_argument("--clear-margin", type=float, default=0.10,
                    help="|p - 0.5| above which a pair counts as humanly decided")
    ap.add_argument("--min-trials", type=int, default=4,
                    help="minimum judgements before a pair is used")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.key and not args.lab_db:
        ap.error("pass --key/--responses (a study export) or --lab-db (a lab log)")

    lab_wins = wins_from_lab(args.lab_db) if args.lab_db else {}
    if lab_wins:
        print(f"[lab] {sum(lab_wins.values())} decided comparisons over "
              f"{len(lab_wins)} ordered dataset pairs")

    if not args.key:
        # Lab-only: no trial list, so no catch trials and no exclusion step.
        # Said out loud because the exclusion is the study's guard against
        # inattentive raters, and a lab log collected by the researcher
        # themselves has no equivalent -- that is a real difference in evidence
        # quality, not a detail.
        print("[lab] no answer key: skipping catch-trial exclusion "
              "(a lab log is self-collected and has no catch trials)")
        return report_from_wins(lab_wins, args, ap)

    key = json.loads(Path(args.key).read_text())
    trials = {t["id"]: t for t in key["trials"]}
    catch_pair = set(key["meta"]["catch_pair"])
    sessions = load_responses(args.responses)
    print(f"[study] {len(sessions)} sessions, "
          f"{sum(len(s.get('responses', [])) for s in sessions)} judgements")

    # --- 1. exclusion, before anything else is computed --------------------
    kept, dropped = [], []
    for s in sessions:
        hit = tot = 0
        for r in s.get("responses", []):
            t = trials.get(r["trial"])
            if not t or not t.get("catch"):
                continue
            tot += 1
            chosen = t["left_arm"] if r["chose"] == "left" else t["right_arm"]
            hit += (chosen == "real")
        rate = hit / tot if tot else float("nan")
        (kept if (tot == 0 or rate >= args.catch_floor) else dropped).append(
            (s, rate, tot))
    print(f"[study] kept {len(kept)} sessions, dropped {len(dropped)} on catch trials"
          + (f" (rates {[round(r,2) for _, r, _ in dropped]})" if dropped else ""))

    # --- 2. pool to arm pairs ----------------------------------------------
    wins: Dict[Tuple[str, str], int] = defaultdict(int)
    for s, _, _ in kept:
        for r in s.get("responses", []):
            t = trials.get(r["trial"])
            if not t:
                continue
            a = t["left_arm"] if r["chose"] == "left" else t["right_arm"]
            b = t["right_arm"] if r["chose"] == "left" else t["left_arm"]
            wins[(a, b)] += 1

    for pair, k in lab_wins.items():           # fold in the lab log, if given
        wins[pair] += k

    arms = sorted({a for pair in wins for a in pair})
    bt = bradley_terry(wins, arms)
    order = sorted(arms, key=lambda a: -bt[a])

    print()
    print("| rank | arm | Bradley-Terry | judgements |")
    print("|---|---|---|---|")
    for i, a in enumerate(order, 1):
        n = sum(v for (x, y), v in wins.items() if a in (x, y))
        print(f"| {i} | `{a}` | {bt[a]:+.3f} | {n} |")

    # --- 3. metric agreement ------------------------------------------------
    report = {"bradley_terry": bt, "order": order,
              "sessions_kept": len(kept), "sessions_dropped": len(dropped),
              "pairs": {}, "metrics": {}}

    pairs: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            k = wins.get((a, b), 0)
            n = k + wins.get((b, a), 0)
            if n >= args.min_trials:
                pairs[(a, b)] = (k, n)
                lo, hi = wilson(k, n)
                report["pairs"][f"{a}|{b}"] = {"prefer_a": k, "n": n,
                                               "p": k / n, "ci": [lo, hi]}

    if args.scorecard:
        card = json.loads(Path(args.scorecard).read_text())
        print()
        print("| metric | pairwise agreement | n pairs | Spearman vs BT |")
        print("|---|---|---|---|")
        from scipy.stats import spearmanr
        for m, (block, direction) in METRICS.items():
            raw = {}
            for arm, blocks in card.get("arms", {}).items():
                v = (blocks.get(block) or {}).get(m)
                if isinstance(v, dict) and v.get("value") is not None:
                    raw[arm] = float(v["value"])
            score = {a: raw[ALIASES.get(a, a)] for a in order
                     if ALIASES.get(a, a) in raw}
            common = [a for a in order if a in score]
            if len(common) < 3:
                print(f"| `{m}` | — (arm names do not overlap the scorecard) | | |")
                continue
            hit = tot = 0
            for (a, b), (k, n) in pairs.items():
                if a not in score or b not in score:
                    continue
                p = k / n
                if abs(p - 0.5) < args.clear_margin:
                    continue                    # humans undecided; not a test
                tot += 1
                human_a = p > 0.5
                metric_a = (score[a] < score[b]) if direction == "lower_better" \
                    else (score[a] > score[b])
                hit += (human_a == metric_a)
            rho = spearmanr([bt[a] for a in common],
                            [-score[a] if direction == "lower_better" else score[a]
                             for a in common]).statistic
            acc = hit / tot if tot else float("nan")
            lo, hi = wilson(hit, tot)
            report["metrics"][m] = {"agreement": acc, "n_pairs": tot,
                                    "ci": [lo, hi], "spearman_vs_bt": float(rho),
                                    "n_arms": len(common)}
            print(f"| `{m}` | {acc:.3f} [{lo:.2f}, {hi:.2f}] | {tot} | {rho:+.3f} |")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
