"""Render a compact cross-arm comparison from one or more bench scorecards.

`scorecard.md` prints everything. This prints the columns you would actually put
in a paper table, in a fixed pillar order, with the real-data floor pinned to the
top row so every other number is read against it.

    python scripts/bench_report.py --run outputs/run_<stamp>_bench
    python scripts/bench_report.py --run outputs/run_*_bench --out deliverables/table.md

Columns marked `?` are metrics the validation ladder rejected for the backbone and
view the run used; they are absent by design, not missing by accident.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# (section, key, header, decimals, higher_is_better)
COLUMNS = [
    ("realism", "mv_dino_kid", "KID↓", 3, False),
    ("fidelity", "recall", "recall↑", 3, True),
    ("novelty", "dino_nn_percentile", "novelty↑", 3, True),
    ("novelty", "voxel_dup_rate", "dup↓", 3, False),
    ("novelty", "voxel_diversity", "diversity↑", 3, True),
    ("dataset_stats", "palette_jsd_family", "palette JSD↓", 3, False),
]

# (key, header, decimals) — always shown with the real value beside them.
COHERENCE = [
    ("lcc_ratio", "connected", 3),
    ("enclosed_air_ratio", "interior", 3),
    ("floating_block_frac", "floating", 3),
    ("n_blocks", "blocks", 0),
]


def load_runs(patterns: List[str]) -> List[dict]:
    cards = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            f = Path(path)
            f = f / "scorecard.json" if f.is_dir() else f
            if f.exists():
                cards.append(json.loads(f.read_text()))
    if not cards:
        raise SystemExit(f"no scorecard.json found under {patterns}")
    return cards


def cell(metric: Optional[dict], nd: int, ci: bool = True) -> str:
    if not isinstance(metric, dict):
        return "—"
    if metric.get("value") is None:
        return "?"
    v = metric["value"]
    lo, hi = (metric.get("ci") or [None, None])[:2]
    if ci and lo is not None and hi is not None:
        return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{v:.{nd}f}"


def coherence_cell(entry: Optional[dict], nd: int) -> str:
    if not isinstance(entry, dict):
        return "—"
    g = entry["gen"]["mean"]
    r = entry["real"]["mean"]
    return f"{g:.{nd}f} ({r:.{nd}f})"


def order_arms(arms: Dict[str, Any]) -> List[str]:
    """Real floor first, then models, then the remaining controls."""
    def rank(name: str) -> tuple:
        track = (arms[name].get("meta") or {}).get("track", "")
        if name == "real_test":
            return (0, name)
        if track == "control":
            return (2, name)
        return (1, name)
    return sorted(arms, key=rank)


def render(card: dict, show_ci: bool = True) -> str:
    arms = card["arms"]
    names = order_arms(arms)
    ctx = card.get("context", {})
    lines: List[str] = []

    passing = (card.get("ladder") or {}).get("passing")
    lines.append(f"**corpus** `{ctx.get('corpus')}` · **split** `{ctx.get('split_key')}` "
                 f"· **reference** {ctx.get('n_ref_used')} held-out real builds")
    if passing:
        lines.append(f"**ladder-validated metrics**: {', '.join(passing)}")
    lines.append("")

    head = ["arm", "n"] + [c[2] for c in COLUMNS]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for name in names:
        arm = arms[name]
        row = [f"`{name}`", str((arm.get("meta") or {}).get("n", "?"))]
        for section, key, _, nd, _ in COLUMNS:
            row.append(cell((arm.get(section) or {}).get(key), nd, show_ci))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("Coherence — generated (real in parentheses); closer to the real "
                 "value is better, in **either** direction.")
    lines.append("")
    head2 = ["arm"] + [c[1] for c in COHERENCE]
    lines.append("| " + " | ".join(head2) + " |")
    lines.append("|" + "---|" * len(head2))
    for name in names:
        block = arms[name].get("coherence") or {}
        row = [f"`{name}`"] + [coherence_cell(block.get(k), nd)
                               for k, _, nd in COHERENCE]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    costed = [n for n in names if arms[n].get("cost")]
    if costed:
        lines.append("| arm | $/coherent build | tokens/coherent build |")
        lines.append("|---|---|---|")
        for n in costed:
            c = arms[n]["cost"]
            lines.append(f"| `{n}` | {cell(c.get('usd_per_coherent_build'), 4, False)} "
                         f"| {cell(c.get('tokens_per_coherent_build'), 0, False)} |")
        lines.append("")

    if card.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in card["warnings"])
        lines.append("")
    lines.append("`?` = rejected by the validation ladder for this backbone/view.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ci", action="store_true")
    args = ap.parse_args()

    text = "\n\n---\n\n".join(render(c, show_ci=not args.no_ci)
                              for c in load_runs(args.run))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
