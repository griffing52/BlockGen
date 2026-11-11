"""Supp. Fig. gate sweep: gate verdicts of the head-to-head rows under alternative thresholds.

Reads the recorded sweep (outputs/run_20261106_193109_gate_provenance/sweep.json,
part_a_ladder) and draws one row per candidate geometry metric and one column per
threshold setting: the four one-at-a-time sweeps, then three joint corners. A cell is
blue when the row passes every gate on the geometry rungs; otherwise it lists the
failing gates by number (1 = G1 ordering under decimation, 2 = G2 ordering under block
noise, 6 = G6 invariance, 7 = G7 self-consistency, 8 = G8 n-stability). The bottom row
counts the passing metrics. Drawn full width at print size, CPU only:

    .venv/bin/python scripts/figs/gate_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figs"))
import figstyle                                         # noqa: E402
from figstyle import BLUE, INK, INK_2, CVPR_FULL        # noqa: E402

SWEEP = ROOT / "outputs" / "run_20261106_193109_gate_provenance" / "sweep.json"
OUT = ROOT / "deliverables" / "paper" / "cvpr27" / "figures"

NAMES = {
    "vkid": "V-KID", "occ_kid": "occupancy KID",
    "uni3d_all_kid": "Uni3D KID, all", "uni3d_vol_kid": "Uni3D KID, vol",
    "ulip2_all_kid": "ULIP-2 KID, all", "ulip2_vol_kid": "ULIP-2 KID, vol",
    "cd_all_mmd": "chamfer MMD, all", "cd_all_one_minus_cov": "chamfer 1-cov, all",
    "cd_all_one_nna": "chamfer 1-NNA, all",
    "cd_vol_mmd": "chamfer MMD, voxel centres", "cd_vol_one_minus_cov": "chamfer 1-cov, voxel centres",
    "cd_vol_one_nna": "chamfer 1-NNA, voxel centres",
    "emd_all_mmd": "EMD MMD, all", "emd_all_one_minus_cov": "EMD 1-cov, all",
    "emd_all_one_nna": "EMD 1-NNA, all",
    "iou_d4_mmd": "IoU MMD, D4", "iou_d4_one_minus_cov": "IoU 1-cov, D4",
    "iou_d4_one_nna": "IoU 1-NNA, D4",
    "iou_corner_mmd": "IoU MMD, corner", "iou_corner_one_minus_cov": "IoU 1-cov, corner",
    "iou_corner_one_nna": "IoU 1-NNA, corner",
}
#: the thresholds of record (ladder.py): ordering tolerance 0, invariance 1 sd,
#: self-consistency 2 sd, n-stability 2 SE
USED = {"order_tol=0", "k6=1", "k7=2", "k8=2"}
GROUPS = [("ordering tolerance", "order_tol"), ("invariance bar", "k6"),
          ("self-consistency bar", "k7"), ("n-stability bar", "k8")]
CORNERS = [("S", "strict"), ("L", "lenient"), ("VL", "very_lenient")]
FAIL = "#ececec"


def main() -> None:
    figstyle.apply(size=7)
    d = json.loads(SWEEP.read_text())["part_a_ladder"]
    one, corners, solid = d["one_at_a_time"], d["corners"], d["solidify_z"]
    cols, heads = [], []
    for title, key in GROUPS:
        ks = [k for k in one if k.split("=")[0] == key]
        cols += [one[k] for k in ks]
        heads += [(title, k.split("=")[1] + ("*" if k in USED else "")) for k in ks]
    cols += [corners[k] for _, k in CORNERS]
    heads += [("joint", lab) for lab, _ in CORNERS]
    rows = sorted(NAMES, key=lambda m: -solid[m])

    fig, ax = plt.subplots(figsize=(CVPR_FULL, 3.5))
    nr, nc = len(rows), len(cols)
    for j, col in enumerate(cols):
        for i, m in enumerate(rows):
            f = col["fails"][m]["geometry"]
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.88, color=BLUE if not f else FAIL, lw=0))
            if f:
                ax.text(j + 0.47, i + 0.46, "".join(g[1:] for g in f), ha="center",
                        va="center", fontsize=6, color=INK_2)
        npass = sum(not col["fails"][m]["geometry"] for m in rows)
        ax.text(j + 0.47, nr + 0.5, str(npass), ha="center", va="center", fontsize=6.5,
                color=INK, fontweight="bold")
    ax.set_xlim(0, nc)
    ax.set_ylim(nr + 1.0, -2.3)
    ax.set_yticks([i + 0.45 for i in range(nr)] + [nr + 0.5])
    ax.set_yticklabels([f"{NAMES[m]} ({solid[m]:+.1f})" for m in rows] + ["metrics passing"],
                       fontsize=6.5)
    ax.set_xticks([])
    for j, h in enumerate(heads):
        ax.text(j + 0.47, -0.45, h[1], ha="center", va="center", fontsize=6.5, color=INK)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    start = 0
    for title in dict.fromkeys(h[0] for h in heads):
        w = sum(1 for h in heads if h[0] == title)
        ax.text(start + w / 2 - 0.03, -1.55, title, ha="center", va="center", fontsize=6.5,
                color=INK)
        ax.plot([start + 0.1, start + w - 0.16], [-1.15, -1.15], color=INK_2, lw=0.5)
        start += w
    fig.subplots_adjust(left=0.245, right=0.995, top=0.995, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"gate_sweep.{ext}", dpi=300)
    print(f"wrote {OUT}/gate_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
