"""Figures for the `train_llm_geobias.py` control-vs-bias pair.

1. `geobias_loss.png`  -- both arms' train/val curves on one axis.
2. `geobias_table.png` -- what the offset table actually learned: orthogonal slices
   through the (dx, dy, dz) cube, and the six 6-adjacent offsets against everything
   else. A bias that helps by learning *neighbourhood* should single those six out;
   one that helps by learning "look nearby-ish" will not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BLUE, ORANGE = "#3b6ea5", "#d1793a"
UNIT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("bias")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ctl, bia = Path(args.control), Path(args.bias)
    out = Path(args.out) if args.out else bia

    # --- 1. loss curves ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    finals = {}
    for run, name, c in ((ctl, "control (Track D)", BLUE), (bia, "+ geometric bias", ORANGE)):
        h = json.loads((run / "history.json").read_text())
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], color=c, ls="--", lw=1.3, alpha=0.75,
                label=f"{name} — train")
        ax.plot(ep, [r["val_loss"] for r in h], color=c, lw=2.2, label=f"{name} — val")
        best = min(h, key=lambda r: r["val_loss"])
        finals[name] = (best["val_loss"], best["epoch"], h[-1]["val_loss"])
        ax.plot(best["epoch"], best["val_loss"], marker="o", color=c, ms=7, mfc="white", mew=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("completion-only cross-entropy")
    ax.legend(fontsize=9)
    sub = "  ·  ".join(f"{k}: best val {v[0]:.4f} @ep{v[1]}" for k, v in finals.items())
    ax.set_title("Injecting geometry into attention — matched arms\n" + sub, fontsize=11)
    fig.savefig(out / "geobias_loss.png", dpi=150)
    plt.close(fig)

    # --- 2. the learned table ----------------------------------------------
    sd = torch.load(bia / "geo_bias.pt", map_location="cpu")
    n = round(sd["table"].numel() ** (1 / 3))
    r = (n - 1) // 2
    T = sd["table"].reshape(n, n, n).numpy()          # [dx, dy, dz], index = d + r

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    v = np.abs(T).max()
    slices = [("dy = 0", T[:, r, :], "dx", "dz"), ("dz = 0", T[:, :, r], "dx", "dy"),
              ("dx = 0", T[r, :, :], "dy", "dz")]
    for ax, (title, M, xl, yl) in zip(axes, slices):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, extent=[-r - .5, r + .5, r + .5, -r - .5])
        ax.set_title(f"learned bias, {title}", fontsize=10)
        ax.set_xlabel(yl); ax.set_ylabel(xl)
        ax.axhline(0, color="k", lw=.5, alpha=.3); ax.axvline(0, color="k", lw=.5, alpha=.3)
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="additive attention bias (logits)")

    unit = np.array([T[dx + r, dy + r, dz + r] for dx, dy, dz in UNIT])
    mask = np.ones_like(T, dtype=bool)
    for dx, dy, dz in UNIT:
        mask[dx + r, dy + r, dz + r] = False
    mask[r, r, r] = False
    other = T[mask]
    ax = axes[3]
    ax.bar(range(6), unit, color=ORANGE, alpha=.9)
    ax.axhline(float(other.mean()), color="k", ls="--", lw=1.4,
               label=f"all other offsets (mean {other.mean():+.3f})")
    ax.axhline(float(T[r, r, r]), color=BLUE, ls=":", lw=1.6,
               label=f"zero offset ({T[r, r, r]:+.3f})")
    ax.set_xticks(range(6))
    ax.set_xticklabels(["+x", "-x", "+y", "-y", "+z", "-z"])
    ax.set_ylabel("additive attention bias (logits)")
    ax.set_title(f"the six 6-adjacent offsets\nmean {unit.mean():+.3f} vs "
                 f"{other.mean():+.3f} elsewhere", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"What the shared offset table learned  ({n}^3 buckets, zero-init, "
                 f"specials: prompt {float(sd['prompt']):+.3f} · "
                 f"same-line {float(sd['same_line']):+.3f} · "
                 f"no-ref {float(sd['no_ref']):+.3f})", fontsize=12)
    fig.savefig(out / "geobias_table.png", dpi=150)
    plt.close(fig)
    print(f"[figures] {out}/geobias_loss.png  geobias_table.png")
    for k, v2 in finals.items():
        print(f"  {k}: best val {v2[0]:.4f} @ep{v2[1]}, final {v2[2]:.4f}")


if __name__ == "__main__":
    main()
