"""Figures for `scripts/probe_llm_attention.py`.

Three panels, in the order the argument runs:
1. `head_auc.png`     -- every layer x head, base vs finetuned, against the recency floor.
2. `probe_summary.png` -- the headline comparison with bootstrap CIs over held-out builds.
3. `example_build.png` -- one build's true adjacency next to the winning head's attention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY, GREEN = "#3b6ea5", "#d1793a", "#8a8a8a", "#4f8a5b"


def _anchors_for(idx: int, n: int):
    """Re-derive one build's piece anchors -- the numbers the serialization prints."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from blockgen.curation.houses import load_house_structures
        from blockgen.export.minecraftace import load_piece_vocab
        from train_llm_pieces import PIECE_VOCAB, build_piece_names
        from probe_llm_attention import piece_instances
        structures, _ = load_house_structures(max_dim=32)
        cv = load_piece_vocab(PIECE_VOCAB)
        id_to_name, _ = build_piece_names(cv)
        inst = piece_instances(structures[idx], cv, id_to_name)
        return np.array([d["anchor"] for d in inst[:n]], dtype=np.int64)
    except Exception as e:  # noqa: BLE001 - the panel is optional
        print(f"[warn] anchors unavailable: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    S = json.loads((run / "summary.json").read_text())
    D = np.load(run / "head_auc.npz")
    setup = S["setup"]
    nl, nh = setup["n_layers"], setup["n_heads"]
    te = D["test_builds"]
    rec = float(np.nanmean(D["recency"][te]))

    # --- 1. per-head heatmaps ---------------------------------------------
    base = np.nanmean(D["base"][te], axis=0).reshape(nl, nh)
    ft = np.nanmean(D["finetuned"][te], axis=0).reshape(nl, nh)
    vmax = max(base.max(), ft.max(), rec)
    vmin = min(base.min(), ft.min())
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    for ax, M, name in zip(axes, (base, ft), ("base Qwen2.5-Coder-1.5B", "+ LoRA (Track D)")):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                       norm=matplotlib.colors.TwoSlopeNorm(
                           vcenter=rec, vmin=min(vmin, rec - 1e-3), vmax=max(vmax, rec + 1e-3)))
        ax.set_title(f"{name}\nbest head {M.max():.3f}", fontsize=11)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        b = np.unravel_index(M.argmax(), M.shape)
        ax.plot(b[1], b[0], marker="o", mfc="none", mec="k", ms=11, mew=1.6)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("adjacency AUC (mean over held-out builds)", labelpad=26)
    cb.ax.axhline(rec, color="k", lw=1.6)
    cb.ax.text(0.5, rec, f"recency {rec:.3f}", va="bottom", ha="center", fontsize=8,
               transform=cb.ax.get_yaxis_transform())
    surf = S.get("surface_baselines", {}).get("anchor_L1", {}).get("auc_on_test")
    tail = (f" — for scale, subtracting the printed coordinates scores {surf:.3f}"
            if surf else "")
    fig.suptitle("Can a single attention head rank the 6-adjacent previous piece first?\n"
                 f"{nl}x{nh} heads, {len(te)} held-out builds, "
                 f"{setup['positive_rate']*100:.1f}% of candidate pairs are true edges{tail}",
                 fontsize=12)
    fig.savefig(run / "head_auc.png", dpi=150)
    plt.close(fig)

    # --- 2. headline bars: raw AUC and the distance-stratified control ----
    sb = S.get("surface_baselines", {})
    rows = [
        ("recency  -(i-j)", S["recency"]["auc_on_test"], S["recency"]["strat_auc_on_test"], GREY),
        ("base: best attention head", S["base"]["auc_selected_on_test"],
         S["base"]["strat_auc_selected_on_test"], BLUE),
        ("finetuned: best attention head", S["finetuned"]["auc_selected_on_test"],
         S["finetuned"]["strat_auc_selected_on_test"], ORANGE),
    ]
    for k, v in sorted(S.get("hidden_probe", {}).items()):
        arm, layer = k.split(":")
        if layer != "L14":
            continue
        rows.append((f"{arm}: trained probe on hidden states ({layer})",
                     v["auc_on_test"], v["strat_auc_on_test"],
                     BLUE if arm == "base" else ORANGE))
    for k, lab in (("coord_match", "surface text: # of x/y/z tokens that match"),
                   ("anchor_L1", "surface text: L1 distance between the\nanchors the two lines literally print")):
        if k in sb:
            rows.append((lab, sb[k]["auc_on_test"], sb[k]["strat_auc_on_test"], GREEN))

    fig, ax = plt.subplots(figsize=(11, 0.85 * len(rows) + 2.6), constrained_layout=True)
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.36
    for yi, (lab, raw, strat, c) in zip(y, rows):
        ax.barh(yi + h / 2 + 0.02, raw - 0.5, left=0.5, color=c, alpha=0.9, height=h)
        ax.barh(yi - h / 2 - 0.02, strat - 0.5, left=0.5, color=c, alpha=0.42, height=h)
        ax.text(raw + 0.006, yi + h / 2 + 0.02, f"{raw:.3f}", va="center", fontsize=9)
        ax.text(strat + 0.006, yi - h / 2 - 0.02, f"{strat:.3f}", va="center", fontsize=9,
                color="#555555")
    ax.axvline(0.5, color="k", lw=1)
    ax.text(0.503, y[-1] - 0.55, "chance", fontsize=8.5, color="k")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0.48, 1.02)
    ax.set_xlabel("AUC — ranking the truly 6-adjacent previous piece above the non-adjacent ones")
    ax.set_title("Nothing to steal: the coordinates are already printed on the page\n"
                 f"solid = raw AUC · faded = distance-stratified (position removed) · "
                 f"{len(te)} held-out builds", fontsize=12)
    fig.savefig(run / "probe_summary.png", dpi=150)
    plt.close(fig)

    # --- 3. one build, side by side ---------------------------------------
    E = np.load(run / "example_build.npz")
    adj, A = E["adj"], E["A"]
    L = int(E["L"])
    best = int(np.nanmean(D["finetuned"][te], axis=0).argmax())
    n = min(L, 120)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(np.tril(adj[:n, :n], -1).astype(float), cmap="magma", interpolation="nearest")
    axes[0].set_title("ground truth: 6-adjacency\n(lower triangle = causal candidates)")
    m = A[best][:n, :n]
    axes[1].imshow(np.tril(m, -1), cmap="magma", interpolation="nearest",
                   vmax=np.percentile(m[np.tril_indices(n, -1)], 99))
    axes[1].set_title(f"best head (layer {best // 12}, head {best % 12})\nrow-normalized attention")
    rec_m = np.zeros((n, n))
    ii, jj = np.tril_indices(n, -1)
    rec_m[ii, jj] = 1.0 / (ii - jj)
    axes[2].imshow(rec_m, cmap="magma", interpolation="nearest",
                   vmax=np.percentile(rec_m[ii, jj], 99))
    axes[2].set_title("recency  1/(i-j)\nwhat position alone predicts")

    anc = _anchors_for(int(E["idx"]), n)
    if anc is not None:
        d = np.abs(anc[ii] - anc[jj]).sum(1).astype(float)
        surf = np.zeros((n, n))
        surf[ii, jj] = 1.0 / (1.0 + d)
        axes[3].imshow(surf, cmap="magma", interpolation="nearest",
                       vmax=np.percentile(surf[ii, jj], 99))
        axes[3].set_title("surface text  1/(1+L1 distance)\nsubtracting the printed coordinates")
    else:
        axes[3].axis("off")
    for ax in axes:
        ax.set_xlabel("piece j (earlier)"); ax.set_ylabel("piece i")
    fig.suptitle(f"Build h{int(E['idx']):05d}, first {n} pieces — the head tracks a recency "
                 "band; the printed coordinates reproduce the adjacency structure", fontsize=12)
    fig.savefig(run / "example_build.png", dpi=150)
    plt.close(fig)
    print(f"[figures] {run}/head_auc.png  probe_summary.png  example_build.png")


if __name__ == "__main__":
    main()
