"""Matplotlib figures for attention-as-adjacency (all saved as PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

SEQ = "Blues"                 # magnitude: one hue, light -> dark
DIV = "RdBu_r"                # AUROC around chance 0.5: two hues, neutral midpoint
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
               "#e34948"]  # fixed order, never cycled past 8
INK, MUTED = "#0b0b0b", "#8a8984"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)


def _save(fig, path) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def token_heatmap(att: np.ndarray, spans: Sequence[Tuple[int, int]], path, title: str = "",
                  max_lines: int = 20) -> Path:
    """One head's token-level attention ``[T, T]`` with block-line boundaries drawn in."""
    spans = list(spans)[:max_lines]
    start, end = spans[0][0], spans[-1][1]
    sub = att[start:end, start:end]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    im = ax.imshow(sub ** 0.5, cmap=SEQ, interpolation="nearest")
    for s, _ in spans[1:]:
        ax.axhline(s - start - 0.5, color=MUTED, lw=0.4)
        ax.axvline(s - start - 0.5, color=MUTED, lw=0.4)
    ax.set_xlabel("key token (lines delimited)")
    ax.set_ylabel("query token")
    ax.set_title(title or "token attention (sqrt scale)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="sqrt(attention)")
    return _save(fig, path)


def head_grid(layer_att: np.ndarray, path, title: str = "", max_nodes: int = 64) -> Path:
    """All heads of one layer, node-level ``[H, N, N]`` (each panel on its own scale,
    clipped at its 99th percentile; cropped to the first ``max_nodes`` lines)."""
    layer_att = layer_att[:, :max_nodes, :max_nodes]
    H = layer_att.shape[0]
    cols = int(np.ceil(np.sqrt(H)))
    rows = int(np.ceil(H / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 1.9 * rows))
    for h, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if h < H:
            ax.imshow(layer_att[h], cmap=SEQ, interpolation="nearest",
                      vmax=float(np.percentile(layer_att[h], 99)) or None)
            ax.set_title(f"head {h}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def versus_truth(panels: Sequence[Tuple[str, np.ndarray, np.ndarray]], path,
                 title: str = "", max_nodes: int = 64) -> Path:
    """Rows of ``(label, score [N,N], adjacency [N,N])``: score left, ground truth right.

    Long sequences are cropped to their first ``max_nodes`` lines (a raster prefix is a
    valid partial build); colour is clipped at the 99th percentile of positive scores.
    """
    fig, axes = plt.subplots(len(panels), 2, figsize=(7, 3.4 * len(panels)), squeeze=False)
    for (label, score, adj), (a0, a1) in zip(panels, axes):
        score, adj = score[:max_nodes, :max_nodes], adj[:max_nodes, :max_nodes]
        pos = score[score > 0]
        a0.imshow(score, cmap=SEQ, interpolation="nearest",
                  vmax=float(np.percentile(pos, 99)) if pos.size else None)
        a0.set_title(f"{label}: attention score", fontsize=8)
        a1.imshow(adj, cmap="Greys", interpolation="nearest")
        edges = int(np.count_nonzero(np.tril(adj, -1) | np.triu(adj, 1).T))
        a1.set_title(f"{label}: true adjacency ({edges} edges)", fontsize=8)
        for ax in (a0, a1):
            ax.set_xlabel("block line", fontsize=7)
            ax.set_ylabel("block line", fontsize=7)
            _style(ax)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def auroc_heatmaps(grids: Mapping[str, np.ndarray], notes: Mapping[str, str], path,
                   title: str = "", label: str = "AUROC (0.5 = chance)") -> Path:
    """Layer x head AUROC panels on one diverging scale centred at chance (0.5)."""
    dev = max(float(np.nanmax(np.abs(g - 0.5))) for g in grids.values())
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.5 - dev, vmax=0.5 + dev)
    n = len(grids)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols + 1, 4.6 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.45)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (name, g) in zip(axes.ravel(), grids.items()):
        im = ax.imshow(g, cmap=DIV, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_title(f"{name}\nbest {np.nanmax(g):.3f}; {notes.get(name, '')}", fontsize=7)
        ax.set_xlabel("head", fontsize=7)
        ax.set_ylabel("layer", fontsize=7)
        _style(ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, label=label)
    fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def probe_curves(results: Mapping[str, Mapping[int, float]], refs: Mapping[str, float],
                 path, title: str = "", ylabel: str = "held-out AUROC") -> Path:
    """Held-out probe metric by layer, one line per condition, model-free references dotted."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (key, by_layer) in enumerate(results.items()):
        layers = sorted(by_layer)
        ax.plot(layers, [by_layer[l] for l in layers], color=CATEGORICAL[k % len(CATEGORICAL)],
                lw=2, marker="o", ms=4, label=key)
    for name, value in refs.items():
        ax.axhline(value, color=MUTED, lw=1, ls=":")
        ax.text(1.0, value, f" {name}", va="center", fontsize=7, color=MUTED,
                transform=ax.get_yaxis_transform())
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.set_xlabel("hidden-state layer (0 = embeddings)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title(title, fontsize=9)
    _style(ax)
    return _save(fig, path)
