"""Animate attachment/growth generation, highlighting the face being decided.

Each op in the stream is a decision about ONE open face popped from the frontier
(implementation_plan.md §2). This replays the *exact* decoder frontier discipline
(`attach_order.attach_ops_to_structure`) but pauses at every placement and renders:

  * everything placed so far (block-coloured voxels),
  * the **current target face** in red -- the port the current token is autoregressing
    on -- drawn as a quad on the shared face plus a wire cube at the cell it leads to,
  * the voxel placed by the previous op in green.

Because the frame is the state *before* the op is applied, watching it shows the model
choosing face-by-face -- and it makes each method's failure mode visible: a filament
(layered_raster), a plate (bfs), or a sphere (radial) grows in front of you.

Writes `<arm>/growth.gif` and a `<arm>/growth_keyframes.png` contact sheet.

Usage:
    python -m scripts.attach_growth_video --arm outputs/run_.../radial --frames 100
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from blockgen.utils.attach_order import (LIFO_ORDERINGS, ORDERINGS, OP_ATTACH,
                                         OP_CLOSE, OP_SEED, OrderingContext,
                                         _unpack_token)
from blockgen.utils.attach_vocab import BOS_ID, EOS_ID, AttachVocab
from blockgen.utils.graph_data import PORT_DIRECTIONS

Coord = Tuple[int, int, int]


def _color_for(block_id: int) -> Tuple[float, float, float]:
    """Stable pastel per block id, so materials read as distinct without a texture."""
    h = (block_id * 2654435761) & 0xFFFFFFFF
    r = 0.45 + 0.40 * (((h >> 0) & 0xFF) / 255.0)
    g = 0.40 + 0.42 * (((h >> 8) & 0xFF) / 255.0)
    b = 0.35 + 0.42 * (((h >> 16) & 0xFF) / 255.0)
    return (r, g, b)


def replay_steps(ops, ordering: str):
    """Yield one event per op, mirroring the decoder frontier exactly.

    Event: dict(kind, cells(copy), target=(parent, dir, child), placed_prev).
    ``cells`` is the state BEFORE the op; ``target`` is the face being decided now.
    """
    prio = ORDERINGS[ordering]
    lifo = ordering in LIFO_ORDERINGS
    grid = 512
    origin = (grid // 2, 2, grid // 2)

    cells: Dict[Coord, Tuple[int, int]] = {}
    placed: set = set()
    frontier: List[tuple] = []
    counter = 0
    visited: set = set()
    ctx = OrderingContext(seed=origin)
    n_seed = 0
    prev_placed: Optional[Coord] = None

    def push(voxel: Coord):
        nonlocal counter
        for d, (dx, dy, dz) in enumerate(PORT_DIRECTIONS):
            child = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            if (voxel, d) in visited or child in placed:
                continue
            visited.add((voxel, d))
            heapq.heappush(frontier, (prio(voxel, d, child, ctx),
                                      -counter if lifo else counter, voxel, d))
            counter += 1

    for op in ops:
        if op.kind == OP_SEED:
            n_seed += 1
            anchor = (origin[0], origin[1], origin[2] + (n_seed - 1) * (grid // 16))
            frontier.clear(); visited.clear(); counter = 0
            ctx = OrderingContext(seed=anchor)
            yield {"kind": OP_SEED, "cells": dict(cells),
                   "target": (anchor, -1, anchor), "placed_prev": prev_placed}
            cells[anchor] = _unpack_token(op.piece, False)
            placed.add(anchor); push(anchor); prev_placed = anchor
            continue

        # Peek the face this op will decide (without disturbing frontier order).
        target = None
        tmp = []
        while frontier:
            item = heapq.heappop(frontier)
            _p, _c, parent, d = item
            dx, dy, dz = PORT_DIRECTIONS[d]
            child = (parent[0] + dx, parent[1] + dy, parent[2] + dz)
            if child in placed:
                continue
            target = (parent, d, child)
            tmp.append(item)
            break
        for it in tmp:
            heapq.heappush(frontier, it)
        if target is None:
            continue

        yield {"kind": op.kind, "cells": dict(cells), "target": target,
               "placed_prev": prev_placed}

        # Apply (same as decoder).
        while frontier:
            _p, _c, parent, d = heapq.heappop(frontier)
            dx, dy, dz = PORT_DIRECTIONS[d]
            child = (parent[0] + dx, parent[1] + dy, parent[2] + dz)
            if child in placed:
                continue
            if op.kind == OP_ATTACH:
                cells[child] = _unpack_token(op.piece, False)
                placed.add(child); push(child); prev_placed = child
            break

    # Final settled state.
    yield {"kind": -1, "cells": dict(cells), "target": None, "placed_prev": prev_placed}


def replay_steps_final(ops, ordering: str) -> dict:
    """Just the settled cell dict -- cheap sizing pass for sample selection."""
    last = None
    for last in replay_steps(ops, ordering):
        pass
    return last["cells"] if last else {}


def _face_quad(parent: Coord, d: int, lo: np.ndarray):
    """4 corners (plot coords) of parent's face in direction d. Plot up = our y."""
    dx, dy, dz = PORT_DIRECTIONS[d]
    # our (x,y,z) -> plot (x, z, y); cell spans [c, c+1]
    px, py, pz = parent[0] - lo[0], parent[1] - lo[1], parent[2] - lo[2]
    # face at +0.5 along the delta axis, spanning the other two
    cx, cy, cz = px + 0.5, py + 0.5, pz + 0.5  # our-coord cell centre
    cx += 0.5 * dx; cy += 0.5 * dy; cz += 0.5 * dz
    if dx != 0:      us, vs = (0, 1, 0), (0, 0, 1)
    elif dy != 0:    us, vs = (1, 0, 0), (0, 0, 1)
    else:            us, vs = (1, 0, 0), (0, 1, 0)
    corners = []
    for su, sv in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]:
        ox = cx + su * us[0] + sv * vs[0]
        oy = cy + su * us[1] + sv * vs[1]
        oz = cz + su * us[2] + sv * vs[2]
        corners.append((ox, oz, oy))  # -> plot (x, z, y)
    return corners


def render_frame(ev, lo, shape, title, elev=24, azim=-58, px=520):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    cells = ev["cells"]
    if cells:
        coords = np.array(list(cells.keys()))
        occ = np.zeros((shape[0], shape[2], shape[1]), dtype=bool)  # plot (x,z,y)
        fc = np.zeros(occ.shape + (4,), dtype=float)
        for (x, y, z), (bid, _bd) in cells.items():
            i, j, k = x - lo[0], z - lo[2], y - lo[1]
            occ[i, j, k] = True
            fc[i, j, k] = (*_color_for(bid), 1.0)
        # newest voxel in green
        pp = ev["placed_prev"]
        if pp is not None and pp in cells:
            i, j, k = pp[0] - lo[0], pp[2] - lo[2], pp[1] - lo[1]
            fc[i, j, k] = (0.15, 0.95, 0.25, 1.0)
        ax.voxels(occ, facecolors=fc, edgecolor=(0, 0, 0, 0.15), linewidth=0.3, shade=True)

    # current target face in red
    if ev["target"] is not None and ev["kind"] != OP_SEED:
        parent, d, child = ev["target"]
        quad = _face_quad(parent, d, lo)
        col = "red" if ev["kind"] == OP_ATTACH else (0.2, 0.5, 1.0)
        ax.add_collection3d(Poly3DCollection([quad], facecolors=col, alpha=0.85,
                                             edgecolors="k", linewidths=0.8))

    ax.set_xlim(0, shape[0]); ax.set_ylim(0, shape[2]); ax.set_zlim(0, shape[1])
    ax.set_box_aspect((shape[0], shape[2], shape[1]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def load_model(arm: Path, device: str):
    import torch
    blob = torch.load(arm / "model.pt", map_location=device, weights_only=False)
    cfg = blob["config"]
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    m = VoxelTransformerAR2(
        vocab_size=blob["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], pe="sin",
    ).to(device)
    m.load_state_dict(blob["state_dict"]); m.eval()
    return m, cfg, blob["ordering"]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--frames", type=int, default=100, help="approx target frame count")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--min-voxels", type=int, default=120)
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arm = Path(args.arm)
    vocab = AttachVocab.load(arm / "attach_vocab.json")
    model, cfg, ordering = load_model(arm, args.device)

    # Pick a non-trivial sample.
    torch.manual_seed(args.seed)
    best_ops, best_n = None, -1
    for _ in range(args.tries):
        toks = model.generate(bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                              max_new_tokens=cfg["max_seq_len"], temperature=args.temp,
                              device=args.device)
        if isinstance(toks, torch.Tensor):
            toks = toks.flatten().tolist()
        ops = vocab.ids_to_ops(toks)
        # Select by ACTUAL placed voxels, not op count: many ATTACH ops hit an empty
        # frontier and place nothing (itself a collapse signature), so op count and
        # voxel count diverge widely.
        n = len(replay_steps_final(ops, ordering))
        if n > best_n:
            best_ops, best_n = ops, n
        if n >= args.min_voxels:
            break
    ops = best_ops

    # Final extent for a fixed camera.
    all_steps = list(replay_steps(ops, ordering))
    final_cells = all_steps[-1]["cells"]
    n_ops = sum(1 for o in ops if o.kind in (OP_SEED, OP_ATTACH))
    print(f"[video] {ordering}: {len(final_cells)} voxels from {n_ops} attach ops "
          f"({len(ops)} total ops)", flush=True)
    coords = np.array(list(final_cells.keys()))
    lo = coords.min(axis=0)
    shape = tuple(int(v) for v in (coords.max(axis=0) - lo + 1))

    # Frame on SEED + ATTACH steps (the ones that build), subsampled to ~args.frames.
    build_steps = [s for s in all_steps if s["kind"] in (OP_SEED, OP_ATTACH)]
    stride = max(1, len(build_steps) // args.frames)
    picks = build_steps[::stride] + [all_steps[-1]]

    import imageio.v2 as imageio
    frames = []
    for i, ev in enumerate(picks):
        placed = len(ev["cells"])
        kind = {OP_SEED: "SEED", OP_ATTACH: "ATTACH", OP_CLOSE: "CLOSE", -1: "DONE"}[ev["kind"]]
        title = f"{ordering}  |  step {i*stride}/{len(build_steps)}  |  {placed} blocks  |  {kind}"
        frames.append(render_frame(ev, lo, shape, title))
        if i % 20 == 0:
            print(f"  frame {i}/{len(picks)}", flush=True)

    gif = arm / "growth.gif"
    imageio.mimsave(gif, frames, fps=args.fps, loop=0)
    print(f"[video] wrote {gif} ({len(frames)} frames)", flush=True)

    # Keyframe contact sheet for quick inspection.
    idxs = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, k in zip(axes.flat, idxs):
        ax.imshow(frames[k]); ax.set_axis_off(); ax.set_title(f"frame {k}", fontsize=8)
    for ax in axes.flat[len(idxs):]:
        ax.set_axis_off()
    fig.suptitle(f"attach-growth :: {ordering} :: face being decided in red, newest block green")
    fig.tight_layout()
    sheet = arm / "growth_keyframes.png"
    fig.savefig(sheet, dpi=90); plt.close(fig)
    print(f"[video] wrote {sheet}", flush=True)


if __name__ == "__main__":
    main()
