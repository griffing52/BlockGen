"""Animate AR generation block-by-block into a GIF + MP4.

Generates one sequence from a train_conditioned.py run (voxel repr), then
renders one frame per placed block on a FIXED-SIZE canvas (constant camera)
with a slow orbit. Frames -> GIF (PIL) and MP4 (ffmpeg).

The generation order is the model's own token order (bottom-up y-raster), so
the video shows the build assembling layer by layer.

Usage:
    python scripts/render_generation_video.py --run outputs/cond/image_run_c16 \
        --mode uncond --seed 3 --out outputs/videos
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import split_indices
from blockgen.models.voxel_transformer_cond import CondVoxelAR2
from blockgen.renderer.textured import load_face_textures, render_structure
from blockgen.training.train_conditioned import BOS, EOS, NUM_SPECIAL, EMBEDS
from blockgen.utils.data import Structure


def parse_placements(tokens, max_dim, block_index_to_pair):
    """Token sequence -> ordered [(x, y, z, block_id, block_data), ...]."""
    coord_off = NUM_SPECIAL
    block_off = NUM_SPECIAL + max_dim
    out, buf = [], []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t in (BOS, 0):
            continue
        if len(buf) < 3:
            if coord_off <= t < block_off:
                buf.append(t - coord_off)
            else:
                buf = []
        else:
            bi = t - block_off
            if 0 <= bi < len(block_index_to_pair):
                bid, bdata = block_index_to_pair[bi]
                out.append((buf[0], buf[1], buf[2], bid, bdata))
            buf = []
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/cond/image_run_c16")
    parser.add_argument("--mode", choices=["uncond", "image", "text"], default="uncond")
    parser.add_argument("--val-index", type=int, default=0,
                        help="which held-out val condition to use (image/text modes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--px", type=int, default=448)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--orbit-deg-per-frame", type=float, default=0.35)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--out", default="outputs/videos")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg.get("repr") == "voxel", "video script supports voxel-repr runs"
    device = "cuda"
    torch.manual_seed(args.seed)

    blob = json.loads((run / "block_vocab.json").read_text())
    max_dim = blob["max_dim"]
    pairs = [tuple(p) for p in blob["block_index_to_pair"]]

    model = CondVoxelAR2(
        cond_dim=cfg["cond_dim"], n_prefix=cfg["n_prefix"],
        vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=8, num_layers=cfg["layers"],
        dim_feedforward=4 * cfg["d_model"], pe=cfg["pe"]).to(device).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))

    if args.mode == "uncond":
        cond = model.null_cond[None]
        cfg_scale = 1.0
        tag = f"uncond_seed{args.seed}"
    else:
        _, manifest = load_house_structures(max_dim=32)
        assignment = split_indices(manifest["items"], seed=0)
        val_idx = [i for i in range(len(manifest["items"])) if assignment[i] == "val"]
        i = val_idx[args.val_index]
        embeds = np.load(EMBEDS)["image_embeds" if args.mode == "image" else "text_embeds"]
        e = embeds[i] if args.mode == "image" else embeds[i, 0][None]
        cond = torch.tensor(e[None], device=device, dtype=torch.float32)
        cfg_scale = args.cfg
        tag = f"{args.mode}_h{i:05d}_seed{args.seed}"

    toks = model.generate_cond(
        cond=cond.to(device), bos_token_id=BOS, eos_token_id=EOS,
        max_new_tokens=cfg["max_seq_len"] - 1, cfg_scale=cfg_scale)[0].cpu().numpy()
    placements = parse_placements(toks, max_dim, pairs)
    print(f"{len(placements)} blocks generated ({tag})", flush=True)

    face_textures = load_face_textures()
    canvas_ids = np.zeros((max_dim, max_dim, max_dim), np.int32)
    canvas_data = np.zeros_like(canvas_ids)
    frames = []
    for k, (x, y, z, bid, bdata) in enumerate(placements):
        canvas_ids[x, y, z] = bid
        canvas_data[x, y, z] = bdata
        s = Structure(block_ids=canvas_ids.copy(), block_data=canvas_data.copy())
        azim = 45.0 + k * args.orbit_deg_per_frame
        arr = render_structure(s, px=args.px, azim_deg=azim, elev_deg=30,
                               bg=(1.0, 1.0, 1.0), face_textures=face_textures)
        frames.append(Image.fromarray(arr).convert("RGB"))
        if k % 50 == 0:
            print(f"frame {k}/{len(placements)}", flush=True)
    frames += [frames[-1]] * args.hold_frames

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gif_path = out / f"gen_{tag}.gif"
    small = [f.resize((320, 320)) for f in frames]
    small[0].save(gif_path, save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0)

    mp4_path = out / f"gen_{tag}.mp4"
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(f"{td}/f_{i:05d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
             "-i", f"{td}/f_%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "22", str(mp4_path)], check=True)
    print(f"saved {gif_path} ({gif_path.stat().st_size//1024} KB) and "
          f"{mp4_path} ({mp4_path.stat().st_size//1024} KB), "
          f"{len(frames)} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
