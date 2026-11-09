"""Sample Dream-Cubed (Merino et al., arXiv:2604.22847) chunks for the external arm.

Runs under the overlay venv `libs/DreamCubed/.venv` (main-venv torch + einops +
ema_pytorch). It does not import Dream-Cubed's `inference.py`: that module pulls
in wandb, torchvision, pyvista and the `denoising_diffusion_pytorch` package
only for training/DDPM/rendering. We build the DiT3D + MD4 sampler exactly the
way `inference._build_model_from_config` / `generate_random_samples` do, with the
unused imports stubbed.

    flock <gpu.lock> libs/DreamCubed/.venv/bin/python scripts/ext/dreamcubed_generate.py \\
        --n 288 --biome village --out <run_dir>/raw_village.pt

Output: torch.save({"names": [N,32,32,32] uint8 compressed-vocab indices,
"vocab": list of 1.12 Bukkit block names, "biome": ..., "meta": {...}}).
Axis order is Dream-Cubed's (x, y, z), y up (its rotation augmentation rotates
dims (0, 2)), which matches our Structure XYZ convention.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "libs" / "DreamCubed"
WEIGHTS = REPO / "weights"


def _stub_imports() -> None:
    for name in ("wandb", "visualization_utils", "render_dataset"):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.save_chunks = None
        mod.MinecraftVisualizerPyVista = None
        mod.render_dataset_to_images = None
        sys.modules[name] = mod
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MD4_p4_boosted")
    ap.add_argument("--mappings", default=str(WEIGHTS / "mappings" / "village_boosted_dataset_cc_oh_mappings.pt"))
    ap.add_argument("--biome", default="village")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cond-scale", type=float, default=None, help="default: config default_cond_scale")
    ap.add_argument("--steps", type=int, default=None, help="default: config reverse_steps (1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cpu only for pipeline smoke tests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    args.out = str(Path(args.out).resolve())
    args.mappings = str(Path(args.mappings).resolve())
    _stub_imports()
    import os
    os.chdir(REPO)  # data_utils reads assets/block_types.json relative to cwd
    import numpy as np
    import torch
    from accelerate import Accelerator
    import data_utils
    from data_utils import BlockBiomeConverter
    # Upstream bug: the converter's default reads assets/block_types.json, which
    # the public repo does not ship. The mappings file carries block_to_str, so
    # the default is overwritten right after; an empty table is equivalent.
    data_utils.load_block_to_str_mapping = lambda *a, **k: {}
    from dit import DiT3D
    from discrete_diffusion_md4 import MD4Discrete3D

    mdir = WEIGHTS / args.model
    config = json.loads((mdir / "config.json").read_text())
    mappings = torch.load(args.mappings, map_location="cpu", weights_only=False)
    conv = BlockBiomeConverter(mappings["block_mappings"], mappings["biome_mappings"])
    K = len(conv.block_to_index)
    biome_idx = conv.biome_to_index[args.biome]
    dev = torch.device(args.device)

    model = DiT3D(
        in_dim=K + 1, out_dim=K,
        hidden_channels=int(config["hidden_channels"]), image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]), time_dim=int(config["time_dim"]),
        depth=int(config["depth"]), num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]), attn_drop=0.0, proj_drop=0.0,
        class_conditional=True, num_classes=int(config["num_classes"]),
        comb_method=str(config.get("comb_method", "add")),
        cond_drop_prob=float(config.get("cond_drop_prob", 0.2)),
    )
    raw = torch.load(mdir / "inference_weights" / "model_best_weights.pt", map_location="cpu", weights_only=False)
    sd = raw.get("ema_model", raw.get("model", raw)) if isinstance(raw, dict) else raw
    if isinstance(sd, dict) and any(k.startswith("ema_model.") for k in sd):
        sd = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {args.model}: K={K} missing={len(missing)} unexpected={len(unexpected)} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)
    if missing:
        raise SystemExit(f"missing keys, e.g. {missing[:5]}")
    model = model.to(dev).eval()

    acc = Accelerator(cpu=args.device == "cpu",
                      mixed_precision=(config.get("mixed_precision_type", "bf16") if config.get("amp") else "no"))
    md4 = MD4Discrete3D(accelerator=acc, num_classes=K, device=dev)
    air = conv.get_air_block_index()
    steps = int(args.steps or config.get("reverse_steps", 1000))
    scale = float(args.cond_scale if args.cond_scale is not None else config.get("default_cond_scale", 4.0))
    S = int(config["image_size"])
    torch.manual_seed(args.seed)

    out, t0 = [], time.time()
    while sum(o.shape[0] for o in out) < args.n:
        b = min(args.batch, args.n - sum(o.shape[0] for o in out))
        cls = torch.full((b,), biome_idx, device=dev, dtype=torch.long)
        idx = md4.sample(model, batch_size=b, spatial_size=(S, S, S), reverse_steps=steps,
                         progress=False, air_index_fallback=int(air), class_cond=cls, cond_scale=scale)
        out.append(idx.to("cpu", torch.uint8))
        done = sum(o.shape[0] for o in out)
        el = time.time() - t0
        print(f"  {done}/{args.n}  {el / done:.2f}s/sample  eta {el / done * (args.n - done) / 60:.1f} min  "
              f"peak {(torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0):.2f} GiB", flush=True)
        # checkpoint partial progress so a kill does not lose everything
        _save(args, out, conv, config, steps, scale, el)
    _save(args, out, conv, config, steps, scale, time.time() - t0)
    print(f"[done] {args.out}", flush=True)
    return 0


def _save(args, out, conv, config, steps, scale, elapsed):
    import torch
    bm = conv.block_to_index
    idx_to_orig = {v: k for k, v in bm.items()}
    b2s = torch.load(args.mappings, map_location="cpu", weights_only=False)["block_mappings"]["block_to_str"]
    vocab = [b2s[idx_to_orig[i]] for i in range(len(bm))]
    torch.save({
        "indices": torch.cat(out, 0), "vocab": vocab, "biome": args.biome,
        "meta": {"model": args.model, "config_run_name": config.get("run_name"), "mappings": Path(args.mappings).name,
                 "reverse_steps": steps, "cond_scale": scale, "seed": args.seed, "batch": args.batch,
                 "elapsed_s": elapsed, "peak_gib": (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0)},
    }, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
