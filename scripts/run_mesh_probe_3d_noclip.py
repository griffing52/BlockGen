"""`run_mesh_probe_3d.py` on a GPU with under 1 GB free.

Identical design, seeds, renders, DINOv2 4-view and 4-cutaway rows, V-KID and every
3D-native row. Three changes, all for memory: DINOv2 runs on CUDA at batch 8; Uni3D,
ULIP-2, chamfer and IoU run on the CPU (`--device cpu`); the CLIP-L rows and the
8/16-view DINOv2 rows are not computed and are dropped from the written tables
(none of them is in tab:probe3d). Arguments are those of `run_mesh_probe_3d.py`;
pass `--device cpu` and `--name`.

    GPU_LOCK=... PYTHONPATH=. .venv/bin/python scripts/run_mesh_probe_3d_noclip.py \
        --objects 512 --n 128 --seed 0 --device cpu --name mesh_probe3d_all_s0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_mesh_probe as rmp            # noqa: E402
import run_mesh_probe_3d as r3          # noqa: E402
from blockgen.eval.bench import features as ft   # noqa: E402
from blockgen.eval.bench import mesh_probe as mp  # noqa: E402

DROP = ("clip4_kid", "clip4_fd", "clip_cut4_kid", "clipscore_drop", "dino8_kid", "dino16_kid")


class DinoOnly:
    def __init__(self, device):
        self.text = np.zeros(768, np.float32)

    def __call__(self, imgs, backbone):
        if backbone == "clipL":
            return np.zeros((len(imgs), 768), np.float32)
        return ft.embed_images(imgs, backbone, "cuda", batch=8)


def gpu_object(m_real, vis, hid, g, oi, variants, emb, rng):
    """`run_mesh_probe_3d.gpu_object` without the 8/16-view renders and CLIP."""
    out = {}
    for vname in variants:
        vm = rmp.make_variant(vname, m_real, vis, hid, g, np.random.default_rng(oi))
        v4, depth_pts = mp.render(vm, mp.VIEWS4, depth=True)
        c4 = mp.render(vm, mp.VIEWS4, cut=True)
        f = {"_v4": v4, "_c4": c4}
        d4 = emb(v4, "dinov2b")
        f["dino4"] = ft.pooled(d4[None])[0]
        crng = np.random.default_rng([oi, len(vname)])   # placeholder rows, dropped below
        f["clip4"] = crng.standard_normal(768).astype(np.float32)
        f["clipscore"] = float(crng.standard_normal())
        f["dino8"] = f["dino4"]
        f["dino16"] = f["dino4"]
        f["dino_cut4"] = ft.pooled(emb(c4, "dinov2b")[None])[0]
        f["clip_cut4"] = crng.standard_normal(768).astype(np.float32)
        rp = np.concatenate(depth_pts) if depth_pts else np.zeros((0, 3), np.float32)
        f["pts_render"] = (rp[rng.choice(len(rp), 1024, replace=len(rp) < 1024)] if len(rp)
                           else np.zeros((1024, 3), np.float32))
        out[vname] = f
    return out


def main():
    r3.Embedder = DinoOnly
    rmp.Embedder = DinoOnly
    r3.gpu_object = gpu_object
    r3.main()
    name = sys.argv[sys.argv.index("--name") + 1]
    run = sorted(Path("outputs").glob(f"run_*_{name}"))[-1]
    res_p = run / "results.json"
    if res_p.exists():
        res = json.loads(res_p.read_text())
        for k in DROP:
            res["results"].pop(k, None)
        res["note"] = ("CLIP-L and 8/16-view DINOv2 rows not computed (GPU memory); "
                       "3D encoders on CPU. scripts/run_mesh_probe_3d_noclip.py")
        res_p.write_text(json.dumps(res, indent=2) + "\n")
        tab = run / "table.md"
        if tab.exists():
            lines = [l for l in tab.read_text().splitlines()
                     if not any(l.startswith(f"| {k} ") for k in DROP)]
            tab.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
