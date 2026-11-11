"""`run_mesh_probe_3d_noclip.py` with no GPU at all.

Identical design, objects, seeds and rows. Two changes so the run never touches the
shared card: DINOv2 runs on the CPU (batch 8), and the renderer runs on OSMesa
(`PYOPENGL_PLATFORM=osmesa`) instead of EGL. Pass `--emd-from` to resume from an
earlier run's emd.npz of the same seed. Run with `CUDA_VISIBLE_DEVICES=` and no
`GPU_LOCK`.

    CUDA_VISIBLE_DEVICES= PYOPENGL_PLATFORM=osmesa PYTHONPATH=. .venv/bin/python \
        scripts/run_mesh_probe_3d_cpuonly.py --objects 600 --n 170 --seed 1 --device cpu \
        --workers 6 --emd-workers 8 --emd-from <run>/emd.npz --name mesh_probe3d_568_s1cpu
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.pop("GPU_LOCK", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_mesh_probe_3d_noclip as nc  # noqa: E402
from blockgen.eval.bench import features as ft  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))


class DinoOnlyCPU(nc.DinoOnly):
    def __call__(self, imgs, backbone):
        if backbone == "clipL":
            return np.zeros((len(imgs), 768), np.float32)
        return ft.embed_images(imgs, backbone, "cpu", batch=8)


nc.DinoOnly = DinoOnlyCPU

# pyrender keeps every uploaded mesh's buffers until its renderer is deleted; under OSMesa
# those buffers are host RAM (about 1 GB per minute here), so drop the renderer per object.
from blockgen.eval.bench import mesh_probe as _mp  # noqa: E402

_gpu_object = nc.gpu_object


def gpu_object_fresh(*a, **k):
    try:
        return _gpu_object(*a, **k)
    finally:
        for r in _mp._RENDERERS.values():
            r.delete()
        _mp._RENDERERS.clear()


nc.gpu_object = gpu_object_fresh

if __name__ == "__main__":
    nc.main()
