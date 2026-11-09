"""CPU (OSMesa) rendering for the paper's figures, opt-in.

    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/paper_figures.py fig7

The textured renderer defaults to pyrender over EGL, which needs the GPU. A
figure renders a few dozen thumbnails, so on a shared card it is cheaper to
render them in software than to queue for the GPU. OSMesa ignores the RGBA flag
and returns an opaque image, so the alpha channel is recovered exactly by
rendering each view twice, over white and over black:

    C_white = a C + (1 - a) 255,   C_black = a C   =>   a = 1 - (C_white - C_black) / 255

which is the same alpha EGL's transparent background gives (glass included).
`install()` must run before anything imports pyrender; it then replaces
`blockgen.renderer.textured.render_structure` in place, so every caller that
imports the function at call time gets the CPU path.
"""

from __future__ import annotations

import os

import numpy as np


def wanted() -> bool:
    return os.environ.get("BLOCKGEN_FIG_RENDER", "").lower() == "cpu"


def install() -> bool:
    """Switch the figure renderer to OSMesa when BLOCKGEN_FIG_RENDER=cpu."""
    if not wanted():
        return False
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    from blockgen.renderer import textured

    if getattr(textured.render_structure, "_cpu_two_pass", False):
        return True
    gpu_render = textured.render_structure

    def render_structure(structure, *args, bg=None, **kw):
        if bg is not None:
            return gpu_render(structure, *args, bg=bg, **kw)
        w = gpu_render(structure, *args, bg=(1.0, 1.0, 1.0), **kw)[..., :3].astype(np.float32)
        b = gpu_render(structure, *args, bg=(0.0, 0.0, 0.0), **kw)[..., :3].astype(np.float32)
        a = np.clip(1.0 - (w - b).max(axis=-1) / 255.0, 0.0, 1.0)
        rgb = np.where(a[..., None] > 0, b / np.maximum(a[..., None], 1e-6), 0.0)
        out = np.concatenate([np.clip(rgb, 0, 255), 255.0 * a[..., None]], axis=-1)
        return np.round(out).astype(np.uint8)

    render_structure._cpu_two_pass = True
    textured.render_structure = render_structure
    return True
