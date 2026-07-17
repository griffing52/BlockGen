"""Perceptual evaluation: score builds by how they LOOK, not by voxel overlap.

Why this exists (results.md T20). `nn_iou` disagreed with the eye three times in one
day: it reported 84% of baseline on canon-16 rubble (T17), and reported the native+BPE
arm *worse* than the control while that arm produced the first samples in the project
with pitched roofs and window openings (T20). It is an occupancy-IoU shape-matching
score, so it is also **grid-dependent**: at 16³ every house is a coarse blob and blobs
overlap generously; at 32³ builds are distinct and matching one is genuinely harder.
Dividing by `val_baseline_nn_iou` does not repair that -- 83% of a blob-matching score
is not commensurable with 75% of a detail-matching score. So cross-resolution arms are
currently not adjudicable at all.

**Rendering fixes exactly that.** A 16³ build and a 32³ build both become a 224x224
image, so the comparison happens in pixel space where resolution is a property of the
*content* rather than of the metric. That is also how a human judge would do it.

What we compute, per structure set:
  * **CMMD** -- CLIP-space Maximum Mean Discrepancy to a real reference set.
    Chosen over FID deliberately: FID is Inception/ImageNet-biased and badly biased at
    small n (it wants 10k+; we have 16-64). CMMD is unbiased and designed for small
    samples (Jayasumana et al., "Rethinking FID", arXiv 2401.09603). LOWER is better.
  * **CLIP text score** -- cosine similarity of renders to a prompt like
    "a Minecraft house". HIGHER is better. Catches "is it even house-like".

**Both arms must be scored against the SAME real reference (native builds).** Scoring
each arm against its own training distribution reintroduces exactly the normalization
problem this module exists to escape -- it would ask "did it learn its own data?" when
the question is "does it produce good Minecraft houses?".

Sanity check before trusting any of it: `--validate` scores real native builds vs the
SAME builds decimated to canon-16. We know from T17/F16 that decimation destroys 86.5%
of blocks and deletes whole roofs, so a metric worth using MUST rank native above
canon-16 by a wide margin. If it cannot see damage that obvious, it cannot adjudicate
anything subtler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from blockgen.utils.data import Structure

_CLIP = {"model": None, "proc": None}
DEFAULT_VIEWS = ((45.0, 30.0), (135.0, 30.0), (225.0, 30.0), (315.0, 30.0))
DEFAULT_PROMPT = "a Minecraft house built from blocks"


def _load_clip(device: str = "cuda"):
    if _CLIP["model"] is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP["model"] = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").to(device).eval()
        _CLIP["proc"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _CLIP["model"], _CLIP["proc"]


def render_views(structures: Sequence[Structure], views=DEFAULT_VIEWS,
                 px: int = 224, verbose: bool = True) -> List[np.ndarray]:
    """Render each structure from several azimuths. Returns a flat list of RGB images.

    Same textured pyrender path as the dataset sheets, so generated and real builds are
    rendered identically -- the comparison lives in pixel space, not voxel space.
    """
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures
    tex = load_face_textures()
    out: List[np.ndarray] = []
    for i, s in enumerate(structures):
        for azim, elev in views:
            try:
                img = render_structure(s, px=px, azim_deg=azim, elev_deg=elev,
                                       face_textures=tex)
            except Exception:  # degenerate/empty samples must not kill a sweep
                img = np.full((px, px, 3), 255, np.uint8)
            if img.shape[-1] == 4:  # composite alpha onto white, as the sheets do
                a = img[..., 3:4].astype(np.float32) / 255.0
                img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
            out.append(img[..., :3])
        if verbose and (i + 1) % 16 == 0:
            print(f"    rendered {i + 1}/{len(structures)}", flush=True)
    return out


@torch.no_grad()
def clip_image_features(images: Sequence[np.ndarray], device: str = "cuda",
                        batch: int = 64) -> torch.Tensor:
    """L2-normalized CLIP image embeddings [N, 512]."""
    model, proc = _load_clip(device)
    feats = []
    for i in range(0, len(images), batch):
        chunk = [np.asarray(im) for im in images[i:i + batch]]
        px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device)
        f = model.get_image_features(pixel_values=px)
        feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def clip_text_features(prompts: Sequence[str], device: str = "cuda") -> torch.Tensor:
    model, proc = _load_clip(device)
    tok = proc(text=list(prompts), return_tensors="pt", padding=True).to(device)
    f = model.get_text_features(**tok)
    return torch.nn.functional.normalize(f, dim=-1).float().cpu()


def cmmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0,
         scale: float = 1000.0) -> float:
    """CLIP-MMD between two sets of L2-normalized features. Lower = more alike.

    Gaussian RBF MMD^2 with the unbiased-in-n formulation from arXiv 2401.09603.
    `scale` only makes the numbers readable (they are otherwise ~1e-3).
    """
    x = x.double()
    y = y.double()
    g = 1.0 / (2.0 * sigma ** 2)

    def _k(a, b):
        # ||a-b||^2 = |a|^2 + |b|^2 - 2ab; features are unit-norm but stay general
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
        return torch.exp(-g * d.clamp(min=0))

    return float(scale * (_k(x, x).mean() + _k(y, y).mean() - 2.0 * _k(x, y).mean()))


@dataclass
class PerceptualReport:
    cmmd: float                # vs the real reference set. LOWER better.
    clip_text: float           # mean cosine to the prompt. HIGHER better.
    n_gen: int
    n_ref: int
    prompt: str

    def row(self, tag: str) -> dict:
        return {"track": tag, "cmmd": round(self.cmmd, 3),
                "clip_text": round(self.clip_text, 4),
                "n_gen": self.n_gen, "n_ref": self.n_ref}


def evaluate_perceptual(generated: Sequence[Structure], reference: Sequence[Structure],
                        prompt: str = DEFAULT_PROMPT, views=DEFAULT_VIEWS,
                        px: int = 224, device: str = "cuda",
                        ref_feats: Optional[torch.Tensor] = None,
                        verbose: bool = True) -> PerceptualReport:
    """Score `generated` against a real `reference` set, in pixel space.

    Pass `ref_feats` to reuse a cached reference embedding across arms (the reference
    must be identical across arms for the numbers to be comparable).
    """
    if verbose:
        print(f"  rendering {len(generated)} generated x {len(views)} views...", flush=True)
    g = clip_image_features(render_views(generated, views, px, verbose), device)
    if ref_feats is None:
        if verbose:
            print(f"  rendering {len(reference)} reference x {len(views)} views...",
                  flush=True)
        ref_feats = clip_image_features(render_views(reference, views, px, verbose), device)
    t = clip_text_features([prompt], device)
    return PerceptualReport(cmmd=cmmd(g, ref_feats),
                            clip_text=float((g @ t.T).mean()),
                            n_gen=len(generated), n_ref=ref_feats.shape[0] // len(views),
                            prompt=prompt)
