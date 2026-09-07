"""Render structures to images and embed them with a frozen vision backbone.

The FULL tier lives in feature space because voxel IoU is grid-dependent: at 16^3
every house is a coarse blob and blobs overlap generously, at 32^3 builds are
distinct and matching one is genuinely harder, so cross-resolution arms are not
adjudicable at all (T20). Rendering turns a 16^3 and a 32^3 build into the same
224x224 image, which is also how a human judge would compare them.

**Pooling: mean of L2-normalized per-view features, re-normalized.**
`blockgen/eval/perceptual.py` flattens views into one flat list, making n four
times the number of structures with 4x-correlated rows. That is not merely a
confidence-interval problem -- measured on real builds, the flattened KID null
is biased to +8e-5 at n=16 (2 sigma) while the pooled null sits at 0.000.
Pooling makes the sample unit the structure, which is what every estimator and
every bootstrap in this suite already assumes.

The cost is real and worth stating: a build with a handsome facade and a hollow
back averages to "fine". `pooled(..., mode="per_view")` keeps the per-view
features so callers can also report the worst view, still at n = n_structures.

Concatenating the four views (3072-d) is deliberately not offered: a 3072x3072
covariance cannot be estimated from 64 samples, k-NN radii degenerate at that
dimensionality, and it presumes a canonical "front" that most builds lack.

Cost, measured on this machine: 21 ms/view to render, 2.4 ms/image to embed, so
the entire 2661-build corpus takes about 4 minutes and 33 MB. The cache exists
for turnaround, not because the computation is expensive -- so it invalidates
aggressively, and it stores features only, never PNGs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.perceptual import DEFAULT_VIEWS
from blockgen.utils.data import Structure

FEATURE_ROOT = Path("data/minecraft/features")

#: name -> (HF model id, feature dimension)
BACKBONES: Dict[str, Tuple[str, int]] = {
    "dinov2b": ("facebook/dinov2-base", 768),
    "clipL": ("openai/clip-vit-large-patch14", 768),
    # Bit-identical to the model `blockgen/eval/perceptual.py` uses, kept so the
    # legacy CMMD numbers can be reproduced inside the new harness.
    "clipB": ("openai/clip-vit-base-patch32", 512),
    "siglipB": ("google/siglip-base-patch16-224", 768),
}

_MODELS: Dict[str, tuple] = {}


@dataclass(frozen=True)
class ViewConfig:
    """Camera setup. Part of the cache key, because it changes every number."""
    views: Tuple[Tuple[float, float], ...] = DEFAULT_VIEWS
    px: int = 224
    ortho: bool = True
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def key(self) -> str:
        az = "-".join(f"{a:g}" for a, _ in self.views)
        el = "-".join(f"{e:g}" for _, e in self.views)
        return (f"v{len(self.views)}_az{az}_el{el}_px{self.px}"
                f"_{'ortho' if self.ortho else 'persp'}")

    @property
    def n_views(self) -> int:
        return len(self.views)


def _load_backbone(name: str, device: str = "cuda"):
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    if name not in _MODELS:
        from transformers import AutoImageProcessor, AutoModel, CLIPModel
        model_id = BACKBONES[name][0]
        proc = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        if "clip" in name.lower():
            model = CLIPModel.from_pretrained(model_id).to(device).eval()
        else:
            model = AutoModel.from_pretrained(model_id).to(device).eval()
        _MODELS[name] = (model, proc)
    return _MODELS[name]


def render_views(structures: Sequence[Structure], view: ViewConfig = ViewConfig(),
                 verbose: bool = True, max_fail_frac: float = 0.02
                 ) -> List[np.ndarray]:
    """Render every structure from every view, in structure-major order.

    Uses the same textured pyrender path and the same alpha-onto-white composite
    as `blockgen.eval.perceptual.render_views`, so pixels are identical to the
    existing pipeline and generated and real builds are rendered the same way.

    **This fails loudly.** An earlier version substituted a blank white frame for
    any exception, which made a broken renderer indistinguishable from a corpus
    of blank builds -- and that is not a hypothetical: a PyOpenGL/Python version
    mismatch made every `render_structure` call raise, every image came back
    white, every DINO feature collapsed to one vector, and every arm scored a
    near-perfect KID of -4e-13. Worse, `load_or_build` verifies the render canary
    before trusting its cache, so the all-white canary would have failed to match
    and the 30 MB feature cache would have been silently *rebuilt from blank
    frames*, destroying every previously valid number.

    A scattered failure on a pathological sample is tolerated up to
    `max_fail_frac`; anything beyond that raises with the first traceback, on the
    principle the rest of the suite already follows -- a metric that cannot be
    computed is reported as null with a reason, never as a number.
    """
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures

    tex = load_face_textures()
    out: List[np.ndarray] = []
    bg255 = np.array(view.bg, dtype=np.float32) * 255.0
    failures: List[BaseException] = []
    for i, s in enumerate(structures):
        for azim, elev in view.views:
            try:
                img = render_structure(s, px=view.px, azim_deg=azim, elev_deg=elev,
                                       ortho=view.ortho, face_textures=tex)
            except Exception as exc:
                failures.append(exc)
                img = np.full((view.px, view.px, 3), 255, np.uint8)
            if img.shape[-1] == 4:
                a = img[..., 3:4].astype(np.float32) / 255.0
                img = (img[..., :3] * a + bg255 * (1 - a)).astype(np.uint8)
            out.append(np.ascontiguousarray(img[..., :3]))
        if verbose and (i + 1) % 128 == 0:
            print(f"    rendered {i + 1}/{len(structures)}", flush=True)

    if failures and len(failures) > max(1, int(max_fail_frac * max(len(out), 1))):
        raise RuntimeError(
            f"renderer failed on {len(failures)}/{len(out)} views "
            f"({len(failures) / max(len(out), 1):.0%}), above the "
            f"{max_fail_frac:.0%} tolerance. Every failed view becomes a blank "
            f"frame, which scores as a perfect match rather than as an error, so "
            f"this run is aborted instead of reported. First failure: "
            f"{type(failures[0]).__name__}: {failures[0]}") from failures[0]
    if failures:
        print(f"[features] WARNING: {len(failures)}/{len(out)} views failed to "
              f"render and were blanked ({type(failures[0]).__name__})", flush=True)
    return out


def embed_images(images: Sequence[np.ndarray], backbone: str = "dinov2b",
                 device: str = "cuda", batch: int = 64) -> np.ndarray:
    """L2-normalized image features `[N, D]`, float32."""
    import torch

    model, proc = _load_backbone(backbone, device)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [np.asarray(im) for im in images[i:i + batch]]
            px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device)
            if hasattr(model, "get_image_features"):        # CLIP / SigLIP
                f = model.get_image_features(pixel_values=px)
            else:                                           # DINOv2 -> CLS token
                f = model(pixel_values=px).last_hidden_state[:, 0]
            f = torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, 0) if feats else np.zeros((0, 0), np.float32)


def embed_views(structures: Sequence[Structure], view: ViewConfig = ViewConfig(),
                backbone: str = "dinov2b", device: str = "cuda", batch: int = 64,
                verbose: bool = True) -> np.ndarray:
    """`[N, V, D]` per-view features, each L2-normalized."""
    if not structures:
        return np.zeros((0, view.n_views, BACKBONES[backbone][1]), np.float32)
    images = render_views(structures, view, verbose=verbose)
    flat = embed_images(images, backbone, device, batch)
    return flat.reshape(len(structures), view.n_views, -1)


def pooled(feats: np.ndarray, mode: str = "mean") -> np.ndarray:
    """`[N, V, D]` -> `[N, D]` (mean) or unchanged (per_view). See module docstring."""
    if feats.ndim != 3:
        raise ValueError(f"expected [N, V, D], got shape {feats.shape}")
    if mode == "per_view":
        return feats
    if mode != "mean":
        raise ValueError(f"pool mode must be 'mean' or 'per_view', got {mode!r}")
    m = feats.mean(axis=1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norm, 1e-12)


# --- cache -----------------------------------------------------------------
CANARY_SEED = 12345


def render_canary(view: ViewConfig, n: int = 4) -> str:
    """Fingerprint of a fixed render, so a changed texture pack invalidates.

    Without this the cache is keyed only on things we *declare* (model id, view
    config) and not on the renderer's actual output. A silently updated texture
    pack or renderer change would otherwise poison every cached number
    permanently, and the poisoning would be invisible.
    """
    rng = np.random.default_rng(CANARY_SEED)
    structs = []
    for k in range(n):
        ids = np.zeros((4 + k, 4, 4), dtype=np.int32)
        ids[:, 0, :] = 4
        ids[0, :, :] = 5
        ids[rng.integers(0, 4 + k), rng.integers(0, 4), rng.integers(0, 4)] = 98
        structs.append(Structure(block_ids=ids, block_data=np.zeros_like(ids)))
    imgs = render_views(structs, view, verbose=False)
    h = hashlib.sha1()
    for im in imgs:
        h.update(np.ascontiguousarray(im).tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class FeatureKey:
    corpus: str
    split: str
    view: str
    backbone: str

    def dir(self, root: Path | str = FEATURE_ROOT) -> Path:
        return Path(root) / self.corpus / self.split / self.view

    def path(self, root: Path | str = FEATURE_ROOT) -> Path:
        return self.dir(root) / f"{self.backbone}.npz"

    def meta_path(self, root: Path | str = FEATURE_ROOT) -> Path:
        return self.dir(root) / f"{self.backbone}_meta.json"


def structures_sha(structures: Sequence[Structure]) -> str:
    """Identity of the exact structure set, independent of file layout."""
    h = hashlib.sha1()
    for s in structures:
        h.update(np.ascontiguousarray(s.block_ids.astype(np.int32)).tobytes())
    return h.hexdigest()[:16]


def load_or_build(
    key: FeatureKey,
    structures: Sequence[Structure],
    view: ViewConfig = ViewConfig(),
    device: str = "cuda",
    root: Path | str = FEATURE_ROOT,
    force: bool = False,
    verbose: bool = True,
    check_canary: bool = True,
) -> np.ndarray:
    """Return `[N, V, D]` features, from cache when it is still valid."""
    path, meta_path = key.path(root), key.meta_path(root)
    want_sha = structures_sha(structures)
    canary = render_canary(view) if check_canary else "skipped"

    if path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            reasons = []
            if meta.get("structures_sha") != want_sha:
                reasons.append("structure set changed")
            if check_canary and meta.get("render_canary_sha") != canary:
                reasons.append("renderer output changed")
            if meta.get("n") != len(structures):
                reasons.append("count changed")
            if not reasons:
                with np.load(path) as z:
                    if verbose:
                        print(f"[features] cache hit {path}", flush=True)
                    return z["feats"]
            if verbose:
                print(f"[features] rebuilding {path.name}: {'; '.join(reasons)}",
                      flush=True)
        except Exception as exc:
            print(f"[features] unreadable cache {path} ({exc}); rebuilding")

    if verbose:
        print(f"[features] embedding {len(structures)} x {view.n_views} views "
              f"with {key.backbone}", flush=True)
    feats = embed_views(structures, view, key.backbone, device, verbose=verbose)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feats=feats.astype(np.float32))
    meta_path.write_text(json.dumps({
        "corpus": key.corpus, "split": key.split, "backbone": key.backbone,
        "model_id": BACKBONES[key.backbone][0], "n": len(structures),
        "view": {"views": [list(v) for v in view.views], "px": view.px,
                 "ortho": view.ortho, "bg": list(view.bg), "key": view.key()},
        "dim": int(feats.shape[-1]) if feats.size else 0,
        "structures_sha": want_sha, "render_canary_sha": canary,
        "git_sha": _git_sha(), "dtype": "float32",
    }, indent=2) + "\n")
    return feats


def _git_sha() -> str:
    from blockgen.eval.bench.scorecard import git_sha
    return git_sha()


def main() -> None:
    import argparse
    from blockgen.eval.bench import splits

    ap = argparse.ArgumentParser(description="Warm the feature cache.")
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--backbone", default="dinov2b", choices=sorted(BACKBONES))
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    view = ViewConfig(px=args.px)
    split = splits.load_split(args.corpus, args.seed)
    for which in args.splits:
        structs = splits.split_structures(split, which)
        key = FeatureKey(args.corpus, which, view.key(), args.backbone)
        feats = load_or_build(key, structs, view, device=args.device,
                              force=args.force)
        print(f"  {which}: {feats.shape} -> {key.path()}")


if __name__ == "__main__":
    main()
