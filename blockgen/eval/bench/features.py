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

#: The four-view orthographic protocol, (azimuth, elevation) in degrees. Equal to
#: `blockgen.eval.perceptual.DEFAULT_VIEWS`, restated here so the FAST tier can
#: import this module without torch, which that legacy module imports at the top.
DEFAULT_VIEWS = ((45.0, 30.0), (135.0, 30.0), (225.0, 30.0), (315.0, 30.0))
from blockgen.utils.data import Structure

FEATURE_ROOT = Path("data/minecraft/features")

#: name -> (model id, feature dimension). A `torchvision:` prefix selects the
#: torchvision path instead of HuggingFace; see `_load_backbone`.
BACKBONES: Dict[str, Tuple[str, int]] = {
    "dinov2b": ("facebook/dinov2-base", 768),
    "clipL": ("openai/clip-vit-large-patch14", 768),
    # Bit-identical to the model `blockgen/eval/perceptual.py` uses, kept so the
    # legacy CMMD numbers can be reproduced inside the new harness.
    "clipB": ("openai/clip-vit-base-patch32", 512),
    "siglipB": ("google/siglip-base-patch16-224", 768),
    # --- encoder-ablation backbones (T26) --------------------------------
    # These exist to answer one reviewer objection: "the interior blindness is a
    # DINOv2 artifact". They deliberately span the three supervision regimes the
    # literature uses -- self-supervised ViT (dinov2b), language-supervised ViT
    # (clipB16/clipB), label-supervised CNN (resnet50, inceptionv3) -- so a null
    # across all of them cannot be blamed on one pretraining objective.
    "clipB16": ("openai/clip-vit-base-patch16", 512),
    # ImageNet-supervised CNN features, the pre-ViT default for perceptual
    # metrics. Penultimate (post-global-pool) activations, 2048-d.
    "resnet50": ("torchvision:resnet50", 2048),
    # The classic FID backbone. torchvision's `inception_v3` with `fc` removed is
    # the 2048-d final average pool -- "pool3" in the original TF graph's naming.
    # Not bit-identical to `pytorch-fid`'s ported weights, so FID numbers here are
    # not comparable to published FIDs; the ablation only needs the same
    # architecture and receptive field, scored against its own noise floor.
    "inceptionv3": ("torchvision:inception_v3", 2048),
}

_MODELS: Dict[str, tuple] = {}


# --- cutaway geometry -------------------------------------------------------
# A cut is part of the *view protocol*, not a corruption of the build: it says
# which part of the structure the camera is allowed to see. It therefore lives
# here beside the cameras and not in `probes`, and it is applied identically to
# every arm -- reference, real and probe -- so the only thing that changes
# between two conditions is what the metric is shown.
_AXES = {"x": 0, "y": 1, "z": 2}


def apply_cut(s: Structure, spec: str) -> Structure:
    """Delete part of a build so the camera can see inside it. `""` is identity.

    Spec grammar (strings, so a `ViewConfig` stays hashable and its cache key
    stays readable):

    * ``half:<axis>:<lo|hi>``        -- keep that half of the bounding box
    * ``slab:<axis>:<pct>:<thick>``  -- keep a `thick`-voxel slab centred at
      `pct`% of the extent along `axis`: an actual cross-section
    * ``quad:<lo|hi>:<lo|hi>``       -- delete the (x, z) quadrant named, keeping
      the L-shaped remainder. Pair it with the camera that faces that corner.

    Why a *slab* is offered at all: F34.2 measured every quadrant and half-space
    cut and found the best of them moved only 4.25/255 mean absolute pixel
    difference against a 1.42 exterior baseline, because cutting a render still
    shows wall faces of the same materials from the same angle. A one-voxel slab
    viewed face-on is different in kind -- it is a picture of the floorplan, air
    reads as background, and filled rooms read as material -- which is the
    condition a positive control needs.

    **Empty cuts fall back to the uncut build.** A blank frame embeds to one
    constant vector and scores as a *perfect* distribution match (the failure
    `render_views` exists to shout about), so a cut that removes everything must
    not silently become one. `cut_empty_rate` reports how often this happens so
    the fallback is a measured number in the findings rather than a hidden one.
    """
    if not spec or spec == "none":
        return s
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    keep = keep_mask(occ.shape, spec)
    if not (keep & occ).any():
        return c
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    bi[~keep] = c.air_block_id
    bd[~keep] = 0
    return Structure(block_ids=bi, block_data=bd, air_block_id=c.air_block_id)


def keep_mask(shape: Tuple[int, int, int], spec: str) -> np.ndarray:
    """Which voxels of a cropped build survive `spec`. Pure geometry, no blocks.

    Split out of `apply_cut` so `cut_empty_rate` can ask "would this cut empty
    the build" without rebuilding a Structure, and so the two can never disagree
    about what a spec means.
    """
    keep = np.zeros(shape, dtype=bool)
    kind, *args = spec.split(":")

    def _span(axis: str, side: str) -> Tuple[int, int]:
        n = shape[_AXES[axis]]
        return (0, n // 2) if side == "lo" else (n - n // 2, n)

    if kind == "half":
        axis, side = args
        lo, hi = _span(axis, side)
        sel: List[slice] = [slice(None)] * 3
        sel[_AXES[axis]] = slice(lo, hi)
        keep[tuple(sel)] = True
    elif kind == "slab":
        axis, pct, thick = args[0], float(args[1]), int(args[2])
        a = _AXES[axis]
        n = shape[a]
        lo = int(round(pct / 100.0 * n)) - thick // 2
        lo = max(0, min(max(n - thick, 0), lo))
        sel = [slice(None)] * 3
        sel[a] = slice(lo, lo + thick)
        keep[tuple(sel)] = True
    elif kind == "quad":
        xside, zside = args
        keep[:] = True
        x0, x1 = _span("x", xside)
        z0, z1 = _span("z", zside)
        keep[x0:x1, :, z0:z1] = False
    else:
        raise ValueError(f"unknown cut spec {spec!r}")
    return keep


def cut_empty_rate(structures: Sequence[Structure], cuts: Sequence[str]) -> float:
    """Fraction of (structure, cut) pairs the cut would empty. See `apply_cut`.

    Reported beside every cutaway condition because those pairs silently fall
    back to the uncut build, which biases that condition *toward* the exterior
    null -- a high rate would make a null there uninterpretable.
    """
    if not cuts:
        return 0.0
    empty = total = 0
    for s in structures:
        occ = s.crop_to_non_air().occupied_mask
        for spec in cuts:
            if not spec or spec == "none":
                continue
            total += 1
            if not (keep_mask(occ.shape, spec) & occ).any():
                empty += 1
    return empty / max(total, 1)


@dataclass(frozen=True)
class ViewConfig:
    """Camera setup. Part of the cache key, because it changes every number."""
    views: Tuple[Tuple[float, float], ...] = DEFAULT_VIEWS
    px: int = 224
    ortho: bool = True
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: One `apply_cut` spec per view, or `()` for the exterior-only default.
    #: Per-view rather than global because a cutaway only reveals anything when
    #: the removed volume faces *that* camera.
    cuts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cuts and len(self.cuts) != len(self.views):
            raise ValueError(
                f"cuts must be empty or one per view, got {len(self.cuts)} cuts "
                f"for {len(self.views)} views")

    def key(self) -> str:
        az = "-".join(f"{a:g}" for a, _ in self.views)
        el = "-".join(f"{e:g}" for _, e in self.views)
        key = (f"v{len(self.views)}_az{az}_el{el}_px{self.px}"
               f"_{'ortho' if self.ortho else 'persp'}")
        # Appended only when non-empty, so every key written before cutaways
        # existed still names the same condition and its cache stays valid.
        if self.cuts:
            digest = hashlib.sha1("|".join(self.cuts).encode()).hexdigest()[:8]
            key += f"_cut{digest}"
        return key

    @property
    def n_views(self) -> int:
        return len(self.views)


def _load_torchvision(model_id: str, device: str):
    """An ImageNet-supervised CNN with its classifier removed, plus its own transform.

    Preprocessing comes from the checkpoint's `weights.transforms()` rather than
    from a hand-written resize, because the whole point of a supervised-CNN arm
    is to reproduce *that* backbone's usual input pipeline -- 299px and Inception
    normalization for inception_v3, 232/224 and ImageNet statistics for
    resnet50. Getting this wrong would let a null be dismissed as "you fed it
    the wrong pixels".
    """
    import torch
    import torchvision

    arch = model_id.split(":", 1)[1]
    weights = {
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "inception_v3": torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
    }[arch]
    model = getattr(torchvision.models, arch)(weights=weights)
    # Penultimate features: drop the 1000-way classifier, keep the global pool.
    model.fc = torch.nn.Identity()
    if hasattr(model, "aux_logits"):        # inception's aux head is eval-inert
        model.aux_logits = False
        model.AuxLogits = None
    return model.to(device).eval(), weights.transforms()


def _load_backbone(name: str, device: str = "cuda"):
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    if name not in _MODELS:
        model_id = BACKBONES[name][0]
        if model_id.startswith("torchvision:"):
            _MODELS[name] = _load_torchvision(model_id, device)
            return _MODELS[name]
        from transformers import AutoImageProcessor, AutoModel, CLIPModel
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
        for j, (azim, elev) in enumerate(view.views):
            try:
                shown = apply_cut(s, view.cuts[j]) if view.cuts else s
                img = render_structure(shown, px=view.px, azim_deg=azim,
                                       elev_deg=elev,
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
    is_tv = BACKBONES[backbone][0].startswith("torchvision:")
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [np.asarray(im) for im in images[i:i + batch]]
            if is_tv:
                # torchvision presets take uint8 NCHW and do their own resize and
                # normalization; the CNN then returns pooled features directly.
                raw = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2)
                f = model(proc(raw).to(device))
            else:
                px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device)
                if hasattr(model, "get_image_features"):    # CLIP / SigLIP
                    f = model.get_image_features(pixel_values=px)
                else:                                       # DINOv2 -> CLS token
                    f = model(pixel_values=px).last_hidden_state[:, 0]
            # Every backbone is L2-normalized before it reaches a kernel. Raw CNN
            # activations have a norm an order of magnitude larger than a
            # projected CLIP embedding, and KID's default gamma=1/D would then
            # mean something different per arm -- the comparison across encoders
            # would be a comparison of feature scales.
            f = torch.nn.functional.normalize(f.flatten(1), dim=-1).float().cpu().numpy()
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


#: A transparent block, and the opaque one we park behind it. The canary has to
#: contain both: whether a face behind glass is drawn is a renderer decision
#: that no opaque test scene can observe, and on 2026-09-25 that blind spot let
#: a real change through. `textured._exposed_masks` culled a face whenever its
#: neighbour was occupied, transparent or not, so filling the cell behind a
#: window made the window show the background instead of the fill. Fixing it
#: changed every textured render of every build with a window, and the canary
#: fingerprint did not move, because the scene below had no glass in it. The
#: cache stayed warm and a re-scored leaderboard came back mixing old reference
#: features with new arm features.
CANARY_GLASS = 102
CANARY_BACKING = 5


def render_canary(view: ViewConfig, n: int = 4) -> str:
    """Fingerprint of a fixed render, so a changed texture pack invalidates.

    Without this the cache is keyed only on things we *declare* (model id, view
    config) and not on the renderer's actual output. A silently updated texture
    pack or renderer change would otherwise poison every cached number
    permanently, and the poisoning would be invisible.

    The scene deliberately exercises three renderer decisions rather than one:
    an opaque surface, a stray block of a third material, and a transparent
    block with an opaque one directly behind it.
    """
    rng = np.random.default_rng(CANARY_SEED)
    structs = []
    for k in range(n):
        ids = np.zeros((4 + k, 4, 4), dtype=np.int32)
        ids[:, 0, :] = 4
        ids[0, :, :] = 5
        ids[rng.integers(0, 4 + k), rng.integers(0, 4), rng.integers(0, 4)] = 98
        # glass with something solid behind it, so the cull rule is fingerprinted
        ids[2, 2, 2] = CANARY_GLASS
        ids[2, 2, 3] = CANARY_BACKING
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
                 "ortho": view.ortho, "bg": list(view.bg),
                 "cuts": list(view.cuts), "key": view.key()},
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
