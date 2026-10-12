"""Measure the interior-blindness decomposition claimed in `math/math_hull.md`.

Four things are checked, in the order the appendix states them:

* **L1 (exact-class demo).** A synthetic pair (opaque build, same build with its
  sealed pocket filled) is rendered and embedded with three backbones. Renders
  bit-identical => features bit-identical, for every encoder. This is the
  trivial lemma made concrete.
* **L2 (surface invariance of `solidify`).** The set of (cell, direction,
  material) faces bordering exterior air is verified equal for `x` and
  `solidify(x)` on every held-out build -- the premise of the occlusion lemma.
* **B (the decomposition).** Over the held-out split, per (build, view): is the
  render bit-identical, and if not, how many pixels move and by how much. Run
  under four pipelines -- the deployed one and three counterfactuals -- because
  the deployed pipeline has *two* leaks and they are not the same kind of thing:
  light transmitted through translucent blocks (real interior information) and
  the texture-atlas repack triggered by the interior faces disappearing from the
  mesh (an implementation artefact carrying no information at all).
* **C (where the leak dies).** DINOv2 CLS vs patch tokens: how far the touched
  patch tokens move compared to the pooled CLS vector, each against the
  between-build scale. Empirical, not a theorem.
* **D (the matching lemma).** The V-KID descriptor is verified bit-identical
  under `shuffle_materials` and `monochrome`, while the renders of the same
  builds are verified to change.

Run: `.venv/bin/python scripts/figs/math_hull_check.py`
Writes `outputs/figstage/math_hull_check.json` and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from blockgen.eval.bench import features as ft
from blockgen.eval.bench import geometry as geo
from blockgen.eval.bench import probes, splits
from blockgen.renderer.textures import DEFAULT_PACK_DIR as _DEFAULT_PACK
from blockgen.utils.data import Structure

OUT = Path("outputs/figstage/math_hull_check.json")
STRUCT6 = ndimage.generate_binary_structure(3, 1)
#: (name, (dx, dy, dz)), matching `renderer.textured._DIRECTIONS`.
DIRS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("x+", (1, 0, 0)), ("x-", (-1, 0, 0)),
    ("y+", (0, 1, 0)), ("y-", (0, -1, 0)),
    ("z+", (0, 0, 1)), ("z-", (0, 0, -1)),
)


# --- render paths -----------------------------------------------------------
#: An empty pack dir makes every block fall back to a 2x2 solid-colour tile --
#: the `flat_textures()` idiom in `scripts/paper_figures.py`, copied per the brief.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def _opaque_library(pack_dir: str):
    """A texture library with every alpha channel forced to 255."""
    from blockgen.renderer.textures import FaceTextureLibrary

    class _Opaque(FaceTextureLibrary):
        def __missing__(self, key):
            faces = super().__missing__(key)
            out = {}
            for k, arr in faces.items():
                a = np.asarray(arr).copy()
                a[..., 3] = 255
                out[k] = a
            self[key] = out
            return out

    return _Opaque(pack_dir=pack_dir)


def pipelines() -> Dict[str, object]:
    """The deployed render path and three counterfactuals that isolate its leaks.

    Only two things can make `solidify` change a pixel (Lemma 2 rules out
    geometry): light transmitted through a translucent block, and the
    texture-atlas repack that `build_mesh` performs when the *interior* faces it
    also meshes stop existing. `flat` removes the second (every tile is one
    2x2 solid colour, so no UV rounding can change a texel), `opaque` removes
    the first, `flat_opaque` removes both and is the lemma's own hypothesis.
    """
    from blockgen.renderer.textures import load_face_textures
    return {
        "paper": load_face_textures(),                 # the deployed pipeline
        "opaque": _opaque_library(_DEFAULT_PACK),      # atlas artefact only
        "flat": load_face_textures(pack_dir=_FLAT_DIR),  # transmission only
        "flat_opaque": _opaque_library(_FLAT_DIR),     # neither: Lemma 2's setting
    }


def atlas_layout(c: Structure, tex) -> dict:
    """The atlas `renderer.textured.build_mesh` would pack for this build.

    Tiles are keyed by `(id(texture array), face brightness)` and numbered in
    first-appearance order over `_DIRECTIONS` and then voxel order -- a loop
    that runs over *every* exposed face, interior ones included. So deleting the
    faces that bordered a sealed pocket renumbers the atlas, moves every tile's
    UV rect, and at NEAREST filtering flips the texel sampled at some pixels of
    surfaces that did not change at all.
    """
    from blockgen.renderer.textured import _DIRECTIONS, _exposed_masks

    occ = c.occupied_mask
    masks = _exposed_masks(occ)
    ids, data = c.block_ids, c.block_data
    seen: Dict[tuple, int] = {}
    sizes: List[int] = []
    for name, _, kind, bright in _DIRECTIONS:
        for x, y, z in np.argwhere(masks[name]):
            t = tex[(int(ids[x, y, z]), int(data[x, y, z]))][kind]
            key = (id(t), bright)
            if key not in seen:
                seen[key] = len(sizes)
                sizes.append(int(np.asarray(t).shape[0]))
    n = len(sizes)
    tile_px = max(max(sizes) if sizes else 2, 2)
    cols = int(np.ceil(np.sqrt(n))) if n else 0
    return {"n_tiles": n, "tile_px": tile_px, "cols": cols,
            "order": tuple(seen.keys())}


def render_with(structures: Sequence[Structure], view: ft.ViewConfig, tex
                ) -> List[np.ndarray]:
    """`features.render_views` with an explicit texture library.

    Duplicated (not imported) only so the texture library can be swapped and so
    one library instance serves every render; `selfcheck_render_path` pins this
    to `features.render_views` bit for bit.
    """
    from blockgen.renderer.textured import render_structure

    out: List[np.ndarray] = []
    bg255 = np.array(view.bg, dtype=np.float32) * 255.0
    for s in structures:
        for j, (azim, elev) in enumerate(view.views):
            shown = ft.apply_cut(s, view.cuts[j]) if view.cuts else s
            img = render_structure(shown, px=view.px, azim_deg=azim,
                                   elev_deg=elev, ortho=view.ortho,
                                   face_textures=tex)
            if img.shape[-1] == 4:
                a = img[..., 3:4].astype(np.float32) / 255.0
                img = (img[..., :3] * a + bg255 * (1 - a)).astype(np.uint8)
            out.append(np.ascontiguousarray(img[..., :3]))
    return out


def selfcheck_render_path(structs: Sequence[Structure], tex) -> dict:
    """`render_with` must equal `features.render_views` pixel for pixel."""
    view = ft.ViewConfig()
    a = ft.render_views(list(structs), view, verbose=False)
    b = render_with(list(structs), view, tex)
    same = all(np.array_equal(u, v) for u, v in zip(a, b))
    # Renderer determinism: the same call twice must give the same pixels, or no
    # bit-identity statement below means anything.
    c = render_with(list(structs), view, tex)
    return {"matches_features_render_views": bool(same),
            "deterministic": bool(all(np.array_equal(u, v) for u, v in zip(b, c))),
            "n_views_checked": len(a)}


# --- pixel-space difference -------------------------------------------------
def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference between two renders, in the units `scripts/bench_hollow.py` uses."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = d.max(axis=-1) > 0
    return {
        "identical": bool(not changed.any()),
        "frac_px_changed": float(changed.mean()),
        "mean_abs_all_px": float(d.mean()),                 # T36's whole-frame mean
        "max_abs": float(d.max()),
        "mean_abs_where_changed": (float(d.max(axis=-1)[changed].mean())
                                   if changed.any() else 0.0),
    }


# --- geometry predicates ----------------------------------------------------
def filled_cells(occ: np.ndarray) -> np.ndarray:
    """The cells `probes.solidify` adds: air sealed off from the outside."""
    return ndimage.binary_fill_holes(occ, structure=STRUCT6) & ~occ


def translucent_mask(c: Structure, tex) -> np.ndarray:
    """Occupied cells whose block has any texel with alpha < 255.

    Exactly the test `renderer.textured.build_mesh` applies to decide whether a
    face goes into the BLEND primitive, so this is the renderer's own notion of
    "light gets through", not an ontology guess.
    """
    occ = c.occupied_mask
    out = np.zeros_like(occ)
    ids, data = c.block_ids, c.block_data
    keys = np.unique(np.stack([ids[occ], data[occ]], 1), axis=0) if occ.any() else []
    for bid, bdata in keys:
        faces = tex[(int(bid), int(bdata))]
        if any(bool((np.asarray(t)[..., 3] < 255).any()) for t in faces.values()):
            out |= occ & (ids == bid) & (data == bdata)
    return out


def exterior_face_set(c: Structure) -> set:
    """(cell, direction, block id, block data) for every face on the outside air.

    The rendered surface: `renderer.textured._exposed_masks` emits a quad per
    air-bordering face, and of those only the ones bordering air that is
    6-connected to outside the bounding box can be reached by a ray.
    """
    occ = c.occupied_mask
    ext = np.pad(probes.exterior_air(occ), 1, constant_values=True)
    faces = set()
    for name, (dx, dy, dz) in DIRS:
        nb = ext[1 + dx: ext.shape[0] - 1 + dx,
                 1 + dy: ext.shape[1] - 1 + dy,
                 1 + dz: ext.shape[2] - 1 + dz]
        for x, y, z in np.argwhere(occ & nb):
            faces.add((int(x), int(y), int(z), name,
                       int(c.block_ids[x, y, z]), int(c.block_data[x, y, z])))
    return faces


def geom_facts(c: Structure, solid: Structure, tex) -> dict:
    occ = c.occupied_mask
    add = filled_cells(occ)
    trans = translucent_mask(c, tex)
    touching = ndimage.binary_dilation(add, STRUCT6) & occ
    a, b = atlas_layout(c, tex), atlas_layout(solid, tex)
    return {
        "n_blocks": int(occ.sum()),
        "n_filled": int(add.sum()),
        "mass_increase": float(add.sum() / max(int(occ.sum()), 1)),
        "any_translucent_block": bool(trans.any()),
        "translucent_touches_pocket": bool((trans & touching).any()),
        "atlas_unchanged": bool(a["order"] == b["order"]),
        "n_tiles": a["n_tiles"], "n_tiles_solid": b["n_tiles"],
    }


# --- part A: the exact class, constructively ---------------------------------
def synthetic_pair() -> Tuple[Structure, Structure]:
    """An opaque box with a sealed 3x3x3 room, and the same box solidified."""
    n = 7
    ids = np.full((n, n, n), 4, dtype=np.int32)      # cobblestone shell, opaque
    ids[2:5, 2:5, 2:5] = 0                          # sealed air pocket
    ids[:, 0, :] = 1                                # a second material, floor
    x = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    return x, probes.solidify(x)


def lemma1_demo(device: str, backbones=("dinov2b", "clipB", "resnet50")) -> dict:
    x, xs = synthetic_pair()
    view = ft.ViewConfig()
    ia = ft.render_views([x], view, verbose=False)
    ib = ft.render_views([xs], view, verbose=False)
    pixels_identical = all(np.array_equal(u, v) for u, v in zip(ia, ib))
    out = {"filled_cells": int(filled_cells(x.crop_to_non_air().occupied_mask).sum()),
           "pixels_identical": bool(pixels_identical), "encoders": {}}
    for b in backbones:
        fa = ft.embed_images(ia, b, device)
        fb = ft.embed_images(ib, b, device)
        out["encoders"][b] = {
            "features_bit_identical": bool(np.array_equal(fa, fb)),
            "max_abs_feature_diff": float(np.abs(fa - fb).max()),
        }
    return out


# --- part B: the decomposition ----------------------------------------------
def decomposition(pool: Sequence[Structure], view: ft.ViewConfig, tex,
                  batch: int = 16, label: str = "") -> dict:
    """Per (build, view) bit-identity and change size, plus the geometry predicates."""
    rows: List[dict] = []
    t0 = time.time()
    for i in range(0, len(pool), batch):
        chunk = list(pool[i:i + batch])
        solid = [probes.solidify(s) for s in chunk]
        ra = render_with(chunk, view, tex)
        rb = render_with(solid, view, tex)
        v = view.n_views
        for k, s in enumerate(chunk):
            facts = geom_facts(s.crop_to_non_air(), solid[k], tex)
            for j in range(v):
                d = diff_stats(ra[k * v + j], rb[k * v + j])
                d.update(facts)
                d["build"] = i + k
                d["view"] = j
                rows.append(d)
        print(f"    [{label}] {min(i + batch, len(pool))}/{len(pool)} builds "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {"view_key": view.key(), "n_views": view.n_views,
            "n_builds": len(pool), "rows": rows}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    changed = [r for r in rows if not r["identical"]]
    ident = np.array([r["identical"] for r in rows])
    v = res["n_views"]
    per_build = ident.reshape(-1, v)
    has_pocket = np.array([r["n_filled"] > 0 for r in rows])
    tp = np.array([r["translucent_touches_pocket"] for r in rows])
    out = {
        "n_pairs": len(rows),
        "bit_identical_frac": float(ident.mean()),
        "builds_all_views_identical_frac": float(per_build.all(1).mean()),
        "builds_no_view_identical_frac": float((~per_build).all(1).mean()),
        "pairs_with_pocket_frac": float(has_pocket.mean()),
        # Conditioned on the probe actually doing something: a build with no
        # sealed pocket is unchanged by solidify and is trivially identical.
        "bit_identical_frac_given_pocket": float(ident[has_pocket].mean()),
        "translucent_touches_pocket_frac": float(tp.mean()),
    }
    at = np.array([r["atlas_unchanged"] for r in rows])
    out["atlas_unchanged_frac"] = float(at.mean())
    # The mechanism test: under an opaque pipeline the render should be
    # bit-identical exactly when the atlas is not repacked.
    out["atlas_predicts_identity"] = float((at == ident).mean())
    # Which way the predicate fails matters: "atlas repacked yet the render is
    # identical" is benign (the moved tiles are not visible from that camera);
    # "atlas intact yet the render changed" would mean a third leak channel.
    out["n_atlas_unchanged_render_changed"] = int((at & ~ident).sum())
    out["n_atlas_changed_render_identical"] = int((~at & ident).sum())
    exc = [r for r, keep in zip(rows, at & ~ident) if keep]
    out["atlas_unchanged_render_changed_frac_px"] = sorted(
        round(r["frac_px_changed"], 6) for r in exc)[:12]
    out["atlas_unchanged_render_changed_max_abs"] = sorted(
        r["max_abs"] for r in exc)[:12]
    for name, sel in (("no_translucent_on_pocket", has_pocket & ~tp),
                      ("translucent_on_pocket", has_pocket & tp),
                      ("atlas_unchanged", has_pocket & at),
                      ("atlas_unchanged_and_no_translucent",
                       has_pocket & at & ~tp)):
        out[f"bit_identical_frac_{name}"] = (float(ident[sel].mean())
                                            if sel.any() else None)
        out[f"n_pairs_{name}"] = int(sel.sum())
    if changed:
        fp = np.array([r["frac_px_changed"] for r in changed])
        ma = np.array([r["mean_abs_all_px"] for r in changed])
        mx = np.array([r["max_abs"] for r in changed])
        mw = np.array([r["mean_abs_where_changed"] for r in changed])
        out["changed"] = {
            "n": len(changed),
            "frac_px_median": float(np.median(fp)),
            "frac_px_mean": float(fp.mean()),
            "frac_px_p90": float(np.percentile(fp, 90)),
            "frac_px_max": float(fp.max()),
            "mean_abs_all_px_median": float(np.median(ma)),
            "max_abs_median": float(np.median(mx)),
            "mean_abs_where_changed_median": float(np.median(mw)),
        }
    mi = np.array([r["mass_increase"] for r in rows if r["view"] == 0])
    out["mass_increase_median"] = float(np.median(mi))
    out["builds_with_pocket_frac"] = float((mi > 0).mean())
    return out


def surface_invariance(pool: Sequence[Structure]) -> dict:
    """L2's premise, on every build: solidify preserves the exterior face set."""
    bad_solid = bad_hollow = 0
    nf = []
    for s in pool:
        c = s.crop_to_non_air()
        f = exterior_face_set(c)
        nf.append(len(f))
        if exterior_face_set(probes.solidify(c)) != f:
            bad_solid += 1
        if exterior_face_set(probes.hollow(c)) != f:
            bad_hollow += 1
    return {"n_builds": len(pool), "solidify_violations": bad_solid,
            "hollow_violations": bad_hollow,
            "median_exterior_faces": float(np.median(nf))}


# --- part C: where in the encoder the change dies ---------------------------
def dino_tokens(images: Sequence[np.ndarray], device: str, batch: int = 32):
    """L2-normalized CLS and patch tokens from the paper's DINOv2 backbone."""
    import torch

    model, proc = ft._load_backbone("dinov2b", device)
    cls, patch = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [np.asarray(im) for im in images[i:i + batch]]
            px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device)
            h = model(pixel_values=px).last_hidden_state
            cls.append(torch.nn.functional.normalize(h[:, 0], dim=-1).cpu().numpy())
            patch.append(torch.nn.functional.normalize(h[:, 1:], dim=-1).cpu().numpy())
    return np.concatenate(cls), np.concatenate(patch)


def patch_grid_mask(changed: np.ndarray, side: int, out_px: int, resize: int
                    ) -> np.ndarray:
    """Push a 224x224 change mask through resize-then-centre-crop to patch cells.

    Approximate by construction (nearest-neighbour stand-in for the processor's
    bicubic resize); it is used only to label which tokens the change touches.
    """
    idx = np.clip(((np.arange(resize) + 0.5) * changed.shape[0] / resize).astype(int),
                  0, changed.shape[0] - 1)
    big = changed[np.ix_(idx, idx)]
    off = (resize - out_px) // 2
    crop = big[off:off + out_px, off:off + out_px]
    ps = out_px // side
    return crop[:side * ps, :side * ps].reshape(side, ps, side, ps).any(axis=(1, 3))


def part_c(pool: Sequence[Structure], tex, device: str, batch: int = 16,
           label: str = "") -> dict:
    view = ft.ViewConfig()
    v = view.n_views
    cls_r, cls_s, masks, view_changed = [], [], [], []
    pr_touch, pr_untouch, pr_between = [], [], []
    side = None
    for i in range(0, len(pool), batch):
        chunk = list(pool[i:i + batch])
        solid = [probes.solidify(s) for s in chunk]
        ra = render_with(chunk, view, tex)
        rb = render_with(solid, view, tex)
        ca, pa = dino_tokens(ra, device)
        cb, pb = dino_tokens(rb, device)
        if side is None:
            side = int(round(np.sqrt(pa.shape[1])))
        cls_r.append(ca)
        cls_s.append(cb)
        # Between-build token scale, computed inside the batch so the 200 MB of
        # patch tokens never has to be held: build k against build k+1 at the
        # same view index, and the pool order is already a random permutation.
        nb = len(chunk)
        for k in range(nb):
            for j in range(v):
                other = ((k + 1) % nb) * v + j
                if other == k * v + j:
                    continue
                tok = 1.0 - (pa[k * v + j] * pa[other]).sum(-1)
                pr_between.append(float(tok.mean()))
        for k in range(len(ra)):
            d = np.abs(ra[k].astype(np.int16) - rb[k].astype(np.int16)).max(-1) > 0
            m = patch_grid_mask(d, side, 224, 256).reshape(-1)
            masks.append(m)
            view_changed.append(bool(d.any()))
            tok = 1.0 - (pa[k] * pb[k]).sum(-1)          # per-token cosine distance
            if m.any():
                pr_touch.append(float(tok[m].mean()))
            if (~m).any():
                pr_untouch.append(float(tok[~m].mean()))
        print(f"    [part C {label}] {min(i + batch, len(pool))}/{len(pool)} builds",
              flush=True)
    cls_r = np.concatenate(cls_r)
    cls_s = np.concatenate(cls_s)
    d_cls = 1.0 - (cls_r * cls_s).sum(-1)

    # Between-build scale, same view index, so the comparison is not confounded
    # by camera azimuth.
    rng = np.random.default_rng(0)
    n = len(pool)
    between = []
    for j in range(v):
        f = cls_r[j::v]
        a = rng.permutation(n)
        b = (a + 1 + rng.integers(0, n - 1, size=n)) % n
        between.append(1.0 - (f[a] * f[b]).sum(-1))
    d_between = np.concatenate(between)

    # What KID actually reads: the mean-pooled, renormalized per-build vector.
    pool_r = ft.pooled(cls_r.reshape(n, v, -1))
    pool_s = ft.pooled(cls_s.reshape(n, v, -1))
    d_pool = 1.0 - (pool_r * pool_s).sum(-1)
    a = rng.permutation(n)
    b = (a + 1 + rng.integers(0, n - 1, size=n)) % n
    d_pool_between = 1.0 - (pool_r[a] * pool_r[b]).sum(-1)

    touched = np.array([m.mean() for m in masks])
    return {
        "n_builds": n, "n_views": v, "patch_grid": side,
        "frac_tokens_touched_median": float(np.median(touched)),
        "frac_tokens_touched_mean": float(touched.mean()),
        "token_dist_touched_median": float(np.median(pr_touch)) if pr_touch else None,
        "token_dist_untouched_median": float(np.median(pr_untouch)) if pr_untouch else None,
        "token_dist_between_builds_median": (float(np.median(pr_between))
                                             if pr_between else None),
        "cls_dist_median": float(np.median(d_cls)),
        # Restricted to views the corruption actually altered: under a pipeline
        # where most views come back bit-identical the unrestricted median is
        # just float epsilon and says nothing.
        "views_changed_frac": float(np.mean(view_changed)),
        "cls_dist_median_where_changed": (
            float(np.median(d_cls[np.array(view_changed)]))
            if any(view_changed) else None),
        "cls_dist_between_builds_median": float(np.median(d_between)),
        "pooled_dist_median": float(np.median(d_pool)),
        "pooled_dist_between_builds_median": float(np.median(d_pool_between)),
    }


# --- part D: the matching lemma ---------------------------------------------
def part_d(pool: Sequence[Structure], ref: Sequence[Structure], tex,
           seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = [s.crop_to_non_air() for s in pool]
    arms = {"material_shuffle": [probes.shuffle_materials(s, rng) for s in base],
            "monochrome": [probes.monochrome(s) for s in base]}
    d_real = geo.descriptor(base)
    std = geo.Standardizer.fit(geo.descriptor(list(ref)))
    f_real = geo.geom_features(base, std)
    kid_real = geo.geom_kid(f_real, geo.geom_features(list(ref), std))
    out = {"n_builds": len(base), "n_ref": len(ref), "arms": {}}
    view = ft.ViewConfig()
    real_imgs = render_with(base, view, tex)
    for name, structs in arms.items():
        d = geo.descriptor(structs)
        f = geo.geom_features(structs, std)
        kid = geo.geom_kid(f, geo.geom_features(list(ref), std))
        imgs = render_with(structs, view, tex)
        st = [diff_stats(a, b) for a, b in zip(real_imgs, imgs)]
        out["arms"][name] = {
            "descriptor_bit_identical": bool(np.array_equal(d_real, d)),
            "max_abs_descriptor_diff": float(np.abs(d_real - d).max()),
            "geom_kid": float(kid),
            "geom_kid_equals_real_exactly": bool(kid == kid_real),
            "renders_identical_frac": float(np.mean([s["identical"] for s in st])),
            "render_frac_px_changed_median": float(
                np.median([s["frac_px_changed"] for s in st])),
            "render_mean_abs_all_px_median": float(
                np.median([s["mean_abs_all_px"] for s in st])),
        }
    out["geom_kid_real"] = float(kid_real)
    return out


# --- a picture of the two leak channels -------------------------------------
def corpus_mask(n: int, corpus: str = "3dcraft") -> List[bool]:
    """Which test-split rows may be shown as images. Copied from `paper_figures`.

    Only 3D-Craft builds are publishable as pictures: GrabCraft's terms name
    photographs of blueprints as prohibited distribution, and text2mc's builds
    come from PlanetMinecraft.
    """
    man = json.loads(Path("data/minecraft/cache/houses_32_manifest.json").read_text())
    rows = list(splits.load_split("houses_32").indices("test"))
    return [man["items"][rows[i]].get("corpus") == corpus for i in range(n)]


def dump_example(test: Sequence[Structure], order: np.ndarray, libs: dict,
                 path: Path) -> dict:
    """Real | solidified | paper change mask | opaque-only (atlas) change mask.

    The panel exists to show that the two masks are different *kinds* of thing:
    the atlas mask is one-texel seams along block edges everywhere, the extra
    pixels in the paper mask are blobs behind windows.
    """
    from PIL import Image

    allowed = corpus_mask(len(test))
    view = ft.ViewConfig()
    best = None
    for idx in order[:64]:
        if not allowed[int(idx)]:
            continue
        s = test[int(idx)]
        sol = probes.solidify(s)
        a = render_with([s], view, libs["paper"])
        b = render_with([sol], view, libs["paper"])
        oa = render_with([s], view, libs["opaque"])
        ob = render_with([sol], view, libs["opaque"])
        for j in range(view.n_views):
            frac = diff_stats(a[j], b[j])["frac_px_changed"]
            if best is None or frac > best[0]:
                best = (frac, int(idx), j, a[j], b[j], oa[j], ob[j])
    if best is None:
        return {"written": False}
    frac, idx, j, a, b, oa, ob = best
    m_paper = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 0)
    m_atlas = (np.abs(oa.astype(np.int16) - ob.astype(np.int16)).max(-1) > 0)
    panel = np.concatenate([a, b,
                            np.repeat((m_paper * 255).astype(np.uint8)[..., None], 3, 2),
                            np.repeat((m_atlas * 255).astype(np.uint8)[..., None], 3, 2)],
                           axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(path)
    return {"written": True, "path": str(path), "test_index": idx, "view": j,
            "frac_px_paper": float(m_paper.mean()),
            "frac_px_atlas_only": float(m_atlas.mean()),
            "corpus": "3dcraft"}


# --- main -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256, help="builds for the 4-view arm")
    ap.add_argument("--n-wide", type=int, default=128,
                    help="builds for the 8- and 16-view arms")
    ap.add_argument("--n-c", type=int, default=128, help="builds for part C")
    ap.add_argument("--n-d", type=int, default=64, help="builds for part D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    split = splits.load_split(args.corpus, args.seed)
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    val = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(test))
    pool = [test[i] for i in order[:args.n]]
    print(f"[hull] test {len(test)} builds, using {len(pool)}", flush=True)

    libs = pipelines()
    tex = libs["paper"]

    res: Dict[str, object] = {"meta": {
        "corpus": args.corpus, "seed": args.seed,
        "split_key": split.key(),
        "n_test": len(test), "n_val": len(val), "device": args.device,
        "views_default": [list(v) for v in ft.ViewConfig().views],
        "px": ft.ViewConfig().px,
    }}

    print("[hull] self-check: render path identical to features.render_views")
    res["selfcheck"] = selfcheck_render_path(pool[:4], tex)
    print("   ", res["selfcheck"], flush=True)

    print("[hull] L1: synthetic exactly-invisible pair")
    res["lemma1_demo"] = lemma1_demo(args.device)
    print("   ", json.dumps(res["lemma1_demo"]), flush=True)

    print("[hull] L2: exterior face-set invariance on every held-out build")
    res["surface_invariance"] = surface_invariance(test)
    print("   ", res["surface_invariance"], flush=True)

    v4 = ft.ViewConfig()
    for name, lib in libs.items():
        print(f"[hull] B: decomposition, {name} pipeline, 4 views")
        res[f"B_{name}_v4"] = summarize(decomposition(pool, v4, lib,
                                                      label=f"{name}/4"))

    wide = pool[:args.n_wide]
    az8 = tuple((a, 30.0) for a in (0, 45, 90, 135, 180, 225, 270, 315))
    v8 = ft.ViewConfig(views=az8)
    v16 = ft.ViewConfig(views=az8 + tuple((a, e) for e in (75.0, -45.0)
                                          for a in (0, 90, 180, 270)))
    for name, vc in (("B_paper_v8", v8), ("B_paper_v16", v16)):
        print(f"[hull] B: {name}")
        res[name] = summarize(decomposition(wide, vc, tex, label=name))

    for name in ("paper", "flat"):
        print(f"[hull] C: CLS vs patch tokens, {name} pipeline")
        res[f"C_tokens_{name}"] = part_c(pool[:args.n_c], libs[name],
                                         args.device, label=name)
        print("   ", json.dumps(res[f"C_tokens_{name}"]), flush=True)

    print("[hull] example panel (3D-Craft builds only, licensing gate)")
    res["example_panel"] = dump_example(test, order, libs,
                                        Path("outputs/figstage/math_hull_leak.png"))
    print("   ", res["example_panel"], flush=True)

    print("[hull] D: V-KID material invariance")
    res["D_matching_lemma"] = part_d(pool[:args.n_d], val, tex, args.seed)
    print("   ", json.dumps(res["D_matching_lemma"], indent=None), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n[hull] wrote {out}")

    print("\n--- solidify vs real, 4 views, by render pipeline "
          f"(n={res['B_paper_v4']['n_pairs']} build-view pairs) ---")
    print(f"{'pipeline':14s} {'bit-ident':>9s} {'|pocket':>8s} "
          f"{'px median':>10s} {'max/255':>8s}  atlas->identity")
    for name in libs:
        s = res[f"B_{name}_v4"]
        ch = s.get("changed", {})
        print(f"{name:14s} {s['bit_identical_frac']:9.3f} "
              f"{s['bit_identical_frac_given_pocket']:8.3f} "
              f"{ch.get('frac_px_median', 0.0):10.4f} "
              f"{ch.get('max_abs_median', 0.0):8.0f}  "
              f"{s['atlas_predicts_identity']:.3f}")
    for name in ("B_paper_v8", "B_paper_v16"):
        s = res[name]
        print(f"{name:14s} {s['bit_identical_frac']:9.3f} "
              f"{s['bit_identical_frac_given_pocket']:8.3f}  "
              f"(all views identical: {s['builds_all_views_identical_frac']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
