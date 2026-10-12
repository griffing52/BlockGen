"""What in V-KID actually detects interior filling, and is it just counting blocks?

Two questions a reviewer will ask about a handcrafted 77-dimensional descriptor,
answered by measurement rather than by assertion.

**1. Which components carry the result?** The descriptor is four blocks: 55
D4-orbit occupancy patterns, 5 wall-thickness bins, 8 height bins, and 9 global
scalars. Two of those scalars are uncomfortable on their own terms:
`interior_ratio_sealed` measures the very thing `Fill-Interior` destroys, and
`log1p(n_blocks)` is pure size. If either carries the separation by itself,
V-KID detects the probe almost tautologically and the structural claim is weak.
So we score each block alone, each block removed, and the two suspect scalars
removed.

**2. Is it a size detector?** `Fill-Interior` adds mass, so a metric that only
counted blocks would separate it. The control is `Grow-Shell`: dilate the solid
outward until it has added *exactly the same number of blocks* the probe adds to
that build, leaving the interior untouched. Same mass, same material, same
build; rooms intact rather than destroyed. A size detector scores the two alike.
A structural metric does not.

`Grow-Shell` also gives a double dissociation, because it changes the silhouette
that `Fill-Interior` leaves untouched: the render metric should see the control
and miss the probe, and the geometric metric the reverse.

Every separation is in that descriptor variant's own noise floor, measured in
the same run from the same draws, because floors are not comparable across
variants any more than across runs.

Run: `.venv/bin/python scripts/vkid_ablation.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N, REPS, SEED = 128, 48, 0
OUT = ROOT / "outputs" / "figstage" / "vkid_ablation.json"


def grow_shell(s, n_add: int, rng):
    """Dilate the solid outward by exactly `n_add` cells; interior untouched.

    Grows in distance order from the existing surface, so the added mass is a
    thickened shell rather than scattered noise, and ties at equal distance are
    broken randomly so the control is not itself a tidy geometric object.
    """
    from scipy import ndimage
    occ = s.occupied_mask.copy()
    if n_add <= 0:
        return s.copy_with_mask(occ) if hasattr(s, "copy_with_mask") else s
    pad = int(np.ceil(n_add ** (1 / 3))) + 2
    big = np.pad(occ, pad)
    dist = ndimage.distance_transform_edt(~big)
    cand = np.argwhere((dist > 0) & (dist < pad))
    order = np.lexsort((rng.random(len(cand)), dist[tuple(cand.T)]))
    take = cand[order[:n_add]]
    big[tuple(take.T)] = True
    grown = big[pad:-pad or None, pad:-pad or None, pad:-pad or None]
    # the dilation may spill past the original array; keep what fits, which
    # under-adds rather than silently resizing the build
    out = s.with_occupancy(grown) if hasattr(s, "with_occupancy") else None
    return out, int(grown.sum() - occ.sum())


def main() -> int:
    from blockgen.eval.bench import splits, probes as pb, geometry as geom
    from blockgen.eval.bench import distances as D, ladder as ld

    split = splits.load_split("houses_32")
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    ref_s = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]
    idx = np.random.default_rng(SEED + 1).permutation(len(test))[:N]
    real = [test[i] for i in idx]
    solid = [pb.solidify(s) for s in real]

    # --- the matched control -------------------------------------------
    rng = np.random.default_rng(SEED)
    grown, deltas = [], []
    for a, b in zip(real, solid):
        n_add = int(b.occupied_mask.sum() - a.occupied_mask.sum())
        g = _grow(a, n_add, rng)
        grown.append(g)
        deltas.append((n_add, int(g.occupied_mask.sum() - a.occupied_mask.sum())))
    want = np.array([d[0] for d in deltas], float)
    got = np.array([d[1] for d in deltas], float)
    ok = want > 0
    print(f"matched control: median blocks added by the probe {np.median(want):.0f}, "
          f"by Grow-Shell {np.median(got):.0f}; "
          f"match within 1 block on {int((abs(want-got) <= 1).sum())}/{N} builds")

    arms = {"real@solidify": solid, "real@grow": grown}
    layout = geom.descriptor_layout()
    names = [n for n, _ in layout]
    widths = [w for _, w in layout]
    bounds, o = {}, 0
    for n_, w in layout:
        bounds[n_] = (o, o + w); o += w
    n_scal = widths[-1]
    scal0 = bounds["scalars"][0]
    from blockgen.eval.bench.geometry import GEOMETRY_METRICS
    scalar_names = list(GEOMETRY_METRICS) + ["log1p_n_blocks"]

    raw_ref = geom.descriptor(ref_s)
    raw_arm = {k: geom.descriptor(v) for k, v in arms.items()}
    raw_real = geom.descriptor(real)
    # The floor is the spread of FRESH real draws of size N, so the pool it
    # draws from must be the whole test split. Passing the N-row arm instead
    # makes every draw identical and the floor exactly zero.
    raw_pool = geom.descriptor(test)

    variants = {"full (all 77)": np.ones(sum(widths), bool)}
    for n_ in names:                                   # each block alone
        m = np.zeros(sum(widths), bool); m[slice(*bounds[n_])] = True
        variants[f"{n_} only ({bounds[n_][1]-bounds[n_][0]}d)"] = m
    for n_ in names:                                   # each block removed
        m = np.ones(sum(widths), bool); m[slice(*bounds[n_])] = False
        variants[f"without {n_}"] = m
    for drop in ("interior_ratio_sealed", "log1p_n_blocks"):
        m = np.ones(sum(widths), bool)
        m[scal0 + scalar_names.index(drop)] = False
        variants[f"without {drop}"] = m
    m = np.ones(sum(widths), bool)
    for drop in ("interior_ratio_sealed", "log1p_n_blocks"):
        m[scal0 + scalar_names.index(drop)] = False
    variants["without both suspects"] = m

    rows = []
    for label, mask in variants.items():
        std = geom.Standardizer.fit(raw_ref[:, mask])
        fr = std(raw_ref[:, mask])
        f_real = std(raw_real[:, mask])
        pool = std(raw_pool[:, mask])
        _, sd, _ = ld.noise_floor(lambda r, p: D.kid(p, r), fr, pool, N,
                                  reps=REPS, rng=np.random.default_rng(SEED + 991))
        base = D.kid(f_real, fr)
        out = {"variant": label, "dims": int(mask.sum()), "floor": float(sd)}
        for arm, raw in raw_arm.items():
            v = D.kid(std(raw[:, mask]), fr)
            out[arm] = float((v - base) / sd) if sd > 0 else float("nan")
        rows.append(out)
        print(f"  {label:<34} {out['dims']:>3}d   "
              f"solidify {out['real@solidify']:>9.1f}   "
              f"grow {out['real@grow']:>8.1f}")

    # --- the double dissociation ---------------------------------------
    # Grow-Shell changes the silhouette that Fill-Interior leaves untouched, so
    # the render metric should see the control and miss the probe while the
    # geometric metric does neither. Same builds, same mass added, opposite
    # verdicts: that is the complementarity claim without a hand-built appeal.
    from blockgen.eval.bench import features as ft
    print("\nrender metric on the same three arms ...", flush=True)
    view = ft.ViewConfig()
    f_ref = ft.pooled(ft.embed_views(ref_s, view, verbose=False))
    f_pool = ft.pooled(ft.embed_views(test, view, verbose=False))
    f_r = ft.pooled(ft.embed_views(real, view, verbose=False))
    _, sd_r, _ = ld.noise_floor(lambda r, p: D.kid(p, r), f_ref, f_pool, N,
                                reps=REPS, rng=np.random.default_rng(SEED + 991))
    base_r = D.kid(f_r, f_ref)
    render = {}
    for arm, structs in arms.items():
        v = D.kid(ft.pooled(ft.embed_views(structs, view, verbose=False)), f_ref)
        render[arm] = float((v - base_r) / sd_r)
        print(f"  {arm:<16} render {render[arm]:+8.2f} floors")
    full = next(r for r in rows if r["variant"].startswith("full"))
    print(f"\n  {'arm':<16} {'render':>10} {'geometry':>10}")
    for arm in arms:
        print(f"  {arm:<16} {render[arm]:>10.2f} {full[arm]:>10.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n": N, "reps": REPS, "seed": SEED,
                               "rows": rows, "render": render,
                               "render_floor": float(sd_r),
                               "matched_median_blocks": float(np.median(want)),
                               "matched_exact": int((abs(want-got) <= 1).sum())},
                              indent=1))
    print(f"\nwrote {OUT}")
    return 0


def _grow(s, n_add, rng):
    """Outward dilation adding `n_add` cells, padding the grid as needed."""
    from scipy import ndimage
    from blockgen.utils.data import Structure
    occ = s.occupied_mask
    ids, dat = s.block_ids, s.block_data
    vals, counts = np.unique(ids[occ], return_counts=True)
    dom_id = int(vals[np.argmax(counts)])
    dm = dat[occ][ids[occ] == dom_id]
    dom_dat = int(np.bincount(dm).argmax()) if dm.size else 0

    pad = max(int(np.ceil((n_add / max(occ.sum(), 1)) * 4)) + 2, 3)
    big = np.pad(occ, pad)
    dist = ndimage.distance_transform_edt(~big)
    cand = np.argwhere((dist > 0) & (dist <= pad))
    order = np.lexsort((rng.random(len(cand)), dist[tuple(cand.T)]))
    take = cand[order[:n_add]]
    big[tuple(take.T)] = True
    bi = np.pad(ids, pad); bd = np.pad(dat, pad)
    bi[tuple(take.T)] = dom_id; bd[tuple(take.T)] = dom_dat
    return Structure(block_ids=bi, block_data=bd).crop_to_non_air()


if __name__ == "__main__":
    raise SystemExit(main())
