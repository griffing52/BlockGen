"""Does the render metric's blindness depend on *which* exterior views it is given?

Two reviewer questions, one harness. The paper's MV-KID renders four corner views
(azimuth 45/135/225/315, elevation 30). A reviewer can ask:

* **A. Facade views.** Would axis-aligned elevations (front, back, left, right,
  roof plan), the views an architect draws, see more? Two readings:
  - ``A-render``: the same textured renderer, four side elevations at elevation 0
    plus a roof plan, and its four-side subset ``A-sides``.
  - ``A-2.5D``: the facade representation itself (`blockgen.corpus.facade`), one
    picture per face of the first block hit and its depth, drawn without a
    renderer. It treats glass as opaque, so a probe that keeps exterior air
    fixed leaves these images bit-identical.
* **B. Cross-section views.** Would cutting the build open and showing the cut
  see the interior? ``B-mid``: one-voxel sections through the middle of x, z and
  y, seen face-on. ``B-9``: sections at 25/50/75% of each axis. ``B-cut``: the
  build with half removed (x, z, then y), seen at the cut face.

Everything else is the leaderboard of record: houses_32 split
houses_32.s0.70-15-15.v1, seed 0, the 399 val builds as reference, the
`fast.control_arms` draws of the n=256 run (rng seed+1), the baselines with the
runner's seeds, the AR and Dream-Cubed npz files of that run, DINOv2-B/14 with
mean-pooling, the unbiased cubic-kernel KID, and the noise floor as the sd of 48
fresh real draws of n from the 399 test builds (rng seed+991). Every arm is
reported in that unit, relative to `real_test`, which is how the leaderboard
states pillars. `geom_kid` (77-d, v1) is recomputed under the same rule as the
comparison column.

Invariance arms: `real_test` rotated 90 degrees about y, and mirrored in x.
Axis-aligned views are not invariant to yaw unless the pooled set is, so these
rows say whether a protocol scores the same building differently by its heading.

    python scripts/ablate_view_facade_section.py --n 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.eval.bench import baselines as bl
from blockgen.eval.bench import distances as D
from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as ld
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits, stats
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

LEADERBOARD = Path("outputs/run_20261103_090315_bench_n_256_leaderboard_of_record")
NPZ_ARMS = {
    "canon_oriented": "outputs/bench_arms/canon_oriented_n256_32.npz",
    "canon_unoriented": "outputs/bench_arms/canon_unoriented_n256_32.npz",
    "native_oriented": "outputs/bench_arms/native_oriented_n256_32.npz",
    "dreamcubed": "outputs/bench_arms/dreamcubed_32_n256.npz",
}
BASELINES = ("shell_box", "patchwork", "gabled_house")
TOP = 89.5          # `textured._look_at` degenerates at exactly 90

#: Arms that are `real_test` build for build, so their delta is paired.
PAIRED = ("real_shuffled_materials", "real@monochrome", "real@solidify",
          "real@hollow", "real@rot90", "real@mirror")
INVARIANCE = ("real@rot90", "real@mirror")

_SEC_AXES = (("x", (90.0, 0.0)), ("z", (0.0, 0.0)), ("y", (0.0, TOP)))
_SEC_PCTS = (25, 50, 75)

RENDERED: Dict[str, ft.ViewConfig] = {
    "P0": ft.ViewConfig(),
    "A": ft.ViewConfig(views=((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0),
                              (0.0, TOP))),
    "B": ft.ViewConfig(
        views=tuple(v for _, v in _SEC_AXES for _ in _SEC_PCTS),
        cuts=tuple(f"slab:{a}:{p}:1" for a, _ in _SEC_AXES for p in _SEC_PCTS)),
    "C": ft.ViewConfig(views=((90.0, 20.0), (0.0, 20.0), (0.0, TOP)),
                       cuts=("half:x:lo", "half:z:lo", "half:y:lo")),
}

#: protocol name -> (source, view slots, what it is)
PROTOCOLS: Dict[str, Tuple[str, Tuple[int, ...], str]] = {
    "corner (paper)": ("P0", (0, 1, 2, 3), "4 corner views, az 45/135/225/315, el 30"),
    "A-sides": ("A", (0, 1, 2, 3), "4 side elevations, el 0"),
    "A-render": ("A", (0, 1, 2, 3, 4), "4 side elevations + roof plan"),
    "A-2.5D": ("F", (0, 1, 2, 3, 4), "5 facade images (block + depth), no renderer"),
    "B-mid": ("B", (1, 4, 7), "mid sections of x, z, y, face-on"),
    "B-9": ("B", tuple(range(9)), "sections at 25/50/75% of x, z, y"),
    "B-cut": ("C", (0, 1, 2), "half removed along x, z, y; seen at the cut"),
}


def facade_images(s: Structure, px: int = 224) -> List[np.ndarray]:
    """The five `SIDES_TOP` face pictures, each padded square and scaled to `px`."""
    out = []
    c = s.crop_to_non_air()
    for face in fc.SIDES_TOP:
        img = fc.face_picture(fc.project(c, face))
        h, w = img.shape[:2]
        side = max(h, w)
        sq = np.full((side, side, 3), 255, np.uint8)
        sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
        k = max(1, px // side)
        sq = np.repeat(np.repeat(sq, k, 0), k, 1)
        big = np.full((px, px, 3), 255, np.uint8)
        hh = min(px, sq.shape[0])
        o = (px - hh) // 2
        big[o:o + hh, o:o + hh] = sq[:hh, :hh]
        out.append(big)
    return out


def embed_facades(structs: Sequence[Structure], cache: Path, device: str) -> np.ndarray:
    sha = ft.structures_sha(structs)
    path = cache / f"facade25d_{sha}.npz"
    if path.exists():
        return np.load(path)["feats"]
    imgs = [im for s in structs for im in facade_images(s)]
    f = ft.embed_images(imgs, "dinov2b", device).reshape(len(structs), 5, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feats=f.astype(np.float32))
    return f


def build_arms(seed: int, n: int) -> Tuple[List[Structure], List[Structure],
                                           Dict[str, List[Structure]], dict]:
    split = splits.load_split("houses_32", seed)
    train = splits.split_structures(split, "train")
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")
    ref = [s.crop_to_non_air() for s in val]
    calib = [s.crop_to_non_air() for s in test]

    arms: Dict[str, List[Structure]] = {}
    for a in fast_tier.control_arms(test, train=train, n=n,
                                    rng=np.random.default_rng(seed + 1),
                                    corpus="houses_32", seed=seed):
        if a.name in ("real_test", "real_shuffled_materials", "real@monochrome",
                      "real@solidify"):
            arms[a.name] = a.structures
    rt = arms["real_test"]
    arms["real@hollow"] = [pb.hollow(s) for s in rt]
    arms["real@rot90"] = [pb.rotate_y(s, 1).crop_to_non_air() for s in rt]
    arms["real@mirror"] = [pb.mirror_x(s).crop_to_non_air() for s in rt]
    for i, name in enumerate(bl.ORDER):
        if name in BASELINES:
            arms[name] = [s.crop_to_non_air()
                          for s in bl.build(name, n, train, seed=seed + i)]
    for name, path in NPZ_ARMS.items():
        arms[name] = sc.ArmSpec(name=name, npz=path).load()[:n]

    # Same builds as the leaderboard of record, or the comparison is void.
    check = {}
    if n == 256 and (LEADERBOARD / "scorecard.json").exists():
        card = json.loads((LEADERBOARD / "scorecard.json").read_text())["arms"]
        for name, structs in arms.items():
            want = (card.get(name) or {}).get("meta", {}).get("structures_sha")
            if want:
                check[name] = ft.structures_sha(structs) == want
    return ref, calib, arms, check


def kid(a: np.ndarray, r: np.ndarray) -> float:
    return D.kid(a, r)


def score_space(ref: np.ndarray, calib: np.ndarray, arms: Dict[str, np.ndarray],
                n: int, seed: int, kidfn=kid, ci_reps: int = 200) -> Dict[str, dict]:
    """Every arm in noise floors above `real_test`, with an interval."""
    _, sd, _ = ld.noise_floor(lambda r, p: kidfn(p, r), ref, calib, n, reps=48,
                              rng=np.random.default_rng(seed + 991))
    base = kidfn(arms["real_test"], ref)
    rng = np.random.default_rng(seed + 1)
    m = int(round(0.8 * n))
    out = {"_sd": float(sd), "_real_kid": float(base)}
    for name, f in arms.items():
        val = kidfn(f, ref)
        if name == "real_test":
            out[name] = {"sep": 0.0, "ci": [0.0, 0.0], "kid": val}
            continue
        d = np.empty(ci_reps)
        for r in range(ci_reps):
            i = rng.choice(n, size=m, replace=False)
            j = i if name in PAIRED else rng.choice(n, size=m, replace=False)
            d[r] = kidfn(f[i], ref) - kidfn(arms["real_test"][j], ref)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {"sep": float((val - base) / sd), "ci": [float(lo / sd), float(hi / sd)],
                     "kid": float(val)}
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def sample_sheet(run_dir: Path, structs: Dict[str, Structure], px: int = 128) -> None:
    """Rows: real, Fill-Interior, Hollow-Shell. Columns: one view per protocol family."""
    from PIL import Image, ImageDraw
    cols = [("corner", "P0", 0), ("A side", "A", 0), ("A roof", "A", 4),
            ("A-2.5D side", "F", 0), ("A-2.5D roof", "F", 4),
            ("B mid x", "B", 1), ("B mid z", "B", 4), ("B mid y", "B", 7),
            ("B-cut x", "C", 0), ("B-cut z", "C", 1), ("B-cut y", "C", 2)]
    rows = list(structs.items())
    W, H, top, left = px * len(cols), px * len(rows), 18, 90
    sheet = Image.new("RGB", (W + left, H + top), "white")
    d = ImageDraw.Draw(sheet)
    for ci, (lab, _, _) in enumerate(cols):
        d.text((left + ci * px + 4, 3), lab, fill="black")
    for ri, (name, s) in enumerate(rows):
        d.text((4, top + ri * px + px // 2), name, fill="black")
        for ci, (_, src, slot) in enumerate(cols):
            if src == "F":
                img = facade_images(s, 224)[slot]
            else:
                v = RENDERED[src]
                one = ft.ViewConfig(views=(v.views[slot],),
                                    cuts=(v.cuts[slot],) if v.cuts else ())
                img = ft.render_views([s], one, verbose=False)[0]
            sheet.paste(Image.fromarray(img).resize((px, px)),
                        (left + ci * px, top + ri * px))
    sheet.save(run_dir / "samples.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-root", default="data/minecraft/features/viewproto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    geom.set_descriptor_version("v1")
    root = Path(args.feature_root)
    run_dir = Path(args.out) if args.out else new_run_dir("view_facade_section")
    run_dir.mkdir(parents=True, exist_ok=True)

    ref, calib, arms, check = build_arms(args.seed, args.n)
    print(f"[view] ref {len(ref)} calib {len(calib)} arms "
          f"{ {k: len(v) for k, v in arms.items()} }", flush=True)
    print(f"[view] same builds as leaderboard: {check}", flush=True)

    sets = {"ref": ref, "calib": calib, **arms}

    def raw(src: str, tag: str, structs: List[Structure]) -> np.ndarray:
        if src == "F":
            return embed_facades(structs, root / "facade25d", args.device)
        view = RENDERED[src]
        key = ft.FeatureKey("viewproto_houses_32", f"{tag}_{ft.structures_sha(structs)}",
                            view.key(), "dinov2b")
        return ft.load_or_build(key, structs, view, device=args.device, root=root,
                                verbose=True)

    feats: Dict[str, Dict[str, np.ndarray]] = {}
    for src in ("P0", "F", "A", "C", "B"):
        print(f"\n=== source {src} ===", flush=True)
        feats[src] = {tag: raw(src, tag, structs) for tag, structs in sets.items()}

    # geometry comparison column, same scoring rule
    std = geom.Standardizer.fit(geom.descriptor(ref))
    gf = {tag: geom.geom_features(structs, std) for tag, structs in sets.items()}

    results: Dict[str, Dict[str, dict]] = {}
    for n in sorted({args.n, 128}):
        res: Dict[str, dict] = {}
        for pname, (src, slots, _) in PROTOCOLS.items():
            f = {t: ft.pooled(x[:, list(slots)], "mean") for t, x in feats[src].items()}
            res[pname] = score_space(f["ref"], f["calib"],
                                     {k: f[k][:n] for k in arms}, n, args.seed)
        res["V-KID (77-d)"] = score_space(gf["ref"], gf["calib"],
                                          {k: gf[k][:n] for k in arms}, n, args.seed,
                                          kidfn=lambda a, r: geom.geom_kid(a, r))
        results[str(n)] = res

    # exactness of the 2.5D null
    ident = {a: bool(np.array_equal(feats["F"][a], feats["F"]["real_test"]))
             for a in ("real@solidify", "real@hollow")}
    ident_img = {}
    for a in ("real@solidify", "real@hollow"):
        same = [all(np.array_equal(x, y) for x, y in
                    zip(facade_images(s), facade_images(t)))
                for s, t in zip(arms["real_test"], arms[a])]
        ident_img[a] = float(np.mean(same))

    ranked = [a for a in arms if a not in ("real_test",) + INVARIANCE]
    corr = {}
    for n, res in results.items():
        base_mv = [res["corner (paper)"][a]["sep"] for a in ranked]
        base_v = [res["V-KID (77-d)"][a]["sep"] for a in ranked]
        corr[n] = {p: {"vs_corner": spearman([res[p][a]["sep"] for a in ranked], base_mv),
                       "vs_vkid": spearman([res[p][a]["sep"] for a in ranked], base_v)}
                   for p in res}

    out = {"protocols": {k: {"source": v[0], "slots": list(v[1]), "what": v[2]}
                         for k, v in PROTOCOLS.items()},
           "views": {k: {"views": [list(x) for x in v.views], "cuts": list(v.cuts)}
                     for k, v in RENDERED.items()},
           "same_builds_as_leaderboard": check, "results": results,
           "spearman": corr, "ranked_arms": ranked,
           "facade25d_identical_features": ident,
           "facade25d_identical_images_frac": ident_img,
           "cut_empty_rate": {k: ft.cut_empty_rate(arms["real_test"], v.cuts)
                              for k, v in RENDERED.items() if v.cuts}}
    (run_dir / "results.json").write_text(json.dumps(out, indent=1))

    lines = []
    for n, res in results.items():
        cols = list(res)
        lines.append(f"\n## n = {n} (noise floors above real_test; 95% interval)\n")
        lines.append("| arm | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---:|" * len(cols))
        for a in arms:
            if a == "real_test":
                continue
            cells = [f"{res[c][a]['sep']:+.2f} [{res[c][a]['ci'][0]:+.1f}, "
                     f"{res[c][a]['ci'][1]:+.1f}]" for c in cols]
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        lines.append("| *noise sd* | " + " | ".join(f"{res[c]['_sd']:.2e}" for c in cols)
                     + " |")
        lines.append("| Spearman vs corner | " + " | ".join(
            f"{corr[n][c]['vs_corner']:+.2f}" for c in cols) + " |")
        lines.append("| Spearman vs V-KID | " + " | ".join(
            f"{corr[n][c]['vs_vkid']:+.2f}" for c in cols) + " |")
    lines.append(f"\nA-2.5D features identical to real_test: {ident}; "
                 f"images identical per build: {ident_img}")
    lines.append(f"\nSame builds as the leaderboard of record: {check}")
    (run_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    k = 0
    sample_sheet(run_dir, {"real": arms["real_test"][k],
                           "Fill-Interior": arms["real@solidify"][k],
                           "Hollow-Shell": arms["real@hollow"][k],
                           "real 2": arms["real_test"][1],
                           "Fill 2": arms["real@solidify"][1]})
    (run_dir / "samples.txt").write_text(
        "rows: real, Fill-Interior, Hollow-Shell (build 0), real and Fill (build 1); "
        "columns: one view of each protocol\n")
    print(f"-> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
