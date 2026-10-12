"""Is the interior blindness a DINOv2 artifact, or a property of the view protocol?

The paper's central negative result is that a multi-view render-and-embed metric
(DINOv2-B, 4 orthographic views, 224px, unbiased KID) cannot separate real houses
from the same houses with every enclosed room filled in (`probes.solidify`),
while a voxel-descriptor metric separates the same probe by ~200 noise floors.

The obvious objection is that this indicts *one encoder at one view budget*
rather than render-and-embed metrics as a class. This script decides between the
two readings by holding the probe, the KID, the split and the sample size fixed
and ablating only the render pipeline:

* **A. encoder** -- DINOv2-B vs CLIP ViT-B/16 vs CLIP ViT-B/32 vs an
  ImageNet-supervised ResNet-50 vs Inception-v3. Self-supervised ViT,
  language-supervised ViT and label-supervised CNN: if all three regimes are
  blind, no pretraining objective is the culprit.
* **B. view budget** -- 4, 8, 16 views. Does more exterior coverage recover the
  interior?
* **C. projection** -- orthographic vs perspective.
* **E. pooling** -- the four views scored one at a time instead of mean-pooled,
  in case averaging is what hides a difference that one view carries.
* **D. cutaway (the positive control)** -- the same encoder and the same KID, but
  the renders show the inside. If the metric separates `solidify` here, the
  information exists and returns the moment the views carry it, which localises
  the blindness to the *view protocol*. Without a working positive control the
  whole result is unfalsifiable: "no condition separates it" would be
  indistinguishable from a broken harness.

Units. Every condition is reported in its own noise floor -- the spread of KID
across repeated draws of a fresh real sample of the same size, `ladder.noise_floor`,
the same denominator BlockScore uses. Encoders with different feature dimensions
produce KIDs on different scales, so raw KID is not comparable across conditions
and noise-floor multiples are.

Cost: ~56k renders, about 25 minutes on this machine. Features are cached under
`--feature-root` so a re-run with an added condition is cheap.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as ld
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import splits, stats
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

RING8 = tuple(float(a) for a in range(0, 360, 45))


def view_budget(n: int) -> Tuple[Tuple[float, float], ...]:
    """Camera rings for the 4/8/16-view arms, each a superset of the last.

    The budgets must nest, otherwise "16 views sees more" could be an artifact of
    16 views seeing *elsewhere*. 8 adds the four intercardinal azimuths at the
    same elevation; 16 adds a high ring (elev 75, looking down on the roof) and a
    low ring (elev -45, looking up at the underside), which is as close to full
    exterior coverage as an exterior protocol can get. Elevations stop short of
    +/-90 because `renderer.textured._look_at` degenerates when the view
    direction is parallel to the up vector.
    """
    if n == 4:
        return ft.DEFAULT_VIEWS
    if n == 8:
        return tuple((a, 30.0) for a in RING8)
    if n == 16:
        return (tuple((a, 30.0) for a in RING8)
                + tuple((a, 75.0) for a in RING8[::2])
                + tuple((a, -45.0) for a in RING8[::2]))
    raise ValueError(f"no view ring defined for {n} views")


#: The camera that faces each quadrant corner, so a `quad` cut actually opens the
#: build toward the camera rather than hiding the hole behind a wall.
_QUAD_FOR_AZIM = {45.0: "quad:hi:hi", 135.0: "quad:hi:lo",
                  225.0: "quad:lo:lo", 315.0: "quad:lo:hi"}


@dataclass(frozen=True)
class Condition:
    name: str
    backbone: str
    view: ft.ViewConfig
    family: str
    sees_inside: bool
    note: str = ""


def conditions() -> List[Condition]:
    """Every ablation arm. The first is the protocol the paper actually used."""
    base = ft.ViewConfig()                       # 4 views, ortho, 224px
    out = [Condition("dinov2b / 4 ortho (paper)", "dinov2b", base, "baseline",
                     False, "the published protocol")]

    for bb in ("clipB16", "clipB", "resnet50", "inceptionv3"):
        out.append(Condition(f"{bb} / 4 ortho", bb, base, "A encoder", False))

    for n in (8, 16):
        out.append(Condition(f"dinov2b / {n} ortho", "dinov2b",
                             ft.ViewConfig(views=view_budget(n)), "B views", False))

    out.append(Condition("dinov2b / 4 perspective", "dinov2b",
                         ft.ViewConfig(ortho=False), "C projection", False))

    # --- D. positive controls, weakest first ------------------------------
    out.append(Condition(
        "cut: facing quadrant", "dinov2b",
        ft.ViewConfig(cuts=tuple(_QUAD_FOR_AZIM[a] for a, _ in ft.DEFAULT_VIEWS)),
        "D cutaway", True, "F34.2's best-effort cutaway, reproduced"))
    out.append(Condition(
        "cut: roof off (top half)", "dinov2b",
        ft.ViewConfig(views=tuple((a, 60.0) for a, _ in ft.DEFAULT_VIEWS),
                      cuts=("half:y:lo",) * 4),
        "D cutaway", True, "upper half deleted, steeper camera"))
    out.append(Condition(
        "cut: floorplan slabs", "dinov2b",
        ft.ViewConfig(views=tuple((45.0, 75.0) for _ in range(4)),
                      cuts=tuple(f"slab:y:{p}:1" for p in (20, 40, 60, 80))),
        "D cutaway", True, "four one-voxel horizontal sections, seen from above"))
    out.append(Condition(
        "cut: vertical sections", "dinov2b",
        ft.ViewConfig(views=((0.0, 0.0), (0.0, 0.0), (90.0, 0.0), (90.0, 0.0)),
                      cuts=("slab:z:35:1", "slab:z:65:1",
                            "slab:x:35:1", "slab:x:65:1")),
        "D cutaway", True, "four one-voxel cross-sections, seen face-on"))
    return out


@dataclass
class Row:
    name: str
    family: str
    backbone: str
    view_key: str
    dim: int
    floor: float
    floor_ci: Tuple[float, float]
    solidify: float
    solidify_ci: Tuple[float, float]
    sd: float
    sep: float
    sep_ci: Tuple[float, float]
    floor_mean: float = float("nan")
    cut_empty: float = 0.0
    note: str = ""

    @classmethod
    def of(cls, name: str, family: str, backbone: str, view_key: str,
           dim: int, sc: "Scored", **kw) -> "Row":
        return cls(name, family, backbone, view_key, dim, sc.floor, sc.floor_ci,
                   sc.solidify, sc.solidify_ci, sc.sd, sc.sep, sc.sep_ci,
                   floor_mean=sc.floor_mean, **kw)

    @property
    def self_consistent(self) -> bool:
        """Is this condition's floor draw a typical real draw? (ladder G7)"""
        return bool(np.isfinite(self.sd) and self.sd > 0
                    and abs(self.floor - self.floor_mean) < 2.0 * self.sd)

    @property
    def sees(self) -> bool:
        """3 noise floors, and the interval must clear zero. Same bar as the ladder."""
        return bool(np.isfinite(self.sep) and self.sep > 3.0 and self.sep_ci[0] > 0)

    def to_json(self) -> dict:
        return {"condition": self.name, "family": self.family,
                "backbone": self.backbone, "view": self.view_key, "dim": self.dim,
                "floor_kid": self.floor, "floor_ci": list(self.floor_ci),
                "solidify_kid": self.solidify, "solidify_ci": list(self.solidify_ci),
                "noise_sd": self.sd, "separation_sd": self.sep,
                "separation_ci": list(self.sep_ci), "sees_interior": self.sees,
                "floor_mean": self.floor_mean,
                "floor_is_a_typical_real_draw": self.self_consistent,
                "cut_empty_rate": self.cut_empty, "note": self.note}


def _kid(arm: np.ndarray, ref: np.ndarray) -> float:
    return D.kid(arm, ref)


@dataclass
class Scored:
    """One condition's four numbers, named rather than positional.

    `floor_mean` is the mean of the repeated real draws and is the condition's
    self-consistency check: if the single `real_test` draw that defines `floor`
    sits far from the mean of fresh draws of the same size, the floor is a quirk
    of that draw and nothing measured against it can be trusted. The ladder's G7
    gate asks the same question with the same 2-sd bar.
    """

    floor: float
    floor_ci: Tuple[float, float]
    floor_mean: float
    solidify: float
    solidify_ci: Tuple[float, float]
    sd: float
    sep: float
    sep_ci: Tuple[float, float]

    @property
    def self_consistent(self) -> bool:
        return bool(np.isfinite(self.sd) and self.sd > 0
                    and abs(self.floor - self.floor_mean) < 2.0 * self.sd)


def score(ref_feats: np.ndarray, pool_feats: np.ndarray, sol_feats: np.ndarray,
          n: int, reps: int, ci_reps: int = 200, seed: int = 0) -> Scored:
    """KID floor, KID under `solidify`, the noise floor, and the separation.

    The probe set is `pool_feats[:n]` and `sol_feats` is that same set solidified,
    **build for build**. That pairing is what lets the separation carry a tight
    interval: the delta is resampled on a shared index set, so between-build
    variance -- which is far larger than the effect -- cancels instead of being
    charged to the effect.
    """
    real_feats = pool_feats[:n]
    floor = _kid(real_feats, ref_feats)
    solid = _kid(sol_feats, ref_feats)
    _, lo_f, hi_f = stats.subsample_ci(_kid, real_feats, ref_feats, n_rep=ci_reps,
                                       rng=np.random.default_rng(seed))
    _, lo_s, hi_s = stats.subsample_ci(_kid, sol_feats, ref_feats, n_rep=ci_reps,
                                       rng=np.random.default_rng(seed))
    floor_mean, sd, _ = ld.noise_floor(lambda r, p: _kid(p, r), ref_feats,
                                       pool_feats, n, reps=reps,
                                       rng=np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)
    m = int(round(0.8 * n))
    deltas = np.empty(ci_reps, dtype=float)
    for r in range(ci_reps):
        idx = rng.choice(n, size=m, replace=False)
        deltas[r] = _kid(sol_feats[idx], ref_feats) - _kid(real_feats[idx], ref_feats)
    sep = (solid - floor) / sd if np.isfinite(sd) and sd > 0 else float("nan")
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    sep_ci = (float(d_lo / sd), float(d_hi / sd)) if np.isfinite(sd) and sd > 0 \
        else (float("nan"), float("nan"))
    return Scored(floor, (lo_f, hi_f), floor_mean, solid, (lo_s, hi_s), sd,
                  sep, sep_ci)


def best_single_view_row(ref_f: np.ndarray, pool_f: np.ndarray, sol_f: np.ndarray,
                         n: int, reps: int, ci_reps: int) -> Row:
    """Score each view slot on its own and keep the most sensitive one.

    Mean-pooling is the suite's default and its cost is stated in
    `features`: "a build with a handsome facade and a hollow back averages to
    fine". If pooling were what hides the interior, one view would still carry
    it, so this asks each of the four independently and reports the best. Taking
    the maximum over four correlated tests is mildly optimistic -- which is the
    right direction for a condition whose job is to *fail* to find the signal.

    Features arrive unpooled as `[N, V, D]`; each slot is already L2-normalized,
    so a slot is a valid feature matrix with the structure as the sample unit and
    the noise floor is computed on that same slot.
    """
    best: Row | None = None
    for v in range(ref_f.shape[1]):
        sc = score(ref_f[:, v], pool_f[:, v], sol_f[:, v], n, reps, ci_reps)
        row = Row.of(f"dinov2b / 4 ortho, single view {v} (no pooling)",
                     "E pooling", "dinov2b", "v4_single", int(ref_f.shape[-1]),
                     sc, note="best of the four view slots")
        if best is None or row.sep > best.sep:
            best = row
    return best


def geometry_row(ref: Sequence[Structure], pool: Sequence[Structure],
                 sol: Sequence[Structure], n: int, reps: int,
                 ci_reps: int = 200) -> Row:
    """The voxel-descriptor comparison point, scored by the identical procedure.

    Included because "no render condition sees it" is only meaningful beside a
    metric that does; it is the same KID on different features.
    """
    std = geom.Standardizer.fit(geom.descriptor(list(ref)))
    ref_f = geom.geom_features(list(ref), std)
    pool_f = geom.geom_features(list(pool), std)
    sol_f = geom.geom_features(list(sol), std)
    sc = score(ref_f, pool_f, sol_f, n, reps, ci_reps)
    return Row.of("geom_kid (voxel descriptor)", "reference metric", "-", "-",
                  int(ref_f.shape[1]), sc,
                  note="not a render metric; the comparison point")


def run(corpus: str, seed: int, n: int, n_ref: int, reps: int, device: str,
        feature_root: Path, only: Sequence[str] | None, ci_reps: int = 200,
        verbose: bool = True) -> Tuple[List[Row], dict]:
    split = splits.load_split(corpus, seed)
    train = [s.crop_to_non_air() for s in splits.split_structures(split, "train")]
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]

    rng = np.random.default_rng(seed)
    # Shuffle before slicing: corpus order groups builds by GrabCraft category,
    # so the first k would be a different category mix from the rest and the
    # noise floor would be measured on a different population than the probes.
    ref = [train[i] for i in rng.permutation(len(train))[:n_ref]]
    pool = [test[i] for i in np.random.default_rng(seed).permutation(len(test))]
    probes = pool[:n]
    sol = [pb.solidify(s) for s in probes]

    meta = {"corpus": corpus, "seed": seed, "n_probe": n, "n_ref": len(ref),
            "n_null_pool": len(pool), "reps": reps, "device": device,
            "split_key": split.key(), "ci_reps": ci_reps,
            "host": platform.node()}

    rows: List[Row] = []
    for cond in conditions():
        if only and cond.name not in only and cond.family not in only:
            continue
        if verbose:
            print(f"\n=== {cond.name} [{cond.family}] ===", flush=True)
        vk = cond.view.key()

        def embed(tag: str, structs: Sequence[Structure]) -> np.ndarray:
            """Unpooled `[N, V, D]`, so the per-view arm can reuse the same cache."""
            key = ft.FeatureKey(f"ablate_{corpus}", tag, vk, cond.backbone)
            return ft.load_or_build(key, structs, cond.view, device=device,
                                    root=feature_root, verbose=verbose)

        raw = {tag: embed(tag, structs) for tag, structs in
               ((f"ref{len(ref)}", ref), (f"test{len(pool)}", pool),
                (f"solidify{n}", sol))}
        ref_f, pool_f, sol_f = (ft.pooled(raw[k], "mean") for k in
                                (f"ref{len(ref)}", f"test{len(pool)}",
                                 f"solidify{n}"))
        sc = score(ref_f, pool_f, sol_f, n, reps, ci_reps)
        row = Row.of(cond.name, cond.family, cond.backbone, vk,
                     int(ref_f.shape[1]), sc,
                     cut_empty=ft.cut_empty_rate(probes, cond.view.cuts),
                     note=cond.note)
        rows.append(row)
        if cond.family == "baseline":
            # Same cached features, no pooling: closes the "mean-pooling hid it"
            # reading of the null before anyone raises it.
            rows.append(best_single_view_row(
                raw[f"ref{len(ref)}"], raw[f"test{len(pool)}"],
                raw[f"solidify{n}"], n, reps, ci_reps))
        if verbose:
            for r in ([row] if row is rows[-1] else [row, rows[-1]]):
                print(f"  [{r.family}] floor {r.floor:+.4f}  "
                      f"solidify {r.solidify:+.4f}  sd {r.sd:.4f}  "
                      f"separation {r.sep:+.1f} sd  sees={r.sees}  "
                      f"typical_floor={r.self_consistent}", flush=True)
    # Last, so the table reads "no render condition sees it -- and here is the
    # metric that does" rather than opening with the comparison point.
    rows.append(geometry_row(ref, pool, sol, n, reps, ci_reps))
    return rows, meta


def view_sheet(run_dir: Path, corpus: str, seed: int, k: int = 3,
               px: int = 256) -> Path:
    """Render what each condition actually shows, real beside solidified.

    A table of KIDs cannot show that a positive control is a positive control --
    it can only show that some number moved. This sheet is the evidence a reader
    can check with their eyes: on the exterior rows the two builds are pixel-wise
    near-identical, on the cross-section rows one is full of rooms and the other
    is a slab of stone. Builds are picked by interior volume, largest first,
    because a build with no enclosed air is not a test of anything.
    """
    from PIL import Image
    from scipy import ndimage

    split = splits.load_split(corpus, seed)
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]

    def interior(s: Structure) -> int:
        o = s.occupied_mask
        f = ndimage.binary_fill_holes(
            o, structure=ndimage.generate_binary_structure(3, 1))
        return int((f & ~o).sum())

    picks = sorted(range(len(test)), key=lambda i: -interior(test[i]))[:k]
    panels = [("exterior (the paper's protocol)", ft.ViewConfig(views=((45.0, 30.0),))),
              ("facing-quadrant cut", ft.ViewConfig(views=((45.0, 30.0),),
                                                    cuts=("quad:hi:hi",))),
              ("roof off", ft.ViewConfig(views=((45.0, 60.0),),
                                         cuts=("half:y:lo",))),
              ("vertical section", ft.ViewConfig(views=((0.0, 0.0),),
                                                 cuts=("slab:z:50:1",)))]
    rows = []
    for i in picks:
        real, sol = test[i], pb.solidify(test[i])
        strip = []
        for _, view in panels:
            v = ft.ViewConfig(views=view.views, px=px, ortho=view.ortho,
                              cuts=view.cuts)
            strip.append(ft.render_views([real], v, verbose=False)[0])
            strip.append(ft.render_views([sol], v, verbose=False)[0])
        rows.append(np.concatenate(strip, axis=1))
    out = run_dir / "views.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    (run_dir / "views.txt").write_text(
        "columns, in pairs of (real, solidified): "
        + "; ".join(name for name, _ in panels) + "\n"
        + f"rows: held-out builds {picks} -- the three largest interiors\n")
    return out


def table(rows: Sequence[Row]) -> str:
    head = ("| condition | family | floor KID [95%] | solidify KID [95%] | "
            "noise sd | separation (noise floors) | sees the interior? |\n"
            "|---|---|---|---|---|---|---|\n")
    out = []
    for r in rows:
        out.append(
            f"| {r.name} | {r.family} | "
            f"{r.floor:+.4f} [{r.floor_ci[0]:+.4f}, {r.floor_ci[1]:+.4f}] | "
            f"{r.solidify:+.4f} [{r.solidify_ci[0]:+.4f}, {r.solidify_ci[1]:+.4f}] | "
            f"{r.sd:.4f} | {r.sep:+.1f} [{r.sep_ci[0]:+.1f}, {r.sep_ci[1]:+.1f}] | "
            f"{'**YES**' if r.sees else 'no'} |")
    return head + "\n".join(out)


def verdict(rows: Sequence[Row]) -> str:
    """State which of the two readings the numbers support, without hedging."""
    render = [r for r in rows if r.family not in ("reference metric",)]
    exterior = [r for r in render if r.family != "D cutaway"]
    cutaway = [r for r in render if r.family == "D cutaway"]
    ext_seen = [r for r in exterior if r.sees]
    cut_seen = [r for r in cutaway if r.sees]
    best_ext = max(exterior, key=lambda r: r.sep) if exterior else None
    best_cut = max(cutaway, key=lambda r: r.sep) if cutaway else None

    lines = []
    if not ext_seen and cut_seen:
        lines.append(
            "**The blindness is a property of the view protocol, not of the "
            "encoder.** No exterior condition separates `solidify` -- not a "
            "different pretraining objective, not a CNN instead of a ViT, not "
            "four times the view budget, not perspective projection -- while the "
            "same encoder and the same KID separate it by "
            f"{best_cut.sep:.0f} noise floors as soon as the renders show the "
            "inside. The information is absent from the input, not mishandled by "
            "the features, so the claim generalises to render-and-embed metrics "
            "built on exterior views.")
        lines.append(
            f"The strongest exterior condition was *{best_ext.name}* at "
            f"{best_ext.sep:+.1f} sd, which is inside the noise.")
    elif ext_seen:
        lines.append(
            "**The paper's claim must weaken.** "
            + ", ".join(f"`{r.name}` ({r.sep:+.1f} sd)" for r in ext_seen)
            + " separates `solidify` from exterior views alone, so the honest "
              "statement is that *the standard protocol is under-viewed / uses "
              "the wrong encoder*, not that render-and-embed metrics cannot see "
              "interiors as a class.")
    elif not cut_seen:
        lines.append(
            "**Inconclusive, and the result must not be reported.** No condition "
            "separates `solidify`, including the positive controls, so a broken "
            "harness and a real informational blindness are indistinguishable "
            "here. Fix the positive control before claiming anything.")
    return "\n\n".join(lines)


def findings(rows: Sequence[Row], meta: dict) -> str:
    return "\n".join([
        "# Is the interior blindness one encoder, or the view protocol?",
        "",
        f"Corpus `{meta['corpus']}`, split `{meta['split_key']}`, seed "
        f"{meta['seed']}. n = {meta['n_probe']} probe builds against "
        f"{meta['n_ref']} reference builds; noise floor from "
        f"{meta['reps']} fresh real draws of the same size out of a "
        f"{meta['n_null_pool']}-build held-out pool, with the finite-population "
        "correction `ladder.noise_floor` applies.",
        "",
        "Every row is the *same* probe (`probes.solidify`: fill every enclosed "
        "air cell with the build's own dominant material), the *same* unbiased "
        "polynomial KID, the *same* builds, and mean-pooled L2-normalized "
        "per-view features. Only the render pipeline changes. Raw KID is not "
        "comparable across encoders of different feature dimension, which is why "
        "the decision column is in noise floors.",
        "",
        table(rows),
        "",
        "## Verdict",
        "",
        verdict(rows),
        "",
        "## Notes",
        "",
        "* `sees the interior?` = separation > 3 noise floors *and* the paired "
        "bootstrap interval clears zero -- the same bar the ladder's sensitivity "
        "flags use.",
        "* Separation intervals come from a paired resample: the same 80% subset "
        "of probe builds is scored real and solidified against the full "
        "reference, so between-build variance cancels.",
        "* `geom_kid` is included only as the comparison point. It is not a "
        "render metric.",
        "* Cutaway conditions fall back to the uncut build when a cut would "
        "remove every block; the rate was "
        + ", ".join(f"{r.name} {r.cut_empty:.1%}"
                    for r in rows if r.family == "D cutaway")
        + ". A blank frame scores as a perfect match, so this fallback is "
          "reported rather than hidden -- it biases a cutaway condition "
          "*toward* the exterior null.",
        "* Every condition's floor draw was checked against the mean of its own "
        "repeated real draws (ladder G7, 2 sd): "
        + ", ".join(f"{r.name} {'ok' if r.self_consistent else 'ATYPICAL'}"
                    for r in rows) + ".",
        "",
        "```json",
        json.dumps({"meta": meta, "rows": [r.to_json() for r in rows]}, indent=2),
        "```",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256, help="probe builds")
    ap.add_argument("--n-ref", type=int, default=512, help="reference builds")
    ap.add_argument("--reps", type=int, default=48,
                    help="draws for the noise floor (ladder default)")
    ap.add_argument("--ci-reps", type=int, default=200,
                    help="resamples behind every reported interval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-root", default=None,
                    help="feature cache root (default: <run>/features)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="condition or family names to run")
    ap.add_argument("--name", default="view_protocol_ablation")
    args = ap.parse_args()

    run_dir = new_run_dir(args.name)
    root = Path(args.feature_root) if args.feature_root else run_dir / "features"
    rows, meta = run(args.corpus, args.seed, args.n, args.n_ref, args.reps,
                     args.device, root, args.only, ci_reps=args.ci_reps)
    meta["feature_root"] = str(root)
    (run_dir / "FINDINGS.md").write_text(findings(rows, meta))
    try:
        print(f"wrote {view_sheet(run_dir, args.corpus, args.seed)}")
    except Exception as exc:                  # a figure must never lose a result
        print(f"[warn] view sheet failed: {type(exc).__name__}: {exc}")
    (run_dir / "ablation.json").write_text(json.dumps(
        {"meta": meta, "rows": [r.to_json() for r in rows]}, indent=2) + "\n")
    print("\n" + table(rows))
    print("\n" + verdict(rows))
    print(f"\nwrote {run_dir / 'FINDINGS.md'}")


if __name__ == "__main__":
    main()
