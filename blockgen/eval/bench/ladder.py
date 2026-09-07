"""The validation ladder: a metric must earn the right to adjudicate anything.

This project has twice adopted a metric that turned out to measure something
other than build quality. `nn_iou` disagreed with the eye three times in one day
(T17/T20). `cmmd` replaced it and was then found to track solidity at r = -0.993
(T21). Both were in use before anyone asked what they do to damage we have
already measured.

So: `full.py` refuses to report a metric that has not passed here. A failing
metric is emitted as null with its failing gate named, never as a blank that
reads like a zero.

The rungs are structures whose quality is already established -- F16 measured
canon-16 decimation as deleting 86.5% of blocks and whole roofs -- plus
invariances that must *not* move the number. A metric that ranks damage
correctly but also penalizes a rotated building is measuring pose, not quality.

Validity gates versus sensitivity, and why they are not the same test
--------------------------------------------------------------------
Two different questions get asked of a metric, and conflating them makes the
ladder either useless or dishonest.

*Is it measuring build quality at all?* A metric must order damage correctly,
must not move when a building is merely rotated, and must give the same answer
on two different samples of real builds. Failing any of these means the number
means something other than what its name says -- exactly the situation that
produced the `nn_iou` and `cmmd` retractions. These are **blocking**: `full.py`
reports null.

*How much damage can it resolve at this sample size?* That is a question about
statistical power, not validity, and its answer is a function of n. A metric
that cleanly separates canon-16 but cannot resolve a 40% interior cut at n=64
is not broken; it is being asked for more resolution than 64 samples buy. These
are recorded as **sensitivity** flags and reported beside the value, never used
to suppress it. The remedy is more samples, which is why the suite pushes for
n >= 256 on headline numbers.

Blocking gates
    G1  ordering          real < canon16 < canon8
    G2  ordering          real < noise_1 < noise_5 < noise_10
    G6  invariance        |rot90_k - real|, |mirror_x - real| < 1 sd
    G7  self-consistency  a fresh real draw sits within 2 sd of the null mean
    G8  n-stability       the null mean at n/2 and n agree within 2 standard
                          errors *of their difference* -- not within 2 sd of a
                          single draw, which is sqrt(reps) times larger and
                          passes estimators with a visible 1/n bias

Sensitivity flags (recorded, non-blocking)
    S3  resolves canon16           canon16 - real          > 3 sd
    S4  resolves material shuffle  material_shuffle - real > 3 sd
    S5  resolves 40% deletion      chunk_delete_40 - real  > 3 sd

G7's tolerance is 2 sd rather than 1 because it compares a *single* draw against
the mean of many: a single draw lands within 1 sd of the mean only about 68% of
the time, so a 1 sd bar would fail roughly a third of perfectly good metrics by
chance alone.

`sd(real)` is the spread of the metric across repeated real-vs-reference draws
under the same protocol the rungs use -- the metric's own noise floor.
Everything is measured in those units, so the criteria are scale-free and apply
unchanged to KID, MMD, FD and PRDC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import probes as pb
from blockgen.utils.data import Structure

LADDER_ROOT = Path("outputs/analysis/ladder")

def build_metrics(sigma: float) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Metric table with the RBF bandwidth frozen on the reference set.

    Letting `mmd_rbf` pick its own bandwidth per call (`sigma=None`) silently
    changes the kernel between rungs, so the numbers are not on one scale and
    the self-consistency gate cannot pass. Bandwidth is a property of the
    reference, chosen once.
    """
    table = dict(METRICS)
    table["mmd_rbf"] = lambda r, p: D.mmd_rbf(p, r, sigma=sigma)
    return table


#: metric name -> fn(reference_feats, probe_feats) -> float.
#: All are oriented so that larger means "more different from the reference".
METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kid": lambda r, p: D.kid(p, r),
    "mmd_rbf": lambda r, p: D.mmd_rbf(p, r, sigma=None),
    "fd": lambda r, p: D.fd(p, r),
    # PRDC members are flipped so that, like the others, larger means worse.
    "one_minus_coverage": lambda r, p: 1.0 - D.prdc(r, p)["coverage"],
    "one_minus_density": lambda r, p: 1.0 - min(D.prdc(r, p)["density"], 1.0),
    "one_minus_precision": lambda r, p: 1.0 - D.prdc(r, p)["precision"],
    "one_minus_recall": lambda r, p: 1.0 - D.prdc(r, p)["recall"],
}

#: rungs whose ordering G1 checks
ORDERED_DAMAGE = ("canon16", "canon8")
ORDERED_NOISE = ("noise_1", "noise_5", "noise_10")
INVARIANT = ("rot90_1", "rot90_2", "mirror_x")


@dataclass(frozen=True)
class RungSpec:
    """Which rungs a family of metrics is gated on.

    A metric is only fairly judged on damage it is built to see. The render
    metrics read appearance, so their noise rung retypes blocks in place
    (`noise_*`). The geometry metrics read occupancy and are *exactly* blind to
    retyping, so gating them on `noise_*` would fail every one of them for doing
    precisely what they are designed to do. Their noise rung moves blocks
    instead (`occ_noise_*`), and material shuffle joins their invariance set as
    a positive statement of that blindness: it must not move them at all.
    """
    ordered_noise: Tuple[str, ...]
    invariant: Tuple[str, ...]
    sensitivity: Dict[str, str]


#: Sensitivity rungs shared by both families. `solidify` is listed for both on
#: purpose -- it is the rung that separates them, and recording that the render
#: metrics fail to resolve it is the point.
RENDER_RUNGS = RungSpec(
    ordered_noise=ORDERED_NOISE,
    invariant=INVARIANT,
    sensitivity={"S3_resolves_canon16": "canon16",
                 "S4_resolves_material_shuffle": "material_shuffle",
                 "S5_resolves_chunk_delete_40": "chunk_delete_40",
                 "S6_resolves_solidify": "solidify"})

GEOMETRY_RUNGS = RungSpec(
    ordered_noise=("occ_noise_1", "occ_noise_5", "occ_noise_10"),
    # Material-only corruptions belong in the *invariance* set here: an
    # occupancy metric that moved under them would be reading something it has
    # no access to, which would mean a bug in the descriptor.
    invariant=INVARIANT + ("material_shuffle", "monochrome"),
    sensitivity={"S3_resolves_canon16": "canon16",
                 "S5_resolves_chunk_delete_40": "chunk_delete_40",
                 "S6_resolves_solidify": "solidify",
                 "S7_resolves_jitter_columns": "jitter_columns"})


def legacy_cmmd(r: np.ndarray, p: np.ndarray) -> float:
    """`blockgen.eval.perceptual.cmmd`, scored as a labelled legacy metric."""
    import torch
    from blockgen.eval.perceptual import cmmd
    return float(cmmd(torch.tensor(p), torch.tensor(r)))


@dataclass
class LadderResult:
    backbone: str
    view: str
    n: int
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    noise: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gates: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    sensitivity: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def passed(self, metric: str) -> bool:
        """Validity only. Sensitivity flags never suppress a value."""
        return bool(self.gates.get(metric)) and all(self.gates[metric].values())

    def resolves(self, metric: str) -> List[str]:
        return [k for k, v in self.sensitivity.get(metric, {}).items() if v]

    def failed_gate(self, metric: str) -> str | None:
        for gate, ok in self.gates.get(metric, {}).items():
            if not ok:
                return gate
        return None

    @property
    def passing(self) -> List[str]:
        return [m for m in self.gates if self.passed(m)]

    def to_json(self) -> dict:
        return {"backbone": self.backbone, "view": self.view, "n": self.n,
                "scores": self.scores, "noise_sd": self.noise, "gates": self.gates,
                "sensitivity": self.sensitivity, "passing": self.passing,
                "resolves": {m: self.resolves(m) for m in self.sensitivity},
                "failing": {m: self.failed_gate(m) for m in self.gates
                            if not self.passed(m)}}


def _noise_floor(fn, ref_feats: np.ndarray, pool_feats: np.ndarray, n: int,
                 reps: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """`(mean, sd_pop, sem)` for *real* probes against the real reference.

    Draws n held-out real builds and scores them against the same full reference
    every rung is scored against. Comparing two halves of the reference instead
    would change both the probe size and the reference size relative to the real
    rungs, which shifts the estimator's bias and makes the self-consistency gate
    unpassable for reasons that have nothing to do with the metric.

    This is the metric's own resolution: differences smaller than this sd are
    not differences, whatever the point estimates say.

    Two corrections, both of which the first version of this function got wrong
    and both of which understated the spread -- which made the gates too strict
    and rejected metrics for being noisy when the noise was mismeasured.

    **The pool must contain the probe set.** `real_heldout` is the "fresh real
    draw" G7 tests, so it has to be a member of the distribution it is tested
    against. Drawing the null from the held-out builds *other than* the probes
    turns G7 into a comparison of two disjoint halves, and with a group-aware
    split two halves of a small corpus genuinely differ.

    **The draws overlap, so their spread is not the spread of an independent
    draw.** Repeated n-subsets of an N-build pool share members, and their
    variance carries the finite-population factor `(1 - n/N)`; dividing it out
    recovers the variance of a draw from the population, which is the quantity
    every gate is stated in. At n=64 out of N=245 the factor is 0.87 in sd, and
    measured on `geom_kid` the uncorrected estimate was 0.21 against a true
    between-sample spread near 0.28 -- enough to fail a metric that separates
    the `solidify` rung at 145 sd. The correction is exact for a sample mean and
    a leading-order approximation for the degree-2 U-statistics used here.
    """
    N = len(pool_feats)
    if N < n + 1:
        return float("nan"), float("nan"), float("nan")
    vals = []
    for _ in range(reps):
        idx = rng.choice(N, size=n, replace=False)
        vals.append(fn(ref_feats, pool_feats[idx]))
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    fpc = max(1.0 - n / float(N), 1e-6)
    sd_obs = float(v.std(ddof=1))
    # Two different spreads, for two different questions, and using one where
    # the other belongs is what made G8 toothless (see the gate).
    #   sd_pop -- how much a single fresh real draw moves. The unit every gate
    #             is stated in.
    #   sem    -- how precisely `reps` draws pin down this pool's mean. Draws
    #             share members, so the mean converges to a property of the
    #             pool and `sd_obs / sqrt(reps)` is its honest precision.
    return float(v.mean()), sd_obs / np.sqrt(fpc), sd_obs / np.sqrt(v.size)


def noise_floor(fn, ref_feats: np.ndarray, pool_feats: np.ndarray, n: int,
                reps: int = 48, rng: np.random.Generator | None = None
                ) -> Tuple[float, float, float]:
    """Public alias for the noise-floor estimator: `(mean, sd_pop, sem)`.

    Exported because `composite` states every pillar in units of `sd_pop` -- the
    spread of a *fresh* real sample of the same size, scored under the same
    protocol. Any other denominator makes "real-sample spreads" a false label.
    """
    return _noise_floor(fn, ref_feats, pool_feats, n, reps,
                        rng or np.random.default_rng(0))


def run_ladder(
    reference: Sequence[Structure],
    probe_pool: Sequence[Structure],
    view: ft.ViewConfig = ft.ViewConfig(),
    backbone: str = "dinov2b",
    n: int = 64,
    reps: int = 48,
    device: str = "cuda",
    seed: int = 0,
    include_legacy: bool = True,
    verbose: bool = True,
) -> LadderResult:
    """Score every metric on every rung and evaluate the gates."""
    rng = np.random.default_rng(seed)

    def embed(structs):
        return ft.pooled(ft.embed_views(structs, view, backbone, device,
                                        verbose=False), "mean")

    ref_pool = [s.crop_to_non_air() for s in reference]
    all_probe = [s.crop_to_non_air() for s in probe_pool]
    # Shuffle before slicing. Corpus order groups builds by category -- the
    # GrabCraft categories are contiguous in index order -- so taking the first
    # n would give the probe set and the null pool different category mixes,
    # and the self-consistency gate would then fail for a reason that has
    # nothing to do with the metric.
    order = np.random.default_rng(seed).permutation(len(all_probe))
    all_probe = [all_probe[i] for i in order]
    probe_src = all_probe[:n]
    # Every held-out real build, used only to estimate each metric's noise floor
    # under the *same* protocol the rungs are scored under. The probe set is
    # deliberately included -- see `_noise_floor`.
    null_pool = all_probe
    if verbose:
        print(f"[ladder] reference {len(ref_pool)}, probes {len(probe_src)}, "
              f"null pool {len(null_pool)}", flush=True)

    ref_feats = embed(ref_pool)
    metrics = build_metrics(D.median_bandwidth(ref_feats))
    if include_legacy:
        metrics["legacy_cmmd"] = legacy_cmmd

    suite = pb.probe_suite(probe_src, np.random.default_rng(seed))
    suite["real_heldout"] = probe_src
    probe_feats = {}
    for name, structs in suite.items():
        if verbose:
            print(f"  [{name}]", flush=True)
        probe_feats[name] = embed(structs)
    null_feats = embed(null_pool)

    return score_rungs(metrics, ref_feats, probe_feats, null_feats, RENDER_RUNGS,
                       backbone=backbone, view=view.key(), reps=reps, rng=rng)


def score_rungs(
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    ref_feats: np.ndarray,
    probe_feats: Dict[str, np.ndarray],
    null_feats: np.ndarray,
    rungs: RungSpec,
    backbone: str,
    view: str,
    reps: int = 48,
    rng: np.random.Generator | None = None,
) -> LadderResult:
    """Evaluate every gate for every metric, given features for every rung.

    Shared by the render and geometry ladders so the two families are judged by
    literally the same code -- if `geom_kid` and `mv_dino_kid` disagree about a
    rung, that is a fact about what their features see, never about how their
    gates were computed.
    """
    rng = rng or np.random.default_rng(0)
    n_probe = len(probe_feats["real_heldout"])
    result = LadderResult(backbone=backbone, view=view, n=n_probe)

    for mname, fn in metrics.items():
        scores = {name: float(fn(ref_feats, f)) for name, f in probe_feats.items()}
        n_floor = min(n_probe, len(null_feats) - 1)
        floor_mean, floor_sd, floor_sem = _noise_floor(fn, ref_feats, null_feats,
                                                       n_floor, reps, rng)
        sd = floor_sd if np.isfinite(floor_sd) and floor_sd > 0 else float("nan")
        base = scores["real_heldout"]

        def sep(name: str, k: float) -> bool:
            return (name in scores and np.isfinite(sd)
                    and (scores[name] - base) > k * sd)

        half = max(8, n_floor // 2)
        m_small, _, sem_small = _noise_floor(fn, ref_feats, null_feats, half,
                                             reps, rng)
        # G8 asks whether two *means* differ, so it must be tested against the
        # standard error of their difference -- not against the spread of a
        # single draw, which is larger by a factor of sqrt(reps) and let a
        # known-biased estimator through. `legacy_cmmd`'s null mean halves every
        # time n doubles (0.173, 0.100, 0.058, 0.039 at n = 16/32/64/128); the
        # old form scored that 0.6 sd and passed it, this form scores it 14.8
        # standard errors and rejects it.
        drift_se = float(np.sqrt(sem_small ** 2 + floor_sem ** 2))

        noise_seq = [base] + [scores[k] for k in rungs.ordered_noise]
        # Blocking: is this measuring build quality at all?
        gates: Dict[str, bool] = {
            "G1_damage_ordering": base < scores["canon16"] < scores["canon8"],
            "G2_noise_ordering": all(a < b for a, b in zip(noise_seq, noise_seq[1:])),
            "G6_invariance": bool(np.isfinite(sd) and all(
                abs(scores[k] - base) < 1.0 * sd for k in rungs.invariant)),
            # 2 sd, not 1: this compares a single draw to a mean of many.
            "G7_self_consistency": bool(
                np.isfinite(sd) and abs(base - floor_mean) < 2.0 * sd),
            "G8_n_stability": bool(
                np.isfinite(drift_se) and drift_se > 0
                and abs(m_small - floor_mean) < 2.0 * drift_se),
        }
        # Non-blocking: how much damage does n buy the resolution to see?
        sensitivity = {flag: sep(rung, 3.0)
                       for flag, rung in rungs.sensitivity.items()}

        result.scores[mname] = {k: float(v) for k, v in scores.items()}
        result.noise[mname] = {"mean": float(floor_mean), "sd": float(floor_sd),
                               "sem": float(floor_sem),
                               "mean_half_n": float(m_small),
                               "drift_se": float(drift_se)}
        # numpy comparisons yield np.bool_, which json refuses to serialize.
        result.gates[mname] = {k: bool(v) for k, v in gates.items()}
        result.sensitivity[mname] = {k: bool(v) for k, v in sensitivity.items()}

    return result


def run_geometry_ladder(
    reference: Sequence[Structure],
    probe_pool: Sequence[Structure],
    n: int = 64,
    reps: int = 48,
    seed: int = 0,
    verbose: bool = True,
) -> LadderResult:
    """The same ladder, on geometric features instead of rendered ones.

    No GPU, no renderer, no network -- which is the point: the render tier is
    the half of the suite that can break silently (a texture pack, a driver, an
    EGL context), and a structural tier that runs anywhere is what makes the
    benchmark reproducible on a reviewer's machine.

    The whitening is fitted on the reference set only, exactly once, and reused
    for every rung. Refitting per rung would let a corruption move the feature
    scale and then be measured against its own distortion.
    """
    from blockgen.eval.bench import geometry as geom

    rng = np.random.default_rng(seed)
    ref_pool = [s.crop_to_non_air() for s in reference]
    all_probe = [s.crop_to_non_air() for s in probe_pool]
    order = np.random.default_rng(seed).permutation(len(all_probe))
    all_probe = [all_probe[i] for i in order]
    probe_src = all_probe[:n]
    null_pool = all_probe          # includes the probes; see `_noise_floor`
    if verbose:
        print(f"[geom-ladder] reference {len(ref_pool)}, probes {len(probe_src)}, "
              f"null pool {len(null_pool)}", flush=True)

    std = geom.Standardizer.fit(geom.descriptor(ref_pool))
    ref_feats = geom.geom_features(ref_pool, std)

    suite = pb.probe_suite(probe_src, np.random.default_rng(seed))
    suite["real_heldout"] = probe_src
    probe_feats = {name: geom.geom_features(structs, std)
                   for name, structs in suite.items()}
    null_feats = geom.geom_features(null_pool, std)

    sigma = D.median_bandwidth(ref_feats)
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "geom_kid": lambda r, p: D.kid(p, r),
        "geom_mmd_rbf": lambda r, p: D.mmd_rbf(p, r, sigma=sigma),
    }
    return score_rungs(metrics, ref_feats, probe_feats, null_feats, GEOMETRY_RUNGS,
                       backbone=GEOM_BACKBONE, view=geom_view_key(), reps=reps,
                       rng=rng)


#: The geometry tier is stored beside the render ladders under a backbone name
#: of its own, so `full.py`'s gate lookup needs no special case. The "view" slot
#: records the descriptor's shape, which is what would invalidate the result.
GEOM_BACKBONE = "geom"


def geom_view_key() -> str:
    from blockgen.eval.bench import geometry as geom
    return (f"patch{geom.PATCH}_orb{geom.N_ORBITS}"
            f"_t{len(geom.THICKNESS_BINS)}_h{geom.HEIGHT_BINS}")


def ladder_path(backbone: str, view_key: str, root: Path | str = LADDER_ROOT) -> Path:
    return Path(root) / f"{backbone}__{view_key}.json"


def save(result: LadderResult, root: Path | str = LADDER_ROOT) -> Path:
    path = ladder_path(result.backbone, result.view, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return path


def load(backbone: str, view_key: str,
         root: Path | str = LADDER_ROOT) -> LadderResult | None:
    path = ladder_path(backbone, view_key, root)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return LadderResult(backbone=d["backbone"], view=d["view"], n=d["n"],
                        scores=d["scores"], noise=d.get("noise_sd", {}),
                        gates=d["gates"], sensitivity=d.get("sensitivity", {}))


def render(result: LadderResult) -> str:
    names = list(next(iter(result.scores.values())).keys())
    order = (["real_heldout"] + [k for k in names if k != "real_heldout"])
    lines = [f"# Metric validation ladder — {result.backbone} / {result.view} "
             f"(n={result.n})", ""]
    lines.append("| metric | " + " | ".join(order)
                 + " | noise sd | validity | resolves |")
    lines.append("|" + "---|" * (len(order) + 4))
    for m, scores in result.scores.items():
        sd = result.noise[m]["sd"]
        verdict = "PASS" if result.passed(m) else f"FAIL {result.failed_gate(m)}"
        res = ",".join(k.replace("S3_resolves_", "").replace("S4_resolves_", "")
                       .replace("S5_resolves_", "") for k in result.resolves(m))
        cells = " | ".join(f"{scores[k]:.3f}" for k in order)
        lines.append(f"| {m} | {cells} | {sd:.3f} | **{verdict}** | {res or '—'} |")
    lines.append("")
    lines.append("Blocking gates: " + ", ".join(next(iter(result.gates.values())).keys()))
    lines.append("")
    lines.append("`resolves` lists the damage each metric separates from real at 3 sd "
                 "and this n. A blank is a power limit, not a defect -- the remedy is "
                 "more samples.")
    return "\n".join(lines)


def main() -> None:
    import argparse
    from blockgen.eval.bench import splits

    ap = argparse.ArgumentParser(description="Validate metrics against known damage.")
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="dinov2b", choices=sorted(ft.BACKBONES))
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--n", type=int, default=64, help="builds per probe rung")
    ap.add_argument("--n-ref", type=int, default=256)
    ap.add_argument("--reps", type=int, default=48,
                    help="draws used to estimate each metric's noise floor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--family", choices=("render", "geometry", "both"),
                    default="render",
                    help="render = DINO on views (needs a GPU); "
                         "geometry = voxel descriptors (CPU only)")
    args = ap.parse_args()

    view = ft.ViewConfig(px=args.px)
    split = splits.load_split(args.corpus, args.seed)
    # Restrict to builds the decimator actually damages, so canon16/canon8 mean
    # something; this is a named subset of the split, not a second split.
    def usable(s: Structure) -> bool:
        return max(s.shape) > 16 and 300 <= int(s.occupied_mask.sum()) <= 4000

    # Shuffle before capping, for the same category-contiguity reason as above.
    ref_all = splits.split_structures(split, "val", predicate=usable)
    rng = np.random.default_rng(args.seed)
    ref = [ref_all[i] for i in rng.permutation(len(ref_all))[:args.n_ref]]
    probe = splits.split_structures(split, "test", predicate=usable)
    print(f"[ladder] reference {len(ref)}, probe pool {len(probe)}")

    results = []
    if args.family in ("geometry", "both"):
        results.append(run_geometry_ladder(ref, probe, n=args.n, reps=args.reps,
                                           seed=args.seed))
    if args.family in ("render", "both"):
        results.append(run_ladder(ref, probe, view, args.backbone, n=args.n,
                                  reps=args.reps, device=args.device,
                                  seed=args.seed,
                                  include_legacy=not args.no_legacy))
    for result in results:
        path = save(result)
        print()
        print(render(result))
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
