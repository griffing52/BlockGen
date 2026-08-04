"""FULL tier: render, embed, and score in feature space -- but only what passed.

Rendering is what makes cross-resolution arms comparable at all: a 16^3 build
and a 32^3 build both become a 224x224 image, so resolution becomes a property
of the content rather than of the metric (T20).

Nothing here reports a metric that has not passed `ladder.py`. A gated-out
metric is emitted as null carrying the name of the gate it failed, so a reader
sees "fd: — (G8_n_stability)" rather than a blank cell that looks like zero.
Set `require_ladder=False` only for exploration, never for a results table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import faithfulness as fth
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import ladder as ld
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import stats
from blockgen.utils.data import Structure


@dataclass
class FullReference:
    """Feature-space reference, built once and shared by every arm."""
    ctx: sc.BenchContext
    view: ft.ViewConfig
    backbone: str
    ref_feats: np.ndarray          # [R, D] pooled, the KID/MMD/PRDC reference
    train_feats: np.ndarray        # [T, D] pooled, the memorization search space
    calib_feats: np.ndarray        # [C, D] pooled, held-out real for calibration
    sigma: float
    ladder: ld.LadderResult | None = None

    def gate(self, metric: str) -> str | None:
        """Reason this metric may not be reported, or None if it is valid.

        Only *validity* failures suppress a value. A metric that is measuring
        the right thing but lacks the power to resolve some damage at this n is
        still reported, with its resolution limits attached (see `resolves`).
        """
        if self.ladder is None:
            return "no ladder result for this backbone/view"
        if metric not in self.ladder.gates:
            return "metric not scored by the ladder"
        return self.ladder.failed_gate(metric)

    def resolves(self, metric: str) -> List[str]:
        """Damage levels this metric separates from real at 3 sd, at ladder n."""
        return self.ladder.resolves(metric) if self.ladder else []


def build_reference(
    train: Sequence[Structure], val: Sequence[Structure], test: Sequence[Structure],
    ctx: sc.BenchContext, view: ft.ViewConfig = ft.ViewConfig(),
    backbone: str = "dinov2b", corpus: str = "houses_32", device: str = "cuda",
    require_ladder: bool = True, verbose: bool = True,
) -> FullReference:
    def feats(split_name: str, structs: Sequence[Structure]) -> np.ndarray:
        key = ft.FeatureKey(corpus, split_name, view.key(), backbone)
        return ft.pooled(ft.load_or_build(key, list(structs), view, device=device,
                                          verbose=verbose), "mean")

    ref_feats = feats("val", val)
    train_feats = feats("train", train)
    calib_feats = feats("test", test)

    result = ld.load(backbone, view.key())
    if result is None and require_ladder:
        raise RuntimeError(
            f"no validation ladder for {backbone}/{view.key()}. Run:\n"
            f"  python -m blockgen.eval.bench.ladder --backbone {backbone}\n"
            "Metrics must be validated against known damage before they are "
            "allowed to rank anything (see results.md T20/T21).")

    return FullReference(ctx=ctx, view=view, backbone=backbone, ref_feats=ref_feats,
                         train_feats=train_feats, calib_feats=calib_feats,
                         sigma=D.median_bandwidth(ref_feats), ladder=result)


def score_full(arm: sc.ArmSpec, ref: FullReference, corpus: str = "houses_32",
               device: str = "cuda", card: sc.Scorecard | None = None,
               captions: Dict[str, List[str]] | None = None,
               ) -> Dict[str, Dict[str, sc.Metric]]:
    ctx = ref.ctx
    structs = [s.crop_to_non_air() for s in arm.load()]
    structs = [s for s in structs if s.occupied_mask.any()]
    n = len(structs)
    warn = card.warn if card else (lambda m: None)

    if n < 2:
        reason = "fewer than 2 non-empty structures"
        warn(f"arm {arm.name!r}: {reason}")
        return {"realism": {"mv_dino_kid": sc.skipped(reason)}}

    rng = ctx.rng()
    feats = ft.pooled(ft.embed_views(structs, ref.view, ref.backbone, device,
                                     verbose=False), "mean")

    too_small = n < ctx.min_n
    if too_small:
        warn(f"arm {arm.name!r}: n={n} < min_n={ctx.min_n}; distribution metrics "
             f"are reported but their intervals are not interpretable")

    def gated(name: str, fn, direction: str = "lower_better", **extra) -> sc.Metric:
        reason = ref.gate(name)
        if reason:
            return sc.skipped(f"{name} failed {reason}", direction=direction)
        triple = stats.subsample_ci(fn, feats, ref.ref_feats, frac=0.8, n_rep=200,
                                    rng=rng)
        return sc.from_ci(triple, direction=direction,
                          resolves=ref.resolves(name), **extra)

    realism = {
        "mv_dino_kid": gated("kid", lambda a, b: D.kid(a, b), x=1000),
        "mv_dino_mmd_rbf": gated("mmd_rbf", lambda a, b: D.mmd_rbf(a, b, sigma=ref.sigma),
                                 sigma=ref.sigma, x=1000),
        # Frechet distance is n-dependent even when it passes, so it carries the
        # n it is valid at and the renderer refuses to line it up across arms.
        "mv_fd_dino": gated("fd", lambda a, b: D.fd(a, b),
                            comparable_only_at_n=n, shrinkage=0.1),
    }

    prdc = D.prdc(ref.ref_feats, feats, k=5)
    fidelity: Dict[str, sc.Metric] = {}
    for key, ladder_name, direction in (
            ("density", "one_minus_density", "higher_better"),
            ("coverage", "one_minus_coverage", "higher_better"),
            ("precision", "one_minus_precision", "higher_better"),
            ("recall", "one_minus_recall", "higher_better")):
        reason = ref.gate(ladder_name)
        if reason:
            fidelity[key] = sc.skipped(f"{key} failed {reason}", direction=direction)
        else:
            fidelity[key] = sc.metric(prdc[key], direction=direction, k=5,
                                      resolves=ref.resolves(ladder_name))

    mem = D.memorization_report(feats, ref.train_feats, ref.calib_feats)
    novelty = {
        "dino_nn_percentile": sc.from_ci(
            stats.bootstrap_ci(mem["nn_percentile"], n_boot=ctx.n_boot, rng=rng),
            direction="higher_better",
            note="percentile within held-out-real-vs-train distances; "
                 "low = closer to training data than a real house ever is"),
        "dino_nn_dist_mean": sc.from_ci(
            stats.bootstrap_ci(mem["nn_dist"], n_boot=ctx.n_boot, rng=rng),
            direction="higher_better", metric="cosine"),
        "dino_nn_dist_p05": sc.metric(
            float(np.percentile(mem["nn_dist"], 5)), direction="higher_better",
            note="the most-memorized tail"),
        "dino_dup_rate": sc.metric(
            float(mem["dup_rate"]), direction="lower_better",
            threshold=float(mem["dup_threshold"]), threshold_pctile=1.0),
    }

    blocks: Dict[str, Dict[str, sc.Metric]] = {
        "realism": realism, "fidelity": fidelity, "novelty": novelty}

    prompts = arm.prompts
    if prompts and any(prompts):
        rep = fth.evaluate(structs, list(prompts)[:len(structs)], view=ref.view,
                           backbone=ctx_text_backbone(ctx), device=device, rng=rng,
                           n_boot=ctx.n_boot)
        blocks["faithfulness"] = {
            "retrieval_acc": sc.metric(rep.retrieval_acc, rep.retrieval_ci,
                                       "higher_better", chance=rep.chance,
                                       n_distractors=rep.n_distractors,
                                       mrr=rep.retrieval_mrr,
                                       caption_idx=rep.caption_idx),
            "mv_clipscore": sc.metric(rep.mv_clipscore, rep.mv_clipscore_ci,
                                      "higher_better", backbone=rep.backbone),
            "material_agreement": sc.metric(rep.material_agreement,
                                            rep.material_agreement_ci,
                                            "higher_better",
                                            coverage=rep.material_coverage),
        }
    return blocks


def ctx_text_backbone(ctx: sc.BenchContext) -> str:
    return getattr(ctx, "text_backbone", "clipL")
