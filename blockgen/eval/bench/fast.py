"""FAST tier: everything measurable without a GPU, a renderer, or a network.

Covers coherence, palette statistics, dimension distributions and voxel-space
novelty. Runs in seconds on CPU, so it can gate a training run rather than being
saved for the end.

Voxel novelty delegates to `blockgen.eval.novelty.evaluate_novelty` *unchanged*,
so `mean_nn_iou` / `duplicate_rate` / `diversity` here are bit-identical to the
numbers already recorded in results.md T1-T22. Its documented weakness -- IoU on
a fixed canonical grid is not comparable across resolutions (T20) -- is why the
FULL tier adds a DINO-feature nearest-neighbour metric rather than replacing this
one: the two answer the same question at different granularities and disagreeing
is informative.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from blockgen.eval.bench import palette as pal
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import stats, topology
from blockgen.eval.novelty import (NoveltyReport, _iou_matrix, evaluate_novelty,
                                   voxelize_occupancy)
from blockgen.utils.data import Structure
from blockgen.utils.serialize import BlockVocab, build_block_vocab


def build_vocab(train: Sequence[Structure], grid: int) -> BlockVocab:
    """Class vocabulary fixed from the reference side, shared by every arm.

    Building it per-arm would give each generator a private alphabet and make the
    voxel metrics incomparable across arms.
    """
    return build_block_vocab(list(train), max_dim=grid)


class FastReference:
    """Precomputed reference statistics, shared across arms.

    Constructed once per run. Every arm is scored against this exact object, so
    two arms can never be compared against differently-derived references.
    """

    def __init__(self, train: Sequence[Structure], val: Sequence[Structure],
                 ctx: sc.BenchContext, vocab: BlockVocab | None = None):
        self.ctx = ctx
        self.train = list(train)
        self.val = list(val)
        self.vocab = vocab if vocab is not None else build_vocab(self.train, ctx.grid)
        self.coherence = topology.coherence_table(self.val)
        self.exact_keys = pal.vocabulary(self.train, "exact")
        self.family_keys = pal.vocabulary(self.train, "family")
        # Voxelizing 1863 training builds is the dominant FAST-tier cost and is
        # identical for every arm, so it is done once here rather than inside
        # each `evaluate_novelty` call.
        self.train_occ, self.train_cls = voxelize_occupancy(
            self.train, ctx.grid, self.vocab)


def novelty_cached(generated: Sequence[Structure], ref: FastReference,
                   k: int = 3, dup_threshold: float = 0.95) -> NoveltyReport:
    """`evaluate_novelty` with the reference voxelization hoisted out.

    Numerically identical to `blockgen.eval.novelty.evaluate_novelty` -- pinned
    by `tests/eval/test_fast.py::test_novelty_cached_matches_published`. It is
    reimplemented rather than refactored so `novelty.py` stays byte-identical and
    every number in results.md T1-T22 remains reproducible from the original
    code path.

    `validity_rate` is omitted: the FAST tier reports connectivity through
    `topology.coherence_report`, against the real distribution rather than
    against an implied 1.0.
    """
    gen_occ, gen_cls = voxelize_occupancy(generated, ref.ctx.grid, ref.vocab)
    train_occ, train_cls = ref.train_occ, ref.train_cls

    iou = _iou_matrix(gen_occ, train_occ)
    k = min(k, train_occ.shape[0])
    nn_index = np.argsort(-iou, axis=1)[:, :k]
    nn_iou_topk = np.take_along_axis(iou, nn_index, axis=1)
    nn_iou = nn_iou_topk[:, 0] if k > 0 else np.zeros(len(generated))

    block_agreement = np.zeros(len(generated))
    for i in range(len(generated)):
        a, b = gen_cls[i], train_cls[nn_index[i, 0]]
        union = (a > 0) | (b > 0)
        if union.sum() > 0:
            block_agreement[i] = float(((a == b) & union).sum()) / float(union.sum())

    if len(generated) > 1:
        gg = _iou_matrix(gen_occ, gen_occ)
        n = len(generated)
        diversity = float(1.0 - (gg.sum() - np.trace(gg)) / (n * (n - 1)))
    else:
        diversity = float("nan")

    return NoveltyReport(
        nn_iou=nn_iou, nn_index=nn_index, nn_iou_topk=nn_iou_topk,
        duplicate_rate=float(np.mean(nn_iou >= dup_threshold)),
        mean_nn_iou=float(np.mean(nn_iou)), diversity=diversity,
        validity_rate=float("nan"), train_validity_rate=float("nan"),
        block_agreement=block_agreement)


def score_fast(arm: sc.ArmSpec, ref: FastReference,
               card: sc.Scorecard | None = None) -> Dict[str, Dict[str, sc.Metric]]:
    """Score one arm on the FAST tier. Returns scorecard blocks."""
    ctx = ref.ctx
    structs: List[Structure] = [s.crop_to_non_air() for s in arm.load()]
    n = len(structs)
    non_empty = [s for s in structs if int(s.occupied_mask.sum()) > 0]
    n_empty = n - len(non_empty)

    rng = ctx.rng()
    warn = card.warn if card else (lambda m: None)
    if n < ctx.min_n:
        warn(f"arm {arm.name!r}: n={n} < min_n={ctx.min_n}; "
             f"intervals are not interpretable")
    if n_empty:
        warn(f"arm {arm.name!r}: {n_empty}/{n} samples are empty")

    meta = {
        "track": arm.track, "n": n, "n_empty": n_empty,
        "source": arm.npz or "in-memory",
        "has_prompts": bool(arm.prompts),
        "palette_oov_frac_exact": pal.palette_oov_frac(non_empty, ref.exact_keys, "exact")
                                  if non_empty else 0.0,
        "palette_oov_frac_family": pal.palette_oov_frac(non_empty, ref.family_keys, "family")
                                   if non_empty else 0.0,
        "family_unclassified_frac": pal.family_other_frac(non_empty) if non_empty else 0.0,
    }

    if not non_empty:
        reason = "arm produced no non-empty structures"
        warn(f"arm {arm.name!r}: {reason}")
        nulls = {k: sc.skipped(reason) for k in
                 ("palette_jsd_exact", "palette_jsd_family", "palette_cooccur_jsd",
                  "palette_size_w1", "n_blocks_w1", "max_dim_w1")}
        return {"meta": meta, "dataset_stats": nulls,
                "novelty": {k: sc.skipped(reason) for k in
                            ("voxel_nn_iou_mean", "voxel_dup_rate", "voxel_diversity")},
                "coherence": {}}

    # --- coherence: always reported against the real distribution -----------
    coherence = topology.coherence_report(
        non_empty, ref.val, n_boot=ctx.n_boot, alpha=ctx.alpha, rng=rng,
        real_table=ref.coherence)

    # --- palette ------------------------------------------------------------
    # Histograms are built once per level and the bootstrap then resamples
    # *rows*. Rebuilding them inside each draw re-derives the reference
    # distribution hundreds of times over and dominates the whole tier.
    rows = np.arange(len(non_empty))

    def _ci(fn, **extra):
        # Every statistic in this block is a divergence or a distance, so the
        # interval is clamped at zero.
        return sc.from_ci(stats.resample_ci(fn, rows, n_boot=ctx.n_boot,
                                            alpha=ctx.alpha, rng=rng,
                                            bounds=(0.0, None)),
                          direction="lower_better", **extra)

    dataset_stats: Dict[str, sc.Metric] = {}
    for level, label in (("exact", "palette_jsd_exact"),
                         ("family", "palette_jsd_family")):
        keys = pal.union_vocabulary(non_empty, ref.val, level)
        gen_counts, _, _ = pal.palette_matrix(non_empty, level, keys)
        ref_counts, _, _ = pal.palette_matrix(ref.val, level, keys)
        ref_macro = pal.macro_distribution(ref_counts)
        dataset_stats[label] = _ci(
            lambda idx, g=gen_counts, m=ref_macro: pal.jsd_from_counts(g[idx], m),
            level=level, n_symbols=len(keys))

    # Co-occurrence: top-K chosen by reference frequency, alphabet is the union.
    keys = pal.union_vocabulary(non_empty, ref.val, "family")
    ref_counts, _, _ = pal.palette_matrix(ref.val, "family", keys)
    if len(keys) > 48:
        keys = [keys[i] for i in np.argsort(-ref_counts.sum(0))[:48]]
        ref_counts, _, _ = pal.palette_matrix(ref.val, "family", keys)
    gen_counts, _, _ = pal.palette_matrix(non_empty, "family", keys)
    gen_pres, ref_pres = pal.presence_matrix(gen_counts), pal.presence_matrix(ref_counts)
    ref_co = pal.cooccurrence_from_presence(ref_pres).ravel()
    dataset_stats["palette_cooccur_jsd"] = _ci(
        lambda idx, p=gen_pres, r=ref_co: stats.jsd(
            pal.cooccurrence_from_presence(p[idx]).ravel(), r), topk=len(keys))

    gen_sizes = (gen_counts > 0).sum(1).astype(float)
    ref_sizes = (ref_counts > 0).sum(1).astype(float)
    dataset_stats["palette_size_w1"] = _ci(
        lambda idx, g=gen_sizes, r=ref_sizes: stats.w1(g[idx], r),
        real_mean=float(ref_sizes.mean()))

    gen_tab = topology.coherence_table(non_empty)
    for key, label in (("n_blocks", "n_blocks_w1"), ("max_dim", "max_dim_w1"),
                       ("bbox_fill", "bbox_fill_w1")):
        dataset_stats[label] = _ci(
            lambda idx, c=gen_tab[key], k=key: stats.w1(c[idx], ref.coherence[k]),
            real_mean=float(ref.coherence[key].mean()))

    # --- voxel novelty (delegated verbatim to the published implementation) --
    rep = novelty_cached(non_empty, ref, k=3, dup_threshold=ctx.dup_threshold)
    novelty = {
        "voxel_nn_iou_mean": sc.from_ci(
            stats.bootstrap_ci(rep.nn_iou, n_boot=ctx.n_boot, alpha=ctx.alpha, rng=rng),
            direction="lower_better", grid=ctx.grid,
            note="IoU to nearest TRAIN build; high = memorized"),
        "voxel_dup_rate": sc.from_ci(
            stats.bootstrap_ci((rep.nn_iou >= ctx.dup_threshold).astype(float),
                               n_boot=ctx.n_boot, alpha=ctx.alpha, rng=rng),
            direction="lower_better", threshold=ctx.dup_threshold),
        "voxel_diversity": sc.metric(rep.diversity, direction="higher_better"),
        "voxel_block_agreement": sc.from_ci(
            stats.bootstrap_ci(rep.block_agreement, n_boot=ctx.n_boot,
                               alpha=ctx.alpha, rng=rng),
            direction="lower_better"),
    }

    return {"meta": meta, "coherence": coherence, "dataset_stats": dataset_stats,
            "novelty": novelty}


def control_arms(test: Sequence[Structure], train: Sequence[Structure] | None = None,
                 n: int | None = None,
                 rng: np.random.Generator | None = None) -> List[sc.ArmSpec]:
    """Real-data controls, as ordinary arms.

    Five rungs, each answering a specific question about the metrics:

    * ``real_test``   -- the floor. Genuinely held-out real builds, *disjoint*
      from the ``val`` reference they are scored against. (Scoring the reference
      against itself gives an exact zero everywhere and calibrates nothing.)
    * ``real@canon16`` / ``real@canon8`` -- known damage. F16 measured canon-16
      as deleting 86.5% of blocks and whole roofs.
    * ``real_shuffled_materials`` -- a **negative** control for this tier.
      Shuffling permutes placement but not the multiset, so every voxel palette
      statistic is blind to it by construction. Its palette row should sit on the
      floor; if it does not, something is reading spatial information it should
      not. The FULL tier is where this probe bites.
    * ``real@monochrome`` -- the converse: geometry intact, palette collapsed to
      one material. A palette metric that cannot separate this is not working.
    * ``train_verbatim`` -- literal copies of training builds. The memorization
      detector must fire at ``voxel_dup_rate == 1.0``; anything less means it
      would miss a model that recites its training set.
    """
    from blockgen.eval.bench import probes

    rng = rng or np.random.default_rng(0)
    pool = [s.crop_to_non_air() for s in test]
    if n is not None and n < len(pool):
        pool = [pool[i] for i in rng.permutation(len(pool))[:n]]

    arms = [
        sc.ArmSpec("real_test", track="control", structures=pool),
        sc.ArmSpec("real@canon16", track="control", structures=probes.canon(pool, 16)),
        sc.ArmSpec("real@canon8", track="control", structures=probes.canon(pool, 8)),
        sc.ArmSpec("real_shuffled_materials", track="control",
                   structures=[probes.shuffle_materials(s, rng) for s in pool]),
        sc.ArmSpec("real@monochrome", track="control",
                   structures=[probes.monochrome(s) for s in pool]),
    ]
    if train:
        copies = [s.crop_to_non_air() for s in train]
        take = min(len(copies), n or len(pool))
        arms.append(sc.ArmSpec(
            "train_verbatim", track="control",
            structures=[copies[i] for i in rng.permutation(len(copies))[:take]]))
    return arms
