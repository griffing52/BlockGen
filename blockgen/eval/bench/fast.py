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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as ld
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
        self.geometry = geom.geometry_table(self.val)
        self.exact_keys = pal.vocabulary(self.train, "exact")
        self.family_keys = pal.vocabulary(self.train, "family")
        # The geometric descriptor is whitened on the reference and reused for
        # every arm, so the kernel is on one scale across the whole run. Fitting
        # per arm would let each generator define its own feature units.
        self.geom_std = geom.Standardizer.fit(geom.descriptor(self.val))
        self.geom_ref = geom.geom_features(self.val, self.geom_std)
        self.geom_ladder = ld.load(ld.GEOM_BACKBONE, ld.geom_view_key())
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


#: How many example builds `score_fast` picks per arm when the runner asks for
#: them. Matches the `--examples` default; the runner passes its own value
#: through `examples_k` when the flag says otherwise.
EXAMPLES_K = 8


def _structures_sha(structures: Sequence[Structure]) -> Optional[str]:
    """Content identity of an arm's SCORED builds, or None if it has none.

    The only cross-run identity an arm will ever have: two runs that scored the
    same 128 builds report the same string, whatever the files were called. It
    is also, deliberately, the same recipe `tools/lab/renders.py` content-
    addresses its thumbnails with, so an arm's id here is an id the render cache
    already knows.

    Imported inside the function because `features` imports
    `blockgen.eval.perceptual`, which imports torch at module scope, and none of
    the arithmetic in this tier needs it. (`ladder` already drags torch onto the
    fast path today; that is a debt to pay down, not a licence to add a second
    one.) The hash itself is a pure SHA-1 over `block_ids` -- no renderer, no
    GPU -- so a failure to import is a failure to *name* the arm, never a
    failure to score it.
    """
    if not structures:
        return None
    try:
        from blockgen.eval.bench import features as ftr
        return ftr.structures_sha(structures)
    except Exception:
        return None


def _source_run_id(path: Optional[str],
                   provenance: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The `run_*` directory an arm's inputs came out of, or None.

    One derived string, and it is what turns a flat leaderboard into a graph:
    from a row, back to the training run that produced the checkpoint that
    produced these builds. The checkpoint is asked first because it is the
    tighter edge -- an arm's npz usually lives under `outputs/bench_arms/`,
    which names no run, while its checkpoint path names the run that trained it.
    """
    for candidate in ((provenance or {}).get("checkpoint"), path):
        if not candidate:
            continue
        for part in str(candidate).replace("\\", "/").split("/"):
            if part.startswith("run_"):
                return part
    return None


def score_fast(arm: sc.ArmSpec, ref: FastReference,
               card: sc.Scorecard | None = None,
               sink: Dict[str, np.ndarray] | None = None,
               *,
               examples: Optional[list] = None,
               examples_k: int = EXAMPLES_K,
               examples_slate: Sequence[str] = (),
               ) -> Dict[str, Dict[str, sc.Metric]]:
    """Score one arm on the FAST tier. Returns scorecard blocks.

    `sink` collects each arm's geometric features so the runner can run
    head-to-head tests without recomputing them. Deliberately a side channel:
    the features are an implementation detail of the score, not part of it, and
    putting them in the scorecard would put a 128x77 array in every JSON.

    `examples` is the second such channel, and its protocol is total: when it is
    not None, EXACTLY ONE tuple `(non_empty, rows)` is appended for every arm
    this function is called with -- `([], [])` for an arm that produced nothing,
    so a caller may index `sink[0]` without first asking whether the arm worked.
    `rows` are `examples_k` indices into `non_empty`, seeded from `ctx.rng(9)`
    and sorted ascending, so the strip the lab renders reads in source order and
    is identical on a re-run. The structures themselves are handed over rather
    than written here because this tier must not import a renderer or an IO
    layer; the runner truncates the list to the chosen rows immediately, which
    is what keeps peak retention at k x n_arms instead of every arm's full set.
    """
    ctx = ref.ctx
    structs: List[Structure] = [s.crop_to_non_air() for s in arm.load()]
    n = len(structs)
    # `keep` rather than a bare comprehension so the arm's prompts can be
    # re-aligned to the surviving builds. `arm.prompts` is parallel to the
    # UNFILTERED list, so after dropping the empties, prompt j belongs to
    # `non_empty[j]` only via `keep[j]` -- and a slate that matched on the wrong
    # index would render one model's cottage under another's prompt, which is a
    # worse failure than showing no slate at all.
    keep = [i for i, s in enumerate(structs) if int(s.occupied_mask.sum()) > 0]
    non_empty = [structs[i] for i in keep]
    n_empty = n - len(non_empty)
    kept_prompts: List[str] = (
        [str(arm.prompts[i]) if i < len(arm.prompts) else "" for i in keep]
        if arm.prompts else [])

    rng = ctx.rng()
    warn = card.warn if card else (lambda m: None)
    if n < ctx.min_n:
        warn(f"arm {arm.name!r}: n={n} < min_n={ctx.min_n}; "
             f"intervals are not interpretable")
    if n_empty:
        warn(f"arm {arm.name!r}: {n_empty}/{n} samples are empty")

    # `source` stays exactly `arm.npz or "in-memory"`: five consumers key off
    # that literal -- the leaderboard's dataset join, `bench_report`,
    # `human_study_analyze`, the Curate link and the thumbnail fallback -- so it
    # is the one string in this block that may never be improved.
    provenance = arm.provenance()
    meta = {
        "track": arm.track, "n": n, "n_empty": n_empty,
        "source": arm.npz or "in-memory",
        "has_prompts": bool(arm.prompts),
        "palette_oov_frac_exact": pal.palette_oov_frac(non_empty, ref.exact_keys, "exact")
                                  if non_empty else 0.0,
        "palette_oov_frac_family": pal.palette_oov_frac(non_empty, ref.family_keys, "family")
                                   if non_empty else 0.0,
        "family_unclassified_frac": pal.family_other_frac(non_empty) if non_empty else 0.0,
        "kind": arm.kind,
        "origin": arm.origin,
        "source_run_id": _source_run_id(arm.npz, provenance),
        # Over the SCORED builds -- cropped and empty-filtered -- because the
        # identity that matters is of what the numbers were computed on.
        "structures_sha": _structures_sha(non_empty),
        "provenance": provenance,
    }

    if examples is not None:
        # Appended before the empty-arm return below, so the protocol holds for
        # every arm. Failure here warns and yields nothing: no amount of missing
        # metadata may sink a two-hour scoring run.
        try:
            from blockgen.eval.bench import examples as ex
            examples.append((non_empty, ex.pick_slate(
                len(non_empty), examples_k, ctx.rng(9),
                prompts=kept_prompts, slate=examples_slate)))
        except Exception as exc:
            examples.append(([], []))
            warn(f"arm {arm.name!r}: example selection failed: "
                 f"{type(exc).__name__}: {exc}")

    if not non_empty:
        reason = "arm produced no non-empty structures"
        warn(f"arm {arm.name!r}: {reason}")
        nulls = {k: sc.skipped(reason) for k in
                 ("palette_jsd_exact", "palette_jsd_family", "palette_cooccur_jsd",
                  "palette_size_w1", "n_blocks_w1", "max_dim_w1")}
        return {"meta": meta, "dataset_stats": nulls,
                "novelty": {k: sc.skipped(reason) for k in
                            ("voxel_nn_iou_mean", "voxel_dup_rate", "voxel_diversity")},
                "realism": {"geom_kid": sc.skipped(reason)},
                "coherence": {}, "geometry_scalars": {}}

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

    # --- geometry: the structural counterpart of the render tier -----------
    # This is the whole reason the geometry tier lives in FAST: `geom_kid` is a
    # validated distribution metric that needs no GPU, no renderer and no
    # network, so a run can be ranked on structure in CI and on a reviewer's
    # laptop. See `geometry` for what it is blind to.
    geometry_scalars = geom.geometry_report(
        non_empty, ref.val, n_boot=ctx.n_boot, alpha=ctx.alpha, rng=rng,
        real_table=ref.geometry)

    gate = None
    if ref.geom_ladder is None:
        gate = "no geometry ladder; run `bench.ladder --family geometry`"
    elif not ref.geom_ladder.passed("geom_kid"):
        gate = f"geom_kid failed {ref.geom_ladder.failed_gate('geom_kid')}"
    if gate:
        realism = {"geom_kid": sc.skipped(gate)}
    elif len(non_empty) < 2:
        realism = {"geom_kid": sc.skipped("fewer than 2 non-empty structures")}
    else:
        gf = geom.geom_features(non_empty, ref.geom_std)
        if sink is not None:
            sink[arm.name] = gf
        realism = {"geom_kid": sc.with_mmd(sc.from_ci(
            stats.subsample_ci(lambda a, b: geom.geom_kid(a, b), gf, ref.geom_ref,
                               frac=0.8, n_rep=200, rng=rng),
            direction="lower_better", x=1000,
            resolves=ref.geom_ladder.resolves("geom_kid"),
            note="unbiased polynomial-kernel MMD on D4-invariant geometric "
                 "descriptors; blind to material by construction"))}

    return {"meta": meta, "coherence": coherence, "geometry_scalars": geometry_scalars,
            "realism": realism, "dataset_stats": dataset_stats, "novelty": novelty}


#: One line per control rung: what it IS, in a sentence, for a reader looking at
#: a leaderboard row. This is the only place that prose lives -- `control_arms`
#: points here rather than restating it, because the version that lived in a
#: docstring reached no screen, and a copy in two places is a copy that drifts.
#: Shipped verbatim as `meta.provenance.recipe`, which is what the lab renders
#: for an arm that has nothing on disk to link to.
#:
#: Keys must stay in step with the arms `control_arms` actually builds.
CONTROL_RECIPES: Dict[str, str] = {
    "real_test":
        "genuinely held-out real builds from the test split, disjoint from the "
        "val reference they are scored against, so the floor is a fresh real "
        "sample rather than the reference scored against itself (which is an "
        "exact zero everywhere and calibrates nothing)",
    "real@canon16":
        "real test builds requantised to a 16-block canonical palette -- known "
        "damage of a known size, measured in F16 as deleting 86.5% of blocks "
        "and whole roofs",
    "real@canon8":
        "the same canonicalisation taken down to 8 blocks: the same kind of "
        "damage, harder, so the two rungs together show whether a metric "
        "orders damage as well as detecting it",
    "real_shuffled_materials":
        "a real build with its block multiset permuted across its own occupied "
        "cells; a negative control for this tier, because placement moves and "
        "the multiset does not, so every voxel palette statistic is blind to it "
        "by construction and its palette row should sit on the floor -- the "
        "FULL tier is where this probe bites",
    "real@monochrome":
        "a real build repainted in one material: geometry intact, palette "
        "collapsed -- the converse of shuffling, and a palette metric that "
        "cannot separate it is not working",
    "real@solidify":
        "every enclosed air cell of a held-out test build filled; a median 31% "
        "more blocks, every room deleted, the silhouette untouched -- the "
        "control that tells the two realism tiers apart, since at n=128 "
        "MV-DINO-KID cannot separate it from real and geom_kid puts it 205 "
        "spreads away",
    "real@single_mode":
        "one real test build repeated n times; it is drawn from the real "
        "distribution, so no distribution distance punishes it and only the "
        "diversity gate stands between the leaderboard and a generator that "
        "memorises a single house",
    "train_verbatim":
        "literal copies of training builds; the memorization detector must fire "
        "at voxel_dup_rate == 1.0 on this arm, and anything less means it would "
        "miss a model that recites its training set",
}


def control_arms(test: Sequence[Structure], train: Sequence[Structure] | None = None,
                 n: int | None = None,
                 rng: np.random.Generator | None = None,
                 *, corpus: str = "", seed: int = 0) -> List[sc.ArmSpec]:
    """Real-data controls, as ordinary arms.

    Eight rungs of real data -- seven, without a train split -- most of them
    carrying known, deliberate damage, all of them pushed through the identical
    scoring path as every submission. That is what makes the numbers on this leaderboard
    readable at all: a distance means nothing until a floor, a ceiling and
    several calibrated steps between them have been measured under *this* run's
    protocol -- this corpus, this split, this n, this reference. Importing a
    floor from another run would rank arms against conditions they were never
    measured under, which is why the controls are re-scored every time rather
    than cached.

    **What each rung is, is one line per rung in `CONTROL_RECIPES`, which is
    also what the leaderboard shows.** The prose lives there and only there, and
    every arm built here carries its own line in `provenance_override` -- an
    in-memory arm has no file to point a reader at, so the recipe is the whole
    of its provenance.

    Two of the rungs measure no damage at all. They exist because a benchmark can
    be won by a degenerate strategy that no *distribution* distance objects to --
    reciting the training set, or finding one good house and emitting it n times
    -- so those two are caught by gates rather than by distances, and a gate
    nobody exercises is a gate nobody knows is broken.

    `corpus` and `seed` are recorded in each arm's provenance, not used to
    build anything; both default so existing callers are unaffected.
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
        sc.ArmSpec("real@solidify", track="control",
                   structures=[probes.solidify(s) for s in pool]),
        sc.ArmSpec("real@single_mode", track="control",
                   structures=[pool[0]] * len(pool)),
    ]
    if train:
        copies = [s.crop_to_non_air() for s in train]
        take = min(len(copies), n or len(pool))
        arms.append(sc.ArmSpec(
            "train_verbatim", track="control",
            structures=[copies[i] for i in rng.permutation(len(copies))[:take]]))

    n_requested = int(n) if n is not None else len(pool)
    for arm in arms:
        arm.provenance_override = {
            "writer": "in_memory",
            "builder": "blockgen.eval.bench.fast.control_arms",
            "recipe": CONTROL_RECIPES[arm.name],
            # `train_verbatim` is the one rung drawn from the other side of the
            # split, and saying "test" here would be a lie in the field whose
            # entire job is to say where the builds came from.
            "drawn_from": f"split:{corpus}:"
                          + ("train" if arm.name == "train_verbatim" else "test"),
            "n_requested": n_requested,
            "seed": seed,
        }
    return arms
