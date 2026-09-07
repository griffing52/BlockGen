"""Cross-track evaluation suite for BlockGen.

Additive by construction: this package never modifies `blockgen/eval/{novelty,
validity,perceptual}.py`, so every number already recorded in results.md T1-T22
stays reproducible. Where a metric here supersedes an older one, the old
implementation is *reused* (novelty) or *documented as legacy* (perceptual.cmmd),
never patched.

Two tiers:
  * FAST  -- voxel-only: coherence, palette, novelty, dimension stats, and the
             structural realism metric `geom_kid`. No GPU, no rendering, no
             network; runs in seconds and is the half that works on a reviewer's
             laptop.
  * FULL  -- render + DINOv2 features (KID/MMD, density/coverage, DINO-NN
             memorization, prompt faithfulness). Gated on `ladder.py`: a metric
             that has not passed the corruption ladder is emitted as null, never
             silently printed.

Two realism metrics, not one, because the render tier is measurably blind to
structure: filling every enclosed air cell in the corpus leaves `mv_dino_kid`
statistically indistinguishable from held-out real, while `geom_kid` puts it 200
real-sample spreads away. The converse also holds -- `geom_kid` reads occupancy
only and a material permutation moves it by exactly zero. They are complementary
halves and the ladder gates each on the rungs it is built to see.

`composite.BlockScore` aggregates them by *worst pillar*, never by average, with
memorization and mode collapse as disqualification gates rather than terms --
because a realism-only ranking is won by reciting the training set (T23c).
"""
