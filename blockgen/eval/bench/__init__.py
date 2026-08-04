"""Cross-track evaluation suite for BlockGen.

Additive by construction: this package never modifies `blockgen/eval/{novelty,
validity,perceptual}.py`, so every number already recorded in results.md T1-T22
stays reproducible. Where a metric here supersedes an older one, the old
implementation is *reused* (novelty) or *documented as legacy* (perceptual.cmmd),
never patched.

Two tiers:
  * FAST  -- voxel-only (coherence, palette, novelty, dimension stats). No GPU,
             no rendering, runs in seconds.
  * FULL  -- render + DINOv2 features (KID/MMD, density/coverage, DINO-NN
             memorization, prompt faithfulness). Gated on `ladder.py`: a metric
             that has not passed the corruption ladder is emitted as null, never
             silently printed.
"""
