"""Uncertainty and distribution-distance primitives.

Why this module exists: at n=16 (the current `--samples` default) the standard
deviation of MV-DINO-KID across resamples is ~0.04, while the gap between real
held-out builds and canon-16 decimation is only 0.12. Point estimates at that
scale are not interpretable, so *every* metric in the suite ships with an
interval and the scorecard schema has no slot for a bare float.

Two resampling schemes, and picking the wrong one is a real bug:

* **Per-structure metrics** (coherence, nn_iou, retrieval accuracy, CLIPScore)
  are means over independent per-sample values -> ordinary percentile bootstrap
  over structures.

* **Kernel two-sample statistics** (KID, MMD, FD, PRDC) are *not* means over
  independent units. Resampling with replacement duplicates rows, and a
  duplicated row contributes k(x, x) = 1 as an *off-diagonal* term, which the
  unbiased estimator assumes is an independent pair. That inflates the statistic
  upward. Use m-out-of-n subsampling *without* replacement instead.

The bootstrap unit is always the structure -- never a view, never a voxel. Four
renders of one build are one sample, not four.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

CI = Tuple[float, float]


def _percentiles(samples: np.ndarray, alpha: float) -> CI:
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over independent per-structure values.

    Returns ``(point, lo, hi)``. With fewer than 2 values the interval is
    ``(nan, nan)`` -- an honest "we cannot say" rather than a zero-width lie.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(v))
    if v.size < 2:
        return point, float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    draws = np.array([stat(v[i]) for i in idx], dtype=float)
    lo, hi = _percentiles(draws, alpha)
    return point, lo, hi


def subsample_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    frac: float = 0.8,
    n_rep: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """m-out-of-n subsampling CI for a two-sample statistic ``fn(a, b)``.

    Draws ``m = round(frac * n)`` rows from each side *without replacement*, so no
    row is ever paired with itself off-diagonal (see module docstring). The point
    estimate is computed on the full sets, not on the subsamples.

    Note the interval describes the statistic at size m, not at size n; for the
    unbiased estimators used here (KID, MMD) the statistic is n-stable, so this
    is the intended reading. It is *not* valid for Frechet distance, which is why
    FD carries `comparable_only_at_n` in the scorecard.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    point = float(fn(a, b))
    ma, mb = int(round(frac * len(a))), int(round(frac * len(b)))
    if ma < 2 or mb < 2:
        return point, float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    draws = np.empty(n_rep, dtype=float)
    for r in range(n_rep):
        ia = rng.choice(len(a), size=ma, replace=False)
        ib = rng.choice(len(b), size=mb, replace=False)
        draws[r] = fn(a[ia], b[ib])
    lo, hi = _percentiles(draws, alpha)
    return point, lo, hi


def resample_ci(
    fn: Callable[[list], float],
    items: Sequence,
    n_boot: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    bounds: Tuple[float | None, float | None] = (None, None),
) -> Tuple[float, float, float]:
    """Bootstrap a set-level statistic by resampling the *generated* items.

    For divergences and Wasserstein distances the reference is held fixed and
    only the arm's own sample is resampled, because the question being asked is
    "how much would this arm's score move if we had drawn different samples from
    it", not "how much would it move if the corpus were different".

    Unlike the kernel statistics in `subsample_ci`, with-replacement resampling
    is safe here in the sense that no spurious self-pair is manufactured -- a
    duplicated structure merely re-weights a histogram.

    **The interval is a normal approximation, not a percentile interval**, and
    that is deliberate. Divergences estimated from a finite sample are biased
    upward, and resampling with replacement makes the arm's own sample less
    diverse, which biases them upward again. Measured on real held-out builds:
    point estimate 0.237, percentile interval [0.246, 0.323] -- an interval that
    does not contain its own point estimate. Recentering on the estimate and
    taking the bootstrap standard error reports the sampling variability
    honestly without pretending the estimator is unbiased. The remaining bias is
    a *floor*, and the floor is what the `real_test` control measures; read
    these numbers against that control, never against zero.
    """
    items = list(items)
    point = float(fn(items))
    if len(items) < 2:
        return point, float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    draws = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = rng.integers(0, len(items), size=len(items))
        draws[r] = fn([items[i] for i in idx])
    from scipy.stats import norm
    half = float(norm.ppf(1.0 - alpha / 2.0) * draws.std(ddof=1))
    lo, hi = point - half, point + half
    # Callers pass bounds for statistics with a known range: divergences and
    # Wasserstein distances are non-negative, and a symmetric interval would
    # otherwise report a negative lower bound for a quantity that cannot be
    # negative. Unbounded by default -- clamping a statistic that legitimately
    # goes negative (a mean, a difference) would push the point outside its own
    # interval.
    if bounds[0] is not None:
        lo = max(lo, bounds[0])
    if bounds[1] is not None:
        hi = min(hi, bounds[1])
    return point, lo, hi


def paired_diff_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI on mean(x) - mean(y) for two independent sets.

    Used by the ladder's separation gates, where the question is whether a probe
    is distinguishable from held-out real, not what each scores alone.
    """
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    point = float(xv.mean() - yv.mean())
    if xv.size < 2 or yv.size < 2:
        return point, float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    xi = rng.integers(0, xv.size, size=(n_boot, xv.size))
    yi = rng.integers(0, yv.size, size=(n_boot, yv.size))
    draws = xv[xi].mean(1) - yv[yi].mean(1)
    lo, hi = _percentiles(draws, alpha)
    return point, lo, hi


# --- distribution distances -------------------------------------------------
def _normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > eps else np.full_like(p, 1.0 / max(p.size, 1))


def kl(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p, q = _normalize(p), _normalize(q)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))) / np.log(base))


def jsd(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Jensen-Shannon divergence. In base 2 it is bounded in [0, 1]."""
    p, q = _normalize(p), _normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * kl(p, m, base) + 0.5 * kl(q, m, base))


def w1(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein (earth-mover) distance between two samples."""
    from scipy.stats import wasserstein_distance
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
    if av.size == 0 or bv.size == 0:
        return float("nan")
    return float(wasserstein_distance(av, bv))


def iqr(values: Sequence[float]) -> float:
    """Interquartile range."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def robust_spread(values: Sequence[float]) -> float:
    """Scale for normalizing a distance onto a readable "how many spreads" axis.

    The larger of IQR and standard deviation. Several coherence metrics are
    heavily concentrated on real data -- `lcc_ratio` sits at median 1.0 with an
    IQR of 0.001 -- so IQR alone would divide a difference of 0.04 by 0.001 and
    report "38 spreads away", which is noise dressed as a finding. Taking the max
    keeps a long-tailed metric on its tail's scale. Returns nan when the real
    values are genuinely constant, which forces the reader onto the raw W1 rather
    than a fabricated ratio.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    spread = max(iqr(v), float(v.std(ddof=1)))
    return spread if spread > 1e-9 else float("nan")


def chance_floor(n_distractors: int) -> float:
    """Top-1 accuracy expected by guessing among 1 true + n_distractors items."""
    return 1.0 / float(n_distractors + 1)
