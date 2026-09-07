"""Distribution distances in feature space, plus the memorization detector.

**Every kernel estimator here is unbiased.** That is not a stylistic preference.
`blockgen/eval/perceptual.py:cmmd` computes ``k(x,x).mean() + k(y,y).mean() -
2*k(x,y).mean()`` with the diagonals included, despite a docstring claiming the
unbiased form. Measured on real DINOv2 features, two disjoint halves of the same
real corpus score:

    n        16      32      64     128     256     400
    cmmd  0.289   0.161   0.071   0.037   0.020   0.014

which decays as ~1/n. Every CMMD number recorded in results.md is therefore
comparable only at fixed n, and the reported "floor on held-out real natives" is
largely estimator bias rather than a property of the data. `perceptual.py` is
left untouched so those numbers stay reproducible; `mmd_rbf` here is the
corrected form, and the ladder scores the legacy one alongside it as evidence.

The same measurement for the unbiased KID in this module gives 0.000 at every n
from 16 to 400 -- which is what makes arms with different sample counts
comparable at all.

Note `cmmd(x, x)` on an *identical* array returns exactly 0.0 because the terms
cancel; the bias is only visible across two disjoint samples from one
distribution. A test asserting `cmmd(x, x) > 0` would wrongly pass.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] features, got shape {x.shape}")
    return x


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


# --- kernel two-sample statistics ------------------------------------------
def kid(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float | None = None,
        coef0: float = 1.0, scale: float = 1000.0) -> float:
    """Unbiased polynomial-kernel MMD^2 (Kernel Inception Distance form).

    k(a, b) = (gamma * a.b + coef0)^degree with gamma = 1/D by default. Diagonal
    terms are excluded from both within-set sums, which is what makes the
    estimator unbiased and therefore stable in n. `scale` only makes the numbers
    readable; it is recorded in the scorecard.

    Validated against known damage on real builds (n=256, scale=1000): held-out
    real 0.001, canon-16 0.121, 10% block noise 0.123, canon-8 0.491, material
    shuffle 0.755.
    """
    x, y = _as2d(x), _as2d(y)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    g = 1.0 / x.shape[1] if gamma is None else gamma

    def k(a, b):
        return (g * (a @ b.T) + coef0) ** degree

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(scale * (term_x + term_y - 2.0 * kxy.mean()))


def kid_arm_term(x: np.ndarray, ref: np.ndarray, degree: int = 3,
                 gamma: float | None = None, coef0: float = 1.0,
                 scale: float = 1000.0) -> float:
    """The part of `kid(x, ref)` that depends on `x`. Use for paired deltas.

    `kid = term_x + term_ref - 2*mean(k_x_ref)`, and `term_ref` depends on the
    reference alone. When two arms are compared against the *same* reference
    draw it cancels exactly, so computing it is pure waste -- and it is the
    expensive one: the reference is typically several times larger than an arm,
    so its self-kernel dominates the cost. Dropping it takes a 66-pair
    head-to-head table from tens of minutes to about one, and removes a term
    that contributes nothing to the difference but floating-point noise.

    Differences of this quantity equal differences of `kid` exactly. The
    absolute value is not a KID and must never be reported as one.
    """
    x, ref = _as2d(x), _as2d(ref)
    m = len(x)
    if m < 2 or len(ref) < 1:
        return float("nan")
    g = 1.0 / x.shape[1] if gamma is None else gamma
    kxx = (g * (x @ x.T) + coef0) ** degree
    kxr = (g * (x @ ref.T) + coef0) ** degree
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    return float(scale * (term_x - 2.0 * kxr.mean()))


def median_bandwidth(x: np.ndarray, max_n: int = 512,
                     rng: np.random.Generator | None = None) -> float:
    """Median pairwise distance -- the standard RBF bandwidth heuristic.

    Computed on the *reference* set once and then frozen. Letting each arm pick
    its own bandwidth would make the arms incomparable, and the existing
    `sigma=10` default is far outside the informative range: on unit-norm
    features squared distances lie in [0, 4], so `sigma=10` puts every kernel
    value in [0.98, 1.0] and the statistic degenerates to a scaled mean
    difference that cannot see higher moments.
    """
    x = _as2d(x)
    if len(x) > max_n:
        rng = rng or np.random.default_rng(0)
        x = x[rng.choice(len(x), max_n, replace=False)]
    d = np.sqrt(_sqdist(x, x))
    iu = np.triu_indices(len(x), k=1)
    med = float(np.median(d[iu])) if iu[0].size else 1.0
    return max(med, 1e-6)


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None,
            scale: float = 1000.0) -> float:
    """Unbiased Gaussian-RBF MMD^2. A second opinion under a different kernel.

    `sigma=None` uses the median heuristic on the concatenation; callers should
    instead compute it once on the reference and pass it explicitly.
    """
    x, y = _as2d(x), _as2d(y)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    if sigma is None:
        sigma = median_bandwidth(np.concatenate([x, y], 0))
    g = 1.0 / (2.0 * sigma ** 2)
    kxx = np.exp(-g * _sqdist(x, x))
    kyy = np.exp(-g * _sqdist(y, y))
    kxy = np.exp(-g * _sqdist(x, y))
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(scale * (term_x + term_y - 2.0 * kxy.mean()))


def fd(x: np.ndarray, y: np.ndarray, shrinkage: float = 0.1) -> float:
    """Frechet distance with shrunk covariances.

    Kept for comparability with the literature, and gated hard. A 768x768
    covariance cannot be estimated from 64 samples, so the estimate is dominated
    by n and this metric is *not* comparable across arms of different size --
    callers must attach `comparable_only_at_n`. Shrinkage toward a scaled
    identity keeps it finite; the coefficient is recorded.
    """
    from scipy import linalg

    x, y = _as2d(x), _as2d(y)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mu1, mu2 = x.mean(0), y.mean(0)
    d = x.shape[1]

    def cov(a):
        c = np.cov(a, rowvar=False)
        return (1.0 - shrinkage) * c + shrinkage * (np.trace(c) / d) * np.eye(d)

    s1, s2 = cov(x), cov(y)
    covmean = linalg.sqrtm(s1 @ s2)      # scipy >= 1.17 dropped `disp`
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


# --- manifold coverage ------------------------------------------------------
def _knn_radius(a: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row to its k-th nearest *other* row."""
    d = np.sqrt(_sqdist(a, a))
    np.fill_diagonal(d, np.inf)
    kk = min(k, max(len(a) - 1, 1))
    return np.partition(d, kk - 1, axis=1)[:, kk - 1]


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Precision, recall, density and coverage (Kynkaanniemi; Naeem et al. 2020).

    Density and coverage are the pair to lead with. Precision/recall estimate
    manifold membership from k-NN radii, which is unstable at the sample sizes
    this project works at and hypersensitive to a single outlying feature;
    density and coverage were introduced specifically to fix that. All four are
    computed because precision/recall remain the literature's common currency.

    Coverage is the mode-collapse detector: it asks what fraction of *real*
    builds have any generated neighbour, so duplicating one good sample 64 times
    cannot inflate it.
    """
    real, fake = _as2d(real), _as2d(fake)
    if len(real) < 2 or len(fake) < 2:
        return {m: float("nan") for m in ("precision", "recall", "density", "coverage")}

    r_real = _knn_radius(real, k)
    r_fake = _knn_radius(fake, k)
    d_rf = np.sqrt(_sqdist(real, fake))                 # [R, F]

    within_real = d_rf <= r_real[:, None]               # fake inside a real ball
    within_fake = d_rf <= r_fake[None, :]               # real inside a fake ball

    return {
        "precision": float(within_real.any(axis=0).mean()),
        "recall": float(within_fake.any(axis=1).mean()),
        "density": float(within_real.sum(axis=0).mean() / max(k, 1)),
        "coverage": float((d_rf.min(axis=1) <= r_real).mean()),
    }


# --- memorization -----------------------------------------------------------
def nn_distance(gen: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine distance from each generated row to its nearest reference row.

    Features are unit-norm, so cosine distance is `1 - a.b` and monotone in
    Euclidean distance. Returns `(distances, indices)`.
    """
    gen, ref = _as2d(gen), _as2d(ref)
    if len(gen) == 0 or len(ref) == 0:
        return np.zeros(len(gen)), np.zeros(len(gen), dtype=int)
    sim = gen @ ref.T
    idx = np.argmax(sim, axis=1)
    return 1.0 - sim[np.arange(len(gen)), idx], idx


def nn_percentile(gen_dists: np.ndarray, calib_dists: np.ndarray) -> np.ndarray:
    """Where each sample's nearest-train distance falls among *real* builds'.

    A raw cosine distance in DINOv2 space means nothing on its own. Calibrating
    against the distribution of held-out-real-to-train distances does: a value
    near 0 says "closer to the training set than a genuine unseen house ever
    is", which is memorization, and it says so independently of voxel resolution
    -- the failure mode that makes `nn_iou` unusable across 16^3 and 32^3 arms.
    """
    gen_dists = np.asarray(gen_dists, dtype=float)
    calib = np.sort(np.asarray(calib_dists, dtype=float))
    if calib.size == 0:
        return np.full_like(gen_dists, np.nan)
    return np.searchsorted(calib, gen_dists, side="left") / float(calib.size)


def memorization_report(gen_feats: np.ndarray, train_feats: np.ndarray,
                        calib_feats: np.ndarray, dup_percentile: float = 1.0
                        ) -> Dict[str, np.ndarray | float]:
    """DINO-space memorization panel.

    `calib_feats` must be *held-out real* builds; their own nearest-train
    distances define what "normally close to training data" looks like.
    """
    calib_d, _ = nn_distance(calib_feats, train_feats)
    gen_d, gen_idx = nn_distance(gen_feats, train_feats)
    pct = nn_percentile(gen_d, calib_d)
    threshold = float(np.percentile(calib_d, dup_percentile)) if calib_d.size else 0.0
    return {
        "nn_dist": gen_d,
        "nn_index": gen_idx,
        "nn_percentile": pct,
        "dup_rate": float(np.mean(gen_d <= threshold)) if gen_d.size else float("nan"),
        "dup_threshold": threshold,
        "calib_nn_dist": calib_d,
    }
