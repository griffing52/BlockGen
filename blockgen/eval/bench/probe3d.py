"""The 3D-native metric rows, shared by the Objaverse mesh probe and the
Minecraft ladder (notes §65).

Every row is a standard point-cloud metric computed under one of three
samplers (`mesh_probe.SAMPLERS`), so the table can say whether a metric's
sensitivity follows what its sampler observes:

    all   area-weighted samples over every face, interior faces included
    vis   area-weighted samples over the faces visible from some exterior
          viewpoint (ray cast; the set Strip-Interior keeps)
    cut   area-weighted samples over the faces visible in the four cutaway
          views (ray cast from the plane through the box centre)

Rows per sampler s:
    {uni3d,ulip2}_{s}_kid / _fd   KID / FD on L2-normalised CLIP-aligned point
                                  embeddings (`pc_encoders`, 10k xyz+rgb points)
    cd_{s}_{mmd,one_minus_cov,one_nna}    chamfer (PointFlow convention: mean
                                  squared NN distance both ways), 2,048 points
    emd_{s}_{mmd,one_minus_cov,one_nna}   exact EMD (mean matched Euclidean
                                  distance), 512 points

Set statistics are `pointcloud.mmd / one_minus_cov / one_nna` (Achlioptas
2018; Yang 2019) and the separations are `ladder.noise_floor` z-scores, the
same as every other row of the suite. Distances are computed only for the
pairs those statistics read (`pc_encoders.needed_pairs`).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import ladder as L
from blockgen.eval.bench import pc_encoders as pe
from blockgen.eval.bench import pointcloud as pc

K_ENC = 10_000
K_CD = 2048
K_EMD = 512
#: prompt embeddings from scripts/pc_text_embed.py; row 0 is the probe prompt
TEXT_EMB = pe.WEIGHTS / "pc_text_emb.npz"


_HELD = [0]


@contextlib.contextmanager
def gpu_lock(path: Optional[str] = None):
    """Hold an exclusive flock on the shared GPU lock file (`$GPU_LOCK`) for
    the GPU phases only, so CPU phases do not block other users of the card.
    Re-entrant within a process: a nested call inside a held lock is a no-op
    (a second flock on a new descriptor would deadlock against the first)."""
    path = path or os.environ.get("GPU_LOCK")
    if not path or _HELD[0]:
        _HELD[0] += 1
        try:
            yield
        finally:
            _HELD[0] -= 1
        return
    with open(path, "a") as fh:
        t0 = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 5:
            print(f"  [gpu-lock] waited {waited:.0f}s", flush=True)
        _HELD[0] += 1
        try:
            yield
        finally:
            _HELD[0] -= 1
            fcntl.flock(fh, fcntl.LOCK_UN)


class FastFD:
    """`distances.fd(p, r)` with the reference's statistics cached.

    tr sqrtm(S_p S_r) = sum sqrt(eig(B S_p B)) with B = S_r^{1/2} (both
    symmetric PSD after the same shrinkage), so one eigendecomposition of the
    fixed reference replaces a general `sqrtm` per call: identical value,
    about 100x faster at d = 1,024, which is what makes noise floors and
    paired CIs affordable for the 1,024/1,280-d point embeddings."""

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        self._ref = None

    def _cov(self, a):
        d = a.shape[1]
        c = np.cov(a, rowvar=False)
        return (1.0 - self.shrinkage) * c + self.shrinkage * (np.trace(c) / d) * np.eye(d)

    def __call__(self, r: np.ndarray, p: np.ndarray) -> float:
        if self._ref is None or self._ref[0] is not r:
            s2 = self._cov(r)
            w, V = np.linalg.eigh(s2)
            B = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
            self._ref = (r, r.mean(0), s2, B)
        _, mu2, s2, B = self._ref
        s1 = self._cov(p)
        ev = np.linalg.eigvalsh(B @ s1 @ B)
        diff = p.mean(0) - mu2
        return float(diff @ diff + np.trace(s1) + np.trace(s2)
                     - 2.0 * np.sqrt(np.clip(ev, 0, None)).sum())


# --- distances on the needed pairs ----------------------------------------------
def chamfer_pairs(clouds: np.ndarray, pairs: np.ndarray, device: str = "cuda",
                  batch: int = 24) -> np.ndarray:
    import torch
    x = torch.as_tensor(clouds, dtype=torch.float32, device=device)
    out = np.empty(len(pairs), np.float32)
    for s in range(0, len(pairs), batch):
        p = torch.as_tensor(pairs[s:s + batch], device=device)
        d = torch.cdist(x[p[:, 0]], x[p[:, 1]]).square_()
        out[s:s + batch] = (d.amin(2).mean(1) + d.amin(1).mean(1)).cpu().numpy()
    return out


def _matrix(N: int, pairs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    m = np.full((N, N), np.nan, np.float32)
    np.fill_diagonal(m, 0.0)
    m[pairs[:, 0], pairs[:, 1]] = vals
    m[pairs[:, 1], pairs[:, 0]] = vals
    return m


# --- the rows -----------------------------------------------------------------------
def emd_dms(clouds: Dict[str, np.ndarray], ref_ix: np.ndarray, sets: Dict[str, np.ndarray],
            n: int, workers: int = 16, log=print, cache=None) -> Dict[str, np.ndarray]:
    """CPU phase: `{emd_<s>: (N, N)}` on the needed pairs. Run before waiting
    for the GPU lock. `cache` (an .npz path) is loaded if it exists, else written,
    so a crash in a later phase does not cost the EMD again (same seed only:
    the clouds are seeded, so a cached matrix belongs to one seed)."""
    from pathlib import Path
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        log(f"  [3d] emd from {cache}")
        return {k: z[k] for k in z.files}
    N = len(next(iter(clouds.values())))
    need = pe.needed_pairs(N, ref_ix, sets, n)
    dms = {}
    for s, cl in clouds.items():
        t0 = time.time()
        dms[f"emd_{s}"] = pe.emd_matrix(np.ascontiguousarray(cl[:, :K_EMD, :3]), need,
                                        workers=workers)
        log(f"  [3d] emd_{s} {time.time() - t0:.0f}s ({int(np.triu(need | need.T, 1).sum())}"
            f" pairs)")
    if cache is not None:
        np.savez_compressed(cache, **dms)
    return dms


def rows_3d(clouds: Dict[str, np.ndarray], ref_ix: np.ndarray, sets: Dict[str, np.ndarray],
            n: int, device: str = "cuda", encoders: Sequence[str] = ("uni3d", "ulip2"),
            emd: Optional[Dict[str, np.ndarray]] = None, emd_workers: int = 16,
            lock: Optional[str] = None, log=print) -> Dict[str, tuple]:
    """`{row: (fn(ref, probe), X)}` where X is `(N, d)` features over every item
    (an embedding, or an index column into a distance matrix), plus
    `"_dms": (None, {name: matrix})`.

    `clouds[s]` is `(N, >=K_ENC, 6)` xyz+rgb per sampler; `sets` holds every
    index set the statistics will score against `ref_ix` (rungs, real held-out
    and the null pool). `emd` is `emd_dms(...)` if already computed."""
    N = len(next(iter(clouds.values())))
    rows: Dict[str, tuple] = {}
    need = pe.needed_pairs(N, ref_ix, sets, n)
    pairs = np.argwhere(np.triu(need | need.T, 1))
    log(f"  [3d] {N} items, {len(pairs)} distance pairs per sampler")
    col = np.arange(N).reshape(-1, 1)
    dms = dict(emd) if emd is not None else emd_dms(clouds, ref_ix, sets, n, emd_workers, log)
    text = np.load(TEXT_EMB) if TEXT_EMB.exists() else None
    with gpu_lock(lock):
        for name in encoders:
            t0 = time.time()
            enc = pe.ENCODERS[name](device)
            for s, cl in clouds.items():
                Z = pe.embed(enc, [c[:K_ENC] for c in cl])
                rows[f"{name}_{s}_kid"] = (lambda r, p: D.kid(p, r), Z)
                rows[f"{name}_{s}_fd"] = (FastFD(), Z)
                if text is not None and name in text.files:
                    # text-shape score (Uni3D-score / ULIP-2 analogue of CLIP score),
                    # negated so larger means further from real
                    cs = (100.0 * Z @ text[name][0].astype(np.float64))[:, None]
                    rows[f"{name}_{s}_textscore_drop"] = (lambda r, p: -float(p.mean()), cs)
            del enc
            import torch
            torch.cuda.empty_cache()
            log(f"  [3d] {name} embeddings {time.time() - t0:.0f}s")
        for s, cl in clouds.items():
            t0 = time.time()
            vals = chamfer_pairs(np.ascontiguousarray(cl[:, :K_CD, :3]), pairs, device)
            dms[f"cd_{s}"] = _matrix(N, pairs, vals)
            log(f"  [3d] cd_{s} {time.time() - t0:.0f}s")
    order = [f"{m}_{s}" for m in ("cd", "emd") for s in clouds]
    for mname, fn in pc.metric_table({k: dms[k] for k in order}).items():
        rows[mname] = (fn, col)
    rows["_dms"] = (None, dms)
    return rows


# --- scoring --------------------------------------------------------------------------
def score(table: Dict[str, tuple], rungs: Sequence[str], n: int, reps: int, ci_reps: int,
          rng: np.random.Generator, log=print) -> Dict[str, dict]:
    """`table[name] = (fn, R, {rung: P, "real_heldout": P0}, null)`.

    Per row: raw scores, the noise floor (`ladder.noise_floor`: n real objects
    drawn from the null pool against the same reference), the separation
    z = (probe - real held-out) / sd, its 95% interval from `ci_reps` paired 80%
    subsamples of the probe objects, and the raw difference beside it."""
    results = {}
    m_sub = int(round(0.8 * n))
    for mname, (fn, R, P, Nl) in table.items():
        scores = {k: float(fn(R, f)) for k, f in P.items()}
        mean, sd, sem = L.noise_floor(fn, R, Nl, min(n, len(Nl) - 1), reps, rng)
        base = scores["real_heldout"]
        sep, ci, raw = {}, {}, {}
        for v in rungs:
            raw[v] = scores[v] - base
            sep[v] = raw[v] / sd if sd > 0 else float("nan")
            draws = []
            for _ in range(ci_reps):
                idx = rng.choice(n, m_sub, replace=False)
                draws.append((fn(R, P[v][idx]) - fn(R, P["real_heldout"][idx])) / sd)
            ci[v] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        results[mname] = {"scores": scores, "raw_delta": raw, "noise_mean": mean,
                          "noise_sd": sd, "noise_sem": sem, "sep": sep, "sep_ci95": ci,
                          "real_vs_null_z": (base - mean) / sd if sd > 0 else float("nan")}
        log(f"  {mname:28s} sd={sd:.4g}  " + "  ".join(f"{v}={sep[v]:+.1f}" for v in rungs))
    return results


def table_md(results: Dict[str, dict], rungs: Sequence[str], raw_rows: Sequence[str] = ()
             ) -> list:
    """Markdown rows: z [95% CI] per rung, then the noise sd; rows whose name
    matches any of `raw_rows` (e.g. the KID rows, raw MMD^2) get the raw
    difference beside the z-score."""
    lines = ["| metric | " + " | ".join(rungs) + " | sd |", "|" + "---|" * (len(rungs) + 2)]
    for m, r in results.items():
        show_raw = any(t in m for t in raw_rows)
        cells = []
        for v in rungs:
            c = f"{r['sep'][v]:+.1f} [{r['sep_ci95'][v][0]:+.1f}, {r['sep_ci95'][v][1]:+.1f}]"
            if show_raw:
                c += f" ({r['raw_delta'][v]:+.3g})"
            cells.append(c)
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {r['noise_sd']:.3g} |")
    return lines


def agreement(a: Dict[str, dict], b: Dict[str, dict], rungs: Sequence[str]) -> dict:
    """Two seeds: Pearson r and Spearman rho of the z-scores over every
    (row, rung) cell both have, the median |dz|, and the share of cells whose
    blindness verdict (|z| < 2) agrees."""
    from scipy.stats import pearsonr, spearmanr
    xs, ys = [], []
    for m in a:
        if m not in b:
            continue
        for v in rungs:
            x, y = a[m]["sep"].get(v), b[m]["sep"].get(v)
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    xs, ys = np.asarray(xs), np.asarray(ys)
    return {"cells": int(len(xs)), "pearson": float(pearsonr(xs, ys)[0]),
            "spearman": float(spearmanr(xs, ys)[0]),
            "median_abs_dz": float(np.median(np.abs(xs - ys))),
            "verdict_agree": float(np.mean((np.abs(xs) < 2) == (np.abs(ys) < 2)))}
