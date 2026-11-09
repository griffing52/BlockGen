"""Conditioned evaluation: does a class-conditioned model build the class it was asked for?

Lives outside ``blockgen.eval.bench`` (frozen) and reuses its pieces unchanged: the
pinned geometry descriptor (``bench.geometry.descriptor``, v2, 80-d), its
reference-fitted ``Standardizer`` and ``geom_kid`` (``distances.kid``). Three
numbers per prompted class ``c``:

* **probe accuracy**: a multinomial logistic regression trained on real builds of
  the training tier (``sft-cat``, the classes only, never ``none``) over the
  geometry descriptor plus a block-palette histogram, applied to the samples
  prompted with ``cat:c``; accuracy is the share it assigns to ``c``. The probe's
  own ceiling is its accuracy on the real eval builds (``eval-cat``), reported
  next to it, and chance is ``1 / n_classes``.
* **geom_kid**: the bench's V-KID between the samples of ``c`` and the real eval
  builds of ``c``, features whitened by a ``Standardizer`` fitted once on the whole
  eval reference (every class), so the per-class numbers share one unit.
* **margin**: the mean KID of the samples of ``c`` to the *other* classes' eval
  builds minus their KID to ``c``'s. Positive when the samples are closer to their
  own class than to the rest, about zero for an unconditioned model.

The probe is fitted on two thirds of each class's training builds (a hash of the
id decides); the remaining third is the ``real`` control, a set of real builds
the probe never saw, scored as if they were samples: it should reach about the
probe ceiling and a near-zero KID. The ``shuffled`` control is the same builds
with their classes permuted, the floor: chance accuracy and a margin about zero.

Per-class n. The eval set holds 32 to 128 real builds per class
(``blockgen.corpus.labels.eval_quota``), so KID is computed at n_ref in that range
against ``n`` samples (128 by default). KID is unbiased in n, but at 32 its
spread is wide; ``--boot`` reports a 90% interval per class over 80% subsamples.

CLI::

    python -m blockgen.eval.conditioned --samples <run>/cat_samples.json [--n 128] [--out DIR]
    python -m blockgen.eval.conditioned --control real      # sanity: real held-out builds
    python -m blockgen.eval.conditioned --control shuffled  # floor: the same with classes permuted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

PROBE_FRACTION = 2 / 3          # of each class's training builds; the rest is the real control
PALETTE_BINS = 256              # block id histogram (legacy ids 0..255), air excluded


# --------------------------------------------------------------------------- data

def tier_structures(tier: str, classes: Optional[Sequence[str]] = None) -> Tuple[List[Structure], List[str], List[str]]:
    """``(structures, classes, ids)`` of a labelled tier's builds, cropped to their boxes."""
    from blockgen.corpus.shards import _load_build, _tier_rows, _tight
    _, df = _tier_rows(tier)
    if classes is not None:
        df = df[df.cat.isin(list(classes))]
    cache: Dict[str, object] = {}
    out, cats, ids = [], [], []
    for r in df.to_dict("records"):
        i, d = _tight(*_load_build(r["path"], cache))
        if i.size == 0:
            continue
        out.append(Structure(block_ids=i.astype(np.int32), block_data=d.astype(np.int32)))
        cats.append(str(r["cat"])); ids.append(str(r["id"]))
    return out, cats, ids


def _in_probe(build_id: str) -> bool:
    h = int.from_bytes(hashlib.blake2b(build_id.encode(), digest_size=8).digest(), "big")
    return (h % 3000) < PROBE_FRACTION * 3000


def load_samples(index: Dict[str, str]) -> Dict[str, List[Structure]]:
    from blockgen.curation.houses import load_structures_from_cache
    return {c: [s.crop_to_non_air() for s in load_structures_from_cache(p)[0]] for c, p in index.items()}


# --------------------------------------------------------------------------- features

def palette(structures: Sequence[Structure]) -> np.ndarray:
    """``[N, 256]`` share of each block id among a build's blocks."""
    out = np.zeros((len(structures), PALETTE_BINS), np.float64)
    for i, s in enumerate(structures):
        b = np.asarray(s.block_ids).ravel()
        b = b[b != 0]
        if b.size:
            out[i] = np.bincount(np.clip(b, 0, PALETTE_BINS - 1), minlength=PALETTE_BINS) / b.size
    return out


def features(structures: Sequence[Structure]) -> Tuple[np.ndarray, np.ndarray]:
    """``(geometry descriptor [N, 80], palette [N, 256])``."""
    from blockgen.eval.bench import geometry as G
    return G.descriptor(structures), palette(structures)


class Probe:
    """Standardised [geometry, sqrt palette] -> class, multinomial logistic regression."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, random_state=seed,
                                                                        class_weight="balanced"))

    @staticmethod
    def _x(geo: np.ndarray, pal: np.ndarray) -> np.ndarray:
        return np.concatenate([geo, np.sqrt(pal)], 1)

    def fit(self, geo, pal, y) -> "Probe":
        self.model.fit(self._x(geo, pal), np.asarray(y))
        self.classes = list(self.model.classes_)
        return self

    def predict(self, geo, pal) -> np.ndarray:
        return self.model.predict(self._x(geo, pal)) if len(geo) else np.array([], dtype=object)


# --------------------------------------------------------------------------- scoring

def _kid(gen: np.ndarray, ref: np.ndarray) -> float:
    from blockgen.eval.bench import geometry as G
    return float(G.geom_kid(gen, ref))


def _boot_kid(gen: np.ndarray, ref: np.ndarray, n_boot: int, rng: np.random.Generator,
              frac: float = 0.8) -> Tuple[float, float]:
    """90% interval of KID over subsamples without replacement (``frac`` of each side): a
    with-replacement bootstrap repeats rows, which the unbiased estimator reads as signal."""
    vals = []
    for _ in range(n_boot):
        g = gen[rng.choice(len(gen), size=max(2, int(frac * len(gen))), replace=False)]
        r = ref[rng.choice(len(ref), size=max(2, int(frac * len(ref))), replace=False)]
        vals.append(_kid(g, r))
    lo, hi = np.nanpercentile(vals, [5, 95])
    return float(lo), float(hi)


def score(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], ref: Dict[str, Tuple[np.ndarray, np.ndarray]],
          probe: Probe, n: Optional[int] = None, n_boot: int = 0, seed: int = 0) -> dict:
    """Probe accuracy, per-class geom_kid and margin of ``samples`` (``{class: (geo, pal)}``) against
    ``ref`` (the eval builds, same layout). ``n`` caps the samples per class."""
    from blockgen.eval.bench import geometry as G
    rng = np.random.default_rng(seed)
    std = G.Standardizer.fit(np.concatenate([g for g, _ in ref.values()]))
    ref_f = {c: std(g) for c, (g, _) in ref.items()}
    classes = [c for c in probe.classes if c in ref]
    per: Dict[str, dict] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    for c, (geo, pal) in samples.items():
        if n is not None and len(geo) > n:
            keep = rng.choice(len(geo), size=n, replace=False)
            geo, pal = geo[keep], pal[keep]
        pred = probe.predict(geo, pal)
        confusion[c] = {k: int((pred == k).sum()) for k in classes if (pred == k).any()}
        row = {"n": int(len(geo)), "n_ref": int(len(ref[c][0])) if c in ref else 0,
               "probe_acc": float((pred == c).mean()) if len(pred) else float("nan")}
        if c in ref and len(geo) >= 2:
            f = std(geo)
            row["geom_kid"] = _kid(f, ref_f[c])
            others = [_kid(f, ref_f[o]) for o in classes if o != c]
            row["kid_to_others"] = float(np.mean(others)) if others else float("nan")
            row["margin"] = row["kid_to_others"] - row["geom_kid"]
            if n_boot:
                row["geom_kid_ci90"] = list(_boot_kid(f, ref_f[c], n_boot, rng))
        per[c] = row
    # a class with no eval builds (one folded into none, still in the vocab) has nothing to be scored against
    acc = [r["probe_acc"] for r in per.values() if r["n_ref"] and np.isfinite(r["probe_acc"])]
    kid = [r["geom_kid"] for r in per.values() if "geom_kid" in r]
    mar = [r["margin"] for r in per.values() if "margin" in r]
    return {"per_class": per, "confusion": confusion, "classes": classes, "chance": 1.0 / max(1, len(classes)),
            "mean_probe_acc": float(np.mean(acc)) if acc else float("nan"),
            "mean_geom_kid": float(np.mean(kid)) if kid else float("nan"),
            "mean_margin": float(np.mean(mar)) if mar else float("nan")}


def build_reference(train_tier: str = "sft-cat", eval_tier: str = "eval-cat", seed: int = 0, verbose: bool = True):
    """The fitted probe, the eval features per class, the real control per class and the probe ceiling."""
    t0 = time.time()
    ev_s, ev_c, _ = tier_structures(eval_tier)
    classes = sorted(set(ev_c))
    tr_s, tr_c, tr_ids = tier_structures(train_tier, classes)
    ev_g, ev_p = features(ev_s)
    tr_g, tr_p = features(tr_s)
    in_probe = np.array([_in_probe(i) for i in tr_ids])
    tr_c = np.asarray(tr_c)
    probe = Probe(seed=seed).fit(tr_g[in_probe], tr_p[in_probe], tr_c[in_probe])
    ev_c = np.asarray(ev_c)
    ref = {c: (ev_g[ev_c == c], ev_p[ev_c == c]) for c in classes}
    ctl = {c: (tr_g[~in_probe & (tr_c == c)], tr_p[~in_probe & (tr_c == c)]) for c in classes}
    ceiling = float((probe.predict(ev_g, ev_p) == ev_c).mean())
    ceiling_by = {c: float((probe.predict(*ref[c]) == c).mean()) for c in classes}
    if verbose:
        print(f"[conditioned] probe on {int(in_probe.sum())} {train_tier} builds, {len(classes)} classes; "
              f"eval {len(ev_s)} {eval_tier} builds; probe accuracy on eval {ceiling:.3f} "
              f"(chance {1 / len(classes):.3f}); {time.time() - t0:.0f}s", flush=True)
    return probe, ref, ctl, {"probe_acc_eval": ceiling, "probe_acc_eval_by_class": ceiling_by,
                             "n_probe": int(in_probe.sum()), "train_tier": train_tier, "eval_tier": eval_tier}


def shuffled(ctl: Dict[str, Tuple[np.ndarray, np.ndarray]], seed: int = 0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The control builds with their classes permuted (sizes per class kept)."""
    rng = np.random.default_rng(seed)
    classes = list(ctl)
    g = np.concatenate([ctl[c][0] for c in classes]); p = np.concatenate([ctl[c][1] for c in classes])
    perm = rng.permutation(len(g))
    out, i = {}, 0
    for c in classes:
        k = len(ctl[c][0])
        out[c] = (g[perm[i:i + k]], p[perm[i:i + k]]); i += k
    return out


def table(res: dict, ceiling: Optional[dict] = None) -> str:
    rows = ["| class | n | n_ref | probe acc | probe ceiling | geom_kid | KID to others | margin |",
            "|---|---|---|---|---|---|---|---|"]
    for c, r in sorted(res["per_class"].items()):
        ceil = (ceiling or {}).get("probe_acc_eval_by_class", {}).get(c, float("nan"))
        ci = r.get("geom_kid_ci90")
        kid = f"{r.get('geom_kid', float('nan')):.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        rows.append(f"| {c} | {r['n']} | {r['n_ref']} | {r['probe_acc']:.3f} | {ceil:.3f} | {kid} | "
                    f"{r.get('kid_to_others', float('nan')):.3f} | {r.get('margin', float('nan')):.3f} |")
    rows.append(f"| mean | | | {res['mean_probe_acc']:.3f} | {(ceiling or {}).get('probe_acc_eval', float('nan')):.3f} "
                f"| {res['mean_geom_kid']:.3f} | | {res['mean_margin']:.3f} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=None, help="cat_samples.json ({class: npz}) from python -m blockgen.train "
                                                     "--sample <run> --cat all")
    ap.add_argument("--control", choices=("real", "shuffled"), default=None)
    ap.add_argument("--train-tier", default="sft-cat"); ap.add_argument("--eval-tier", default="eval-cat")
    ap.add_argument("--n", type=int, default=128, help="samples per class scored (a class with more is subsampled)")
    ap.add_argument("--boot", type=int, default=0, help="subsamples (80%% of each side) for a 90%% KID interval per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for conditioned.json; default next to --samples")
    a = ap.parse_args(argv)
    if not a.samples and not a.control:
        ap.error("give --samples or --control")
    probe, ref, ctl, ceil = build_reference(a.train_tier, a.eval_tier, a.seed)
    if a.control == "real":
        smp, what = ctl, "control:real"
    elif a.control == "shuffled":
        smp, what = shuffled(ctl, a.seed), "control:shuffled"
    else:
        index = json.load(open(a.samples))
        smp = {c: features(s) for c, s in load_samples(index).items()}
        what = a.samples
    res = score(smp, ref, probe, n=a.n, n_boot=a.boot, seed=a.seed)
    res.update({"samples": what, **ceil, "n_cap": a.n, "descriptor": "bench.geometry v2 + palette (probe)"})
    print(table(res, ceil))
    out = a.out or (os.path.dirname(os.path.abspath(a.samples)) if a.samples else None)
    if out:
        os.makedirs(out, exist_ok=True)
        name = "conditioned.json" if not a.control else f"conditioned_{a.control}.json"
        json.dump(res, open(os.path.join(out, name), "w"), indent=1)
        print(f"[conditioned] -> {os.path.join(out, name)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
