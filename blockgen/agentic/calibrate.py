"""Scorer calibration on oracle dreams: does a dream scorer recognise the right build?

Before any paid dream run, a dream-vs-render scorer has to pass one sanity check:
given a "dream" that IS a picture of a known build, it must rank that build's own
render above the renders of other builds. The oracle dream is what ``mock:render``
produces (a render of a houses_32 TRAIN-split build at ``iso``); the candidate
renders are taken from *different* views (``iso_left``/``iso_right`` by default,
which the scorer searches like a ``view_search``), so an identical-pixels shortcut
cannot pass. Distractors are other train builds rendered the same way.

Reported per scorer:

``top1``      fraction of dreams whose own build beats every distractor (chance 1/(k+1))
``margin``    mean (own score - best distractor score)
``mrr``       mean reciprocal rank of the own build
``own_mean``  mean score of the own build
``baseline``  mean score of a distractor pair: the "unrelated house" floor. For
              ``embed?baseline=0`` this is the raw random-pair cosine that
              :data:`BASELINES` ships, which ``embed`` subtracts.

Oracle dreams are renders, so this measures the *ceiling* of a scorer (no
image-model domain gap). ``dreams_dir`` swaps in real generated images later:
a directory of ``<corpus index>.png`` files, each an image of that train build.

Usage::

    python -m blockgen.agentic.calibrate --scorer 'embed?backbone=siglipB&baseline=0' --n 64
    python -m blockgen.agentic.calibrate --scorer silhouette --renderer flat
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blockgen.agentic import core
from blockgen.agentic.core import Attempt, DreamSpec, DreamVersion, Spec
from blockgen.utils.data import Structure

#: Mean cosine of an oracle dream (textured iso render of one houses_32 train build)
#: against the iso_left/iso_right renders of a DIFFERENT train build (max over the
#: two views, as embed's default 'paired' reduce does), 224^2 foreground crops.
#: Measured 2026-10-03 with ``run('embed?backbone=<b>&baseline=0&device=cpu', n=64,
#: distractors=31, renderer='textured')``:
#:
#:   siglipB  baseline 0.817 (sd 0.071), own build 0.970, top1 1.00 (chance 0.03)
#:   clipB16  baseline 0.869 (sd 0.059), own build 0.979, top1 1.00
#:
#: i.e. raw cosines live in a 0.1-wide band, which is why ``embed`` maps
#: cos -> (cos - baseline) / (1 - baseline). Re-measure when the renderer, the crop
#: or the backbone changes. Oracle dreams are renders, so real dreams will sit
#: lower (domain gap); ``dreams_dir`` re-measures on real images.
BASELINES: Dict[str, float] = {
    "siglipB": 0.817,
    "clipB16": 0.869,
}

ORACLE_KEY = "oracle"


# ===========================================================================
# A minimal episode for scorers
# ===========================================================================
@dataclass
class _FrozenWorld:
    """World stand-in over a fixed Structure (scorers only read it)."""

    structure: Structure

    @property
    def size(self) -> Tuple[int, int, int]:
        return tuple(self.structure.shape)  # type: ignore[return-value]

    def to_structure(self, crop: bool = True) -> Structure:
        return self.structure.crop_to_non_air() if crop else self.structure

    def block_count(self) -> int:
        return int(self.structure.occupied_mask.sum())

    def clone(self) -> "_FrozenWorld":
        return self


@dataclass
class CalibEpisode:
    """Just enough of :class:`core.EpisodeLike` for image scorers: one dream, a
    renderer, and no LLM (``ask`` raises, so a ``vlm`` scorer cannot be
    calibrated by accident and silently cost money)."""

    dream: DreamVersion
    spec_: DreamSpec
    renderer: Any
    prompt: str = ""
    seed: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kit = SimpleNamespace(dreams=[self.spec_], renderer=self.renderer)
        self.dreams = {self.spec_.key: [self.dream]}

    def latest_dream(self, key: str) -> Optional[DreamVersion]:
        vs = self.dreams.get(key) or []
        return vs[-1] if vs else None

    def dream_id(self) -> str:
        return self.dream.digest

    def render(self, target: Any, views: Union[str, Sequence[str]] = "score") -> Dict[str, bytes]:
        from blockgen.agentic.render import render_pngs
        s = target.world.to_structure() if isinstance(target, Attempt) else target.to_structure()
        if views == "score":        # EpisodeLike's alias: the dream's matched views
            views = [self.spec_.view or "iso", *self.spec_.view_search]
        return render_pngs(self.renderer, s, [views] if isinstance(views, str) else list(views))

    def ask(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("calibration episodes have no LLM: calibrate image scorers only")


def attempt_for(s: Structure, renders: Optional[Dict[str, bytes]] = None, idx: int = 0) -> Attempt:
    """A scorer-ready Attempt over a fixed structure (no program, no LLM)."""
    from blockgen.agentic.dsl import ExecutionReport
    from blockgen.agentic.providers import LLMResponse
    n = int(s.occupied_mask.sum())
    return Attempt("calib", "", ExecutionReport(blocks=n), LLMResponse("", "calib"), n,
                   idx=idx, world=_FrozenWorld(s), renders=dict(renders or {}))


# ===========================================================================
# Builds
# ===========================================================================
def train_builds(n: int, *, corpus: str = "houses_32", split: str = "train",
                 seed: int = 0) -> Tuple[List[Structure], List[int]]:
    """``n`` builds of the canonical split (TRAIN by default: never the test split,
    which is where the paper's prompts come from)."""
    from blockgen.agentic.render import REPO_ROOT
    from blockgen.eval.bench.splits import SPLIT_ROOT, load_corpus, load_split
    sp = load_split(corpus, root=REPO_ROOT / SPLIT_ROOT)
    structs, _ = load_corpus(corpus)
    pool = np.asarray(sp.indices(split))
    pick = np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False)
    return [structs[int(i)].crop_to_non_air() for i in pick], [int(i) for i in pick]


#: (block id, data) materials for synthetic builds: visually distinct on purpose.
_MATS: Tuple[Tuple[int, int], ...] = (
    (1, 0), (5, 0), (45, 0), (24, 0), (35, 14), (35, 11), (155, 0), (4, 0),
    (5, 2), (35, 4), (98, 0), (35, 5), (17, 0), (35, 1), (87, 0), (35, 10),
)


def synthetic_builds(n: int = 8, seed: int = 0, size: int = 24) -> List[Structure]:
    """Offline stand-ins for corpus builds: hollow boxes of varied footprint and
    height with a different wall/roof material each, some with a tower or an
    L-wing. Distinct enough that a working scorer separates them; cheap enough
    for CI."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ids = np.zeros((size, size, size), np.int32)
        data = np.zeros_like(ids)
        w, d = int(rng.integers(6, size - 4)), int(rng.integers(6, size - 4))
        h = int(rng.integers(3, size // 2))
        wall, roof = _MATS[(2 * i) % len(_MATS)], _MATS[(2 * i + 1) % len(_MATS)]

        def put(sl, mat):
            ids[sl], data[sl] = mat

        put((slice(0, w), slice(0, h), slice(0, d)), wall)
        put((slice(1, w - 1), slice(1, h), slice(1, d - 1)), (0, 0))     # hollow
        for k in range(min(w, d) // 2):                                  # stepped roof
            if h + k >= size:
                break
            put((slice(k, w - k), slice(h + k, h + k + 1), slice(k, d - k)), roof)
        if i % 3 == 1:                                                   # tower
            tw = max(2, w // 4)
            put((slice(0, tw), slice(0, min(size, h + 6)), slice(0, tw)), roof)
        if i % 3 == 2 and d + 4 < size:                                  # L-wing
            put((slice(0, max(3, w // 2)), slice(0, max(2, h - 1)), slice(d, min(size, d + 4))), wall)
        out.append(Structure(ids, data).crop_to_non_air())
    return out


def _load_dir_dreams(dreams_dir: Union[str, Path]) -> Dict[int, bytes]:
    out: Dict[int, bytes] = {}
    for p in sorted(Path(dreams_dir).glob("*.png")):
        try:
            out[int(p.stem.split("_")[-1])] = p.read_bytes()
        except ValueError:
            continue
    if not out:
        raise FileNotFoundError(f"{dreams_dir}: no '<corpus index>.png' files")
    return out


# ===========================================================================
# run
# ===========================================================================
def run(scorer_spec: Spec, n: int = 32, distractors: int = 31,
        dreams_dir: Optional[Union[str, Path]] = None, *,
        renderer: Any = "textured", structures: Optional[Sequence[Structure]] = None,
        corpus: str = "houses_32", split: str = "train", seed: int = 0,
        dream_view: str = "iso", match_views: Sequence[str] = ("iso_left", "iso_right"),
        verbose: bool = False) -> Dict[str, Any]:
    """Rank-the-own-build test for a dream scorer (see module docstring).

    ``structures`` replaces the corpus (tests use :func:`synthetic_builds`).
    ``renderer`` is a registry spec or a Renderer instance. With ``dreams_dir``,
    the dreams are those files and the builds are the corpus indices they name.
    """
    from blockgen.agentic import scoring  # noqa: F401  (registers the scorers)
    from blockgen.agentic.render import render_pngs

    t0 = time.time()
    name = core.parse_spec(scorer_spec)[0]
    if not core.meta("scorer", name).get("needs_dream"):
        raise ValueError(f"calibrate: scorer '{name}' does not compare against a dream")
    if core.meta("scorer", name).get("llm"):
        raise ValueError(f"calibrate: scorer '{name}' calls an LLM; calibrate image scorers only")
    fn = core.make("scorer", scorer_spec)
    rend = core.make("renderer", renderer) if isinstance(renderer, (str, dict)) else renderer
    for v in (dream_view, *match_views):
        core.get_view(v)
    if dream_view in match_views:
        raise ValueError("calibrate: the dream view must not be a match view "
                         "(identical pixels would pass any scorer)")

    dir_dreams: Dict[int, bytes] = {}
    if dreams_dir is not None:
        dir_dreams = _load_dir_dreams(dreams_dir)
        from blockgen.eval.bench.splits import load_corpus
        structs_all, _ = load_corpus(corpus)
        ids = sorted(dir_dreams)[:n]
        builds = [structs_all[i].crop_to_non_air() for i in ids]
        source = f"dir:{dreams_dir}"
    elif structures is not None:
        builds, ids = list(structures)[:n], list(range(min(n, len(structures))))
        source = "structures"
    else:
        builds, ids = train_builds(n, corpus=corpus, split=split, seed=seed)
        source = f"oracle:{corpus}/{split}"
    if len(builds) < 2:
        raise ValueError("calibrate: need at least 2 builds")
    k = min(int(distractors), len(builds) - 1)

    ds = DreamSpec(ORACLE_KEY, "", view=match_views[0], view_search=tuple(match_views[1:]))
    dreams = [dir_dreams[i] if dir_dreams else render_pngs(rend, s, [dream_view])[dream_view]
              for i, s in zip(ids, builds)]
    atts = [attempt_for(s, render_pngs(rend, s, list(match_views)), idx=j)
            for j, s in enumerate(builds)]

    rng = np.random.default_rng(seed + 1)
    items, own, dist_all = [], [], []
    for i, png in enumerate(dreams):
        ep = CalibEpisode(DreamVersion(ORACLE_KEY, 0, png, "oracle", source), ds, rend)
        others = rng.choice(np.delete(np.arange(len(builds)), i), size=k, replace=False)
        s_own = float(fn(ep, atts[i]).value)
        s_dist = np.array([float(fn(ep, atts[int(j)]).value) for j in others])
        rank = 1 + int((s_dist >= s_own).sum())          # ties count against the scorer
        items.append({"build": ids[i], "own": round(s_own, 4),
                      "best_distractor": round(float(s_dist.max()), 4), "rank": rank})
        own.append(s_own)
        dist_all.append(s_dist)
        if verbose and (i + 1) % 8 == 0:
            print(f"  calibrate {name}: {i + 1}/{len(dreams)}", flush=True)

    ranks = np.array([it["rank"] for it in items])
    d = np.concatenate(dist_all)
    return {
        "scorer": core.spec_str(scorer_spec), "n": len(builds), "distractors": k,
        "chance_top1": round(1.0 / (k + 1), 4),
        "top1": round(float((ranks == 1).mean()), 4),
        "mrr": round(float((1.0 / ranks).mean()), 4),
        "margin": round(float(np.mean([it["own"] - it["best_distractor"] for it in items])), 4),
        "own_mean": round(float(np.mean(own)), 4),
        "baseline": round(float(d.mean()), 4), "baseline_std": round(float(d.std()), 4),
        "renderer": getattr(rend, "name", str(renderer)), "dream_view": dream_view,
        "match_views": list(match_views), "source": source, "seed": seed,
        "elapsed_s": round(time.time() - t0, 2), "items": items,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer", required=True, help="scorer spec, e.g. 'embed?backbone=siglipB'")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--distractors", type=int, default=31)
    ap.add_argument("--dreams-dir", default=None)
    ap.add_argument("--renderer", default="textured")
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="synthetic builds (offline)")
    ap.add_argument("--out", default=None, help="write the full JSON here")
    a = ap.parse_args(argv)
    res = run(a.scorer, a.n, a.distractors, a.dreams_dir, renderer=a.renderer,
              corpus=a.corpus, seed=a.seed, verbose=True,
              structures=synthetic_builds(a.n, a.seed) if a.synthetic else None)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


__all__ = ["BASELINES", "CalibEpisode", "attempt_for", "main", "run", "synthetic_builds",
           "train_builds"]

if __name__ == "__main__":
    main()
