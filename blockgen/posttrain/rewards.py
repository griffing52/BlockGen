"""Rewards: higher is better, finite, pure functions of a list of structures.

Every reward here is CPU only and reads nothing under ``blockgen.eval.bench``;
the novelty penalty reuses ``blockgen.eval.novelty`` so its nearest-train IoU is
the number the fast tier reports. The rewards a rung may train on are the ones
with ``pillar=False``; the pillar inputs of BlockScore are listed by name as
placeholders so a config that asks for one is refused, not silently honoured.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
from scipy import ndimage as ndi

from blockgen.eval.facade_stats import face_stats
from blockgen.utils.data import Structure

#: BlockScore pillar inputs. Training on any of these breaks the honesty rule in
#: the package docstring, so each is a placeholder that ``get`` refuses by default.
PILLAR_SIGNALS = ("coherence", "geometry_scalars", "traversal", "geom_kid", "mv_dino_kid", "palette_jsd_exact")

#: What an empty build scores on face solidity: below any non-empty build.
EMPTY_SCORE = -4.0

#: Rewards a rung may train or select on from the command line: non-pillar and
#: constructible without arguments. ``composite`` is reached through ``--weights``.
CLI_REWARDS = ("face_solidity", "lcc26", "novelty_penalty", "face_door", "face_solidity_door")

#: Separates a signal name from its weight in ``trained_on_signals`` tokens
#: (``lcc26@0.5``). Shell-, path- and override-safe: no ``:`` (train specs, arm
#: specs), no ``,`` (override lists), no ``/`` or space (tier directories).
WEIGHT_SEP = "@"


class Reward(Protocol):
    name: str
    pillar: bool

    def score(self, structs: Sequence[Structure]) -> np.ndarray: ...


class PillarReward(ValueError):
    """Raised when a BlockScore pillar input is requested as a reward without ``allow_pillar``."""


def _finite(x: np.ndarray, fill: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, fill)


class FaceSolidity:
    """Solid, readable faces: ``lcc - (face_jumps + face_holes + face_lone + dust_blocks)``.

    Every term is a fraction in [0, 1], so a clean hollow box scores 1 and a build
    of relief speckle, pinholes and loose blocks falls toward -3. The terms are the
    ``face_stats`` columns of the facade sheets, which is why this is a reward and
    not a pillar: the fast tier never reads a face image.
    """
    name = "face_solidity"
    pillar = False

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        out = np.full(len(structs), EMPTY_SCORE)
        for i, s in enumerate(structs):
            if not s.occupied_mask.any():
                continue
            st = face_stats(s)
            out[i] = st["lcc"] - (st["face_jumps"] + st["face_holes"] + st["face_lone"] + st["dust_blocks"])
        return _finite(out, EMPTY_SCORE)


#: Door blocks (wooden and iron), as in ``blockgen.eval.traversal.DOOR_IDS``.
DOOR_IDS = (64, 71, 193, 194, 195, 196, 197)


class FaceDoor:
    """1 when a door block is the first hit of some side-face ray, else 0.

    A property of the faces, not of the walk: it reads the four side projections
    only and never runs the player model, so it is not the traversal pillar (whose
    enterable share is measured by the bench and is the transfer to report). Real
    houses_32 builds show a door on at least one side 86% of the time.
    """
    name = "face_door"
    pillar = False

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        out = np.zeros(len(structs))
        for i, s in enumerate(structs):
            ids = s.block_ids
            if not (ids != 0).any():
                continue
            for axis in (0, 2):
                for rev in (False, True):
                    a = np.moveaxis(ids, axis, 0)
                    a = a[::-1] if rev else a
                    occ = a != 0
                    first = np.take_along_axis(a, occ.argmax(0)[None], 0)[0]
                    if np.isin(first[occ.any(0)], DOOR_IDS).any():
                        out[i] = 1.0
                        break
                if out[i]:
                    break
        return out


class FaceSolidityDoor:
    """``face_solidity + face_door``: solid, readable faces, and a visible door when one is on offer."""
    name = "face_solidity_door"
    pillar = False

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        return FaceSolidity().score(structs) + FaceDoor().score(structs)


class LCC26:
    """Share of the blocks in the largest 26-connected component; empty builds score 0.

    26-connectivity, not the 6-connectivity of the coherence pillar: a diagonal
    roof line is one piece here and several there, and that gap is the reason
    coherence movement is pre-registered as transfer in the package docstring.
    """
    name = "lcc26"
    pillar = False

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        out = np.zeros(len(structs))
        for i, s in enumerate(structs):
            occ = s.occupied_mask
            n = int(occ.sum())
            if n == 0:
                continue
            lab, k = ndi.label(occ, structure=np.ones((3, 3, 3)))
            out[i] = np.bincount(lab.ravel())[1:].max() / n if k else 0.0
        return out


class NoveltyPenalty:
    """Minus the nearest-train occupancy IoU, minus one more for a duplicate.

    The IoU is ``blockgen.eval.novelty``'s on the canonical grid, so the number a
    best-of-N run selected on is the number the fast tier later reports as
    ``voxel_nn_iou``. ``train`` defaults to the train side of the houses32-v1
    split, loaded read-only; pass the structures to score against anything else.
    Novelty is a disqualification gate in BlockScore, not a pillar, so this is
    allowed as a reward.
    """
    name = "novelty_penalty"
    pillar = False

    def __init__(self, train: Optional[Sequence[Structure]] = None, grid: int = 24,
                 dup_threshold: float = 0.95, corpus: str = "houses_32", seed: int = 0):
        from blockgen.eval.novelty import voxelize_occupancy
        from blockgen.utils.serialize import build_block_vocab
        if train is None:
            from blockgen.eval.bench.splits import load_split, split_structures
            train = split_structures(load_split(corpus, seed), "train")
        self.train = list(train)
        self.grid, self.dup_threshold = grid, dup_threshold
        self.vocab = build_block_vocab(self.train, max_dim=grid)
        self.train_occ, _ = voxelize_occupancy(self.train, grid, self.vocab)

    def nn_iou(self, structs: Sequence[Structure]) -> np.ndarray:
        from blockgen.eval.novelty import _iou_matrix, voxelize_occupancy
        if not len(structs) or not len(self.train):
            return np.zeros(len(structs))
        gen_occ, _ = voxelize_occupancy(structs, self.grid, self.vocab)
        return _iou_matrix(gen_occ, self.train_occ).max(axis=1)

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        iou = self.nn_iou(structs)
        return -(iou + (iou >= self.dup_threshold))


class Composite:
    """Weighted sum of other rewards' raw scores: ``{"face_solidity": 1.0, "novelty_penalty": 0.5}``.

    Raw scales, not z-scores, so a weight means the same thing on every batch.
    Pillar status is inherited: a composite over a pillar input is a pillar input.
    The name is ``composite-<signal>[@w]-...`` over the sorted signal names, so it
    is safe inside a tier name, a best-of-N directory and an ``--arms`` spec;
    ``signals`` are the same tokens one per part, which is what a run records in
    ``trained_on_signals`` and what ``from_signals`` rebuilds the reward from.
    """
    pillar = False

    def __init__(self, weights: Dict[str, float], allow_pillar: bool = False, **kw):
        if not weights:
            raise ValueError("composite needs at least one weighted reward")
        self.weights = {k: float(weights[k]) for k in sorted(weights)}
        if "composite" in self.weights:
            raise ValueError("a composite cannot nest another composite")
        self.parts = {k: get(k, allow_pillar=allow_pillar, **kw.get(k, {})) for k in self.weights}
        self.pillar = any(r.pillar for r in self.parts.values())
        self.signals: List[str] = [signal_token(k, w) for k, w in self.weights.items()]
        self.name = "composite-" + "-".join(self.signals)

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        out = np.zeros(len(structs))
        for k, w in self.weights.items():
            if w:
                out += w * np.asarray(self.parts[k].score(structs), dtype=np.float64)
        return out


def signal_token(name: str, weight: float = 1.0) -> str:
    """``name`` for weight 1, else ``name@w``; the form ``trained_on_signals`` carries."""
    if any(c in name for c in ":,/ " + WEIGHT_SEP):
        raise ValueError(f"reward name {name!r} is not safe in a tier or arm spec")
    return name if float(weight) == 1.0 else f"{name}{WEIGHT_SEP}{float(weight):g}"


def parse_signal(token: str) -> tuple:
    """Inverse of ``signal_token``: ``(name, weight)``."""
    name, _, w = str(token).partition(WEIGHT_SEP)
    return name, float(w) if w else 1.0


def parse_weights(spec: str) -> Dict[str, float]:
    """``--weights face_solidity:2,lcc26:0.5`` (a missing weight is 1) to a weights dict."""
    out: Dict[str, float] = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        out[name] = float(w) if w else 1.0
    if not out:
        raise ValueError("--weights needs at least one name[:weight]")
    return out


def signals_of(reward: Reward) -> List[str]:
    """The ``trained_on_signals`` tokens a run rewarded on ``reward`` records."""
    return list(getattr(reward, "signals", None) or [reward.name])


def from_signals(tokens: Sequence[str], allow_pillar: bool = False, **kw) -> Reward:
    """Rebuild the reward a run trained on from its ``trained_on_signals``.

    One unweighted token is that reward; anything else is the composite of the
    parsed ``name[@w]`` tokens. ``select=reward`` uses this so a run that trained
    on a weighted composite is selected on the same composite.
    """
    pairs = [parse_signal(t) for t in tokens if str(t)]
    if not pairs:
        raise ValueError("no signals to build a reward from")
    if len(pairs) == 1 and pairs[0][1] == 1.0:
        return get(pairs[0][0], allow_pillar=allow_pillar, **kw.get(pairs[0][0], {}))
    return get("composite", weights=dict(pairs), allow_pillar=allow_pillar, **kw)


class PillarSignal:
    """A BlockScore pillar input, named so a request for it fails loudly.

    Scoring is deliberately unimplemented: the pillars are computed by the bench
    against a fixed reference, and a reward that recomputed one here would be
    training on the test.
    """
    pillar = True

    def __init__(self, name: str):
        if name not in PILLAR_SIGNALS:
            raise KeyError(name)
        self.name = name

    def score(self, structs: Sequence[Structure]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} is a BlockScore pillar input; it is measured by the bench, "
                                  "never optimised. See blockgen/posttrain/__init__.py.")


REWARDS: Dict[str, Callable[..., Reward]] = {
    "face_solidity": FaceSolidity,
    "lcc26": LCC26,
    "novelty_penalty": NoveltyPenalty,
    "face_door": FaceDoor,
    "face_solidity_door": FaceSolidityDoor,
    "composite": Composite,
    **{name: (lambda name=name, **kw: PillarSignal(name)) for name in PILLAR_SIGNALS},
}


def get(name: str, allow_pillar: bool = False, **kw) -> Reward:
    """Build a reward by name. A pillar input is refused unless ``allow_pillar=True``.

    ``allow_pillar`` only lets the placeholder be constructed (``Composite`` uses
    it to inherit pillar status, tests to read the name); a pillar placeholder
    still cannot score, so no run can be rewarded or selected on one.
    """
    if name not in REWARDS:
        raise KeyError(f"unknown reward {name!r}; known: {sorted(REWARDS)}")
    if name == "composite":
        kw.setdefault("allow_pillar", allow_pillar)
    r = REWARDS[name](**kw)
    if r.pillar and not allow_pillar:
        raise PillarReward(f"{r.name} is a BlockScore pillar input; a post-trained arm may not be rewarded on it "
                           "(blockgen/posttrain/__init__.py); the pillars are measured by the bench, never optimised.")
    return r
