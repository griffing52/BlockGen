"""Face nodes: each block has 6 faces, an edge glues face (i, +a) to face (j, -a).

Directions ``d = 0..5`` are ``+x, -x, +y, -y, +z, -z`` (the order of ``data.NBR``);
axis ``a = d // 2``; the complementary face of ``d`` is ``d ^ 1``.

The symmetric binary face-glue matrix (6N x 6N) has, by the constraints, only three
non-zero blocks: ``X_a[i, j]`` = "block i's +a face is glued to block j's -a face".
Storing those three N x N matrices makes symmetry hold by construction (one entry per
unordered face pair), complementarity hold by construction (+a only meets -a), and
"never within one block" is the zero diagonal. "At most one partner per face" is a
row/column constraint on each X_a (each row and each column has at most one 1), which
real chunks satisfy exactly (a face touches at most one cell).

A sequential *attachment* is ``(d, p, j)``: open face d of placed block p, unplaced
block j attached with its face ``d ^ 1``; j lands at ``pos[p] + e_d``. Its score is
``X_a[p, j]`` for ``d = +a`` and ``X_a[j, p]`` for ``d = -a`` (:func:`action_scores`).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np

DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.int64)
OPP = np.array([1, 0, 3, 2, 5, 4])


def cell_index(coords: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    return {tuple(int(v) for v in c): i for i, c in enumerate(np.asarray(coords))}


def glue_matrices(coords: np.ndarray) -> np.ndarray:
    """[3, N, N] bool, ``G[a, i, j]`` = coords[j] == coords[i] + e_{+a}."""
    N = len(coords)
    idx = cell_index(coords)
    G = np.zeros((3, N, N), bool)
    for i, c in enumerate(np.asarray(coords)):
        for a in range(3):
            j = idx.get(tuple(int(v) for v in c + DIRS[2 * a]))
            if j is not None:
                G[a, i, j] = True
    return G


def face_glue_full(G: np.ndarray) -> np.ndarray:
    """Expand the three axis matrices to the symmetric [6N, 6N] face matrix
    (face index ``6 * block + d``). Used by tests and as documentation."""
    _, N, _ = G.shape
    F = np.zeros((6 * N, 6 * N), bool)
    for a in range(3):
        for i, j in zip(*np.nonzero(G[a])):
            F[6 * i + 2 * a, 6 * j + 2 * a + 1] = True
            F[6 * j + 2 * a + 1, 6 * i + 2 * a] = True
    return F


def exposed_faces(coords: np.ndarray) -> np.ndarray:
    """[N, 6] bool: face d of block i is exposed (no block in the cell it faces)."""
    G = glue_matrices(coords)
    buried = np.zeros((len(coords), 6), bool)
    for a in range(3):
        buried[:, 2 * a] = G[a].any(1)
        buried[:, 2 * a + 1] = G[a].any(0)
    return ~buried


def typed_glue(coords: np.ndarray, types: np.ndarray, keep=None) -> Counter:
    """Multiset of glued face pairs as ``(type on the +a side, type on the -a side, a)``.
    Same-type blocks are exchangeable, so face-matching F1 compares these multisets."""
    coords = np.asarray(coords)
    if keep is not None:
        keep = list(keep)
        coords, types = coords[keep], np.asarray(types)[keep]
    G = glue_matrices(coords)
    out = Counter()
    for a in range(3):
        for i, j in zip(*np.nonzero(G[a])):
            out[(int(types[i]), int(types[j]), a)] += 1
    return out


def multiset_f1(pred: Counter, real: Counter) -> Tuple[float, float, float]:
    inter = sum(min(pred[k], real[k]) for k in pred.keys() & real.keys())
    p = inter / max(sum(pred.values()), 1)
    r = inter / max(sum(real.values()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# --------------------------------------------------------------------------- #
# attachment orders and teacher-forcing targets
# --------------------------------------------------------------------------- #
def random_attachment_order(coords: np.ndarray, rng: np.random.Generator,
                            seed_block: Optional[int] = None) -> np.ndarray:
    """A random valid attachment order of a connected chunk: start from a random block,
    then repeatedly pick a uniformly random (open face, unplaced neighbour) pair."""
    N = len(coords)
    idx = cell_index(coords)
    nbr = [[idx.get(tuple(int(v) for v in coords[i] + DIRS[d])) for d in range(6)] for i in range(N)]
    s = int(rng.integers(N)) if seed_block is None else int(seed_block)
    order, placed = [s], np.zeros(N, bool)
    placed[s] = True
    frontier = [j for j in nbr[s] if j is not None]
    while frontier:
        j = frontier.pop(int(rng.integers(len(frontier))))
        if placed[j]:
            continue
        placed[j] = True
        order.append(j)
        frontier += [k for k in nbr[j] if k is not None and not placed[k]]
    if len(order) < N:                         # disconnected input: append the rest
        order += [i for i in range(N) if not placed[i]]
    return np.array(order, np.int64)


def attach_targets(coords: np.ndarray, types: np.ndarray, placed: np.ndarray) -> np.ndarray:
    """[6, N] int: for placed block p and direction d, the type of the real block in the
    cell p faces if that block is still unplaced, else -1. An attachment ``(d, p, j)`` is
    correct iff ``types[j] == tgt[d, p]`` (any unplaced block of the right type: same-type
    blocks are exchangeable, and every face pointing at the same cell counts, which is
    T24e's cell-marginalized target)."""
    N = len(coords)
    idx = cell_index(coords)
    tgt = -np.ones((6, N), np.int64)
    for p in np.nonzero(placed)[0]:
        for d in range(6):
            k = idx.get(tuple(int(v) for v in coords[p] + DIRS[d]))
            if k is not None and not placed[k]:
                tgt[d, p] = types[k]
    return tgt


def correct_actions(tgt: np.ndarray, types: np.ndarray, placed: np.ndarray) -> np.ndarray:
    """[6, N, N] bool correct set C[d, p, j] from :func:`attach_targets`."""
    return (tgt[:, :, None] == types[None, None, :]) & placed[None, :, None] & ~placed[None, None, :]


def action_scores(X):
    """[3, N, N] axis scores -> [6, N, N] attachment scores A[d, p, j] (numpy or torch;
    batched inputs [..., 3, N, N] are supported)."""
    try:
        import torch
        if isinstance(X, torch.Tensor):
            T = X.transpose(-1, -2)
            return torch.stack([X[..., 0, :, :], T[..., 0, :, :], X[..., 1, :, :], T[..., 1, :, :],
                                X[..., 2, :, :], T[..., 2, :, :]], -3)
    except ImportError:                      # pragma: no cover
        pass
    T = np.swapaxes(X, -1, -2)
    return np.stack([X[..., 0, :, :], T[..., 0, :, :], X[..., 1, :, :], T[..., 1, :, :],
                     X[..., 2, :, :], T[..., 2, :, :]], -3)


def buried_state(pos: np.ndarray, placed: np.ndarray) -> np.ndarray:
    """[N, 6] int face state fed to the model: 0 unplaced, 1 placed and open,
    2 placed and facing a placed block."""
    N = len(pos)
    st = np.zeros((N, 6), np.int64)
    occ = {tuple(int(v) for v in pos[i]) for i in np.nonzero(placed)[0]}
    for i in np.nonzero(placed)[0]:
        for d in range(6):
            st[i, d] = 2 if tuple(int(v) for v in pos[i] + DIRS[d]) in occ else 1
    return st
