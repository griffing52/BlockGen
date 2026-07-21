"""3D byte-pair encoding: learn a vocabulary of connected block *clusters*.

Motivation. Our AR track emits one token per voxel with no locality prior, so
(a) samples fragment into incoherent voxel clouds, and (b) sequences are so long
that dense builds must be crushed to 12³ to fit. Both hurt cohesion.

3D-BPE attacks both. Exactly like text BPE, we start from atomic tokens (here one
per block-class voxel) and greedily merge the most frequent *adjacent* pair into a
new composite token — but the pair is 6-connected in 3D and the merge records the
**relative offset** between the two pieces. Iterating grows a vocabulary of rigid,
connected multi-voxel "pieces" (a wall corner, a 1x3 plank run, a roof-slope unit).

A build is then re-tokenized by **replaying the learned merges** on its atomic
labeling — deterministic, so a build tokenizes identically whether or not it was in
the BPE-learning corpus. Emission is ``[BOS, (X, Y, Z, PIECE)*, EOS]`` where X,Y,Z
is the piece's anchor (min corner). Two consequences:

* **Cohesion**: every emitted token is a connected chunk, so the model cannot emit
  the single-voxel noise that flat per-voxel AR does.
* **Reach**: pieces cover several voxels each, so sequences shrink ~2-4x and AR can
  train at higher canonical resolution.

Air is never a token — pieces are occupied-only sets of offsets; internal gaps are
absence. This is exactly the representation that ports to a LEGO piece/brick.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure, _token_for
from blockgen.utils.serialize import (BOS_TOKEN, EOS_TOKEN, NUM_SPECIAL, PAD_TOKEN,
                                       BlockVocab, build_block_vocab)

Coord = Tuple[int, int, int]
Cell = Tuple[int, int, int, int]          # (dx, dy, dz, block_index)
Pattern = Tuple[Cell, ...]
_NBR = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _reanchor(cells: Sequence[Cell]) -> Pattern:
    """Shift a set of cells so its min corner is (0,0,0); return sorted tuple."""
    mx = min(c[0] for c in cells); my = min(c[1] for c in cells); mz = min(c[2] for c in cells)
    return tuple(sorted((x - mx, y - my, z - mz, b) for x, y, z, b in cells))


@dataclass
class ClusterVocab:
    """Token vocabulary over learned block clusters.

    Layout: [PAD,BOS,EOS] + coord tokens (max_dim) + piece tokens (n_pieces).
    Piece 0..num_blocks-1 are the atomic single-voxel pieces (piece id == base
    block index); ids >= num_blocks are learned merges.
    """

    max_dim: int
    patterns: List[Pattern]                          # piece id -> pattern
    block_index_to_pair: List[Tuple[int, int]]       # base block index -> (id, data)
    merges: List[Tuple[int, int, Coord]] = field(default_factory=list)  # (pa, pb, delta) in order
    num_blocks: int = 0
    oriented: bool = False                           # orientation-preserving token keys

    @property
    def coord_offset(self) -> int:
        return NUM_SPECIAL

    @property
    def piece_offset(self) -> int:
        return NUM_SPECIAL + self.max_dim

    @property
    def n_pieces(self) -> int:
        return len(self.patterns)

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.max_dim + self.n_pieces

    def coord_id(self, c: int) -> int:
        if not (0 <= c < self.max_dim):
            raise ValueError(f"coordinate {c} out of range [0,{self.max_dim})")
        return self.coord_offset + int(c)

    def piece_id(self, pid: int) -> int:
        return self.piece_offset + int(pid)

    def is_coord(self, tok: int) -> bool:
        return self.coord_offset <= tok < self.coord_offset + self.max_dim

    def is_piece(self, tok: int) -> bool:
        return self.piece_offset <= tok < self.piece_offset + self.n_pieces

    def decode_coord(self, tok: int) -> int:
        return int(tok) - self.coord_offset

    def decode_piece(self, tok: int) -> int:
        return int(tok) - self.piece_offset


# --------------------------------------------------------------------------- #
# per-build labeling: occupied voxels grouped into piece instances
# --------------------------------------------------------------------------- #
def _atomic_labeling(s: Structure, base: BlockVocab):
    """Init: one atomic instance per occupied voxel. Returns (inst, owner, next_id)."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    inst: Dict[int, dict] = {}
    owner: Dict[Coord, int] = {}
    nid = 0
    for x, y, z in np.argwhere(occ).tolist():
        tok = _token_for(int(c.block_ids[x, y, z]), int(c.block_data[x, y, z]),
                         oriented=base.oriented)
        bidx = base.block_token_to_id.get(tok)
        if bidx is None:
            continue
        inst[nid] = {"piece": bidx, "anchor": (x, y, z), "vox": [(x, y, z)]}
        owner[(x, y, z)] = nid
        nid += 1
    return inst, owner, nid


def _canon_pair(pa: int, aa: Coord, pb: int, ab: Coord):
    """Canonical (piece_first, piece_second, delta) for an adjacency, direction-free."""
    dab = (ab[0] - aa[0], ab[1] - aa[1], ab[2] - aa[2])
    if pa < pb:
        return (pa, pb, dab)
    if pa > pb:
        return (pb, pa, (-dab[0], -dab[1], -dab[2]))
    neg = (-dab[0], -dab[1], -dab[2])
    return (pa, pa, max(dab, neg))


def _apply_merge(inst: Dict[int, dict], owner: Dict[Coord, int], nid: int,
                 pa: int, pb: int, delta: Coord, new_pid: int) -> int:
    """Greedily merge every non-conflicting (pa @ A, pb @ A+delta) pair into new_pid."""
    by_key: Dict[Tuple[int, Coord], List[int]] = {}
    for i, d in inst.items():
        by_key.setdefault((d["piece"], d["anchor"]), []).append(i)
    consumed: set = set()
    same = pa == pb
    for i in sorted(inst.keys()):
        if i in consumed:
            continue
        di = inst[i]
        # try instance i as the "first" piece (pa) with partner at anchor+delta
        for first, second, dl in ((pa, pb, delta),) + (((pb, pa, (-delta[0], -delta[1], -delta[2])),) if same else ()):
            if di["piece"] != first:
                continue
            ax, ay, az = di["anchor"]
            want = (ax + dl[0], ay + dl[1], az + dl[2])
            cand = [j for j in by_key.get((second, want), []) if j != i and j not in consumed]
            if not cand:
                continue
            j = cand[0]
            dj = inst[j]
            vox = di["vox"] + dj["vox"]
            anchor = (min(v[0] for v in vox), min(v[1] for v in vox), min(v[2] for v in vox))
            consumed.add(i); consumed.add(j)
            inst.pop(i); inst.pop(j)
            inst[nid] = {"piece": new_pid, "anchor": anchor, "vox": vox}
            for v in vox:
                owner[v] = nid
            by_key.setdefault((new_pid, anchor), []).append(nid)
            nid += 1
            break
    return nid


# Legacy block ids whose objects occupy TWO voxels that share the id — beds (26),
# wooden/iron doors (64/71/193-197), and double-plants (175, sunflower/rose bush/tall
# grass/large fern). Their two halves should be ONE piece token, not two, both for
# semantic coherence and because the Minecraft mod otherwise places only one half.
MULTI_BLOCK_IDS = frozenset({26, 64, 71, 193, 194, 195, 196, 197, 175})


def _forced_family_keys(labelings, base: BlockVocab):
    """Canonical (pa, pb, delta) merge keys for adjacent same-family atomic voxels.

    Both halves of a bed/door/double-plant share a legacy id, so a qualifying pair is
    two 6-adjacent atomic voxels whose block ids are equal and in ``MULTI_BLOCK_IDS``.
    Returned deterministically sorted so the vocab is reproducible.
    """
    id_of = [base.block_index_to_pair[b][0] for b in range(base.num_blocks)]
    keys: set = set()
    for inst, owner, _ in labelings:
        for (x, y, z), a in owner.items():
            pa = inst[a]["piece"]
            if pa >= len(id_of) or id_of[pa] not in MULTI_BLOCK_IDS:
                continue
            for dx, dy, dz in _NBR:
                w = (x + dx, y + dy, z + dz)
                b = owner.get(w)
                if b is None or b == a or w < (x, y, z):
                    continue
                pb = inst[b]["piece"]
                if pb < len(id_of) and id_of[pb] == id_of[pa]:  # same-family halves
                    keys.add(_canon_pair(pa, inst[a]["anchor"], pb, inst[b]["anchor"]))
    return sorted(keys)


def learn_clusters(structs: Sequence[Structure], max_dim: int, n_merges: int = 120,
                   max_corpus: int = 400, min_count: int = 3, seed: int = 0,
                   force_families: bool = True, oriented: bool = False,
                   verbose: bool = True) -> ClusterVocab:
    """Learn a ClusterVocab by BPE-style greedy merging over a corpus.

    With ``force_families`` (default), multi-voxel objects (beds/doors/double-plants;
    see ``MULTI_BLOCK_IDS``) are merged into single pieces FIRST, as guaranteed
    top-priority merges, before frequency-based BPE — so they always tokenize as one
    piece regardless of how often they occur. These are extra merges, not counted
    against ``n_merges``.
    """
    base = build_block_vocab(structs, max_dim=max_dim, oriented=oriented)
    patterns: List[Pattern] = [((0, 0, 0, b),) for b in range(base.num_blocks)]
    merges: List[Tuple[int, int, Coord]] = []

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(structs))[:max_corpus]
    corpus = [structs[int(i)] for i in idx]
    labelings = []
    for s in corpus:
        inst, owner, nid = _atomic_labeling(s, base)
        labelings.append([inst, owner, nid])

    def _commit(pa, pb, delta):
        new_pid = len(patterns)
        merged = list(patterns[pa]) + [(dx + delta[0], dy + delta[1], dz + delta[2], bb)
                                       for dx, dy, dz, bb in patterns[pb]]
        patterns.append(_reanchor(merged))
        merges.append((pa, pb, delta))
        for lab in labelings:
            lab[2] = _apply_merge(lab[0], lab[1], lab[2], pa, pb, delta, new_pid)
        return new_pid

    if force_families:
        forced = _forced_family_keys(labelings, base)
        for pa, pb, delta in forced:
            _commit(pa, pb, delta)
        if verbose:
            print(f"[BPE] forced {len(forced)} multi-block merges (beds/doors/plants) "
                  f"-> {len(patterns)} pieces", flush=True)

    for step in range(n_merges):
        counts: Counter = Counter()
        for inst, owner, _ in labelings:
            for (x, y, z), a in owner.items():
                for dx, dy, dz in _NBR:
                    w = (x + dx, y + dy, z + dz)
                    b = owner.get(w)
                    if b is None or b == a or w < (x, y, z):
                        continue  # w<v guard: count each face once
                    key = _canon_pair(inst[a]["piece"], inst[a]["anchor"],
                                      inst[b]["piece"], inst[b]["anchor"])
                    counts[key] += 1
        if not counts:
            break
        (pa, pb, delta), cnt = counts.most_common(1)[0]
        if cnt < min_count:
            break
        _commit(pa, pb, delta)
        if verbose and (step % 20 == 0 or step == n_merges - 1):
            avg = np.mean([len(l[0]) for l in labelings])
            print(f"[BPE] merge {step:3d}/{n_merges}  count={cnt}  "
                  f"pieces={len(patterns)}  avg_instances/build={avg:.0f}", flush=True)

    return ClusterVocab(max_dim=max_dim, patterns=patterns,
                        block_index_to_pair=base.block_index_to_pair,
                        merges=merges, num_blocks=base.num_blocks, oriented=oriented)


# --------------------------------------------------------------------------- #
# tokenize / detokenize
# --------------------------------------------------------------------------- #
def structure_to_cluster_tokens(s: Structure, cv: ClusterVocab) -> List[int]:
    """Tokenize by replaying the learned merges, then emit [BOS,(X,Y,Z,PIECE)*,EOS]."""
    base = BlockVocab(
        max_dim=cv.max_dim,
        block_token_to_id={_token_for(*p, oriented=cv.oriented): i
                           for i, p in enumerate(cv.block_index_to_pair)},
        id_to_block_token=[_token_for(*p, oriented=cv.oriented)
                           for p in cv.block_index_to_pair],
        block_index_to_pair=cv.block_index_to_pair, oriented=cv.oriented)
    c = s.crop_to_non_air()
    if max(c.shape) > cv.max_dim:
        raise ValueError(f"structure max dim {max(c.shape)} exceeds max_dim {cv.max_dim}")
    inst, owner, nid = _atomic_labeling(c, base)
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)

    order = sorted(inst.values(), key=lambda d: (d["anchor"][1], d["anchor"][2], d["anchor"][0]))
    tokens: List[int] = [BOS_TOKEN]
    for d in order:
        ax, ay, az = d["anchor"]
        tokens.extend([cv.coord_id(ax), cv.coord_id(ay), cv.coord_id(az), cv.piece_id(d["piece"])])
    tokens.append(EOS_TOKEN)
    assert max(tokens) < cv.vocab_size, "token id out of vocab range"
    return tokens


def cluster_tokens_to_structure(tokens: Sequence[int], cv: ClusterVocab) -> Structure:
    """Inverse: place each piece's pattern at its anchor. Robust to malformed output."""
    placed: List[Tuple[int, int, int, int]] = []  # (x,y,z,block_index)
    buf: List[int] = []
    for tok in tokens:
        tok = int(tok)
        if tok == EOS_TOKEN:
            break
        if tok in (BOS_TOKEN, PAD_TOKEN):
            continue
        if len(buf) < 3:
            if cv.is_coord(tok):
                buf.append(cv.decode_coord(tok))
            else:
                buf.clear()
        else:
            if cv.is_piece(tok):
                ax, ay, az = buf
                for dx, dy, dz, bidx in cv.patterns[cv.decode_piece(tok)]:
                    placed.append((ax + dx, ay + dy, az + dz, bidx))
            buf = []

    if not placed:
        return Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                         block_data=np.zeros((1, 1, 1), np.int32))
    arr = np.array(placed, dtype=np.int64)
    sx, sy, sz = int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1, int(arr[:, 2].max()) + 1
    bi = np.zeros((sx, sy, sz), np.int32)
    bd = np.zeros((sx, sy, sz), np.int32)
    for x, y, z, bidx in arr.tolist():
        if 0 <= bidx < len(cv.block_index_to_pair):
            rid, rdata = cv.block_index_to_pair[bidx]
            bi[x, y, z] = rid
            bd[x, y, z] = rdata
    return Structure(block_ids=bi, block_data=bd).crop_to_non_air()


def build_cluster_sequences(structs: Sequence[Structure], cv: ClusterVocab,
                            max_seq_len: int) -> Tuple[List[List[int]], List[int]]:
    """Tokenize structs; keep those within max_seq_len. Returns (sequences, kept_indices)."""
    seqs: List[List[int]] = []
    kept: List[int] = []
    for i, s in enumerate(structs):
        try:
            t = structure_to_cluster_tokens(s, cv)
        except (ValueError, AssertionError):
            continue
        if 2 < len(t) <= max_seq_len:
            seqs.append(t)
            kept.append(i)
    return seqs, kept
