"""Factor a learned ``ClusterVocab`` into (shape ↑D4, rotation, family, variant).

Why. A 3D-BPE audit on ``houses_32`` (256 merges) found the piece vocabulary is
almost entirely single-material runs — 253/256 pieces use exactly one block token —
and that its *shapes* are trivial: canonicalized up to the horizontal dihedral group
D4, 256 merge slots encode only **16 distinct shapes**. 105/256 slots (41%) are the
same shape AND the same block family, differing only by rotation or by the variant
bits in ``data``. A flat ``nn.Embedding`` over piece ids learns every one of those
from scratch — the same pathology T7 measured at the atomic level, where wood
variants came out near-orthogonal (within-family cos-sim 0.006 vs random 0.002).

Factoring the *embedding* rather than canonicalizing the *tokens* is deliberate.
Collapsing rotations into one token would require remapping the orientation bits in
``block_data`` (a log's axis, a stair's facing), and getting that wrong silently
emits wrong-facing blocks — the failure ``utils/augment.py`` already documents and
tolerates. Here the token stream is untouched, so a bad decomposition costs only
parameter sharing, never correctness. It also keeps the change comparable against
the T11/T12 baseline, since tokenization and data are unchanged.

A vertical oak-log pillar and a horizontal oak-log beam land at "same shape, same
family, different variant" — which is *true*: they really are different blocks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.tokenizers.cluster_bpe import ClusterVocab, Cell, Pattern


def d4_images(cells: Sequence[Cell]) -> List[Pattern]:
    """The 8 horizontal-D4 images of a cell set, re-anchored and sorted.

    Y (axis 1) is up; rotations act on X-Z, the mirror flips X. Block indices ride
    along unchanged — this canonicalizes *geometry* only, which is exactly why it
    is safe: no ``block_data`` orientation bits are touched.
    """
    out: List[Pattern] = []
    for mirror in (False, True):
        for k in range(4):
            t = []
            for (x, y, z, b) in cells:
                X, Z = (-x, z) if mirror else (x, z)
                for _ in range(k):
                    X, Z = Z, -X
                t.append((X, y, Z, b))
            mx = min(c[0] for c in t); my = min(c[1] for c in t); mz = min(c[2] for c in t)
            out.append(tuple(sorted((a - mx, b_ - my, c - mz, d) for a, b_, c, d in t)))
    return out


def canon_shape(cells: Sequence[Cell]) -> Tuple[Pattern, int]:
    """Canonical block-type-agnostic shape, and which D4 element reaches it."""
    stripped = [(x, y, z, 0) for x, y, z, _ in cells]
    imgs = d4_images(stripped)
    best = min(range(len(imgs)), key=lambda i: imgs[i])
    return imgs[best], best


@dataclass
class PieceFactors:
    """Per-piece factor indices, parallel to ``ClusterVocab.patterns``."""

    shape_idx: np.ndarray     # [n_pieces] canonical-shape id
    rot_idx: np.ndarray       # [n_pieces] D4 element (0-7) mapping piece -> canonical
    family_idx: np.ndarray    # [n_pieces] block family (legacy id)
    variant_idx: np.ndarray   # [n_pieces] block variant (legacy data), 0 if mixed
    n_shapes: int
    n_rots: int
    n_families: int
    n_variants: int

    def summary(self) -> str:
        return (f"pieces={len(self.shape_idx)} -> shapes={self.n_shapes} "
                f"rots={self.n_rots} families={self.n_families} "
                f"variants={self.n_variants} "
                f"(flat table would be {len(self.shape_idx)} rows)")


def build_piece_factors(cv: ClusterVocab) -> PieceFactors:
    """Decompose every piece in ``cv`` into (shape, rotation, family, variant)."""
    shape_ids: Dict[Pattern, int] = {}
    fam_ids: Dict[int, int] = {}
    var_ids: Dict[int, int] = {}

    shape_idx, rot_idx, family_idx, variant_idx = [], [], [], []
    for pattern in cv.patterns:
        canon, rot = canon_shape(pattern)
        shape_idx.append(shape_ids.setdefault(canon, len(shape_ids)))
        rot_idx.append(rot)

        # 98.8% of learned pieces are single-material; for the rare mixed piece take
        # the majority family and drop the variant (there is no single right answer).
        fams = Counter(cv.block_index_to_pair[b][0] for _, _, _, b in pattern)
        fam = fams.most_common(1)[0][0]
        family_idx.append(fam_ids.setdefault(fam, len(fam_ids)))

        variants = {cv.block_index_to_pair[b][1] for _, _, _, b in pattern}
        var = variants.pop() if len(variants) == 1 else 0
        variant_idx.append(var_ids.setdefault(var, len(var_ids)))

    return PieceFactors(
        shape_idx=np.asarray(shape_idx, dtype=np.int64),
        rot_idx=np.asarray(rot_idx, dtype=np.int64),
        family_idx=np.asarray(family_idx, dtype=np.int64),
        variant_idx=np.asarray(variant_idx, dtype=np.int64),
        n_shapes=len(shape_ids), n_rots=8,
        n_families=len(fam_ids), n_variants=len(var_ids),
    )
