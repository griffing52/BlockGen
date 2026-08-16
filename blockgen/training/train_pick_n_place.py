"""Training for the pick-and-place growth model.

Two losses over one encoder pass per build:

    pick   cross-entropy over the palette + STOP, at every step
    place  cross-entropy over open faces, at every step after the seed

The seed contributes a pick target and no place target (it has no parent), and
the final step contributes a STOP target and no place target. Both are handled by
`ignore_index`, so no step is silently dropped or double-counted.

**Reported separately, always.** A model can look fine on total loss while being
useless at one of the two jobs — picking the modal block everywhere is cheap, and
so is attaching everything to the most recent node. Per-head loss and per-head
top-1 accuracy are what tell those apart, and the placer's accuracy is reported
against the *legal-face baseline* (1 / number of legal faces) rather than against
zero, because the mask already does much of the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from blockgen.models.pick_n_place import (N_DIR, PAD, PIECE_OFFSET, STOP,
                                          PickAndPlace, PickAndPlaceConfig)
from blockgen.utils.data import Structure
from blockgen.utils.growth_order import GrowthSequence, structure_to_growth


@dataclass
class PieceCodec:
    """Palette <-> model ids. Ids 0/1 are reserved for STOP/PAD."""
    tokens: List[int]
    index: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_sequences(cls, seqs: Sequence[GrowthSequence],
                       limit: Optional[int] = None) -> "PieceCodec":
        from collections import Counter
        counts: Counter = Counter()
        for s in seqs:
            counts.update(s.pieces.tolist())
        toks = [t for t, _ in counts.most_common(limit)]
        return cls(tokens=sorted(toks),
                   index={t: i + PIECE_OFFSET for i, t in enumerate(sorted(toks))})

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_pieces(self) -> int:
        return len(self.tokens)

    def encode(self, token: int) -> Optional[int]:
        return self.index.get(int(token))

    def decode(self, model_id: int) -> int:
        i = int(model_id) - PIECE_OFFSET
        return self.tokens[i] if 0 <= i < len(self.tokens) else 0

    def to_json(self) -> dict:
        return {"tokens": [int(t) for t in self.tokens]}

    @classmethod
    def from_json(cls, d: dict) -> "PieceCodec":
        toks = sorted(int(t) for t in d["tokens"])
        return cls(tokens=toks, index={t: i + PIECE_OFFSET for i, t in enumerate(toks)})


class GrowthDataset(Dataset):
    """Growth sequences, truncated to `max_nodes` and encoded to model ids.

    Truncation is graceful: a prefix of a BFS growth order is still a connected
    build, so a truncated example is a smaller valid house rather than a corrupt
    one. `stats()` reports how much is being cut.
    """

    def __init__(self, structures: Sequence[Structure], codec: PieceCodec,
                 *, max_nodes: int = 256, ordering: str = "bfs",
                 oriented: bool = False, min_nodes: int = 8,
                 complete_only: bool = False):
        self.codec = codec
        self.max_nodes = max_nodes
        self.complete_only = complete_only
        self.n_truncated = 0
        self.items: List[GrowthSequence] = []
        self.n_dropped_oov = 0
        self.n_too_small = 0
        for s in structures:
            seq = structure_to_growth(s, ordering=ordering, oriented=oriented,
                                      max_nodes=max_nodes)
            if seq is None or seq.n_nodes < min_nodes:
                self.n_too_small += 1
                continue
            if complete_only and seq.n_nodes >= max_nodes:
                # Only builds that ended on their own. Truncated ones cannot
                # teach STOP, and when they dominate they drown it: at
                # max_nodes=384 only 13.8% of builds were complete, STOP fell to
                # 0.036% of pick targets, and the model stopped emitting it at
                # all (measured STOP rate 0.00).
                self.n_truncated += 1
                continue
            if any(codec.encode(t) is None for t in seq.pieces.tolist()):
                # A build containing a piece outside the palette would train the
                # picker on a target it cannot represent. T21 hit exactly this
                # and silently avoided the builds; count them instead.
                self.n_dropped_oov += 1
                continue
            self.items.append(seq)

    def __len__(self) -> int:
        return len(self.items)

    def stats(self) -> dict:
        sizes = [s.n_nodes for s in self.items]
        complete = [s.n_nodes < self.max_nodes for s in self.items]
        return {"n": len(self.items), "dropped_oov": self.n_dropped_oov,
                "too_small": self.n_too_small, "truncated_skipped": self.n_truncated,
                "median_nodes": float(np.median(sizes)) if sizes else 0.0,
                "max_nodes_seen": int(max(sizes)) if sizes else 0,
                # Only these carry a STOP target. If it is ~0 the model has no
                # signal for when a build is finished, whatever the losses say.
                "complete_frac": float(np.mean(complete)) if complete else 0.0}

    def __getitem__(self, i: int) -> dict:
        seq = self.items[i]
        n = seq.n_nodes
        pieces = np.array([self.codec.encode(t) for t in seq.pieces.tolist()],
                          dtype=np.int64)
        parent_piece = np.where(seq.parent >= 0, pieces[seq.parent.clip(min=0)], PAD)
        direction = np.where(seq.direction >= 0, seq.direction, N_DIR)
        return {
            "pieces": pieces,
            "direction": direction.astype(np.int64),
            "parent_piece": parent_piece.astype(np.int64),
            "coords": seq.coords.astype(np.int64),
            "legal": seq.placement_masks(),
            "port_target": seq.target_ports(),
            "n": n,
            # A sequence that hit the cap was cut mid-build, so its last step is
            # not an ending. Supervising STOP there teaches "stop at max_nodes"
            # instead of "stop when the build is done" -- measured: with every
            # build truncated to 384, the model emitted ~350 nodes regardless of
            # what it was shown (prefix-test length_corr = -0.05).
            "complete": n < self.max_nodes,
        }


def collate(batch: List[dict]) -> dict:
    """Right-pad to the batch max. Padded nodes are masked everywhere."""
    B = len(batch)
    N = max(int(b["n"]) for b in batch)

    pieces = np.full((B, N), PAD, dtype=np.int64)
    direction = np.full((B, N), N_DIR, dtype=np.int64)
    parent_piece = np.full((B, N), PAD, dtype=np.int64)
    coords = np.zeros((B, N, 3), dtype=np.int64)
    legal = np.zeros((B, N, N, N_DIR), dtype=bool)
    pad = np.ones((B, N), dtype=bool)

    pick_target = np.full((B, N + 1), -100, dtype=np.int64)
    place_target = np.full((B, N), -100, dtype=np.int64)

    for b, item in enumerate(batch):
        n = int(item["n"])
        pieces[b, :n] = item["pieces"]
        direction[b, :n] = item["direction"]
        parent_piece[b, :n] = item["parent_piece"]
        coords[b, :n] = item["coords"]
        legal[b, :n, :n, :] = item["legal"]
        pad[b, :n] = False
        # Pick: predict every piece, then STOP -- but only if this build really
        # ended. A truncated sequence gets no STOP target (stays -100), so STOP
        # is learned exclusively from genuinely-complete builds and keeps meaning
        # "finished" rather than "hit the cap".
        pick_target[b, :n] = item["pieces"]
        if item.get("complete", True):
            pick_target[b, n] = STOP
        # Place: node 0 is the seed and has no parent -> stays ignored.
        pt = item["port_target"]
        for t in range(1, n):
            parent_i, d = divmod(int(pt[t]), N_DIR)
            place_target[b, t] = parent_i * N_DIR + d

    return {
        "pieces": torch.from_numpy(pieces),
        "direction": torch.from_numpy(direction),
        "parent_piece": torch.from_numpy(parent_piece),
        "coords": torch.from_numpy(coords),
        "legal": torch.from_numpy(legal),
        "pad_mask": torch.from_numpy(pad),
        "pick_target": torch.from_numpy(pick_target),
        "place_target": torch.from_numpy(place_target),
    }


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    place_weight: float = 1.0
    grad_clip: float = 1.0
    device: str = "cuda"
    log_every: int = 1
    amp: bool = True
    num_workers: int = 0


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
    keep = target != -100
    if not keep.any():
        return 0.0, 0
    pred = logits.argmax(dim=-1)
    return float((pred[keep] == target[keep]).float().sum()), int(keep.sum())


def evaluate(model: PickAndPlace, loader: DataLoader, cfg: TrainConfig) -> dict:
    """Per-head loss and accuracy, plus the placer's legal-face chance baseline."""
    model.eval()
    dev = cfg.device
    tot = {k: 0.0 for k in ("pick_loss", "place_loss", "pick_hit", "place_hit",
                            "pick_n", "place_n", "chance_sum")}
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(dev) for k, v in batch.items()}
            pick, place = model(b["pieces"], b["direction"], b["parent_piece"],
                                b["coords"], b["legal"], b["pad_mask"])
            pl = F.cross_entropy(pick.reshape(-1, pick.size(-1)),
                                 b["pick_target"].reshape(-1), ignore_index=-100)
            ql = F.cross_entropy(place.reshape(-1, place.size(-1)),
                                 b["place_target"].reshape(-1), ignore_index=-100)
            h1, n1 = _accuracy(pick, b["pick_target"])
            h2, n2 = _accuracy(place, b["place_target"])
            tot["pick_loss"] += float(pl) * max(n1, 1)
            tot["place_loss"] += float(ql) * max(n2, 1)
            tot["pick_hit"] += h1; tot["pick_n"] += n1
            tot["place_hit"] += h2; tot["place_n"] += n2
            keep = b["place_target"] != -100
            if keep.any():
                n_legal = b["legal"].flatten(2).sum(-1).clamp(min=1).float()
                tot["chance_sum"] += float((1.0 / n_legal)[keep].sum())
    out = {
        "pick_loss": tot["pick_loss"] / max(tot["pick_n"], 1),
        "place_loss": tot["place_loss"] / max(tot["place_n"], 1),
        "pick_acc": tot["pick_hit"] / max(tot["pick_n"], 1),
        "place_acc": tot["place_hit"] / max(tot["place_n"], 1),
        "place_chance": tot["chance_sum"] / max(tot["place_n"], 1),
    }
    out["place_lift"] = out["place_acc"] / max(out["place_chance"], 1e-9)
    return out


def train(model: PickAndPlace, train_ds: GrowthDataset,
          val_ds: Optional[GrowthDataset] = None,
          cfg: Optional[TrainConfig] = None) -> Tuple[PickAndPlace, dict]:
    cfg = cfg or TrainConfig()
    dev = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg.device = dev
    model = model.to(dev)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=cfg.num_workers)
    val_loader = (DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=cfg.num_workers)
                  if val_ds is not None and len(val_ds) else None)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and dev != "cpu" and torch.cuda.is_bf16_supported()
    history: Dict[str, List[float]] = {"pick_loss": [], "place_loss": [],
                                       "pick_acc": [], "place_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        acc = {"pl": 0.0, "ql": 0.0, "n1": 0, "n2": 0, "h1": 0.0, "h2": 0.0}
        for batch in loader:
            b = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pick, place = model(b["pieces"], b["direction"], b["parent_piece"],
                                    b["coords"], b["legal"], b["pad_mask"])
                pick_loss = F.cross_entropy(pick.reshape(-1, pick.size(-1)),
                                            b["pick_target"].reshape(-1),
                                            ignore_index=-100)
                place_loss = F.cross_entropy(place.reshape(-1, place.size(-1)),
                                             b["place_target"].reshape(-1),
                                             ignore_index=-100)
                loss = pick_loss + cfg.place_weight * place_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            h1, n1 = _accuracy(pick.detach(), b["pick_target"])
            h2, n2 = _accuracy(place.detach(), b["place_target"])
            acc["pl"] += pick_loss.detach().item() * max(n1, 1); acc["n1"] += n1
            acc["ql"] += place_loss.detach().item() * max(n2, 1); acc["n2"] += n2
            acc["h1"] += h1; acc["h2"] += h2

        row = {"pick_loss": acc["pl"] / max(acc["n1"], 1),
               "place_loss": acc["ql"] / max(acc["n2"], 1),
               "pick_acc": acc["h1"] / max(acc["n1"], 1),
               "place_acc": acc["h2"] / max(acc["n2"], 1)}
        for k, v in row.items():
            history[k].append(v)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            msg = (f"[pnp] epoch {epoch:3d}  pick {row['pick_loss']:.4f}"
                   f"/{row['pick_acc']:.3f}  place {row['place_loss']:.4f}"
                   f"/{row['place_acc']:.3f}")
            if val_loader is not None:
                v = evaluate(model, val_loader, cfg)
                msg += (f"  | val place_acc {v['place_acc']:.3f}"
                        f" (chance {v['place_chance']:.3f},"
                        f" lift {v['place_lift']:.1f}x)")
                model.train()
            print(msg, flush=True)

    return model, history
