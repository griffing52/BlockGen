"""Token-budget batches over one or more shards, mixed by weight."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from blockgen.corpus.shards import Shard, find
from blockgen.utils.serialize import PAD_TOKEN


def open_shards(specs: Sequence[Tuple[str, float]], tok_hash: Optional[str],
                kind: str = "voxels", augment: str = "") -> Tuple[List[Shard], np.ndarray, str]:
    shards = [Shard(find(name, tok_hash, kind, augment)) for name, _ in specs]
    hashes = {s.spec["tokenizer"] for s in shards}
    if len(hashes) != 1:
        raise ValueError(f"shards disagree on tokenizer: {hashes}")
    w = np.array([w for _, w in specs], np.float64)
    return shards, w / w.sum(), hashes.pop()


class MixedBatches:
    """Infinite iterator of (input_ids, targets, pad_mask) within a token budget.

    Draws a buffer of sequences (shard chosen by weight, sequence uniformly within
    it), sorts the buffer by length so a batch pads little, cuts it into batches no
    larger than ``batch_tokens`` padded, and yields the batches in random order.
    """

    def __init__(self, shards: List[Shard], weights: np.ndarray, batch_tokens: int, max_seq_len: int,
                 seed: int = 0, buffer_batches: int = 64, quiet: bool = False):
        self.shards, self.weights = shards, weights
        self.batch_tokens, self.max_seq_len = batch_tokens, max_seq_len
        self.rng = np.random.default_rng(seed)
        self.lengths = [s.lengths() for s in shards]
        # A sequence longer than the context is dropped, never truncated: a truncated
        # sequence has no EOS, and a model that sees those learns that long builds
        # never end (T58: samples ran to the token budget).
        self.eligible = [np.flatnonzero(l <= max_seq_len) for l in self.lengths]
        for s, e, l in zip(shards, self.eligible, self.lengths):
            if len(e) < len(l) and not quiet:
                print(f"[data] {s.spec['tier']}: {len(l) - len(e)} of {len(l)} sequences exceed max_seq_len {max_seq_len}; dropped", flush=True)
        self.buffer_batches = buffer_batches
        self.seqs_seen = 0

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "seqs_seen": self.seqs_seen}

    def load_state(self, st: dict) -> None:
        self.rng.bit_generator.state = st["rng"]; self.seqs_seen = st["seqs_seen"]

    def _draw(self, n: int) -> List[Tuple[int, int]]:
        which = self.rng.choice(len(self.shards), size=n, p=self.weights)
        out = []
        for s in which:
            e = self.eligible[s]
            out.append((int(s), int(e[self.rng.integers(len(e))])))
        return out

    def _collate(self, items: List[np.ndarray]):
        L = max(len(x) for x in items)
        arr = np.full((len(items), L), PAD_TOKEN, np.int64)
        for i, x in enumerate(items):
            arr[i, :len(x)] = x
        t = torch.from_numpy(arr)
        inp, tgt = t[:, :-1], t[:, 1:]
        return inp, tgt, inp == PAD_TOKEN

    def __iter__(self) -> Iterator:
        while True:
            # enough sequences for ~buffer_batches batches at the median length
            med = max(64, int(np.median(np.concatenate([l[:10000] for l in self.lengths]))))
            n = max(8, self.buffer_batches * self.batch_tokens // med)
            picks = self._draw(n)
            lens = np.array([self.lengths[s][i] for s, i in picks])
            order = np.argsort(lens, kind="stable")
            batches, cur, cur_max = [], [], 0
            for j in order:
                L = int(lens[j])
                if cur and max(cur_max, L) * (len(cur) + 1) > self.batch_tokens:
                    batches.append(cur); cur, cur_max = [], 0
                cur.append(picks[j]); cur_max = max(cur_max, L)
            if cur:
                batches.append(cur)
            self.rng.shuffle(batches)
            for b in batches:
                items = [self.shards[s][i] for s, i in b]
                self.seqs_seen += len(items)
                yield self._collate(items)


def eval_batches(shard: Shard, n_seqs: int, batch_tokens: int, max_seq_len: int):
    """The first ``n_seqs`` sequences that fit the context, in fixed order, token-budget batched."""
    all_lens = shard.lengths()
    idx = np.flatnonzero(all_lens <= max_seq_len)[:n_seqs]
    lens = all_lens[idx]
    order = np.argsort(lens, kind="stable")
    cur, cur_max = [], 0
    mb = MixedBatches([shard], np.array([1.0]), batch_tokens, max_seq_len, quiet=True)
    for j in order:
        L = int(lens[j])
        if cur and max(cur_max, L) * (len(cur) + 1) > batch_tokens:
            yield mb._collate([shard[i] for i in cur]); cur, cur_max = [], 0
        cur.append(int(idx[j])); cur_max = max(cur_max, L)
    if cur:
        yield mb._collate([shard[i] for i in cur])
