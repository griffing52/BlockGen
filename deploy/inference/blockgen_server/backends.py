"""Model backends: everything needed to turn a request into a stream of blocks.

A backend owns one checkpoint plus the vocabulary it was trained against, and
exposes a single ``stream()`` generator. The server never imports a model class or
touches a token id — adding a new model is a ``models.json`` entry plus, at most, a
new ``Backend`` subclass here.

Three kinds exist today:

* ``piece_ar``      — ``VoxelTransformerAR2`` over 3D-BPE piece tokens (native_bpe).
* ``voxel_ar``      — ``VoxelTransformerAR2`` over flat per-voxel tokens (canon16_flat).
* ``cond_piece_ar`` — ``CondVoxelAR2`` over piece tokens, conditioned on a CLIP text
  embedding. With no prompt it samples its learned *null* condition, which is the
  same unconditional branch classifier-free guidance uses, so one checkpoint serves
  both ``/gen`` and ``/gen <text>``.

**A checkpoint is not self-describing.** Nothing in a ``model.pt`` records its
vocabulary, and a piece token id is meaningless without the patterns it expands to,
so every entry must name a saved vocab file. ``load()`` cross-checks the checkpoint's
``lm_head`` against the vocabulary size and refuses to run on a mismatch — that check
is the difference between a clear startup error and a model that silently generates
garbage from a vocabulary it was never trained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch

from blockgen_server.decode import Block, PieceDecoder, VoxelDecoder
from blockgen_server.sampling import stream_cond_tokens, stream_tokens

BOS_TOKEN, EOS_TOKEN = 1, 2


@dataclass
class GenerateRequest:
    prompt: Optional[str] = None
    seed: Optional[int] = None
    temperature: float = 1.0
    top_k: Optional[int] = 40
    cfg_scale: float = 3.0
    max_tokens: Optional[int] = None


@dataclass
class ModelSpec:
    """One entry from models.json."""
    name: str
    kind: str
    checkpoint: str
    description: str = ""
    piece_vocab: Optional[str] = None
    block_vocab: Optional[str] = None
    max_seq_len: int = 5480
    pe: str = "phase4"
    d_model: int = 256
    nhead: int = 8
    layers: int = 6
    dim_feedforward: Optional[int] = None
    cond_dim: int = 512
    n_prefix: int = 1
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ff(self) -> int:
        return self.dim_feedforward or 4 * self.d_model


class Backend:
    """Base class. Subclasses implement ``load()`` and ``stream()``."""

    supports_text = False

    def __init__(self, spec: ModelSpec, repo_root: Path) -> None:
        self.spec = spec
        self.root = repo_root
        self.model = None

    # --- helpers ---------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _check_vocab(self, state_dict: dict, vocab_size: int) -> None:
        lm = state_dict["lm_head.weight"].shape[0]
        if lm != vocab_size:
            raise ValueError(
                f"{self.spec.name}: checkpoint lm_head={lm} but vocabulary implies "
                f"{vocab_size}. The checkpoint and the vocab file do not belong "
                f"together; generating anyway would produce garbage.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        return {"name": self.spec.name, "kind": self.spec.kind,
                "description": self.spec.description, "notes": self.spec.notes,
                "supports_text": self.supports_text, "loaded": self.is_loaded()}

    def load(self) -> None:
        raise NotImplementedError

    def stream(self, req: GenerateRequest) -> Iterator[List[Block]]:
        """Yield lists of blocks as tokens decode. Empty lists are not yielded."""
        raise NotImplementedError

    # --- shared plumbing --------------------------------------------------
    def _generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        return torch.Generator(device=self._device()).manual_seed(int(seed))

    def _load_piece_vocab(self):
        if not self.spec.piece_vocab:
            raise ValueError(f"{self.spec.name}: kind={self.spec.kind} requires "
                             f"'piece_vocab' (piece ids are meaningless without it)")
        from blockgen.export.minecraftace import load_piece_vocab
        p = self.path(self.spec.piece_vocab)
        if not p.exists():
            raise FileNotFoundError(
                f"{self.spec.name}: piece vocab not found at {p}. If this run never "
                f"saved one, rebuild it with scripts/rebuild_native_vocab.py.")
        return load_piece_vocab(str(p))

    def _load_block_vocab(self):
        from blockgen.utils.serialize import load_block_vocab
        if not self.spec.block_vocab:
            raise ValueError(f"{self.spec.name}: kind={self.spec.kind} requires "
                             f"'block_vocab'")
        p = self.path(self.spec.block_vocab)
        if not p.exists():
            raise FileNotFoundError(
                f"{self.spec.name}: block vocab not found at {p}. If this run never "
                f"saved one, rebuild it with scripts/rebuild_native_vocab.py.")
        return load_block_vocab(str(p))

    def _build_ar2(self, vocab_size: int):
        from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
        return VoxelTransformerAR2(
            vocab_size=vocab_size, max_seq_len=self.spec.max_seq_len,
            d_model=self.spec.d_model, nhead=self.spec.nhead,
            num_layers=self.spec.layers, dim_feedforward=self.spec.ff,
            pe=self.spec.pe)


class PieceARBackend(Backend):
    """Unconditional AR over 3D-BPE piece tokens."""

    def load(self) -> None:
        self.cv = self._load_piece_vocab()
        sd = torch.load(self.path(self.spec.checkpoint),
                        map_location=self._device(), weights_only=True)
        self._check_vocab(sd, self.cv.vocab_size)
        m = self._build_ar2(self.cv.vocab_size)
        m.load_state_dict(sd)
        self.model = m.to(self._device()).eval()

    def stream(self, req: GenerateRequest) -> Iterator[List[Block]]:
        dec = PieceDecoder(self.cv)
        max_new = req.max_tokens or (self.spec.max_seq_len - 1)
        for tok in stream_tokens(self.model, bos_token_id=BOS_TOKEN,
                                 eos_token_id=EOS_TOKEN, max_new_tokens=max_new,
                                 temperature=req.temperature, top_k=req.top_k,
                                 generator=self._generator(req.seed)):
            blocks = dec.feed(tok)
            if blocks:
                yield blocks


class VoxelARBackend(Backend):
    """Unconditional AR over flat per-voxel tokens."""

    def load(self) -> None:
        self.vocab = self._load_block_vocab()
        sd = torch.load(self.path(self.spec.checkpoint),
                        map_location=self._device(), weights_only=True)
        self._check_vocab(sd, self.vocab.vocab_size)
        m = self._build_ar2(self.vocab.vocab_size)
        m.load_state_dict(sd)
        self.model = m.to(self._device()).eval()

    def stream(self, req: GenerateRequest) -> Iterator[List[Block]]:
        dec = VoxelDecoder(self.vocab)
        max_new = req.max_tokens or (self.spec.max_seq_len - 1)
        for tok in stream_tokens(self.model, bos_token_id=BOS_TOKEN,
                                 eos_token_id=EOS_TOKEN, max_new_tokens=max_new,
                                 temperature=req.temperature, top_k=req.top_k,
                                 generator=self._generator(req.seed)):
            blocks = dec.feed(tok)
            if blocks:
                yield blocks


class CondPieceARBackend(Backend):
    """CLIP-text-conditioned AR over piece tokens; no prompt -> null (uncond) branch."""

    supports_text = True
    _clip = None  # lazily loaded, shared across instances

    def load(self) -> None:
        from blockgen.models.voxel_transformer_cond import CondVoxelAR2
        self.cv = self._load_piece_vocab()
        sd = torch.load(self.path(self.spec.checkpoint),
                        map_location=self._device(), weights_only=True)
        self._check_vocab(sd, self.cv.vocab_size)
        m = CondVoxelAR2(cond_dim=self.spec.cond_dim, n_prefix=self.spec.n_prefix,
                         vocab_size=self.cv.vocab_size,
                         max_seq_len=self.spec.max_seq_len, d_model=self.spec.d_model,
                         nhead=self.spec.nhead, num_layers=self.spec.layers,
                         dim_feedforward=self.spec.ff, pe=self.spec.pe)
        m.load_state_dict(sd)
        self.model = m.to(self._device()).eval()

    @torch.no_grad()
    def embed_text(self, prompt: str) -> torch.Tensor:
        """CLIP ViT-B/32 pooled text features -- the exact encoder training used.

        Must stay in step with ``blockgen.labeling.embed_conditions.embed_texts``;
        a different CLIP variant or a normalized embedding would land the prefix
        somewhere the projection was never trained on.
        """
        if CondPieceARBackend._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            name = self.spec.extra.get("clip", "openai/clip-vit-base-patch32")
            proc = CLIPProcessor.from_pretrained(name)
            clip = CLIPModel.from_pretrained(name).to(self._device()).eval()
            CondPieceARBackend._clip = (proc, clip)
        proc, clip = CondPieceARBackend._clip
        inputs = proc(text=[prompt], return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(self._device())
        emb = clip.get_text_features(**inputs)          # (1, cond_dim)
        return emb.float()[:, None]                      # (1, 1, cond_dim)

    def stream(self, req: GenerateRequest) -> Iterator[List[Block]]:
        cond = self.embed_text(req.prompt) if req.prompt else None
        dec = PieceDecoder(self.cv)
        max_new = req.max_tokens or (self.spec.max_seq_len - self.spec.n_prefix - 1)
        for tok in stream_cond_tokens(self.model, cond=cond, bos_token_id=BOS_TOKEN,
                                      eos_token_id=EOS_TOKEN, max_new_tokens=max_new,
                                      temperature=req.temperature, top_k=req.top_k,
                                      cfg_scale=req.cfg_scale,
                                      generator=self._generator(req.seed)):
            blocks = dec.feed(tok)
            if blocks:
                yield blocks


KINDS = {
    "piece_ar": PieceARBackend,
    "voxel_ar": VoxelARBackend,
    "cond_piece_ar": CondPieceARBackend,
}


def build_backend(spec: ModelSpec, repo_root: Path) -> Backend:
    if spec.kind not in KINDS:
        raise ValueError(f"{spec.name}: unknown kind {spec.kind!r} "
                         f"(have {sorted(KINDS)})")
    return KINDS[spec.kind](spec, repo_root)
