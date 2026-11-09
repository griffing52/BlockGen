"""The architecture seam of the shared trainer: one ``Arch`` per model family.

The step loop in ``trainer.py`` knows nothing about a model beyond this record.
``build`` constructs it from the config and the shard tokenizer, ``loss`` turns a
collated batch into a scalar, ``sample`` turns a trained model into structures
(``sample(model, tokenizer, cfg, n, seed=None) -> (structures, extra)``; the
trainer's ``sample_and_render`` calls it, so a family with its own decoding, a
diffusion or a graph model, is sampled by its own code). ``ar2`` is the only entry
registered here; a new family registers its own ``Arch`` (``register``) and is
then selected by ``arch: <name>`` in a YAML.

The model contract
------------------
Every arch's ``build`` returns an ``nn.Module`` the loop can drive:

* ``model(input_ids, pad_mask=None) -> logits`` with ``input_ids`` of shape
  ``[B, L]`` (right-padded with ``PAD_TOKEN``), ``pad_mask`` a bool ``[B, L]``
  that is True where padded, and ``logits`` of shape ``[B, L, vocab_size]``.
  Training and validation only ever call this form.
* ``model.max_seq_len``: the longest sequence it accepts. The batchers drop
  longer sequences and the sampler caps its token budget by it.
* Optional, for fast sampling: ``model.forward(step, past=past, return_cache=True)
  -> (logits, past)`` where ``past`` is whatever the model wants to carry between
  steps (``None`` on the first call, which then receives the whole prompt) and
  ``step`` is the single new token on later calls. A model that does not accept
  these keywords, or whose ``supports_cache()`` returns False, is sampled by a
  full re-forward per step instead (``sample.generate``); slower, same tokens.

The loss contract: ``loss(model, batch)`` receives ``(input_ids, targets, pad_mask)``
already on the device and returns the mean loss per non-pad target token, so
``evaluate`` can turn it back into a token-weighted sum across batches.

``cfg.model`` (a dict) reaches the constructor as keyword arguments, so an arch
exposes its own knobs without adding fields to ``TrainConfig``. For ``ar2`` these
are the extra ``VoxelTransformerAR2`` arguments (``geo_bias``, ``piece_factors``,
``semantic_embedding``, ...); a key the constructor does not take, or one that
duplicates a config field, raises before any weights are allocated.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.utils.data import Structure
from blockgen.utils.serialize import PAD_TOKEN

from .config import TrainConfig


@dataclass
class Arch:
    name: str
    build: Callable[[TrainConfig, CorpusTokenizer], nn.Module]
    loss: Callable[[nn.Module, tuple], torch.Tensor]
    sample: Callable[..., Tuple[List[Structure], dict]]
    track: str = "ar"             # which benchmark track its samples belong to


ARCHS: Dict[str, Arch] = {}


def register(arch: Arch) -> Arch:
    """Add an arch under its name. A second, different ``Arch`` under a taken name is
    refused so an import cannot silently swap out what a config means by ``arch:``."""
    have = ARCHS.get(arch.name)
    if have is not None and have is not arch:
        raise ValueError(f"arch {arch.name!r} is already registered; pop it from ARCHS first to replace it")
    ARCHS[arch.name] = arch
    return arch


def get(name: str) -> Arch:
    try:
        return ARCHS[name]
    except KeyError:
        raise KeyError(f"unknown arch {name!r}; known: {sorted(ARCHS)}") from None


def constructor_kwargs(ctor: Callable, fixed: Dict[str, Any], extra: Dict[str, Any], arch: str) -> Dict[str, Any]:
    """Merge the config-derived arguments with ``cfg.model``, refusing anything the
    constructor cannot take. ``**kwargs`` constructors are trusted as they are."""
    params = inspect.signature(ctor).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    clash = sorted(set(extra) & set(fixed))
    if clash:
        raise ValueError(f"arch {arch!r}: model.{clash[0]} duplicates a config field; set it at the top level")
    unknown = sorted(k for k in extra if k not in params) if not accepts_any else []
    if unknown:
        allowed = sorted(k for k in params if k not in fixed and k != "self")
        raise ValueError(f"arch {arch!r}: unknown model option(s) {unknown}; the constructor takes {allowed}")
    return {**fixed, **extra}


# --- ar2: the causal voxel transformer ------------------------------------------

def build_ar2(cfg: TrainConfig, tokenizer: CorpusTokenizer) -> nn.Module:
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    fixed = dict(vocab_size=tokenizer.vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
                 nhead=cfg.nhead, num_layers=cfg.layers, dim_feedforward=cfg.ff, dropout=cfg.dropout, pe=cfg.pe)
    return VoxelTransformerAR2(**constructor_kwargs(VoxelTransformerAR2, fixed, dict(cfg.model or {}), "ar2"))


def loss_ar2(model: nn.Module, batch: tuple) -> torch.Tensor:
    """Next-token cross-entropy; PAD targets are ignored, so the mean is per real token."""
    inp, tgt, pad = batch
    logits = model(inp, pad_mask=pad)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD_TOKEN)


def sample_ar2(model: nn.Module, tokenizer: CorpusTokenizer, cfg: TrainConfig, n: int, seed=None):
    """The next-token loop, dispatched on the tokenizer's kind (``sample.sample_structures``)."""
    from .sample import sample_structures
    return sample_structures(model, tokenizer, cfg, n, seed=seed)


register(Arch(name="ar2", build=build_ar2, loss=loss_ar2, sample=sample_ar2, track="ar"))
