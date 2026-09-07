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
* ``agentic``       — no checkpoint at all: a frontier LLM writes a build *program*
  (``blockgen.agentic``) which is then executed command by command. It is a *group*:
  one registry entry serves every model the vendor offers, addressed as
  ``agentic:<model>`` (see ``registry.py``).

**The agentic backend streams differently, and deliberately.** The neural backends
stream because sampling is incremental — blocks appear as tokens arrive. The agentic
backend has the whole program before it places anything, and replays it *one command
at a time* instead of dumping the result: the foundation appears, then the walls,
then the openings are cut, then the roof. That is the build order the model actually
wrote, and watching it is the demo. Each batch carries the command that produced it
so the mod can name the step.

**A checkpoint is not self-describing.** Nothing in a ``model.pt`` records its
vocabulary, and a piece token id is meaningless without the patterns it expands to,
so every entry must name a saved vocab file. ``load()`` cross-checks the checkpoint's
``lm_head`` against the vocabulary size and refuses to run on a mismatch — that check
is the difference between a clear startup error and a model that silently generates
garbage from a vocabulary it was never trained on.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch

from blockgen_server.decode import Block, PieceDecoder, VoxelDecoder
from blockgen_server.sampling import stream_cond_tokens, stream_tokens

BOS_TOKEN, EOS_TOKEN = 1, 2


@dataclass
class BlockBatch:
    """A batch of placements plus optional metadata for the client.

    ``stream()`` may yield a bare ``List[Block]`` (what every neural backend does) or
    one of these. ``step`` names the command that produced the batch; ``stats`` is
    end-of-build accounting (tokens, dollars) that the server attaches to ``done``.
    Both ride on existing message types as extra JSON fields, so a mod built before
    this change still works — it just ignores them.
    """

    blocks: List[Block] = field(default_factory=list)
    step: Optional[dict] = None
    stats: Optional[dict] = None


def as_batch(item) -> BlockBatch:
    """Normalize whatever a backend yielded into a :class:`BlockBatch`."""
    return item if isinstance(item, BlockBatch) else BlockBatch(blocks=list(item))


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
    # Empty for kinds that have no weights (``agentic``); the registry's
    # missing-file check skips empty paths.
    checkpoint: str = ""
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

    def unavailable_reason(self) -> Optional[str]:
        """Why this backend cannot serve right now, beyond missing files.

        The registry shows unusable entries *with a reason* rather than hiding them
        — a model that needs an API key you have not set should say so.
        """
        return None

    def group_members(self) -> List[str]:
        """Sub-models this entry can serve as ``<name>:<member>`` (empty = not a group)."""
        return []

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


class AgenticBackend(Backend):
    """Track E: an LLM writes a build program, which is executed command by command.

    No weights, no vocabulary — the "model" is a provider string
    (``openai:gpt-5-mini``, ``gemini:gemini-3.5-flash``). One registry entry serves
    every model in ``extra["models"]``; the registry derives ``agentic:<model>``
    on demand, so adding a model is a list entry, not a new backend.

    Two things differ from the neural backends and both are on purpose:

    * **Text is required, not optional.** This is the only backend with real text
      conditioning, so a bare ``/gen`` picks a prompt from the built-in short set
      (reported back in ``begin``) rather than erroring — an empty demo is worse
      than an arbitrary house.
    * **The stream is the program, replayed.** Generation finishes before the first
      block is placed, so batching by command is what makes the build legible in
      world: one batch per command, in the order the model wrote them.
    """

    supports_text = True

    # A short, curated allowlist. These are the models worth pointing at for this
    # task, not everything the vendors sell; `/model agentic list` shows exactly
    # this, and an unlisted model still works when `strict_models` is off.
    DEFAULT_MODEL = "gpt-5-mini"

    # Vendor inferred from the model name, so `/model agentic gemini-3.5-flash` is
    # what you type instead of `agentic:gemini:gemini-3.5-flash`. An explicit
    # "vendor:model" always wins, which is the escape hatch for a name that does not
    # match any prefix.
    _VENDOR_PREFIXES = (
        ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
        ("gemini", "gemini"), ("claude", "anthropic"), ("mock", "mock"),
    )

    def __init__(self, spec: ModelSpec, repo_root: Path) -> None:
        super().__init__(spec, repo_root)
        e = spec.extra
        self.model_name: str = e.get("provider") or self.DEFAULT_MODEL
        self.provider_spec: str = self.qualify(self.model_name)
        self.models: List[str] = list(e.get("models") or [self.model_name])
        self._provider = None

    @classmethod
    def qualify(cls, model: str) -> str:
        """``"gemini-3.5-flash"`` -> ``"gemini:gemini-3.5-flash"``; pass through
        anything already vendor-qualified."""
        model = model.strip()
        if ":" in model:
            return model
        low = model.lower()
        for prefix, vendor in cls._VENDOR_PREFIXES:
            if low.startswith(prefix):
                return f"{vendor}:{model}" if vendor != "mock" else "mock"
        raise ValueError(
            f"cannot tell which vendor serves {model!r}; qualify it as "
            f"'<vendor>:{model}' (openai, gemini, anthropic)")

    # --- config ----------------------------------------------------------
    def agent_config(self):
        """Build the :class:`AgentConfig` this entry pins (all loop knobs)."""
        from blockgen.agentic.agent import AgentConfig
        e = self.spec.extra
        canvas = int(e.get("canvas", 48))
        height = int(e.get("canvas_height", canvas))
        return AgentConfig(
            provider=self.provider_spec,
            canvas_size=(canvas, height, canvas),
            plan=bool(e.get("plan", False)),
            n_examples=int(e.get("examples", 1)),
            repair_rounds=int(e.get("repair_rounds", 1)),
            critique_rounds=int(e.get("critique_rounds", 0)),
            critique_mode=e.get("critique_mode", "rewrite"),
            target_blocks=e.get("target_blocks"),
            max_tokens=int(e.get("max_tokens", 16000)),
            reasoning_effort=e.get("reasoning_effort", "low"),
            # Block ontology (blockgen/ontology): "none" keeps the bare palette
            # list this demo shipped with, so an entry that does not mention it
            # behaves exactly as before. Set "mined" on an entry to serve the
            # measured catalog; the server needs data/ontology/ built for that.
            ontology=str(e.get("ontology", "none")),
            ontology_path=e.get("ontology_path"),
            cache=bool(e.get("cache", True)),
            verbose=False,
        )

    def group_members(self) -> List[str]:
        return list(self.models)

    def unavailable_reason(self) -> Optional[str]:
        """No API key for this vendor is the one failure worth reporting up front —
        it is the difference between "why did /gen do nothing" and a clear answer."""
        import os
        from blockgen.agentic.providers import load_env
        load_env()
        vendor = self.provider_spec.split(":", 1)[0].lower()
        needed = {"openai": ["OPENAI_API_KEY"],
                  "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                  "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                  "anthropic": ["ANTHROPIC_API_KEY"],
                  "claude": ["ANTHROPIC_API_KEY"]}.get(vendor, [])
        if needed and not any(os.environ.get(k) for k in needed):
            return f"no API key: set {' or '.join(needed)} in the repo .env"
        return None

    def info(self) -> dict:
        out = super().info()
        cfg = self.agent_config()
        out.update({"provider": self.provider_spec, "group": self.spec.name,
                    "members": self.group_members(),
                    "canvas": list(cfg.canvas_size),
                    "loop": {"plan": cfg.plan, "examples": cfg.n_examples,
                             "repair_rounds": cfg.repair_rounds,
                             "critique_rounds": cfg.critique_rounds}})
        return out

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        from blockgen.agentic.providers import get_provider
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.spec.name}: {reason}")
        cfg = self.agent_config()
        self._provider = get_provider(cfg.provider, cache=cfg.cache,
                                      **cfg.provider_params())
        self.model = self._provider          # marks the backend as loaded

    # --- generation ------------------------------------------------------
    def _prompt_for(self, req: GenerateRequest) -> str:
        if req.prompt:
            return req.prompt
        from blockgen.agentic.tasks import SHORT_PROMPTS
        import random
        return random.Random(req.seed).choice(SHORT_PROMPTS)

    def stream(self, req: GenerateRequest) -> Iterator[BlockBatch]:
        from blockgen.agentic.agent import BuildAgent
        from blockgen.agentic.dsl import ProgramRunner, parse_program
        from blockgen_server.blockmap import modern_state

        cfg = self.agent_config()
        prompt = self._prompt_for(req)
        agent = BuildAgent(self._provider, cfg)
        result = agent.build(prompt, seed=req.seed)

        if not result.program_text.strip():
            raise RuntimeError(f"{self.spec.name}: the model returned no program "
                               f"(prompt: {prompt!r})")

        # Replay the final program so the build appears in the order it was written.
        # Re-execution costs nothing (no API calls) and reproduces the agent's canvas
        # exactly -- the loop only ever hands forward a program, never a canvas diff.
        runner = ProgramRunner(parse_program(result.program_text),
                               size=cfg.canvas_size, track_voxels=True)
        for step in runner.steps():
            blocks = [Block(x, y, z, modern_state(bid, bdata, oriented=True))
                      for x, y, z, bid, bdata in step.voxels]
            if not blocks and step.issue is None:
                continue
            yield BlockBatch(blocks=blocks, step={
                "index": step.index + 1, "total": step.total,
                "command": step.call.text, "changed": step.changed,
                "error": step.issue.message if step.failed else None})

        prompt_tokens, completion_tokens = result.tokens
        yield BlockBatch(blocks=[], stats={
            "provider": self.provider_spec, "prompt": prompt,
            "cost_usd": round(result.cost_usd, 6),
            "cost_known": result.cost_known,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "commands": result.report.n_commands,
            "failed_commands": result.report.n_failed,
            "rounds": [r.stage for r in result.rounds],
            "cached": result.cached,
            "llm_seconds": round(result.elapsed_s, 1)})


KINDS = {
    "piece_ar": PieceARBackend,
    "voxel_ar": VoxelARBackend,
    "cond_piece_ar": CondPieceARBackend,
    "agentic": AgenticBackend,
}


def build_backend(spec: ModelSpec, repo_root: Path) -> Backend:
    if spec.kind not in KINDS:
        raise ValueError(f"{spec.name}: unknown kind {spec.kind!r} "
                         f"(have {sorted(KINDS)})")
    return KINDS[spec.kind](spec, repo_root)
