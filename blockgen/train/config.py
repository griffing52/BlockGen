"""Flat, typed training config. YAML (with ``extends``) plus ``key=value`` overrides."""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PRESETS: Dict[str, Dict[str, int]] = {
    # name: (d_model, nhead, layers, ff)  -- parameter counts in docs/pipeline.md §5
    "5m": dict(d_model=256, nhead=8, layers=6, ff=1024),
    "20m": dict(d_model=512, nhead=8, layers=8, ff=2048),
    "60m": dict(d_model=768, nhead=12, layers=10, ff=3072),
    "150m": dict(d_model=1024, nhead=16, layers=12, ff=4096),
    "300m": dict(d_model=1280, nhead=20, layers=16, ff=5120),
}


@dataclass
class TrainConfig:
    name: str = "run"
    stage: str = "pretrain"               # pretrain | anneal | sft | posttrain: a label for the registry
    preset: Optional[str] = None          # one of PRESETS; sets d_model/nhead/layers/ff unless given
    # model
    arch: str = "ar2"                     # blockgen.train.archs; the registered Arch builds, scores and samples the model
    model: Dict[str, Any] = field(default_factory=dict)   # extra constructor kwargs for the arch (archs.py docstring)
    d_model: int = 256
    nhead: int = 8
    layers: int = 6
    ff: int = 1024
    dropout: float = 0.0
    pe: str = "phase4"
    max_seq_len: int = 8200
    # data: "tier" or "tier:weight"; weights are relative sampling probabilities
    train: List[str] = field(default_factory=lambda: ["sft-houses32"])
    val: List[str] = field(default_factory=lambda: ["eval-holdout"])
    tokenizer: Optional[str] = None       # shard tokenizer hash; default = newest shard of the first train tier
    kind: str = "voxels"                  # which shards and sampler: voxels | facades | pieces (blockgen.corpus.tokenizer)
    augment: str = ""                     # training shard variant: "" | d4 (blockgen.corpus.shards); never the val tiers
    train_split: Optional[str] = None     # provenance the bench ranks by; None = "canonical" iff train == [sft-houses32], else the tiers
    trained_on_signals: List[str] = field(default_factory=list)   # rewards or preference signals a post-training stage optimised
    batch_tokens: int = 32768             # padded tokens per micro-batch
    grad_accum: int = 1
    seed: int = 0
    # optimisation (WSD)
    steps: int = 2000
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 100
    decay_frac: float = 0.2               # last fraction of steps is the decay phase
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0
    amp: bool = True
    compile: bool = False
    # bookkeeping
    log_every: int = 20
    eval_every: int = 500
    eval_seqs: int = 256                  # holdout sequences per periodic validation pass
    final_eval_all: bool = True           # the last pass scores every holdout sequence that fits the context
    ckpt_every: int = 1000
    sample_every: int = 0                 # 0 = only at the end
    n_samples: int = 16
    sample_tags: Dict[str, str] = field(default_factory=lambda: {"src": "curated", "q": "4", "size": "32", "win": "whole"})
    sample_max_blocks: int = 2048
    sample_batch: int = 64                # builds per sampling batch; lower it when the GPU is shared (facades cap at 32)
    temperature: float = 1.0
    top_k: int = 40
    bench: bool = True
    bench_tier: str = "fast"              # fast | full | both (blockgen.eval.bench)
    bench_n: int = 128                    # base.yaml pins 64 for every recorded run; 128 is what a bare TrainConfig gets
    bench_corpus: str = "houses_32"
    select: str = "best"                  # best | last | reward: which checkpoint is sampled, benchmarked and registered
    init_from: Optional[str] = None       # checkpoint path: weights only (a new stage)
    allow_dropout_change: bool = False    # init_from a run trained at a different dropout (T59: wrecks the weights)
    allow_partial_init: bool = False      # init_from a checkpoint whose keys do not all match the model (counts are recorded)
    resume: Optional[str] = None          # run dir: full state (continue the same run)
    out: str = "outputs"
    device: str = "cuda"
    note: str = ""
    config_path: Optional[str] = None     # the YAML this run was loaded from, for the registry

    def __post_init__(self):
        if self.model is None:
            self.model = {}
        if not isinstance(self.model, dict):
            raise TypeError(f"model must be a mapping of constructor kwargs, got {type(self.model).__name__}")
        if self.trained_on_signals is None:
            self.trained_on_signals = []

    def apply_preset(self) -> "TrainConfig":
        if self.preset:
            for k, v in PRESETS[self.preset].items():
                setattr(self, k, v)
        return self

    def train_specs(self) -> List[tuple]:
        out = []
        for s in self.train:
            name, _, w = str(s).partition(":")
            out.append((name, float(w) if w else 1.0))
        return out

    def split_name(self) -> str:
        """What the bench is told the model trained on: the explicit ``train_split``,
        else the inference every run before the field was added recorded."""
        if self.train_split:
            return self.train_split
        return "canonical" if list(self.train) == ["sft-houses32"] else ",".join(self.train)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(field_type: Any, raw: str) -> Any:
    """Parse a ``key=value`` override by the field's annotation (resolved, not the string form)."""
    if raw.lower() in ("none", "null"):
        return None
    if field_type in (int, Optional[int]):
        return int(raw)
    if field_type in (float, Optional[float]):
        return float(raw)
    if field_type is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if typing.get_origin(field_type) is dict:          # model={"geo_bias": true}: JSON or nothing
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"a mapping field takes a JSON object, got {raw!r} ({e.msg})") from None
        if not isinstance(value, dict):
            raise SystemExit(f"a mapping field takes a JSON object, got {raw!r}")
        return value
    if raw.startswith("[") or raw.startswith("{"):
        return json.loads(raw)
    if field_type in (List[str],):
        return [x for x in raw.split(",") if x]
    return raw


def field_types() -> Dict[str, Any]:
    return typing.get_type_hints(TrainConfig)


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """``key=value`` strings to typed values; an unknown key is a usage error (SystemExit)."""
    types = field_types()
    out: Dict[str, Any] = {}
    for ov in overrides:
        k, _, v = ov.partition("=")
        if k not in types:
            raise SystemExit(f"unknown config key {k!r}; known: {sorted(types)}")
        out[k] = _coerce(types[k], v)
    return out


def apply_overrides(cfg: TrainConfig, overrides: List[str]) -> TrainConfig:
    """Set ``key=value`` overrides on an existing config with the same coercion ``load`` uses
    (``bench=false`` is False, not ``bool("false")``)."""
    for k, v in parse_overrides(overrides).items():
        setattr(cfg, k, v)
    return cfg


def load(config_path: Optional[str], overrides: List[str]) -> TrainConfig:
    values: Dict[str, Any] = {}
    types = field_types()
    if config_path:
        from blockgen.config import load_config
        values.update(load_config(config_path))
        unknown = sorted(k for k in values if k not in types)
        if unknown:
            raise ValueError(f"{config_path}: unknown config key(s) {unknown}; known: {sorted(types)}")
    values.update(parse_overrides(overrides))
    cfg = TrainConfig(**values).apply_preset()
    if config_path and cfg.config_path is None:
        cfg.config_path = str(config_path)
    return cfg
