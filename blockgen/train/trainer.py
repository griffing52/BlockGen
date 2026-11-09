"""The step loop. See the package docstring and docs/pipeline.md §5.

The loop is architecture-blind: ``archs.get(cfg.arch)`` supplies how the model
is built, how a batch becomes a loss and how the trained model is sampled
(``archs.py`` documents the model contract). Checkpoint selection is a second
seam: ``select=best`` and ``select=last`` are built in, any other name resolves
through ``SELECT_FNS`` (post-training registers ``reward``).
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from blockgen.corpus.shards import Shard, find
from blockgen.corpus.tiers import TIER_DIR
from blockgen.utils.runs import new_run_dir
from blockgen.utils.serialize import BOS_TOKEN, PAD_TOKEN

from . import archs, registry
from .config import TrainConfig
from .data import MixedBatches, eval_batches, open_shards
from .sample import bench, sample_and_render

COMPILED_PREFIX = "_orig_mod."

# select=<name> hooks beyond best/last. A hook has the signature of ``select_checkpoint``
# (plus ``tokenizer``), loads the weights it chooses into ``model`` and returns
# (val_nll of those weights, their step).
SELECT_FNS: Dict[str, Callable] = {}


def register_select(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        SELECT_FNS[name] = fn
        return fn
    return deco


def resolve_select(name: str) -> Optional[Callable]:
    """None for the built-in rules; the registered hook otherwise. Unknown names fail
    before any training happens."""
    if name in ("best", "last"):
        return None
    if name not in SELECT_FNS:
        # Post-training registers its hooks on import; the package and the module that
        # holds the hook are both tried, so a bare package __init__ does not hide it.
        for mod in ("blockgen.posttrain", "blockgen.posttrain.select"):
            try:
                importlib.import_module(mod)
            except ImportError:
                pass
    if name not in SELECT_FNS:
        raise ValueError(f"unknown select {name!r}; known: {['best', 'last'] + sorted(SELECT_FNS)}")
    return SELECT_FNS[name]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def unwrap(model):
    """The module under ``torch.compile``, or the model itself."""
    return getattr(model, "_orig_mod", model)


def strip_compiled(sd: dict) -> dict:
    """A state dict saved from a compiled model carries ``_orig_mod.``; the plain model does not."""
    if any(k.startswith(COMPILED_PREFIX) for k in sd):
        return {(k[len(COMPILED_PREFIX):] if k.startswith(COMPILED_PREFIX) else k): v for k, v in sd.items()}
    return sd


def load_weights(model, sd: dict, strict: bool = True):
    """``load_state_dict`` that accepts compiled or plain checkpoints into either kind of model."""
    return unwrap(model).load_state_dict(strip_compiled(sd), strict=strict)


class _Vocab:
    """Stand-in for a tokenizer when only the vocabulary size is known (``build_model(cfg, 507)``)."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size


def build_model(cfg: TrainConfig, tokenizer):
    """The arch's ``build``; ``tokenizer`` may be a bare vocabulary size."""
    if isinstance(tokenizer, int):
        tokenizer = _Vocab(tokenizer)
    return archs.get(cfg.arch).build(cfg, tokenizer)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Warmup, stable, decay (WSD). The decay is linear to ``min_lr_frac * lr``."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    decay_start = int(cfg.steps * (1.0 - cfg.decay_frac))
    if step < decay_start or cfg.steps <= decay_start:
        return cfg.lr
    frac = (step - decay_start) / max(1, cfg.steps - decay_start)
    return cfg.lr * (1.0 - (1.0 - cfg.min_lr_frac) * frac)


def make_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "embedding" not in n else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=torch.cuda.is_available())


def tier_hashes(names: List[str]) -> Dict[str, str]:
    out = {}
    for n in names:
        p = os.path.join(TIER_DIR, f"{n}.json")
        out[n] = json.load(open(p))["sha256"] if os.path.exists(p) else ""
    return out


def holdout_sizes(val: Dict[str, Shard], n_seqs: Optional[int], max_seq_len: int) -> Dict[str, int]:
    """How many sequences a validation pass scores per tier: those that fit the context,
    capped at ``n_seqs`` (None = all of them)."""
    out = {}
    for name, shard in val.items():
        fits = int((shard.lengths() <= max_seq_len).sum())
        out[name] = fits if n_seqs is None else min(fits, n_seqs)
    return out


@torch.no_grad()
def evaluate(model, val: Dict[str, Shard], cfg: TrainConfig, device, use_amp: bool,
             n_seqs: Optional[int] = -1) -> Dict[str, float]:
    """Mean NLL per target token on the first ``n_seqs`` holdout sequences of every val
    tier (default ``cfg.eval_seqs``; None = every sequence that fits the context)."""
    loss_fn = archs.get(cfg.arch).loss
    if n_seqs == -1:
        n_seqs = cfg.eval_seqs
    model.eval()
    out = {}
    for name, shard in val.items():
        tot = torch.zeros((), device=device); n = torch.zeros((), device=device)
        for inp, tgt, pad in eval_batches(shard, n_seqs if n_seqs is not None else len(shard), cfg.batch_tokens, cfg.max_seq_len):
            inp, tgt, pad = inp.to(device), tgt.to(device), pad.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model, (inp, tgt, pad))
            k = (tgt != PAD_TOKEN).sum()
            tot += loss.float() * k; n += k
        out[name] = float(tot / n.clamp(min=1))
    model.train()
    return out


def check_init_compat(ck: dict, cfg: TrainConfig) -> None:
    """Refuse to fine-tune at a dropout the checkpoint was not trained with.

    T59: a 5M model pretrained at dropout 0 and fine-tuned at 0.1 scored 5.7 nats on
    its first SFT batch (uniform is 6.2) against 0.6 in eval mode; two of three
    seeds lost the pretraining before the first update. Override with
    ``allow_dropout_change=true`` when that is intended.
    """
    prev = (ck.get("config") or {}).get("dropout")
    if prev is not None and abs(prev - cfg.dropout) > 1e-9 and not cfg.allow_dropout_change:
        raise ValueError(f"init_from checkpoint was trained at dropout {prev}, this run asks for {cfg.dropout}; "
                         f"a model trained without dropout collapses under it (T59). Set dropout={prev} "
                         f"or allow_dropout_change=true")


def extend_vocab(model, sd: dict, old_vocab: int, init_row: int = BOS_TOKEN) -> Tuple[dict, List[str]]:
    """Grow every vocabulary-sized tensor of ``sd`` from ``old_vocab`` rows to the model's.

    For a tokenizer that appends ids to the one the checkpoint used (the category
    group, ``CorpusTokenizer.extends``): each tensor whose first dimension is
    ``old_vocab`` in the checkpoint and larger in the model (token embedding,
    ``lm_head`` weight and bias) keeps its old rows and gets the new ones as copies
    of row ``init_row``. With BOS as that row, a class token placed where BOS was
    starts out as BOS, so the extended model computes exactly what the checkpoint
    did on every old sequence, and the new tokens are as unlikely as a target as
    BOS is (never). Returns the new state dict and the names it grew.
    """
    want = unwrap(model).state_dict()
    out, grown = dict(sd), []
    for k, v in sd.items():
        w = want.get(k)
        if w is None or not torch.is_tensor(v) or v.ndim == 0 or v.shape[0] != old_vocab:
            continue
        if w.shape[0] > old_vocab and tuple(w.shape[1:]) == tuple(v.shape[1:]):
            extra = v[init_row:init_row + 1].expand(w.shape[0] - old_vocab, *v.shape[1:])
            out[k] = torch.cat([v, extra.to(v.dtype)], 0)
            grown.append(k)
    return out, grown


def init_from_checkpoint(model, cfg: TrainConfig, tok_hash: Optional[str], device, tokenizer=None) -> dict:
    """Load ``cfg.init_from`` as weights only and say what did not line up.

    A checkpoint whose keys do not match the model (a renamed module, a different
    preset or arch) is refused: a stage that trains a half-random model is a failed
    run, not a warning. ``allow_partial_init=true`` accepts it, in which case keys
    the checkpoint lacks keep their fresh initialisation, keys the model lacks are
    dropped, and both are counted on stdout, in ``meta["init_from"]`` and in the
    registry record (``init_missing`` / ``init_unexpected``).

    A checkpoint on another tokenizer is refused, except one this run's tokenizer
    extends (``tokenizer.extends``: the same ids plus the category group appended):
    its vocabulary tensors are grown by ``extend_vocab`` and the grown keys are
    recorded in ``meta["init_from"]["extended"]``.
    """
    ck = torch.load(cfg.init_from, map_location=device, weights_only=False)
    check_init_compat(ck, cfg)
    sd = strip_compiled(ck["model"] if "model" in ck else ck)
    ck_tok = (ck.get("meta") or {}).get("tokenizer")
    extended = None
    if tok_hash is not None and ck_tok not in (None, tok_hash):
        base = tokenizer.base() if tokenizer is not None and getattr(tokenizer, "cats", None) else None
        if base is None or base.hash != ck_tok:
            raise ValueError(f"init_from checkpoint used tokenizer {ck_tok}, shards use {tok_hash}")
        sd, grown = extend_vocab(model, sd, base.vocab_size)
        extended = {"from": ck_tok, "to": tok_hash, "old_vocab": base.vocab_size,
                    "new_vocab": tokenizer.vocab_size, "init_row": BOS_TOKEN, "keys": grown}
        print(f"[train] init_from vocabulary {base.vocab_size} -> {tokenizer.vocab_size} "
              f"({ck_tok} -> {tok_hash}); grew {grown}, new rows start as BOS", flush=True)
    missing, unexpected = load_weights(model, sd, strict=False)
    info = {"path": cfg.init_from, "run": os.path.dirname(os.path.abspath(cfg.init_from)), "step": ck.get("step"),
            "tokenizer": ck_tok, "missing": list(missing), "unexpected": list(unexpected)}
    if extended:
        info["extended"] = extended
    if missing or unexpected:
        what = (f"{len(missing)} missing key(s) (fresh weights) {list(missing)[:5]}{'...' if len(missing) > 5 else ''}, "
                f"{len(unexpected)} unexpected key(s) (dropped) {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not cfg.allow_partial_init:
            raise ValueError(f"init_from {cfg.init_from} does not match the model: {what}; "
                             f"set allow_partial_init=true to train from the keys that do line up")
        print(f"[train] WARNING init_from loaded partially: {what}", flush=True)
    print(f"[train] init from {cfg.init_from} (step {ck.get('step')})", flush=True)
    return info


def select_checkpoint(model, run_dir: str, cfg: TrainConfig, best: dict, step: int, final_nll: dict, device,
                      tokenizer=None):
    """Load the selected weights into ``model`` and return (their val_nll, their step).

    ``select=best`` loads ``best.pt`` when an earlier eval beat the final one; ``last``
    keeps the final weights; any other name is a registered hook (``SELECT_FNS``).
    The model that is sampled, benchmarked and registered is the selected one; the
    final weights stay in ``last.pt``.
    """
    hook = resolve_select(cfg.select)
    if hook is not None:
        return hook(model, run_dir, cfg, best, step, final_nll, device, tokenizer=tokenizer)
    path = os.path.join(run_dir, "best.pt")
    if cfg.select != "best" or best["step"] == step or not os.path.exists(path):
        return final_nll, step
    bk = torch.load(path, map_location=device, weights_only=False)
    load_weights(model, bk["model"])
    print(f"[train] select=best: step {best['step']} ({cfg.val[0]} {best['nll']:.4f}) over final step {step} "
          f"({final_nll[cfg.val[0]]:.4f})", flush=True)
    return bk["val"], best["step"]


def save_checkpoint(path: str, model, opt, step: int, tokens_seen: int, cfg: TrainConfig, meta: dict,
                    batches: MixedBatches, history: list, best: Optional[dict] = None) -> None:
    torch.save({"model": unwrap(model).state_dict(), "opt": opt.state_dict(), "step": step, "tokens_seen": tokens_seen,
                "config": cfg.to_dict(), "meta": meta, "batches": batches.state(), "history": history, "best": best,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def save_best(path: str, model, step: int, cfg: TrainConfig, meta: dict, ev: dict) -> None:
    torch.save({"model": unwrap(model).state_dict(), "step": step, "config": cfg.to_dict(), "meta": meta, "val": ev}, path)


def run_record(cfg: TrainConfig, run_dir: str, meta: dict, tokenizer, params: int, step: int, tokens_seen: int,
               minutes: float, history: list, val_nll: dict, final_nll: dict, val_nll_all: Optional[dict],
               eval_seqs: dict, selected: int, selected_ckpt: str, best: dict, score: dict) -> dict:
    """The registry line (``outputs/registry.jsonl``) and ``summary.json`` of a run."""
    train_loss = next((h["loss"] for h in reversed(history) if "loss" in h), None)
    init = meta.get("init_from") or {}
    parent = init.get("run")
    return {"run": run_dir, "name": cfg.name, "stage": cfg.stage, "arch": cfg.arch, "kind": cfg.kind,
            "layout": getattr(tokenizer, "layout", None) if cfg.kind == "facades" else None,
            "augment": cfg.augment, "preset": cfg.preset,
            "params": params, "seed": cfg.seed, "dropout": cfg.dropout, "lr": cfg.lr, "pe": cfg.pe,
            "model": cfg.model, "steps": step, "tokens": tokens_seen, "minutes": round(minutes, 1),
            "train": cfg.train, "val": cfg.val, "train_split": cfg.split_name(),
            "tokenizer": meta["tokenizer"], "tiers": meta["tiers"], "git": meta["git"],
            "config": cfg.config_path, "init_from": cfg.init_from, "parent_run": parent,
            "init_missing": len(init.get("missing") or []), "init_unexpected": len(init.get("unexpected") or []),
            "trained_on_signals": list(cfg.trained_on_signals or []),
            "train_loss": train_loss, "val_nll": val_nll, "val_nll_final": final_nll, "val_nll_all": val_nll_all,
            "eval_seqs": eval_seqs, "selected_step": selected, "selected_ckpt": selected_ckpt, "select": cfg.select,
            "best": best, "bench_n": cfg.bench_n, "bench": {k: v for k, v in score.items() if k not in ("dir", "card")},
            "card": score.get("card"), "samples_png": os.path.exists(os.path.join(run_dir, "samples.png")),
            "note": cfg.note}


def train(cfg: TrainConfig) -> str:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    arch = archs.get(cfg.arch)
    resolve_select(cfg.select)

    shards, weights, tok_hash = open_shards(cfg.train_specs(), cfg.tokenizer, cfg.kind, cfg.augment)
    tokenizer = shards[0].tokenizer
    if tokenizer.kind != cfg.kind:
        raise ValueError(f"config kind {cfg.kind!r} but the shards are {tokenizer.kind!r}")
    val = {n: Shard(find(n, tok_hash, cfg.kind)) for n in cfg.val}
    meta = {"tokenizer": tok_hash, "tiers": tier_hashes([n for n, _ in cfg.train_specs()] + cfg.val),
            "shards": {s.spec["tier"]: s.path for s in shards}, "git": git_sha(), "vocab_size": tokenizer.vocab_size,
            "arch": cfg.arch, "track": arch.track}

    run_dir = cfg.resume or str(new_run_dir(cfg.name, base=cfg.out))
    os.makedirs(run_dir, exist_ok=True)
    model = arch.build(cfg, tokenizer).to(device)
    opt = make_optimizer(model, cfg)
    batches = MixedBatches(shards, weights, cfg.batch_tokens, cfg.max_seq_len, seed=cfg.seed)
    step, tokens_seen, history = 0, 0, []
    best = {"nll": float("inf"), "step": 0}            # on the first val tier; best.pt follows it
    if cfg.resume:
        ck = torch.load(os.path.join(run_dir, "last.pt"), map_location=device, weights_only=False)
        load_weights(model, ck["model"]); opt.load_state_dict(ck["opt"])
        step, tokens_seen, history = ck["step"], ck["tokens_seen"], ck["history"]
        best = ck.get("best", best)
        batches.load_state(ck["batches"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng"]["cuda"]])
        print(f"[train] resumed {run_dir} at step {step}", flush=True)
    elif cfg.init_from:
        meta["init_from"] = init_from_checkpoint(model, cfg, tok_hash, device, tokenizer=tokenizer)
    params = n_params(model)
    json.dump({"config": cfg.to_dict(), "meta": meta, "params": params}, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    print(f"[train] {run_dir}: {params / 1e6:.1f}M params ({cfg.arch}), vocab {tokenizer.vocab_size}, "
          f"train {[(s.spec['tier'], len(s)) for s in shards]}, val {[(n, len(s)) for n, s in val.items()]}, "
          f"{cfg.steps} steps x {cfg.batch_tokens * cfg.grad_accum} tokens", flush=True)

    if cfg.compile:
        model = torch.compile(model)
    metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a")

    def log(rec: dict):
        rec = {"step": step, "tokens": tokens_seen, "elapsed": round(time.time() - t_start, 1), **rec}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush(); history.append(rec)

    it = iter(batches)
    t_start = time.time(); t_log = time.time(); tok_log = tokens_seen
    acc_loss = torch.zeros((), device=device); acc_n = torch.zeros((), device=device)
    model.train()
    while step < cfg.steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(cfg.grad_accum):
            inp, tgt, pad = next(it)
            inp, tgt, pad = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True), pad.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = arch.loss(model, (inp, tgt, pad))
            (loss / cfg.grad_accum).backward()
            n = (tgt != PAD_TOKEN).sum()
            acc_loss += loss.detach().float() * n; acc_n += n
            tokens_seen += int(n)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            l = float(acc_loss / acc_n.clamp(min=1)); acc_loss.zero_(); acc_n.zero_()
            dt = time.time() - t_log; tps = (tokens_seen - tok_log) / max(dt, 1e-6); t_log = time.time(); tok_log = tokens_seen
            log({"loss": round(l, 4), "lr": lr, "grad_norm": round(float(gn), 3), "tok_s": int(tps)})
            print(f"[train] step {step}/{cfg.steps} loss {l:.4f} lr {lr:.2e} gn {float(gn):.2f} {tps / 1e3:.0f}k tok/s "
                  f"{tokens_seen / 1e6:.0f}M tokens {(time.time() - t_start) / 60:.1f} min", flush=True)
        if cfg.eval_every and (step % cfg.eval_every == 0) and step < cfg.steps:
            ev = evaluate(model, val, cfg, device, use_amp)
            log({"val": ev}); print(f"[eval] step {step} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
            if cfg.val and ev[cfg.val[0]] < best["nll"]:
                best = {"nll": ev[cfg.val[0]], "step": step}
                save_best(os.path.join(run_dir, "best.pt"), model, step, cfg, meta, ev)
        if cfg.sample_every and step % cfg.sample_every == 0 and step < cfg.steps:
            sample_and_render(model, tokenizer, cfg, run_dir, step, cfg.n_samples, ckpt=None)
        if cfg.ckpt_every and step % cfg.ckpt_every == 0 and step < cfg.steps:
            save_checkpoint(os.path.join(run_dir, f"ckpt_{step:07d}.pt"), model, opt, step, tokens_seen, cfg, meta, batches, history, best)
            save_checkpoint(os.path.join(run_dir, "last.pt"), model, opt, step, tokens_seen, cfg, meta, batches, history, best)

    save_checkpoint(os.path.join(run_dir, "last.pt"), model, opt, step, tokens_seen, cfg, meta, batches, history, best)
    # The final pass on the same eval_seqs subset as the periodic ones, so best/final compare like with like.
    ev = evaluate(model, val, cfg, device, use_amp)
    log({"val": ev, "final": True}); print(f"[eval] final " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
    final_nll = ev
    if cfg.val and ev[cfg.val[0]] < best["nll"]:
        best = {"nll": ev[cfg.val[0]], "step": step}
        save_best(os.path.join(run_dir, "best.pt"), model, step, cfg, meta, ev)
    ev, selected = select_checkpoint(model, run_dir, cfg, best, step, final_nll, device, tokenizer=tokenizer)
    selected_ckpt = "last.pt" if selected == step else "best.pt"
    eval_seqs = {"periodic": cfg.eval_seqs, "final": holdout_sizes(val, cfg.eval_seqs, cfg.max_seq_len)}
    val_nll_all = None
    if cfg.final_eval_all:
        # The last pass: the selected weights on every holdout sequence that fits the context.
        val_nll_all = evaluate(model, val, cfg, device, use_amp, n_seqs=None)
        eval_seqs["final"] = holdout_sizes(val, None, cfg.max_seq_len)
        log({"val_all": val_nll_all, "eval_seqs": eval_seqs["final"], "selected_step": selected, "final_all": True})
        print(f"[eval] all holdout (step {selected}) " + " ".join(f"{k} {v:.4f} (n={eval_seqs['final'][k]})" for k, v in val_nll_all.items()), flush=True)
    npz = sample_and_render(model, tokenizer, cfg, run_dir, selected, max(cfg.n_samples, cfg.bench_n if cfg.bench else 0),
                            ckpt=selected_ckpt)
    score = bench(npz, cfg, run_dir) if (cfg.bench and npz) else {"ok": False, "skipped": True}
    rec = run_record(cfg, run_dir, meta, tokenizer, params, step, tokens_seen, (time.time() - t_start) / 60, history,
                     ev, final_nll, val_nll_all, eval_seqs, selected, selected_ckpt, best, score)
    registry.append(rec)
    json.dump(rec, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    metrics.close()
    print(f"[train] done -> {run_dir}", flush=True)
    return run_dir


def load_run_checkpoint(run_dir: str, ckpt: str = "last.pt", device: str = "cuda"):
    """``load_run`` plus the checkpoint header: ``(model, tokenizer, config, info)`` with
    ``info = {"file", "step", "val"}``, so a sampler can report the step the weights
    are from without reading the file twice."""
    from blockgen.corpus.tokenizer import CorpusTokenizer
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(run_dir, ckpt), map_location=dev, weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__})
    shard_dir = next(iter(ck["meta"]["shards"].values()))
    tokenizer = CorpusTokenizer.load(os.path.join(shard_dir, "tokenizer.json"))
    model = build_model(cfg, tokenizer).to(dev)
    load_weights(model, ck["model"])
    model.eval()
    info = {"file": ckpt, "step": ck.get("step"), "val": ck.get("val")}
    return model, tokenizer, cfg, info


def load_run(run_dir: str, ckpt: str = "last.pt", device: str = "cuda"):
    """(model, tokenizer, config) from a run directory, for sampling and evaluation."""
    return load_run_checkpoint(run_dir, ckpt, device)[:3]
