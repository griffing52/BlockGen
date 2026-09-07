"""Track D + a structural attention bias: inject adjacency instead of extracting it.

`scripts/probe_llm_attention.py` asks whether a pretrained LLM's attention already
*contains* 3D neighbourhood. This asks the complementary question, and it is the one
the graph-transformer literature actually answers: what if we **put the geometry
into** attention rather than trying to read it out?

Everything here is the Track D piece-serialization run (`train_llm_pieces.py`)
held fixed -- same 3D-BPE piece vocabulary, same `"<piece_name> <x> <y> <z>"` lines,
same LoRA rank/target modules, same optimizer, same 2048-token cap, same seed and
split -- plus one addition in the `--bias` arm: a learned additive term on the
attention scores, indexed by the **relative 3D offset between the pieces two token
positions refer to**. This is the Graphormer/relative-position-bias move, and it is
the same mechanism `blockgen/models/pick_n_place.py` uses to fix the blindness
diagnosed in notes.md §20 ("cannot read its own op history as geometry").

**Causality, which is the whole design constraint.** At generation time the anchor of
the line being written is not known until its coordinates have been emitted, so the
query side cannot use it. The reference point for query token `q` is therefore the
anchor of the last *completed* line, and keys inside `q`'s own line get a separate
learned scalar rather than geometry. Nothing in the bias depends on a coordinate the
model has not already produced, so training and sampling see the same function.

    bias[q, k] = table[ anchor(line(q) - 1) - anchor(line(k)) ]   (clamped to +-r)
               = same_line   if line(k) == line(q)
               = prompt      if k is a prompt token
               = no_ref      if line(q) == 0

The table is zero-initialized, so at step 0 the `--bias` arm is *exactly* the control
arm; anything it gains, it learns. Injected through `attention_mask` as a prepared
`{"full_attention": ...}` mask, so it is shared across all layers and heads -- the
minimal intervention, and a blunt one: a null result here bounds a shared-scalar bias,
not per-head structure.

Run::

    .venv/bin/python scripts/train_llm_geobias.py --arm control
    .venv/bin/python scripts/train_llm_geobias.py --arm bias
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import load_piece_vocab
from blockgen.utils.runs import new_run_dir
from train_llm_pieces import (MODEL_NAME, CAPTIONS_PATH, PIECE_VOCAB, build_piece_names,
                              parse_piece_completion, serialize_pieces)
from probe_llm_attention import piece_instances, line_spans


# --- the bias ---------------------------------------------------------------
class GeoBias(nn.Module):
    """Additive attention bias from the relative offset between referenced pieces."""

    def __init__(self, radius: int = 4):
        super().__init__()
        self.r = radius
        self.n = 2 * radius + 1
        self.table = nn.Parameter(torch.zeros(self.n ** 3))
        self.prompt = nn.Parameter(torch.zeros(()))
        self.same_line = nn.Parameter(torch.zeros(()))
        self.no_ref = nn.Parameter(torch.zeros(()))

    def matrix(self, tok_line: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """`[T, T]` float32 bias. `tok_line[t]` is -1 for prompt tokens."""
        T = tok_line.numel()
        dev = tok_line.device
        ql = tok_line
        ref = ql - 1                                     # last completed line
        a_ref = anchors[ref.clamp(min=0)]                # [T, 3]
        a_key = anchors[ql.clamp(min=0)]                 # [T, 3]
        idx = torch.zeros((T, T), dtype=torch.long, device=dev)
        for ax in range(3):
            d = (a_ref[:, ax].unsqueeze(1) - a_key[:, ax].unsqueeze(0)).clamp(-self.r, self.r) + self.r
            idx = idx * self.n + d
        out = self.table[idx.reshape(-1)].reshape(T, T)
        same = ql.unsqueeze(1) == ql.unsqueeze(0)
        out = torch.where(same, self.same_line.expand_as(out), out)
        out = torch.where((ql < 0).unsqueeze(0).expand_as(out), self.prompt.expand_as(out), out)
        out = torch.where((ref < 0).unsqueeze(1).expand_as(out), self.no_ref.expand_as(out), out)
        return out


def causal_mask(T: int, dtype, device, bias: torch.Tensor | None = None):
    """`{"full_attention": [1,1,T,T]}` -- a prepared mask transformers uses verbatim."""
    m = torch.zeros((T, T), dtype=torch.float32, device=device)
    if bias is not None:
        m = m + bias
    m = m.masked_fill(torch.ones((T, T), dtype=torch.bool, device=device).triu(1),
                      torch.finfo(dtype).min)
    return {"full_attention": m.to(dtype)[None, None]}


# --- data -------------------------------------------------------------------
def build_examples(structures, captions, tokenizer, cv, id_to_name, max_tokens: int,
                   limit: int | None):
    """Track D's examples, plus the per-token line ids and per-line anchors."""
    eos_id = tokenizer.eos_token_id
    out, considered = [], 0
    rng = range(len(structures)) if limit is None else sorted(
        random.Random(0).sample(range(len(structures)), min(limit, len(structures))))
    for i in rng:
        caps = captions.get(f"h{i:05d}")
        if not caps:
            continue
        inst = piece_instances(structures[i], cv, id_to_name)
        if inst is None:
            continue
        considered += 1
        lines = [f"{d['name']} {d['anchor'][0]} {d['anchor'][1]} {d['anchor'][2]}" for d in inst]
        prompt = f"Build: {caps[0]}\n"
        # Track D dropped any build over the cap; keep that exactly (no truncation).
        ids, spans, keep = line_spans(tokenizer, prompt, lines, max_tokens)
        if keep < len(lines) or not spans:
            continue
        ids = list(ids) + [eos_id]
        n_prompt = spans[0][0]
        tok_line = np.full(len(ids), -1, dtype=np.int64)
        for li, (a, b) in enumerate(spans):
            tok_line[a:b] = li
        tok_line[spans[-1][1]:] = len(spans) - 1          # trailing eos joins the last line
        labels = [-100] * n_prompt + ids[n_prompt:]
        out.append({"input_ids": ids, "labels": labels, "tok_line": tok_line,
                    "anchors": np.array([d["anchor"] for d in inst], dtype=np.int64),
                    "caption": caps[0], "idx": i})
    return out, considered


def loss_for(model, ex, geo: GeoBias | None, device, use_ckpt: bool = True):
    """Completion-only CE, chunked over the 152k-vocab head (Track D's recipe)."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([ex["labels"]], dtype=torch.long, device=device)
    T = ids.shape[1]
    bias = None
    if geo is not None:
        tl = torch.as_tensor(ex["tok_line"], device=device)
        an = torch.as_tensor(ex["anchors"], device=device)
        bias = geo.matrix(tl, an)
    qwen = model.base_model.model
    mask = causal_mask(T, qwen.model.embed_tokens.weight.dtype, device, bias)
    hidden = qwen.model(input_ids=ids, attention_mask=mask).last_hidden_state
    lm_head = qwen.lm_head
    sh, sl = hidden[:, :-1, :], labels[:, 1:]

    def _chunk(h, l):
        lg = lm_head(h).float()
        return F.cross_entropy(lg.reshape(-1, lg.size(-1)), l.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total = sh.new_zeros((), dtype=torch.float32)
    count = 0
    for a in range(0, sh.shape[1], 512):
        b = min(a + 512, sh.shape[1])
        hs, ls = sh[:, a:b, :], sl[:, a:b]
        total = total + (checkpoint(_chunk, hs, ls, use_reentrant=False)
                         if use_ckpt and torch.is_grad_enabled() else _chunk(hs, ls))
        count += int((ls != -100).sum())
    return total / max(count, 1)


# --- sampling ---------------------------------------------------------------
@torch.no_grad()
def sample(model, tokenizer, geo: GeoBias | None, caption: str, device,
           max_new: int = 1600, temperature: float = 0.8, top_p: float = 0.95, seed: int = 0):
    """Recompute-the-mask sampling (no KV cache): the bias needs the full mask each step."""
    torch.manual_seed(seed)
    prompt = f"Build: {caption}\n"
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n_prompt = len(ids)
    dtype = model.base_model.model.model.embed_tokens.weight.dtype
    text, spans_char = "", []           # generated text + per-token char span
    tok_line = [-1] * n_prompt
    anchors: List[Tuple[int, int, int]] = []
    line_start_char = 0
    cur_line = 0

    for _ in range(max_new):
        T = len(ids)
        bias = None
        if geo is not None:
            tl = torch.tensor(tok_line, dtype=torch.long, device=device)
            an = torch.tensor(anchors + [(0, 0, 0)], dtype=torch.long, device=device)
            bias = geo.matrix(tl, an)
        mask = causal_mask(T, dtype, device, bias)
        qwen = model.base_model.model
        h = qwen.model(input_ids=torch.tensor([ids], device=device),
                       attention_mask=mask).last_hidden_state[:, -1]
        logits = qwen.lm_head(h).float()[0] / temperature
        probs = torch.softmax(logits, -1)
        sp, si = torch.sort(probs, descending=True)
        cut = int(torch.searchsorted(torch.cumsum(sp, 0), top_p)) + 1
        tok = int(si[torch.multinomial(sp[:cut] / sp[:cut].sum(), 1)])
        if tok == tokenizer.eos_token_id:
            break
        ids.append(tok)
        piece = tokenizer.decode([tok])
        text += piece
        tok_line.append(cur_line)
        if "\n" in piece:
            line = text[line_start_char:].split("\n")[0]
            parts = line.split()
            anchors.append(tuple(int(v) for v in parts[1:4])
                           if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts[1:4])
                           else (anchors[-1] if anchors else (0, 0, 0)))
            line_start_char = len(text)
            cur_line += 1
    if len(anchors) < cur_line + 1:
        anchors.append((0, 0, 0))
    return text


# --- main -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "bias"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bias-lr", type=float, default=1e-2)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    run = new_run_dir(f"llm_geobias_{args.arm}")
    print(f"[run] {run}", flush=True)

    structures, _ = load_house_structures(max_dim=32)
    captions = json.loads(Path(CAPTIONS_PATH).read_text())
    cv = load_piece_vocab(PIECE_VOCAB)
    id_to_name, name_to_id = build_piece_names(cv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples, considered = build_examples(structures, captions, tokenizer, cv, id_to_name,
                                          args.max_tokens, args.limit)
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(round(len(examples) * 0.10)))
    train = [examples[i] for i in order[n_val:]]
    val = [examples[i] for i in order[:n_val]]
    print(f"[data] {len(examples)}/{considered} builds fit {args.max_tokens} tokens; "
          f"train {len(train)} val {len(val)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             bias="none", task_type="CAUSAL_LM")).to(device)
    model.config.use_cache = False
    # No layer-level gradient checkpointing: the bias mask is a single non-leaf tensor
    # consumed by all 28 layers, and recomputation backprops through its graph once per
    # layer ("backward through the graph a second time"). The chunked LM head below is
    # what actually keeps the 152k-vocab logits off the GPU, so we keep that and drop
    # this. Both arms run identically configured.
    model.enable_input_require_grads()

    geo = GeoBias(args.radius).to(device) if args.arm == "bias" else None
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if geo is not None:
        groups.append({"params": list(geo.parameters()), "lr": args.bias_lr})
        print(f"[bias] radius {args.radius} -> {geo.table.numel()} offset buckets "
              f"+ 3 specials, shared across all layers/heads (zero-init)", flush=True)
    opt = torch.optim.AdamW(groups)
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps) + 1, steps)

    cfg = {"arm": args.arm, "model": MODEL_NAME, "epochs": args.epochs, "lr": args.lr,
           "bias_lr": args.bias_lr, "radius": args.radius, "grad_accum": args.grad_accum,
           "max_tokens": args.max_tokens, "n_examples": len(examples), "n_train": len(train),
           "n_val": len(val), "n_considered": considered, "seed": args.seed,
           "piece_vocab": os.path.abspath(PIECE_VOCAB), "n_pieces": cv.n_pieces}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    (run / "piece_name_map.json").write_text(json.dumps(
        {"piece_vocab": os.path.abspath(PIECE_VOCAB),
         "id_to_name": id_to_name, "name_to_id": name_to_id}, indent=1))

    history = []
    for ep in range(args.epochs):
        model.train()
        random.Random(args.seed + ep).shuffle(train)
        tot, n = 0.0, 0
        opt.zero_grad()
        for k, ex in enumerate(train):
            with torch.autocast("cuda", torch.bfloat16):
                l = loss_for(model, ex, geo, device)
            (l / args.grad_accum).backward()
            tot += float(l); n += 1
            if (k + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_for(model, e, geo, device, use_ckpt=False))
                                for e in val]))
        history.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "val_loss": vl})
        print(f"[epoch {ep + 1}/{args.epochs}] train {tot / max(n, 1):.4f}  val {vl:.4f}",
              flush=True)
        (run / "history.json").write_text(json.dumps(history, indent=1))

    model.save_pretrained(run / "adapter")
    if geo is not None:
        torch.save(geo.state_dict(), run / "geo_bias.pt")

    # --- samples from held-out captions ------------------------------------
    print(f"[sample] {args.num_samples} held-out captions", flush=True)
    model.eval()
    report, sample_structs, prompts = [], [], []
    for i, ex in enumerate(val[: args.num_samples]):
        txt = sample(model, tokenizer, geo, ex["caption"], device,
                     max_new=args.max_new_tokens, seed=args.seed + i)
        st, n_valid, n_lines = parse_piece_completion(txt, cv, name_to_id)
        report.append({"caption": ex["caption"], "idx": int(ex["idx"]),
                       "n_lines": n_lines, "n_valid": n_valid,
                       "parse_rate": n_valid / max(n_lines, 1),
                       "blocks": int(st.occupied_mask.sum()), "shape": list(st.shape)})
        sample_structs.append(st); prompts.append(ex["caption"])
        np.savez_compressed(run / f"sample_{i:02d}.npz",
                            block_ids=st.block_ids, block_data=st.block_data)
        (run / f"sample_{i:02d}.txt").write_text(txt)
        print(f"  [{i}] {ex['caption'][:60]!r} lines={n_lines} valid={n_valid} "
              f"blocks={int(st.occupied_mask.sum())}", flush=True)
    (run / "sample_report.json").write_text(json.dumps(
        {"arm": args.arm, "samples": report,
         "parse_rate": float(np.mean([r["parse_rate"] for r in report])),
         "median_blocks": float(np.median([r["blocks"] for r in report]))}, indent=1))

    from blockgen.eval.cond_render import textured_prompt_grid
    pr = float(np.mean([r["parse_rate"] for r in report]))
    textured_prompt_grid(sample_structs, prompts, run / "samples_textured.png",
                         suptitle=f"Track D + geometric attention bias — arm={args.arm} "
                                  f"(mean parse {pr:.2f})")
    print(f"[render] {run}/samples_textured.png", flush=True)
    print(f"[done] {run}", flush=True)


if __name__ == "__main__":
    main()
