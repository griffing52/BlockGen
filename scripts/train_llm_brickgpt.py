"""BrickGPT/LegoGPT-style text->build generation with a LoRA-finetuned LLM.

We text-serialize each Minecraft house as one line per occupied voxel
(``"<name> <x> <y> <z>"`` in ``(y, z, x)`` raster order, mirroring
``blockgen.utils.serialize.structure_to_tokens``) and LoRA-finetune a small
pretrained causal LM (``Qwen/Qwen2.5-1.5B``, the non-gated BrickGPT-analog base)
to map a text caption to that block listing.  Completion-only loss, exactly like
BrickGPT trains bricks conditioned on the prompt.

Pipeline: serialize -> filter by token length -> LoRA SFT -> sample held-out val
captions -> parse generated lines back into ``Structure`` -> TEXTURED render.

Run::

    .venv/bin/python scripts/train_llm_brickgpt.py                       # full run
    .venv/bin/python scripts/train_llm_brickgpt.py --limit 64 --epochs 1 --num-samples 2   # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

# Large (152k) vocab logits fragment memory during the CE loss; expandable
# segments keeps a shared 16GB GPU (other jobs present) from OOM'ing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Dict, List, Tuple

import numpy as np
import torch

from blockgen.curation.houses import load_house_structures
from blockgen.utils.data import Structure, _resource_location_for
from blockgen.utils.runs import new_run_dir

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"  # code-tuned, non-gated; good for the structured line format
CAPTIONS_PATH = "data/minecraft/labels/houses_32_captions.json"


# --- reversible block-name map --------------------------------------------
def _name_for_pair(block_id: int, block_data: int) -> str:
    """Readable modern-ish block name from a legacy ``(id, data)`` pair.

    Derived from ``_resource_location_for`` (strip ``minecraft:``, snake-case),
    so it stays consistent with the rest of the codebase.  Orientation/variant
    pairs that share a resource location collapse to one name (fine for v1 --
    BrickGPT has no orientation either).
    """
    rl = _resource_location_for(int(block_id), int(block_data))
    name = rl.replace("minecraft:", "").lower()
    name = re.sub(r"[^0-9a-z_]+", "_", name).strip("_")
    return name or f"block_{int(block_id)}"


def build_name_map(structures) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Return ``(name_to_pair, pair_to_name)``.

    ``pair_to_name`` maps every distinct ``"id:data"`` string that appears to its
    name; ``name_to_pair`` maps each name to its FIRST (canonical) ``[id, data]``
    pair (so a name that several pairs share inverts to the first-seen pair).
    """
    name_to_pair: Dict[str, list] = {}
    pair_to_name: Dict[str, str] = {}
    for s in structures:
        occ = s.occupied_mask
        ids = s.block_ids[occ].tolist()
        datas = s.block_data[occ].tolist()
        for bid, bdata in zip(ids, datas):
            key = f"{int(bid)}:{int(bdata)}"
            if key in pair_to_name:
                continue
            name = _name_for_pair(bid, bdata)
            pair_to_name[key] = name
            if name not in name_to_pair:  # keep FIRST pair as canonical
                name_to_pair[name] = [int(bid), int(bdata)]
    return name_to_pair, pair_to_name


# --- serialization ---------------------------------------------------------
def serialize_structure(s: Structure, pair_to_name: Dict[str, str]) -> str:
    """One line per occupied voxel ``"<name> <x> <y> <z>"`` in (y, z, x) order."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    coords = np.argwhere(occ)  # rows of (x, y, z)
    order = np.lexsort((coords[:, 0], coords[:, 2], coords[:, 1]))  # y, then z, then x
    coords = coords[order]
    lines: List[str] = []
    for x, y, z in coords.tolist():
        key = f"{int(c.block_ids[x, y, z])}:{int(c.block_data[x, y, z])}"
        name = pair_to_name[key]
        lines.append(f"{name} {x} {y} {z}")
    return "\n".join(lines)


def parse_completion(text: str, name_to_pair: Dict[str, list]) -> Tuple[Structure, int, int]:
    """Parse generated block lines back into a ``Structure``.

    Returns ``(structure, n_valid, n_lines)``.  Malformed lines and unknown
    names are skipped so a bad generation still yields a renderable build.
    """
    placed: List[Tuple[int, int, int, int, int]] = []
    n_lines = 0
    n_valid = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        n_lines += 1
        parts = line.split()
        if len(parts) != 4:
            continue
        name, xs, ys, zs = parts
        pair = name_to_pair.get(name)
        if pair is None:
            continue
        try:
            x, y, z = int(xs), int(ys), int(zs)
        except ValueError:
            continue
        if x < 0 or y < 0 or z < 0 or max(x, y, z) > 255:
            continue
        n_valid += 1
        placed.append((x, y, z, pair[0], pair[1]))

    if not placed:
        return (Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                          block_data=np.zeros((1, 1, 1), np.int32)),
                n_valid, n_lines)
    arr = np.array(placed, dtype=np.int32)
    sx, sy, sz = int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1, int(arr[:, 2].max()) + 1
    block_ids = np.zeros((sx, sy, sz), np.int32)
    block_data = np.zeros((sx, sy, sz), np.int32)
    for x, y, z, bid, bdata in arr.tolist():
        block_ids[x, y, z] = bid
        block_data[x, y, z] = bdata
    return Structure(block_ids=block_ids, block_data=block_data), n_valid, n_lines


# --- dataset ---------------------------------------------------------------
def build_examples(structures, captions, tokenizer, pair_to_name,
                   max_tokens: int, limit: int | None):
    """Build tokenized (prompt-masked) examples <= ``max_tokens`` total.

    Returns ``(examples, n_considered)`` where each example is a dict with
    ``input_ids``, ``labels`` (prompt + pad masked to -100), ``caption``, ``idx``.
    """
    eos_id = tokenizer.eos_token_id
    examples = []
    n_considered = 0
    if limit is None:
        rng = range(len(structures))
    else:
        # Random (seeded) subset, not first-N: the first structures are all large
        # houses, so a first-N slice would contain no <=max_tokens builds.
        rng = sorted(random.Random(0).sample(range(len(structures)),
                                             min(limit, len(structures))))
    for i in rng:
        key = f"h{i:05d}"
        caps = captions.get(key)
        if not caps:
            continue
        caption = caps[0]  # the short tag (caption[0])
        completion = serialize_structure(structures[i], pair_to_name)
        if not completion:
            continue
        n_considered += 1
        prompt = f"Build: {caption}\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [eos_id]
        input_ids = prompt_ids + comp_ids
        if len(input_ids) > max_tokens:
            continue
        labels = [-100] * len(prompt_ids) + list(comp_ids)  # completion-only loss
        examples.append({"input_ids": input_ids, "labels": labels,
                         "caption": caption, "idx": i})
    return examples, n_considered


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        ids = b["input_ids"]
        lab = b["labels"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)  # pad masked from loss
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def chunked_ce_loss(model, input_ids, attn, labels, *, chunk: int = 512,
                    use_ckpt: bool = True):
    """Mean completion-only CE loss without materializing full [S, 152k] logits.

    Qwen2.5's 152k-vocab LM head produces a huge logits tensor; on a shared 16GB
    GPU that OOMs in backward. We run the base model once for hidden states, then
    apply the LM head + cross-entropy in sequence chunks, each wrapped in an
    activation checkpoint so only one chunk of logits exists at a time during
    backward (labels are already -100 on prompt + pad, so this stays
    completion-only). Mirrors HF's shift (logits[:-1] predict labels[1:]).
    """
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    qwen = model.base_model.model  # Qwen2ForCausalLM under the LoRA wrapper
    hidden = qwen.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    lm_head = qwen.lm_head
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    _, sm1, _ = shift_hidden.shape

    def _chunk_loss(h_slice, lbl_slice):
        logits = lm_head(h_slice).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               lbl_slice.reshape(-1), ignore_index=-100, reduction="sum")

    total = shift_hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for a in range(0, sm1, chunk):
        b = min(a + chunk, sm1)
        h_slice, lbl_slice = shift_hidden[:, a:b, :], shift_labels[:, a:b]
        if use_ckpt and torch.is_grad_enabled():
            loss_c = checkpoint(_chunk_loss, h_slice, lbl_slice, use_reentrant=False)
        else:
            loss_c = _chunk_loss(h_slice, lbl_slice)
        total = total + loss_c
        count += int((lbl_slice != -100).sum())
    return total / max(count, 1)


@torch.no_grad()
def eval_loss(model, examples, pad_id, micro_batch, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for k in range(0, len(examples), micro_batch):
        batch = examples[k:k + micro_batch]
        input_ids, attn, labels = collate(batch, pad_id)
        input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = chunked_ce_loss(model, input_ids, attn, labels, use_ckpt=False)
        total += float(loss) * len(batch)
        count += len(batch)
    return total / max(count, 1)


# --- main ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #structures considered (smoke)")
    # ~5 epochs (as specified) is too few here: each build has ~150-200 block
    # lines, so the prompt->first-block transition token gets only ~1/N of the
    # mean completion loss and free generation never learns to START the format.
    # Many small builds x ~20 epochs lets that transition signal accumulate
    # (verified: greedy then emits "<name> x y z" lines). Hence default 20.
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--micro-batch", type=int, default=1)
    # The dataset is tiny (~70 small builds). An effective batch of ~16 gives only
    # ~5 optimizer steps/epoch -- far too few for the diluted format-initiation
    # signal to accumulate. Effective batch 4 gives hundreds of opt steps and the
    # model reliably learns to emit the block format (verified). micro=1 keeps
    # 152k-vocab logits within a shared 16GB GPU.
    ap.add_argument("--grad-accum", type=int, default=4)
    # NOTE: the task's 1024 cap assumed ~few-hundred builds would fit, but this
    # per-voxel serialization costs ~10 tokens/block, so only 3/2661 houses fit in
    # 1024 tokens. 2048 tokens yields ~70 of the SMALLEST houses -- and smaller
    # builds are exactly what train well here (less first-token dilution; see the
    # --epochs note). Larger caps add more (diluted) data but degrade format
    # emission. See the run's config.json / stdout for the exact filtered count.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="generation budget; match --max-tokens so builds aren't truncated")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model

    run = new_run_dir("llm_brickgpt")
    print(f"[run] {run}", flush=True)

    # --- data ---
    structures, _ = load_house_structures(max_dim=32)
    captions = json.loads(Path(CAPTIONS_PATH).read_text())
    print(f"[data] {len(structures)} structures, {len(captions)} caption sets", flush=True)

    name_to_pair, pair_to_name = build_name_map(structures)
    print(f"[names] {len(name_to_pair)} distinct block names "
          f"({len(pair_to_name)} legacy pairs collapsed)", flush=True)
    (run / "name_map.json").write_text(json.dumps(
        {"name_to_pair": name_to_pair, "pair_to_name": pair_to_name}, indent=1))

    print(f"[model] loading tokenizer + {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    examples, n_considered = build_examples(
        structures, captions, tokenizer, pair_to_name, args.max_tokens, args.limit)
    print(f"[filter] {len(examples)}/{n_considered} builds <= {args.max_tokens} tokens "
          f"(of 2661 total houses)", flush=True)
    if not examples:
        raise SystemExit("no examples after filtering")

    # Deterministic val split.
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(round(len(examples) * args.val_frac)))
    val_idx = set(order[:n_val])
    train = [examples[i] for i in order[n_val:]]
    val = [examples[i] for i in order[:n_val]]
    print(f"[split] train {len(train)}  val {len(val)}", flush=True)

    # --- model + LoRA ---
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    lora = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.to(device)
    # Gradient checkpointing keeps ~3k-token sequences within 16GB.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable params {trainable:,} / {total:,} "
          f"({100 * trainable / total:.4f}%)", flush=True)

    # --- config dump ---
    cfg = {"model": MODEL_NAME, "lora": {"r": 32, "alpha": 16, "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"]},
           "trainable_params": int(trainable), "total_params": int(total),
           "max_tokens": args.max_tokens, "epochs": args.epochs, "lr": args.lr,
           "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
           "n_examples": len(examples), "n_train": len(train), "n_val": len(val),
           "n_considered": n_considered, "seed": args.seed}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))

    # --- training ---
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(train) / (args.micro_batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps) + 1, total_steps)
    print(f"[train] {steps_per_epoch} opt-steps/epoch x {args.epochs} = {total_steps} steps",
          flush=True)

    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train)
        run_loss, run_n = 0.0, 0
        opt.zero_grad()
        micro = 0
        for k in range(0, len(train), args.micro_batch):
            batch = train[k:k + args.micro_batch]
            input_ids, attn, labels = collate(batch, pad_id)
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                full_loss = chunked_ce_loss(model, input_ids, attn, labels)
                loss = full_loss / args.grad_accum
            loss.backward()
            run_loss += float(full_loss) * len(batch)
            run_n += len(batch)
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        # flush trailing grads
        if micro % args.grad_accum != 0:
            opt.step()
            sched.step()
            opt.zero_grad()
        tr = run_loss / max(run_n, 1)
        vl = eval_loss(model, val, pad_id, args.micro_batch, device)
        print(f"[epoch {epoch + 1}/{args.epochs}] train_loss {tr:.4f}  val_loss {vl:.4f}",
              flush=True)

    # --- save adapter ---
    model.save_pretrained(str(run / "adapter"))
    print(f"[save] adapter -> {run / 'adapter'}", flush=True)

    # --- sampling + render ---
    n_samples = min(args.num_samples, len(val))
    picks = val[:n_samples]
    model.eval()
    model.config.use_cache = True  # re-enable KV cache for fast generation
    sample_structs, prompts, parse_rates = [], [], []
    for ex in picks:
        prompt = f"Build: {ex['caption']}\n"
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = model.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                                 eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        struct, n_valid, n_lines = parse_completion(text, name_to_pair)
        rate = n_valid / n_lines if n_lines else 0.0
        parse_rates.append(rate)
        sample_structs.append(struct)
        prompts.append(ex["caption"])
        print(f"[sample idx={ex['idx']}] blocks={int(struct.occupied_mask.sum())} "
              f"lines={n_lines} valid={n_valid} parse_rate={rate:.2f}", flush=True)

    overall = float(np.mean(parse_rates)) if parse_rates else 0.0
    print(f"[parse] mean parse-success rate {overall:.3f}", flush=True)

    from blockgen.eval.cond_render import textured_prompt_grid
    out_png = run / "samples_textured.png"
    textured_prompt_grid(sample_structs, prompts, out_png,
                         suptitle=f"LoRA {MODEL_NAME} text->build  "
                                  f"(mean parse {overall:.2f})")
    print(f"[render] {out_png}", flush=True)
    (run / "sample_report.json").write_text(json.dumps(
        {"mean_parse_rate": overall,
         "samples": [{"idx": ex["idx"], "caption": ex["caption"],
                      "parse_rate": r, "blocks": int(s.occupied_mask.sum())}
                     for ex, r, s in zip(picks, parse_rates, sample_structs)]}, indent=1))
    print(f"[done] run dir: {run}", flush=True)


if __name__ == "__main__":
    main()
