"""BrickGPT-style text->build, v2: serialize 3D-BPE PIECES, not voxels.

v1 (``train_llm_brickgpt.py``) emits one line per occupied voxel -- ~10 tokens per
block, so a median house is ~7.8k tokens and only the ~70 smallest builds fit under
the LLM's context. This v2 emits one line per *piece* (a learned 3D-BPE cluster: a
wall run, a roof-slope unit, a whole bed/door), so a build is a much shorter list of
``"<piece_name> <x> <y> <z>"`` placements. Pieces cover several voxels each, cutting
sequence length several-fold and letting far more (and larger) builds train.

Each piece gets a stable readable name ``<majority_block>[_k]`` -- the piece's
dominant material, with a per-material counter to disambiguate. That keeps the name
grounded in the caption's vocabulary ("brick cottage" -> brick_* pieces) while staying
reversible: name -> piece id -> pattern, placed at the line's anchor.

Everything else mirrors v1: completion-only LoRA SFT of Qwen2.5-Coder-1.5B, held-out
val sampling, parse-back to ``Structure``, textured render.

Reuses the piece vocabulary learned for the AR track
(``data/minecraftace/houses_32_bpe``); pass ``--piece-vocab`` to swap it.

Run::

    .venv/bin/python scripts/train_llm_pieces.py                      # full run
    .venv/bin/python scripts/train_llm_pieces.py --limit 200 --epochs 1 --num-samples 2  # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))  # for train_llm_brickgpt

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import load_piece_vocab
from blockgen.tokenizers.cluster_bpe import ClusterVocab, _apply_merge, _atomic_labeling
from blockgen.utils.data import Structure, _token_for
from blockgen.utils.serialize import BlockVocab
from blockgen.utils.runs import new_run_dir
# Shared training machinery -- identical loss/collate/eval as v1.
from train_llm_brickgpt import chunked_ce_loss, collate, eval_loss, _name_for_pair

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
CAPTIONS_PATH = "data/minecraft/labels/houses_32_captions.json"
PIECE_VOCAB = "data/minecraftace/houses_32_bpe/houses_32_bpe_piece_vocab.json"


# --- reversible piece-name map --------------------------------------------
def build_piece_names(cv: ClusterVocab) -> Tuple[Dict[int, str], Dict[str, int]]:
    """``(id_to_name, name_to_id)``; name = ``<majority_block>[_k]`` (k per material)."""
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    counter: Dict[str, int] = {}
    for pid, pat in enumerate(cv.patterns):
        bidxs = [cell[3] for cell in pat]
        maj = max(set(bidxs), key=bidxs.count)
        bid, bdata = cv.block_index_to_pair[maj]
        base = _name_for_pair(bid, bdata)
        k = counter.get(base, 0)
        counter[base] = k + 1
        name = base if k == 0 else f"{base}_{k}"
        id_to_name[pid] = name
        name_to_id[name] = pid
    return id_to_name, name_to_id


def _base_vocab(cv: ClusterVocab) -> BlockVocab:
    return BlockVocab(
        max_dim=cv.max_dim,
        block_token_to_id={_token_for(*p, oriented=cv.oriented): i
                           for i, p in enumerate(cv.block_index_to_pair)},
        id_to_block_token=[_token_for(*p, oriented=cv.oriented)
                           for p in cv.block_index_to_pair],
        block_index_to_pair=cv.block_index_to_pair, oriented=cv.oriented)


# --- serialization ---------------------------------------------------------
def serialize_pieces(s: Structure, cv: ClusterVocab, id_to_name: Dict[int, str]) -> str | None:
    """One line per piece instance ``"<piece_name> <x> <y> <z>"`` in (y, z, x) order.

    Tokenizes by replaying the learned merges (deterministic), exactly like
    ``structure_to_cluster_tokens``. Returns None if the build exceeds ``max_dim``.
    """
    base = _base_vocab(cv)
    c = s.crop_to_non_air()
    if max(c.shape) > cv.max_dim:
        return None
    inst, owner, nid = _atomic_labeling(c, base)
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)
    order = sorted(inst.values(), key=lambda d: (d["anchor"][1], d["anchor"][2], d["anchor"][0]))
    return "\n".join(f"{id_to_name[d['piece']]} {d['anchor'][0]} {d['anchor'][1]} {d['anchor'][2]}"
                     for d in order)


def parse_piece_completion(text: str, cv: ClusterVocab, name_to_id: Dict[str, int]
                           ) -> Tuple[Structure, int, int]:
    """Parse generated ``"<piece_name> <x> <y> <z>"`` lines back into a ``Structure``.

    Each valid line stamps the piece's pattern at the anchor. Malformed lines and
    unknown piece names are skipped so a bad generation still renders.
    Returns ``(structure, n_valid, n_lines)``.
    """
    placed: List[Tuple[int, int, int, int]] = []  # (x, y, z, block_index)
    n_lines = n_valid = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        n_lines += 1
        parts = line.split()
        if len(parts) != 4:
            continue
        name, xs, ys, zs = parts
        pid = name_to_id.get(name)
        if pid is None:
            continue
        try:
            ax, ay, az = int(xs), int(ys), int(zs)
        except ValueError:
            continue
        if min(ax, ay, az) < 0 or max(ax, ay, az) > 255:
            continue
        n_valid += 1
        for dx, dy, dz, bidx in cv.patterns[pid]:
            placed.append((ax + dx, ay + dy, az + dz, bidx))

    if not placed:
        return (Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                          block_data=np.zeros((1, 1, 1), np.int32)), n_valid, n_lines)
    arr = np.array(placed, dtype=np.int64)
    sx, sy, sz = int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1, int(arr[:, 2].max()) + 1
    bi = np.zeros((sx, sy, sz), np.int32)
    bd = np.zeros((sx, sy, sz), np.int32)
    for x, y, z, bidx in arr.tolist():
        if 0 <= bidx < len(cv.block_index_to_pair):
            rid, rdata = cv.block_index_to_pair[bidx]
            bi[x, y, z] = rid
            bd[x, y, z] = rdata
    return Structure(block_ids=bi, block_data=bd).crop_to_non_air(), n_valid, n_lines


# --- dataset ---------------------------------------------------------------
def build_examples(structures, captions, tokenizer, cv, id_to_name,
                   max_tokens: int, limit: int | None):
    """Tokenized (prompt-masked) examples <= ``max_tokens`` total. Mirrors v1."""
    eos_id = tokenizer.eos_token_id
    examples = []
    n_considered = 0
    if limit is None:
        rng = range(len(structures))
    else:
        rng = sorted(random.Random(0).sample(range(len(structures)),
                                              min(limit, len(structures))))
    for i in rng:
        caps = captions.get(f"h{i:05d}")
        if not caps:
            continue
        completion = serialize_pieces(structures[i], cv, id_to_name)
        if not completion:
            continue
        n_considered += 1
        prompt = f"Build: {caps[0]}\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [eos_id]
        input_ids = prompt_ids + comp_ids
        if len(input_ids) > max_tokens:
            continue
        labels = [-100] * len(prompt_ids) + list(comp_ids)
        examples.append({"input_ids": input_ids, "labels": labels,
                         "caption": caps[0], "idx": i})
    return examples, n_considered


# --- main ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    # Pieces cost ~3-4x fewer tokens/block than voxels, so a smaller cap already
    # admits many builds; keep 2048 so bigger houses now fit that per-voxel could not.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piece-vocab", default=PIECE_VOCAB)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    run = new_run_dir("llm_pieces")
    print(f"[run] {run}", flush=True)

    structures, _ = load_house_structures(max_dim=32)
    captions = json.loads(Path(CAPTIONS_PATH).read_text())
    cv = load_piece_vocab(args.piece_vocab)
    id_to_name, name_to_id = build_piece_names(cv)
    print(f"[data] {len(structures)} structures; piece vocab {cv.n_pieces} pieces "
          f"({cv.num_blocks} atomic, {len(cv.merges)} merges), oriented={cv.oriented}", flush=True)

    # Persist the name map + which vocab it came from -- decoding is meaningless without it.
    (run / "piece_name_map.json").write_text(json.dumps(
        {"piece_vocab": os.path.abspath(args.piece_vocab),
         "id_to_name": id_to_name, "name_to_id": name_to_id}, indent=1))

    print(f"[model] loading tokenizer + {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    examples, n_considered = build_examples(
        structures, captions, tokenizer, cv, id_to_name, args.max_tokens, args.limit)
    print(f"[filter] {len(examples)}/{n_considered} builds <= {args.max_tokens} tokens "
          f"(vs 73 for per-voxel v1)", flush=True)
    if not examples:
        raise SystemExit("no examples after filtering")

    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(round(len(examples) * args.val_frac)))
    train = [examples[i] for i in order[n_val:]]
    val = [examples[i] for i in order[:n_val]]
    print(f"[split] train {len(train)}  val {len(val)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    lora = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)", flush=True)

    cfg = {"model": MODEL_NAME, "kind": "pieces", "piece_vocab": os.path.abspath(args.piece_vocab),
           "n_pieces": cv.n_pieces, "lora": {"r": 32, "alpha": 16},
           "max_tokens": args.max_tokens, "epochs": args.epochs, "lr": args.lr,
           "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
           "n_examples": len(examples), "n_train": len(train), "n_val": len(val),
           "n_considered": n_considered, "seed": args.seed}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))

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
                opt.step(); sched.step(); opt.zero_grad()
        if micro % args.grad_accum != 0:
            opt.step(); sched.step(); opt.zero_grad()
        tr = run_loss / max(run_n, 1)
        vl = eval_loss(model, val, pad_id, args.micro_batch, device)
        print(f"[epoch {epoch + 1}/{args.epochs}] train_loss {tr:.4f}  val_loss {vl:.4f}",
              flush=True)

    model.save_pretrained(str(run / "adapter"))
    print(f"[save] adapter -> {run / 'adapter'}", flush=True)

    n_samples = min(args.num_samples, len(val))
    picks = val[:n_samples]
    model.eval()
    model.config.use_cache = True
    sample_structs, prompts, parse_rates = [], [], []
    for ex in picks:
        prompt = f"Build: {ex['caption']}\n"
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = model.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                                 eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        struct, n_valid, n_lines = parse_piece_completion(text, cv, name_to_id)
        rate = n_valid / n_lines if n_lines else 0.0
        parse_rates.append(rate)
        sample_structs.append(struct)
        prompts.append(ex["caption"])
        print(f"[sample idx={ex['idx']}] blocks={int(struct.occupied_mask.sum())} "
              f"pieces={n_lines} valid={n_valid} parse_rate={rate:.2f}", flush=True)

    overall = float(np.mean(parse_rates)) if parse_rates else 0.0
    print(f"[parse] mean parse-success rate {overall:.3f}", flush=True)

    from blockgen.eval.cond_render import textured_prompt_grid
    out_png = run / "samples_textured.png"
    textured_prompt_grid(sample_structs, prompts, out_png,
                         suptitle=f"LoRA {MODEL_NAME} PIECES text->build  (mean parse {overall:.2f})")
    print(f"[render] {out_png}", flush=True)
    (run / "sample_report.json").write_text(json.dumps(
        {"mean_parse_rate": overall,
         "samples": [{"idx": ex["idx"], "caption": ex["caption"],
                      "parse_rate": r, "blocks": int(s.occupied_mask.sum())}
                     for ex, r, s in zip(picks, parse_rates, sample_structs)]}, indent=1))
    print(f"[done] run dir: {run}", flush=True)


if __name__ == "__main__":
    main()
