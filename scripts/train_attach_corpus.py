"""Phase-1 MVP: train the attachment/growth AR model, one arm per ordering.

The growth model's op stream is a token sequence (``utils/attach_vocab``), so this
reuses the existing AR stack unchanged -- ``VoxelTransformerAR2`` +
``train_ar_ext.train_from_sequences``. All the geometry lives in the decoder
(``attach_order.attach_ops_to_structure``), which replays the frontier and derives
pose from each connection. Consequences worth stating up front:

  * **Validity should be ~1.0 by construction** for every arm. That is NOT evidence
    the model learned anything (implementation_plan.md §10, the artifact confound) --
    it is a property of the decoder. The number that carries information here is
    sample *quality*, and the ordering comparison.
  * **There is no adjacency gate and no coordinate to get wrong.** The model chooses
    only CLOSE vs (piece, direction) per open face.

Arms are orderings. This is the experiment implementation_plan.md §2 assumes away:
it asserts bottom-center BFS as canonical while notes.md §8/T11 measured BFS-from-
ground as the worst non-broken arm for raster AR. Same corpus, same budget, same
eval -- only the linearization changes.

Usage:
    python -m scripts.train_attach_corpus --orderings bfs_bottom_center,layered_raster \
        --epochs 30 --limit 4000
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch

from blockgen.curation.houses import load_structures_from_cache
from blockgen.eval.validity import n_components
from blockgen.training.train_ar import ARTrainConfig
from blockgen.training.train_ar_ext import train_from_sequences
from blockgen.utils.attach_order import ORDERINGS, structure_to_attach_ops
from blockgen.utils.attach_vocab import (BOS_ID, EOS_ID, AttachVocab,
                                         build_sequences, build_vocab)
from blockgen.utils.runs import new_run_dir


def sample_structures(model, vocab: AttachVocab, *, n: int, ordering: str,
                      max_new: int, device: str, temperature: float = 1.0):
    """Generate op streams and decode them. Connectivity holds by construction."""
    from blockgen.utils.attach_order import attach_ops_to_structure

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(n):
            try:
                toks = model.generate(
                    bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                    max_new_tokens=max_new, temperature=temperature,
                    device=device)
                if isinstance(toks, torch.Tensor):
                    toks = toks.flatten().tolist()
                ops = vocab.ids_to_ops(toks)
                if not ops:
                    continue
                s = attach_ops_to_structure(ops, ordering=ordering,
                                            oriented=vocab.oriented)
                if int(s.occupied_mask.sum()) > 0:
                    out.append(s)
            except Exception as exc:  # keep one bad sample from killing the arm
                print(f"[sample] {i} failed: {exc}", flush=True)
    return out


def run_arm(ordering: str, structures, args, run_dir: Path) -> dict:
    arm_dir = run_dir / ordering
    arm_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== arm: {ordering} ===", flush=True)
    t0 = time.time()

    vocab = build_vocab(structures, ordering=ordering, oriented=args.oriented,
                        limit=args.vocab_limit)
    # Save BEFORE training: notes.md §18(b) burned two unloadable checkpoints here.
    vocab.save(arm_dir / "attach_vocab.json")
    print(f"[{ordering}] vocab: {len(vocab.pieces)} pieces -> {vocab.size} tokens",
          flush=True)

    seqs = build_sequences(structures, vocab, ordering=ordering,
                           max_seq_len=args.max_seq_len)
    kept = len(seqs) / max(1, len(structures))
    if not seqs:
        print(f"[{ordering}] NO sequences fit max_seq_len={args.max_seq_len}", flush=True)
        return {"ordering": ordering, "error": "no sequences fit",
                "frac_fit": 0.0, "max_seq_len": args.max_seq_len}
    lens = [len(s) for s in seqs]
    print(f"[{ordering}] {len(seqs)}/{len(structures)} builds fit "
          f"max_seq_len={args.max_seq_len} ({kept:.1%}); "
          f"len p50={np.percentile(lens,50):.0f} p90={np.percentile(lens,90):.0f}",
          flush=True)

    n_val = max(1, int(0.1 * len(seqs)))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(seqs))
    val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:].tolist()
    train_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in sorted(val_idx)]

    cfg = ARTrainConfig(
        max_seq_len=args.max_seq_len, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.d_model * 4,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, amp=True)

    model, history = train_from_sequences(train_seqs, vocab_size=vocab.size,
                                          config=cfg, pe="sin")

    torch.save({"state_dict": model.state_dict(), "vocab_size": vocab.size,
                "ordering": ordering, "config": cfg.__dict__},
               arm_dir / "model.pt")

    # Held-out NLL: the primary cross-arm comparison. Same corpus, same budget,
    # same vocab construction -- only the linearization differs, so bits/op here is
    # directly comparable across orderings.
    val_nll = _eval_nll(model, val_seqs, cfg, args.device)
    print(f"[{ordering}] val NLL = {val_nll:.4f} nats/op "
          f"({val_nll / np.log(2):.4f} bits/op)", flush=True)

    samples = sample_structures(model, vocab, n=args.n_samples, ordering=ordering,
                                max_new=args.max_seq_len, device=args.device,
                                temperature=args.temperature)
    print(f"[{ordering}] generated {len(samples)}/{args.n_samples} samples", flush=True)

    occ = [int(s.occupied_mask.sum()) for s in samples]
    comps = [n_components(s) for s in samples]
    # IMPORTANT (found 2026-07-21): validity is NOT 1.0 "by construction". Pose is
    # derived from the connection, so no single component can be disconnected --
    # but nothing stops the model emitting a second SEED, which starts a NEW
    # component. So multi-component samples are exactly multi-SEED samples.
    #
    # This is the artifact confound (implementation_plan.md §10) in miniature:
    # banning SEED after position 0 would force validity to 1.0 as a decode-time
    # FILTER. We deliberately do not mask, so the number below is the LEARNED
    # validity -- the DiGress-style ablation, obtained for free.
    validity = float(np.mean([c == 1 for c in comps])) if comps else 0.0

    result = {
        "ordering": ordering,
        "n_pieces": len(vocab.pieces),
        "vocab_size": vocab.size,
        "n_seqs": len(seqs),
        "frac_fit": kept,
        "seq_p50": float(np.percentile(lens, 50)),
        "seq_p90": float(np.percentile(lens, 90)),
        "val_nll_nats": val_nll,
        "val_bits_per_op": val_nll / float(np.log(2)),
        "final_train_loss": history.get("loss", [None])[-1] if isinstance(history, dict) else None,
        "n_samples": len(samples),
        "sample_occ_mean": float(np.mean(occ)) if occ else 0.0,
        "sample_occ_p50": float(np.percentile(occ, 50)) if occ else 0.0,
        "sample_components_mean": float(np.mean(comps)) if comps else 0.0,
        # Learned (UNMASKED) validity -- see the note above; not a by-construction 1.0.
        "validity_learned_unmasked": validity,
        "validity_by_construction": validity,
        "secs": round(time.time() - t0, 1),
    }
    (arm_dir / "result.json").write_text(json.dumps(result, indent=2))

    # Every training run renders its samples (project standard).
    if samples:
        try:
            from blockgen.eval.cond_render import textured_prompt_grid
            textured_prompt_grid(
                samples[:16], [f"{ordering} #{i}" for i in range(len(samples[:16]))],
                arm_dir / "samples.png", cols=4,
                suptitle=f"attach-growth :: {ordering} :: val {result['val_bits_per_op']:.3f} bits/op")
            print(f"[{ordering}] rendered {arm_dir/'samples.png'}", flush=True)
        except Exception as exc:
            print(f"[{ordering}] render failed: {exc}", flush=True)
        np.savez_compressed(arm_dir / "samples.npz",
                            **{f"s{i}": s.block_ids for i, s in enumerate(samples)})
    return result


def _eval_nll(model, seqs: List[List[int]], cfg: ARTrainConfig, device: str) -> float:
    """Mean per-op cross-entropy on held-out sequences."""
    import torch.nn.functional as F

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in seqs:
            if len(s) < 2:
                continue
            x = torch.tensor(s[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(s[1:], dtype=torch.long, device=device).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device != "cpu"):
                logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum")
            total += float(loss)
            count += y.numel()
    return total / max(1, count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--orderings", default="bfs_bottom_center,layered_raster,radial,dfs")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--vocab-limit", type=int, default=1500)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oriented", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="attach_growth")
    args = ap.parse_args()

    run_dir = new_run_dir(args.name)
    print(f"[attach] run dir: {run_dir}", flush=True)

    structures, _m = load_structures_from_cache(args.cache)
    structures = structures[:args.limit]
    print(f"[attach] {len(structures)} structures", flush=True)

    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]
    for o in orderings:
        if o not in ORDERINGS:
            raise SystemExit(f"unknown ordering {o}; have {sorted(ORDERINGS)}")

    results = []
    for o in orderings:
        try:
            results.append(run_arm(o, structures, args, run_dir))
        except Exception:
            traceback.print_exc()
            results.append({"ordering": o, "error": traceback.format_exc(limit=3)})
        (run_dir / "summary.json").write_text(json.dumps(
            {"args": vars(args), "results": results}, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    ok = [r for r in results if "val_bits_per_op" in r]
    for r in sorted(ok, key=lambda r: r["val_bits_per_op"]):
        print(f"  {r['ordering']:20s} val={r['val_bits_per_op']:.4f} bits/op  "
              f"fit={r['frac_fit']:.1%}  occ_p50={r['sample_occ_p50']:.0f}  "
              f"valid={r['validity_by_construction']:.2f}", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['ordering']:20s} ERROR", flush=True)
    print(f"[attach] wrote {run_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
