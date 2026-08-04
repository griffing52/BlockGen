"""Sample a served checkpoint into the standard structure-cache `.npz`.

`python -m blockgen.eval.bench` consumes structure caches only -- it never loads
a checkpoint -- so a neural arm needs one conversion step before it can be
benchmarked. The agentic track already writes this format; this script closes the
gap for the AR tracks.

Architecture comes from `deploy/inference/models.json`, which already records
`d_model`/`layers`/`nhead`/`dim_feedforward`/`pe`/`max_seq_len` per checkpoint and
is cross-checked against the weights on load. Re-declaring those constants here
would be a second place to get them wrong.

    python scripts/sample_to_npz.py --model native_oriented --n 64 \\
        --out-dir outputs/bench_samples --name native_oriented

Generation is autoregressive without a KV cache (`train_ar_ext.generate_from_prefix`
re-runs the whole prefix each step), so cost grows quadratically in sequence
length. The native BPE arm has a median sequence of ~1750 piece tokens, so budget
accordingly and prefer running this in the background.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

MODELS_JSON = Path("deploy/inference/models.json")


def load_spec(name: str) -> dict:
    spec = json.loads(MODELS_JSON.read_text())["models"].get(name)
    if spec is None:
        avail = sorted(json.loads(MODELS_JSON.read_text())["models"])
        raise SystemExit(f"unknown model {name!r}; available: {avail}")
    return spec


def build_model(spec: dict, vocab_size: int, device: str):
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    model = VoxelTransformerAR2(
        vocab_size=vocab_size, max_seq_len=spec["max_seq_len"],
        d_model=spec["d_model"], nhead=spec["nhead"],
        num_layers=spec["layers"], dim_feedforward=spec["dim_feedforward"],
        pe=spec.get("pe", "phase4"))
    sd = torch.load(spec["checkpoint"], map_location=device, weights_only=True)
    lm = sd["lm_head.weight"].shape[0]
    if lm != vocab_size:
        raise SystemExit(
            f"vocab mismatch: checkpoint lm_head is {lm}, vocab is {vocab_size}. "
            "The piece vocab does not belong to this checkpoint.")
    model.load_state_dict(sd)
    return model.to(device).eval()


def sample_piece_ar(spec: dict, n: int, temperature: float, top_k: int,
                    device: str, seed: int, verbose: bool = True) -> List:
    from blockgen.export.minecraftace import load_piece_vocab
    from blockgen.tokenizers.cluster_bpe import cluster_tokens_to_structure
    from blockgen.training.train_ar_ext import (BOS_TOKEN, EOS_TOKEN,
                                                generate_from_prefix)

    cv = load_piece_vocab(spec["piece_vocab"])
    model = build_model(spec, cv.vocab_size, device)
    torch.manual_seed(seed)

    out, t0 = [], time.time()
    for i in range(n):
        toks = generate_from_prefix(model, [BOS_TOKEN], EOS_TOKEN,
                                    max_new_tokens=model.max_seq_len - 1,
                                    temperature=temperature, top_k=top_k)
        out.append(cluster_tokens_to_structure(toks, cv))
        if verbose and (i + 1) % 4 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i + 1}/{n}  {rate:.1f}s/sample  "
                  f"eta {rate * (n - i - 1) / 60:.1f} min", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="key in deploy/inference/models.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-dim", type=int, default=32)
    args = ap.parse_args()

    spec = load_spec(args.model)
    if spec["kind"] != "piece_ar":
        raise SystemExit(f"only piece_ar is supported here, got {spec['kind']!r}")

    print(f"[sample] {args.model}: {spec['checkpoint']}")
    t0 = time.time()
    structs = sample_piece_ar(spec, args.n, args.temperature, args.top_k,
                              args.device, args.seed)
    structs = [s.crop_to_non_air() for s in structs]
    n_empty = sum(1 for s in structs if not s.occupied_mask.any())
    occ = [int(s.occupied_mask.sum()) for s in structs]
    print(f"[sample] {len(structs)} samples in {(time.time() - t0) / 60:.1f} min; "
          f"{n_empty} empty; median blocks {int(np.median(occ))}")

    from blockgen.curation.houses import save_house_cache
    name = args.name or args.model
    path = save_house_cache(
        structs, args.max_dim, cache_dir=args.out_dir, name=name,
        report={"model": args.model, "checkpoint": spec["checkpoint"],
                "n": len(structs), "n_empty": n_empty, "seed": args.seed,
                "temperature": args.temperature, "top_k": args.top_k})
    print(f"bench with:\n  python -m blockgen.eval.bench --arms ar/{name}:{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
