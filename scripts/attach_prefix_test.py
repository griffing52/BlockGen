"""Does the growth model collapse because it can't SEE, or because it drifts?

Open question after T21: every arm degenerates (lines, plates, balls) and no metric
caught it. Two competing explanations, with very different consequences:

  (a) **Exposure bias / drift.** The model has learned real structure but, free-running
      over ~2,000 ops, accumulates error and slides into a low-entropy attractor (a line
      is the most compressible op stream, so NLL actively rewards it). Fix: scheduled
      sampling, better decoding, sequence-level objectives. Architecture is fine.
  (b) **Blindness.** The op-token MVP sees only its own op history, never the partial
      *geometry*, so it cannot tell mid-wall from mid-air at any point, prefix or not.
      Fix: the graph/state encoder of `implementation_plan.md` §3 — which this run
      previously (and wrongly, on a buggy thickness metric) claimed was unnecessary.

**The test.** Teacher-force the first K ops of a REAL build, then let the model continue
freely. Under (a) a real prefix should hold the model on-distribution for a while and
continuation quality should rise with K. Under (b) the prefix should not help much at any
K — the model has no way to read it as geometry.

Reports, per K, over real val builds: continuation thickness (corrected metric) and
occupancy vs the ground-truth completion of the same build, plus the fraction of the
continuation that is CLOSE (a runaway CLOSE rate is the line-mode signature).

Usage:
    python -m scripts.attach_prefix_test --arm outputs/run_.../bfs_bottom_center
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.curation.houses import load_structures_from_cache
from blockgen.utils.attach_order import (OP_ATTACH, OP_CLOSE, attach_ops_to_structure,
                                         structure_to_attach_ops)
from blockgen.utils.attach_vocab import BOS_ID, EOS_ID, AttachVocab
from scripts.attach_diagnose import thickness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fracs", default="0.0,0.1,0.25,0.5",
                    help="prefix lengths as a fraction of the build's true op count")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arm = Path(args.arm)
    vocab = AttachVocab.load(arm / "attach_vocab.json")
    blob = torch.load(arm / "model.pt", map_location=args.device, weights_only=False)
    cfg, ordering = blob["config"], blob["ordering"]

    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    model = VoxelTransformerAR2(
        vocab_size=blob["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], pe="sin",
    ).to(args.device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    structures, _ = load_structures_from_cache(args.cache)
    # Held-out-ish tail of the corpus, and only builds this arm could train on.
    pool = []
    skipped_oov = 0
    for s in reversed(structures):
        if len(pool) >= args.n:
            break
        try:
            ops, info = structure_to_attach_ops(s, ordering=ordering)
        except Exception:
            continue
        if not (info["n_ops"] + 2 <= cfg["max_seq_len"] and info["n_ops"] > 200):
            continue
        try:
            ids = vocab.ops_to_ids(ops)
        except KeyError:
            # The arm's vocab was built from a --vocab-limit subset, so corpus-tail
            # builds can contain unseen pieces. Skip them: the model has no embedding
            # for such a token, so a prefix containing one is not a valid probe.
            skipped_oov += 1
            continue
        pool.append((s, ids))
    if skipped_oov:
        print(f"[prefix] skipped {skipped_oov} builds with out-of-vocab pieces", flush=True)

    ref_thick = float(np.mean([thickness(s.crop_to_non_air().occupied_mask)
                               for s, _ in pool]))
    ref_occ = float(np.median([int(s.crop_to_non_air().occupied_mask.sum())
                               for s, _ in pool]))
    print(f"[prefix] {len(pool)} real builds  thickness={ref_thick:.2f}  "
          f"occ_p50={ref_occ:.0f}\n", flush=True)

    rows = []
    for frac in [float(f) for f in args.fracs.split(",")]:
        th, oc, closerate = [], [], []
        for _s, ids in pool:
            k = max(1, int(frac * (len(ids) - 2))) + 1  # +1 keeps BOS
            prefix = ids[:k]
            inp = torch.tensor(prefix, dtype=torch.long, device=args.device).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(
                    bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                    max_new_tokens=cfg["max_seq_len"] - len(prefix),
                    temperature=args.temp, device=args.device, prompt=inp)
            if isinstance(out, torch.Tensor):
                out = out.flatten().tolist()
            cont = out[len(prefix):]
            cont_ops = vocab.ids_to_ops(cont)
            if cont_ops:
                n_cl = sum(1 for o in cont_ops if o.kind == OP_CLOSE)
                closerate.append(n_cl / len(cont_ops))
            full = vocab.ids_to_ops(out)
            if not full:
                continue
            st = attach_ops_to_structure(full, ordering=ordering, oriented=vocab.oriented)
            if int(st.occupied_mask.sum()) == 0:
                continue
            th.append(thickness(st.occupied_mask))
            oc.append(int(st.occupied_mask.sum()))
        row = {"prefix_frac": frac,
               "thickness": float(np.mean(th)) if th else 0.0,
               "occ_p50": float(np.median(oc)) if oc else 0.0,
               "close_rate": float(np.mean(closerate)) if closerate else 0.0,
               "n": len(th)}
        rows.append(row)
        print(f"prefix={frac:4.2f}  thickness={row['thickness']:5.2f}  "
              f"occ_p50={row['occ_p50']:7.0f}  close_rate={row['close_rate']:.3f}  "
              f"n={row['n']}", flush=True)

    (arm / "prefix_test.json").write_text(json.dumps(
        {"reference": {"thickness": ref_thick, "occ_p50": ref_occ}, "rows": rows}, indent=2))

    base, best = rows[0]["thickness"], max(r["thickness"] for r in rows)
    print()
    print(f"reference thickness {ref_thick:.2f}; free-running {base:.2f}; "
          f"best-with-prefix {best:.2f}")
    if best >= base + 0.5 and best >= 0.85 * ref_thick:
        print("VERDICT: DRIFT (a). A real prefix substantially rescues structure, so the")
        print("  model HAS learned it and loses it while free-running. Attack exposure")
        print("  bias (scheduled sampling / sequence-level objective) before adding an encoder.")
    else:
        print("VERDICT: BLINDNESS (b). A real prefix does NOT rescue structure — the model")
        print("  cannot read its own history as geometry. implementation_plan.md §3's state/")
        print("  graph encoder is then a REQUIREMENT, not a capacity upgrade.")


if __name__ == "__main__":
    main()
