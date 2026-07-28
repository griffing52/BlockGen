"""Diagnose the filament failure mode of the attachment/growth MVP.

Symptom (run `20260721_071702`, arm `bfs_bottom_center`): val NLL is excellent
(1.077 bits/op) but free-running samples are 1-voxel-wide tendrils, not volumes.
Round-trip IoU is 1.0 on 1,200 real builds, so the representation and decoder are
correct -- this is a *generation* failure.

Mechanism under test. Every op is a per-face binary-ish choice between CLOSE and
ATTACH. A build is ~2,000 such decisions, and solidity requires ATTACHing on many
faces. If free-running P(ATTACH) is even slightly under-calibrated, the error
compounds multiplicatively and a 3D volume collapses into a 1D filament. Two
candidate causes, distinguished here:

  (a) **Calibration** -- the marginal is close but sampling noise pushes it down.
      Then temperature should move attach-rate and solidity a lot, and some
      temperature should produce volumes.
  (b) **Structural** -- the model cannot tell where it is. The token-sequence MVP
      sees only its own op history, never the partial *geometry*, so after a few
      hundred ops it has no idea whether it is mid-wall or mid-air. Then no
      temperature helps, and the fix is the graph encoder of
      `implementation_plan.md` §3 -- i.e. the encoder is a REQUIREMENT, not the
      "capacity upgrade" the Phase-1 MVP assumed it could defer.

Reports, per temperature: attach rate, mean occupancy, component count, and
**thickness** (mean count of occupied 6-neighbours) -- the statistic that separates
a filament (~2) from a wall or volume (~4-5). Training-set thickness is printed as
the reference.

Usage:
    python -m scripts.attach_diagnose --arm outputs/run_.../bfs_bottom_center
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.curation.houses import load_structures_from_cache
from blockgen.eval.validity import n_components
from blockgen.utils.attach_order import (OP_ATTACH, OP_CLOSE, OP_SEED,
                                         attach_ops_to_structure,
                                         structure_to_attach_ops)
from blockgen.utils.attach_vocab import BOS_ID, EOS_ID, AttachVocab
from blockgen.utils.graph_data import PORT_DIRECTIONS


def thickness(occ: np.ndarray) -> float:
    """Mean occupied 6-neighbour count. Filament ~2, wall ~4, solid ~5-6.

    Implemented with **zero padding, not np.roll**. `np.roll` wraps, so voxels on
    opposite faces of the array count as neighbours — and because builds are cropped
    to their bounding box, boundary voxels are ALWAYS on those faces. That inflated
    every thickness reading, worst for thin structures spanning a large box (a
    1-voxel diagonal line scored ~3.9 instead of ~2). Verified: two voxels at
    opposite corners now score 0.0, previously 1.0.
    """
    if occ.sum() == 0:
        return 0.0
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    nb = np.zeros(occ.shape, dtype=np.int32)
    for dx, dy, dz in PORT_DIRECTIONS:
        sl = tuple(slice(1 + d, 1 + d + n) for d, n in zip((dx, dy, dz), occ.shape))
        nb += padded[sl].astype(np.int32)
    return float(nb[occ].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm dir with model.pt + attach_vocab.json")
    ap.add_argument("--temps", default="0.5,0.7,0.85,1.0")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--ref-n", type=int, default=300,
                    help="corpus builds used for reference statistics")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arm = Path(args.arm)
    vocab = AttachVocab.load(arm / "attach_vocab.json")
    blob = torch.load(arm / "model.pt", map_location=args.device, weights_only=False)
    cfg = blob["config"]
    ordering = blob["ordering"]

    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    model = VoxelTransformerAR2(
        vocab_size=blob["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], pe="sin",
    ).to(args.device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    # Reference statistics from real builds under the SAME ordering.
    #
    # Reported BOTH unfiltered and restricted to builds this arm could actually train
    # on (seq <= max_seq_len). The filter is NOT neutral: it discards large builds
    # specifically, halving median occupancy (1058 -> 755 on all_32 @ 4096) while
    # leaving thickness untouched. Comparing generated size against the *unfiltered*
    # figure overstates the model's deficit, which is a mistake this script previously
    # invited -- so the filtered row is the one to judge size against.
    structures, _ = load_structures_from_cache(args.cache)
    max_len = cfg["max_seq_len"]
    ref_thick, ref_occ, ref_attach = [], [], []
    fit_thick, fit_occ = [], []
    for s in structures[:args.ref_n]:
        try:
            ops, info = structure_to_attach_ops(s, ordering=ordering)
        except Exception:
            continue
        c = s.crop_to_non_air()
        t, o = thickness(c.occupied_mask), int(c.occupied_mask.sum())
        ref_thick.append(t)
        ref_occ.append(o)
        ref_attach.append(info["n_attach"] / max(1, info["n_ops"]))
        if info["n_ops"] + 2 <= max_len:
            fit_thick.append(t)
            fit_occ.append(o)
    print(f"[reference] ALL builds        n={len(ref_occ):4d} "
          f"thickness={np.mean(ref_thick):.2f} occ_p50={np.percentile(ref_occ,50):.0f} "
          f"attach_rate={np.mean(ref_attach):.3f}", flush=True)
    if fit_occ:
        print(f"[reference] TRAINABLE @{max_len} n={len(fit_occ):4d} "
              f"thickness={np.mean(fit_thick):.2f} occ_p50={np.percentile(fit_occ,50):.0f} "
              f"({100*len(fit_occ)/max(1,len(ref_occ)):.0f}% kept)  <-- judge SIZE vs this",
              flush=True)
    print(flush=True)

    rows = []
    for temp in [float(t) for t in args.temps.split(",")]:
        th, oc, cp, ar, ns = [], [], [], [], []
        for _ in range(args.n):
            with torch.no_grad():
                toks = model.generate(bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                                      max_new_tokens=cfg["max_seq_len"],
                                      temperature=temp, device=args.device)
            if isinstance(toks, torch.Tensor):
                toks = toks.flatten().tolist()
            ops = vocab.ids_to_ops(toks)
            if not ops:
                continue
            n_at = sum(1 for o in ops if o.kind == OP_ATTACH)
            n_cl = sum(1 for o in ops if o.kind == OP_CLOSE)
            ar.append(n_at / max(1, n_at + n_cl))
            ns.append(sum(1 for o in ops if o.kind == OP_SEED))
            s = attach_ops_to_structure(ops, ordering=ordering, oriented=vocab.oriented)
            occ = s.occupied_mask
            if occ.sum() == 0:
                continue
            th.append(thickness(occ))
            oc.append(int(occ.sum()))
            cp.append(n_components(s))
        row = {"temp": temp,
               "attach_rate": float(np.mean(ar)) if ar else 0.0,
               "thickness": float(np.mean(th)) if th else 0.0,
               "occ": float(np.mean(oc)) if oc else 0.0,
               "components": float(np.mean(cp)) if cp else 0.0,
               "seeds": float(np.mean(ns)) if ns else 0.0}
        rows.append(row)
        print(f"T={temp:4.2f}  attach_rate={row['attach_rate']:.3f}  "
              f"thickness={row['thickness']:.2f}  occ={row['occ']:7.0f}  "
              f"comps={row['components']:5.2f}  seeds={row['seeds']:.2f}", flush=True)

    ref = {"thickness": float(np.mean(ref_thick)),
           "attach_rate": float(np.mean(ref_attach)),
           "occ_p50": float(np.percentile(ref_occ, 50))}
    (arm / "diagnose.json").write_text(json.dumps({"reference": ref, "sweep": rows}, indent=2))

    best = max(rows, key=lambda r: r["thickness"]) if rows else None
    print()
    if best and best["thickness"] < 0.75 * ref["thickness"]:
        print("VERDICT: STRUCTURAL. No temperature recovers real-build thickness.")
        print("  The op-token MVP never observes the partial geometry, only its own op")
        print("  history, so it cannot tell mid-wall from mid-air. The graph encoder of")
        print("  implementation_plan.md §3 is a REQUIREMENT, not a capacity upgrade.")
    else:
        print("VERDICT: CALIBRATION. Temperature materially moves thickness --")
        print("  tune sampling before redesigning the model.")


if __name__ == "__main__":
    main()
