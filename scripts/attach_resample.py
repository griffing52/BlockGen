"""Re-sample trained attachment/growth arms at a chosen temperature, and render.

Why this exists. The Phase-1 arms sampled at temperature 1.0 and produced 1-voxel
filaments; `scripts/attach_diagnose.py` showed the cause is *calibration*, not model
capacity -- lower temperature recovers (and at 0.5 exceeds) real-build thickness. The
training runs also predate the sparse-decoder fix, so some of their samples were lost
to an out-of-grid IndexError. This regenerates samples for every arm with the fixed
decoder at a sane temperature, so the morning comparison is on quality rather than on
a sampling artifact.

Reports per arm: attach rate, thickness (filament ~2, wall ~4, solid ~5+), occupancy,
components, and LEARNED (unmasked) validity.

Usage:
    python -m scripts.attach_resample --run outputs/run_..._attach_growth_ordering --temp 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.eval.validity import n_components
from blockgen.utils.attach_order import OP_ATTACH, OP_CLOSE, OP_SEED, attach_ops_to_structure
from blockgen.utils.attach_vocab import BOS_ID, EOS_ID, AttachVocab
from scripts.attach_diagnose import thickness


def resample_arm(arm: Path, temp: float, n: int, device: str) -> dict | None:
    if not (arm / "model.pt").exists():
        return None
    vocab = AttachVocab.load(arm / "attach_vocab.json")
    blob = torch.load(arm / "model.pt", map_location=device, weights_only=False)
    cfg, ordering = blob["config"], blob["ordering"]

    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    model = VoxelTransformerAR2(
        vocab_size=blob["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], pe="sin",
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    samples, th, ar, seeds = [], [], [], []
    for _ in range(n):
        with torch.no_grad():
            toks = model.generate(bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                                  max_new_tokens=cfg["max_seq_len"],
                                  temperature=temp, device=device)
        if isinstance(toks, torch.Tensor):
            toks = toks.flatten().tolist()
        ops = vocab.ids_to_ops(toks)
        if not ops:
            continue
        n_at = sum(1 for o in ops if o.kind == OP_ATTACH)
        n_cl = sum(1 for o in ops if o.kind == OP_CLOSE)
        ar.append(n_at / max(1, n_at + n_cl))
        seeds.append(sum(1 for o in ops if o.kind == OP_SEED))
        s = attach_ops_to_structure(ops, ordering=ordering, oriented=vocab.oriented)
        if int(s.occupied_mask.sum()) == 0:
            continue
        th.append(thickness(s.occupied_mask))
        samples.append(s)

    if not samples:
        return {"ordering": ordering, "error": "no samples"}
    occ = [int(s.occupied_mask.sum()) for s in samples]
    comps = [n_components(s) for s in samples]
    out = {
        "ordering": ordering, "temp": temp, "n_samples": len(samples),
        "attach_rate": float(np.mean(ar)),
        "thickness": float(np.mean(th)),
        "occ_p50": float(np.percentile(occ, 50)),
        "components_mean": float(np.mean(comps)),
        "seeds_mean": float(np.mean(seeds)),
        "validity_learned_unmasked": float(np.mean([c == 1 for c in comps])),
    }
    tag = f"T{temp:g}".replace(".", "")
    (arm / f"resample_{tag}.json").write_text(json.dumps(out, indent=2))
    try:
        from blockgen.eval.cond_render import textured_prompt_grid
        textured_prompt_grid(
            samples[:16], [f"{ordering} #{i}" for i in range(len(samples[:16]))],
            arm / f"samples_{tag}.png", cols=4,
            suptitle=f"attach-growth :: {ordering} :: T={temp} :: thickness {out['thickness']:.2f}")
    except Exception as exc:
        print(f"  render failed: {exc}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run = Path(args.run)
    rows = []
    for arm in sorted(p for p in run.iterdir() if p.is_dir()):
        print(f"[resample] {arm.name} @ T={args.temp}", flush=True)
        r = resample_arm(arm, args.temp, args.n, args.device)
        if r:
            rows.append(r)
            print(f"  thickness={r.get('thickness',0):.2f} occ={r.get('occ_p50',0):.0f} "
                  f"valid={r.get('validity_learned_unmasked',0):.2f} "
                  f"seeds={r.get('seeds_mean',0):.2f}", flush=True)

    (run / f"resample_T{args.temp:g}".replace(".", "") ).with_suffix(".json").write_text(
        json.dumps(rows, indent=2))
    # Reference is the TRAINABLE slice (seq <= max_seq_len), not the whole corpus: the
    # length filter removes large builds specifically, so the unfiltered occ_p50 (~1000+)
    # is not the target these arms were trained toward. See results.md T21.
    print("\n=== RESAMPLE SUMMARY (trainable-slice reference @4096: "
          "thickness 4.20, occ_p50 806) ===")
    for r in sorted(rows, key=lambda r: -r.get("thickness", 0)):
        print(f"  {r['ordering']:20s} thick={r.get('thickness',0):5.2f} "
              f"occ={r.get('occ_p50',0):6.0f} valid={r.get('validity_learned_unmasked',0):.2f} "
              f"seeds={r.get('seeds_mean',0):.2f}")


if __name__ == "__main__":
    main()
