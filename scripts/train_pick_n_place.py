"""Train the pick-and-place growth model, sample, render, and dump for the bench.

    .venv/bin/python scripts/train_pick_n_place.py --quick        # ~2 min smoke
    .venv/bin/python scripts/train_pick_n_place.py --epochs 60 --max-nodes 256

One command produces a full run directory: checkpoint, palette, metrics, a
textured sample sheet, and a structure cache ready for
`python -m blockgen.eval.bench`.

Read `place_lift` first. The placer picks among *legal* faces, so its accuracy
has a floor set by the mask, not by zero -- `place_lift` is accuracy divided by
the 1/n_legal chance baseline. A lift near 1.0 means the model has learned
nothing the mask did not already give it, whatever the raw accuracy looks like.
That distinction is the whole reason T21's headline numbers had to be retracted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blockgen.eval.bench import splits
from blockgen.models.pick_n_place import (PickAndPlace, PickAndPlaceConfig, generate,
                                          rollout_to_structure)
from blockgen.training.train_pick_n_place import (GrowthDataset, PieceCodec, TrainConfig,
                                                  evaluate, train)
from blockgen.utils.growth_order import growth_stats, structure_to_growth
from blockgen.utils.runs import new_run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-nodes", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="cap training builds")
    ap.add_argument("--vocab-limit", type=int, default=256)
    ap.add_argument("--ordering", default="bfs", choices=("bfs", "layered", "dfs"))
    ap.add_argument("--oriented", action="store_true")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--rel-clamp", type=int, default=4)
    ap.add_argument("--place-weight", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-extent", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="pick_n_place")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run: 3 epochs, 200 builds, 96 nodes")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.limit, args.max_nodes = 3, 200, 96
        args.d_model, args.layers, args.samples = 128, 3, 6

    run = new_run_dir(args.name)
    print(f"[run] {run}", flush=True)
    (run / "cmd.txt").write_text(json.dumps(vars(args), indent=2) + "\n")

    split = splits.load_split(args.corpus, args.seed)
    train_s = splits.split_structures(split, "train")
    val_s = splits.split_structures(split, "val")
    if args.limit:
        train_s = train_s[:args.limit]
        val_s = val_s[:max(8, args.limit // 8)]
    print(f"[data] {len(train_s)} train / {len(val_s)} val  ({split.key()})",
          flush=True)

    # Phase 0 first: if the representation does not round-trip, nothing
    # downstream means anything (T21's gate).
    gs = growth_stats(train_s[:100], ordering=args.ordering, oriented=args.oriented)
    print(f"[repr] roundtrip IoU min {gs['min_roundtrip_iou']:.4f} "
          f"mean {gs['mean_roundtrip_iou']:.4f} | block retention "
          f"{gs['block_retention']:.3f} | median nodes {gs['median_nodes']:.0f}",
          flush=True)
    if not gs["min_roundtrip_iou"] >= 1.0 - 1e-9:
        raise SystemExit("representation does not round-trip; fix that first")

    seqs = [s for s in (structure_to_growth(x, ordering=args.ordering,
                                            oriented=args.oriented,
                                            max_nodes=args.max_nodes)
                        for x in train_s) if s is not None]
    codec = PieceCodec.from_sequences(seqs, limit=args.vocab_limit)
    train_ds = GrowthDataset(train_s, codec, max_nodes=args.max_nodes,
                             ordering=args.ordering, oriented=args.oriented)
    val_ds = GrowthDataset(val_s, codec, max_nodes=args.max_nodes,
                           ordering=args.ordering, oriented=args.oriented)
    print(f"[dataset] palette {codec.n_pieces} | train {train_ds.stats()} | "
          f"val {val_ds.stats()}", flush=True)
    if len(train_ds) == 0:
        raise SystemExit("no usable training sequences -- raise --vocab-limit")

    cfg = PickAndPlaceConfig(n_pieces=codec.n_pieces, d_model=args.d_model,
                             nhead=args.nhead, num_layers=args.layers,
                             rel_clamp=args.rel_clamp, max_nodes=args.max_nodes)
    model = PickAndPlace(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par/1e6:.2f}M params", flush=True)

    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                       place_weight=args.place_weight, device=args.device)
    t0 = time.time()
    model, hist = train(model, train_ds, val_ds, tcfg)
    train_min = (time.time() - t0) / 60
    torch.save(model.state_dict(), run / "model.pt")
    (run / "palette.json").write_text(json.dumps(codec.to_json()) + "\n")

    from torch.utils.data import DataLoader
    from blockgen.training.train_pick_n_place import collate
    val_metrics = evaluate(model, DataLoader(val_ds, batch_size=args.batch,
                                             collate_fn=collate), tcfg) \
        if len(val_ds) else {}
    print(f"[train] {args.epochs} epochs in {train_min:.1f} min; val {val_metrics}",
          flush=True)

    print(f"[sample] generating {args.samples}...", flush=True)
    rolls, structs = [], []
    for i in range(args.samples):
        r = generate(model, max_nodes=args.max_nodes, temperature=args.temperature,
                     top_k=args.top_k, max_extent=args.max_extent,
                     device=tcfg.device)
        rolls.append(r)
        structs.append(rollout_to_structure(r, codec.decode, args.oriented))
    occ = [int(s.occupied_mask.sum()) for s in structs]
    stop_rate = float(np.mean([r.stopped for r in rolls]))
    print(f"[sample] median blocks {int(np.median(occ))} | "
          f"emitted STOP {stop_rate:.2f} | max {max(occ)}", flush=True)

    try:
        from blockgen.renderer.grid import save_grid
        save_grid(structs, str(run / "samples.png"), cols=4, tile_px=256)
        print(f"[render] {run / 'samples.png'}", flush=True)
    except Exception as exc:                                   # headless is optional
        print(f"[render] skipped ({type(exc).__name__}: {exc})", flush=True)

    from blockgen.curation.houses import save_house_cache
    npz = save_house_cache([s.crop_to_non_air() for s in structs], 32,
                           cache_dir=str(run), name=args.name,
                           report={"model": "pick_n_place", "epochs": args.epochs})

    metrics = {"params": n_par, "train_min": round(train_min, 2),
               "history": {k: [round(x, 4) for x in v] for k, v in hist.items()},
               "val": {k: round(v, 4) for k, v in val_metrics.items()},
               "median_blocks": int(np.median(occ)), "stop_rate": stop_rate,
               "repr": gs, "palette": codec.n_pieces,
               "train_stats": train_ds.stats()}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\n-> {run}")
    print(f"bench:\n  .venv/bin/python -m blockgen.eval.bench "
          f"--arms ar/{args.name}:{npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
