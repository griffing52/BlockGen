"""CLI for the two-stage cell-assignment generator on full houses (README, "Full houses").

    .venv/bin/python -m tools.misc.adjacency_gen cells-stage1 --f 4 --minutes 10
    .venv/bin/python -m tools.misc.adjacency_gen cells-grow --f 4 --minutes 40
    .venv/bin/python -m tools.misc.adjacency_gen cells-merge --f 4 --minutes 40 [--dagger]
    .venv/bin/python -m tools.misc.adjacency_gen cells-eval --f 4 --stage1 <run> --grow <run> [--merge <run> ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _device(max_gb):
    from .train import pick_device
    return pick_device(max_gb, need_free_gb=3.0)


def quick_samples(run: Path, ds, f: int, rows: dict, n: int = 8):
    """samples.png for a training run: condition, real, then the given rows
    ({name: list of (coords, types, house)})."""
    from .coarse import coarse_structure
    from .cells import house_coarse
    from .eval import save_samples
    from .eval_cells import to_structure
    T = ds["vocab"].n_types
    hs = ds["test"][:n]
    sm = {"condition": [coarse_structure(house_coarse(h, f, T)) for h in hs],
          "real": [to_structure(h.coords, h.types, h) for h in hs]}
    for k, v in rows.items():
        sm[k] = [to_structure(c, t, h) for c, t, h in v]
    save_samples(sm, run / "samples.png")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adjacency_gen cells")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cells-stage1", "cells-grow", "cells-merge"):
        p = sub.add_parser(name)
        p.add_argument("--f", type=int, default=4)
        p.add_argument("--minutes", type=float, default=30)
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--name", default=None)
        p.add_argument("--max-gb", type=float, default=6.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-n", type=int, default=None, help="train only on houses with at most this many blocks")
        p.add_argument("--smoke", action="store_true")
        if name == "cells-stage1":
            p.add_argument("--d", type=int, default=192)
            p.add_argument("--layers", type=int, default=4)
            p.add_argument("--heads", type=int, default=6)
            p.add_argument("--dropout", type=float, default=0.1)
            p.add_argument("--wd", type=float, default=0.01)
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--mirror", action="store_true", help="add x-mirrored training houses")
            p.add_argument("--eval-every", type=int, default=1000)
        if name == "cells-merge":
            p.add_argument("--dagger", action="store_true", help="mix in states from the decoder's own rollouts")
            p.add_argument("--init", default=None, help="initialize from a merge run's merge.pt")
    e = sub.add_parser("cells-eval")
    e.add_argument("--f", type=int, default=4)
    e.add_argument("--stage1", required=True)
    e.add_argument("--grow", default=None)
    e.add_argument("--merge", nargs="*", default=[])
    e.add_argument("--arms", nargs="+", default=None)
    e.add_argument("--n", type=int, default=399)
    e.add_argument("--merge-max-n", type=int, default=1024)
    e.add_argument("--steps", type=int, default=256)
    e.add_argument("--n-div", type=int, default=64)
    e.add_argument("--k-div", type=int, default=3)
    e.add_argument("--name", default=None)
    e.add_argument("--max-gb", type=float, default=6.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None, help="write into this existing run dir (e.g. to redraw a figure)")
    e.add_argument("--tag", default="", help="suffix for results/rows files")
    a = ap.parse_args(argv)

    from blockgen.utils.runs import new_run_dir
    from .cells import assignment, build_house_cells_dataset, house_coarse
    device = _device(a.max_gb)
    print(f"[device] {device}", flush=True)
    t0 = time.time()
    ds = build_house_cells_dataset()
    T = ds["vocab"].n_types
    print(f"[data] {ds['stats']} in {time.time() - t0:.0f}s", flush=True)
    if a.cmd != "cells-eval" and a.max_n:
        ds["train"] = [h for h in ds["train"] if len(h) <= a.max_n]
        ds["val"] = [h for h in ds["val"] if len(h) <= a.max_n]
        print(f"[data] max_n {a.max_n}: {len(ds['train'])} train, {len(ds['val'])} val", flush=True)
    if getattr(a, "smoke", False):
        a.minutes = 1.0
        ds["train"] = ds["train"][:200]

    if a.cmd == "cells-stage1":
        from .stage1 import Stage1Config, stage1_items, train_stage1
        from .eval_cells import stage1_report
        run = new_run_dir(f"adjacency_cells_{a.name or f'stage1_f{a.f}'}{'_smoke' if a.smoke else ''}")
        print(f"[run] {run}", flush=True)
        cfg = Stage1Config(n_types=T, f=a.f, d=a.d, layers=a.layers, heads=a.heads, dropout=a.dropout)
        (run / "config.json").write_text(json.dumps({"args": vars(a), "model": cfg.to_dict(),
                                                    "data": ds["stats"]}, indent=1))
        trh = ds["train"]
        if a.mirror:
            from .cells import mirror_x
            trh = trh + [mirror_x(h) for h in trh]
        tr = stage1_items(trh, a.f, T)
        va = stage1_items(ds["val"], a.f, T)
        model = train_stage1(tr, va, cfg, run, device, minutes=a.minutes, bs=a.bs, seed=a.seed, wd=a.wd,
                             lr=a.lr, eval_every=a.eval_every)
        rep, rows = stage1_report(model, ds, a.f, device, seed=a.seed)
        (run / "results.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1), flush=True)
        quick_samples(run, ds, a.f, rows)
    elif a.cmd == "cells-grow":
        from .grow_cells import GrowConfig, grow_decode, train_grow
        run = new_run_dir(f"adjacency_cells_{a.name or f'grow_f{a.f}'}{'_smoke' if a.smoke else ''}")
        print(f"[run] {run}", flush=True)
        cfg = GrowConfig(n_types=T, f=a.f)
        (run / "config.json").write_text(json.dumps({"args": vars(a), "model": cfg.to_dict(),
                                                    "data": ds["stats"]}, indent=1))
        model = train_grow(ds["train"], ds["val"], cfg, run, device, minutes=a.minutes, bs=a.bs, seed=a.seed)
        hs = ds["test"][:8]
        A = [assignment(h, a.f, T) for h in hs]
        C = [house_coarse(h, a.f, T) for h in hs]
        o = grow_decode(model, hs, A, C, device, a.f, bs=8)
        quick_samples(run, ds, a.f, {"oracle+grow": [(x["coords"], x["types"], h) for x, h in zip(o, hs)]})
    elif a.cmd == "cells-merge":
        from .merge_cells import MergeConfig, merge_cells_decode, train_merge
        run = new_run_dir(f"adjacency_cells_{a.name or ('merge_dagger' if a.dagger else 'merge') + f'_f{a.f}'}"
                          f"{'_smoke' if a.smoke else ''}")
        print(f"[run] {run}", flush=True)
        cfg = MergeConfig(n_types=T, f=a.f)
        (run / "config.json").write_text(json.dumps({"args": vars(a), "model": cfg.to_dict(),
                                                    "data": ds["stats"]}, indent=1))
        model = train_merge(ds["train"], ds["val"], cfg, run, device, minutes=a.minutes, bs=a.bs,
                            seed=a.seed, dagger=a.dagger, init=a.init)
        hs = [h for h in ds["test"] if len(h) <= (a.max_n or 1024)][:8]
        A = [assignment(h, a.f, T) for h in hs]
        o = merge_cells_decode(model, hs, A, device, a.f)
        from .coarse import coarse_structure
        from .eval import save_samples
        from .eval_cells import to_structure
        save_samples({"condition": [coarse_structure(house_coarse(h, a.f, T)) for h in hs],
                      "real": [to_structure(h.coords, h.types, h) for h in hs],
                      "oracle+merge": [to_structure(x["coords"], x["types"], h) for x, h in zip(o, hs)]},
                     run / "samples.png")
    else:
        from .eval_cells import evaluate_cells
        run = Path(a.out) if a.out else new_run_dir(f"adjacency_cells_{a.name or f'eval_f{a.f}'}")
        print(f"[run] {run}", flush=True)
        (run / f"config{a.tag}.json").write_text(json.dumps({"args": vars(a), "data": ds["stats"]}, indent=1))
        evaluate_cells(run, ds, a, device)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
