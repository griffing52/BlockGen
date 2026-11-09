"""BrickNet track driver: the Qwen3-0.6B + LoRA text-walk model through the pipeline.

Runs in the conda env (blockgen importable) and shells into the BrickNet venv
(`libs/BrickNet/.venv`) for the GPU steps, so the BrickNet stack (Qwen tokenizer,
PEFT, transformers 4.57) never meets the native trainer. Plan:
`deliverables/reports/brickgpt-pipeline-plan.md`.

    PYTHONPATH=. python scripts/bricknet_pipeline.py sample            # stage 0 readout
    PYTHONPATH=. python scripts/bricknet_pipeline.py sample --dry-run  # print the plan only

`sample` draws one build per training-house caption, realises each walk, crops it to
32 and writes a bench structure cache (`samples_n<n>_32.npz` + `_manifest.json`),
`samples.png`, `draws.jsonl`, `run.json` and `summary.json` into
`outputs/run_<stamp>_bricknet_sample_<name>`, then appends one registry line.

Adapter stack. `train.py sft --pt_adapter A` merges A into Qwen3 and trains a new LoRA
on top, so a saved SFT adapter is only the delta over base + A. The stack is therefore
every adapter in training order, passed as repeated `--lora` (generate.py merges them
in order). It is recorded in `run.json` and the manifest.

Prompts are captions of TRAINING houses (`sft-houses32`, checked disjoint from
`eval-holdout`), never holdout captions: the arm is unconditional in distribution, not
prompt-matched. Raw sampling, no collision mask.

Draw accounting: a draw is valid when the whole text parses (`parse_path`, grammar
errors and truncated tails count as unparsable) and realises at least one block.
Collisions (a node on an occupied cell; the node is dropped, the walk continues) do
not invalidate a draw; they are counted. The driver draws until `n` valid or
`--max-draws`, keeps the first `n` valid in draw order and records every draw;
nothing is dropped silently. Exit code 3 marks a shortfall.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRICKNET = ROOT / "libs" / "BrickNet"
VENV_PY = BRICKNET / ".venv" / "bin" / "python"
RUNS = BRICKNET / "repro" / "runs"
CAPTIONS = ROOT / "data" / "minecraft" / "labels" / "houses_32_captions.json"
BASE_MODEL = "Qwen/Qwen3-0.6B"
MAX_DIM = 32

#: Ordered adapter stacks (merged in this order onto Qwen3-0.6B), from the
#: `--pt_adapter` each run was trained with (repro/run_mc_chain*.sh).
STACKS: Dict[str, List[str]] = {
    "mc_pt_base": ["repro/runs/mc_pt_base/adapter"],
    "mc_pt_legoinit": ["kulits/BrickNet-0.6B-PT", "repro/runs/mc_pt_legoinit/adapter"],
    "mc_sft_base": ["repro/runs/mc_pt_base/adapter", "repro/runs/mc_sft_base/adapter"],
    "mc_sft_legoinit": ["kulits/BrickNet-0.6B-PT", "repro/runs/mc_pt_legoinit/adapter",
                        "repro/runs/mc_sft_legoinit/adapter"],
}
#: SFT eval loss (Qwen-token NLL on `paths_val` walks, HF Trainer `eval_loss`) at the
#: saved adapter, from repro/runs/<run>.log. Not comparable to the native `val_nll`.
EVAL_LOSS = {"mc_sft_base": {"step": 2000, "eval_loss": 0.64927, "best_step": 1500, "best_eval_loss": 0.64904}}


@functools.lru_cache(maxsize=None)
def _repro():
    """Import the BrickNet converter (`libs/BrickNet/repro/mc_to_bricknet.py`) read-only."""
    sys.path.insert(0, str(BRICKNET / "repro"))
    import mc_to_bricknet  # noqa: E402
    return mc_to_bricknet


# ----------------------------------------------------------------- prompts

def training_prompts(seed: int) -> List[dict]:
    """Every `sft-houses32` house with captions, in a seeded order, one caption each.

    The order is the draw order: draw k uses entry k, so redraws for a shortfall
    continue down the list on houses not yet used. After the last house the list
    repeats with fresh caption picks.
    """
    from blockgen.corpus.shards import Shard, find
    mc = _repro()
    train_ids = list(dict.fromkeys(str(i) for i in Shard(find("sft-houses32")).meta().id.values))
    holdout = set(str(i) for i in Shard(find("eval-holdout")).meta().id.values)
    leak = holdout.intersection(train_ids)
    if leak:
        raise SystemExit(f"{len(leak)} sft-houses32 ids are also in eval-holdout, e.g. {sorted(leak)[:3]}")
    caps = json.loads(CAPTIONS.read_text())
    houses = [(mid, caps[k]) for mid in train_ids
              if (k := mc.caption_key(mid)) and caps.get(k)]
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(8):
        for j in rng.permutation(len(houses)):
            mid, cl = houses[int(j)]
            ci = int(rng.integers(len(cl)))
            out.append({"meta_id": mid, "caption_index": ci, "caption": cl[ci], "pass": rep})
    return out


# ----------------------------------------------------------------- realise

def crop32(ids: np.ndarray, dat: np.ndarray, max_dim: int = MAX_DIM) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Window an XYZ build to at most ``max_dim`` per axis.

    x and z keep the centred window; y keeps the bottom ``max_dim`` layers, so a tall
    build loses its roof, not its floor. The result is re-cropped to its occupied box.
    """
    sl, cut = [], False
    for ax, d in enumerate(ids.shape):
        if d <= max_dim:
            sl.append(slice(0, d))
            continue
        cut = True
        s = 0 if ax == 1 else (d - max_dim) // 2
        sl.append(slice(s, s + max_dim))
    ids, dat = ids[tuple(sl)], dat[tuple(sl)]
    occ = np.argwhere(ids != 0)
    if len(occ):
        lo, hi = occ.min(0), occ.max(0) + 1
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        ids, dat = ids[box], dat[box]
    return ids, dat, cut


def realise(text: str, table) -> dict:
    """One generated walk -> a record with its status and, when valid, the arrays."""
    mc = _repro()
    cells, n_nodes, coll, err = mc.parse_path(text)
    rec = {"n_nodes": n_nodes, "n_cells": len(cells), "collisions": coll, "error": err,
           "trailing_newline": text.endswith("\n")}
    if err is not None:
        rec["status"] = "unparsable"
        return rec
    if not cells:
        rec["status"] = "empty"
        return rec
    rec["unknown_names"] = sum(1 for nd in cells.values() if (nd[0], int(nd[1])) not in table)
    ids, dat = mc.cells_to_arrays(cells, table)
    rec["dims_raw"] = list(ids.shape)
    ids, dat, cut = crop32(ids, dat)
    rec.update(cropped=cut, dims=list(ids.shape), n_blocks=int((ids != 0).sum()), status="valid",
               _arrays=(ids, dat))
    return rec


# ----------------------------------------------------------------- generate

def generate_chunk(prompts: Sequence[dict], out: Path, adapters: Sequence[str], a, seed: int) -> Path:
    """One generate.py call in the BrickNet venv; skipped when ``out`` is complete."""
    if out.exists() and sum(1 for l in out.open() if l.strip()) == len(prompts):
        print(f"[bricknet] reuse {out.name}", flush=True)
        return out
    pfile = out.with_suffix(".prompts.jsonl")
    pfile.write_text("".join(json.dumps(p) + "\n" for p in prompts))
    cmd = [str(VENV_PY), "scripts/generate.py", "--model", BASE_MODEL]
    for ad in adapters:
        cmd += ["--lora", ad]
    cmd += ["--output", str(out.resolve()), "--prompts_file", str(pfile.resolve()), "--batch_size", str(a.batch_size),
            "--max_new_tokens", str(a.max_new_tokens), "--seed", str(seed), "--temperature", str(a.temperature),
            "--top_k", str(a.top_k), "--top_p", str(a.top_p)]
    env = {**os.environ, "HF_HOME": "/home/griffing52/hf_cache", "BRICKNET_DATA": str(BRICKNET / ".bricknet_data"),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print("[bricknet] " + " ".join(cmd), flush=True)
    if a.dry_run:
        return out
    subprocess.run(cmd, cwd=BRICKNET, env=env, check=True)
    return out


def read_rows(path: Path) -> List[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return sorted(rows, key=lambda r: (r.get("id", 0), r.get("sample", 0)))


# ----------------------------------------------------------------- sample

def cmd_sample(a) -> int:
    from blockgen.corpus.shards import Shard, find
    from blockgen.curation.houses import save_house_cache
    from blockgen.utils.data import Structure
    from blockgen.utils.runs import new_run_dir

    mc = _repro()
    adapters = a.adapters or STACKS[a.model]
    for ad in adapters:
        if not ad.startswith("kulits/") and not (BRICKNET / ad / "adapter_model.safetensors").exists():
            raise SystemExit(f"adapter not found: {BRICKNET / ad}")
    run_dir = Path(a.run_dir) if a.run_dir else new_run_dir(f"bricknet_sample_{a.model}", base=a.base)
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = run_dir / "gen"
    gen_dir.mkdir(exist_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    t0 = time.time()

    pool = training_prompts(a.seed)
    prompts_file = run_dir / "prompts.jsonl"
    prompts_file.write_text("".join(json.dumps({"draw": i, **p}) + "\n" for i, p in enumerate(pool)))
    n_houses = len({p["meta_id"] for p in pool})
    table = mc.name_to_pair_table(Shard(find("eval-holdout")).tokenizer)
    run = {"stage": "sample", "arch": "bricknet", "model": a.model, "base_model": BASE_MODEL,
           "adapters": adapters, "generate": "libs/BrickNet/scripts/generate.py",
           "args": {k: v for k, v in vars(a).items() if k != "func"},
           "prompts": {"file": str(prompts_file), "source": "sft-houses32 house captions "
                       f"({CAPTIONS.relative_to(ROOT)}), {n_houses} houses, seeded order", "seed": a.seed}}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))

    if a.gen_jsonl:                       # offline test path: treat existing generations as the draws
        rows = [r for p in a.gen_jsonl for r in read_rows(Path(p))][: a.max_draws]
        draws = [(i, r["text"]) for i, r in enumerate(rows)]
    else:
        draws = []
    records, valid = [], []

    def absorb(batch):
        for i, text in batch:
            rec = realise(text, table)
            rec["draw"] = i
            rec["prompt"] = pool[i] if not a.gen_jsonl else None
            records.append(rec)
            if rec["status"] == "valid" and len(valid) < a.n:
                valid.append(rec)

    if a.gen_jsonl:
        absorb(draws)
    else:
        drawn = 0
        while len(valid) < a.n and drawn < a.max_draws:
            want = min(a.n - len(valid), a.max_draws - drawn)
            for s in range(drawn, drawn + want, a.chunk):
                e = min(s + a.chunk, drawn + want)
                out = generate_chunk(pool[s:e], gen_dir / f"draw_{s:05d}_{e:05d}.jsonl", adapters, a, a.seed + s)
                if a.dry_run:
                    continue
                rows = read_rows(out)
                if len(rows) != e - s:
                    raise SystemExit(f"{out}: {len(rows)} rows for {e - s} prompts")
                absorb([(s + k, r["text"]) for k, r in enumerate(rows)])
                print(f"[bricknet] draws {e}: {len(valid)} valid, "
                      f"{sum(r['status'] == 'unparsable' for r in records)} unparsable, "
                      f"{sum(r['status'] == 'empty' for r in records)} empty, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
            drawn += want
            if a.dry_run:
                print("[bricknet] dry run: stopping after the first round", flush=True)
                return 0

    n_drawn = len(records)
    count = lambda st: sum(r["status"] == st for r in records)  # noqa: E731
    with (run_dir / "draws.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_arrays"}) + "\n")

    structs = []
    for r in valid:
        ids, dat = r["_arrays"]
        structs.append(Structure(block_ids=ids, block_data=dat, metadata={"corpus": f"bricknet_{a.model}"}))
    report = {
        "model": f"bricknet_{a.model}", "arch": "bricknet", "kind": "text_walk",
        "checkpoint": {"file": f"libs/BrickNet/{adapters[-1]}", **EVAL_LOSS.get(a.model, {})},
        "adapters": [f"libs/BrickNet/{ad}" if not ad.startswith("kulits/") else ad for ad in adapters],
        "train_split": "canonical",
        "base_model": f"{BASE_MODEL.split('/')[-1]} (external pretraining)",
        "train_split_note": "label undecided: Qwen3-0.6B was pretrained on an external web corpus; "
                            "the Minecraft PT pool excludes eval-holdout and SFT is on sft-houses32",
        "trained_on_signals": [], "n": len(structs), "n_target": a.n, "seed": a.seed,
        "temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p,
        "max_new_tokens": a.max_new_tokens, "batch_size": a.batch_size, "decoding": "raw sampling, no collision mask",
        "prompts": str(prompts_file), "prompt_source": "sft-houses32 training-house captions (not eval-holdout)",
        "n_drawn": n_drawn, "n_unparsable": count("unparsable"), "n_empty": count("empty"),
        "n_valid_drawn": count("valid"),
        "n_collisions": sum(r["collisions"] > 0 for r in valid),
        "collisions_total": int(sum(r["collisions"] for r in valid)),
        "n_cropped": sum(bool(r["cropped"]) for r in valid),
        "n_unknown_names": int(sum(r["unknown_names"] for r in valid)),
        "n_unparsable_no_trailing_newline": sum(r["status"] == "unparsable" and not r["trailing_newline"]
                                                for r in records),
        "valid_rule": "whole text parses (parse_path) and realises >= 1 block; collisions counted, not rejected; "
                      "first n valid in draw order kept",
        "n_collisions_rule": "kept builds with >= 1 collision (colliding nodes dropped by the realiser)",
        "crop": "x/z centred 32 window, y bottom 32 layers, then occupied-box crop; unknown names -> stone",
        "draws": str(run_dir / "draws.jsonl"),
    }
    if a.gen_jsonl:
        report["test_input"] = list(a.gen_jsonl)
    npz = None
    if structs:
        npz = save_house_cache(structs, MAX_DIM, cache_dir=str(run_dir), name=f"samples_n{a.n}", report=report)
        try:
            from blockgen.renderer.grid import save_grid
            save_grid(structs[:64], str(run_dir / "samples.png"), cols=8)
        except Exception as e:  # rendering must not lose a multi-hour draw
            print(f"[bricknet] render failed: {e}", flush=True)
    shortfall = a.n - len(structs)
    summary = {"run": str(run_dir), "name": f"bricknet_{a.model}", "stage": "sample", "arch": "bricknet",
               "model": a.model, "npz": npz, "n_valid": len(structs), "shortfall": shortfall,
               "minutes": round((time.time() - t0) / 60, 1), **{k: report[k] for k in (
                   "adapters", "train_split", "base_model", "trained_on_signals", "seed", "temperature", "top_k",
                   "top_p", "n_drawn", "n_unparsable", "n_empty", "n_collisions", "n_cropped")}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if not a.no_registry:
        from blockgen.train import registry
        registry.append({"run": str(run_dir), "name": f"bricknet_{a.model}", "stage": "sample", "arch": "bricknet",
                         "kind": "text_walk", "base_model": report["base_model"], "adapters": report["adapters"],
                         "init_from": report["adapters"], "train_split": "canonical", "trained_on_signals": [],
                         "eval_loss_bricknet": EVAL_LOSS.get(a.model), "bench_n": len(structs), "bench": None,
                         "samples_png": (run_dir / "samples.png").exists(), "npz": npz,
                         "note": f"n_drawn={n_drawn} unparsable={report['n_unparsable']} empty={report['n_empty']}"
                                 + (f" SHORTFALL {shortfall}" if shortfall > 0 else "")})
    print(f"NPZ {npz}", flush=True)
    print(f"[bricknet] {len(structs)}/{a.n} valid from {n_drawn} draws "
          f"({report['n_unparsable']} unparsable, {report['n_empty']} empty, "
          f"{report['n_collisions']} kept with collisions, {report['n_cropped']} cropped)", flush=True)
    if shortfall > 0:
        print(f"SHORTFALL {shortfall}: {len(structs)} valid < {a.n}; a card at this n is not comparable", flush=True)
        return 3
    return 0


def not_yet(a) -> int:
    print(f"bricknet_pipeline {a.cmd}: not yet (plan section 2, deliverables/reports/brickgpt-pipeline-plan.md)")
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw n builds from an adapter stack into a bench cache")
    s.add_argument("--model", default="mc_sft_base", choices=sorted(STACKS), help="run under libs/BrickNet/repro/runs")
    s.add_argument("--adapters", nargs="+", default=None, help="override the stack (paths relative to libs/BrickNet)")
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top_k", type=int, default=20)
    s.add_argument("--top_p", type=float, default=0.95)
    s.add_argument("--max_new_tokens", type=int, default=20480, help="p99 house is ~19k tokens")
    s.add_argument("--batch_size", type=int, default=4, help="KV ~114 KB/token: 4 x 20k tokens ~9.3 GB on 16 GB")
    s.add_argument("--chunk", type=int, default=32, help="prompts per generate.py call (resume granularity)")
    s.add_argument("--max_draws", type=int, default=1024)
    s.add_argument("--base", default="outputs", help="parent of the run dir")
    s.add_argument("--run_dir", default=None, help="resume into an existing run dir (completed chunks are reused)")
    s.add_argument("--gen_jsonl", nargs="+", default=None, help="test: parse these generate.py outputs, no GPU")
    s.add_argument("--no_registry", action="store_true")
    s.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the generate.py commands only")
    s.set_defaults(func=cmd_sample)
    for name in ("data", "pt", "sft", "bon", "rft"):
        sub.add_parser(name, help="not yet").set_defaults(func=not_yet)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
