"""Generate BrickGPT structures from recorded prompts (published-model scoring).

Runs in BrickGPT's own venv (libs/BrickGPT/.venv), never the project's:

    flock <gpu.lock> libs/BrickGPT/.venv/bin/python scripts/brickgpt_generate.py \
        --run outputs/run_<stamp>_brickgpt_lego

Reads `<run>/prompts.jsonl` (written by the prompt picker; see notes.md), writes
one `<run>/gen/<idx>.json` per prompt (bricks text, rejection reasons, number of
rollbacks, wall time) and skips prompts already done, so a killed job resumes.

Everything is BrickGPT's default `BrickGPTConfig` except three things, each
forced and each recorded in `<run>/gen_config.json`:

* **base weights**: the adapter's base `meta-llama/Llama-3.2-1B-Instruct` is gated
  and this account has no access, so the base is loaded from the
  `unsloth/Llama-3.2-1B-Instruct` mirror (same 2,471,645,608-byte safetensors;
  the Llama 3.2 Community License permits redistribution).
* **dtype**: bf16 rather than the reference fp32, to fit the ~4.5 GB of shared
  VRAM available.
* **stability check during rollback**: `use_gurobi=False` (the repo's documented
  connectivity fallback) by default, because the pip gurobipy licence is
  size-limited to 2,000 variables and the stability programme uses ~22 per
  brick. Pass `--use-gurobi` once a full licence is installed.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BASE = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER = "AvaLovelace/BrickGPT"


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def patch_llm(dtype: torch.dtype) -> None:
    """Load base mirror + BrickGPT LoRA (merged) in `dtype` inside BrickGPT's LLM."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brickgpt.models import llm as llm_mod

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
        self.model.to(device).eval()
        self.kv_cache = None
        self.kv_cache_saved = None
        self.input_ids_cache = None
        self.input_ids_cache_saved = None

    llm_mod.LLM.__init__ = __init__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--use-gurobi", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0,
                    help="exit after generating this many (lets a driver loop "
                         "release the shared GPU lock between chunks)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="wall-clock cap per structure (s). Not in BrickGPT; a "
                         "timed-out prompt is recorded with no structure and "
                         "counted as a failure, never dropped")
    ap.add_argument("--reverse", action="store_true",
                    help="work from the last prompt backwards (a second worker)")
    ap.add_argument("--shard", default="0/1", help="i/k: do prompts with idx %% k == i")
    args = ap.parse_args()

    import transformers
    from brickgpt.models import BrickGPT, BrickGPTConfig

    run = Path(args.run)
    gen = run / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in (run / "prompts.jsonl").read_text().splitlines() if l.strip()]
    si, sk = (int(x) for x in args.shard.split("/"))
    prompts = [p for p in prompts if p["idx"] % sk == si]
    if args.limit:
        prompts = prompts[:args.limit]

    prompts = [p for p in prompts if not (gen / f"{p['idx']:04d}.json").exists()]
    if args.reverse:
        prompts = prompts[::-1]
    if args.max_new:
        prompts = prompts[:args.max_new]
    if not prompts:
        print("nothing to do", flush=True)
        return 0
    patch_llm(getattr(torch, args.dtype))
    cfg = BrickGPTConfig(use_gurobi=args.use_gurobi, device=args.device)
    (run / f"gen_config_{args.device}.json").write_text(json.dumps({
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "base_weights": BASE, "adapter": ADAPTER, "dtype": args.dtype,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "deviations": ["base weights from ungated mirror", f"dtype {args.dtype}",
                       "use_gurobi=%s" % args.use_gurobi],
    }, indent=2, default=str))
    model = BrickGPT(cfg)

    for p in prompts:
        out = gen / f"{p['idx']:04d}.json"
        if out.exists():
            continue
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        transformers.set_seed(int(p["seed"]))
        t0 = time.time()
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.timeout)
        try:
            res = model(p["caption"])
        except _Timeout:
            dt = time.time() - t0
            out.write_text(json.dumps({**p, "timed_out": True, "bricks": "",
                                       "n_bricks": 0, "rejection_reasons": {},
                                       "n_rejections": 0, "n_regenerations": -1,
                                       "seconds": dt, "device": args.device}))
            print(f"[{p['idx']}] TIMEOUT after {dt:.0f}s", flush=True)
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        rec = {**p, "bricks": res["bricks"].to_txt(), "n_bricks": len(res["bricks"]),
               "rejection_reasons": dict(res["rejection_reasons"]),
               "n_rejections": int(sum(res["rejection_reasons"].values())),
               "n_regenerations": int(res["n_regenerations"]),
               "seconds": dt,
               "device": args.device, "dtype": args.dtype,
               "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30
                                if args.device == "cuda" else 0.0)}
        out.write_text(json.dumps(rec))
        print(f"[{p['idx']}] {p['category']:8s} bricks={rec['n_bricks']:4d} "
              f"rej={rec['n_rejections']:4d} regen={rec['n_regenerations']:3d} "
              f"{dt:6.1f}s {rec['peak_vram_gb']:.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
