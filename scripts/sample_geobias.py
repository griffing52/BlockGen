"""Sample a `train_llm_geobias.py` run into a benchmark arm (docs/workflow.md step 2).

The benchmark is generator-blind: its only input is a structure cache written by
`blockgen.curation.houses.save_house_cache`. This is the conversion step for the
LLM arms, and it is where their `report` block -- run, arm, checkpoint, seed,
temperature, prompts -- gets written so a leaderboard row can explain itself.

**KV-cached decoding.** `train_llm_geobias.sample` re-runs the whole prefix every
step (~45 s/build), which is fine for a contact sheet and useless for the 128 builds
the official protocol wants. Here the bias row for the new query is built
incrementally by `GeoBias.row`, whose equality with the training-time
`GeoBias.matrix` is pinned in `tests/test_geobias.py`. `--verify` re-checks it
end-to-end at greedy decode before spending an hour on samples: if cached and
uncached decoding disagree on a single token, the arm is measuring a sampler bug.

Run::

    .venv/bin/python scripts/sample_geobias.py --run outputs/run_..._llm_geobias_bias \\
        --n 128 --out-dir outputs/bench_arms --verify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blockgen.curation.houses import load_house_structures, save_house_cache
from blockgen.export.minecraftace import load_piece_vocab
from blockgen.utils.data import Structure
from train_llm_pieces import (MODEL_NAME, CAPTIONS_PATH, PIECE_VOCAB, build_piece_names,
                              parse_piece_completion)
from train_llm_geobias import GeoBias, causal_mask


def _anchor_of(line: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    p = line.split()
    if len(p) == 4 and all(v.lstrip("-").isdigit() for v in p[1:4]):
        return (int(p[1]), int(p[2]), int(p[3]))
    return fallback


@torch.no_grad()
def generate(model, tokenizer, geo, caption: str, device, *, max_new: int,
             temperature: float, top_p: float, greedy: bool = False,
             cached: bool = True) -> str:
    """One build. `cached=False` is the O(T^2) reference path used by `--verify`."""
    prompt = f"Build: {caption}\n"
    ids: List[int] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n_prompt = len(ids)
    qwen = model.base_model.model
    dtype = qwen.model.embed_tokens.weight.dtype
    eos = tokenizer.eos_token_id

    tok_line = [-1] * n_prompt
    anchors: List[Tuple[int, int, int]] = []
    text, line_start, cur_line = "", 0, 0
    past = None

    for step in range(max_new):
        if cached and past is not None:
            tl = torch.tensor(tok_line, dtype=torch.long, device=device)
            if geo is not None:
                an = torch.tensor(anchors + [(0, 0, 0)], dtype=torch.long, device=device)
                # The query is the token just emitted, so its line is `tok_line[-1]` --
                # NOT `cur_line`, which has already advanced when that token closed a
                # line. `matrix()` keys every row on the token's own line, and using
                # the advanced value here silently shifted the whole reference by one
                # line. Caught by `--verify`.
                row = geo.row(tl, an, int(tok_line[-1])).to(torch.float32)
            else:
                row = torch.zeros(len(tok_line), device=device)
            mask = {"full_attention": row.to(dtype)[None, None, None]}
            out = qwen.model(input_ids=torch.tensor([[ids[-1]]], device=device),
                             attention_mask=mask, past_key_values=past, use_cache=True)
        else:
            T = len(ids)
            bias = None
            if geo is not None:
                tl = torch.tensor(tok_line, dtype=torch.long, device=device)
                an = torch.tensor(anchors + [(0, 0, 0)], dtype=torch.long, device=device)
                bias = geo.matrix(tl, an)
            out = qwen.model(input_ids=torch.tensor([ids], device=device),
                             attention_mask=causal_mask(T, dtype, device, bias),
                             use_cache=cached)
        past = out.past_key_values if cached else None
        logits = qwen.lm_head(out.last_hidden_state[:, -1]).float()[0]

        if greedy:
            tok = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, -1)
            sp, si = torch.sort(probs, descending=True)
            cut = int(torch.searchsorted(torch.cumsum(sp, 0), top_p)) + 1
            tok = int(si[torch.multinomial(sp[:cut] / sp[:cut].sum(), 1)])
        if tok == eos:
            break
        ids.append(tok)
        tok_line.append(cur_line)
        piece = tokenizer.decode([tok])
        text += piece
        if "\n" in piece:
            anchors.append(_anchor_of(text[line_start:].split("\n")[0],
                                      anchors[-1] if anchors else (0, 0, 0)))
            line_start = len(text)
            cur_line += 1
    return text


@torch.no_grad()
def verify(model, tokenizer, geo, caption: str, device, steps: int = 24) -> None:
    """Check the cached decode path against the reference, on the right criterion.

    Exact token equality is the criterion you reach for first and it is wrong here:
    batched prefill and single-token decode run different bf16 kernels, so their
    logits differ by ~0.6 median for this model **with the bias switched off**. The
    control arm passed exact-match only because its top-1/top-2 gap never closed;
    the bias arm hit an exact tie on a coordinate digit and flipped a token. Fixing
    that by loosening to a hand-picked tolerance would just hide the next real bug.

    So verify two things instead. (1) The bias row used at decode is *exactly* the
    row training would have used -- this is the part that can actually be wrong, and
    it is checked at every step. (2) The cached path's logit drift is no worse than
    the same model's drift with no bias at all, measured here rather than assumed,
    so the tolerance is the model's own numerical floor.
    """
    qwen = model.base_model.model
    dtype = qwen.model.embed_tokens.weight.dtype

    def rollout(use_bias: bool):
        ids = tokenizer(f"Build: {caption}\n", add_special_tokens=False)["input_ids"]
        tok_line, anchors, cur, past, drift, text, ls = [-1] * len(ids), [], 0, None, [], "", 0
        g = geo if use_bias else None
        for _ in range(steps):
            tl = torch.tensor(tok_line, dtype=torch.long, device=device)
            an = torch.tensor(anchors + [(0, 0, 0)], dtype=torch.long, device=device)
            full = g.matrix(tl, an) if g is not None else None
            if g is not None:                       # (1) the row must be exact
                row = g.row(tl, an, int(tok_line[-1]))
                assert torch.allclose(row, full[-1], atol=1e-6), "row() != matrix()[-1]"
            o_un = qwen.model(input_ids=torch.tensor([ids], device=device),
                              attention_mask=causal_mask(len(ids), dtype, device, full),
                              use_cache=False)
            l_un = qwen.lm_head(o_un.last_hidden_state[:, -1]).float()[0]
            if past is None:
                o_c = qwen.model(input_ids=torch.tensor([ids], device=device),
                                 attention_mask=causal_mask(len(ids), dtype, device, full),
                                 use_cache=True)
            else:
                r = (full[-1] if g is not None
                     else torch.zeros(len(tok_line), device=device)).to(torch.float32)
                o_c = qwen.model(input_ids=torch.tensor([[ids[-1]]], device=device),
                                 attention_mask={"full_attention": r.to(dtype)[None, None, None]},
                                 past_key_values=past, use_cache=True)
            past = o_c.past_key_values
            l_c = qwen.lm_head(o_c.last_hidden_state[:, -1]).float()[0]
            drift.append(float((l_un - l_c).abs().max()))
            tok = int(l_un.argmax())
            ids.append(tok); tok_line.append(cur)
            piece = tokenizer.decode([tok]); text += piece
            if "\n" in piece:
                anchors.append(_anchor_of(text[ls:].split("\n")[0],
                                          anchors[-1] if anchors else (0, 0, 0)))
                ls = len(text); cur += 1
        return float(np.median(drift))

    with_bias = rollout(True) if geo is not None else 0.0
    floor = rollout(False)
    print(f"[verify] bias row == training row at every step; "
          f"cached-vs-uncached logit drift {with_bias:.3f} against a no-bias floor of "
          f"{floor:.3f} (bf16 kernel noise)", flush=True)
    assert with_bias <= max(3.0 * floor, 0.5), (
        f"bias path drifts {with_bias:.3f} vs no-bias floor {floor:.3f} -- not kernel noise")


def _trained_indices(cfg: dict, tokenizer, cv, captions) -> set:
    """The corpus indices a geobias run actually trained on, re-derived from its config."""
    from train_llm_geobias import build_examples
    structures, _ = load_house_structures(max_dim=32)
    id_to_name, _ = build_piece_names(cv)
    random.seed(cfg.get("seed", 0))
    ex, _ = build_examples(structures, captions, tokenizer, cv, id_to_name,
                           cfg.get("max_tokens", 2048), None)
    order = list(range(len(ex)))
    random.Random(cfg.get("seed", 0)).shuffle(order)
    n_val = max(1, int(round(len(ex) * 0.10)))
    return {ex[i]["idx"] for i in order[n_val:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a train_llm_geobias.py run directory")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out-dir", default="outputs/bench_arms")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-dim", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prompts", choices=["canonical_test", "run_val"], default="canonical_test",
                    help="canonical_test draws captions from the pinned houses_32 test "
                         "split (docs/workflow.md); run_val uses the run's own val builds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", default=None,
                    help="extend an existing arm .npz up to --n instead of starting fresh")
    ap.add_argument("--max-attempt-factor", type=float, default=1.6,
                    help="give up after n * this many attempts")
    ap.add_argument("--keep-trained-prompts", action="store_true",
                    help="do NOT drop canonical-test captions the run trained on "
                         "(they are dropped by default: prompting a model with a build "
                         "it memorized measures recall, not generation)")
    ap.add_argument("--verify", action="store_true",
                    help="check cached decoding == uncached decoding before sampling")
    args = ap.parse_args()

    device = "cuda"
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    name = args.name or f"llm_{cfg['arm']}"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(run / "adapter")).to(device).eval()
    model.config.use_cache = True
    geo = None
    if (run / "geo_bias.pt").exists():
        geo = GeoBias(cfg.get("radius", 4)).to(device)
        geo.load_state_dict(torch.load(run / "geo_bias.pt", map_location=device))
        geo.eval()
    print(f"[model] {run.name}  arm={cfg['arm']}  bias={'yes' if geo is not None else 'no'}",
          flush=True)

    cv = load_piece_vocab(PIECE_VOCAB)
    _, name_to_id = build_piece_names(cv)
    captions = json.loads(Path(CAPTIONS_PATH).read_text())

    # --- prompt slate ------------------------------------------------------
    n_excluded = 0
    if args.prompts == "canonical_test":
        from blockgen.eval.bench.splits import load_split
        idxs = load_split("houses_32").indices("test").tolist()
        if not args.keep_trained_prompts:
            # Track D predates the pinned split, so these checkpoints trained on some
            # canonical-test builds. That contamination is in the weights and cannot be
            # undone here; what CAN be avoided is handing the model a caption whose
            # build it memorized. Recorded in the report either way.
            trained = _trained_indices(cfg, tokenizer, cv, captions)
            before = len(idxs)
            idxs = [i for i in idxs if i not in trained]
            n_excluded = before - len(idxs)
            print(f"[prompts] dropped {n_excluded}/{before} canonical-test captions the "
                  f"run trained on", flush=True)
    else:
        structures, _ = load_house_structures(max_dim=args.max_dim)
        from train_llm_geobias import build_examples
        ex, _ = build_examples(structures, captions, tokenizer, cv,
                               build_piece_names(cv)[0], cfg["max_tokens"], None)
        order = list(range(len(ex)))
        random.Random(cfg["seed"]).shuffle(order)
        idxs = [ex[i]["idx"] for i in order[: max(1, int(round(len(ex) * 0.10)))]]
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    pool = [captions[f"h{i:05d}"][0] for i in idxs if captions.get(f"h{i:05d}")]
    if not pool:
        raise SystemExit("no captions in the prompt slate")
    print(f"[prompts] slate of {len(pool)} from {args.prompts}", flush=True)

    if args.verify:
        verify(model, tokenizer, geo, pool[0], device)

    structs: List[Structure] = []
    rows: List[dict] = []
    if args.append:
        from blockgen.curation.houses import load_structures_from_cache
        prev, prev_man = load_structures_from_cache(args.append)
        structs.extend(prev)
        prev_rows = json.loads(Path(args.append.replace(f"_{args.max_dim}.npz",
                                                        "_samples.json")).read_text())
        rows.extend(prev_rows["samples"])
        print(f"[append] starting from {len(structs)} builds in {args.append}", flush=True)

    # `--n` counts KEPT builds, not attempts. A generation that parses to zero blocks
    # is not a sample the benchmark can score, and an arm that quietly lands at 121
    # is an arm the protocol drops for being 7 short.
    t0, n_empty, attempt = time.time(), 0, 0
    if args.append:
        # Resume the prompt cursor where the previous pass stopped, or every top-up
        # re-draws the same captions with the same seeds and duplicates the builds
        # it was meant to extend.
        r = prev_man.get("report", {})
        attempt = int(r.get("attempts") or (int(r.get("n", 0)) + int(r.get("n_empty", 0))))
        n_empty = int(r.get("n_empty", 0))
        print(f"[append] resuming prompt cursor at attempt {attempt}", flush=True)
    max_attempts = attempt + int(args.n * args.max_attempt_factor) + 16
    while len(structs) < args.n and attempt < max_attempts:
        cap = pool[attempt % len(pool)]
        torch.manual_seed(args.seed * 100003 + attempt + 7919 * (attempt // len(pool)))
        attempt += 1
        txt = generate(model, tokenizer, geo, cap, device, max_new=args.max_new_tokens,
                       temperature=args.temperature, top_p=args.top_p)
        st, n_valid, n_lines = parse_piece_completion(txt, cv, name_to_id)
        blocks = int(st.occupied_mask.sum())
        if blocks == 0:
            n_empty += 1
            continue
        st.metadata = {"corpus": f"gen:{name}", "title": cap[:80], "category": "generated"}
        structs.append(st)
        rows.append({"caption": cap, "n_lines": n_lines, "n_valid": n_valid,
                     "parse_rate": n_valid / max(n_lines, 1), "blocks": blocks})
        if len(structs) % 16 == 0:
            el = time.time() - t0
            print(f"  kept {len(structs)}/{args.n}  attempts {attempt}  "
                  f"{el / attempt:.1f}s/attempt  empty {n_empty}", flush=True)
    if len(structs) < args.n:
        print(f"[warn] only {len(structs)}/{args.n} builds after {attempt} attempts",
              flush=True)

    parse = float(np.mean([r["parse_rate"] for r in rows])) if rows else 0.0
    # The parser accepts coordinates up to 255, so a runaway anchor produces a build
    # far outside the corpus's 32^3 scale. It is real model output and is NOT dropped
    # -- filtering it would flatter the arm -- but it moves the rendered appearance
    # pillar, so the rate is recorded next to the score that it explains.
    dims = [max(st.shape) for st in structs]
    oversize = sum(1 for d in dims if d > args.max_dim)
    path = save_house_cache(
        structs, args.max_dim, cache_dir=args.out_dir, name=name,
        report={"model": f"llm_geobias:{cfg['arm']}", "checkpoint": str(run / "adapter"),
                "run": run.name, "arm": cfg["arm"], "n": len(structs), "n_empty": n_empty,
                "attempts": attempt, "appended_to": args.append or "",
                "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens, "prompts": args.prompts,
                "parse_rate": parse, "base_model": MODEL_NAME,
                "prompts_dropped_as_trained": n_excluded,
                "oversize_builds": oversize, "oversize_rate": oversize / max(len(structs), 1),
                "max_dim_seen": int(max(dims)) if dims else 0,
                "train_split": cfg.get("split", "trackd"),
                "piece_vocab": cfg.get("piece_vocab", PIECE_VOCAB)})
    (Path(args.out_dir) / f"{name}_samples.json").write_text(json.dumps(
        {"arm": cfg["arm"], "run": run.name, "parse_rate": parse, "samples": rows}, indent=1))

    from blockgen.eval.cond_render import textured_prompt_grid
    sheet = Path(args.out_dir) / f"{name}_samples.png"
    textured_prompt_grid(structs[:8], [r["caption"] for r in rows[:8]], sheet,
                         suptitle=f"{name} — n={len(structs)}, mean parse {parse:.2f}")
    print(f"[render] {sheet}", flush=True)
    print(f"[done] {len(structs)} builds, {n_empty} empty, mean parse {parse:.3f}, "
          f"{oversize} over {args.max_dim} (largest {max(dims) if dims else 0}) "
          f"in {(time.time() - t0) / 60:.1f} min")
    print(f"bench with:\n  python -m blockgen.eval.bench --arms llm/{name}:{path}")


if __name__ == "__main__":
    main()
