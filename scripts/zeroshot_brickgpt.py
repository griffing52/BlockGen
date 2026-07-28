"""Zero-shot baseline: prompt a frontier LLM (gpt-5-mini) to emit a Minecraft build,
no finetuning — with and without an in-context example. The MC-Bench/T2BM baseline.

Same caption->block-lines format, name map, parser, and textured render as the
finetuned model (`train_llm_brickgpt.py`), so it's apples-to-apples. Two conditions:

* ``zeroshot`` — format spec + caption only.
* ``oneshot``  — format spec + ONE real serialized build as an in-context example,
  then the caption.

Both parse back to a ``Structure`` and render textured, titled with the prompt.
VoxelCodeBench's finding is the hypothesis: a frontier LLM emits valid-looking output
but spatially-incoherent builds; the in-context example should help format + grounding.

    python scripts/zeroshot_brickgpt.py --n 12 --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # for train_llm_brickgpt
from train_llm_brickgpt import build_name_map, parse_completion, serialize_structure  # noqa: E402

from blockgen.curation.houses import load_house_structures  # noqa: E402
from blockgen.eval.cond_render import textured_prompt_grid  # noqa: E402
from blockgen.utils.runs import new_run_dir  # noqa: E402

SYSTEM = (
    "You generate Minecraft structures as a flat list of blocks. Output ONLY lines "
    "of the form `<block_name> <x> <y> <z>` — a modern Minecraft block name (e.g. "
    "oak_planks, cobblestone, glass, oak_stairs) then three 0-based integer "
    "coordinates, y is up. One block per line, no prose, no code fences, no coordinates "
    "outside 0..31. Build the structure described in the prompt as a coherent, "
    "connected build with walls, a roof, and a floor."
)


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    from openai import OpenAI
    return OpenAI()


def generate(client, model, caption, example=None, max_tokens=16000,
             reasoning_effort="low"):
    # gpt-5 is a REASONING model: reasoning tokens count against max_completion_tokens,
    # so a small budget -> empty content. A house block-list is also large (thousands of
    # lines), so give a big budget and keep reasoning low (this is a serialization task,
    # not a reasoning one).
    msgs = [{"role": "system", "content": SYSTEM}]
    if example is not None:
        ex_cap, ex_body = example
        msgs.append({"role": "user", "content": f"Build: {ex_cap}"})
        msgs.append({"role": "assistant", "content": ex_body})
    msgs.append({"role": "user", "content": f"Build: {caption}"})
    kw = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kw)
    return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--conditions", nargs="+", default=["zeroshot", "oneshot"])
    args = ap.parse_args()

    structures, manifest = load_house_structures(max_dim=32)
    captions = json.load(open("data/minecraft/labels/houses_32_captions.json"))
    name_to_pair, pair_to_name = build_name_map(structures)

    occ = [(int(s.occupied_mask.sum()), i) for i, s in enumerate(structures)]
    occ.sort()
    # In-context example = a SMALL real build (keeps the prompt short); pick ~120 blocks.
    ex_i = next(i for n, i in occ if n >= 120 and captions.get(f"h{i:05d}"))
    example = (captions[f"h{ex_i:05d}"][0], serialize_structure(structures[ex_i], pair_to_name))
    # Eval on a deterministic spread of small/medium builds with captions.
    eval_idx = [i for n, i in occ if 120 <= n <= 700 and captions.get(f"h{i:05d}")
                and i != ex_i][: args.n]
    print(f"example build h{ex_i:05d} ({int(structures[ex_i].occupied_mask.sum())} blocks); "
          f"{len(eval_idx)} eval captions", flush=True)

    client = _client()
    out = new_run_dir("zeroshot_brickgpt")
    report = {}
    for cond in args.conditions:
        ex = example if cond == "oneshot" else None
        samples, prompts, rates = [], [], []
        for k, i in enumerate(eval_idx):
            cap = captions[f"h{i:05d}"][0]
            text = generate(client, args.model, cap, example=ex)
            struct, n_valid, n_lines = parse_completion(text, name_to_pair)
            rate = n_valid / max(1, n_lines)
            samples.append(struct)
            prompts.append(cap)
            rates.append(rate)
            print(f"[{cond} {k+1}/{len(eval_idx)}] '{cap[:40]}' -> "
                  f"{int(struct.occupied_mask.sum())} blocks, parse {rate:.2f}", flush=True)
        textured_prompt_grid(
            samples, prompts, out / f"samples_{cond}.png",
            suptitle=f"{args.model} {cond} (no finetune) — prompt -> output")
        report[cond] = {"mean_parse_rate": round(sum(rates) / max(1, len(rates)), 3),
                        "median_blocks": sorted(int(s.occupied_mask.sum()) for s in samples)
                        [len(samples) // 2] if samples else 0}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\n" + json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
