"""Price the full-corpus VLM captioning from a small live sample.

Runs N real caption requests (sync), records token usage, and projects the full-batch
cost. Captions are written to the same resumable JSONL, so the sample is not wasted —
the full run skips these ids. Reports tokens precisely and a $ estimate under a stated
price assumption (correct the per-1M rates to your gpt-5-mini pricing).

    python scripts/vlm_price_probe.py --n 100 \
        --renders outputs/renders/all_32 \
        --index data/minecraft/labels/all_32_index.json \
        --out data/minecraft/labels/all_32_vlm.jsonl
"""

from __future__ import annotations

import argparse
import json

from blockgen.labeling.vlm_captions import (DEFAULT_MODEL, _load_env, _parse_result,
                                            build_request_body, load_done)

# EDIT to your gpt-5-mini pricing ($ per 1M tokens). Batch API is 50% off both.
PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    _load_env()
    from openai import OpenAI
    client = OpenAI()

    index = json.load(open(args.index))
    done = load_done(args.out)
    todo = [sid for sid in index if sid not in done][: args.n]
    print(f"probing {len(todo)} builds with {args.model}", flush=True)

    tin = tout = n_ok = n_build = 0
    with open(args.out, "a") as out:
        for i, sid in enumerate(todo):
            body = build_request_body(sid, args.renders, index[sid], args.model)
            if body is None:
                continue
            resp = client.chat.completions.create(**body)
            u = resp.usage
            tin += u.prompt_tokens
            tout += u.completion_tokens
            ch = resp.choices[0]
            if not ch.message.content:
                continue
            rec = _parse_result(ch.message.content)
            out.write(json.dumps({"id": sid, **rec}) + "\n")
            out.flush()
            n_ok += 1
            n_build += int(rec["is_build"])
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(todo)}] tags e.g. {rec.get('short_tag')!r}; "
                      f"is_build so far {n_build}/{n_ok}", flush=True)

    if n_ok == 0:
        print("no successful calls"); return
    ain, aout = tin / n_ok, tout / n_ok
    total = len(index)
    proj_in, proj_out = ain * total, aout * total
    sync_cost = proj_in / 1e6 * PRICE_IN_PER_M + proj_out / 1e6 * PRICE_OUT_PER_M
    print("\n" + "=" * 68)
    print(f"sample: {n_ok} ok, is_build {n_build}/{n_ok} ({100*n_build/n_ok:.0f}%)")
    print(f"avg tokens/build: {ain:.0f} in + {aout:.0f} out")
    print(f"full corpus ({total}): {proj_in/1e6:.1f}M in + {proj_out/1e6:.1f}M out")
    print(f"est. cost @ ${PRICE_IN_PER_M}/{PRICE_OUT_PER_M} per-1M "
          f"(EDIT to real rates): sync ${sync_cost:.2f}, batch(-50%) ${sync_cost/2:.2f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
