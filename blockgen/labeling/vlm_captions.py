"""Cap3D-style VLM labeling of rendered structures via the OpenAI API.

Sends each structure's 4 rendered views (from ``render_views.py``) plus a
metadata hint to a vision model and gets back, in ONE call:
  * ``captions``    — 3 diverse text-to-structure captions
  * ``is_build``    — coherence filter (false = terrain/tree/fragment/junk)
  * ``category``    — 1-3 word build type (house/castle/tower/statue/...)
  * ``flag_reason`` — why it was flagged, when ``is_build`` is false

So one sweep yields captions (for text conditioning), a semantic junk filter
(complements the geometric enclosed-air gate), and a category — same render,
same API call, near-zero marginal cost over captioning alone. Uses the OpenAI
Batch API (50% cheaper) by default; ``--sync`` does immediate requests for small
tests.

Reads ``OPENAI_API_KEY`` from the environment or the repo ``.env``.

Progress is a JSONL file keyed by structure id — reruns skip completed ids.

Usage:
    python -m blockgen.labeling.vlm_captions \
        --renders outputs/renders/houses_32 \
        --index data/minecraftace/houses_32/index.json \
        --out data/minecraft/labels/houses_32_vlm.jsonl \
        [--only-missing-titles] [--limit 8 --sync]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from typing import Dict, List, Optional

DEFAULT_MODEL = "gpt-5-mini"
N_VIEWS = 4
BATCH_CHUNK = 1000            # max requests per batch job
BATCH_MAX_BYTES = 60_000_000  # well under OpenAI's ~200MB limit; big uploads 504

SYSTEM = (
    "You label 3D Minecraft structures for a text-to-structure dataset. You are "
    "shown 4 views (rotated 90° apart) of the same build rendered on a white "
    "background. Do three things:\n"
    "1. Judge whether this is a single, coherent, intentionally-built structure. "
    "Set is_build=false for terrain/landscape, a lone tree or plant, a floating or "
    "cropped fragment, a random world-chunk, or an incoherent blob — anything that "
    "is not a usable standalone build.\n"
    "2. Give a short build type in `category` (1-3 lowercase words, e.g. house, "
    "castle, church, tower, bridge, statue, ship, garden, farm, pixel art, "
    "redstone, tree, terrain, fragment).\n"
    "3. Write captions a player might type to request this build: mention the "
    "structure type, notable architecture (roof shape, stories, towers, porches), "
    "and main materials. Caption what is actually shown even when is_build=false. "
    "Do not mention the white background, the rendering, or the camera views."
)

# One VLM call returns: coherence filter + category + captions.
LABEL_SCHEMA = {
    "name": "structure_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_build": {
                "type": "boolean",
                "description": "true only if a single coherent intentionally-built "
                               "structure; false for terrain, plants, floating/cropped "
                               "fragments, world-chunks, or incoherent blobs.",
            },
            "category": {
                "type": "string",
                "description": "1-3 word lowercase build type (house, castle, tower, "
                               "statue, bridge, pixel art, redstone, tree, terrain, "
                               "fragment, ...).",
            },
            "short_tag": {
                "type": "string",
                "description": "a terse 2-4 word tag naming the main material + type, "
                               "e.g. 'spruce house', 'stone castle', 'brick tower'. This "
                               "is how a user would quickly type it.",
            },
            "captions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3 diverse captions (8-25 words each), from plain to "
                               "detailed. Name materials accurately using the provided "
                               "dominant-blocks list; describe naturally, do not list "
                               "the blocks verbatim.",
            },
            "flag_reason": {
                "type": "string",
                "description": "empty string if is_build is true and quality is fine; "
                               "otherwise a short reason (e.g. 'mostly terrain', "
                               "'floating fragment', 'incoherent blob').",
            },
        },
        "required": ["is_build", "category", "short_tag", "captions", "flag_reason"],
        "additionalProperties": False,
    },
}

# Fields a completed label record carries (besides "id").
LABEL_FIELDS = ("captions", "short_tag", "is_build", "category", "flag_reason")


def _load_env() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        # repo root .env (two levels up from blockgen/labeling/)
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def _image_url(path: str) -> dict:
    # Re-encode PNG -> JPEG: renders are on an opaque white bg, and ~140KB PNGs
    # would blow OpenAI's batch-file size limit (4 views x 1000 requests).
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": "low"}}


def build_request_body(sid: str, renders_dir: str, meta: dict,
                       model: str) -> Optional[dict]:
    """Chat-completions request body for one structure, or None if renders missing."""
    paths = [os.path.join(renders_dir, f"{sid}_view{v}.png") for v in range(N_VIEWS)]
    if not all(os.path.exists(p) for p in paths):
        return None
    hint_bits = []
    if meta.get("title"):
        hint_bits.append(f"builder's title: {meta['title']!r}")
    if meta.get("category"):
        hint_bits.append(f"category: {meta['category']}")
    hint = ("Metadata hint (may be noisy): " + "; ".join(hint_bits) + ". "
            if hint_bits else "")
    # Ground-truth block histogram (idea: block-stats grounding). The VLM cannot read
    # exact block types off a render, so we tell it the actual dominant materials.
    mats = meta.get("materials")
    mat_hint = (f"Dominant blocks (ground truth, most first): {', '.join(mats)}. "
                if mats else "")
    content: List[dict] = [_image_url(p) for p in paths]
    content.append({"type": "text",
                    "text": f"{hint}{mat_hint}Judge if this is a real build, categorize "
                            "it, give a short 2-4 word tag, and write 3 diverse captions. "
                            "Use the dominant-blocks list to name materials accurately."})
    return {
        "model": model,
        "max_completion_tokens": 2000,  # headroom for reasoning models
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": LABEL_SCHEMA},
    }


def load_done(out_path: str) -> Dict[str, dict]:
    """id -> full label record. Back-compatible with old caption-only JSONL."""
    done: Dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def _parse_result(content: str) -> dict:
    """Parse the VLM JSON into a normalized label record (without the id)."""
    obj = json.loads(content)
    captions = [c.strip() for c in obj.get("captions", []) if c and c.strip()]
    return {
        "captions": captions,
        "short_tag": (obj.get("short_tag") or "").strip(),
        "is_build": bool(obj.get("is_build", True)),
        "category": (obj.get("category") or "").strip().lower(),
        "flag_reason": (obj.get("flag_reason") or "").strip(),
    }


def run_sync(client, todo, renders_dir, index, out_path, model):
    with open(out_path, "a") as out:
        for i, sid in enumerate(todo):
            body = build_request_body(sid, renders_dir, index[sid], model)
            if body is None:
                print(f"{sid}: renders missing, skipped", flush=True)
                continue
            resp = client.chat.completions.create(**body)
            choice = resp.choices[0]
            if choice.finish_reason == "content_filter" or not choice.message.content:
                print(f"{sid}: no content ({choice.finish_reason}), skipped", flush=True)
                continue
            rec = _parse_result(choice.message.content)
            out.write(json.dumps({"id": sid, **rec}) + "\n")
            out.flush()
            flag = "" if rec["is_build"] else f" [FLAGGED: {rec['flag_reason']}]"
            head = rec["captions"][0] if rec["captions"] else "(no caption)"
            print(f"[{i + 1}/{len(todo)}] {sid} ({rec['category']}){flag}: {head}",
                  flush=True)


def _chunk_requests(todo, renders_dir, index, model):
    """Yield lists of JSONL lines bounded by count AND total bytes."""
    lines, nbytes = [], 0
    for sid in todo:
        body = build_request_body(sid, renders_dir, index[sid], model)
        if body is None:
            continue
        line = json.dumps({"custom_id": sid, "method": "POST",
                           "url": "/v1/chat/completions", "body": body})
        if lines and (nbytes + len(line) > BATCH_MAX_BYTES or len(lines) >= BATCH_CHUNK):
            yield lines
            lines, nbytes = [], 0
        lines.append(line)
        nbytes += len(line)
    if lines:
        yield lines


def run_batches(client, todo, renders_dir, index, out_path, model):
    n_queued = 0
    for lines in _chunk_requests(todo, renders_dir, index, model):
        n_queued += len(lines)
        payload = "\n".join(lines).encode()
        batch = None
        for attempt in range(5):  # transient 5xx/timeouts on large uploads
            try:
                upload = client.files.create(
                    file=("batch.jsonl", io.BytesIO(payload)), purpose="batch")
                batch = client.batches.create(
                    input_file_id=upload.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h")
                break
            except Exception as exc:
                wait = 30 * (2 ** attempt)
                print(f"batch submit failed ({type(exc).__name__}), "
                      f"retry in {wait}s", flush=True)
                time.sleep(wait)
        if batch is None:
            print("batch submit failed after retries — rerun later to resume",
                  flush=True)
            return
        print(f"batch {batch.id}: {len(lines)} requests "
              f"({n_queued}/{len(todo)} queued)", flush=True)
        while True:
            batch = client.batches.retrieve(batch.id)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(60)
        if batch.status != "completed":
            print(f"batch {batch.id}: {batch.status} — "
                  f"{batch.errors}", flush=True)
            continue
        n_ok = 0
        results = client.files.content(batch.output_file_id).text
        with open(out_path, "a") as out:
            for line in results.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec["custom_id"]
                if rec.get("error") or rec["response"]["status_code"] != 200:
                    print(f"{sid}: request failed ({rec.get('error')})", flush=True)
                    continue
                choice = rec["response"]["body"]["choices"][0]
                if not choice["message"].get("content"):
                    print(f"{sid}: no content ({choice.get('finish_reason')})", flush=True)
                    continue
                try:
                    label = _parse_result(choice["message"]["content"])
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"{sid}: parse failed ({exc})", flush=True)
                    continue
                out.write(json.dumps({"id": sid, **label}) + "\n")
                n_ok += 1
        print(f"batch {batch.id} done: {n_ok}/{len(lines)} captioned", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True, help="dir of <id>_view{0..3}.png")
    parser.add_argument("--index", required=True,
                        help="index.json from blockgen.export.minecraftace")
    parser.add_argument("--out", required=True, help="output JSONL (appended, resumable)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only-missing-titles", action="store_true",
                        help="caption only structures with no title metadata")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sync", action="store_true",
                        help="immediate requests instead of the Batch API (small tests)")
    args = parser.parse_args()

    _load_env()
    from openai import OpenAI
    client = OpenAI()

    with open(args.index) as f:
        index = json.load(f)
    done = load_done(args.out)

    todo = [sid for sid, meta in sorted(index.items())
            if sid not in done and meta.get("split") != "filtered"
            and (not args.only_missing_titles or not meta.get("title"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to caption", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.sync:
        run_sync(client, todo, args.renders, index, args.out, args.model)
    else:
        run_batches(client, todo, args.renders, index, args.out, args.model)


if __name__ == "__main__":
    main()
