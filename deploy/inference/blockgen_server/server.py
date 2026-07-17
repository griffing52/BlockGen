"""WebSocket inference server: streams generated blocks to the Minecraft mod.

Run it on the GPU box; the mod connects from wherever Minecraft runs.

    python -m blockgen_server.server --host 0.0.0.0 --port 8000

Protocol (JSON over a single WebSocket, ``/ws``). Client -> server:

    {"type": "models"}
    {"type": "generate", "model": "native_bpe", "prompt": null, "seed": 7,
     "temperature": 1.0, "top_k": 40, "cfg_scale": 3.0, "max_tokens": null}
    {"type": "cancel"}

Server -> client:

    {"type": "models",  "default": "native_bpe", "models": [...]}
    {"type": "begin",   "model": "native_bpe", "seed": 7, "supports_text": false}
    {"type": "blocks",  "blocks": [[x, y, z, "minecraft:oak_planks"], ...]}
    {"type": "done",    "blocks": 1752, "elapsed": 10.4, "reason": "complete"}
    {"type": "error",   "message": "..."}

Coordinates are the model's local frame (0-based, y-up); the mod anchors them.

**Why a thread.** Sampling is a long synchronous CUDA loop. Running it on the event
loop would block the socket, so ``cancel`` would not be read until generation
finished -- i.e. cancel would not work at all, which is exactly when you want it
(these builds take ~10s and a bad one is obvious in 2s). Generation runs in a worker
thread and hands batches to the loop through a queue; the socket stays responsive.

One generation per connection at a time: a second ``generate`` while one is running
is rejected rather than queued, since the mod has one world anchor per player and
interleaving two builds into it produces rubble.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from blockgen_server.backends import Block, GenerateRequest
from blockgen_server.registry import Registry

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]

# Flush a batch when it reaches this many blocks or this many seconds have passed,
# whichever comes first. One message per decoded piece would be thousands of tiny
# frames; one message per build would defeat the entire point of streaming.
BATCH_BLOCKS = 64
BATCH_SECONDS = 0.05

app = FastAPI(title="BlockGen inference")
registry: Optional[Registry] = None


def get_registry() -> Registry:
    global registry
    if registry is None:
        registry = Registry(HERE / "models.json", REPO)
    return registry


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/models")
def models() -> dict:
    reg = get_registry()
    return {"default": reg.default, "models": reg.describe()}


def _encode(blocks: List[Block]) -> list:
    return [[b.x, b.y, b.z, b.state] for b in blocks]


def _worker(backend, req: GenerateRequest, loop: asyncio.AbstractEventLoop,
            queue: asyncio.Queue, cancel: threading.Event) -> None:
    """Run the blocking sampler, pushing batches onto the asyncio queue."""
    def put(item):
        # The queue belongs to the event loop; hand items over thread-safely.
        asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

    pending: List[Block] = []
    last = time.time()
    try:
        for batch in backend.stream(req):
            if cancel.is_set():
                put(("done", "cancelled"))
                return
            pending.extend(batch)
            now = time.time()
            if len(pending) >= BATCH_BLOCKS or (now - last) >= BATCH_SECONDS:
                put(("blocks", pending))
                pending, last = [], now
        if pending:
            put(("blocks", pending))
        put(("done", "complete"))
    except Exception as exc:  # noqa: BLE001 - surface to the client, never hang it
        put(("error", f"{type(exc).__name__}: {exc}"))


async def _generate(ws: WebSocket, msg: dict, cancel: threading.Event) -> None:
    reg = get_registry()
    try:
        backend = reg.get(msg.get("model"))
    except Exception as exc:  # noqa: BLE001 - unknown model / bad checkpoint / vocab
        await ws.send_json({"type": "error", "message": str(exc)})
        return

    prompt = msg.get("prompt") or None
    if prompt and not backend.supports_text:
        await ws.send_json({"type": "error", "message":
                            f"{backend.spec.name} does not take a prompt; use a "
                            f"cond_piece_ar model or drop the text"})
        return

    seed = msg.get("seed")
    if seed is None:
        seed = random.randrange(2 ** 31)  # report it so a build is reproducible
    req = GenerateRequest(
        prompt=prompt, seed=int(seed),
        temperature=float(msg.get("temperature", 1.0)),
        top_k=msg.get("top_k", 40),
        cfg_scale=float(msg.get("cfg_scale", 3.0)),
        max_tokens=msg.get("max_tokens"))

    await ws.send_json({"type": "begin", "model": backend.spec.name, "seed": seed,
                        "prompt": prompt, "supports_text": backend.supports_text})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel.clear()
    t0 = time.time()
    total = 0
    thread = threading.Thread(target=_worker, args=(backend, req, loop, queue, cancel),
                              daemon=True)
    thread.start()
    while True:
        kind, payload = await queue.get()
        if kind == "blocks":
            total += len(payload)
            await ws.send_json({"type": "blocks", "blocks": _encode(payload)})
        elif kind == "error":
            await ws.send_json({"type": "error", "message": payload})
            return
        else:
            await ws.send_json({"type": "done", "blocks": total,
                                "elapsed": round(time.time() - t0, 2),
                                "reason": payload})
            return


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    cancel = threading.Event()
    task: Optional[asyncio.Task] = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "models":
                reg = get_registry()
                await ws.send_json({"type": "models", "default": reg.default,
                                    "models": reg.describe()})
            elif kind == "generate":
                if task and not task.done():
                    await ws.send_json({"type": "error", "message":
                                        "a generation is already running; /gen cancel first"})
                    continue
                cancel.clear()
                task = asyncio.create_task(_generate(ws, msg, cancel))
            elif kind == "cancel":
                cancel.set()          # the worker notices between batches
            else:
                await ws.send_json({"type": "error",
                                    "message": f"unknown message type {kind!r}"})
    except WebSocketDisconnect:
        cancel.set()                   # don't keep sampling for a client that left
    except Exception as exc:  # noqa: BLE001
        cancel.set()
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001 - socket already gone
            pass


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 so the laptop on your LAN can reach it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true",
                    help="load the default model at startup instead of on first /gen")
    args = ap.parse_args()

    reg = get_registry()
    print(f"models: {', '.join(reg.names())} (default: {reg.default})")
    for row in reg.describe():
        if not row["available"]:
            print(f"  ! {row['name']}: {row['unavailable_reason']}")
    if args.preload:
        t0 = time.time()
        reg.get(None)
        print(f"preloaded {reg.default} in {time.time() - t0:.1f}s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
