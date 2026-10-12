"""The recipe backend (``recipe:<name>``), offline.

``recipe:mock`` runs the dream recipe on the mock LLM and image backends, so the
whole live path -- episode thread, narration, dream billboard, per-attempt diffs and
the final restore -- runs with no API key and no spend.

Run:  python -m pytest deploy/inference/tests/ -q
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen_server.backends import GenerateRequest, as_batch
from blockgen_server.billboard import billboard_blocks
from blockgen_server.decode import Block
from blockgen_server.recipe_backend import diff_blocks, summarize_reply
from blockgen_server.registry import Registry

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry(HERE / "models.json", REPO)


def test_recipe_group_lists_benched_recipes(registry):
    members = registry.groups()["recipe"]
    for name in ("dream-caption", "dream", "classic-matched", "mock"):
        assert name in members


def test_diff_removes_then_builds_bottom_up():
    have = {(0, 0, 0): "minecraft:stone", (0, 1, 0): "minecraft:stone"}
    want = {(0, 0, 0): "minecraft:stone", (1, 2, 0): "minecraft:oak_planks",
            (1, 0, 0): "minecraft:dirt"}
    out = diff_blocks(have, want)
    assert out[0] == Block(0, 1, 0, "minecraft:air")
    assert [b.y for b in out[1:]] == [0, 2]
    assert diff_blocks(want, want) == []


def test_billboard_drops_background():
    from PIL import Image
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (150, 97, 83))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    blocks = billboard_blocks(buf.getvalue(), (32, 32, 32), size=16, gap=3)
    assert blocks and all(p[2] == -4 for p in blocks)
    # The 8x8 centre square survives; the white border is dropped. Resampling
    # blends the square's edge, so only its interior is pinned to bricks.
    assert 64 <= len(blocks) <= 100
    assert blocks[(16 - 8 + 8, 16 - 1 - 8, -4)] == "minecraft:bricks"


def test_summarize_reply():
    assert summarize_reply("builder", "# walls first\nfill stone 0 0 0 3 0 3\nwalls oak_planks 0 1 0 3 3 3") \
        == "wrote a 2-command program: walls first"
    s = summarize_reply("judge", '{"score": 0.4, "critique": "roof is flat", "ok": false}')
    assert "score 0.4" in s and "roof is flat" in s


def test_mock_recipe_streams_live(registry):
    backend = registry.get("recipe:mock")
    backend.spec.extra["trace_dir"] = ""          # no disk trace in tests
    batches = [as_batch(b) for b in backend.stream(GenerateRequest(prompt="a hut", seed=0))]
    lines = [l for b in batches for l in (b.log or [])]
    styles = {l["style"] for l in lines}
    assert {"stage", "dream", "think", "build", "score"} <= styles
    stats = batches[-1].stats
    assert stats["attempts"] >= 1 and stats["best_attempt"] is not None
    # Replaying every batch leaves exactly the selected attempt plus the billboard.
    world = {}
    for b in batches:
        for blk in b.blocks:
            if blk.state == "minecraft:air":
                world.pop((blk.x, blk.y, blk.z), None)
            else:
                world[(blk.x, blk.y, blk.z)] = blk.state
    build = {p for p in world if p[2] >= 0}
    wall = {p for p in world if p[2] < 0}
    assert build and wall


def test_cancel_stops_the_stream(registry):
    import threading
    backend = registry.get("recipe:mock")
    backend.spec.extra["trace_dir"] = ""
    ev = threading.Event()
    ev.set()
    batches = list(backend.stream(GenerateRequest(prompt="a hut", seed=1, cancel=ev)))
    assert all(as_batch(b).stats is None for b in batches)
