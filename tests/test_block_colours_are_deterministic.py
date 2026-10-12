"""Fallback block colours must not change between processes.

`_hashed_color` assigns an arbitrary colour to any block with no palette entry.
Arbitrary is fine; *unstable* is not. It previously seeded the colour with
builtin `hash()`, which CPython salts per process, so every flat-colour render
in the paper came out in a different palette on each run -- including the teaser
figure. Nothing failed, nothing warned, and rebuilding the figures for a
camera-ready copy would have silently recoloured every building in the paper
while the captions and the measured numbers stayed put.

This test pins the stable-digest behaviour. It deliberately does NOT assert
specific RGB values: the colours may be re-chosen deliberately, but they must be
reproducible, so what is pinned is agreement across independent interpreters
under different hash seeds.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from blockgen.utils.data import _color_for, _hashed_color

PROBE = ("minecraft:block_251", "minecraft:planks", "minecraft:unknown_thing",
         "", "minecraft:stone")

_SNIPPET = (
    "from blockgen.utils.data import _hashed_color;"
    "print([ _hashed_color(r) for r in %r ])" % (PROBE,)
)


def _colours_in_fresh_process(hash_seed: str) -> str:
    """Render the probe colours in a new interpreter with a given hash seed."""
    out = subprocess.run([sys.executable, "-c", _SNIPPET], check=True,
                         capture_output=True, text=True,
                         env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"})
    return out.stdout.strip()


def test_fallback_colours_agree_across_hash_seeds():
    """The regression itself: two hash seeds used to give two palettes."""
    seeds = ["0", "1", "12345", "random"]
    results = {s: _colours_in_fresh_process(s) for s in seeds}
    distinct = set(results.values())
    assert len(distinct) == 1, (
        "fallback block colours differ between processes: "
        + "; ".join(f"PYTHONHASHSEED={s} -> {v[:70]}" for s, v in results.items())
        + ". A figure rebuilt later will not match the one in the paper. "
        "Seed the colour with a digest (hashlib), never builtin hash().")


def test_fallback_colours_are_in_the_intended_band():
    """The colours stay mid-tone, so text over a render keeps its contrast."""
    for name in PROBE:
        r, g, b, a = _hashed_color(name)
        assert a == 1.0
        for channel in (r, g, b):
            assert 0.35 <= channel <= 0.80, f"{name}: channel {channel} out of band"


def test_named_blocks_are_unaffected():
    """Only the fallback path was changed; assigned palette entries must not move."""
    assert _color_for("minecraft:wool", 0) == _color_for("minecraft:wool", 0)
    # a block with a real entry must not be routed through the fallback
    assert _color_for("minecraft:wool", 0) != _hashed_color("minecraft:wool")


@pytest.mark.parametrize("name", PROBE)
def test_stable_within_a_process(name: str):
    assert _hashed_color(name) == _hashed_color(name)
