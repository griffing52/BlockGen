"""Live smoke tests of the paid / quota backends. Skipped unless BLOCKGEN_ALLOW_SPEND=1.

Total cost when run: about $0.01 of API credit (one gpt-image-1-mini low image, one
gpt-5-mini call) plus one small claude-code:haiku call on subscription quota. The
caches are written to a temp dir so every run really calls out.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from blockgen.agentic.core import ImageRequest, Ledger, Charge
from blockgen.agentic.imagegen import get_image_gen
from blockgen.agentic.providers import Message, Part, encode_png, get_provider, text_message

pytestmark = pytest.mark.spend


def test_live_gpt_image_mini_low(tmp_path):
    g = get_image_gen("openai:gpt-image-1-mini?quality=low", cache_dir=tmp_path)
    r = g.generate(ImageRequest("a small stone cottage, isometric, white background"))
    assert r.images and r.images[0][:4] == b"\x89PNG" and r.priced
    assert r.cost_usd == pytest.approx(0.005)
    assert g.generate(ImageRequest("a small stone cottage, isometric, white background")).cached


def test_live_gpt5_mini(tmp_path):
    p = get_provider("openai:gpt-5-mini", cache_dir=tmp_path, spend_gate=True,
                     max_tokens=2000, reasoning_effort="minimal")
    r = p.complete([text_message("user", "Reply with the single word: pong")])
    assert "pong" in r.text.lower() and r.priced
    led = Ledger()
    led.charge(Charge.from_llm("builder", r))
    assert led.summary()["usd"] < 0.01


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_live_claude_code_haiku_reads_image(tmp_path):
    red = encode_png(np.full((64, 64, 3), (220, 20, 20), dtype=np.uint8))
    p = get_provider("claude-code:haiku", cache_dir=tmp_path, spend_gate=True)
    r = p.complete([text_message("system", "You name colours. Answer with one lowercase word."),
                    Message("user", [Part("text", text="What colour is this image?"),
                                     Part("image", image_png=red)])])
    assert "red" in r.text.lower()
    assert r.model == "claude-code/haiku" and not r.priced and r.cost_usd == 0.0
    assert "total_cost_usd" in r.meta
