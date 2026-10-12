"""imagegen.py: prices, the content-addressed cache, the spend gate, OpenAI wiring
(against a fake client), the lazy diffusers backend and the offline mocks."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from blockgen.agentic import core, imagegen
from blockgen.agentic.core import CacheMiss, ImageRequest, SpendNotAllowed
from blockgen.agentic.imagegen import (IMAGE_PRICES, CachingImageGen, MockImageGen,
                                       OpenAIImageGen, get_image_gen, image_price)
from blockgen.utils.data import Structure

from .fakes import FakeImageGen, solid_png


class PayingFake(FakeImageGen):
    name = "paying_fake"
    spends = True
    billing = "api"


def _req(**kw):
    return ImageRequest(**{"prompt": "a small oak cottage", **kw})


# ===========================================================================
# Prices
# ===========================================================================
def test_prices_and_unknown_is_unpriced():
    assert IMAGE_PRICES[("gpt-image-1-mini", "1024x1024", "low")] == 0.005
    assert IMAGE_PRICES[("gpt-image-1", "1024x1024", "high")] == 0.167
    assert image_price("gpt-image-1-mini", "1024x1024", "low", n=2) == pytest.approx(0.010)
    assert image_price("gpt-image-1-mini", "1024x1024", "low", n_refs=2) > 0.005   # edit surcharge
    assert image_price("gpt-image-9", "1024x1024", "low") is None
    g = OpenAIImageGen("gpt-image-9")
    assert g.estimate_usd(_req()) is None


def test_openai_estimate_uses_effective_quality():
    text = imagegen.INPUT_TOKEN_USD["gpt-image-1"][0] * imagegen._prompt_tokens_upper(
        _req().prompt)      # the prompt's text tokens are billed too
    assert OpenAIImageGen("gpt-image-1").estimate_usd(_req(quality="medium")) == \
        pytest.approx(0.042 + text)
    assert OpenAIImageGen("gpt-image-1", quality="high").estimate_usd(_req()) == \
        pytest.approx(0.167 + text)


# ===========================================================================
# Cache
# ===========================================================================
def test_cache_key_includes_refs_seed_view_and_settings(tmp_path):
    c = CachingImageGen(FakeImageGen(), tmp_path)
    base = c.key(_req())
    assert len({base, c.key(_req(seed=1)), c.key(_req(view="iso_back")),
                c.key(_req(refs=(b"ref",))), c.key(_req(refs=(b"other",))),
                c.key(_req(quality="medium"))}) == 6
    # generator overrides land in the key (what was generated, not what was asked)
    o1 = CachingImageGen(OpenAIImageGen("gpt-image-1-mini", quality="medium"), tmp_path)
    o2 = CachingImageGen(OpenAIImageGen("gpt-image-1-mini"), tmp_path)
    assert o1.key(_req(quality="low")) == o2.key(_req(quality="medium"))
    o3 = CachingImageGen(OpenAIImageGen("gpt-image-1-mini", background="transparent"), tmp_path)
    assert o3.key(_req()) != o2.key(_req())


def test_hit_is_cached_with_stored_price(tmp_path):
    inner = FakeImageGen(usd=0.01)
    c = CachingImageGen(inner, tmp_path)
    a = c.generate(_req(n=2))
    b = c.generate(_req(n=2))
    assert not a.cached and b.cached and inner.calls == 1
    assert b.images == a.images and len(b.images) == 2
    assert b.cost_usd == pytest.approx(0.02) and b.priced
    names = sorted(p.name for p in tmp_path.iterdir())
    key = c.key(_req(n=2))
    assert names == sorted([f"{key}.json", f"{key}.png", f"{key}_1.png"])
    # a fresh wrapper (new process) hits the same files
    assert CachingImageGen(FakeImageGen(), tmp_path).generate(_req(n=2)).cached
    charge = core.Charge.from_image("image", b)
    assert charge.usd == 0.0 and charge.would_cost_usd == pytest.approx(0.02)


def test_read_only_miss(tmp_path):
    inner = FakeImageGen()
    c = CachingImageGen(inner, tmp_path, mode="read_only")
    with pytest.raises(CacheMiss):
        c.generate(_req())
    assert inner.calls == 0
    CachingImageGen(inner, tmp_path).generate(_req())
    assert c.generate(_req()).cached


def test_mode_off_is_passthrough(tmp_path):
    inner = FakeImageGen()
    c = CachingImageGen(inner, tmp_path, mode="off")
    c.generate(_req()); c.generate(_req())
    assert inner.calls == 2 and not list(tmp_path.iterdir()) and not c.peek(_req())
    with pytest.raises(ValueError):
        CachingImageGen(inner, tmp_path, mode="bogus")


def test_spend_gate(tmp_path, monkeypatch):
    inner = PayingFake()
    c = CachingImageGen(inner, tmp_path, spend_gate=True)
    with pytest.raises(SpendNotAllowed):
        c.generate(_req())
    with pytest.raises(SpendNotAllowed):                       # 'off' is gated too
        CachingImageGen(inner, tmp_path, mode="off").generate(_req())
    assert inner.calls == 0
    assert not CachingImageGen(inner, tmp_path, spend_gate=False).generate(_req()).cached
    assert c.generate(_req()).cached                           # hits never need permission
    monkeypatch.setenv(core.SPEND_ENV, "1")
    c.generate(_req(seed=5))
    assert inner.calls == 2


def test_peek(tmp_path):
    c = CachingImageGen(FakeImageGen(), tmp_path)
    assert not c.peek(_req())
    c.generate(_req())
    assert c.peek(_req()) and not c.peek(_req(seed=9))


def test_atomic_writes(tmp_path, monkeypatch):
    c = CachingImageGen(FakeImageGen(), tmp_path)
    real = os.replace

    def flaky(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("disk full")
        real(src, dst)
    monkeypatch.setattr(imagegen.os, "replace", flaky)
    with pytest.raises(OSError):
        c.generate(_req())
    assert not list(tmp_path.glob("*.json")) and not list(tmp_path.glob("*.tmp"))
    assert not c.peek(_req())                                  # PNG alone is not an entry
    monkeypatch.setattr(imagegen.os, "replace", real)
    assert not c.generate(_req()).cached and c.generate(_req()).cached


def test_damaged_entry_regenerates(tmp_path):
    inner = FakeImageGen()
    c = CachingImageGen(inner, tmp_path)
    c.generate(_req())
    (tmp_path / f"{c.key(_req())}.json").write_text("{not json")
    assert not c.generate(_req()).cached and inner.calls == 2


def test_wrapper_delegates_protocol(tmp_path):
    c = CachingImageGen(PayingFake(usd=0.03), tmp_path)
    assert isinstance(c, core.ImageGen)
    assert (c.name, c.model, c.spends, c.billing) == ("paying_fake", "fake-solid", True, "api")
    assert c.estimate_usd(_req()) == 0.03 and c.supports_refs() and c.usd == 0.03
    assert core.billing_of(c) == "api"
    c.unload()                                                 # absent on inner: no-op


# ===========================================================================
# get_image_gen and the registry
# ===========================================================================
def test_registry_names():
    assert {"openai", "mock", "diffusers"} <= set(core.names("image"))
    assert core.lookup("image", "local-image").name == "diffusers"
    assert core.meta("image", "openai")["env"] == "OPENAI_API_KEY"


def test_get_image_gen_wraps_and_mocks_use_memory(tmp_path):
    g = get_image_gen("openai:gpt-image-1-mini?quality=medium", cache_dir=tmp_path)
    assert isinstance(g, CachingImageGen) and isinstance(g.inner, OpenAIImageGen)
    assert g.inner.quality == "medium" and g.spend_gate and not g.memory
    m = get_image_gen({"type": "mock", "arg": "solid"}, cache_dir=tmp_path)
    assert m.memory and m.describe() == "mock:solid"
    m.generate(_req())
    assert m.generate(_req()).cached and not list(tmp_path.iterdir())
    ro = get_image_gen("mock:solid", cache="read_only", spend_gate=False)
    assert ro.mode == "read_only" and not ro.spend_gate


# ===========================================================================
# OpenAI (fake client; the network is never touched)
# ===========================================================================
class FakeImages:
    def __init__(self):
        self.calls = []

    def _resp(self, n):
        png = solid_png((10, 20, 30))
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(),
                                                     revised_prompt=None) for _ in range(n)])

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        return self._resp(kw["n"])

    def edit(self, **kw):
        self.calls.append(("edit", kw))
        return self._resp(kw["n"])


def test_openai_gated_before_client(tmp_path, monkeypatch):
    def boom(self):
        raise AssertionError("client constructed")
    monkeypatch.setattr(OpenAIImageGen, "_client_lazy", boom)
    g = get_image_gen("openai:gpt-image-1-mini", cache_dir=tmp_path)
    with pytest.raises(SpendNotAllowed, match="openai:gpt-image-1-mini"):
        g.generate(_req())


def test_openai_generate_and_edit_wiring(tmp_path, monkeypatch):
    fake = FakeImages()
    gen = OpenAIImageGen("gpt-image-1-mini")
    gen._client = SimpleNamespace(images=fake)
    r = gen.generate(_req(quality="low", n=2))
    kind, kw = fake.calls[-1]
    assert kind == "generate" and kw == {"model": "gpt-image-1-mini", "prompt": "a small oak cottage",
                                         "n": 2, "size": "1024x1024", "quality": "low",
                                         "background": "opaque"}
    assert len(r.images) == 2 and r.images[0] == solid_png((10, 20, 30))
    text = imagegen.INPUT_TOKEN_USD["gpt-image-1-mini"][0] * imagegen._prompt_tokens_upper(
        "a small oak cottage")
    assert r.priced and r.cost_usd == pytest.approx(0.010 + text) and not r.cached
    r = gen.generate(_req(refs=(b"r1", b"r2"), background="transparent"))
    kind, kw = fake.calls[-1]
    assert kind == "edit" and kw["background"] == "transparent"
    assert [f[0] for f in kw["image"]] == ["ref0.png", "ref1.png"]
    assert kw["image"][1][1].read() == b"r2"
    assert r.cost_usd == pytest.approx(0.005 + 2 * imagegen.REF_SURCHARGE_USD["gpt-image-1-mini"]
                                       + text)


# ===========================================================================
# diffusers (lazy, optional)
# ===========================================================================
def test_diffusers_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", None)        # import -> ImportError
    with pytest.raises(ImportError, match="pip install diffusers accelerate"):
        core.make("image", "diffusers:sdxl-turbo")


def test_importing_imagegen_does_not_import_heavy_deps():
    code = ("import sys, blockgen.agentic.imagegen as m; "
            "bad = [x for x in ('diffusers', 'torch', 'openai') if x in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=imagegen._REPO, timeout=120)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("BLOCKGEN_TEST_DIFFUSERS") != "1",
                    reason="downloads/loads sdxl-turbo on the GPU; set BLOCKGEN_TEST_DIFFUSERS=1")
def test_diffusers_generates_deterministically():
    g = core.make("image", "diffusers:sdxl-turbo?offload=true")
    a = g.generate(_req(size="512x512", seed=1))
    b = g.generate(_req(size="512x512", seed=1))
    assert a.images == b.images and not a.priced
    g.unload()


def test_diffusers_meta_without_loading(monkeypatch):
    """Construction is cheap (no weights) when the packages import."""
    pytest.importorskip("diffusers")
    g = core.make("image", "diffusers:flux-schnell?steps=2")
    assert g.model_id == "black-forest-labs/FLUX.1-schnell" and g.billing == "local"
    assert not g.spends and g.estimate_usd(_req()) is None and g.describe() == "diffusers:flux-schnell"
    assert CachingImageGen(g).model_key() != \
        CachingImageGen(core.make("image", "diffusers:flux-schnell?steps=4")).model_key()


# ===========================================================================
# Mocks
# ===========================================================================
def test_mock_solid_deterministic_and_varies():
    g = MockImageGen("solid")
    a, b = g.generate(_req(seed=1)), g.generate(_req(seed=1))
    assert a.images == b.images and not a.priced and a.cost_usd == 0.0
    assert a.model == "mock-solid"
    assert g.generate(_req(seed=2)).images != a.images
    assert g.generate(_req(seed=1, refs=(b"x",))).images != a.images
    two = g.generate(_req(n=2))
    assert len(two.images) == 2 and two.images[0] != two.images[1]
    from PIL import Image
    assert Image.open(io.BytesIO(MockImageGen("solid", px=64).generate(_req()).images[0])).size \
        == (64, 64)


def test_mock_dir(tmp_path):
    for i in range(3):
        (tmp_path / f"d{i}.png").write_bytes(solid_png((i * 50, 0, 0)))
    g = core.make("image", f"mock:dir?path={tmp_path}")
    a = g.generate(_req(seed=4))
    assert a.images == g.generate(_req(seed=4)).images
    pool = {(tmp_path / f"d{i}.png").read_bytes() for i in range(3)}
    assert a.images[0] in pool
    picks = {g.generate(_req(seed=s)).images[0] for s in range(12)}
    assert len(picks) > 1
    with pytest.raises(ValueError, match="path"):
        MockImageGen("dir")
    with pytest.raises(FileNotFoundError):
        MockImageGen("dir", path=str(tmp_path / "empty")).generate(_req())


def test_mock_bad_mode_and_test_split_refused():
    with pytest.raises(ValueError):
        MockImageGen("glitter")
    with pytest.raises(ValueError, match="leak"):
        MockImageGen("render", split="test")


def _house(h: int) -> Structure:
    ids = np.zeros((6, 8, 6), dtype=np.int32)
    ids[:, 0, :] = 4                     # cobblestone floor
    ids[0, 1:h, :] = 5                   # planks wall
    ids[:, 1:h, 0] = 5
    return Structure(ids, np.zeros_like(ids), air_block_id=0)


def test_mock_render_from_pool_with_renderer(monkeypatch):
    pool = [_house(3), _house(7)]
    g = MockImageGen("render", structures=pool, renderer="fake_renderer", px=48)
    a = g.generate(_req(view="iso"))
    assert a.images == g.generate(_req(view="iso")).images and g.renderer_used == "fake_renderer"
    from PIL import Image
    assert Image.open(io.BytesIO(a.images[0])).size == (64, 64)   # FakeRenderer pins its own px
    assert g.generate(_req(view="plan")).images != a.images     # the view matters
    k1 = CachingImageGen(g).model_key()
    assert "fake_renderer" in k1 and k1 == CachingImageGen(g).model_key()


def test_mock_render_numpy_fallback(monkeypatch):
    def no_renderers(kind, spec, **kw):
        raise KeyError(f"unknown {kind} {spec}")
    monkeypatch.setattr(imagegen, "make", no_renderers)
    g = MockImageGen("render", structures=[_house(5)], px=40)
    with pytest.warns(UserWarning, match="falling back"):
        img = g.generate(_req()).images[0]
    assert g.renderer_used == "numpy"
    from PIL import Image
    arr = np.asarray(Image.open(io.BytesIO(img)))
    assert arr.shape == (40, 40, 3) and (arr < 250).any() and (arr == 255).any()
    # cutaway view removes the top of the build -> fewer non-white pixels
    cut = np.asarray(Image.open(io.BytesIO(g.generate(_req(view="plan")).images[0])))
    assert (cut < 250).any(axis=-1).sum() < (arr < 250).any(axis=-1).sum()


@pytest.mark.slow
@pytest.mark.skipif(not (imagegen._REPO / "data/minecraft/cache/houses_32.npz").exists(),
                    reason="houses_32 corpus cache not present")
def test_mock_render_real_train_split():
    g = core.make("image", "mock:render?renderer=flat&px=64")
    a = g.generate(_req())
    assert a.images == g.generate(_req()).images
    pool = imagegen._train_pool("houses_32", "train")
    assert len(pool) == 1863


# ===========================================================================
# Verifier regressions
# ===========================================================================
def _episode_request(gen):
    """How Episode.dream builds its request from the (cache-wrapped) generator."""
    return ImageRequest("a cottage", size=str(getattr(gen, "size", "1024x1024")),
                        quality=str(getattr(gen, "quality", "low")))


def test_openai_attributes_are_concrete_for_episode(tmp_path):
    """None quality/size used to reach the API, the key and the price as 'None'."""
    g = get_image_gen("openai:gpt-image-1-mini", cache_dir=tmp_path)
    req = _episode_request(g)
    assert (req.size, req.quality) == ("1024x1024", "low")
    # (+ abs tolerance: the estimate also prices the prompt's few text tokens)
    assert g.estimate_usd(req) == pytest.approx(0.005, abs=1e-4)
    o = get_image_gen("openai:gpt-image-1?quality=medium&size=1536x1024", cache_dir=tmp_path)
    assert o.estimate_usd(_episode_request(o)) == pytest.approx(0.063, abs=1e-4)
    # an explicit request value still wins when the generator sets no override
    assert OpenAIImageGen("gpt-image-1").estimate_usd(_req(quality="high")) == \
        pytest.approx(0.167, abs=1e-4)


def test_mock_memory_cache_read_only_still_generates():
    """Mocks are free: `--mock --cache read_only` must not die on the first dream
    (mock LLMs are never cached either)."""
    g = get_image_gen("mock:solid", cache="read_only")
    first = g.generate(_req())
    assert not first.cached and g.generate(_req()).cached
    assert g.peek(_req())


def test_store_records_effective_request(tmp_path):
    import json as _json
    inner = OpenAIImageGen("gpt-image-1-mini", quality="medium", size="1024x1536")
    c = CachingImageGen(inner, tmp_path, spend_gate=False)
    res = core.ImageResult(images=[solid_png((1, 2, 3), 8)], model="gpt-image-1-mini",
                           cost_usd=0.015)
    c._store(c.key(_req()), c._effective(_req()), res)
    blob = _json.loads(next(tmp_path.glob("*.json")).read_text())
    assert (blob["quality"], blob["size"]) == ("medium", "1024x1536")


def test_diffusers_dtype_auto_and_native_size():
    """FLUX overflows in fp16; sdxl-turbo is distilled at 512px."""
    pytest.importorskip("diffusers")
    flux = core.make("image", "diffusers:flux-schnell")
    turbo = core.make("image", "diffusers:stabilityai/sdxl-turbo?steps=4")
    assert flux.dtype == "bf16" and turbo.dtype == "fp16"
    assert core.make("image", "diffusers:flux-schnell?dtype=fp32").dtype == "fp32"
    with pytest.raises(ValueError):
        core.make("image", "diffusers:sdxl-turbo?dtype=fp8")
    assert _episode_request(turbo).size == "512x512" and flux.size == "1024x1024"
    assert core.make("image", "diffusers:sdxl-turbo?size=768x768").size == "768x768"
    # no override: the request's size is what is generated (and keyed)
    assert turbo.effective(_req(size="256x256")).size == "256x256"
