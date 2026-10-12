"""WP5 renderers: flat / null offline, textured behind the `slow` marker (EGL)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from blockgen.agentic import core, render
from blockgen.agentic.core import VIEWS, EmptyBuild, RenderError, View
from blockgen.agentic.render import (FlatRenderer, NullRenderer, TexturedRenderer, null_image,
                                     png_bytes, png_to_rgb, render_pngs, to_rgb)
from blockgen.utils.data import Structure


def tower(h: int = 12, roof_id: int = 45) -> Structure:
    """5x5 stone tower of height h with a brick cap: roof colour is easy to find."""
    ids = np.zeros((7, h + 2, 7), np.int32)
    ids[1:6, 0:h, 1:6] = 1
    ids[2:5, 1:h, 2:5] = 0
    ids[1:6, h, 1:6] = roof_id
    return Structure(ids, np.zeros_like(ids))


def empty() -> Structure:
    ids = np.zeros((4, 4, 4), np.int32)
    return Structure(ids, np.zeros_like(ids))


def _textured_or_skip(**kw) -> TexturedRenderer:
    try:
        r = TexturedRenderer(**kw)
        r.render(tower(3), ["iso"])
    except (RenderError, ImportError) as exc:
        pytest.skip(f"textured renderer unavailable here: {exc}")
    return r


# --- registry --------------------------------------------------------------
def test_registered_names():
    assert {"textured", "flat", "null"} <= set(core.names("renderer"))
    r = core.make("renderer", "flat?px=48")
    assert isinstance(r, FlatRenderer) and r.px == 48
    with pytest.raises(TypeError):
        core.make("renderer", "flat?pixels=48")


# --- flat / null: every view -------------------------------------------------
@pytest.mark.parametrize("cls", [FlatRenderer, NullRenderer])
def test_every_view_shape_and_dtype(cls):
    r = cls(px=40)
    imgs = r.render(tower(), list(VIEWS))
    assert len(imgs) == len(VIEWS)
    for im in imgs:
        assert im.shape == (40, 40, 3) and im.dtype == np.uint8


def test_flat_uses_view_px_when_unset():
    im, = FlatRenderer().render(tower(), [View("tiny", 45, 30, px=33)])
    assert im.shape == (33, 33, 3)


def test_flat_white_background_and_foreground():
    im, = FlatRenderer(px=64).render(tower(), ["iso"])
    assert (im[0, 0] == 255).all() and (im[-1, -1] == 255).all()
    fg = (im < 250).any(-1)
    assert 0.1 < fg.mean() < 0.9


def test_flat_deterministic_and_view_dependent():
    a = FlatRenderer(px=64).render(tower(), ["iso", "iso_back", "top"])
    b = FlatRenderer(px=64).render(tower(), ["iso", "iso_back", "top"])
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], a[2])


def test_flat_without_textures_still_renders():
    im, = FlatRenderer(px=32, textures=False).render(tower(), ["iso"])
    assert (im < 250).any()


def test_null_is_view_keyed_solid_tile():
    r = NullRenderer(px=16)
    a, b = r.render(tower(), ["iso", "top"])
    assert np.array_equal(a, null_image("iso", 16))
    assert len(np.unique(a.reshape(-1, 3), axis=0)) == 1
    assert not np.array_equal(a, b)
    assert not (a == 255).all()


@pytest.mark.parametrize("cls", [FlatRenderer, NullRenderer])
def test_empty_build_raises(cls):
    with pytest.raises(EmptyBuild):
        cls(px=16).render(empty(), ["iso"])


def test_textured_empty_raises_before_gl():
    if not render.DEFAULT_PACK.is_dir():
        pytest.skip("no texture pack")
    with pytest.raises(EmptyBuild):
        TexturedRenderer().render(empty(), ["iso"])


# --- view transforms ---------------------------------------------------------
def test_cutaway_and_plan_reduce_visible_height():
    s = tower(12)
    full_h = s.crop_to_non_air().shape[1]
    for name in ("cutaway", "plan"):
        cut = VIEWS[name].apply(s).crop_to_non_air()
        assert cut.shape[1] < full_h
    assert VIEWS["plan"].apply(s).crop_to_non_air().shape[1] < \
        VIEWS["cutaway"].apply(s).crop_to_non_air().shape[1]


def test_cutaway_render_removes_roof_colour():
    """The brick cap is visible from above in 'top' and gone in a cutaway rendered
    from the same camera."""
    r = FlatRenderer(px=64)
    roof = np.array(render.block_rgb(45, 0, "top"))
    top, = r.render(tower(), ["top"])
    cut, = r.render(tower(), [View("topcut", 0, 89, keep_below=0.5)])

    def roof_px(im):
        return int((np.abs(im.astype(int) - roof).max(-1) <= 2).sum())
    assert roof_px(top) > 100
    assert roof_px(cut) == 0


def test_section_halves_the_axis():
    s = tower()
    sec = VIEWS["section_x"].apply(s).crop_to_non_air()
    assert sec.shape[0] < s.crop_to_non_air().shape[0]
    im, = FlatRenderer(px=32).render(s, ["section_x"])
    assert (im < 250).any()


# --- helpers -------------------------------------------------------------------
def test_png_roundtrip_and_alpha_composite():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:2, :, 3] = 255                     # top half opaque red, bottom transparent
    rgb = png_to_rgb(png_bytes(rgba))
    assert rgb.shape == (4, 4, 3)
    assert tuple(rgb[0, 0]) == (200, 0, 0)
    assert tuple(rgb[3, 3]) == (255, 255, 255)
    assert not rgb.flags.writeable           # shared through the decode cache
    assert np.array_equal(to_rgb(rgb), rgb)


def test_render_pngs_keys_by_view_name():
    out = render_pngs(FlatRenderer(px=24), tower(), ["iso", VIEWS["top"]])
    assert set(out) == {"iso", "top"}
    assert png_to_rgb(out["iso"]).shape == (24, 24, 3)


def test_camera_basis_orthonormal():
    r, u, d = render.camera_basis(45, 30)
    for a, b in ((r, u), (r, d), (u, d)):
        assert abs(float(a @ b)) < 1e-9
    assert u[1] > 0


# --- textured -------------------------------------------------------------------
def test_textured_missing_pack_fails_loudly(tmp_path):
    with pytest.raises(RenderError, match="texture pack"):
        TexturedRenderer(pack_dir=str(tmp_path / "nope"))
    (tmp_path / "emptypack").mkdir()
    with pytest.raises(RenderError):
        TexturedRenderer(pack_dir=str(tmp_path / "emptypack"))


def test_textured_gl_failure_is_render_error(monkeypatch):
    if not render.DEFAULT_PACK.is_dir():
        pytest.skip("no texture pack")
    import sys
    import types

    def boom(*a, **k):
        raise RuntimeError("driver hiccup")
    fake = types.ModuleType("blockgen.renderer.textured")
    fake.render_structure = boom
    monkeypatch.setitem(sys.modules, "blockgen.renderer.textured", fake)
    with pytest.raises(RenderError, match="driver hiccup"):
        TexturedRenderer().render(tower(), ["iso"])


def test_texture_library_shared():
    if not render.DEFAULT_PACK.is_dir():
        pytest.skip("no texture pack")
    assert render.texture_library() is render.texture_library(str(render.DEFAULT_PACK))


@pytest.mark.slow
def test_textured_renders_every_view():
    r = _textured_or_skip(px=96)
    imgs = r.render(tower(), list(VIEWS))
    assert len(imgs) == len(VIEWS)
    for im in imgs:
        assert im.shape == (96, 96, 3) and im.dtype == np.uint8
        assert (im[0, 0] >= 250).all()          # white background
        assert (im < 240).any()                 # something drawn
    assert r.fallback_frac == 0.0


@pytest.mark.slow
def test_textured_and_flat_share_a_silhouette():
    from blockgen.agentic.scoring import foreground_mask
    r = _textured_or_skip(px=96)
    s = tower()
    for v in ("iso", "iso_back", "front", "top"):
        t, = r.render(s, [v])
        f, = FlatRenderer(px=96).render(s, [v])
        mt, mf = foreground_mask(t), foreground_mask(f)
        iou = (mt & mf).sum() / max(1, (mt | mf).sum())
        assert iou > 0.85, (v, iou)


@pytest.mark.slow
def test_textured_thread_safe():
    r = _textured_or_skip(px=64)
    out, errs = [], []

    def work():
        try:
            out.append(r.render(tower(), ["iso", "top"]))
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)
    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs and len(out) == 4
    for o in out[1:]:
        assert np.array_equal(o[0], out[0][0])


@pytest.mark.slow
def test_textured_warns_once_on_fallback_textures(monkeypatch):
    r = _textured_or_skip(px=32)
    monkeypatch.setattr(render, "_WARNED_FALLBACK", False)
    ids = np.zeros((3, 3, 3), np.int32)
    ids[:, 0, :] = 250                           # an id with no texture mapping
    s = Structure(ids, np.zeros_like(ids))
    with pytest.warns(RuntimeWarning, match="fallback"):
        r.render(s, ["iso"])
    assert r.fallback_frac == 1.0
