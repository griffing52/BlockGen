"""Regression tests for the perception/scoring review fixes (2026-10-03).

* dream_palette: shading-aware candidates + block-scale clustering
* texture_sheet{blocks: used}: no wrong-material names, rejected when empty
* foreground_mask: vignette / gradient / contact-shadow backdrops, near-white
  blocks on a pure-white render
* image/text features: an already-loaded backbone's device wins
* material scorer: warns that it optimizes a bench metric
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

from blockgen.agentic import context as ctx
from blockgen.agentic import scoring
from blockgen.agentic.calibrate import attempt_for
from blockgen.agentic.core import RenderError
from blockgen.agentic.providers import encode_png
from blockgen.utils.data import Structure


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bc(name, rgb):
    return ctx.BlockColor(name, (1, 0), tuple(rgb), tuple(ctx.rgb_to_lab(np.array(rgb))))


def _iso_like(rgb, px=192, shades=(1.0, 0.85, 0.70)) -> bytes:
    """A flat-shaded 'iso' cube: top lit, two sides at the renderer's face shades."""
    img = np.full((px, px, 3), 255, np.uint8)
    c = np.asarray(rgb, np.float64)
    img[30:80, 30:160] = (c * shades[0]).round()          # top
    img[80:170, 30:95] = (c * shades[1]).round()          # left side
    img[80:170, 95:160] = (c * shades[2]).round()         # right side
    return encode_png(img)


class _World:
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def block_count(self):
        return len(self.pairs)

    def to_structure(self, crop=True):
        ids = np.zeros((len(self.pairs), 1, 1), np.int32)
        data = np.zeros_like(ids)
        for i, (a, b) in enumerate(self.pairs):
            ids[i], data[i] = a, b
        return Structure(ids, data)


def _ep(pairs=()):
    return types.SimpleNamespace(world=_World(pairs), notes={}, trace=None)


# ---------------------------------------------------------------------------
# dream_palette
# ---------------------------------------------------------------------------
def test_palette_shaded_faces_name_the_lit_block_not_its_darker_neighbour():
    # purple is EXACTLY magenta at 0.79 brightness: unshaded matching named the
    # shaded sides 'purple' and listed it first.
    magenta = (150, 88, 109)
    purple = tuple(int(round(v * 0.79)) for v in magenta)
    cands = [_bc("magenta", magenta), _bc("purple", purple), _bc("white", (233, 236, 236)),
             _bc("black", (21, 21, 26))]
    entries = ctx.match_palette(_iso_like(magenta), 4, candidates=cands)
    assert entries[0].name == "magenta"
    assert entries[0].share > 0.9


def test_palette_clusters_block_colours_not_texel_colours():
    # A fine brick/mortar pattern averages to 'brick'; raw texels named 'mortar'
    # and 'dark' blocks that are not in the build at all.
    brick, mortar = np.array([150, 70, 55]), np.array([200, 195, 185])
    tile = np.empty((4, 4, 3))
    tile[:] = brick
    tile[0] = mortar                                     # one mortar row in four
    mean = tile.reshape(-1, 3).mean(0)
    img = np.full((256, 256, 3), 255, np.uint8)
    img[40:216, 40:216] = np.tile(tile, (44, 44, 1)).astype(np.uint8)
    cands = [_bc("brick_block", mean.round()), _bc("mortar_block", mortar),
             _bc("red_block", brick), _bc("black", (21, 21, 26))]
    entries = ctx.match_palette(encode_png(img), 3, candidates=cands)
    assert entries[0].name == "brick_block" and entries[0].share > 0.9


def _textured_or_skip():
    from blockgen.agentic.render import TexturedRenderer
    try:
        r = TexturedRenderer(px=256)
        ids = np.ones((2, 2, 2), np.int32)
        r.render(Structure(ids, np.zeros_like(ids)), ["iso"])
    except (RenderError, ImportError) as exc:
        pytest.skip(f"textured renderer unavailable here: {exc}")
    return r


def _box(name, n=16):
    from blockgen.agentic.blockstate import try_resolve_block
    i, d = try_resolve_block(name)
    return Structure(np.full((n, n, n), i, np.int32), np.full((n, n, n), d, np.int32))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["magenta_terracotta", "bricks", "quartz_block", "white_wool",
                                  "sandstone", "red_wool", "oak_planks", "birch_planks",
                                  "spruce_planks", "light_gray_wool"])
def test_oracle_single_material_box_lists_that_material_first(name):
    from blockgen.agentic.render import png_bytes
    if not ctx.block_colors():
        pytest.skip("texture pack missing")
    r = _textured_or_skip()
    png = png_bytes(r.render(_box(name), ["iso"])[0])
    assert ctx.match_palette(png, 4)[0].name == name


@pytest.mark.slow
@pytest.mark.parametrize("name", ["stone_bricks", "cobblestone"])
def test_oracle_grey_stone_box_is_in_top_four(name):
    # Grey stones share a mean colour; they must at least be listed.
    from blockgen.agentic.render import png_bytes
    if not ctx.block_colors():
        pytest.skip("texture pack missing")
    r = _textured_or_skip()
    png = png_bytes(r.render(_box(name), ["iso"])[0])
    assert name in [e.name for e in ctx.match_palette(png, 4)]


# ---------------------------------------------------------------------------
# texture_sheet{blocks: used}
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair,wrong", [((44, 7), "stone_slab"), ((126, 5), "oak_slab"),
                                        ((95, 14), "white_stained_glass"),
                                        ((17, 5), "oak_log"), ((18, 7), "oak_leaves")])
def test_used_names_never_names_a_different_material(pair, wrong):
    assert wrong not in ctx._used_names(_ep([pair]))


def test_used_names_masks_orientation_bits_only():
    names = ctx._used_names(_ep([(17, 5), (126, 5), (53, 3), (35, 14), (5, 0)]))
    assert names == ["oak_planks", "spruce_log", "red_wool", "oak_stairs", "dark_oak_planks"]


def test_texture_sheet_used_rejected_before_any_build():
    with pytest.raises(ValueError, match="once before the first build"):
        ctx.texture_sheet(_ep([]), blocks="used")


# ---------------------------------------------------------------------------
# foreground_mask
# ---------------------------------------------------------------------------
_PX = 256


def _synthetic_house():
    """(rgba-ish: rgb, coverage) of a brick-brown house with a light quartz roof."""
    rgb = np.zeros((_PX, _PX, 3), np.float64)
    cov = np.zeros((_PX, _PX), bool)
    rgb[120:200, 70:186] = (150, 80, 60)                   # walls
    cov[120:200, 70:186] = True
    for k in range(8):                                     # stepped light roof
        rgb[64 + 7 * k:71 + 7 * k, 62 + 2 * k:194 - 2 * k] = (232, 228, 222)
        cov[64 + 7 * k:71 + 7 * k, 62 + 2 * k:194 - 2 * k] = True
    rgb[120:200, 70:186] += np.random.default_rng(1).normal(0, 6, (80, 116, 3))
    return rgb, cov


def _on(bg):
    rgb, cov = _synthetic_house()
    out = np.where(cov[..., None], rgb, bg)
    noise = np.random.default_rng(0).normal(0, 2.5, out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8), cov


def _backdrop(kind):
    yy, xx = np.mgrid[0:_PX, 0:_PX]
    r = np.hypot(yy - _PX / 2, xx - _PX / 2) / (_PX / 2 * 1.414)
    if kind == "vignette20":
        return (250 - 20 * r)[..., None] * np.ones(3)
    if kind == "vignette40":
        return (250 - 40 * r)[..., None] * np.ones(3)
    if kind == "sky":
        t = (yy / _PX)[..., None]
        return np.array([150, 190, 235]) + t * np.array([90, 55, 15])
    if kind == "shadow":
        from scipy import ndimage
        ell = (((xx - 128) / 80.0) ** 2 + ((yy - 200) / 14.0) ** 2) <= 1
        sh = ndimage.gaussian_filter(ell.astype(float), 8)
        return np.full((_PX, _PX, 3), 245.0) * (1 - 0.3 * sh[..., None])
    raise KeyError(kind)


@pytest.mark.parametrize("kind", ["vignette20", "vignette40", "sky", "shadow"])
def test_foreground_mask_on_dream_backdrops_matches_clean_silhouette(kind):
    scoring._mask_png.cache_clear()
    dream, cov = _on(_backdrop(kind))
    clean = np.where(cov[..., None], _synthetic_house()[0], 255).clip(0, 255).astype(np.uint8)
    m = scoring.foreground_mask(dream)
    recall = (m & cov).sum() / cov.sum()
    precision = (m & cov).sum() / max(1, m.sum())
    assert recall >= 0.95 and precision >= 0.95, (kind, recall, precision)
    assert scoring.silhouette_iou(encode_png(dream), encode_png(clean)) >= 0.9


def test_foreground_mask_keeps_near_white_blocks_on_a_pure_white_render():
    # Lit snow is within 10 of white and ~46% of its texels ARE pure white: the old
    # tolerance of 10 merged a snow roof into the background.
    rng = np.random.default_rng(0)
    img = np.full((128, 128, 3), 255, np.uint8)
    snow = np.where(rng.random((40, 100, 1)) < 0.46, 255,
                    rng.integers(240, 255, (40, 100, 3))).astype(np.uint8)
    snow[:, :, 0] = np.minimum(snow[:, :, 0], 252)        # never a full pure-white run
    img[20:60, 14:114] = snow
    img[60:110, 14:114] = (150, 80, 60)
    m = scoring.foreground_mask(img)
    assert m[20:110, 14:114].mean() >= 0.97
    assert not m[:20].any() and not m[110:].any()


@pytest.mark.slow
def test_textured_snow_roof_is_foreground():
    from blockgen.agentic.blockstate import try_resolve_block
    r = _textured_or_skip()
    snow, planks = try_resolve_block("snow_block"), try_resolve_block("oak_planks")
    ids = np.zeros((16, 16, 16), np.int32)
    data = np.zeros_like(ids)
    ids[1:-1, :8, 1:-1], data[1:-1, :8, 1:-1] = planks
    for k in range(6):
        ids[1 + k:15 - k, 8 + k, 1:-1], data[1 + k:15 - k, 8 + k, 1:-1] = snow
    s = Structure(ids, data)
    from blockgen.agentic.core import get_view
    from blockgen.agentic.render import DEFAULT_PACK, _GL_LOCK, texture_library
    from blockgen.renderer.textured import render_structure
    lib = texture_library(DEFAULT_PACK)
    for view in ("iso", "top"):
        v = get_view(view)
        with _GL_LOCK:
            rgba = render_structure(s.crop_to_non_air(), px=256, azim_deg=v.azim,
                                    elev_deg=v.elev, ortho=True, bg=None, face_textures=lib)
        truth = rgba[..., 3] > 0
        m = scoring.foreground_mask(np.asarray(r.render(s, [view])[0]))
        assert (m & truth).sum() / truth.sum() >= 0.97, view


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, device):
        self._p = types.SimpleNamespace(device=device)

    def parameters(self):
        yield self._p


def test_image_features_follow_the_loaded_models_device(monkeypatch):
    from blockgen.eval.bench import features as ft
    seen = []
    monkeypatch.setitem(ft._MODELS, "siglipB", (_FakeModel("cpu"), None))
    monkeypatch.setattr(ft, "embed_images",
                        lambda imgs, backbone, device: seen.append(device)
                        or np.ones((len(imgs), 4), np.float32))
    monkeypatch.setattr(scoring, "_WARNED", set())
    scoring._FEAT_CACHE.clear()
    img = np.full((32, 32, 3), 255, np.uint8)
    img[8:24, 8:24] = 0
    with pytest.warns(RuntimeWarning, match="already loaded on cpu"):
        scoring.image_features([encode_png(img)], "siglipB", device="cuda")
    assert seen == ["cpu"]


def test_text_features_follow_the_loaded_models_device(monkeypatch):
    from blockgen.eval.bench import faithfulness as fa
    from blockgen.eval.bench import features as ft
    seen = []
    monkeypatch.setitem(ft._MODELS, "clipB16", (_FakeModel("cpu"), None))
    monkeypatch.setattr(fa, "embed_texts", lambda texts, backbone, device: seen.append(device)
                        or np.ones((len(texts), 4), np.float32))
    monkeypatch.setattr(scoring, "_WARNED", set())
    scoring._FEAT_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scoring.text_features(["a stone tower (device test)"], "clipB16", device="cuda:0")
    assert seen == ["cpu"]


def test_backbone_device_is_the_request_when_nothing_is_loaded(monkeypatch):
    from blockgen.eval.bench import features as ft
    monkeypatch.delitem(ft._MODELS, "dinov2s_unloaded_test", raising=False)
    assert scoring.backbone_device("dinov2s_unloaded_test", "cpu") == "cpu"


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
def test_material_warns_it_optimizes_a_bench_metric(monkeypatch):
    from blockgen.agentic.blockstate import try_resolve_block
    monkeypatch.setattr(scoring, "_WARNED", set())
    i, d = try_resolve_block("stone_bricks")
    s = Structure(np.full((3, 3, 3), i, np.int32), np.full((3, 3, 3), d, np.int32))
    att = attempt_for(s)
    ep = types.SimpleNamespace(prompt="a stone brick tower")
    with pytest.warns(RuntimeWarning, match="material_agreement"):
        term = scoring.material(ep, att)
    assert "optimizes bench metric" in term.note
