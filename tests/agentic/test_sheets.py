"""Turnaround sheets (sheets.py) and the mock sheet images (imagegen.MockImageGen).

Covers the WP-C acceptance list for the splitter: split(compose(x)) recovers the
four panels (IoU > 0.9) with even gutters, uneven gutters and a panel that
touches its gutter; a single building is rejected; the cyclic assignment
recovers a known rotation (and a reversed sheet) on real flat renders.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from blockgen.agentic import scoring, sheets, turns
from blockgen.agentic.core import Lowered, MinecraftWorld, get_view
from blockgen.agentic.imagegen import GRID_HINT, MockImageGen, parse_grid_hint
from blockgen.agentic.providers import encode_png
from blockgen.agentic.render import FlatRenderer

ORDER = ("iso", "iso_left", "iso_back", "iso_right")

#: An asymmetric build (a long hall, a corner tower, a red block on one side), so
#: its four iso silhouettes differ and a wrong camera assignment is detectable.
ASYM = ("fill stone 0 0 0 11 0 7\nwalls oak_planks 0 1 0 11 3 7\n"
        "fill cobblestone 0 0 0 2 12 2\nfill red_wool 9 4 5 11 5 7\n")


def asym_world() -> MinecraftWorld:
    w = MinecraftWorld.empty((32, 32, 32))
    w.run(Lowered(ASYM, ASYM))
    return w


@pytest.fixture(scope="module")
def renders():
    s = asym_world().to_structure()
    r = FlatRenderer(px=192)
    return {v: encode_png(r.render(s, [get_view(v)])[0]) for v in ORDER}


def mask_iou(a: bytes, b: bytes) -> float:
    ma = scoring.normalized_mask(scoring._mask_png(a, scoring.FG_THRESH))
    mb = scoring.normalized_mask(scoring._mask_png(b, scoring.FG_THRESH))
    return float((ma & mb).sum()) / float((ma | mb).sum())


def blob(rgb, w: int = 120, h: int = 90, px: int = 160) -> bytes:
    """A coloured rectangle on white."""
    arr = np.full((px, px, 3), 255, np.uint8)
    y0, x0 = (px - h) // 2, (px - w) // 2
    arr[y0:y0 + h, x0:x0 + w] = rgb
    return encode_png(arr)


def size_of(png: bytes):
    return Image.open(io.BytesIO(png)).size


# ===========================================================================
# split / compose
# ===========================================================================
def test_split_recovers_composed_panels(renders):
    tiles = sheets.split_grid(sheets.compose_grid([renders[v] for v in ORDER]))
    assert len(tiles) == 4
    for v, t in zip(ORDER, tiles):
        assert mask_iou(t, renders[v]) > 0.9
        w, h = size_of(t)
        assert w == h                                  # padded square
    assert sheets.tiles_ok(tiles)


def test_split_with_uneven_gutters_and_a_panel_touching_its_gutter():
    """Hand-made sheet: gutters at 40% (top row) and 60% (bottom row) of the width,
    the row gutter at 45% of the height, and the top-left panel running right up to
    its gutter. Each panel must still come back whole."""
    px = 600
    arr = np.full((px, px, 3), 255, np.uint8)
    cols = [(0, 0, 0), (200, 30, 30), (30, 160, 30), (30, 30, 200)]
    boxes = [(30, 60, 220, 230),      # (y0, x0, y1, x1); x1 = 230 touches the gutter at 230..250
             (40, 300, 230, 560),
             (310, 40, 560, 330),
             (300, 390, 570, 560)]
    for (y0, x0, y1, x1), c in zip(boxes, cols):
        arr[y0:y1, x0:x1] = c
    arr[0:270, 230:250] = 255         # top row gutter (x 230..249)
    arr[270:290] = 255                # row gutter
    arr[290:, 350:370] = 255          # bottom row gutter
    tiles = sheets.split_grid(encode_png(arr))
    for (y0, x0, y1, x1), c, t in zip(boxes, cols, tiles):
        ref = np.full((px, px, 3), 255, np.uint8)
        ref[y0:y1, x0:x1] = c
        assert mask_iou(t, encode_png(ref)) > 0.9
        rgb = np.asarray(Image.open(io.BytesIO(t)).convert("RGB"))
        fg = rgb[(rgb != 255).any(-1)]
        assert (np.abs(fg.astype(int) - np.array(c)).max(axis=1) <= 2).mean() > 0.95
    assert sheets.tiles_ok(tiles)


@pytest.mark.parametrize("backdrop", ["flat", "gradient"])
def test_split_mattes_an_off_white_backdrop(renders, backdrop):
    """An image model's backdrop is rarely pure white. The tile is padded with
    white, so a kept off-white margin would read as part of the building (the
    silhouette became the crop rectangle, IoU ~0.5). The split mattes it away."""
    sheet = np.asarray(Image.open(io.BytesIO(
        sheets.compose_grid([renders[v] for v in ORDER]))).convert("RGB")).astype(int)
    rng = np.random.default_rng(0)
    if backdrop == "flat":
        back = 238 + rng.integers(-3, 4, size=sheet.shape)
    else:
        yy = np.linspace(0.0, 1.0, sheet.shape[0])[:, None, None]
        back = 250 - 18 * yy + rng.integers(-3, 4, size=sheet.shape)
    white = (sheet == 255).all(-1, keepdims=True)
    noisy = np.where(white, back, sheet).clip(0, 255).astype(np.uint8)
    tiles = sheets.split_grid(encode_png(noisy))
    assert sheets.tiles_ok(tiles)
    for v, t in zip(ORDER, tiles):
        assert scoring.silhouette_iou(t, renders[v]) > 0.9, v


def test_split_falls_back_to_exact_fractions_on_a_blank_image():
    tiles = sheets.split_grid(encode_png(np.full((200, 200, 3), 255, np.uint8)))
    assert len(tiles) == 4
    assert not sheets.tiles_ok(tiles)                  # nothing in them


def test_tiles_carry_their_cut_fraction():
    tiles = sheets.split_grid(sheets.compose_grid([blob((200, 0, 0))] * 4))
    assert all(sheets.tile_meta(t)[sheets.CUT_KEY] == 0.0 for t in tiles)
    assert sheets.tile_meta(blob((1, 2, 3))) == {}     # an ordinary PNG has none


def test_tiles_ok_rejects_a_single_building(renders):
    tiles = sheets.split_grid(renders["iso"])
    assert not sheets.tiles_ok(tiles)
    ok, why = sheets.tile_check(tiles[0])
    assert not ok and "sliced" in why


def test_tiles_ok_rejects_tiny_full_and_scattered_tiles():
    tiny = blob((200, 0, 0), w=4, h=4, px=200)
    full = encode_png(np.full((50, 50, 3), (90, 90, 200), np.uint8))
    arr = np.full((200, 200, 3), 255, np.uint8)
    for k in range(4):
        arr[20 + 45 * k:50 + 45 * k, 20 + 45 * k:50 + 45 * k] = (0, 0, 0)
    scattered = encode_png(arr)
    assert not sheets.tile_check(tiny)[0]
    assert not sheets.tile_check(full)[0]
    ok, why = sheets.tile_check(scattered)
    assert not ok and "component" in why
    assert not sheets.tiles_ok([])


# ===========================================================================
# assignment
# ===========================================================================
def test_cyclic_orders_are_four_rotations_in_two_directions():
    cands = sheets.cyclic_orders(ORDER)
    assert len(cands) == 8 and len(set(cands)) == 8
    assert cands[0] == ORDER
    assert ("iso_left", "iso_back", "iso_right", "iso") in cands
    assert ("iso", "iso_right", "iso_back", "iso_left") in cands     # reversed
    assert ("iso", "iso_back", "iso_left", "iso_right") not in cands  # not a turnaround


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_best_assignment_recovers_a_rotation(renders, k):
    drawn = ORDER[k:] + ORDER[:k]                      # panel i shows view drawn[i]
    tiles = sheets.split_grid(sheets.compose_grid([renders[v] for v in drawn]))
    chosen, value = sheets.best_assignment(tiles, renders, ORDER)
    assert chosen == drawn
    assert value > 0.9
    assert value >= sheets.assignment_score(tiles, renders, ORDER)


def test_best_assignment_recovers_a_reversed_sheet(renders):
    drawn = ("iso", "iso_right", "iso_back", "iso_left")
    tiles = sheets.split_grid(sheets.compose_grid([renders[v] for v in drawn]))
    assert sheets.best_assignment(tiles, renders, ORDER)[0] == drawn


def test_best_assignment_keeps_the_order_on_a_tie_and_without_renders():
    tiles = sheets.split_grid(sheets.compose_grid([blob((200, 0, 0))] * 4))
    same = {v: blob((0, 0, 200)) for v in ORDER}
    assert sheets.best_assignment(tiles, same, ORDER)[0] == ORDER
    assert sheets.best_assignment(tiles, {}, ORDER) == (ORDER, 0.0)


# ===========================================================================
# review turn
# ===========================================================================
class _Ep:
    prompt = "a stone tower"
    size = (32, 32, 32)


def test_sheet_review_turn_has_the_tag_the_image_and_ends_with_the_json_rule():
    msg = sheets.sheet_review(_Ep(), blob((1, 2, 3)), ORDER)
    assert sheets.SHEET_REVIEW_TAG in msg.text
    assert msg.parts[-1].text == turns.JSON_RULE
    assert sum(p.kind == "image" for p in msg.parts) == 1
    assert "panel 3 (bottom-left): view iso_back" in msg.text


def test_normalize_sheet_verdict_defaults():
    v = sheets.normalize_sheet_verdict({"score": "6/10", "consistent": "no", "issues": "x"})
    assert v == {"consistent": False, "score": 6.0, "issues": ["x"], "revised_prompt": ""}
    assert sheets.normalize_sheet_verdict({})["consistent"] is True


def test_sheet_template_fills_every_field():
    for f in ("{style}", "{desc}", "{W}", "{H}", "{D}"):
        assert f in sheets.SHEET_TEMPLATE


# ===========================================================================
# mock image generator
# ===========================================================================
def req(view: str, prompt: str = "a hut"):
    from blockgen.agentic.core import ImageRequest
    return ImageRequest(prompt, seed=1, view=view)


def test_parse_grid_hint():
    assert parse_grid_hint("iso") is None
    assert parse_grid_hint(GRID_HINT + ",".join(ORDER)) == list(ORDER)
    assert parse_grid_hint(GRID_HINT + "iso") == ["iso", "iso", "iso", "iso"]


def test_mock_solid_sheet_splits_into_four_distinct_panels():
    gen = MockImageGen("solid", px=96)
    png = gen.generate(req(GRID_HINT + ",".join(ORDER))).images[0]
    tiles = sheets.split_grid(png)
    assert sheets.tiles_ok(tiles)
    def centre(t: bytes):
        rgb = np.asarray(Image.open(io.BytesIO(t)).convert("RGB"))
        return tuple(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2])
    colours = {centre(t) for t in tiles}
    assert len(colours) == 4
    # a plain request is unchanged: one full-colour placeholder
    plain = np.asarray(Image.open(io.BytesIO(gen.generate(req("iso")).images[0])))
    assert plain.shape[:2] == (96, 96)


def test_mock_render_sheet_shows_one_house_at_the_four_views(renders):
    s = asym_world().to_structure()
    gen = MockImageGen("render", renderer="flat", px=192, structures=[s])
    png = gen.generate(req(GRID_HINT + ",".join(ORDER))).images[0]
    tiles = sheets.split_grid(png)
    assert sheets.tiles_ok(tiles)
    for v, t in zip(ORDER, tiles):
        assert mask_iou(t, renders[v]) > 0.9


def test_mock_render_unknown_view_falls_back_to_iso(renders):
    s = asym_world().to_structure()
    gen = MockImageGen("render", renderer="flat", px=192, structures=[s])
    a = gen.generate(req("no_such_view")).images[0]
    b = gen.generate(req("iso", prompt="other")).images[0]
    assert a == b == renders["iso"]
