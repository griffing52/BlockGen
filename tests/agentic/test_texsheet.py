"""Texture contact sheet (WP6): deterministic, cached, one labelled tile per block, paged."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from blockgen.agentic import texsheet
from blockgen.agentic.blockstate import PALETTE

NAMES = ["stone", "oak_planks", "bricks", "glass", "grass_block", "red_wool"]


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(texsheet, "CACHE_DIR", tmp_path / "texsheets")
    texsheet.clear_cache()
    yield tmp_path / "texsheets"
    texsheet.clear_cache()


def _img(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def _tile_px(sheet, i):
    page, x, y = sheet.cells[i]
    return _img(sheet.pages[page])[y:y + sheet.tile, x:x + sheet.tile]


def test_deterministic_bytes_and_key():
    a = texsheet.build_sheet(NAMES, tile=32)
    texsheet.clear_cache()
    b = texsheet.build_sheet(NAMES, tile=32)
    assert a.pages == b.pages and a.key == b.key


def test_cached_on_disk_and_reused(tmp_cache, monkeypatch):
    a = texsheet.build_sheet(NAMES, tile=32)
    files = sorted(tmp_cache.glob("*.png"))
    assert len(files) == len(a.pages) and not a.cached
    assert {f.read_bytes() for f in files} == set(a.pages)
    texsheet.clear_cache()

    def boom(*args, **kw):
        raise AssertionError("cache miss: page was recomposed")
    monkeypatch.setattr(texsheet, "_compose", boom)
    b = texsheet.build_sheet(NAMES, tile=32)
    assert b.cached and b.pages == a.pages and b.cells == a.cells
    # In-process memo: no disk read either.
    assert texsheet.build_sheet(NAMES, tile=32) is b


def test_cache_key_depends_on_params():
    keys = {texsheet.build_sheet(NAMES, tile=32).key,
            texsheet.build_sheet(NAMES, tile=48).key,
            texsheet.build_sheet(NAMES, tile=32, cols=3).key,
            texsheet.build_sheet(NAMES, tile=32, face="top").key,
            texsheet.build_sheet(NAMES, tile=32, shuffled=True).key,
            texsheet.build_sheet(NAMES[:-1], tile=32).key}
    assert len(keys) == 6


def test_one_tile_per_block_with_exact_texture_pixels():
    sheet = texsheet.build_sheet(NAMES, tile=32)
    assert len(sheet.cells) == len(NAMES) == len(sheet.labels)
    assert sheet.names == NAMES and sheet.labels == NAMES and not sheet.shuffled
    tiles = texsheet.resolve_tiles(NAMES)
    for i, t in enumerate(tiles):
        np.testing.assert_array_equal(_tile_px(sheet, i), texsheet._tile_image(t, 32))
    # Cells never overlap.
    boxes = {(p, x, y) for p, x, y in sheet.cells}
    assert len(boxes) == len(NAMES)


def test_labels_under_tiles_are_drawn():
    sheet = texsheet.build_sheet(["stone"], tile=32)
    img = _img(sheet.pages[0])
    _, x, y = sheet.cells[0]
    label = img[y + 33:y + 33 + 14, :]
    assert (label < 100).any()                  # dark label text below the tile
    blank = texsheet.build_sheet(["stone"], tile=32, labels=[""])
    lab2 = _img(blank.pages[0])[y + 33:y + 33 + 14, :]
    assert not (lab2 < 100).any()


def test_paged_at_max_px():
    names = list(PALETTE)
    sheet = texsheet.build_sheet(names, tile=192, cols=10)
    assert len(sheet.pages) > 1
    for png in sheet.pages:
        w, h = Image.open(io.BytesIO(png)).size
        assert w <= texsheet.MAX_PX and h <= texsheet.MAX_PX
    assert sorted({c[0] for c in sheet.cells}) == list(range(len(sheet.pages)))
    assert len(sheet.cells) == len(names)
    # Every page but the last is full.
    lay = texsheet.layout(len(names), tile=192, cols=10, labels=names)
    per_page = lay.cols * lay.rows_per_page
    assert sum(1 for c in sheet.cells if c[0] == 0) == per_page


def test_small_max_px_pages_more():
    one = texsheet.build_sheet(NAMES, tile=32)
    many = texsheet.build_sheet(NAMES, tile=32, max_px=150)
    assert len(one.pages) == 1 and len(many.pages) == 3
    for png in many.pages:
        assert max(Image.open(io.BytesIO(png)).size) <= 150


def test_shuffled_same_tiles_different_labels():
    plain = texsheet.build_sheet(NAMES, tile=32)
    shuf = texsheet.build_sheet(NAMES, tile=32, shuffled=True, seed=0)
    assert shuf.shuffled and shuf.names == plain.names
    assert sorted(shuf.labels) == sorted(plain.labels)
    assert all(a != b for a, b in zip(shuf.labels, shuf.names))      # derangement
    assert shuf.cells == plain.cells
    for i in range(len(NAMES)):
        np.testing.assert_array_equal(_tile_px(shuf, i), _tile_px(plain, i))
    assert shuf.pages != plain.pages
    again = texsheet.build_sheet(NAMES, tile=32, shuffled=True, seed=0)
    assert again.labels == shuf.labels
    other = texsheet.build_sheet(NAMES, tile=32, shuffled=True, seed=5)
    assert other.labels != shuf.labels


@pytest.mark.parametrize("n", range(2, 15))
def test_shuffled_labels_is_a_derangement(n):
    names = [f"b{i}" for i in range(n)]
    for seed in range(5):
        lab = texsheet.shuffled_labels(names, seed)
        assert sorted(lab) == sorted(names)
        assert all(a != b for a, b in zip(lab, names))


def test_duplicates_dropped_and_unknown_kept_in_place():
    sheet = texsheet.build_sheet(["stone", "stone", "not_a_block_xyz", "bricks"], tile=16)
    assert sheet.names == ["stone", "not_a_block_xyz", "bricks"]
    assert sheet.unknown == ["not_a_block_xyz"]
    hatched = _tile_px(sheet, 1)
    assert set(map(tuple, hatched.reshape(-1, 3))) == {(235, 235, 235), (205, 205, 205)}


def test_missing_textures_fall_back_gracefully(tmp_path, monkeypatch):
    empty = tmp_path / "pack"
    empty.mkdir()
    monkeypatch.setattr(texsheet, "PACK_DIR", empty)
    texsheet.clear_cache()
    sheet = texsheet.build_sheet(NAMES, tile=16)
    assert sheet.missing == NAMES and sheet.unknown == []
    assert len(sheet.pages) == 1
    # Fallback tiles are solid colours, still distinct per block.
    colours = {tuple(_tile_px(sheet, i)[8, 8]) for i in range(len(NAMES))}
    assert len(colours) >= 4


def test_faces_differ_for_grass():
    side = texsheet.resolve_tiles(["grass_block"], face="side")[0]
    top = texsheet.resolve_tiles(["grass_block"], face="top")[0]
    if side.fallback:
        pytest.skip("vanilla textures not fetched")
    assert not np.array_equal(side.rgba, top.rgba)
    with pytest.raises(ValueError):
        texsheet.resolve_tiles(["stone"], face="north")


def test_validation_errors():
    with pytest.raises(ValueError, match="empty"):
        texsheet.build_sheet([])
    with pytest.raises(ValueError, match="labels"):
        texsheet.build_sheet(["stone"], labels=["a", "b"])
    with pytest.raises(ValueError, match="either"):
        texsheet.build_sheet(["stone"], labels=["a"], shuffled=True)
    with pytest.raises(ValueError, match="does not fit"):
        texsheet.build_sheet(["stone"], tile=2000)


def test_wrap_label_round_trips_and_fits():
    font = texsheet.load_font(11)
    for name in PALETTE:
        lines = texsheet.wrap_label(name, 82, font)
        assert "".join(lines) == name
        assert all(texsheet.text_width(line, font) <= 82 for line in lines)
    lines = texsheet.wrap_label("a_" * 40 + "b", 30, font, max_lines=2)
    assert len(lines) == 2 and lines[-1].endswith("~")


def test_tall_cell_rejected_instead_of_overflowing_page():
    """A cell taller than the page used to clamp to 1 row and exceed max_px."""
    with pytest.raises(ValueError, match="does not fit"):
        texsheet.layout(1, tile=200, cols=1, labels=["stone"], max_px=230)


def test_memory_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(texsheet, "_MEM_MAX", 3)
    names = ["stone", "bricks", "oak_planks", "red_wool", "glass"]
    for i in range(len(names)):
        texsheet.build_sheet(names[:i + 1], tile=16)
    assert len(texsheet._MEM) == 3
    # The most recent sheets survive (LRU), so a repeat is an in-memory hit.
    last = texsheet.build_sheet(names, tile=16)
    assert texsheet.build_sheet(names, tile=16) is last
