"""The `grid` context: axis rules and which build side each camera shows."""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.agentic.context import VIEW_SIDES, grid_text
from blockgen.agentic.core import RenderError, get_view, make
from blockgen.agentic.dsl import run_program
from blockgen.agentic.recipe import load_recipe


def test_grid_text_names_the_fixed_axis_and_iso_faces():
    t = grid_text(("iso",))
    assert "windmill sails" in t and "ONE" in t
    assert "LEFT of the image is the z=max side" in t and "RIGHT is the x=max side" in t
    assert "Put the front" in t


def test_grid_text_skips_unknown_views_and_is_plain_rules_without_views():
    assert "Cameras:" not in grid_text(())
    assert "nope" not in grid_text(("nope",))


def test_dream_grid_recipe_adds_only_the_grid_context():
    a, b = load_recipe("dream"), load_recipe("dream-grid")
    assert len(b.contexts) == len(a.contexts) + 1 and b.hash() != a.hash()
    assert load_recipe("dream").hash() == a.hash()       # dream itself unchanged


@pytest.mark.slow
def test_iso_left_face_is_z_max_in_a_real_render():
    """Blue wall on z=max, red wall on x=max: from 'iso' blue must be on the left."""
    try:
        r = make("renderer", "textured?px=128")
    except RenderError as exc:
        pytest.skip(f"textured renderer unavailable: {exc}")
    canvas, _, _ = run_program("fill stone 0 0 0 15 0 15\n"
                               "fill red_wool 15 1 0 15 3 15\n"
                               "fill blue_wool 0 1 15 15 3 15\n", size=(16, 16, 16))
    try:
        (img,) = r.render(canvas.to_structure(), [get_view("iso")])
    except RenderError as exc:
        pytest.skip(f"textured renderer unavailable: {exc}")
    img = img.astype(int)
    blue = (img[..., 2] > img[..., 0] + 40)
    red = (img[..., 0] > img[..., 2] + 40)
    half = img.shape[1] // 2
    assert VIEW_SIDES["iso"] == ("z=max", "x=max")
    assert blue[:, :half].sum() > blue[:, half:].sum()
    assert red[:, half:].sum() > red[:, :half].sum()
