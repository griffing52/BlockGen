"""The block ontology: measurement, control, and the prompt it produces.

Three of these tests exist because of a specific way this can go wrong.

`test_mined_statistics_are_exact` builds a structure whose every statistic can be
worked out by hand and asserts the numbers, because the whole claim of the mined
ontology is "these are measurements" -- a plausible-looking wrong number is the
failure mode that would survive every other check here.

`test_shuffled_is_token_matched` pins the control's only real requirement. If the
shuffled table were even slightly shorter than the real one, an arm difference
could be a context-length difference, and the experiment would prove nothing.

`test_stats_are_json_serializable` pins a bug already paid for: numpy scalars
leaking into the stats made `Catalog.save` raise on `np.float64`, after a full
corpus mine had already run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.agentic.blockstate import PALETTE, resolve_block
from blockgen.ontology import authored, mine, minecraft
from blockgen.ontology.schema import Catalog
from blockgen.utils.data import Structure

STONE, OAK_LOG, OAK_PLANKS = (1, 0), (17, 0), (5, 0)


def _floor_and_column() -> Structure:
    """A 4x4x4 volume: a stone floor at y=0 and a 3-tall oak column on one corner.

    Every statistic below is derivable from this by hand, which is the point.
    """
    ids = np.zeros((4, 4, 4), dtype=np.int32)
    data = np.zeros((4, 4, 4), dtype=np.int32)
    ids[:, 0, :] = STONE[0]
    ids[0, 1:4, 0] = OAK_LOG[0]
    return Structure(block_ids=ids, block_data=data)


# --- the miner -------------------------------------------------------------
def test_mined_statistics_are_exact():
    stats = mine.mine_corpus([_floor_and_column()], min_support=1, min_pair_count=1)
    stone = stats.get(mine.symbol_for(*STONE))
    log = stats.get(mine.symbol_for(*OAK_LOG))

    assert stone.count == 16 and log.count == 3
    # Height is normalized inside the build's occupied bbox (y spans 0..3).
    assert stone.height_mean == pytest.approx(0.0)
    assert log.height_mean == pytest.approx((1 + 2 + 3) / 3 / 3)
    # The floor is one run of 4 per row in x and in z; the column is one run of 3.
    assert stone.horizontal_run == pytest.approx(4.0)
    assert stone.vertical_run == pytest.approx(1.0)
    assert log.vertical_run == pytest.approx(3.0)
    assert log.horizontal_run == pytest.approx(1.0)
    assert log.anisotropy == pytest.approx(3.0)
    # Everything rests on something (y=0 counts as ground).
    assert stone.support_frac == pytest.approx(1.0)
    assert log.support_frac == pytest.approx(1.0)
    # Open faces: 4, 4 and 5 going up the column.
    assert log.exposure == pytest.approx((4 + 4 + 5) / 3)
    # The column touches the floor exactly once, so stone is its only neighbour.
    assert [n[0] for n in log.neighbors] == [stone.symbol]


def test_empty_and_degenerate_inputs_do_not_raise():
    empty = Structure(block_ids=np.zeros((2, 2, 2), dtype=np.int32),
                      block_data=np.zeros((2, 2, 2), dtype=np.int32))
    stats = mine.mine_corpus([empty])
    assert stats.n_builds == 0 and stats.blocks == {}
    assert mine.mine_corpus([]).n_placements == 0


def test_stats_are_json_serializable():
    stats = mine.mine_corpus([_floor_and_column()], min_support=1, min_pair_count=1)
    json.dumps(stats.to_json())          # numpy scalars here used to raise


def test_symbols_collapse_orientation_but_not_species():
    """Stairs facing four ways are one entry; oak and spruce are not."""
    facings = {mine.symbol_for(53, d) for d in range(8)}
    assert len(facings) == 1
    assert mine.symbol_for(*resolve_block("oak_planks")) != \
           mine.symbol_for(*resolve_block("spruce_planks"))


# --- words from numbers ----------------------------------------------------
@pytest.mark.parametrize("height,word", [(0.0, "ground"), (0.19, "ground"),
                                         (0.3, "low"), (0.45, "mid"),
                                         (0.55, "upper"), (0.9, "roof")])
def test_layer_word_cutoffs(height, word):
    assert minecraft.layer_word(height) == word


def test_form_word_prefers_exposure_then_verticality():
    def stats(**kw):
        return mine.BlockStats(symbol="x", family="x", **kw)

    assert minecraft.form_word(stats(exposure=3.0, vertical_run=9, horizontal_run=1)) == "trim"
    assert minecraft.form_word(stats(exposure=1.0, vertical_run=3, horizontal_run=1)) == "post"
    assert minecraft.form_word(stats(exposure=1.0, vertical_run=1, horizontal_run=4)) == "plate"
    assert minecraft.form_word(stats(exposure=1.0, vertical_run=1.6, horizontal_run=1.7)) == "mass"


# --- authored / derived rules ----------------------------------------------
def test_physical_rules():
    assert authored.flammable("oak_planks") and not authored.flammable("stone_bricks")
    assert authored.gravity("sand") and not authored.gravity("white_concrete")
    assert authored.light_level("glowstone") == 15
    assert not authored.full_block("oak_fence") and authored.full_block("bricks")


def test_state_properties_match_the_executor():
    """The advertised grammar has to be what blockstate actually honours; a
    property we print but drop is a silent no-op in a build program."""
    assert authored.state_properties("oak_stairs") == ["facing", "half"]
    assert authored.state_properties("oak_log") == ["axis"]
    assert authored.state_properties("stone_slab") == ["type"]
    assert authored.state_properties("bricks") == []


# --- the catalog -----------------------------------------------------------
@pytest.fixture(scope="module")
def small_catalog() -> Catalog:
    stats = mine.mine_corpus([_floor_and_column()] * 40, corpus="synthetic",
                             min_support=1, min_pair_count=1, top_neighbors=25)
    return minecraft.build_catalog(stats, min_support=1)


def test_catalog_covers_the_whole_palette(small_catalog):
    assert set(small_catalog.ids()) == set(PALETTE)
    assert len(small_catalog) == len(PALETTE)


def test_catalog_round_trips_through_json(tmp_path, small_catalog):
    path = small_catalog.save(tmp_path / "cat.json")
    back = Catalog.load(path)
    assert back.ids() == small_catalog.ids()
    assert back.render_table() == small_catalog.render_table()
    assert [s.name for s in back.schema] == [s.name for s in small_catalog.schema]


def test_neighbour_lists_stay_inside_the_palette(small_catalog):
    """A neighbour the DSL cannot resolve would teach the model a name that fails
    to parse -- knowledge converted into per-line errors."""
    for part in small_catalog:
        for name in part.get("build.pairs_with") or []:
            assert name in PALETTE
            assert name != part.id


def test_family_collisions_are_recorded_not_hidden(small_catalog):
    """`stripped_oak_log` remaps onto `oak_log`; the catalog must say so."""
    assert "same legacy block as" in small_catalog.get("stripped_oak_log").notes


def test_low_support_blocks_get_numbers_but_not_tags():
    stats = mine.mine_corpus([_floor_and_column()], min_support=1000,
                             min_pair_count=1)
    catalog = minecraft.build_catalog(stats, min_support=1000)
    stone = catalog.get("stone")
    assert stone.get("stats.count") == 16          # the measurement is reported
    assert stone.get("build.layer") is None        # the word is withheld
    assert "tags withheld" in stone.notes


# --- the control -----------------------------------------------------------
def test_shuffled_preserves_every_marginal(small_catalog):
    shuffled = small_catalog.shuffled(0)
    assert set(shuffled.ids()) == set(small_catalog.ids())
    for spec in small_catalog.schema:
        if spec.kind == "refs":
            continue                    # self-references are dropped, see below
        def values(cat):
            return sorted(json.dumps(p.attrs.get(spec.name), sort_keys=True)
                          for p in cat)
        assert values(shuffled) == values(small_catalog), spec.name


def test_shuffled_is_token_matched(small_catalog):
    a = minecraft.prompt_block(small_catalog)
    b = minecraft.prompt_block(small_catalog.shuffled(0))
    assert abs(len(a) - len(b)) <= 0.02 * len(a)


def test_shuffled_actually_moves_attributes(small_catalog):
    shuffled = small_catalog.shuffled(0)
    moved = sum(1 for p in small_catalog
                if shuffled.get(p.id).attrs.get("appearance.color")
                != p.attrs.get("appearance.color"))
    assert moved > 0.5 * len(small_catalog)


def test_shuffled_never_lists_a_block_as_its_own_neighbour(small_catalog):
    shuffled = small_catalog.shuffled(3)
    for part in shuffled:
        assert part.id not in (part.get("build.pairs_with") or [])


def test_subset_by_source_drops_the_other_fields(small_catalog):
    mined = small_catalog.subset_by_source(("mined",))
    assert all(s.source == "mined" for s in mined.schema)
    assert all("appearance.color" not in p.attrs for p in mined)
    assert mined.ids() == small_catalog.ids()


# --- the prompt seam -------------------------------------------------------
def test_prompt_table_is_readable_and_closed(small_catalog):
    text = minecraft.prompt_block(small_catalog)
    assert "Block reference" in text and "oak_planks" in text
    assert "Orientable blocks:" in text
    for line in text.splitlines():
        assert len(line) < 260          # a runaway column would wreck the prompt


def test_render_table_respects_field_selection(small_catalog):
    text = small_catalog.render_table(["build.layer"], legend=False)
    assert text.splitlines()[0].split() == ["block", "layer"]


def test_agent_system_prompt_swaps_the_palette_for_the_table(tmp_path, small_catalog):
    from blockgen.agentic.agent import AgentConfig, BuildAgent
    from blockgen.agentic.providers import get_provider

    path = small_catalog.save(tmp_path / "cat.json")
    provider = get_provider("mock", cache=False)

    plain = BuildAgent(provider, AgentConfig(verbose=False))._system().text
    with_ontology = BuildAgent(provider, AgentConfig(
        ontology="mined", ontology_path=str(path), verbose=False))._system().text

    assert "Block reference" not in plain and "Palette (known-good" in plain
    assert "Block reference" in with_ontology
    # The table replaces the bare list rather than joining it, so the vocabulary
    # is never printed twice.
    assert "Palette (known-good" not in with_ontology
    assert len(with_ontology) > len(plain)


def test_missing_catalog_fails_loudly(tmp_path):
    """An ontology arm that silently ran without an ontology would land in the
    results table as a null result."""
    from blockgen.agentic.agent import AgentConfig, BuildAgent
    from blockgen.agentic.providers import get_provider

    agent = BuildAgent(get_provider("mock", cache=False),
                       AgentConfig(ontology="mined", verbose=False,
                                   ontology_path=str(tmp_path / "nope.json")))
    with pytest.raises(FileNotFoundError):
        agent._system()


def test_unknown_variant_is_an_error(tmp_path, small_catalog):
    from blockgen.ontology import variant_catalog

    small_catalog.save(tmp_path / "cat.json")
    with pytest.raises(ValueError):
        variant_catalog("bogus", str(tmp_path / "cat.json"))


# --- appearance ------------------------------------------------------------
def test_appearance_measures_the_texture_not_a_guess(tmp_path):
    from PIL import Image

    from blockgen.ontology import appearance

    pack = tmp_path / "pack"
    pack.mkdir()
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 0], pixels[..., 3] = 200, 255          # flat opaque red
    Image.fromarray(pixels, "RGBA").save(pack / "stone.png")

    look = appearance.measure(1, 0, pack_dir=pack)
    assert look.rgb == (200, 0, 0) and look.texture == "flat"
    assert look.transparency == "opaque"
    assert appearance.measure(35, 0, pack_dir=pack) is None   # no texture on disk


def test_colour_names_are_the_words_a_builder_uses():
    from blockgen.ontology.appearance import color_name

    assert color_name((59, 38, 17)) == "dark brown"       # spruce log
    assert color_name((162, 131, 79)) == "tan"            # oak planks
    assert color_name((128, 127, 128)) == "grey"          # cobblestone
    assert color_name((8, 10, 15)) == "near-black"


def test_every_palette_block_has_a_texture():
    """A gap here means a swatch the lab cannot draw and a colour the catalog
    cannot measure -- worth failing on, since the pack is in the repo."""
    from blockgen.ontology.appearance import DEFAULT_PACK_DIR, measure

    if not DEFAULT_PACK_DIR.is_dir():
        pytest.skip("no texture pack fetched")
    missing = [n for n in PALETTE if measure(*resolve_block(n)) is None]
    assert missing == []
