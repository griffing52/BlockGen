"""Context modules (WP6): legacy-text parity, examples, dream palette, texture sheet."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from blockgen.agentic import context as ctx
from blockgen.agentic import core, texsheet
from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic.core import ContextParts, DreamVersion, Lowered, MinecraftWorld
from blockgen.agentic.prompts import _palette_block
from blockgen.agentic.providers import encode_png

from .fakes import HUT_PROGRAM, FakeActions, RecordingTrace, ScriptedLLM, fake_spec


# ---------------------------------------------------------------------------
# A minimal EpisodeLike: contexts only touch these attributes.
# ---------------------------------------------------------------------------
@dataclass
class _Kit:
    actions: Any = field(default_factory=FakeActions)


@dataclass
class FakeEp:
    prompt: str = "a small stone tower"
    contexts: List[Any] = field(default_factory=lambda: ["palette"])
    kit: _Kit = field(default_factory=_Kit)
    seed: int = 0
    world: Any = None
    dreams: Dict[str, List[DreamVersion]] = field(default_factory=dict)
    reference_images: List[bytes] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)
    trace: RecordingTrace = field(default_factory=RecordingTrace)

    def __post_init__(self) -> None:
        self.spec = fake_spec(contexts=self.contexts)
        if self.world is None:
            self.world = MinecraftWorld.empty((32, 32, 32))

    def latest_dream(self, key: str) -> Optional[DreamVersion]:
        vs = self.dreams.get(key) or []
        return vs[-1] if vs else None

    def add_dream(self, key: str, png: bytes) -> DreamVersion:
        dv = DreamVersion(key, len(self.dreams.get(key, [])), png, "p", "fake")
        self.dreams.setdefault(key, []).append(dv)
        return dv


def two_colour_png(c1=(160, 39, 34), c2=(53, 57, 157), bg=(255, 255, 255), px=64) -> bytes:
    img = np.full((px, px, 3), bg, np.uint8)
    img[8:36, 8:56] = c1          # 28 rows
    img[36:56, 8:56] = c2         # 20 rows: c1 is the larger share
    return encode_png(img)


@pytest.fixture
def tmp_sheet_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(texsheet, "CACHE_DIR", tmp_path / "texsheets")
    texsheet.clear_cache()
    yield tmp_path / "texsheets"
    texsheet.clear_cache()


@pytest.fixture
def no_textures(tmp_path, monkeypatch, tmp_sheet_cache):
    empty = tmp_path / "no_pack"
    empty.mkdir()
    monkeypatch.setattr(texsheet, "PACK_DIR", empty)
    texsheet.clear_cache()
    ctx.block_colors.cache_clear()
    yield empty
    texsheet.clear_cache()
    ctx.block_colors.cache_clear()


def _has_catalog() -> bool:
    from blockgen.ontology import load_catalog
    return load_catalog() is not None


def _has_textures() -> bool:
    return (texsheet.PACK_DIR / "oak_planks.png").is_file()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_all_contexts_registered():
    names = core.names("context")
    for n in ("palette", "ontology", "examples", "texture_sheet", "dream_palette", "reference"):
        assert n in names
    assert core.meta("context", "dream_palette").get("needs_dream") is True
    assert core.meta("context", "texture_sheet").get("images") is True


def test_make_validates_params():
    with pytest.raises(TypeError, match="did you mean"):
        core.make("context", {"texture_sheet": {"tiles": 64}})
    fn = core.make("context", "examples?k=0")
    assert fn(FakeEp()).turns == []


# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
def test_palette_equals_legacy_block():
    parts = ctx.palette(FakeEp())
    assert parts.system_text == _palette_block()
    assert parts.user_parts == []
    assert core.make("context", "palette")(FakeEp()).system_text == _palette_block()


def test_palette_defers_to_active_ontology_like_legacy_system_prompt():
    ep = FakeEp(contexts=["palette", {"ontology": {"variant": "mined"}}])
    assert ctx.palette(ep).system_text == ""
    assert ctx.palette(ep, skip_if_ontology=False).system_text == _palette_block()
    ep_none = FakeEp(contexts=["palette", {"ontology": {"variant": "none"}}])
    assert ctx.palette(ep_none).system_text == _palette_block()
    ep_str = FakeEp(contexts=["palette", "ontology"])          # default variant = mined
    assert ctx.palette(ep_str).system_text == ""


# ---------------------------------------------------------------------------
# ontology
# ---------------------------------------------------------------------------
def test_ontology_none_is_empty():
    assert ctx.ontology(FakeEp(), variant="none").system_text == ""
    assert ctx.ontology_text("none") == ""


@pytest.mark.skipif(not _has_catalog(), reason="ontology catalog not built")
@pytest.mark.parametrize("variant", ["none", "mined", "shuffled", "stats"])
@pytest.mark.parametrize("seed", [0, 3])
def test_ontology_equals_buildagent_block(variant, seed):
    agent = BuildAgent(ScriptedLLM(), AgentConfig(ontology=variant, ontology_seed=seed))
    ep = FakeEp()
    got = core.make("context", {"ontology": {"variant": variant, "seed": seed}})(ep)
    assert got.system_text == agent._ontology_block()
    assert ep.notes["ontology"]["variant"] == variant


def test_ontology_missing_catalog_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m blockgen.ontology"):
        ctx.ontology(FakeEp(), variant="mined", path=str(tmp_path / "nope.json"))


def test_ontology_unknown_variant_raises():
    if not _has_catalog():
        pytest.skip("ontology catalog not built")
    with pytest.raises(ValueError):
        ctx.ontology(FakeEp(), variant="bogus")


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------
def test_examples_builtin_asks_action_space():
    ep = FakeEp()
    ep.inserts_example_turns = True          # an Episode that inserts prior turns
    parts = ctx.examples(ep, k=1)
    assert isinstance(parts, ContextParts) and parts.system_text == "" and not parts.user_parts
    assert [m.role for m in parts.turns] == ["user", "assistant"]
    assert parts.turns[0].text == "Build: a small hut"
    assert parts.turns[1].text == HUT_PROGRAM.strip()
    assert ep.notes["example_turns"] == parts.turns
    assert ctx.prior_turns([ContextParts(system_text="x"), parts]) == parts.turns


def test_examples_ride_in_first_turn_when_episode_ignores_turns():
    """The current Episode reads only system_text/user_parts: the demonstrations
    must reach the model through user_parts, not be dropped silently."""
    parts = ctx.examples(FakeEp(), k=1)
    assert len(parts.user_parts) == 1 and parts.user_parts[0].kind == "text"
    text = parts.user_parts[0].text
    assert "Build: a small hut" in text and HUT_PROGRAM.strip() in text
    assert [m.role for m in parts.turns] == ["user", "assistant"]   # still offered


def test_examples_reach_the_model_through_a_real_episode():
    """Regression: recipes like python-oneshot/dream list `examples` as a context
    and rely on it (classic zeroes n_examples); the first builder request must
    contain the demonstration."""
    from blockgen.agentic.episode import Episode

    from .fakes import make_fake_kit

    recipe = ["palette", {"examples": {"k": 1}}]
    kit = make_fake_kit(spec_kw={"contexts": recipe},
                        contexts=[(core.parse_spec(c)[0], core.make("context", c))
                                  for c in recipe])
    ep = Episode("a small stone tower", kit)
    ep.ask("Build: a small stone tower", stage="generate")
    first = ep.transcript[0]["request"]
    assert "Build: a small hut" in first and HUT_PROGRAM.strip() in first


def test_examples_mixed_pool_uses_space_converter(tmp_path):
    """mixed with parts [python, blocks] ignores unfenced DSL: the example must be
    converted by the space itself, not passed through raw."""
    from blockgen.agentic.actions import MixedActions

    acts = MixedActions(parts=["python", "blocks"])
    ep = FakeEp(prompt="a blue house", kit=_Kit(actions=acts))
    ans = ctx.examples(ep, k=1, pool=_pool(tmp_path)).turns[1].text
    assert ans.startswith("```python") and "fill blue_wool 0 0 0 2 2 2" not in ans
    low = acts.lower(ans)
    assert not low.issues and low.program_text.strip()


def test_examples_k0_is_empty():
    parts = ctx.examples(FakeEp(), k=0)
    assert parts.turns == [] and parts.system_text == ""


def test_examples_match_legacy_build_prompt_shape():
    """Same message shape legacy build_prompt uses for its examples."""
    from blockgen.agentic.examples import select_examples
    from blockgen.agentic.prompts import build_prompt

    class DslLike(FakeActions):
        name = "dsl"

        def examples(self, query, k):
            return [(e.caption, e.program) for e in select_examples(query, k)]

    ep = FakeEp(prompt="a stone tower", kit=_Kit(actions=DslLike()))
    turns = ctx.examples(ep, k=2).turns
    legacy = build_prompt("a stone tower", examples=select_examples("a stone tower", 2))[:-1]
    assert [(m.role, [p.fingerprint() for p in m.parts]) for m in turns] == \
           [(m.role, [p.fingerprint() for p in m.parts]) for m in legacy]


def _pool(tmp_path) -> str:
    path = tmp_path / "pool.json"
    path.write_text(json.dumps([
        {"caption": "a red hut", "program": "fill red_wool 0 0 0 2 2 2", "tags": ["red"]},
        {"caption": "a blue hut", "program": "fill blue_wool 0 0 0 2 2 2", "tags": ["blue"]},
    ]))
    return str(path)


def test_examples_file_pool_dsl(tmp_path):
    class Dsl(FakeActions):
        name = "dsl"

    ep = FakeEp(prompt="a blue house", kit=_Kit(actions=Dsl()))
    turns = ctx.examples(ep, k=1, pool=_pool(tmp_path)).turns
    assert turns[0].text == "Build: a blue hut"
    assert turns[1].text == "fill blue_wool 0 0 0 2 2 2"


def test_examples_file_pool_converts_for_python(tmp_path, monkeypatch):
    import blockgen.agentic as pkg

    fake = types.ModuleType("blockgen.agentic.actions")
    fake.dsl_to_python = lambda text: f"# py\n{text}"
    fake.dsl_to_blocks = lambda text, size: f"# blocks {size}\n{text}"
    monkeypatch.setitem(sys.modules, "blockgen.agentic.actions", fake)
    monkeypatch.setattr(pkg, "actions", fake, raising=False)

    class Py(FakeActions):
        name = "python"

    class Blocks(FakeActions):
        name = "blocks"

    t = ctx.examples(FakeEp(prompt="red", kit=_Kit(actions=Py())), pool=_pool(tmp_path)).turns
    assert t[1].text.startswith("# py")
    t = ctx.examples(FakeEp(prompt="red", kit=_Kit(actions=Blocks())), pool=_pool(tmp_path)).turns
    assert t[1].text.startswith("# blocks (32, 32, 32)")


def test_examples_bad_pool_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ctx.examples(FakeEp(), pool=str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------
def test_rgb_to_lab_reference_values():
    lab = ctx.rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)


def test_kmeans_is_deterministic_and_caps_k():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, (50, 3)), rng.normal(40, 1, (50, 3))])
    c1, l1 = ctx.kmeans(x, 2)
    c2, l2 = ctx.kmeans(x, 2)
    np.testing.assert_array_equal(c1, c2)
    assert len(set(l1[:50])) == 1 and len(set(l1[50:])) == 1 and l1[0] != l1[-1]
    c, _ = ctx.kmeans(np.array([[1.0, 2, 3]] * 10 + [[5.0, 5, 5]] * 4), 12)
    assert len(c) == 2                                  # only 2 distinct colours


def test_foreground_ignores_coloured_background():
    png = two_colour_png(bg=(135, 206, 235))           # sky-blue backdrop
    fg = ctx.foreground_lab(png)
    sky = ctx.rgb_to_lab(np.array([135, 206, 235]))
    assert len(fg) > 0         # box-filtered to PALETTE_SIDE and eroded: not 48 * 48
    assert np.linalg.norm(fg - sky, axis=-1).min() > 12


def _synthetic_candidates():
    def bc(name, rgb):
        return ctx.BlockColor(name, (1, 0), rgb, tuple(ctx.rgb_to_lab(np.array(rgb))))
    return [bc("red_a", (161, 39, 35)), bc("red_b", (150, 40, 40)), bc("blue_a", (53, 57, 157)),
            bc("green_a", (40, 160, 40)), bc("white_a", (250, 250, 250)),
            bc("black_a", (10, 10, 10))]


def test_match_palette_synthetic_candidates_no_textures_needed():
    entries = ctx.match_palette(two_colour_png(), 12, candidates=_synthetic_candidates())
    names = [e.name for e in entries]
    assert names[:2] == ["red_a", "blue_a"]             # by share, red is larger
    assert entries[0].share > entries[1].share > 0
    assert abs(entries[0].share + entries[1].share - 1.0) < 1e-6
    # Fill only adds near neighbours, never green/white/black.
    assert set(names) <= {"red_a", "red_b", "blue_a"}
    assert ctx.match_palette(two_colour_png(), 1, candidates=_synthetic_candidates())[0].name \
        == "red_a"


def test_match_palette_blank_image_is_empty():
    blank = encode_png(np.full((32, 32, 3), 255, np.uint8))
    assert ctx.match_palette(blank, 12, candidates=_synthetic_candidates()) == []


# ---------------------------------------------------------------------------
# dream_palette
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_textures(), reason="vanilla textures not fetched")
def test_dream_palette_two_colours_returns_blocks_of_those_colours():
    ep = FakeEp()
    ep.add_dream("iso", two_colour_png())
    parts = core.make("context", {"dream_palette": {"k": 6}})(ep)
    names = ep.notes["dream_palette"]
    assert 2 <= len(names) <= 6
    by_name = {b.name: b for b in ctx.block_colors()}
    red, blue = ctx.rgb_to_lab(np.array((160, 39, 34))), ctx.rgb_to_lab(np.array((53, 57, 157)))
    for n in names:
        lab = np.asarray(by_name[n].lab)
        assert min(np.linalg.norm(lab - red), np.linalg.norm(lab - blue)) < 25, n
    first_two = [by_name[n] for n in names[:2]]
    assert np.linalg.norm(np.asarray(first_two[0].lab) - red) < 10
    assert np.linalg.norm(np.asarray(first_two[1].lab) - blue) < 10
    assert names[0] in parts.system_text and "Dream palette" in parts.system_text
    assert ep.notes["dream_palette_detail"][0]["name"] == names[0]


def test_dream_palette_uses_latest_version(monkeypatch):
    monkeypatch.setattr(ctx, "block_colors", lambda *a, **k: tuple(_synthetic_candidates()))
    ep = FakeEp()
    ep.add_dream("iso", two_colour_png(c1=(40, 160, 40), c2=(40, 160, 40)))
    ep.add_dream("iso", two_colour_png())
    ctx.dream_palette(ep, k=2)
    assert ep.notes["dream_palette"] == ["red_a", "blue_a"]


def test_dream_palette_falls_back_to_another_key(monkeypatch):
    monkeypatch.setattr(ctx, "block_colors", lambda *a, **k: tuple(_synthetic_candidates()))
    ep = FakeEp()
    ep.add_dream("front", two_colour_png())
    ctx.dream_palette(ep, k=2, key="iso")
    assert ep.notes["dream_palette"] == ["red_a", "blue_a"]
    assert "dream_palette_key" in ep.trace.kinds()


def test_dream_palette_without_dream_is_empty():
    ep = FakeEp()
    parts = ctx.dream_palette(ep)
    assert parts.system_text == "" and ep.notes["dream_palette"] == []
    assert "dream_palette_skipped" in ep.trace.kinds()


def test_dream_palette_without_textures_is_empty_not_fatal(no_textures):
    ep = FakeEp()
    ep.add_dream("iso", two_colour_png())
    assert ctx.block_colors() == ()
    assert ctx.dream_palette(ep).system_text == ""
    assert "dream_palette_skipped" in ep.trace.kinds()


# ---------------------------------------------------------------------------
# texture_sheet
# ---------------------------------------------------------------------------
def test_texture_sheet_palette_parts(tmp_sheet_cache):
    from blockgen.agentic.blockstate import PALETTE

    ep = FakeEp()
    parts = core.make("context", {"texture_sheet": {"blocks": "palette", "tile": 32}})(ep)
    assert parts.system_text == ""
    assert parts.user_parts[0].kind == "text" and "Texture sheet" in parts.user_parts[0].text
    images = [p for p in parts.user_parts if p.kind == "image"]
    assert images and all(p.image_png.startswith(b"\x89PNG") for p in images)
    assert ep.notes["texture_sheet"]["n"] == len(PALETTE)
    assert ep.notes["texture_sheet"]["shuffled"] is False


def test_texture_sheet_shuffled_is_token_matched(tmp_sheet_cache):
    a = ctx.texture_sheet(FakeEp(), blocks=["stone", "oak_planks", "bricks"], tile=32)
    b = ctx.texture_sheet(FakeEp(), blocks=["stone", "oak_planks", "bricks"], tile=32,
                          shuffled=True, seed=1)
    assert a.user_parts[0].text == b.user_parts[0].text      # identical caption
    assert a.user_parts[1].image_png != b.user_parts[1].image_png


def test_texture_sheet_used_blocks(tmp_sheet_cache):
    ep = FakeEp()
    ep.world.run(Lowered(HUT_PROGRAM, HUT_PROGRAM))
    assert ctx._sheet_names(ep, "used") == ["cobblestone", "oak_planks"]
    parts = ctx.texture_sheet(ep, blocks="used")
    assert ep.notes["texture_sheet"]["n"] == 2 and len(parts.user_parts) == 2


def test_texture_sheet_used_on_empty_world_is_rejected(tmp_sheet_cache):
    # Contexts run once, before the first build turn: 'used' would always be empty.
    ep = FakeEp()
    with pytest.raises(ValueError, match="used"):
        ctx.texture_sheet(ep, blocks="used")


def test_texture_sheet_dream_palette_computed_on_demand(tmp_sheet_cache, monkeypatch):
    cands = [c for c in _synthetic_candidates()]
    cands = [ctx.BlockColor(n, (1, 0), c.rgb, c.lab) for n, c in
             zip(["red_wool", "red_concrete", "blue_wool", "green_wool", "white_wool",
                  "black_wool"], cands)]
    monkeypatch.setattr(ctx, "block_colors", lambda *a, **k: tuple(cands))
    ep = FakeEp()
    ep.add_dream("iso", two_colour_png())
    parts = ctx.texture_sheet(ep, blocks="dream_palette")
    assert ep.notes["dream_palette"][:2] == ["red_wool", "blue_wool"]
    assert ep.notes["texture_sheet"]["n"] == len(ep.notes["dream_palette"])
    assert parts.user_parts


def test_texture_sheet_bad_blocks_raises(tmp_sheet_cache):
    with pytest.raises(ValueError, match="texture_sheet blocks"):
        ctx.texture_sheet(FakeEp(), blocks="everything")


def test_texture_sheet_without_textures_warns_but_works(no_textures):
    ep = FakeEp()
    with pytest.warns(UserWarning, match="fallback"):
        parts = ctx.texture_sheet(ep, blocks=["stone", "oak_planks"], tile=16)
    assert len(parts.user_parts) == 2
    assert "texture_sheet_degraded" in ep.trace.kinds()


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------
def test_reference_parts():
    assert ctx.reference(FakeEp()).user_parts == []
    img = two_colour_png()
    parts = core.make("context", "reference")(FakeEp(reference_images=[img, img]))
    assert parts.user_parts[0].text == ctx.REFERENCE_NOTE
    assert [p.image_png for p in parts.user_parts[1:]] == [img, img]
    assert parts.system_text == ""


def test_reference_note_is_legacy_text():
    from blockgen.agentic.prompts import build_prompt

    msg = build_prompt("x", images=[b"img"])[-1]
    assert msg.parts[0].text.startswith(ctx.REFERENCE_NOTE)


def test_block_colors_follow_current_pack_dir(tmp_path, monkeypatch):
    """match_palette must not keep serving the colours of the first pack it saw."""
    ctx.block_colors.cache_clear()
    seen = []
    real = ctx.block_colors.__wrapped__

    def spy(pack_dir=None):
        seen.append(pack_dir)
        return real(pack_dir)

    monkeypatch.setattr(ctx, "block_colors", spy)
    empty = tmp_path / "no_pack"
    empty.mkdir()
    monkeypatch.setattr(texsheet, "PACK_DIR", empty)
    assert ctx.match_palette(two_colour_png(), 4) == []
    assert seen == [str(empty)]
