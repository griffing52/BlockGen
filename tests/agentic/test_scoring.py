"""WP5 scorers: offline (flat renderer, numpy) plus `slow` real-model checks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from blockgen.agentic import core, scoring
from blockgen.agentic.calibrate import attempt_for, synthetic_builds
from blockgen.agentic.core import DreamSpec, DreamVersion, ScoreTerm
from blockgen.agentic.render import FlatRenderer, png_bytes, render_pngs
from blockgen.utils.data import Structure

from .fakes import HUT_PROGRAM, make_attempt

R = FlatRenderer(px=96)
BUILDS = synthetic_builds(6, seed=3)


class Ep:
    """Minimal EpisodeLike for scorers: dreams, a renderer, a scripted judge."""

    def __init__(self, dreams: Sequence[DreamSpec] = (), pngs: Optional[Dict[str, bytes]] = None,
                 prompt: str = "a small house", reply: Any = None, renderer: Any = R) -> None:
        self.kit = SimpleNamespace(dreams=list(dreams), renderer=renderer)
        self.dreams = {k: [DreamVersion(k, 0, p, "prompt", "test")] for k, p in (pngs or {}).items()}
        self.prompt = prompt
        self.reply = reply
        self.asks: List[Any] = []
        self.rendered: List[List[str]] = []

    def latest_dream(self, key):
        v = self.dreams.get(key)
        return v[-1] if v else None

    def render(self, target, views="score"):
        views = [views] if isinstance(views, str) else list(views)
        self.rendered.append(views)
        return render_pngs(self.kit.renderer, target.world.to_structure(), views)

    def ask(self, turn, **kw):
        self.asks.append((turn, kw))
        return self.reply


def dream_of(s: Structure, view: str = "iso") -> bytes:
    return render_pngs(R, s, [view])[view]


def ep_for(dream_png: bytes, view="iso", view_search=(), **kw) -> Ep:
    return Ep([DreamSpec("iso", "{desc}", view=view, view_search=tuple(view_search))],
              {"iso": dream_png}, **kw)


# --- registry -------------------------------------------------------------------
EXPECTED_META = {
    "embed": (True, True), "silhouette": (True, True), "color_lab": (True, True),
    "pixel": (True, True), "clip_text": (False, True), "material": (False, False),
    "validity": (False, False), "vlm": (True, True),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_META))
def test_scorers_registered_with_meta(name):
    m = core.meta("scorer", name)
    assert (m["needs_dream"], m["needs_renders"]) == EXPECTED_META[name]


def test_make_validates_params():
    f = core.make("scorer", "silhouette?thresh=240")
    assert f.keywords == {"thresh": 240}
    with pytest.raises(TypeError, match="thresh"):
        core.make("scorer", "silhouette?thrseh=240")


def test_accept_policies_registered():
    assert {"always", "nonempty_legacy", "nonempty", "improve", "improve_valid",
            "anneal"} <= set(core.names("accept"))


# --- foreground segmentation --------------------------------------------------------
def test_foreground_mask_render():
    img = np.full((64, 64, 3), 255, np.uint8)
    img[20:40, 10:50] = (120, 80, 40)
    m = scoring.foreground_mask(img)
    assert m[30, 30] and not m[5, 5]
    assert m.sum() == 20 * 40


def test_foreground_mask_noisy_dream_keeps_enclosed_white():
    """Off-white noisy backdrop is background; a white window inside the house is
    NOT (it is not connected to the border)."""
    rng = np.random.default_rng(0)
    img = (235 + rng.integers(-6, 7, (128, 128, 3))).astype(np.uint8)
    img[30:100, 20:110] = (150, 60, 50)
    img[50:60, 40:55] = 255                                  # window
    img[2, 2] = (0, 0, 0)                                    # speck on the border region
    m = scoring.foreground_mask(img)
    assert m[55, 45]                                         # window is foreground
    assert not m[10, 10] and not m[120, 120]
    assert abs(int(m.sum()) - 70 * 90) < 50


def test_normalized_mask_is_scale_and_translation_invariant():
    a = np.zeros((100, 100), bool)
    a[10:30, 10:50] = True
    b = np.zeros((300, 300), bool)
    b[150:210, 60:180] = True
    na, nb = scoring.normalized_mask(a), scoring.normalized_mask(b)
    assert (na & nb).sum() / (na | nb).sum() > 0.95


def test_fg_crop_square_and_whole_image_fallback():
    img = np.full((50, 80, 3), 255, np.uint8)
    img[10:20, 30:60] = 0
    out = scoring.fg_crop(img, size=32)
    assert out.shape == (32, 32, 3)
    blank = scoring.fg_crop(np.full((10, 10, 3), 255, np.uint8), size=16)
    assert blank.shape == (16, 16, 3)


def test_lab_known_values():
    lab = scoring.rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], np.uint8))
    assert lab[0, 0] == pytest.approx(100, abs=0.05) and abs(lab[0, 1]) < 0.05
    assert lab[1, 0] == pytest.approx(0, abs=0.05)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


# --- dream-vs-render scorers rank identical above different --------------------------
@pytest.mark.parametrize("name", ["pixel", "silhouette", "color_lab"])
def test_identical_beats_different(name):
    fn = core.make("scorer", name)
    own, other = BUILDS[0], BUILDS[1]
    ep = ep_for(dream_of(own))
    s_own = fn(ep, attempt_for(own)).value
    s_other = fn(ep, attempt_for(other)).value
    assert s_own == pytest.approx(1.0, abs=1e-6)
    assert s_own > s_other
    assert 0.0 <= s_other <= 1.0


def test_view_search_takes_the_max_and_reports_best_view():
    s = BUILDS[2]
    ep = ep_for(dream_of(s, "iso_left"), view="iso", view_search=("iso_left",))
    t = scoring.silhouette(ep, attempt_for(s))
    assert t.value == pytest.approx(1.0)
    assert "best view iso_left" in t.note
    assert set(t.per_view) == {"iso@iso", "iso@iso_left"}
    assert t.per_view["iso@iso_left"] == pytest.approx(1.0)
    assert t.per_view["iso@iso"] < 1.0
    ep2 = ep_for(dream_of(s, "iso_left"), view="iso")              # no search: lower
    assert scoring.silhouette(ep2, attempt_for(s)).value < t.value


def test_reduce_modes():
    s = BUILDS[2]
    ep = ep_for(dream_of(s, "iso_left"), view="iso", view_search=("iso_left",))
    att = attempt_for(s)
    paired = scoring.pixel(ep, att).value
    mean = scoring.pixel(ep, att, reduce="mean").value
    assert mean < paired
    att.renders.update(render_pngs(R, s, ["iso_left", "top"]))
    assert scoring.pixel(ep, att, reduce="max").value == pytest.approx(paired)
    with pytest.raises(ValueError):
        scoring.pixel(ep, att, reduce="median")


def test_weighted_mean_over_dreams_and_unscored_dreams_ignored():
    s = BUILDS[0]
    good, bad = dream_of(s, "iso"), dream_of(BUILDS[3], "iso_back")
    specs = [DreamSpec("iso", "t", view="iso", weight=3.0),
             DreamSpec("back", "t", view="iso_back", weight=1.0),
             DreamSpec("mood", "t", view=None)]
    ep = Ep(specs, {"iso": good, "back": bad, "mood": bad})
    t = scoring.color_lab(ep, attempt_for(s))
    v_back = scoring.color_lab_sim(bad, render_pngs(R, s, ["iso_back"])["iso_back"])
    assert t.value == pytest.approx((3 * 1.0 + v_back) / 4, abs=1e-6)
    assert set(t.per_view) == {"iso@iso", "back@iso_back"}


def test_missing_renders_are_rendered_and_stored():
    s = BUILDS[1]
    ep = ep_for(dream_of(s), view="iso", view_search=("top",))
    att = attempt_for(s)
    scoring.pixel(ep, att)
    assert set(att.renders) == {"iso", "top"}
    n = len(ep.rendered)
    scoring.pixel(ep, att)
    assert len(ep.rendered) == n                       # second call reuses them


def test_no_dream_and_empty_build_score_zero():
    att = attempt_for(BUILDS[0])
    for name in ("silhouette", "color_lab", "embed"):
        t = core.make("scorer", name)(Ep(), att)
        assert t.value == 0.0 and "no scored dream" in t.note
    empty = make_attempt("")
    assert empty.blocks == 0
    t = scoring.silhouette(ep_for(dream_of(BUILDS[0])), empty)
    assert t.value == 0.0 and "empty" in t.note


def test_pixel_without_dream_is_coverage_proxy():
    small, big = make_attempt(HUT_PROGRAM), attempt_for(BUILDS[0])
    t_small = scoring.pixel(Ep(), small)
    assert "coverage" in t_small.note and 0 < t_small.value < 1
    assert set(small.renders) == set(scoring.TEXT_VIEWS)
    assert scoring.pixel(Ep(), make_attempt("")).value == 0.0
    assert scoring.pixel(Ep(), big).value > 0


# --- dream-free scorers ---------------------------------------------------------------
def two_blobs(diagonal: bool = False) -> Structure:
    ids = np.zeros((8, 8, 8), np.int32)
    ids[0:2, 0:2, 0:2] = 1                              # 8 blocks
    if diagonal:
        ids[2, 2, 2] = 1                                # touches corner: 26-connected
    else:
        ids[6, 6, 6] = 1                                # 1 separate block
    return Structure(ids, np.zeros_like(ids))


def test_validity_two_blobs():
    t = scoring.validity(Ep(), attempt_for(two_blobs()))
    assert t.value == pytest.approx(8 / 9)
    assert scoring.validity(Ep(), attempt_for(two_blobs(diagonal=True))).value == 1.0
    assert scoring.validity(Ep(), make_attempt("")).value == 0.0
    assert scoring.largest_component_frac(None) == 0.0


def test_material():
    att = make_attempt(HUT_PROGRAM)                    # cobblestone floor, oak plank walls
    assert scoring.material(Ep(prompt="an oak cabin"), att).value == 1.0
    assert scoring.material(Ep(prompt="a quartz temple"), att).value == 0.0
    t = scoring.material(Ep(prompt="a cosy cabin"), att)
    assert t.value == 1.0 and "no material" in t.note
    assert scoring.material(Ep(prompt="an oak cabin"), make_attempt("")).value == 0.0


# --- vlm judge ------------------------------------------------------------------------
def test_vlm_parses_score_and_note():
    s = BUILDS[0]
    reply = {"score": 7, "missing": ["chimney", "door"], "wrong": "roof too flat",
             "fix_hint": "add a gable roof"}
    ep = ep_for(dream_of(s), reply=reply)
    t = scoring.vlm(ep, attempt_for(s))
    assert t.value == pytest.approx(0.7)
    assert "missing: chimney; door" in t.note and "fix_hint: add a gable roof" in t.note
    (turn, kw), = ep.asks
    assert kw["role"] == "judge" and kw["json_reply"] is True and kw["history"] is False
    imgs = [p.image_png for p in turn.parts if p.kind == "image"]
    assert imgs[0] == ep.latest_dream("iso").png and len(imgs) == 2      # TARGET, YOURS
    assert "TARGET iso:" in turn.text and "YOURS iso (view iso):" in turn.text
    assert '"score"' in turn.text
    from blockgen.agentic.turns import JSON_RULE
    assert turn.text.rstrip().endswith(JSON_RULE)        # mock:auto keys on the sentinel


@pytest.mark.parametrize("reply,val", [({}, 0.0), ({"score": "great"}, 0.0),
                                       ({"score": 15}, 1.0), ({"score": -2}, 0.0), (None, 0.0)])
def test_vlm_bad_replies(reply, val):
    s = BUILDS[0]
    t = scoring.vlm(ep_for(dream_of(s), reply=reply), attempt_for(s))
    assert t.value == val


def test_vlm_prompt_rubric_without_dreams_and_empty():
    s = BUILDS[0]
    ep = Ep(reply={"score": 5})
    assert scoring.vlm(ep, attempt_for(s), rubric="prompt").value == 0.5
    turn = ep.asks[0][0]
    assert any(p.kind == "image" for p in turn.parts)  # renders were shown
    assert "TARGET" not in turn.text and "build request" in turn.text
    ep2 = Ep(reply={"score": 5})
    assert scoring.vlm(ep2, make_attempt("")).value == 0.0 and not ep2.asks
    with pytest.raises(ValueError):
        scoring.vlm(ep, attempt_for(s), rubric="vibes")


# --- embed / clip_text maths (model replaced by fixed vectors) --------------------------
@pytest.fixture
def fake_features(monkeypatch):
    """image_features -> unit vectors keyed by image content; text -> fixed vector."""
    def img_feats(pngs, backbone="siglipB", *, crop_fg=True, device=None):
        out = []
        for p in pngs:
            seed = int.from_bytes(__import__("hashlib").sha256(p).digest()[:4], "big")
            v = np.random.default_rng(seed).normal(size=8)
            v[0] += 6.0                                  # everything is fairly similar
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def txt_feats(texts, backbone="clipB16", *, device=None):
        v = np.zeros(8)
        v[0] = 1.0
        return np.stack([v for _ in texts])
    monkeypatch.setattr(scoring, "image_features", img_feats)
    monkeypatch.setattr(scoring, "text_features", txt_feats)
    return img_feats, txt_feats


def test_embed_calibration_formula(fake_features):
    img_feats, _ = fake_features
    s = BUILDS[0]
    d = dream_of(BUILDS[1])
    ep = ep_for(d)
    att = attempt_for(s)
    raw = scoring.embed(ep, att, baseline=0.0).value
    fa, fb = img_feats([d, att.renders["iso"]])
    assert raw == pytest.approx(max(0.0, float(fa @ fb)), abs=1e-6)
    b = 0.5
    assert scoring.embed(ep, att, baseline=b).value == pytest.approx(
        np.clip((raw - b) / (1 - b), 0, 1), abs=1e-6)
    assert scoring.embed(ep, attempt_for(BUILDS[1])).value == pytest.approx(1.0)


def test_embed_default_baseline_is_shipped_table(fake_features):
    from blockgen.agentic.calibrate import BASELINES
    assert scoring.embed_baseline("siglipB") == BASELINES["siglipB"]
    assert 0.0 < BASELINES["siglipB"] < 1.0 and 0.0 < BASELINES["clipB16"] < 1.0
    scoring._WARNED.clear()
    with pytest.warns(RuntimeWarning, match="baseline"):
        assert scoring.embed_baseline("nonesuch") == 0.0


def test_embed_warns_on_bench_backbone(fake_features):
    scoring._WARNED.clear()
    s = BUILDS[0]
    with pytest.warns(RuntimeWarning, match="bench"):
        scoring.embed(ep_for(dream_of(s)), attempt_for(s), backbone="dinov2b", baseline=0.0)


def test_clip_text_offline(fake_features):
    s = BUILDS[0]
    att = attempt_for(s, render_pngs(R, s, ["iso", "iso_back"]))
    t = scoring.clip_text(Ep(), att)
    assert set(t.per_view) == {"iso", "iso_back"}
    assert 0.0 <= t.value <= 1.0
    assert scoring.clip_text(Ep(), make_attempt("")).value == 0.0
    t2 = scoring.clip_text(Ep(), attempt_for(s))                   # no renders: renders TEXT_VIEWS
    assert set(t2.per_view) == set(scoring.TEXT_VIEWS)
    scoring._WARNED.clear()
    with pytest.warns(RuntimeWarning, match="clipL"):
        scoring.clip_text(Ep(), att, backbone="clipL")


def test_feature_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(scoring, "_FEAT_MAX", 3)
    for i in range(5):
        scoring._cache_put(("k", str(i)), np.zeros(2))
    assert len(scoring._FEAT_CACHE) == 3 and ("k", "4") in scoring._FEAT_CACHE
    scoring._FEAT_CACHE.clear()


# --- real models (slow) ------------------------------------------------------------------
def _model_or_skip(backbone: str):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from blockgen.eval.bench.features import BACKBONES
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(BACKBONES[backbone][0], "config.json")
    except Exception:  # noqa: BLE001
        hit = None
    if not isinstance(hit, str):
        pytest.skip(f"{backbone} weights not in the local HF cache")


@pytest.mark.slow
def test_embed_real_siglip_ranks_own_build():
    _model_or_skip("siglipB")
    s, other = BUILDS[0], BUILDS[4]
    ep = ep_for(dream_of(s))
    own = scoring.embed(ep, attempt_for(s), device="cpu")
    oth = scoring.embed(ep, attempt_for(other), device="cpu")
    assert own.value == pytest.approx(1.0, abs=1e-3)
    assert own.value > oth.value


@pytest.mark.slow
def test_clip_text_real_prefers_matching_prompt():
    _model_or_skip("clipB16")
    att = attempt_for(BUILDS[0], render_pngs(R, BUILDS[0], ["iso"]))
    house = scoring.clip_text(Ep(prompt="a voxel house with a stepped roof"), att, device="cpu")
    other = scoring.clip_text(Ep(prompt="a bowl of ramen noodles"), att, device="cpu")
    assert house.value > other.value


# --- view sets are fixed per recipe, not "whatever the attempt has rendered" -----------
def test_dream_free_scorers_ignore_extra_cached_renders(fake_features):
    """Feedback turns add renders (legacy_a, cutaway) to SOME attempts. clip_text,
    pixel's proxy and reduce='max' must not change because of that, or a
    dream-independent final_score compares attempts over different view sets."""
    s = BUILDS[0]
    plain = attempt_for(s)
    extra = attempt_for(s, render_pngs(R, s, ["legacy_a", "cutaway", "top"]))
    assert scoring.clip_text(Ep(), plain).value == pytest.approx(
        scoring.clip_text(Ep(), extra).value)
    assert set(scoring.clip_text(Ep(), extra).per_view) == set(scoring.TEXT_VIEWS)
    assert scoring.pixel(Ep(), plain).value == pytest.approx(scoring.pixel(Ep(), extra).value)
    ep = ep_for(dream_of(s, "top"), view="iso", view_search=("iso_left",))
    t_plain = scoring.pixel(ep, attempt_for(s), reduce="max")
    t_extra = scoring.pixel(ep, attempt_for(s, render_pngs(R, s, ["top"])), reduce="max")
    assert t_plain.value == pytest.approx(t_extra.value)
    assert set(t_extra.per_view) == {"iso@iso", "iso@iso_left"}


def test_per_view_feeds_turns_matched_view():
    """turns.matched_view picks the YOURS camera from per_view; the scorer must
    report every candidate view under '<dream>@<view>' so it can."""
    from blockgen.agentic import turns
    s = BUILDS[2]
    ds = DreamSpec("iso", "{desc}", view="iso", view_search=("iso_back", "iso_left"))
    ep = Ep([ds], {"iso": dream_of(s, "iso_left")})
    att = attempt_for(s)
    parts = {n: core.make("scorer", n)(ep, att) for n in ("silhouette", "pixel")}
    att.score = core.Score(1.0, {n: t.value for n, t in parts.items()},
                           {n: t.per_view for n, t in parts.items()})
    assert turns.matched_view(att, ds) == "iso_left"


def test_vlm_prompt_rubric_uses_fixed_views():
    s = BUILDS[0]
    att = attempt_for(s, render_pngs(R, s, ["legacy_a", "cutaway"]))
    ep = Ep(reply={"score": 5})
    scoring.vlm(ep, att, rubric="prompt")
    turn = ep.asks[-1][0]
    labels = [p.text for p in turn.parts if p.kind == "text" and p.text.startswith("YOURS")]
    assert labels == [f"YOURS (view {v}):" for v in scoring.TEXT_VIEWS]
