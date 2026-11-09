"""VLM-judge arm, offline: a mocked provider, synthetic stimuli, no API and no GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.agentic.providers import LLMProvider, LLMResponse
from blockgen.eval.bench import vlm_judge as vj


class FakeStore(vj.StimulusStore):
    """Solid-colour PNGs; the colour encodes the arm so the mock judge can 'see' it."""

    COLORS = {"real_test": (0, 200, 0), "real@solidify": (0, 200, 0),
              "uniform_random": (200, 0, 0), "gabled_house": (0, 0, 200),
              "patchwork": (200, 200, 0), "real@hollow": (0, 200, 0),
              "real3dc": (0, 200, 0), "real3dc@solidify": (0, 200, 0),
              "real3dc@hollow": (0, 200, 0)}

    def _make(self, b):
        img = np.zeros((8, 8, 3), np.uint8)
        img[:] = self.COLORS.get(b.arm, (90, 90, 90))
        img[0, 0] = (b.idx % 256, 0, 0)  # distinct bytes per build
        return vj.png_bytes(img)


class MockJudge(LLMProvider):
    """Prefers green (real-looking) over red; otherwise always answers 'A'."""

    name = "mockjudge"
    spends = False

    def __init__(self):
        super().__init__("mock-vlm")
        self.calls = 0

    def complete(self, messages, **kw):
        from PIL import Image
        import io
        self.calls += 1
        imgs = [p.image_png for m in messages for p in m.parts if p.kind == "image"]
        assert len(imgs) == 2
        g = [np.asarray(Image.open(io.BytesIO(x)).convert("RGB"))[4, 4] for x in imgs]
        if g[0][1] > g[0][0] and g[1][0] > g[1][1]:
            choice = "A"
        elif g[1][1] > g[1][0] and g[0][0] > g[0][1]:
            choice = "B"
        else:
            choice = "A"            # pure position bias when it cannot tell
        return LLMResponse(text=json.dumps({"choice": choice, "reason": "x"}),
                           model="gpt-4o-2024-11-20", prompt_tokens=1000,
                           completion_tokens=20)


def _plan(conds, n=128, system=(), k=0):
    arms_len = {a: n for a in list(vj.SYSTEM_ARMS) + ["real@hollow", vj.REAL3,
                                                      vj.REAL3 + "@solidify",
                                                      vj.REAL3 + "@hollow"]}
    return vj.make_trials(conds, arms_len, seed=0, system_arms=system, system_k=k)


def test_both_orders_and_pairing():
    conds = [c for c in vj.default_conditions() if c.name in ("interior_ext", "catch")]
    trials = _plan(conds)
    by_pair = {}
    for t in trials:
        by_pair.setdefault(t.pair, []).append(t)
    assert all(len(v) == 2 for v in by_pair.values())
    for a, b in by_pair.values():
        assert (a.a, a.b) == (b.b, b.a) and {a.order, b.order} == {0, 1}
        if a.condition == "interior_ext":       # paired: the same house
            assert a.a.idx == a.b.idx
    assert vj.plan_sha(trials) == vj.plan_sha(_plan(conds))   # deterministic


def test_parse_choice():
    assert vj.parse_choice('{"choice": "B", "reason": "r"}') == "B"
    assert vj.parse_choice('```json\n{"choice":"a"}\n```') == "A"
    assert vj.parse_choice('blah "choice": "A", trailing') == "A"
    assert vj.parse_choice("B") == "B"
    assert vj.parse_choice("I cannot decide") is None


def test_end_to_end_with_mock(tmp_path):
    conds = [c for c in vj.default_conditions()
             if c.name in ("interior_ext", "catch", "gabled_patchwork")]
    for c in conds:
        c.n_pairs = 20
    trials = _plan(conds)
    store = FakeStore(tmp_path / "stim")
    judge = MockJudge()
    recs = vj.run_trials(judge, trials, store, tmp_path / "r.jsonl", workers=4,
                         log=lambda m: None)
    assert judge.calls == len(trials)
    an = vj.analyze(recs, conds, n_boot=200)
    cat = an["conditions"]["catch"]
    assert cat["rate_a"] == 1.0 and cat["prediction_met"] is True
    assert cat["order_agreement"] == 1.0
    ext = an["conditions"]["interior_ext"]
    # Indistinguishable stimuli + an always-'A' judge: exactly chance, all position.
    assert ext["rate_a"] == 0.5 and ext["p_first_slot"] == 1.0
    assert ext["order_averaged_rate_a"] == 0.5 and ext["position_driven_pairs"] == 1.0
    assert ext["prediction_met"] is True
    assert an["overall"]["cost_usd"] == pytest.approx(len(trials) * (1000 * 2.5 + 20 * 10) / 1e6)
    # Resumable: nothing left to do, no new calls.
    vj.run_trials(judge, trials, store, tmp_path / "r.jsonl", log=lambda m: None)
    assert judge.calls == len(trials)


def test_system_bt_and_correlations(tmp_path):
    system = ("real_test", "gabled_house", "patchwork", "uniform_random")
    trials = _plan([], system=system, k=6)
    recs = vj.run_trials(MockJudge(), trials, FakeStore(tmp_path / "s"),
                         tmp_path / "r.jsonl", log=lambda m: None)
    metrics = {"real_test": {"blockscore": 0.0, "mv_kid": 0.0, "v_kid": 0.0},
               "gabled_house": {"blockscore": 300.0, "mv_kid": 2.0, "v_kid": 30.0},
               "patchwork": {"blockscore": 246.0, "mv_kid": 1.5, "v_kid": 18.0},
               "uniform_random": {"blockscore": 603.0, "mv_kid": 5.0, "v_kid": 34.0}}
    an = vj.system_analysis(recs, metrics, arms=system, n_boot=50)
    assert an["order"][0] == "real_test" and an["order"][-1] == "uniform_random"
    c = an["correlations"]["all/blockscore"]
    assert -1 <= c["kendall"] <= 1 and c["kendall_ci"] is not None


def test_bt_matches_the_rater_study_estimator():
    from scripts.human_study_analyze import bradley_terry as ref
    arms = ["a", "b", "c", "d"]
    rng = np.random.default_rng(3)
    w = rng.integers(0, 9, size=(4, 4)).astype(float)
    np.fill_diagonal(w, 0)
    wins = {(arms[i], arms[j]): int(w[i, j]) for i in range(4) for j in range(4) if i != j}
    want = ref(wins, arms, iters=5000)
    got = vj.bradley_terry(w, iters=5000)
    assert np.allclose([want[a] for a in arms], got, atol=1e-6)


def test_grid_and_protocols():
    v = [np.full((5, 6, 3), k * 40, np.uint8) for k in range(4)]
    g = vj.grid_2x2(v, gutter=2)
    assert g.shape == (12, 14, 3) and g[0, 0, 0] == 0 and g[-1, -1, 0] == 120
    assert vj.view_config("ext4").key() == vj.view_config("ext4").key()
    assert len(vj.view_config("cut4_roof").cuts) == 4
    for p in ("ext4", "cut4_roof", "cut4_quad", *vj.HUMAN_PROTOCOLS):
        assert p in vj.VIEW_NOTES


def test_door_diagnostic_splits_pairs():
    conds = [c for c in vj.default_conditions() if c.name == "interior_ext"]
    conds[0].n_pairs = 10
    trials = _plan(conds, n=10)
    recs = []
    for t in trials:      # a judge that prefers real only when the house is "changed"
        idx = t.a.idx
        pref = idx < 5
        recs.append({**t.to_dict(), "valid": True, "prefers_arm_a": pref,
                     "choice": "A" if pref == (t.order == 0) else "B"})
    vis = [{"added": 3, "visible_added": 2 if i < 5 else 0, "behind_window": 0}
           for i in range(10)]
    dd = vj.door_diagnostic(recs, vis)["reachable"]
    assert dd["frac_houses_changed"] == 0.5
    assert dd["interior_ext"]["changed"]["rate_real"] == 1.0
    assert dd["interior_ext"]["unchanged"]["rate_real"] == 0.0
