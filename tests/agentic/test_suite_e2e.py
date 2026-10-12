"""End-to-end suite runs, fully offline (spec WP7 acceptance).

Every run here is ``mock=True`` with ``mock:solid`` dreams (no corpus needed) and
the flat renderer, into a tmp run dir, so the whole file runs in seconds with no
network, no GPU and no spend. The textured samples.png path has its own test,
skipped when EGL is unavailable.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from blockgen.agentic import core
from blockgen.agentic.recipe import load_battery, load_recipe
from blockgen.agentic import suite

OFFLINE = {"image": "mock:solid", "renderer": "flat"}
E2E_RECIPES = ["dream", "classic-matched", "dream-multiview", "mixed-dream",
               "python-oneshot", "blocks-oneshot", "texsheet-oneshot", "dsl-classic"]
PROMPTS = ["a brick house", "a wooden cabin"]


@pytest.fixture(autouse=True)
def _clean_kit_cache():
    from blockgen.agentic import kit
    kit.clear_cache()
    yield
    kit.clear_cache()


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    """One mock run of the acceptance recipes (shared: it is the expensive part)."""
    from blockgen.agentic import kit
    kit.clear_cache()
    base = tmp_path_factory.mktemp("runs")
    specs = [load_recipe(n) for n in E2E_RECIPES]
    run = suite.run_recipes(specs, PROMPTS, name="e2e", mock=True, post_mock=OFFLINE,
                            render=False, base=base)
    return run


def test_run_dir_layout(mock_run):
    run = mock_run
    assert run.name.endswith("_e2e_mock")
    for f in ("config.json", "metrics.json", "summary.md", "variants.png"):
        assert (run / f).is_file(), f
    for arm in E2E_RECIPES:
        assert (run / f"{arm}_32.npz").is_file()
        assert (run / f"{arm}_32_manifest.json").is_file()
        assert (run / f"contact_{arm}.png").is_file()
        for i in range(len(PROMPTS)):
            root = run / "builds" / arm / f"{i:03d}"
            assert (root / "contact.png").is_file(), root
            assert (root / "result.json").is_file()
            assert (run / "builds" / arm / f"{i:03d}_program.txt").is_file()
            assert list((root / "iters").glob("*/structure.npz"))


def test_manifest_provenance_and_reload(mock_run):
    from blockgen.curation.houses import load_structures_from_cache
    for arm in E2E_RECIPES:
        structs, manifest = load_structures_from_cache(str(mock_run / f"{arm}_32.npz"))
        assert len(structs) == len(PROMPTS)
        assert all(int(s.occupied_mask.sum()) > 0 for s in structs)
        rep = manifest["report"]
        for key in ("method", "recipe", "recipe_hash", "action_repr", "backend",
                    "image_generator", "renderer", "contexts", "dreams", "loop",
                    "train_split", "n_requested", "git_commit"):
            assert key in rep, (arm, key)
        assert rep["mock"] is True and rep["n_requested"] == len(PROMPTS)
        assert [it["title"] for it in manifest["items"]] == PROMPTS
    assert json.loads((mock_run / "dream_32_manifest.json").read_text())["report"][
        "train_split"] == "none"       # mock:solid is not a train-split render


def test_metrics_cost_fields(mock_run):
    m = json.loads((mock_run / "metrics.json").read_text())
    for arm in E2E_RECIPES:
        row = m["arms"][arm]
        assert row["cost_known"] is False           # mocks are unpriced, never "$0"
        assert "total_would_cost_usd" in row and row["n_truncated"] == 0
        assert row["ledger"]["llm_calls"] > 0
        assert row["provenance"]["recipe"] == arm
    assert m["ledger"]["usd"] == 0.0
    assert m["arms"]["dream"]["n_attempts"] >= 2
    assert m["arms"]["dream-multiview"]["dream_score_best"] is not None
    cfg = json.loads((mock_run / "config.json").read_text())
    assert set(cfg["recipes"]) == set(E2E_RECIPES)
    assert "git_commit" in cfg and "git_dirty" in cfg and cfg["mock"] is True
    assert cfg["recipes"]["dream"]["image"] == "mock:solid"


def test_multiview_trace_has_every_dream(mock_run):
    root = mock_run / "builds" / "dream-multiview" / "000"
    keys = {p.stem.rsplit("_v", 1)[0] for p in (root / "dreams").glob("*.png")}
    assert keys == {"iso", "back", "interior", "plan", "mood"}
    msgs = [json.loads(line) for line in (root / "messages.jsonl").read_text().splitlines()]
    assert msgs and all("role" in r for r in msgs)


def test_read_only_rerun_is_identical(mock_run, tmp_path):
    first = json.loads((mock_run / "metrics.json").read_text())
    specs = [load_recipe(n) for n in ("dream", "classic-matched")]
    run = suite.run_recipes(specs, PROMPTS, name="ro", mock=True, post_mock=OFFLINE,
                            render=False, base=tmp_path, cache_mode="read_only")
    again = json.loads((run / "metrics.json").read_text())
    for arm in ("dream", "classic-matched"):
        assert again["arms"][arm]["n_truncated"] == 0
        a = [(b["blocks"], b["n_attempts"]) for b in first["builds"][arm]]
        b = [(b["blocks"], b["n_attempts"]) for b in again["builds"][arm]]
        assert a == b
        for i in range(len(PROMPTS)):
            p1 = (mock_run / "builds" / arm / f"{i:03d}_program.txt").read_text()
            p2 = (run / "builds" / arm / f"{i:03d}_program.txt").read_text()
            assert p1 == p2


def test_real_backend_without_allow_spend_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(core.SPEND_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(core.SpendNotAllowed, match="allow-spend"):
        suite.run_recipes([load_recipe("classic-full", {"cache": "off"})], PROMPTS[:1],
                          name="gate", render=False, base=tmp_path)


def test_kit_failure_is_recorded_not_fatal(tmp_path):
    # Un-mocked on purpose (mock would replace the bad llm): the arm fails while its
    # kit is built, before any backend call, so nothing can spend.
    bad = load_recipe("classic-full", {"llm": "nope:model", "name": "bad"})
    run = suite.run_recipes([bad], PROMPTS[:1], name="kitfail", render=False, base=tmp_path)
    m = json.loads((run / "metrics.json").read_text())
    assert m["builds"]["bad"][0]["error"].startswith("kit: ValueError")
    assert "kit_error" in m["arms"]["bad"]["provenance"]


def test_battery_with_inline_override_and_sweep(tmp_path):
    (tmp_path / "b.yaml").write_text(
        "name: tiny\nprompts: {source: short, n: 1}\nbudget: {usd: 1.0}\n"
        "recipes:\n  - classic-matched\n  - {dream: {as: dream_vdm, params.feedback: vdm}}\n"
        "sweep:\n  base: dream\n  grid: {params.max_iters: [1, 2]}\n")
    b = load_battery(tmp_path / "b.yaml")
    run = suite.run_battery(b, mock=True, post_mock=OFFLINE, render=False,
                            base=tmp_path / "out")
    m = json.loads((run / "metrics.json").read_text())
    assert set(m["arms"]) == {"classic-matched", "dream_vdm", "dream@max_iters=1",
                              "dream@max_iters=2"}
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["battery"]["name"] == "tiny" and len(cfg["prompts"]) == 1
    assert (m["arms"]["dream@max_iters=1"]["n_attempts"]
            < m["arms"]["dream@max_iters=2"]["n_attempts"])


def test_dreams_first(tmp_path):
    run = suite.run_recipes([load_recipe("dream")], PROMPTS[:1], name="df", mock=True,
                            post_mock=OFFLINE, render=False, base=tmp_path,
                            dreams_first=True)
    res = json.loads((run / "builds" / "dream" / "000" / "result.json").read_text())
    assert res["dreams"]["iso"] and not res.get("truncated")
    # The build phase read the pre-generated dream from the cache.
    charges = [json.loads(x) for x in
               (run / "builds" / "dream" / "000" / "charges.jsonl").read_text().splitlines()]
    assert any(c["kind"] == "image" and c["cached"] for c in charges)


def test_estimate_prices_real_models_without_network(monkeypatch):
    def no_net(*a, **k):
        raise AssertionError("estimate tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.delenv(core.SPEND_ENV, raising=False)
    rows = suite.estimate([load_recipe("dream-refine"), load_recipe("classic-matched")],
                          PROMPTS[:1])
    by = {r["arm"]: r for r in rows}
    assert set(by) == {"dream-refine", "classic-matched", "TOTAL"}
    dr = by["dream-refine"]
    # Upper bound: the worst-case critic forces all K=2 regenerations (+ a re-dream).
    assert dr["images"] >= 3 and dr["est_full_usd"] > 0 and dr["cost_known"] is True
    assert by["classic-matched"]["images"] == 0 and by["classic-matched"]["est_full_usd"] > 0
    assert by["TOTAL"]["est_full_usd"] == pytest.approx(
        dr["est_full_usd"] + by["classic-matched"]["est_full_usd"], abs=1e-3)
    assert "est_full_usd" in suite.format_estimate(rows)


def test_estimate_marks_unpriced_backends(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    spec = load_recipe("dream", {"llm": "local:qwen", "image": "mock:solid"})
    row = suite.estimate([spec], PROMPTS[:1])[0]
    assert row["cost_known"] is False


def test_rescore_offline(mock_run):
    path = suite.rescore(mock_run, ["validity", "pixel"], arms=["dream", "classic-matched"],
                         mock=True, renderer="flat")
    blob = json.loads(Path(path).read_text())
    assert path.name.startswith("rescore_")
    for arm in ("dream", "classic-matched"):
        builds = blob["arms"][arm]["builds"]
        assert set(builds) == {"000", "001"}
        rows = [r for b in builds.values() for r in b["attempts"].values()]
        assert rows and all(set(r["parts"]) == {"validity", "pixel"} for r in rows)
        assert all(r["parts"]["validity"] == pytest.approx(1.0) for r in rows)
    assert blob["arms"]["dream"]["builds"]["000"]["attempts"]
    # Dream-anchored: the dream arm's pixel term compares against its trace dream.
    assert any(r["dream_id"] for b in blob["arms"]["dream"]["builds"].values()
               for r in b["attempts"].values())


def test_prepare_spec_tags_oracle_and_keeps_explicit_choices(monkeypatch):
    from blockgen.agentic import kit as kitmod
    monkeypatch.setattr(kitmod, "textured_available", lambda: True)
    spec = suite.prepare_spec(load_recipe("dream"), mock=True)
    assert spec.image == "mock:render" and spec.tags["train_split"] == "oracle_train"
    assert spec.llm == "mock:auto" and spec.renderer == "textured"   # --mock keeps real renders
    spec = suite.prepare_spec(load_recipe("dream"), mock=True,
                              post_mock={"renderer": "textured", "image": "mock:solid"})
    assert spec.renderer == "textured" and spec.image == "mock:solid"
    assert spec.tags.get("train_split") is None and spec.tags["mock"] is True
    assert load_recipe("dream-oracle").provenance()["train_split"] == "oracle_train"


def test_parse_prompts():
    assert len(suite.parse_prompts("short:3")) == 3
    assert len(suite.parse_prompts("short:3", n=1)) == 1
    assert len(suite.parse_prompts("detailed", n=2)) == 2


def _egl_ok() -> bool:
    try:
        from blockgen.agentic.core import make, VIEWS
        from blockgen.agentic.core import MinecraftWorld
        from blockgen.agentic.core import Lowered
        w = MinecraftWorld.empty((8, 8, 8))
        w.run(Lowered("fill stone 0 0 0 2 2 2", "fill stone 0 0 0 2 2 2"))
        make("renderer", "textured").render(w.to_structure(), [VIEWS["iso"]])
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.slow
def test_textured_samples_png(tmp_path):
    if not _egl_ok():
        pytest.skip("textured renderer (EGL) unavailable")
    run = suite.run_recipes([load_recipe("dream")], PROMPTS[:1], name="tex", mock=True,
                            post_mock={"image": "mock:solid", "renderer": "textured"},
                            render=True, base=tmp_path)
    assert (run / "samples.png").is_file()
    assert (run / "builds" / "dream" / "000" / "contact.png").stat().st_size > 10_000
