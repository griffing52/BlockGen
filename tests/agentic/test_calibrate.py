"""WP5 calibration: oracle-dream ranking (offline with flat + synthetic builds)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.agentic import calibrate
from blockgen.agentic.calibrate import (BASELINES, CalibEpisode, attempt_for, run,
                                        synthetic_builds)
from blockgen.agentic.core import DreamSpec, DreamVersion
from blockgen.agentic.render import FlatRenderer


def test_synthetic_builds_distinct_and_deterministic():
    a, b = synthetic_builds(8, seed=0), synthetic_builds(8, seed=0)
    assert len(a) == 8
    for x, y in zip(a, b):
        assert np.array_equal(x.block_ids, y.block_ids)
    sigs = {(x.shape, int(x.occupied_mask.sum())) for x in a}
    assert len(sigs) == 8
    assert all(int(x.occupied_mask.sum()) > 0 for x in a)


@pytest.mark.parametrize("scorer", ["pixel", "color_lab"])
def test_calibrate_offline_top1(scorer):
    res = run(scorer, n=8, distractors=7, renderer=FlatRenderer(px=96),
              structures=synthetic_builds(8))
    assert res["n"] == 8 and res["distractors"] == 7
    assert res["top1"] > 0.8
    assert res["chance_top1"] == pytest.approx(1 / 8)
    assert res["margin"] > 0
    assert 0 <= res["baseline"] < res["own_mean"] <= 1
    assert len(res["items"]) == 8 and {"own", "best_distractor", "rank"} <= set(res["items"][0])
    assert res["renderer"] == "flat" and res["source"] == "structures"
    json.dumps(res)


def test_calibrate_renderer_spec_string_and_fewer_builds_than_distractors():
    res = run("pixel", n=4, distractors=31, renderer="flat?px=64", structures=synthetic_builds(4))
    assert res["distractors"] == 3


def test_calibrate_rejects_dream_free_llm_and_identical_views():
    builds = synthetic_builds(3)
    with pytest.raises(ValueError, match="dream"):
        run("validity", n=3, renderer="flat", structures=builds)
    with pytest.raises(ValueError, match="LLM"):
        run("vlm", n=3, renderer="flat", structures=builds)
    with pytest.raises(ValueError, match="identical"):
        run("pixel", n=3, renderer="flat", structures=builds, match_views=("iso", "iso_left"))
    with pytest.raises(ValueError, match="2 builds"):
        run("pixel", n=1, renderer="flat", structures=builds)


def test_calib_episode_has_no_llm():
    s = synthetic_builds(1)[0]
    ep = CalibEpisode(DreamVersion("oracle", 0, b"", "p", "g"), DreamSpec("oracle", ""),
                      FlatRenderer(px=32))
    with pytest.raises(RuntimeError, match="no LLM"):
        ep.ask("hi")
    out = ep.render(attempt_for(s), ["iso", "top"])
    assert set(out) == {"iso", "top"}
    assert ep.latest_dream("oracle").key == "oracle" and ep.latest_dream("x") is None


def test_dir_dream_loader(tmp_path):
    (tmp_path / "00012.png").write_bytes(b"a")
    (tmp_path / "dream_7.png").write_bytes(b"b")
    (tmp_path / "notes.png").write_bytes(b"c")
    assert calibrate._load_dir_dreams(tmp_path) == {12: b"a", 7: b"b"}
    with pytest.raises(FileNotFoundError):
        calibrate._load_dir_dreams(tmp_path / "missing")


def test_baselines_table():
    assert set(BASELINES) >= {"siglipB", "clipB16"}
    assert all(0.0 < v < 1.0 for v in BASELINES.values())


def test_cli_synthetic(capsys, tmp_path):
    out = tmp_path / "cal.json"
    calibrate.main(["--scorer", "pixel", "--n", "4", "--distractors", "3", "--renderer", "flat",
                    "--synthetic", "--out", str(out)])
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{"):])
    assert "items" not in summary and summary["top1"] >= 0
    data = json.loads(out.read_text())
    assert data["n"] == 4 and data["scorer"] == "pixel" and "items" in data


@pytest.mark.slow
def test_calibrate_on_corpus_train_split():
    from blockgen.agentic.render import REPO_ROOT
    if not (REPO_ROOT / "data/minecraft/cache/houses_32.npz").exists():
        pytest.skip("houses_32 corpus cache not present")
    res = run("color_lab", n=8, distractors=7, renderer="flat?px=128")
    assert res["source"] == "oracle:houses_32/train"
    assert res["top1"] > 0.5
