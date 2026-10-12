"""report.py stays backward compatible: legacy BuildResults get only new aggregate keys."""

from __future__ import annotations

import json

import numpy as np

from blockgen.agentic import report
from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic.canvas import Canvas
from blockgen.agentic.dsl import ExecutionReport
from blockgen.agentic.agent import BuildResult
from blockgen.agentic.providers import ScriptedProvider

#: metrics['arms'][arm] keys report.write_run produced before the suite (git HEAD).
LEGACY_ARM_KEYS = {"n_builds", "n_empty", *report.AGGREGATE_KEYS, "total_cost_usd",
                   "total_elapsed_s", "coherence_rate"}
NEW_ARM_KEYS = {"cost_known", "total_would_cost_usd", "n_truncated"}


def _legacy_results():
    cfg = AgentConfig(provider="mock", canvas_size=(16, 16, 16), verbose=False)
    ok = BuildAgent(ScriptedProvider(), cfg).build("a hut")
    empty_canvas = Canvas((16, 16, 16))
    empty = BuildResult(description="nothing", structure=empty_canvas.to_structure(crop=False),
                        canvas=empty_canvas, program_text="", report=ExecutionReport(),
                        config=cfg, error="boom")
    return [ok, empty]


def test_write_run_on_plain_buildresults_adds_only_new_keys(tmp_path):
    m = report.write_run(tmp_path, arms={"a": _legacy_results()}, config={"x": 1},
                         render=False, max_dim=16)
    keys = set(m["arms"]["a"])
    assert keys == LEGACY_ARM_KEYS | NEW_ARM_KEYS
    assert m["arms"]["a"]["n_truncated"] == 0
    assert m["arms"]["a"]["total_would_cost_usd"] == m["arms"]["a"]["total_cost_usd"]
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert set(on_disk) == {"arms", "builds"}
    assert "Method metrics" not in (tmp_path / "summary.md").read_text()
    manifest = json.loads((tmp_path / "a_16_manifest.json").read_text())
    assert manifest["report"] == {"arm": "a", "n_requested": 2, "n_nonempty": 1}


def test_provenance_and_keep_empty(tmp_path):
    from blockgen.curation.houses import load_structures_from_cache
    res = _legacy_results()
    report.write_run(tmp_path, arms={"a": res}, config={}, render=False, max_dim=16,
                     provenance={"a": {"method": "legacy", "train_split": "none"}})
    rep = json.loads((tmp_path / "a_16_manifest.json").read_text())["report"]
    assert rep["method"] == "legacy" and rep["n_requested"] == 2
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["arms"]["a"]["provenance"]["train_split"] == "none"

    path = report.save_structures(res, tmp_path / "k", "a", max_dim=16, keep_empty=True,
                                  extra_report={"recipe": "r"})
    structs, manifest = load_structures_from_cache(path)
    assert len(structs) == 2 and manifest["report"]["recipe"] == "r"
    assert int(np.asarray(structs[1].block_ids).sum()) == 0
    assert manifest["items"][1]["title"] == "nothing"
    assert report.save_structures(res[1:], tmp_path / "e", "a", max_dim=16) is None
