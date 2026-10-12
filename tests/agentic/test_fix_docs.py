"""Regression tests for the docs review fixes (agentic-suite.md, ontology.md)."""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from blockgen.agentic import dreams
from blockgen.agentic.episode import Episode

from .fakes import make_fake_kit

ROOT = Path(__file__).resolve().parents[2]
SUITE_DOC = ROOT / "docs" / "agentic-suite.md"
CALIB_DIR = ROOT / "outputs" / "run_20261003_130834_calibrate"


def _method_example() -> str:
    text = SUITE_DOC.read_text()
    sec = text[text.index("### 6.1 A method"):]
    return re.search(r"```python\n(.*?)```", sec, re.S).group(1)


def test_doc_dream_example_keeps_style_and_template(registry_snapshot):
    """The §6.1 example must fill only {desc}; a raw prompt= drops style and camera."""
    src = _method_example()
    ns: dict = {}
    exec(compile(src, "agentic-suite.md#6.1", "exec"), ns)

    def _plan(ep):
        ep.plan = "LAYER 0 stone floor"
        return ep.plan

    ns["steps"] = types.SimpleNamespace(plan=_plan, generate=lambda ep: None,
                                        match_step=None)
    kit = make_fake_kit(dreams=["iso"])
    ep = Episode("a small oak cottage", kit)
    ns["plan_then_dream"](ep, max_iters=0)
    sent = kit.image.requests[-1].prompt
    assert sent.startswith(dreams.DEFAULT_STYLE)
    assert "a small oak cottage" in sent and "LAYER 0 stone floor" in sent


def test_suite_doc_does_not_link_docs_results_page():
    # docs/results.md has no suite entry; T41 lives in the repo-root results.md.
    assert "[Results](results.md)" not in SUITE_DOC.read_text()


@pytest.mark.skipif(not CALIB_DIR.is_dir(), reason="calibration run dir not present")
def test_calibration_numbers_match_saved_run():
    text = " ".join(SUITE_DOC.read_text().split())
    assert CALIB_DIR.name in text
    expect = {"embed_siglipB": "siglipB top-1", "embed_clipB16": "clipB16",
              "color_lab": "`color_lab`", "pixel": "`pixel`", "silhouette": "`silhouette`"}
    for stem, label in expect.items():
        d = json.loads((CALIB_DIR / f"{stem}.json").read_text())
        assert d["n"] == 32 and d["seed"] == 0
        assert f"{label} {d['top1']:.2f}" in text, (stem, d["top1"])


def test_ontology_doc_prompt_sizes_match_agentic_doc_and_measurement():
    from blockgen.ontology import variant_catalog
    from blockgen.ontology.minecraft import prompt_block

    onto = (ROOT / "docs" / "ontology.md").read_text()
    agentic = (ROOT / "docs" / "agentic.md").read_text()
    for arm, variant in (("ont_mined", "mined"), ("ont_shuffled", "shuffled"),
                         ("ont_stats", "stats")):
        chars = len(prompt_block(variant_catalog(variant)))
        label = f"~{chars / 4000:.1f}k"
        row_o = next(l for l in onto.splitlines() if l.startswith(f"| `{arm}`"))
        row_a = next(l for l in agentic.splitlines() if l.startswith(f"| `{arm}`"))
        assert label in row_o and label in row_a, (arm, label, row_o, row_a)
        assert f"{chars / 1000:.1f}k chars" in row_o
