"""Trace files and contact sheet carry the selection value (pilot fix D1) and the
attempt annotations of staged methods (unit scores, reverted writes)."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from blockgen.agentic import trace as tr
from blockgen.agentic.core import ScoreTerm
from blockgen.agentic.episode import run_episode

from .fakes import make_fake_kit

SEL_RGB = (40, 40, 140)          # the third header line's ink


def _third_line_has_ink(img: np.ndarray, col: int) -> bool:
    band = img[31:45, col * tr.TILE + 2:(col + 1) * tr.TILE - 4]
    return bool((np.abs(band.astype(int) - SEL_RGB).sum(axis=-1) < 90).any())


def _run(tmp_path, final: bool, method=None):
    def default(ep):
        ep.try_("generate", "fill stone 0 0 0 3 0 3\n")
        ep.try_("match0", "fill stone 0 0 0 7 0 7\n")

    small_wins = lambda ep, a: ScoreTerm(1.0 / max(1, a.blocks))  # noqa: E731
    kw = {"final_scorers": {"inv": (small_wins, 1.0)}} if final else {}
    kit = make_fake_kit(method=method or default, spec_kw={"select": "best"}, **kw)
    root = tmp_path / ("final" if final else "plain")
    res = run_episode(kit, "a hut", trace=tr.Trace.for_kit(root, kit))
    return root, res


def test_header_is_three_lines():
    assert tr.HEADER == 48


def test_score_json_has_selection_and_extra(tmp_path):
    root, res = _run(tmp_path, final=True)
    sc = json.loads((root / "iters/00_generate/score.json").read_text())
    assert sc["selection"] == 1 / 16 and sc["extra"] == {}
    assert res.best_idx == 0
    blob = json.loads((root / "result.json").read_text())
    assert blob["attempts"][1]["selection"] == 1 / 64


def test_contact_sheet_shows_sel_when_final_scorers_exist(tmp_path):
    root, _ = _run(tmp_path, final=True)
    img = np.asarray(Image.open(root / "contact.png").convert("RGB"))
    lay = tr.contact_layout(root)
    assert img.shape[0] == tr.HEADER + lay["rows"] * tr.TILE
    assert _third_line_has_ink(img, 1) and _third_line_has_ink(img, 2)
    plain, _ = _run(tmp_path, final=False)
    img2 = np.asarray(Image.open(plain / "contact.png").convert("RGB"))
    assert not _third_line_has_ink(img2, 1)


def test_contact_sheet_shows_unit_score(tmp_path):
    def method(ep):
        att = ep.try_("u_kitchen", "fill stone 0 0 0 3 0 3\n")
        att.extra.update({"unit": "kitchen", "unit_score": 0.41})

    root, _ = _run(tmp_path, final=False, method=method)
    sc = json.loads((root / "iters/00_u_kitchen/score.json").read_text())
    assert sc["extra"] == {"unit": "kitchen", "unit_score": 0.41}
    img = np.asarray(Image.open(root / "contact.png").convert("RGB"))
    assert _third_line_has_ink(img, 1)


def test_gated_selection_reads_gated(tmp_path):
    def method(ep):
        ep.try_("generate", "fill stone 0 0 0 3 0 3\n")
        ep.try_("empty", "clear 0 0 0 0 0 0\n")

    root, res = _run(tmp_path, final=True, method=method)
    sc = json.loads((root / "iters/01_empty/score.json").read_text())
    assert sc["selection"] == -1.0
    img = np.asarray(Image.open(root / "contact.png").convert("RGB"))
    assert _third_line_has_ink(img, 2)
