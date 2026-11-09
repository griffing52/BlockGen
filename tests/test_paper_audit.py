"""scripts/audit_claims.py (TeX cross-check) and scripts/paper_readiness.py.

The fast tests build a toy paper in tmp_path; the two that read the real run
directories skip when those runs are not on disk.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_claims as ac  # noqa: E402
import paper_readiness as pr  # noqa: E402


def _paper(tmp_path, results="", appendix="", supp=None):
    paper = tmp_path / "paper"
    (paper / "sec").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\input{math_commands}\n\\begin{document}\n"
        "\\input{sec/6_results}\n\\bibliography{references}\n\\appendix\n"
        "\\input{sec/X_appendix}\n\\end{document}\n")
    (paper / "math_commands.tex").write_text("\\newcommand{\\x}{3.14159}\n")
    (paper / "sec" / "6_results.tex").write_text(results)
    (paper / "sec" / "X_appendix.tex").write_text(appendix)
    if supp is not None:
        (paper / "supp.tex").write_text("\\input{sec/S_extra}\n")
        (paper / "sec" / "S_extra.tex").write_text(supp)
    return paper


# --- parsing printed numbers ------------------------------------------------

def _vals(text):
    return [(n.value, n.decimals, n.percent) for n in ac.parse_numbers(text)]


def test_parse_handles_math_signs_thousands_and_percent():
    got = _vals("reads $\\mathbf{+0.38}$ vs $-220.83$; $1{,}441$ and 1,441 "
                "builds; 28.7\\% leak; $33\\%$; $1.34\\times$")
    assert (0.38, 2, False) in got
    assert (220.83, 2, False) in got
    assert got.count((1441.0, 0, False)) == 2
    assert (0.287, 3, True) in got
    assert (0.33, 2, True) in got
    assert (1.34, 2, False) in got


def test_parse_ignores_comments_notes_refs_lengths_and_names():
    text = ("% stale 0.38 in a comment\n"
            "\\com{is 0.77 right?} kept 0.51\n"
            "see Table~\\ref{tab:5} and \\cite{a2019}\n"
            "\\includegraphics[width=0.48\\linewidth]{fig2_dose}\n"
            "\\vspace{-2pt} ResNet-50 on 3D builds with DINOv2\n"
            "\\begin{tabular}{rlr} \\multicolumn{6}{l}{x 0.25}\n")
    assert _vals(text) == [(0.51, 2, False), (0.25, 2, False)]


def test_parse_reports_line_numbers():
    nums = ac.parse_numbers("a\nb $0.38$\n% 9.99\nc 12.5\n", "sec/x.tex")
    assert [(n.where, n.raw) for n in nums] == [("sec/x.tex:2", "0.38"),
                                                ("sec/x.tex:4", "12.5")]


def test_matching_respects_printed_precision():
    n = lambda t: ac.parse_numbers(t)[0]                       # noqa: E731
    assert n("0.14").matches(0.139)
    assert not n("0.140").matches(0.139)
    assert n("530").matches(531.05)        # rounded to tens
    assert not n("530").matches(1441.4)
    assert n("91\\%").matches(0.905)
    assert n("0.069").matches(-0.069)      # sign is not compared
    assert n("200").matches(202.0) and not n("200").matches(230.0)


# --- the cross-check --------------------------------------------------------

def _claim(name, value, run_dependent=True):
    c = ac.Claim(name, value, lambda: value)
    c.run_dependent = run_dependent
    return c


def test_stale_found_near_and_unregistered(tmp_path):
    paper = _paper(tmp_path,
                   results="appearance $+0.38$ and geometry $201.80$; "
                           "shuffle $+220.83$; extra 57 builds of 128\n",
                   appendix="again $0.14$\n")
    claims = [_claim("appearance", 0.14), _claim("geometry", 201.8),
              _claim("shuffle", 231.52, run_dependent=False)]
    got = {"appearance": 0.14, "geometry": 201.8, "shuffle": 231.52}
    alt = {"old": {"appearance": 0.38, "geometry": 201.8}}
    rep = ac.tex_crosscheck(claims, got, paper, alt=alt)
    assert [(s["where"], s["printed"], s["alt_run"]) for s in rep.stale] == \
        [("sec/6_results.tex:1", "0.38", "old")]
    assert rep.found["appearance"] == ["sec/X_appendix.tex:1"]
    assert rep.found["geometry"] == ["sec/6_results.tex:1"]
    assert [(n["printed"], n["name"], n["claim_printed_elsewhere"])
            for n in rep.near] == [("220.83", "shuffle", False)]
    assert rep.missing == ["shuffle"]
    # 57 matches nothing; 128 is a protocol constant, not a claim
    assert [u["printed"] for u in rep.unregistered] == ["57"]


def test_registered_value_of_a_run_dependent_claim_is_not_proof(tmp_path):
    """Under a different run of record the registry's value IS the stale one."""
    paper = _paper(tmp_path, results="appearance $+0.14$\n")
    claims = [_claim("appearance", 0.14)]
    rep = ac.tex_crosscheck(claims, {"appearance": 0.38}, paper,
                            alt={"0925": {"appearance": 0.14}})
    assert [s["printed"] for s in rep.stale] == ["0.14"]
    assert "appearance" not in rep.found


def test_unregistered_leaderboard_cells_are_caught_as_stale(tmp_path):
    paper = _paper(tmp_path, results="& $+63.98$ & $+63.98$ \\\\\n")
    rep = ac.tex_crosscheck([], {}, paper,
                            alt={"old": {"leaderboard x/appearance": 63.98}},
                            cells={"leaderboard x/appearance": 69.76})
    assert len(rep.stale) == 1 and not rep.stale[0]["registered"]


def test_coarse_two_digit_hits_are_not_called_stale(tmp_path):
    paper = _paper(tmp_path, results="CI [0.7, 1.3]\n")
    claims = [_claim("ratio", 1.34)]
    rep = ac.tex_crosscheck(claims, {"ratio": 1.22}, paper,
                            alt={"old": {"ratio": 1.34}})
    assert rep.stale == []


def test_paper_files_order_appendix_and_supplement(tmp_path):
    paper = _paper(tmp_path, supp="x")
    files = [(p.name, a) for p, a in ac.paper_files(paper)]
    assert files == [("6_results.tex", False), ("X_appendix.tex", True),
                     ("S_extra.tex", True)]
    assert [p.name for p, _ in ac.paper_files(paper, appendix=False)] == \
        ["6_results.tex"]


def test_default_paper_prefers_cvpr27(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "PAPERS", tmp_path)
    assert ac.default_paper() == tmp_path / "iclr"
    (tmp_path / "cvpr27").mkdir()
    assert ac.default_paper() == tmp_path / "iclr"     # no main.tex yet
    (tmp_path / "cvpr27" / "main.tex").write_text("")
    assert ac.default_paper() == tmp_path / "cvpr27"


def test_run_of_record_is_one_of_the_registered_runs():
    assert ac.RUN_OF_RECORD in ac.RUNS and ac.LEADERBOARD_RUN in ac.RUNS
    assert ac.BENCH == ac.RUNS[ac.RUN_OF_RECORD]
    assert ac.LB_BENCH == ac.RUNS[ac.LEADERBOARD_RUN]
    assert pr.RUN == ac.RUN_OF_RECORD and pr.LB_RUN == ac.LEADERBOARD_RUN
    with pytest.raises(SystemExit):
        ac.resolve_run("no_such_run")


@pytest.mark.skipif(not all((p / "scorecard.md").exists() for p in ac.RUNS.values()),
                    reason="bench runs not on disk")
def test_switching_the_run_moves_only_run_dependent_claims():
    """Leaderboard claims follow the leaderboard run, record claims the record run."""
    a = {c.name: c for c in ac.build_claims(ac.RUNS["0925"], ac.RUNS["1103_n256"])}
    b = {c.name: c for c in ac.build_claims(ac.RUNS["0925"], ac.RUNS["0917"])}
    lb = "leaderboard real@solidify/appearance"
    head = "solidify separation, geometry metric"
    assert a[lb].role == "leaderboard" and a[head].role == "record"
    assert not a["canonical test split size"].run_dependent
    assert a[lb].compute() == pytest.approx(0.35)
    assert b[lb].compute() == pytest.approx(0.38)
    assert a[head].compute() == pytest.approx(b[head].compute())
    assert not a["abstract bound: solidify separation exceeds 200 floors"].printed


def test_resolve_paper_accepts_a_short_name():
    assert ac.resolve_paper(Path("cvpr27")) == (ac.PAPERS / "cvpr27").resolve()


# --- readiness --------------------------------------------------------------

def test_main_text_end_from_pdf_text():
    body = "This is a line of real main text here.\n"
    pages = [body * 3, body * 3,
             "Under review as a conference paper at ICLR 2027\n540\n"
             "      U SE OF G ENERATIVE AI\n" + body,
             "R EFERENCES\n"]
    assert pr.main_text_pages(pages) == (2, "Use of Generative AI")
    pages = [body * 3, body + "References\n" + "[1] A. Author.\n"]
    assert pr.main_text_pages(pages) == (2, "References")
    assert pr.main_text_pages([body]) is None


def test_notes_check_blocks_live_notes_only(tmp_path, monkeypatch):
    paper = _paper(tmp_path,
                   results="fine \\com{fix this}\n% \\com{old note}\n",
                   appendix="\\TODO{url}\n")
    (paper / "main.tex").write_text(
        "\\newcommand{\\com}[1]{#1}\n" + (paper / "main.tex").read_text())
    monkeypatch.setattr(pr, "PAPER", paper)
    monkeypatch.setattr(pr, "SEC", paper / "sec")
    c = pr.check_notes()
    assert not c.ok and c.blocking
    assert c.detail == "2 live · 1 more inside % comments"


def test_sample_size_reads_the_leaderboard_run(tmp_path, monkeypatch):
    run = tmp_path / "run_x"
    run.mkdir()
    (run / "scorecard.json").write_text(json.dumps(
        {"arms": {"a": {"meta": {"n": 300}}, "b": {"meta": {"n": 256}}}}))
    monkeypatch.setattr(pr, "LB_RUN", str(run))
    c = pr.check_sample_size()
    assert c.ok and "min n=256" in c.detail


def test_venue_detection(tmp_path, monkeypatch):
    paper = tmp_path / "cvpr27"
    paper.mkdir()
    monkeypatch.setattr(pr, "PAPER", paper)
    assert pr.venue() == "cvpr"
    monkeypatch.setattr(pr, "PAPER", tmp_path / "iclr")
    assert pr.venue() == "iclr"
