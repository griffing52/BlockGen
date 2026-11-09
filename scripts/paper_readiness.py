"""Is the paper submittable yet? One command, one honest answer.

Checks the things that are cheap to get wrong and expensive to discover late:
a build with undefined references (read from a log this run wrote), author
notes and unmeasured claims still in the LaTeX, figures that do not exist,
numbers that disagree with the run of record, the venue's page budget, runs
below the protocol's own minimum sample size, and citations never verified
against the record.

    .venv/bin/python scripts/paper_readiness.py               # rebuilds in a temp copy
    .venv/bin/python scripts/paper_readiness.py --verbose
    .venv/bin/python scripts/paper_readiness.py --paper deliverables/paper/iclr --no-build
    .venv/bin/python scripts/paper_readiness.py --run 0917    # the other run of record

The paper defaults to deliverables/paper/cvpr27 when it exists, else iclr; the
run of record is audit_claims.RUN_OF_RECORD. Exit code is 0 only when every
BLOCKING check passes, so this can gate a release commit. Advisory checks are
reported but do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_claims  # noqa: E402  -- RUNS / RUN_OF_RECORD / default_paper live there

PAPER = audit_claims.default_paper()
SEC = PAPER / "sec"
#: The leaderboard run the paper cites. Never set here: audit_claims.RUN_OF_RECORD
#: (or `--run`) is the one place that choice is made.
RUN = audit_claims.RUN_OF_RECORD
#: The leaderboard run (Table tab:leaderboard). audit_claims.LEADERBOARD_RUN or
#: `--leaderboard-run`; the sample-size check reads this one.
LB_RUN = audit_claims.LEADERBOARD_RUN

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")

#: Headings after which the PDF stops counting toward the page limit. Matched
#: with whitespace removed and upper-cased, because small-caps headings come
#: out of pdftotext as "U SE OF G ENERATIVE AI".
END_OF_MAIN = {h.replace(" ", "").upper(): h for h in (
    "References", "Bibliography", "LLM Usage Statement", "LLM Usage",
    "Use of Generative AI", "Use of Large Language Models",
    "Reproducibility Statement", "Ethics Statement")}
#: Running headers that sit above a heading without being main text.
_RUNNING_HEADER = re.compile(r"Under review as|CONFIDENTIAL REVIEW COPY|"
                             r"DO NOT DISTRIBUTE|Submission #", re.I)


def set_paper(paper: Path) -> None:
    global PAPER, SEC
    PAPER, SEC = paper, paper / "sec"


def venue() -> str:
    """'cvpr' or 'iclr', from the paper directory and its style file."""
    main = PAPER / "main.tex"
    text = main.read_text(errors="ignore") if main.exists() else ""
    return "cvpr" if ("cvpr" in PAPER.name.lower() or "{cvpr}" in text) else "iclr"


@dataclass
class Check:
    name: str
    ok: bool
    blocking: bool
    detail: str = ""
    lines: List[str] = field(default_factory=list)


@dataclass
class Build:
    """The outcome of compiling the paper: where the fresh log/aux/pdf are."""
    ok: bool
    dir: Path
    fresh: bool
    detail: str = ""
    errors: List[str] = field(default_factory=list)


def build_paper(no_build: bool = False, keep: Optional[Path] = None) -> Build:
    """Compile the paper in a temporary copy and return the fresh artifacts.

    Building in a copy means a readiness run never leaves aux files behind in
    a directory someone else may be editing, and never reads a main.log that
    predates the last edit -- which is how a stale "no undefined refs" PASS
    happened. `--no-build` reads the paper directory's own artifacts instead,
    and says so.
    """
    if no_build:
        return Build((PAPER / "main.pdf").exists(), PAPER, False,
                     "--no-build: reading existing main.log/main.pdf")
    if shutil.which("latexmk") is None:
        return Build(False, PAPER, False, "latexmk not on PATH")
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="paper_build_"))
    dst = tmp / PAPER.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PAPER, dst, symlinks=True, ignore=shutil.ignore_patterns(
        ".backup*", "main.pdf", "main.log", "main.aux", "main.fdb_latexmk",
        "main.fls", "main.bbl", "main.blg", "main.out", "*.synctex.gz"))
    try:
        r = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error",
             "-f", "main.tex"], cwd=dst, capture_output=True, text=True,
            timeout=900, env={**os.environ, "max_print_line": "1000"})
    except subprocess.TimeoutExpired:
        return Build(False, dst, True, "latexmk timed out after 900 s")
    log = dst / "main.log"
    errs = []
    if log.exists():
        errs = [l for l in log.read_text(errors="ignore").splitlines()
                if re.match(r"^(\S+\.tex:\d+:|! )", l)]
    ok = (dst / "main.pdf").exists() and r.returncode == 0 and not errs
    return Build(ok, dst, True,
                 f"fresh build in {dst}" + ("" if ok else
                                            f" (latexmk exit {r.returncode})"),
                 errs)


def _tex_files() -> List[Path]:
    """main.tex, its preamble, and every section it inputs (appendix included)."""
    files = [PAPER / "main.tex"]
    files += [p for p in (PAPER / "preamble.tex", PAPER / "math_commands.tex")
              if p.exists()]
    files += [p for p, _ in audit_claims.paper_files(PAPER, appendix=True)]
    if len(files) <= 1:
        files += sorted(SEC.glob("*.tex"))
    return list(dict.fromkeys(files))


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _scan(pattern: str, *, skip_definitions: bool = True):
    """(live, commented) hits of `pattern` across the paper's TeX."""
    live: List[str] = []
    commented: List[str] = []
    rx = re.compile(pattern)
    for f in _tex_files():
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if skip_definitions and re.search(r"\\(re|provide)?newcommand", line):
                continue                     # the macro definitions are not notes
            pct = re.search(r"(?<!\\)%", line)
            code_end = pct.start() if pct else len(line)
            for m in rx.finditer(line):
                where = f"{_rel(f)}:{i}  {line.strip()[:90]}"
                (live if m.start() < code_end else commented).append(where)
    return live, commented


def check_unmeasured() -> Check:
    """No \\UNMEASURED may survive into a submission."""
    live, _ = _scan(r"\\UNMEASURED\b")
    return Check("no unmeasured claims in the LaTeX", not live, True,
                 f"{len(live)} outstanding" if live else "clean", live)


def check_notes() -> Check:
    """No author note (\\com{...}) or \\TODO may survive into a submission.

    Live notes block: with \\draftfalse each one is a build error, and with
    \\drafttrue each one prints in blue or red on the page. Notes inside a
    TeX comment do not print; they are counted so they are cleaned up too,
    but do not block.
    """
    live, commented = _scan(r"\\(?:com\s*\{|TODO\b)")
    detail = f"{len(live)} live" if live else "clean"
    if commented:
        detail += f" · {len(commented)} more inside % comments"
    return Check("no leftover \\com{} notes or \\TODO", not live, True, detail,
                 live + [f"(commented) {c}" for c in commented])


def check_figures() -> Check:
    want = ["fig1_teaser", "fig2_dose_response", "fig3_pillars",
            "fig4_casestudy"]
    missing = [w for w in want if not (PAPER / "figures" / f"{w}.pdf").exists()]
    # every \includegraphics must resolve
    broken: List[str] = []
    for f in _tex_files():
        if not f.exists():
            continue
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                             f.read_text()):
            ref = m.group(1)
            if not any((PAPER / f"{ref}{e}").exists() for e in (".pdf", ".png", "")):
                broken.append(f"{f.name}: {ref}")
    # The check that was missing: a figure can be BUILT and silently dropped from
    # the paper by an edit. That is content loss, and the reference check above
    # cannot see it -- it only verifies that what IS referenced exists.
    referenced = set()
    for f in _tex_files():
        if not f.exists():
            continue
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                             f.read_text()):
            referenced.add(Path(m.group(1)).stem)
    built = {p.stem for p in (PAPER / "figures").glob("*.pdf")}
    orphaned = sorted(built - referenced - {"cutaway_check"})

    bad = [f"missing: {m}" for m in missing] + [f"unresolved: {b}" for b in broken]
    detail = "all present" if not bad else f"{len(bad)} problems"
    if orphaned:
        detail += f" · {len(orphaned)} built but unreferenced"
        bad += [f"orphaned (built, never included): {o}" for o in orphaned]
    return Check("figures built, referenced, and none orphaned", not bad, False,
                 detail, bad)


def check_build(build: Build) -> Check:
    """Undefined references and citations, read from a log this run wrote."""
    if not build.fresh and not (build.dir / "main.pdf").exists():
        return Check("paper builds", False, True, "main.pdf not built")
    if build.fresh and not (build.dir / "main.pdf").exists():
        return Check("paper builds", False, True, build.detail, build.errors[:40])
    log = build.dir / "main.log"
    issues: List[str] = []
    lines: List[str] = []
    if not log.exists():
        return Check("paper builds without undefined refs", False, True,
                     f"no main.log in {build.dir}")
    text = log.read_text(errors="ignore")
    for pat, what in ((r"Warning: Citation .* undefined", "undefined citation"),
                      (r"Warning: Reference .* undefined", "undefined reference")):
        found = re.findall(pat + r".*", text)
        if found:
            issues.append(f"{len(found)} {what}(s)")
            lines += found
    if build.errors:
        issues.append(f"{len(build.errors)} LaTeX error(s)")
        lines += build.errors
    pages = re.search(r"Output written on .*? \((\d+) pages", text)
    if pages:
        issues.append(f"{pages.group(1)} pages total")
    if not build.fresh:
        import time
        age = (time.time() - log.stat().st_mtime) / 3600
        issues.append(f"log not rebuilt (--no-build), {age:.1f} h old")
    ok = build.ok and not any("undefined" in i or "error" in i for i in issues)
    return Check("paper builds without undefined refs", ok, True,
                 "; ".join(issues) if issues else "clean", lines)


def check_citations() -> Check:
    """Entries still flagged NEEDS-VERIFY have never been checked."""
    bib = PAPER / "references.bib"
    if not bib.exists():
        return Check("citations verified", False, True, "references.bib missing")
    text = bib.read_text()
    unverified = re.findall(r"@\w+\{([^,]+),(?:(?!\n@).)*?NEEDS-VERIFY",
                            text, re.S)
    total = len(re.findall(r"^@\w+\{", text, re.M))
    return Check("every citation verified against the record",
                 not unverified, True,
                 f"{len(unverified)} of {total} unverified",
                 [f"{k.strip()}" for k in unverified])


def check_sample_size(min_n: int = 256) -> Check:
    """The protocol asks for n >= 256; report what the paper's leaderboard run used.

    Reads the leaderboard run (audit_claims.LEADERBOARD_RUN, or
    `--leaderboard-run`), not the protocol run of record and not the
    newest run in outputs/: the newest run is whatever was launched last (a
    facade experiment, say), which says nothing about the table in the paper.
    """
    try:
        bench = audit_claims.resolve_run(LB_RUN)
    except SystemExit as exc:
        return Check(f"leaderboard arms at n >= {min_n}", False, False, str(exc))
    card = bench / "scorecard.json"
    if not card.exists():
        return Check(f"leaderboard arms at n >= {min_n}", False, False,
                     f"no scorecard in {bench.name}")
    try:
        arms = json.loads(card.read_text())["arms"]
        arms = arms if isinstance(arms, dict) else {a["name"]: a for a in arms}
        ns = {a: v["meta"]["n"] for a, v in arms.items()}
    except Exception as exc:                       # noqa: BLE001
        return Check(f"leaderboard arms at n >= {min_n}", False, False, str(exc))
    worst = min(ns.values()) if ns else 0
    small = sorted(a for a, n in ns.items() if n < min_n)
    return Check(f"leaderboard arms at n >= {min_n}", worst >= min_n, False,
                 f"leaderboard run {LB_RUN} ({bench.name}): min n={worst}, "
                 f"{len(small)} of {len(ns)} arms below",
                 [f"{a}: n={ns[a]}" for a in small])


def check_instrument_tests() -> Check:
    try:
        r = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest",
             "tests/test_human_study.py", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        ok = r.returncode == 0
        tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
    except Exception as exc:                       # noqa: BLE001
        return Check("2AFC instrument tests pass", False, True, str(exc))
    return Check("2AFC instrument tests pass", ok, True, tail)


_AUDIT: dict = {}


def run_audit() -> dict:
    """audit_claims.py once per readiness run, under the same paper and run."""
    if _AUDIT:
        return _AUDIT
    out = Path(tempfile.mkstemp(prefix="audit_", suffix=".json")[1])
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "audit_claims.py"),
             "--paper", str(PAPER), "--run", RUN, "--leaderboard-run", LB_RUN,
             "--json", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        _AUDIT.update(json.loads(out.read_text()) if out.stat().st_size else
                      {"error": (r.stderr or r.stdout).strip()[-300:]})
    except Exception as exc:                       # noqa: BLE001
        _AUDIT["error"] = str(exc)
    finally:
        out.unlink(missing_ok=True)
    return _AUDIT


def check_claims() -> Check:
    """Every registered numeric claim must still match its data.

    Separate from the other checks because it is the one that catches a number
    going stale: a re-scored run, an edited table, a hand-typed figure. The
    build cannot see any of those.
    """
    a = run_audit()
    if "error" in a:
        return Check("registered claims match the data", False, True, a["error"])
    bad = [c for c in a["claims"] if not c["ok"] and not c["needs_corpus"]]
    return Check("registered claims match the data", not bad, True,
                 f"{len(bad)} of {len(a['claims'])} disagree with run {a['run']}"
                 if bad else f"all {len(a['claims'])} match run {a['run']}",
                 [f"{c['name']}: registered {c['paper_value']:g}, computed "
                  + ("error" if c["computed"] is None else f"{c['computed']:g}")
                  for c in bad])


def check_tex_numbers() -> Check:
    """The numbers the TeX prints are the run of record's, not another run's.

    The registry check above cannot see this: it compares the registry with
    the data and never opens the paper, so a table transcribed from a
    superseded run passed it for weeks.
    """
    a = run_audit()
    if "error" in a:
        return Check("TeX prints the run of record's numbers", False, True,
                     a["error"])
    t = a["tex"]
    strong = [n for n in t["near"] if not n.get("claim_printed_elsewhere")]
    detail = (f"{len(t['stale'])} stale (another run's value), "
              f"{len(strong)} near-miss, {len(t['missing'])} claim(s) not in "
              f"the TeX, {len(t['unregistered'])} unregistered headline numbers")
    lines = [f"STALE {s['where']}  {s['printed']} — {s['name']}: "
             + ("absent" if s["record"] is None else f"{s['record']:g}")
             + f" under {a['run']}, {s['alt']:g} under {s['alt_run']}"
             for s in t["stale"]]
    lines += [f"NEAR  {n['where']}  {n['printed']} — {n['name']} = {n['value']:g}"
              for n in strong]
    return Check(f"TeX prints the run of record's numbers ({a['run']})",
                 not t["stale"], True, detail, lines)


def _pdf_pages(pdf: Path) -> List[str]:
    """Page texts in reading order. Not `-layout`: in a two-column page that
    mode glues a right-column heading onto a left-column line."""
    r = subprocess.run(["pdftotext", str(pdf), "-"],
                       capture_output=True, text=True, timeout=120)
    pages = r.stdout.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def main_text_pages(pages: List[str]) -> Optional[tuple]:
    """(last page of main text, heading that ends it), 1-based, or None.

    Main text ends at the first bibliography or statement heading. If that
    heading is the first real text on its page, the previous page is the last
    one of main text; otherwise its own page still carries main text.
    """
    for p, page in enumerate(pages, 1):
        substantive = False
        for line in page.splitlines():
            norm = re.sub(r"^[\d.]+", "", re.sub(r"\s+", "", line).upper())
            if norm in END_OF_MAIN:
                return (p if substantive else p - 1), END_OF_MAIN[norm]
            stripped = re.sub(r"^\s*\d+\s*", "", line).strip()
            if (len(stripped.split()) >= 4
                    and not _RUNNING_HEADER.search(stripped)):
                substantive = True
    return None


def check_page_budget(build: Build, limit: Optional[int] = None) -> Check:
    """Main text must fit the venue limit; references and statements do not count.

    CVPR: 8 pages, excluding references and the LLM Usage Statement. ICLR: 9,
    excluding references and the ethics / reproducibility / AI-use statements.
    The end of main text is read from the freshly built PDF, as the page that
    holds the first such heading; the `endofmain` label, where present, is
    reported alongside as a cross-check.
    """
    limit = limit or (8 if venue() == "cvpr" else 9)
    name = f"main text <= {limit} pages ({venue().upper()})"
    pdf = build.dir / "main.pdf"
    if not pdf.exists():
        return Check(name, False, True, "no main.pdf")
    if shutil.which("pdftotext") is None:
        return Check(name, False, True, "pdftotext not on PATH")
    pages = _pdf_pages(pdf)
    end = main_text_pages(pages)
    if end is None:
        return Check(name, False, True,
                     f"no bibliography or statement heading in {len(pages)} pages")
    used, heading = end
    detail = f"{used} pages (main text ends before '{heading}')"
    aux = build.dir / "main.aux"
    if aux.exists():
        m = re.search(r"\\newlabel\{endofmain\}\{\{[^}]*\}\{(\d+)\}",
                      aux.read_text(errors="ignore"))
        if m:
            detail += f"; endofmain label on p.{m.group(1)}"
    if used > limit:
        detail += f" — {used - limit} over, see README trimming plan"
    return Check(name, used <= limit, True, detail)


def check_structure() -> Check:
    """Load-bearing sections and tables must exist, in the main text.

    Added after an over-greedy regex silently swallowed the arms table, the
    metric definitions and the decimation-policy paragraph while moving a
    different table to the appendix. The build stayed clean and the page count
    went down, so nothing flagged it. Content loss during restructuring is
    exactly the failure a page-count check cannot see.
    """
    required_labels = {
        "tab:arms": "the table defining every arm name",
        "sec:metrics": "the MV-KID / V-KID definitions",
        "fig:teaser": "Figure 1",
        "app:repro": "the reproducibility appendix",
    }
    #: phrase alternatives -> why; any one alternative satisfies the row
    required_phrases = {
        ("What is not a control",): "the decimation-policy paragraph",
        ("Use of Generative AI", "LLM Usage"): "the mandatory AI use statement",
    }
    if venue() == "iclr":
        required_phrases[("Reproducibility Statement",)] = \
            "the main-text reproducibility statement"
    body = "\n".join(f.read_text() for f in _tex_files() if f.exists())
    missing = [f"{k} — {why}" for k, why in required_labels.items()
               if f"label{{{k}}}" not in body]
    missing += [f"'{' / '.join(k)}' — {why}" for k, why in required_phrases.items()
                if not any(alt in body for alt in k)]
    return Check("load-bearing sections present", not missing, True,
                 "all present" if not missing else f"{len(missing)} missing",
                 missing)


def check_release_decisions() -> Check:
    """Things only a human can settle, tracked so they are not forgotten."""
    repro = (SEC / "X_appendix.tex")
    text = repro.read_text() if repro.exists() else ""
    todo = "TODO" in text and "anonymized code URL" in text
    return Check("code/data release + licensing settled", not todo, False,
                 "still a TODO in the reproducibility appendix" if todo
                 else "settled")


def main() -> int:
    global RUN, LB_RUN
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true",
                    help="list every offending line")
    ap.add_argument("--paper", type=Path, default=None,
                    help="paper directory (default: deliverables/paper/cvpr27 "
                         "if it exists, else deliverables/paper/iclr)")
    ap.add_argument("--run", default=audit_claims.RUN_OF_RECORD,
                    help="protocol run of record (default "
                         f"audit_claims.RUN_OF_RECORD = {audit_claims.RUN_OF_RECORD})")
    ap.add_argument("--leaderboard-run", default=audit_claims.LEADERBOARD_RUN,
                    help="leaderboard run (default audit_claims.LEADERBOARD_RUN = "
                         f"{audit_claims.LEADERBOARD_RUN})")
    ap.add_argument("--no-build", action="store_true",
                    help="skip latexmk and read the paper directory's existing "
                         "main.log / main.pdf (may be stale)")
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="build here instead of a fresh temp dir (kept afterwards)")
    ap.add_argument("--pages", type=int, default=None,
                    help="override the venue page limit")
    args = ap.parse_args()
    if args.paper is not None:
        set_paper(audit_claims.resolve_paper(args.paper))
    RUN, LB_RUN = args.run, args.leaderboard_run

    build = build_paper(args.no_build, args.build_dir)
    # human study removed 2026-11-02 (cut from the paper; no IRB) -- do not re-add
    checks = [check_build(build), check_structure(), check_figures(),
              check_instrument_tests(),
              check_unmeasured(), check_notes(), check_citations(),
              check_claims(), check_tex_numbers(),
              check_page_budget(build, args.pages),
              check_sample_size(),
              check_release_decisions()]

    print(f"\n  paper readiness — {_rel(PAPER)} · run of record {RUN}"
          f"\n  {DIM}{build.detail}{RESET}\n")
    blocking_failed = 0
    for c in checks:
        if c.ok:
            mark, colour = "PASS", GREEN
        elif c.blocking:
            mark, colour = "BLOCK", RED
            blocking_failed += 1
        else:
            mark, colour = "TODO ", YELLOW
        print(f"  {colour}{mark}{RESET}  {c.name}"
              + (f"  {DIM}— {c.detail}{RESET}" if c.detail else ""))
        if args.verbose and c.lines:
            for line in c.lines[:40]:
                print(f"         {DIM}{line}{RESET}")
            if len(c.lines) > 40:
                print(f"         {DIM}... and {len(c.lines)-40} more{RESET}")

    if build.fresh and args.build_dir is None:
        shutil.rmtree(build.dir.parent, ignore_errors=True)
    print()
    if blocking_failed:
        runbook = PAPER / "RUNBOOK.md"
        print(f"  {RED}NOT SUBMITTABLE{RESET} — {blocking_failed} blocking "
              f"check(s) failing."
              + (f" See {_rel(runbook)}" if runbook.exists() else "") + "\n")
    else:
        print(f"  {GREEN}All blocking checks pass.{RESET} Review the TODO rows, "
              f"then set \\draftfalse in main.tex and rebuild.\n")
    return 1 if blocking_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
