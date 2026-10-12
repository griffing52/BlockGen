"""Is the paper submittable yet? One command, one honest answer.

Checks the things that are cheap to get wrong and expensive to discover late:
unmeasured claims still marked in the LaTeX, figures that do not exist, runs
below the protocol's own minimum sample size, citations never verified against
the record, and human data that has not been collected.

    .venv/bin/python scripts/paper_readiness.py
    .venv/bin/python scripts/paper_readiness.py --verbose

Exit code is 0 only when every BLOCKING check passes, so this can gate a
release commit. Advisory checks are reported but do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "deliverables" / "paper" / "iclr"
SEC = PAPER / "sec"
STUDY = ROOT / "outputs" / "run_20260918_human_v2"

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")


@dataclass
class Check:
    name: str
    ok: bool
    blocking: bool
    detail: str = ""
    lines: List[str] = field(default_factory=list)


def _tex_files() -> List[Path]:
    return sorted(SEC.glob("*.tex")) + [PAPER / "main.tex"]


def check_unmeasured() -> Check:
    """No \\TODO or \\UNMEASURED may survive into a submission."""
    hits: List[str] = []
    for f in _tex_files():
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\\(TODO|UNMEASURED)\{", line):
                # the macro definitions in main.tex are not claims
                if f.name == "main.tex" and "newcommand" in line:
                    continue
                kind = "UNMEASURED" if "UNMEASURED" in line else "TODO"
                hits.append(f"{f.relative_to(ROOT)}:{i}  [{kind}]")
    return Check("no unmeasured claims in the LaTeX", not hits, True,
                 f"{len(hits)} outstanding" if hits else "clean", hits)


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


def check_build() -> Check:
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        return Check("paper builds", False, True, "main.pdf not built")
    log = PAPER / "main.log"
    issues: List[str] = []
    if log.exists():
        text = log.read_text(errors="ignore")
        for pat, what in ((r"Warning: Citation .* undefined", "undefined citation"),
                          (r"Warning: Reference .* undefined", "undefined reference")):
            n = len(re.findall(pat, text))
            if n:
                issues.append(f"{n} {what}(s)")
        pages = re.search(r"Output written on .*? \((\d+) pages", text)
        if pages:
            issues.append(f"{DIM}{pages.group(1)} pages total{RESET}")
    return Check("paper builds without undefined refs",
                 not any("undefined" in i for i in issues), True,
                 "; ".join(issues) if issues else "clean")


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


def check_human_data() -> Check:
    if not (STUDY / "study.html").exists():
        return Check("human study collected", False, True,
                     "study not generated; see RUNBOOK.md §1")
    responses = sorted(STUDY.glob("rater*.json"))
    n = len(responses)
    return Check("human study collected", n >= 8, True,
                 f"{n} rater file(s); target >= 8"
                 + ("" if n else " — see RUNBOOK.md §1"))


def check_sample_size(min_n: int = 256) -> Check:
    """The protocol asks for n >= 256; report what the newest run actually used."""
    cards = sorted(ROOT.glob("outputs/*/scorecard.json"))
    if not cards:
        return Check(f"headline arms at n >= {min_n}", False, False,
                     "no scorecards found")
    newest = max(cards, key=lambda p: p.parent.name)
    try:
        arms = json.loads(newest.read_text())["arms"]
        arms = arms if isinstance(arms, dict) else {a["name"]: a for a in arms}
        ns = {a: v["meta"]["n"] for a, v in arms.items()}
    except Exception as exc:                       # noqa: BLE001
        return Check(f"headline arms at n >= {min_n}", False, False, str(exc))
    worst = min(ns.values()) if ns else 0
    return Check(f"headline arms at n >= {min_n}", worst >= min_n, False,
                 f"newest run {newest.parent.name} has n={worst}")


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


def check_claims() -> Check:
    """Every registered numeric claim must still match its data.

    Separate from the other checks because it is the one that catches a number
    going stale: a re-scored run, an edited table, a hand-typed figure. The
    build cannot see any of those.
    """
    try:
        r = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"),
             str(ROOT / "scripts" / "audit_claims.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
    except Exception as exc:                           # noqa: BLE001
        return Check("registered claims match the data", False, True, str(exc))
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
    msg = re.sub(r"\033\[[0-9;]*m", "", tail[-1]) if tail else "no output"
    return Check("registered claims match the data", r.returncode == 0, True,
                 msg.strip())


def check_page_budget(limit: int = 9) -> Check:
    """Main text must fit the venue limit; references and appendix do not count.

    `\\label{endofmain}` sits at the end of the Conclusion, so the page it
    resolves to in main.aux is the last page of main text.
    """
    aux = PAPER / "main.aux"
    if not aux.exists():
        return Check(f"main text <= {limit} pages", False, True, "no main.aux")
    m = re.search(r"\\newlabel\{endofmain\}\{\{[^}]*\}\{(\d+)\}", aux.read_text())
    if not m:
        return Check(f"main text <= {limit} pages", False, False,
                     "endofmain label missing")
    pages = int(m.group(1))
    return Check(f"main text <= {limit} pages", pages <= limit, True,
                 f"{pages} pages"
                 + (f" — {pages - limit} over, see README trimming plan"
                    if pages > limit else ""))


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
    required_phrases = {
        "What is not a control": "the decimation-policy paragraph",
        "Use of Generative AI": "the mandatory ICLR AI use statement",
        "Reproducibility Statement": "the main-text reproducibility statement",
    }
    body = "\n".join(f.read_text() for f in _tex_files() if f.exists())
    missing = [f"{k} — {why}" for k, why in required_labels.items()
               if f"label{{{k}}}" not in body]
    missing += [f"'{k}' — {why}" for k, why in required_phrases.items()
                if k not in body]
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true",
                    help="list every offending line")
    args = ap.parse_args()

    checks = [check_build(), check_structure(), check_figures(),
              check_instrument_tests(),
              check_unmeasured(), check_citations(), check_claims(),
              check_page_budget(),
              check_human_data(), check_sample_size(),
              check_release_decisions()]

    print(f"\n  paper readiness — {PAPER.relative_to(ROOT)}\n")
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

    print()
    if blocking_failed:
        print(f"  {RED}NOT SUBMITTABLE{RESET} — {blocking_failed} blocking "
              f"check(s) failing. See deliverables/paper/iclr/RUNBOOK.md\n")
    else:
        print(f"  {GREEN}All blocking checks pass.{RESET} Review the TODO rows, "
              f"then set \\draftfalse in main.tex and rebuild.\n")
    return 1 if blocking_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
