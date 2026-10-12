"""Build the anonymized supplementary-code release from this repository.

The release is a whitelisted, scrubbed copy: the evaluation suite, its probes and
procedural baselines, the pinned corpus splits, the human-study instrument, the
figure and audit scripts, and the recorded runs every number in the paper is
read from. Nothing else. No git history, no notes, no corpus content, no
textures, no model weights.

    .venv/bin/python deliverables/release/make_release.py --out ../blockgen-release

The file list is *derived*, not maintained: it starts from the entry points a
reader would run and follows `blockgen.*` imports transitively, so a module the
suite needs cannot be forgotten and a module it does not need cannot sneak in.
The recorded runs and data files are listed explicitly because they are the
paper's evidence and should be read as a list.

Everything the script writes into the release is checked afterwards for
identifying strings; the build fails if any survive.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# 1. Code: entry points, and the closure of what they import
# --------------------------------------------------------------------------

ENTRY_POINTS = [
    "scripts/audit_claims.py",          # re-derives every number in the paper
    "scripts/paper_figures.py",         # the paper's figures, from recorded runs
    "scripts/bench_doseresponse.py",    # dose-response validation
    "scripts/ablate_view_protocol.py",  # encoder / view-budget / cutaway ablation
    "scripts/human_study_export.py",    # builds the rater study
    "scripts/human_study_serve.py",     # serves it and collects responses
    "scripts/human_study_analyze.py",   # Bradley-Terry + agreement
    "scripts/bench_hollow.py",          # the converse probe (hollowing) run
    "scripts/vkid_ablation.py",         # descriptor ablation + grow-shell mass-matched control
    "scripts/build_influence.py",       # per-build influence, read by figs/influence_pair.py
    "blockgen/eval/bench/__main__.py",  # the benchmark runner
    "blockgen/eval/bench/full.py",
    "blockgen/eval/bench/lego.py",      # the second domain
    "blockgen/eval/bench/ladder.py",
    "blockgen/eval/bench/splits.py",
    "blockgen/eval/bench/features.py",
    "blockgen/curation/houses.py",      # corpus rebuild
    "blockgen/data/fetch_omr.py",       # LEGO OMR fetch
    "blockgen/data/grabcraft_scraper.py",  # corpus source 2
    "legogen/renderer/ldraw_voxel.py",
    "legogen/renderer/blender_render.py",
    "legogen/renderer/ldraw_grid.py",
] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts" / "figs").glob("*.py")) \
  + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "eval").glob("test_*.py"))

# Entry points that exist in the repo but are not part of the release.
DROP = {
    "scripts/figs/math_max_check.py",   # internal audit-agent checks that read a
    "scripts/figs/math_sqrt_check.py",  # scratch cache; not cited by the paper
    "tests/eval/test_examples.py",            # exercise tools/lab, which is not released
    "tests/eval/test_bench_report_compat.py",
}

# Modules the closure would pull in that the release must not carry.
DROP_MODULES = {
    "blockgen/utils/graph_data.py",     # torch-geometric; reached only lazily
}

_IMPORT = re.compile(
    r"^\s*(?:from\s+((?:blockgen|legogen)[\w.]*)\s+import\s+([\w, ()*]+)"
    r"|import\s+((?:blockgen|legogen)[\w.]*))", re.M)


def _module_path(mod: str) -> Path | None:
    p = ROOT / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = ROOT / mod.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def import_closure(seeds: list[str]) -> set[str]:
    seen: set[Path] = set()
    todo = [ROOT / s for s in seeds]
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        for m in _IMPORT.finditer(f.read_text()):
            base = m.group(1) or m.group(3)
            names = [n.split()[0] for n in (m.group(2) or "")
                     .replace("(", "").replace(")", "").split(",") if n.strip()]
            parts = base.split(".")
            # the package chain (blockgen/__init__.py, blockgen/eval/__init__.py, ...)
            for i in range(1, len(parts) + 1):
                p = _module_path(".".join(parts[:i]))
                if p:
                    todo.append(p)
            for n in names:  # `from pkg import submodule`
                p = _module_path(f"{base}.{n}")
                if p:
                    todo.append(p)
    rel = {str(p.relative_to(ROOT)) for p in seen}
    return {r for r in rel if r not in DROP_MODULES}


# --------------------------------------------------------------------------
# 2. Data and recorded runs: explicit, because they are the evidence
# --------------------------------------------------------------------------

BENCH = "outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs"
LEGO = "outputs/run_20260918_200852_bench_lego_interior_blindness_replication"
DOSE = "outputs/run_20260918_203311_doseresponse_uniform_random-shell_box-canon8"
DOSE_N64 = "outputs/run_20260831_224723_doseresponse"
ABLATION = "outputs/run_20260918_205019_view_protocol_ablation"
HOLLOW = "outputs/run_20260918_203947_bench_hollow"
SIGNIF = "outputs/run_20260919_022424_significance_recompute"
HUMAN = "outputs/run_20260923_human_v5"

EVIDENCE = [
    # the pinned splits and the manifests (source URL, title, dims per build; no content)
    "data/minecraft/splits/houses_32.s0.70-15-15.v1.json",
    "data/minecraft/splits/houses_48.s0.70-15-15.v1.json",
    "data/minecraft/splits/lego_void_20.s0.70-15-15.v1.json",
    "data/minecraft/cache/houses_32_manifest.json",
    "data/minecraft/cache/houses_48_manifest.json",
    "data/minecraft/cache/lego_void_20_manifest.json",
    "data/minecraft/text2mc_index.json",   # per-file shape/occupancy index the curation reads
    # the canonical benchmark run (Table 2, the case study)
    f"{BENCH}/scorecard.json", f"{BENCH}/scorecard.md",
    # the LEGO replication
    f"{LEGO}/scorecard.json", f"{LEGO}/scorecard.md",
    f"{LEGO}/ladder/dinov2b__v4_az45-135-225-315_el30-30-30-30_px224_ortho.json",
    f"{LEGO}/ladder/geom__patch2_orb55_t5_h8.json",
    # dose-response
    f"{DOSE}/doseresponse.json", f"{DOSE}/doseresponse.md",
    f"{DOSE_N64}/doseresponse.json", f"{DOSE_N64}/doseresponse.md",
    # view-protocol ablation (the cutaway positive control)
    f"{ABLATION}/ablation.json", f"{ABLATION}/views.png", f"{ABLATION}/views.txt",
    # the converse probe (hollowing beside solidify)
    f"{HOLLOW}/hollow.json", f"{HOLLOW}/hollow.md",
    # the significance recompute behind the head-to-head table and its appendix
    f"{SIGNIF}/recompute.py", f"{SIGNIF}/recompute_r1.json", f"{SIGNIF}/recompute_r2.json",
    # the validation ladder
    "outputs/analysis/ladder/dinov2b__v4_az45-135-225-315_el30-30-30-30_px224_ortho.json",
    "outputs/analysis/ladder/geom__patch2_orb55_t5_h8.json",
    # small derived checks the audit reads
    "outputs/figstage/contiguous_draw_check.json",
    "outputs/figstage/influence_pair.json",
    "outputs/figstage/interior_counts.json",
    "outputs/figstage/math_hull_check.json",
    "outputs/figstage/rescore_flat_opaque.json",
    "outputs/figstage/vkid_ablation.json",
    # the human study: instrument, key, every response, analysis
    f"{HUMAN}/study.html", f"{HUMAN}/answer_key.json", f"{HUMAN}/analysis.json",
    # licence text
    "LICENSE",
]
EVIDENCE += sorted(str(p.relative_to(ROOT)) for p in (ROOT / HUMAN).glob("rater_*.json"))
# the scored arms' samples, so the canonical run can be re-scored
for arm in ("canon_oriented", "canon_unoriented", "native_oriented",
            "llm_control", "llm_bias"):
    EVIDENCE += [f"outputs/bench_arms/{arm}_32.npz",
                 f"outputs/bench_arms/{arm}_32_manifest.json"]
EVIDENCE += sorted(str(p.relative_to(ROOT))
                   for p in (ROOT / "tests" / "fixtures").rglob("*") if p.is_file())

# --------------------------------------------------------------------------
# 3. Scrubbing
# --------------------------------------------------------------------------

TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".html", ".csv", ".toml", ".cfg", ".ldr"}

SCRUB = [
    (re.compile(r"/home/griffing52/tmp/claude-1000/-home-griffing52-programming-gen-BlockGen"
                r"/[0-9a-f-]+/scratchpad"), "<scratch>"),
    (re.compile(r"/home/griffing52/programming/gen/BlockGen"), "<repo>"),
    (re.compile(r"/home/griffing52/[^\s\"']*blender[^\s\"']*"), "blender"),
    (re.compile(r"/home/griffing52"), "<home>"),
    (re.compile(r"ggalimi@math\.ucla\.edu|emailgriffinnow@gmail\.com"), "<email>"),
    (re.compile(r"\bgg-ml\b"), "host-a"),
    (re.compile(r"griffing52"), "user"),
    (re.compile(r"Griffin Galimi|Galimi|Griffin", re.I), "anon"),
]

# Anything matching these after scrubbing fails the build. Base64 payloads in
# study.html produce random matches for short words, so the html is checked
# with the strict subset only.
LEAK = re.compile(r"griffin|galimi|ucla\.edu|math\.ucla|gg-ml|/home/|claude-1000"
                  r"|emailgriffin|griffing52", re.I)
LEAK_STRICT = re.compile(r"galimi|ucla\.edu|math\.ucla|gg-ml|/home/griffing|claude-1000"
                         r"|emailgriffin|griffing52", re.I)


def scrub(text: str) -> str:
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text


# --------------------------------------------------------------------------
# 4. Build
# --------------------------------------------------------------------------

def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    seeds = [s for s in ENTRY_POINTS if s not in DROP]
    code = import_closure(seeds)
    files = sorted(code | set(seeds) | set(EVIDENCE))

    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        sys.exit("missing from repo:\n  " + "\n  ".join(missing))

    n_bytes = 0
    for rel in files:
        src, dst = ROOT / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES:
            dst.write_text(scrub(src.read_text()))
        else:
            shutil.copy2(src, dst)
        n_bytes += dst.stat().st_size

    # package files: an empty __init__ for every package directory that lacks one
    for pkg in ("blockgen", "legogen"):
        for d in (out / pkg).rglob("*"):
            if d.is_dir() and not (d / "__init__.py").exists():
                (d / "__init__.py").write_text("")
    for d in [out / "blockgen", out / "legogen"]:
        if not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("")

    # release-only files
    for name in ("README.md", "pyproject.toml", "requirements.txt",
                 "requirements-full.txt", ".gitignore"):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, out / name)

    # leak check
    leaks = []
    for p in out.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        pat = LEAK_STRICT if p.suffix == ".html" else LEAK
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if pat.search(line):
                leaks.append(f"{p.relative_to(out)}:{i}: {line.strip()[:100]}")
    if leaks:
        sys.exit("identifying strings survived scrubbing:\n  " + "\n  ".join(leaks))

    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"release: {out}")
    print(f"  {n_files} files, {n_bytes / 1e6:.1f} MB")
    print(f"  code modules: {len(code)}   entry points: {len(seeds)}   evidence files: {len(EVIDENCE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="release directory (recreated)")
    ap.add_argument("--list", action="store_true", help="print the file list and stop")
    args = ap.parse_args()
    if args.list:
        seeds = [s for s in ENTRY_POINTS if s not in DROP]
        for f in sorted(import_closure(seeds) | set(seeds) | set(EVIDENCE)):
            print(f)
    else:
        build(Path(args.out).resolve())
