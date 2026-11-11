"""Verify every numeric claim in the paper against the data that produced it.

A paper's numbers drift. A run gets re-scored, a probe changes, a table is
edited by hand, and the text quietly stops matching the artifact it cites. That
failure is invisible at build time -- LaTeX compiles a wrong number exactly as
happily as a right one -- and expensive at review time.

This is the registry that closes that gap. Each entry names a claim, where it
appears in the paper, and how to recompute it from a scorecard or a recorded
run. Run it before every submission:

    .venv/bin/python scripts/audit_claims.py
    .venv/bin/python scripts/audit_claims.py --verbose        # show the derivation
    .venv/bin/python scripts/audit_claims.py --unregistered   # headline numerals no claim covers
    .venv/bin/python scripts/audit_claims.py --run 0917       # another run of record
    .venv/bin/python scripts/audit_claims.py --compare-runs   # what changes under each run
    .venv/bin/python scripts/audit_claims.py --paper deliverables/paper/iclr

Besides checking the registry against the data, it reads the paper's TeX
(default deliverables/paper/cvpr27 if it exists, else iclr) and reports every
printed number that equals the same quantity under a DIFFERENT run (STALE,
fails the audit), near-misses of registered values, registered claims the TeX
never prints, and headline numerals no claim covers (a warning). There are
two runs of record, one per role, and nothing else names them: RUN_OF_RECORD
(the protocol run: probes, case study, headline separations) and
LEADERBOARD_RUN (Table tab:leaderboard and its prose). `--run` and
`--leaderboard-run` override them.

Adding a number to the paper means adding it here. That is the point: a claim
nobody can recompute is a claim nobody can check.

Note on noise floors
--------------------
Separations are quoted in units of the *run's own* noise floor, stored in
`context.noise_floor_sd`. Dividing an n=128 difference by an n=64 noise floor
gives a different and meaningless answer -- KID's sampling noise falls with n,
so the unit is only defined within a run. `sep()` below enforces that by
reading both numerator and denominator from the same scorecard.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "deliverables" / "paper"


def resolve_paper(arg: Optional[Path]) -> Path:
    """A paper directory, or a short name under deliverables/paper (`cvpr27`)."""
    if arg is None:
        return PAPER.resolve()
    for p in (Path(arg), PAPERS / str(arg)):
        if p.is_dir():
            return p.resolve()
    return Path(arg).resolve()


def default_paper() -> Path:
    """The CVPR 2027 port once it exists, the ICLR draft until then."""
    cvpr = PAPERS / "cvpr27"
    return cvpr if (cvpr / "main.tex").exists() else PAPERS / "iclr"


PAPER = default_paper()

#: The bench runs the paper could cite. Choosing between them is the author's
#: call; this table and RUN_OF_RECORD below are the ONLY places that choice is
#: written down, so switching it (or passing `--run 0917`) moves every claim,
#: the TeX cross-check and paper_readiness.py's sample-size check together.
#: 0917 is what the ICLR draft's tables were transcribed from; 0925 re-scored
#: the same arms after the renderer's transparency-cull fix.
RUNS = {
    "0917": ROOT / "outputs" / "run_20260917_043807_bench_canonical-split_native_arms_oriented_vs",
    "0925": ROOT / "outputs" / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull",
    # n=256 rerun of 0925's protocol (--descriptor v1, 77-d), no LLM arms, plus
    # the external Dream-Cubed arms; validated against 0925 by
    # run_20261102_214725_bench_validation_0925_re-score_under_the_v1 (notes §69).
    "1103_n256": ROOT / "outputs" / "run_20261103_090315_bench_n_256_leaderboard_of_record",
}
#: Two roles, two runs. RUN_OF_RECORD is the PROTOCOL run of record: the
#: headline probe separations, the case study, and every claim that reads
#: `card` below. LEADERBOARD_RUN is the run Table~tab:leaderboard, its rank
#: intervals and the leaderboard prose come from: every claim that reads `lb`
#: or `lbc`. The split is deliberate (user decision 2026-11-04): the n=256
#: rerun pins 0925's protocol (`--descriptor v1`) and adds the external arms,
#: while the ladder, dose-response, gates and probe numbers stay on 0925.
#: A claim's role is read off its closure (see build_claims), so a claim
#: cannot be registered against one run and silently computed on the other.
RUN_OF_RECORD = "0925"
LEADERBOARD_RUN = "1103_n256"


def resolve_run(key: str) -> Path:
    """A RUNS key, a run directory name, or a path, to the run directory."""
    if key in RUNS:
        return RUNS[key]
    for p in (Path(key), ROOT / "outputs" / key):
        if (p / "scorecard.json").exists():
            return p.resolve()
    raise SystemExit(f"unknown run {key!r}; known: {', '.join(RUNS)}")


def run_key(bench: Path) -> str:
    for k, p in RUNS.items():
        if p.resolve() == bench.resolve():
            return k
    return bench.name


BENCH = RUNS[RUN_OF_RECORD]
LB_BENCH = RUNS[LEADERBOARD_RUN]

#: Free-variable names that bind a claim to a run role. A claim reading
#: `card` moves with RUN_OF_RECORD; one reading `lb` / `lbc` moves with
#: LEADERBOARD_RUN. Claims that read a run through any other name are pinned
#: to that run and move with neither.
ROLE_VARS = {"card": "record", "lb": "leaderboard", "lbc": "leaderboard",
             "lbdir": "leaderboard"}
#: The dose run of record. The earlier `run_20260831_224723_doseresponse` built
#: its mixture axis from decimated real data, which this project no longer
#: treats as a quality reference, so its coefficients are superseded. Solidify
#: and occupancy-noise are unaffected by that choice and reproduce across both.
DOSE = ROOT / "outputs" / "run_20260925_010828_doseresponse_uniform_random-shell_box"
DOSE_N64 = ROOT / "outputs" / "run_20260925_011210_doseresponse_uniform_random-shell_box"
#: The render tier's validation ladder. Appendix Table `tab:ladder-render` is a
#: transcription of this file; the three appendix tables were each found to be
#: hand-edited copies of a superseded `results.md` table, which is why every
#: cell of all three now has a claim below.
#: Post-fix rerun (2026-11-04, notes §77) of the 2026-08-31 ladder in outputs/analysis/ladder.
LADDER = (ROOT / "outputs" / "run_20261104_221808_bench_mc_ladder_postfix" / "ladder"
          / "dinov2b__v4_az45-135-225-315_el30-30-30-30_px224_ortho.json")
CACHE = ROOT / "data" / "minecraft" / "cache"

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")


def _load(p: Path) -> dict:
    with open(p) as fh:
        return json.load(fh)


class Card:
    """A scorecard, with the accessors the claims actually need."""

    def __init__(self, path: Path):
        self.d = _load(path)
        self.arms = self.d["arms"]
        self.sd = self.d["context"]["noise_floor_sd"]

    def realism(self, arm: str, metric: str) -> float:
        return self.arms[arm]["realism"][metric]["value"]

    def sep(self, arm: str, metric: str, ref: str = "real_test") -> float:
        """Separation from the floor in this run's own noise floors."""
        num = self.realism(arm, metric) - self.realism(ref, metric)
        return num / self.sd[f"realism.{metric}"]



class Leaderboard:
    """The BlockScore pillar table, parsed out of the recorded `scorecard.md`.

    `scorecard.json` stores per-arm metrics but not the aggregate, which the
    scoring code derives; the leaderboard block of `scorecard.md` is therefore
    the recorded artifact for Table~\\ref{tab:leaderboard}. Parsing it catches
    the failure mode that actually threatens a hand-built LaTeX table --- a
    transcribed cell that is off by a digit --- which is worth more than nothing
    while the aggregation is re-derived. The pillar order is read from the
    header rather than assumed, so a change in the scorer's column order shows
    up as a KeyError instead of as silently shifted numbers.
    """

    def __init__(self, path: Path):
        rows, pillars = {}, None
        in_block = False
        for line in path.read_text().splitlines():
            if line.startswith("## leaderboard"):
                in_block = True
                continue
            if in_block and line.startswith("## "):
                break
            if not in_block or not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == "#":
                pillars = cells[4:]
                continue
            if set(cells[0]) <= set("- :") and len(cells[0]) > 1:
                continue
            if pillars is None or len(cells) < 4 + len(pillars):
                continue
            arm = cells[1]
            def num(t):
                t = t.replace("*", "").strip()
                return None if t in {"", "-", "\u2014", "DQ"} else float(t.replace("+", ""))
            rows[arm] = {"blockscore": num(cells[2]), "worst": cells[3],
                         **{p: num(v) for p, v in zip(pillars, cells[4:])}}
        if not rows:
            raise ValueError(f"no leaderboard block in {path}")
        self.rows = rows

    def pillar(self, arm: str, pillar: str) -> float:
        v = self.rows[arm][pillar]
        if v is None:
            raise KeyError(f"{arm}/{pillar} is not scored in this run")
        return v

    def ratio(self, arm: str, hi: str, lo: str) -> float:
        return self.pillar(arm, hi) / self.pillar(arm, lo)



@lru_cache(maxsize=1)
def _random_split_leakage() -> tuple:
    """(mean, sd, seed0) fraction of validation items sharing a family with train.

    Recomputed rather than quoted, because the paper's original figure was a
    single seed that happened to be the worst of twenty -- the exact shape of
    error this registry exists to catch.
    """
    import numpy as np
    from blockgen.eval.bench import splits as _sp
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    gk = np.array([_sp.group_key(it) for it in man["items"]])
    n = len(gk)
    rates = []
    for seed in range(20):
        perm = np.random.default_rng(seed).permutation(n)
        n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
        train_families = set(gk[perm[:n_tr]].tolist())
        val = gk[perm[n_tr:n_tr + n_va]]
        rates.append(float(np.mean([g in train_families for g in val])))
    r = np.asarray(rates)
    return float(r.mean()), float(r.std(ddof=1)), float(r[0])


@dataclass
class Claim:
    name: str
    paper_value: float
    compute: Callable[[], float]
    tol: float = 0.0
    unit: str = ""
    note: str = ""
    appears_in: str = ""
    #: True when the value is read from the bench run of record, so it moves
    #: when RUN_OF_RECORD does. Set by build_claims, not by hand.
    run_dependent: bool = False
    #: False for claims the paper states in words or as a verdict (a bound
    #: that holds, a gate that passes, "exactly two"); the TeX cross-check
    #: has no numeral to look for. 0.0/1.0-valued claims are skipped anyway.
    printed: bool = True
    #: "record" (moves with RUN_OF_RECORD), "leaderboard" (moves with
    #: LEADERBOARD_RUN) or "" (pinned). Set by build_claims, not by hand.
    role: str = ""

    def check(self) -> tuple[bool, float, Optional[str]]:
        try:
            got = self.compute()
        except Exception as exc:                       # noqa: BLE001
            return False, float("nan"), f"{type(exc).__name__}: {exc}"
        ok = abs(got - self.paper_value) <= max(self.tol, 1e-9)
        return ok, got, None


def needs_corpus(err: Optional[str]) -> bool:
    """A claim that could not be computed because the corpus is not on disk.

    The released code ships the splits and manifests but no structure content
    (Appendix: data), so a reader who has not rebuilt the corpus cannot recompute
    the claims that read it. Those are reported as such rather than as failures.
    """
    return bool(err) and "FileNotFoundError" in err and "data/minecraft/cache" in err


def build_claims(bench: Optional[Path] = None,
                 leaderboard: Optional[Path] = None) -> List[Claim]:
    """Every registered claim, with the bench-reading ones bound to their run.

    `bench` (default: RUN_OF_RECORD) is what claims reading `card` see;
    `leaderboard` (default: LEADERBOARD_RUN) is what claims reading `lb` (the
    recorded leaderboard block) or `lbc` (that run's scorecard) see. Each claim
    is tagged with its role from its closure, so a new claim cannot forget it.
    """
    bench = BENCH if bench is None else bench
    leaderboard = LB_BENCH if leaderboard is None else leaderboard
    card = Card(bench / "scorecard.json")
    lbc = Card(leaderboard / "scorecard.json")
    dose = _load(DOSE / "doseresponse.json")
    claims: List[Claim] = []
    C = claims.append

    # --- the headline -----------------------------------------------------
    C(Claim("solidify separation, geometry metric", 202.0,
            lambda: card.sep("real@solidify", "geom_kid"), tol=0.5,
            unit="noise floors", appears_in="abstract, intro, §4",
            note="the paper's central positive number"))
    C(Claim("solidify separation, render metric", 0.139,
            lambda: card.sep("real@solidify", "mv_dino_kid"), tol=0.02,
            unit="noise floors", appears_in="§4",
            note="under half a noise floor -- the blindness itself"))
    C(Claim("render metric, real floor", -0.0018,
            lambda: card.realism("real_test", "mv_dino_kid"), tol=0.0005,
            appears_in="abstract, §4"))
    C(Claim("render metric, solidify", -0.0014,
            lambda: card.realism("real@solidify", "mv_dino_kid"), tol=0.0005,
            appears_in="abstract, §4"))
    C(Claim("render metric, verbatim training copies", 0.017,
            lambda: card.realism("train_verbatim", "mv_dino_kid"), tol=0.001,
            appears_in="abstract, §4",
            note="the difference the same test DOES resolve"))
    C(Claim("geometry metric is exactly invariant to material shuffle", 0.0,
            lambda: (card.realism("real_shuffled_materials", "geom_kid")
                     - card.realism("real_test", "geom_kid")),
            tol=1e-12, appears_in="abstract, §4",
            note="must be bit-identical, not merely close"))
    C(Claim("geometry metric is exactly invariant to monochrome", 0.0,
            lambda: (card.realism("real@monochrome", "geom_kid")
                     - card.realism("real_test", "geom_kid")),
            tol=1e-12, appears_in="§4"))

    # --- dose response ----------------------------------------------------
    sol = dose["axes"]["solidify"]
    dose64 = _load(DOSE_N64 / "doseresponse.json")["axes"]["solidify"]
    C(Claim("solidify dose, render range at n=64", 0.48,
            lambda: dose64["mv_dino_kid"]["range_sd"], tol=0.02,
            unit="noise floors", appears_in="abstract, intro, §4"))
    C(Claim("solidify dose, render range at n=128", 0.34,
            lambda: sol["mv_dino_kid"]["range_sd"], tol=0.02,
            unit="noise floors", appears_in="abstract, intro, §4",
            note="the paper quotes BOTH n, because the rank correlation of a "
                 "flat sequence is unstable (+0.300 at n=64, +0.900 at n=128) "
                 "while the range barely moves -- the range is the claim"))
    C(Claim("solidify dose, geometry range at n=128", 286.7,
            lambda: sol["geom_kid"]["range_sd"], tol=1.0,
            unit="noise floors", appears_in="abstract, §4"))
    C(Claim("solidify dose, geometry Spearman", 1.000,
            lambda: sol["geom_kid"]["spearman"], tol=0.005,
            appears_in="§4"))
    C(Claim("solidify dose, geometry range at n=64", 133.0,
            lambda: dose64["geom_kid"]["range_sd"], tol=1.0,
            unit="noise floors", appears_in="§4"))
    C(Claim("render metric is monotone on occupancy noise", 1.000,
            lambda: dose["axes"]["occ_noise"]["mv_dino_kid"]["spearman"],
            tol=0.005, appears_in="§4",
            note="the contrast that makes the solidify row specific"))

    # --- the quadratic response ------------------------------------------
    #: The mixture axis is now keyed by which bad pool it used. We register
    #: `uniform_random` -- a procedural baseline that is bad along a *quality*
    #: axis -- because the decimation pool the earlier run used changes a
    #: build's resolution, not its quality, and this project no longer treats
    #: that as a degradation.
    mix = dose["axes"]["mixture_uniform_random"]["mv_dino_kid"]

    def bad_quarter_ratio() -> float:
        """How bad a 25%-failing arm READS, against a fully-bad arm."""
        v = mix["value"]
        return v[1] / v[-1]

    C(Claim("a 25%-bad arm reads this fraction of fully-bad", 0.060,
            bad_quarter_ratio, tol=0.0006, appears_in="app:failrate (as 6.0%); cut from main §6.4 2026-11-07",
            note="measured against a PROCEDURAL bad pool, not decimation; "
                 "(1-a)^2 predicts 0.0625. NOT the measured/linear ratio, "
                 "which a previous draft conflated with this"))

    # --- the case study ---------------------------------------------------
    C(Claim("contaminated arm, geometry metric", 11.624,
            lambda: card.realism("native_oriented", "geom_kid"), tol=0.01,
            appears_in="§6"))
    C(Claim("clean retrain, geometry metric", 14.898,
            lambda: card.realism("canon_oriented", "geom_kid"), tol=0.01,
            appears_in="§6"))
    C(Claim("LLM baseline, geometry metric", 15.006,
            lambda: card.realism("llm_control", "geom_kid"), tol=0.01,
            appears_in="§6",
            note="reproduces to 3 decimals across two runs -- the "
                 "cross-run consistency check"))
    C(Claim("clean retrain is NOT separated from the LLM baseline", 1.0,
            lambda: float(abs(card.realism("canon_oriented", "geom_kid")
                              - card.realism("llm_control", "geom_kid")) < 0.2),
            appears_in="§6",
            note="the whole point of the case study"))
    C(Claim("contaminated arm IS separated from the LLM baseline", 1.0,
            lambda: float(card.realism("llm_control", "geom_kid")
                          - card.realism("native_oriented", "geom_kid") > 3.0),
            appears_in="§6"))

    # --- corpus facts -----------------------------------------------------
    def split_sizes() -> float:
        from blockgen.eval.bench import splits
        sp = splits.load_split("houses_32")
        return float(len(sp.test))

    C(Claim("canonical test split size", 399.0, split_sizes, tol=0,
            appears_in="§3, everywhere as the reference size"))

    claims += corpus_claims() + ladder_claims() + baseline_claims()
    # --- Table 1, the restored leaderboard --------------------------------
    # Grouped and renumbered for print; every cell is checked against the
    # recorded scorecard so the regrouping cannot smuggle in a wrong number.
    lb = Leaderboard(leaderboard / "scorecard.md")
    lbdir = leaderboard
    #: (arm, pillar, value). Every cell Table~tab:leaderboard prints. Re-registered
    #: 2026-11-04 against the n=256 leaderboard of record; the LLM arms are not on
    #: it (they remain in the protocol run's case study, read through `card`).
    for arm, pillar, want in [
            ("real_test", "appearance", 0.00), ("real_test", "geometry", 0.00),
            ("dreamcubed", "blockscore", 165.71), ("dreamcubed", "appearance", 165.71),
            ("dreamcubed", "geometry", 146.64), ("dreamcubed", "palette", 87.65),
            ("native_oriented", "appearance", 128.87),
            ("native_oriented", "geometry", 229.29),
            ("canon_oriented", "appearance", 157.40),
            ("canon_oriented", "geometry", 285.22),
            ("canon_unoriented", "appearance", 182.86),
            ("canon_unoriented", "geometry", 345.10),
            ("patchwork", "appearance", 431.70), ("patchwork", "geometry", 368.21),
            ("gabled_house", "appearance", 535.62), ("gabled_house", "geometry", 620.98),
            ("shell_box", "appearance", 864.00), ("shell_box", "geometry", 645.86),
            ("uniform_random", "appearance", 1087.06),
            ("uniform_random", "geometry", 727.48),
            ("real@solidify", "appearance", 0.35), ("real@solidify", "geometry", 444.36),
            ("real_shuffled_materials", "appearance", 422.43),
            ("real_shuffled_materials", "geometry", 0.00),
            ("real@monochrome", "appearance", 535.36),
            ("real@monochrome", "geometry", 0.00),
            ("train_copy_noise", "appearance", 271.94),
            ("train_copy_noise", "geometry", 619.13),
    ]:
        C(Claim(f"leaderboard {arm}/{pillar}", want,
                lambda a=arm, q=pillar: lb.pillar(a, q),
                tol=1e-9 if want == 0 else 0.005,
                unit="noise floors", appears_in="Table 1",
                note="cell of the n=256 leaderboard"))
    #: raw MMD^2 (x1e3) beside the z-scores, which are not comparable across n
    for arm, mv, gk in [
            ("real_test", 0.001, -0.14), ("dreamcubed", 0.268, 7.67),
            ("native_oriented", 0.208, 12.08), ("canon_oriented", 0.254, 15.06),
            ("canon_unoriented", 0.295, 18.25), ("patchwork", 0.696, 19.48),
            ("gabled_house", 0.863, 32.96), ("shell_box", 1.392, 34.28),
            ("uniform_random", 1.752, 38.63), ("real@solidify", 0.001, 23.54),
            ("real_shuffled_materials", 0.681, -0.14),
            ("real@monochrome", 0.863, -0.14), ("train_copy_noise", 0.439, 32.86)]:
        C(Claim(f"leaderboard raw mv_dino_kid {arm}", mv,
                lambda a=arm: lbc.realism(a, "mv_dino_kid"), tol=0.0005,
                appears_in="Table 1, raw MMD^2 column"))
        C(Claim(f"leaderboard raw geom_kid {arm}", gk,
                lambda a=arm: lbc.realism(a, "geom_kid"), tol=0.005,
                appears_in="Table 1, raw MMD^2 column"))
    C(Claim("verbatim copies, raw render metric on the leaderboard", 0.012,
            lambda: lbc.realism("train_verbatim", "mv_dino_kid"), tol=0.0005,
            appears_in="app:tables, why coherence is distance to real"))
    C(Claim("verbatim copies rank third of sixteen on raw render metric", 3.0,
            lambda: float(1 + sum(lbc.realism(a, "mv_dino_kid")
                                  < lbc.realism("train_verbatim", "mv_dino_kid")
                                  for a in lbc.arms)),
            appears_in="app:tables", printed=False))
    C(Claim("pillar gap on the interior probe", 1270.0,
            lambda: lb.ratio("real@solidify", "geometry", "appearance"), tol=1.0,
            unit="x", appears_in="§6.4",
            note="printed as 1,270. The appearance pillar is +0.35 at n=256; the "
                 "ratio of a near-zero pillar is noisy (530 at 0917, 1441 at 0925), "
                 "so it is read as 'three orders of magnitude', never compared across runs"))
    C(Claim("best attack on the geometric metric", 1.17,
            lambda: lb.ratio("patchwork", "appearance", "geometry"), tol=0.02,
            unit="x", appears_in="§6.4 (printed 1.2x)"))

    # --- rank intervals and ties (rank_intervals.json of the leaderboard run) ----
    # Read from the twin-gate re-score of the leaderboard (notes §78): same
    # cells, plus the within-set near-twin gate, which disqualifies Dream-Cubed,
    # gabled_house and shell_box, so the ranks of every arm below them move.
    def _rank_iv(arm: str, end: int) -> float:
        ri = _load(TWIN_RUN / "rank_intervals.json")
        for r in ri["rows"]:
            if r.get("arm") == arm:
                return float(r["rank_ci"][end])
        raise KeyError(arm)
    for arm, lo, hi in [("native_oriented", 2, 2),
                        ("canon_oriented", 3, 3), ("canon_unoriented", 4, 4),
                        ("real_shuffled_materials", 5, 7), ("patchwork", 5, 7),
                        ("real@solidify", 5, 7), ("real@monochrome", 8, 8),
                        ("train_copy_noise", 9, 9), ("uniform_random", 10, 10)]:
        C(Claim(f"rank interval {arm} lower", float(lo),
                lambda a=arm: _rank_iv(a, 0), appears_in="Table 1 rank column",
                printed=False))
        C(Claim(f"rank interval {arm} upper", float(hi),
                lambda a=arm: _rank_iv(a, 1), appears_in="Table 1 rank column",
                printed=False))

    # --- Dream-Cubed, the published generator (notes §66, §69) ----------------
    C(Claim("Dream-Cubed NN-IoU to train", 0.42,
            lambda: lbc.arms["dreamcubed"]["novelty"]["voxel_nn_iou_mean"]["value"],
            tol=0.005, appears_in="§6.4, app:dreamcubed"))
    C(Claim("held-out real NN-IoU to train, leaderboard", 0.38,
            lambda: lbc.arms["real_test"]["novelty"]["voxel_nn_iou_mean"]["value"],
            tol=0.005, appears_in="§6.4, app:dreamcubed"))
    C(Claim("Dream-Cubed duplicate rate", 0.004,
            lambda: lbc.arms["dreamcubed"]["novelty"]["voxel_dup_rate"]["value"],
            tol=0.0005, appears_in="§6.4, app:dreamcubed"))
    C(Claim("held-out real duplicate rate, leaderboard", 0.008,
            lambda: lbc.arms["real_test"]["novelty"]["voxel_dup_rate"]["value"],
            tol=0.0005, appears_in="app:dreamcubed"))
    _dcs = _load(ROOT / "outputs" / "figstage" / "dreamcubed_size.json")["arms"]["dreamcubed"]
    C(Claim("Dream-Cubed raw V-KID, full descriptor", 7.67,
            lambda: _dcs["geom_kid_full"], tol=0.005, appears_in="§6.4, app:dreamcubed"))
    C(Claim("Dream-Cubed raw V-KID without the size term", 7.17,
            lambda: _dcs["geom_kid_no_size"], tol=0.005,
            appears_in="§6.4, app:dreamcubed"))
    C(Claim("size share of Dream-Cubed's geometric gap, upper bound", 0.07,
            lambda: _dcs["size_share_upper_bound"], tol=0.006, unit="fraction",
            appears_in="§6.4 (as 7%)"))
    C(Claim("Dream-Cubed raw V-KID against a size-matched reference", 8.57,
            lambda: _dcs["geom_kid_size_matched_ref"], tol=0.005,
            appears_in="app:dreamcubed"))
    C(Claim("reference builds inside Dream-Cubed's block-count IQR", 111.0,
            lambda: float(_dcs["ref_inside_iqr"]), appears_in="app:dreamcubed"))
    C(Claim("Dream-Cubed median blocks", 272.0,
            lambda: _dcs["median_blocks"], tol=0.5, appears_in="§6.4, app:dreamcubed"))

    # --- the memorization gate on Voxel-AR-U, both n ----------------------------
    C(Claim("Voxel-AR-U retrieval percentile, n=256", 0.403,
            lambda: lbc.arms["canon_unoriented"]["novelty"]["dino_nn_percentile"]["value"],
            tol=0.0005, appears_in="§6.5"))
    C(Claim("real retrieval percentile floor, n=256", 0.507,
            lambda: lbc.arms["real_test"]["novelty"]["dino_nn_percentile"]["value"],
            tol=0.0005, appears_in="§6.5, app:tables"))
    C(Claim("Voxel-AR-U retrieval percentile, n=128", 0.382,
            lambda: card.arms["canon_unoriented"]["novelty"]["dino_nn_percentile"]["value"],
            tol=0.0005, appears_in="§6.5"))
    C(Claim("real retrieval percentile floor, n=128", 0.534,
            lambda: card.arms["real_test"]["novelty"]["dino_nn_percentile"]["value"],
            tol=0.0005, appears_in="§6.5"))

    # --- group-aware split leakage ----------------------------------------
    C(Claim("random-split family leakage, mean over 20 seeds", 0.287,
            lambda: _random_split_leakage()[0], tol=0.003,
            unit="fraction", appears_in="§5",
            note="the paper quoted seed 0's 35.8% bare until an audit found it "
                 "is the MAXIMUM of 20 seeds at z=+2.60; the mean is the honest "
                 "figure and the argument survives it"))
    C(Claim("random-split family leakage, seed 0", 0.358,
            lambda: _random_split_leakage()[2], tol=0.003,
            unit="fraction", appears_in="§5"))

    # --- the sqrt readout, after an audit found the intro bullet wrong -------
    _mf = dose["mixture_fits"]["mixture_uniform_random"]
    C(Claim("sqrt readout MAE, render metric", 0.013,
            lambda: _mf["mv_dino_kid"]["mae_sqrt"], tol=0.0006,
            appears_in="app:failrate; cut from main 2026-11-07",
            note="the intro said 0.022, which is the GEOMETRIC metric's value"))
    C(Claim("raw readout MAE, render metric", 0.128,
            lambda: _mf["mv_dino_kid"]["mae_raw"], tol=0.0015,
            appears_in="app:failrate; cut from main 2026-11-07",
            note="the intro said 0.119, traceable only to an n=64 run whose bad "
                 "pool (canon8) this project disavowed as a quality reference"))
    C(Claim("sqrt readout MAE, geometric metric", 0.020,
            lambda: _mf["geom_kid"]["mae_sqrt"], tol=0.0006,
            appears_in="app:failrate"))
    C(Claim("raw readout MAE, geometric metric", 0.131,
            lambda: _mf["geom_kid"]["mae_raw"], tol=0.0015,
            appears_in="app:failrate"))
    C(Claim("mixture exponent, render metric", 2.05,
            lambda: _mf["mv_dino_kid"]["exponent_best_fit"], tol=0.05,
            appears_in="app:failrate", note="quadratic response, as the identity predicts"))

    # --- the render-leak decomposition (Appendix app:hull) ------------------
    _hull = _load(ROOT / "outputs" / "figstage" / "math_hull_check.json")
    C(Claim("solidify is EXACTLY invisible to an opaque flat renderer", 1.0,
            lambda: _hull["B_flat_opaque_v4"]["bit_identical_frac"], tol=0.0,
            unit="fraction bit-identical", appears_in="abstract, §6.3, app:hull",
            note="the impossibility result: must be exactly 1.0, not approximately"))
    C(Claim("deployed renderer, fraction of views altered", 0.905,
            lambda: 1.0 - _hull["B_paper_v4"]["bit_identical_frac"], tol=0.006,
            appears_in="abstract (as 91%), §6.3",
            note="the paper said 86%, which came from a 60-build subsample"))
    C(Claim("flat textures, fraction of views altered", 0.284,
            lambda: 1.0 - _hull["B_flat_v4"]["bit_identical_frac"], tol=0.01,
            appears_in="app:hull",
            note="the corruption's own reach once the atlas artefact is removed"))
    C(Claim("atlas artefact alone still alters most views", 0.917,
            lambda: 1.0 - _hull["B_opaque_v4"]["bit_identical_frac"], tol=0.01,
            appears_in="app:hull",
            note="translucency off, textures on: the renderer defect by itself"))
    C(Claim("surface invariance violations under solidify", 0.0,
            lambda: float(_hull["surface_invariance"]["solidify_violations"]),
            tol=0.0, appears_in="app:hull (Lemma 2)",
            note="exhaustive over 399 held-out builds; must be exactly zero"))
    C(Claim("V-KID is an identity under material shuffle", 0.0,
            lambda: float(_hull["D_matching_lemma"]["arms"]["material_shuffle"]
                          ["max_abs_descriptor_diff"]),
            tol=0.0, appears_in="app:hull (Lemma 4)",
            note="bit-identical descriptors, not a small number"))

    # The abstract rounds the headline down to "more than 200". Register the
    # BOUND, not just the value, so a future run that drops the separation below
    # the rounding fails here instead of quietly making the abstract false.
    C(Claim("abstract bound: solidify separation exceeds 200 floors", 1.0,
            lambda: float(card.sep("real@solidify", "geom_kid") > 200.0),
            tol=0.0, unit="holds", appears_in="abstract",
            note="1.0 means the bound holds; 0.0 means the abstract overstates"))

    # --- the flat+opaque rescore (T38) -------------------------------------
    _rs = _load(ROOT / "outputs" / "figstage" / "rescore_flat_opaque.json")["pipelines"]
    C(Claim("rescore: opaque flat renderer separates solidify by exactly zero", 0.0,
            lambda: _rs["flat_opaque"]["separation_floors"], tol=0.0,
            unit="noise floors", appears_in="§6.1, app:hull",
            note="end-to-end confirmation of the occlusion lemma through the real "
                 "bench path: bit-identical features, so the KID difference is an "
                 "identity. Zero tolerance -- an approximate zero here would mean "
                 "the proof or the implementation is wrong"))
    C(Claim("rescore: opaque flat features are bit-identical", 0.0,
            lambda: _rs["flat_opaque"]["max_abs_feature_diff"], tol=0.0,
            appears_in="app:hull"))
    C(Claim("rescore: deployed renderer residual", 0.119,
            lambda: _rs["paper"]["separation_floors"], tol=0.01,
            unit="noise floors", appears_in="app:hull",
            note="the bench run of record reads 0.38 for the same quantity; both "
                 "are well under one floor and the runs estimate different floors"))
    C(Claim("rescore: residual with transmission only", -0.069,
            lambda: _rs["flat"]["separation_floors"], tol=0.01,
            unit="noise floors", appears_in="app:hull",
            note="NEGATIVE -- the probe reads closer to real than real does, which "
                 "is what sub-noise-floor wobble looks like"))
    C(Claim("rescore: residual with the atlas artefact only", 0.026,
            lambda: _rs["opaque"]["separation_floors"], tol=0.01,
            unit="noise floors", appears_in="app:hull"))

    # --- contiguous vs shuffled calibration draw (§5) -----------------------
    _cd = _load(ROOT / "outputs" / "figstage" / "contiguous_draw_check.json")
    C(Claim("contiguous draw inflates the noise floor", 10.9,
            lambda: _cd["sd_inflation"], tol=0.05, unit="x",
            appears_in="app:repro (main §5 still prints the stale 10.4x; re-point pending)",
            note="replaces an unbacked 'roughly 45%' that traced to no recorded run"))
    C(Claim("shuffled calibration draw reproduces the bench floor", 0.0029,
            lambda: _cd["shuffled"]["sd"], tol=0.00005,
            appears_in="app:repro",
            note="re-derived 2026-11-04 from contiguous_draw_check.json (0.002945); "
                 "the protocol run's realism.mv_dino_kid floor is 0.002887, so the "
                 "independent recomputation agrees to 2%. The old 0.00305 / 10.4x "
                 "were an earlier draw of this check"))
    C(Claim("contiguous calibration draw sd", 0.0320,
            lambda: _cd["contiguous"]["sd"], tol=0.00005, appears_in="app:repro"))
    C(Claim("contiguous calibration draw centre", 0.0410,
            lambda: _cd["contiguous"]["mean"], tol=0.00005, appears_in="app:repro"))

    # --- per-build influence, the probe-free evidence (fig:influence) -------
    _inf = _load(ROOT / "outputs" / "figstage" / "influence_pair.json")
    C(Claim("render metric's per-build ranking vs sealed interior volume", 0.020,
            lambda: _inf["rho_render_sealed"], tol=0.01, unit="Spearman rho",
            appears_in="fig:influence, app:vkid-robust (cut from main §6.4 2026-11-07)",
            note="not significantly different from zero; this is interior "
                 "blindness measured on real builds with no probe"))
    C(Claim("geometric metric's per-build ranking vs sealed interior volume", -0.343,
            lambda: _inf["rho_geom_sealed"], tol=0.01, unit="Spearman rho",
            appears_in="fig:influence, app:vkid-robust (cut from main §6.4 2026-11-07)"))
    C(Claim("the two per-build rankings are nearly independent", 0.122,
            lambda: _inf["rho_render_geom"], tol=0.01, unit="Spearman rho",
            appears_in="fig:influence, app:vkid-robust (cut from main §6.4 2026-11-07)"))

    # --- V-KID descriptor ablation and the mass-matched control ------------
    _ab = _load(ROOT / "outputs" / "figstage" / "vkid_ablation.json")
    _rows = {r["variant"]: r for r in _ab["rows"]}

    def _ab_get(variant, arm="real@solidify"):
        return _rows[variant][arm]

    C(Claim("V-KID separation survives deleting the size term", 202.2,
            lambda: _ab_get("without log1p_n_blocks"), tol=1.0,
            unit="noise floors", appears_in="§6.4, app:descriptor",
            note="if V-KID were counting blocks this would collapse"))
    C(Claim("V-KID separation survives deleting the sealed-interior scalar", 197.1,
            lambda: _ab_get("without interior_ratio_sealed"), tol=1.0,
            unit="noise floors", appears_in="§6.4"))
    C(Claim("V-KID separation survives deleting both suspect scalars", 197.3,
            lambda: _ab_get("without both suspects"), tol=1.0,
            unit="noise floors", appears_in="§6.4"))
    C(Claim("best single descriptor block on the probe", 164.5,
            lambda: _ab_get("scalars only (9d)"), tol=1.0,
            unit="noise floors", appears_in="§6.4, app:descriptor",
            note="no single block reproduces the full 201.8"))
    C(Claim("mass-matched control, render metric", 102.27,
            lambda: _ab["render"]["real@grow"], tol=0.5,
            unit="noise floors", appears_in="§6.4",
            note="the render metric SEES the matched control it cannot see the "
                 "probe: same mass added, outside instead of inside"))
    C(Claim("probe on the render metric, this run", 0.26,
            lambda: _ab["render"]["real@solidify"], tol=0.05,
            unit="noise floors", appears_in="§6.4"))
    C(Claim("mass-matched control, geometric metric", 214.5,
            lambda: _ab_get("full (all 77)", "real@grow"), tol=1.0,
            unit="noise floors", appears_in="§6.4"))
    C(Claim("mass match is exact on every build", 128.0,
            lambda: float(_ab["matched_exact"]), tol=0.0,
            unit="of 128 builds", appears_in="§6.4",
            note="within one block; the control is matched, not approximately matched"))

    # --- the aperture-aware interior counter on real houses ---------------
    # These four decide how far the sealed-only counter understates real data,
    # and app:interior carried stale values for all of them until they were
    # recorded here: a build is altered by the probe iff it has a sealed air
    # cell, so `altered_frac` and `no_sealed_frac` must sum to exactly 1 and the
    # appendix's old 9.0% could not coexist with the method section's 94%.
    _ic = _load(ROOT / "outputs" / "figstage" / "interior_counts.json")["rows"]["draw_200"]

    C(Claim("real houses with no sealed interior at all", 0.060,
            lambda: _ic["no_sealed_frac"], tol=0.002,
            unit="of 200 held-out houses", appears_in="app:interior",
            note="complement of the 94% the probe alters, by construction"))
    C(Claim("those builds that do have a room through a doorway", 12.0,
            lambda: float(_ic["no_sealed_with_aperture_room"]), tol=0.0,
            unit="of 12 builds", appears_in="app:interior",
            note="every one of them: a sealed-only counter calls these roomless"))
    C(Claim("pooled share of real interior volume behind an aperture", 0.256,
            lambda: _ic["aperture_share_pooled"], tol=0.005,
            unit="fraction of interior volume", appears_in="app:interior",
            note="pooled over builds, not a per-build mean; the per-build sealed "
                 "share is the 85.5% in \u00a74.1"))
    C(Claim("per-build sealed share of interior volume", 0.855,
            lambda: _ic["sealed_share_per_build_mean"], tol=0.005,
            unit="mean over builds", appears_in="\u00a74.1"))
    # The other half of the fig:lego comparison. It was the one number in that
    # caption nothing checked, which mattered once the voxelizer feeding it was
    # repaired: restoring the dropped baseplates moved the population this
    # median is taken over (models with >= 8 sealed cells, 121/400 -> 145/400)
    # without moving the median itself.
    _omr = _load(ROOT / "outputs" / "figstage" / "omr_audit.json")
    C(Claim("OMR damage on the same gated basis", 0.012,
            lambda: _omr["void_frac_of_mass_median_gated"], tol=0.001,
            unit="median block increase", appears_in="fig:lego caption",
            note="official LEGO sets enclose far less air than either measured "
                 "corpus; compare the 0.358 registered below for houses"))

    C(Claim("probe damage on the basis the LEGO audit reports", 0.358,
            lambda: _ic["block_delta_median_gated"], tol=0.005,
            unit="median block increase", appears_in="fig:lego caption",
            note="conditioned on at least 8 sealed cells, the lego.py gate, so "
                 "the +1.2% OMR median is a like-for-like comparison"))

    # The rater study (run_20260923_human_v5) was removed from the paper on
    # 2026-11-02: it had no human-subjects determination. Its claims are gone too.

    claims += added_claims_20261104(card)
    claims += added_claims_hollow_n256()
    claims += added_claims_views25d()
    claims += added_claims_unregistered_20261104()
    claims += added_claims_vkid_robust()
    claims += added_claims_rankflips()
    claims += added_claims_effect_sizes()
    claims += added_claims_twin_gate()
    claims += added_claims_leak_postfix_20261106()
    claims += added_claims_consistency_20261109()
    claims += added_claims_rankflips_exact()
    claims += added_claims_functional_rebase()
    claims += added_claims_stats_supp()
    claims += added_claims_round2_text()
    for c in claims:
        free = set(getattr(getattr(c.compute, "__code__", None), "co_freevars", ()))
        roles = {ROLE_VARS[v] for v in free & set(ROLE_VARS)}
        if len(roles) > 1:
            raise ValueError(f"claim {c.name!r} reads both runs; split it")
        c.role = roles.pop() if roles else ""
        c.run_dependent = bool(c.role)
        if c.unit == "holds" or c.name.startswith("exactly two"):
            c.printed = False
    return claims



def added_claims_consistency_20261109() -> List[Claim]:
    """Numbers introduced or relabelled by the 2026-11-09 main/supplement consistency pass.

    Each reads the run that produced it. The Hollow-Shell row of tab:observers and
    the corner column of tab:views25d are put on the leaderboard's own noise floor:
    the facade run scores the leaderboard's builds against its reference (identical
    raw kernel distances) but drew its own corner-view floor.
    """
    import re as _re
    fac = _load(ROOT / "outputs" / "run_20261104_213620_view_facade_section"
                / "results.json")["results"]["256"]
    corner, vk = fac["corner (paper)"], fac["V-KID (77-d)"]
    lbs = Card(RUNS["1103_n256"] / "scorecard.json")
    lb_sd = lbs.sd["realism.mv_dino_kid"]
    on_lb = lambda a: (corner[a]["kid"] - corner["real_test"]["kid"]) / lb_sd
    h2h = _load(ROOT / "outputs" / "run_20261104_235615_bench_headtohead" / "results.json")
    dose = _load(DOSE / "doseresponse.json")["context"]
    rec = Card(RUNS["0925"] / "scorecard.json")
    rec_md = (RUNS["0925"] / "scorecard.md").read_text()
    nov = lambda a: rec.arms[a]["novelty"]["dino_nn_percentile"]["value"]
    vr = _load(ROOT / "outputs" / "run_20261106_193358_vkid_robustness" / "results.json")
    cc = vr["board"]["rows"]["generic: CNN w0 + neighbour counts (392)"]
    hp = _load(ROOT / "outputs" / "run_20261104_131806_bench_hollow_v1_n256"
               / "hollow.json")["pixels"]["real@solidify"]
    nscale = lambda m: (rec.sd[f"realism.{m}"], lbs.sd[f"realism.{m}"])
    return [
        Claim("Hollow-Shell MV-KID on the leaderboard floor", 0.04, lambda: on_lb("real@hollow"),
              tol=0.0051, appears_in="tab:observers, §6.3, app:views25d"),
        Claim("Fill-Interior MV-KID on the leaderboard floor (facade run kid)", 0.35,
              lambda: on_lb("real@solidify"), tol=0.0051, appears_in="app:views25d"),
        Claim("Material-Shuffle MV-KID on the leaderboard floor (facade run kid)", 422.43,
              lambda: on_lb("real_shuffled_materials"), tol=0.06, appears_in="app:views25d"),
        Claim("Monochrome MV-KID on the leaderboard floor (facade run kid)", 535.36,
              lambda: on_lb("real@monochrome"), tol=0.06, appears_in="app:views25d"),
        Claim("facade-run corner floor over the leaderboard floor", 1.22,
              lambda: corner["_sd"] / lb_sd, tol=0.0051, appears_in="app:views25d"),
        Claim("Hollow-Shell V-KID on the leaderboard floor", 265.4,
              lambda: vk["real@hollow"]["sep"], tol=0.051,
              appears_in="tab:observers, §6.3, fig:gallery, app:views25d"),
        Claim("V-KID cost, ladder pass (s)", 15.0,
              lambda: h2h["ladder"]["cost_s"]["vkid (CPU)"], tol=0.5, appears_in="§4.2, app:headtohead"),
        Claim("V-KID cost, board pass (s)", 18.0,
              lambda: h2h["board"]["cost_s"]["vkid (CPU)"], tol=0.5, appears_in="§4.2, app:headtohead"),
        Claim("head-to-head ladder reference size", 248.0,
              lambda: float(h2h["ladder"]["n_ref"]), appears_in="app:headtohead, app:nscale"),
        Claim("dose run V-KID floor at n=128", 0.0848,
              lambda: dose["noise_sd"]["geom_kid"], tol=0.00005, appears_in="app:nscale"),
        Claim("noise-floor ratio per doubling, V-KID, n=128 to 256", 0.49,
              lambda: nscale("geom_kid")[1] / nscale("geom_kid")[0], tol=0.0051,
              appears_in="tab:leaderboard caption"),
        Claim("contaminated Voxel-AR retrieval percentile, protocol run", 0.509,
              lambda: nov("native_oriented"), tol=0.0005, appears_in="app:casestudy"),
        Claim("Train-Copy memorization spreads, protocol run", 10.7,
              lambda: float(_re.search(r"train_verbatim.*?\(([+-][\d.]+) spreads\)", rec_md).group(1)),
              tol=0.051, appears_in="app:casestudy"),
        Claim("generic CNN+counts: Fill-Interior", 396.4, lambda: cc["sep"]["real@solidify"],
              tol=0.051, appears_in="tab:vkid-robust"),
        Claim("generic CNN+counts: Hollow-Shell", 230.7, lambda: cc["sep"]["real@hollow"],
              tol=0.051, appears_in="tab:vkid-robust"),
        Claim("generic CNN+counts: Patch-Stitch", 406.1, lambda: cc["sep"]["patchwork"],
              tol=0.051, appears_in="tab:vkid-robust"),
        Claim("generic CNN+counts: rho vs full V-KID", 0.78, lambda: cc["rho_full"],
              tol=0.0051, appears_in="tab:vkid-robust"),
        Claim("generic CNN+counts passes every gate", 1.0,
              lambda: float(vr["ladder"]["rows"]["generic: CNN w0 + neighbour counts (392)"]["passed"]),
              appears_in="app:vkid-robust", printed=False),
        Claim("exterior frames in the converse-probe pixel residue", 128.0,
              lambda: float(hp["n_views"]), appears_in="fig:pipeline caption"),
        Claim("fig:cutaway exterior mean pixel difference, build 249 view 1 (/255)", 0.8,
              lambda: 0.80, tol=0.051, appears_in="fig:cutaway caption",
              note="printed on the figure by scripts/figs/cutaway.py (0.80/255, CPU raster, "
                   "PYTHONHASHSEED=0); recompute by rerunning that script"),
    ]


LEAK_RUN = ROOT / "outputs" / "run_20261106_193345_leak_opaque_dose"


def added_claims_leak_postfix_20261106() -> List[Claim]:
    """Leak census, opaque-renderer MV-KID and dose curve under the current renderer
    (notes §82). Replaces the T37 (2026-09-20) census numbers 1.75% / 212/255 and the
    unbacked "89% of houses leak through a window"."""
    r = _load(LEAK_RUN / "results.json")
    H = r["houses_32"]
    c, s = H["census"], H["separation"]
    dose = H["dose"]["translucent_fixed|leak=translucent_fixed"]
    dep = H["dose"]["deployed|leak=translucent_fixed"]
    cpu = r["cpu_rasterizer_bit_identity"]
    lego = r["lego_void_20"]["separation"]
    out = [
        Claim("current renderer: Fill-Interior alters this share of the 1,596 test views", 0.909,
              lambda: c["deployed"]["solidify"]["views_altered_frac"], tol=0.006,
              appears_in="§6.2 (91%)"),
        Claim("current renderer: median share of pixels changed in an altered view", 0.0123,
              lambda: c["deployed"]["solidify"]["frac_px_changed_median_altered"], tol=0.0005,
              appears_in="§6.2 (1.23%)", note="replaces 1.75% (T37, pre-cull-fix census)"),
        Claim("current renderer: median max channel difference of an altered view", 128.0,
              lambda: c["deployed"]["solidify"]["max_ch_median_altered"], tol=0.0,
              appears_in="§6.2 (128/255)", note="replaces 212/255"),
        Claim("translucency alone (atlas fixed): share of views altered", 0.587,
              lambda: c["translucent_fixed"]["solidify"]["views_altered_frac"], tol=0.006,
              appears_in="§6.2, app:hull (59%)"),
        Claim("translucency alone (atlas fixed): share of houses with a leaking view", 0.672,
              lambda: c["translucent_fixed"]["solidify"]["houses_any_view_altered_frac"],
              tol=0.003, appears_in="§6.2 (67%)", note="replaces '89% of houses'"),
        Claim("translucency alone: median share of pixels changed in an altered view", 0.0021,
              lambda: c["translucent_fixed"]["solidify"]["frac_px_changed_median_altered"],
              tol=0.00005, appears_in="app:hull (0.21%)"),
        Claim("CPU rasterizer, glass opaque, atlas fixed: Fill-Interior views altered", 0.0,
              lambda: float(cpu["houses_32|opaque_fixed"]["solidify"]["views_altered"]), tol=0.0,
              appears_in="§6.2, app:hull", note="bit-identical in all 1,596 views; must be 0"),
        Claim("CPU rasterizer, glass opaque, atlas fixed: Hollow-Shell views altered", 0.0,
              lambda: float(cpu["houses_32|opaque_fixed"]["hollow"]["views_altered"]), tol=0.0,
              appears_in="§6.2, app:hull"),
        Claim("opaque fixed-atlas MV-KID: |Fill-Interior separation| under 1e-3 floors at n=256",
              1.0, lambda: float(abs(s["opaque_fixed"]["solidify"]["separation_floors"]) < 1e-3),
              tol=0.0, unit="holds", appears_in="§6.2"),
        Claim("opaque fixed-atlas MV-KID: |Hollow-Shell separation| under 1e-3 floors at n=256",
              1.0, lambda: float(abs(s["opaque_fixed"]["hollow"]["separation_floors"]) < 1e-3),
              tol=0.0, unit="holds", appears_in="§6.2"),
        Claim("deployed MV-KID, Fill-Interior, this run's draw, n=256", 0.29,
              lambda: s["deployed"]["solidify"]["separation_floors"], tol=0.005,
              unit="noise floors", appears_in="app:hull"),
        Claim("dose: Spearman of per-house leak vs feature shift (atlas fixed)", 0.96,
              lambda: dose["spearman_leak_vs_feature_shift"], tol=0.005, appears_in="§6.2"),
        Claim("dose: largest |separation| over leak bins, deployed renderer", 1.36,
              lambda: max(abs(b["separation_floors"]) for b in dep["bins"]), tol=0.005,
              unit="noise floors", appears_in="§6.2"),
        Claim("dose: top-quartile median leak (share of silhouette pixels)", 0.044,
              lambda: dose["bins"][-1]["leak_median"], tol=0.0005, appears_in="§6.2 (4.4%)"),
        Claim("LEGO deployed Fill-Interior residual is the atlas artefact: fixed-atlas |sep| < 1e-3",
              1.0, lambda: float(abs(lego["translucent_fixed"]["solidify"]["separation_floors"]) < 1e-3),
              tol=0.0, unit="holds", appears_in="§6.3 (LEGO)"),
    ]
    return out


def added_claims_20261104(card: "Card") -> List[Claim]:
    """Claims added with the n=256 leaderboard of record (notes §66-§71, T79-T85).

    `card` is the protocol run (record role). Everything else here reads a
    pinned artifact: the n=128 board for the comparison table, the judge, the
    3D-native runs, the BrickGPT run, the nearest-neighbour run, the dose runs.
    """
    claims: List[Claim] = []
    C = claims.append

    # Appendix tab:blindness, from the protocol run (pinned: the table is labelled as the protocol run).
    # Cut from the supplement PDF in the 2026-11-07 restructure (superseded by
    # tab:leaderboard / tab:n256); kept verbatim in SUPP_MOVED_TO_REPO.md, so the
    # values are still checked against the data but no longer looked up in the TeX.
    _MOVED_BLINDNESS = "SUPP_MOVED_TO_REPO.md (tab:blindness, cut from the supplement 2026-11-07)"
    c0925 = Card(RUNS["0925"] / "scorecard.json")
    for arm, mv, mv_ci in [("real_test", -0.002, (-0.004, 0.002)),
                           ("real@solidify", -0.001, (-0.004, 0.002)),
                           ("real_shuffled_materials", 0.667, (0.636, 0.706)),
                           ("real@monochrome", 0.859, (0.833, 0.890)),
                           ("train_verbatim", 0.017, (0.008, 0.026))]:
        C(Claim(f"tab:blindness {arm} mv_dino_kid", mv,
                lambda a=arm: c0925.realism(a, "mv_dino_kid"), tol=0.0005,
                appears_in=_MOVED_BLINDNESS, printed=False))
        for j, v in enumerate(mv_ci):
            C(Claim(f"tab:blindness {arm} mv_dino_kid ci{j}", v,
                    lambda a=arm, j=j: c0925.arms[a]["realism"]["mv_dino_kid"]["ci"][j],
                    tol=0.0005, appears_in=_MOVED_BLINDNESS, printed=False))
    for arm, gk, ci in [("real_test", -0.09, (-0.22, 0.03)),
                        ("real@solidify", 22.03, (20.65, 23.70)),
                        ("train_verbatim", 0.96, (0.60, 1.51))]:
        C(Claim(f"tab:blindness {arm} geom_kid", gk,
                lambda a=arm: c0925.realism(a, "geom_kid"), tol=0.005,
                appears_in=_MOVED_BLINDNESS, printed=False))
        for j, v in enumerate(ci):
            C(Claim(f"tab:blindness {arm} geom_kid ci{j}", v,
                    lambda a=arm, j=j: c0925.arms[a]["realism"]["geom_kid"]["ci"][j],
                    tol=0.005, appears_in=_MOVED_BLINDNESS, printed=False))
    C(Claim("verbatim copies' render difference from real, protocol run", 0.019,
            lambda: c0925.realism("train_verbatim", "mv_dino_kid")
            - c0925.realism("real_test", "mv_dino_kid"), tol=0.0005,
            appears_in="app:tables"))

    # Appendix tab:n256: the protocol run's board at n=128, PINNED to 0925 (it
    # is the comparison column and must not move with either role).
    lb128 = Leaderboard(RUNS["0925"] / "scorecard.md")
    for arm, bs in [("native_oriented", 106.87), ("canon_oriented", 136.73),
                    ("real_shuffled_materials", 231.52), ("patchwork", 246.15),
                    ("real@solidify", 201.80), ("real@monochrome", 297.95),
                    ("train_copy_noise", 302.73), ("gabled_house", 300.15),
                    ("shell_box", 476.22), ("uniform_random", 603.09)]:
        C(Claim(f"n=128 board {arm} BlockScore", bs,
                lambda a=arm: lb128.pillar(a, "blockscore"), tol=0.005,
                appears_in="app tab:n256"))
    _sd = {k: Card(RUNS[k] / "scorecard.json").sd for k in ("0925", "1103_n256")}
    for k, m, v, tol in [("0925", "geom_kid", 0.110, 0.0005),
                         ("1103_n256", "geom_kid", 0.053, 0.0005),
                         ("0925", "mv_dino_kid", 0.00289, 0.000005),
                         ("1103_n256", "mv_dino_kid", 0.00161, 0.000005)]:
        C(Claim(f"noise floor sd {m} {k}", v, lambda k=k, m=m: _sd[k][f"realism.{m}"],
                tol=tol, appears_in="app:n256, app:hull"))

    # The VLM judge (T80) and its correlation with both boards.
    _vj = _load(ROOT / "outputs" / "run_20261102_214026_vlm_judge" / "analysis.json")
    _cond = _vj["conditions"]
    for cond, rate, ci in [("interior_ext", 0.53, (0.50, 0.56)),
                           ("interior_cut", 0.70, (0.64, 0.75)),
                           ("interior_cutq", 0.625, (0.53, 0.72)),
                           ("hollow_ext", 0.52, (0.49, 0.54)),
                           ("hollow_cut", 0.73, (0.66, 0.80)),
                           ("gabled_patchwork", 0.28, (0.21, 0.35))]:
        C(Claim(f"VLM {cond} P(first)", rate,
                lambda c=cond: _cond[c]["rate_a"], tol=0.005,
                appears_in="§6.2, app:vlm"))
        for j, v in enumerate(ci):
            C(Claim(f"VLM {cond} CI{j}", v,
                    lambda c=cond, j=j: _cond[c]["cluster_boot_ci"][j], tol=0.005,
                    appears_in="§6.2, app:vlm"))
    C(Claim("VLM prefers Patch-Stitch over Gabled", 0.72,
            lambda: 1 - _cond["gabled_patchwork"]["rate_a"], tol=0.005,
            appears_in="§6.2"))
    C(Claim("VLM first-slot rate on exterior interior pairs", 0.93,
            lambda: _cond["interior_ext"]["p_first_slot"], tol=0.005,
            appears_in="app:vlm (as 93%)"))
    C(Claim("VLM order agreement on exterior interior pairs", 0.13,
            lambda: _cond["interior_ext"]["order_agreement"], tol=0.005,
            appears_in="app:vlm (as 13%)"))
    _vl = _load(ROOT / "outputs" / "figstage" / "vlm_vs_leaderboard.json")["boards"]
    for b, m, v in [("n256_1103", "mv_kid", 0.74), ("n256_1103", "blockscore", 0.54),
                    ("n256_1103", "v_kid", 0.20), ("n128_0925", "mv_kid", 0.76),
                    ("n128_0925", "blockscore", 0.57), ("n128_0925", "v_kid", 0.16)]:
        C(Claim(f"VLM Spearman vs {m}, {b}", v,
                lambda b=b, m=m: _vl[b][f"spearman_{m}"], tol=0.005,
                appears_in="app:vlm"))
    C(Claim("VLM ranks held-out real sixth", 6.0,
            lambda: float(_vl["n256_1103"]["real_rank_by_judge"]),
            appears_in="§6.2, app:vlm", printed=False))

    # 3D-native rows (T81).
    def _p3(run: str) -> dict:
        return _load(ROOT / "outputs" / run / "results.json")
    _mc0 = _p3("run_20261103_004358_mc_probe3d_s0")
    _ob0 = _p3("run_20261103_004358_mesh_probe3d_s0")
    C(Claim("Uni3D all-face KID on filled Minecraft interiors", 21.7,
            lambda: _mc0["results"]["uni3d_all_kid"]["sep"]["solidify"], tol=0.05,
            unit="noise floors", appears_in="§6.2, app:probe3d"))
    C(Claim("Objaverse strict multi-room objects", 290.0,
            lambda: float(_ob0["n_objects"]), appears_in="§6.2, app:probe3d"))

    def _vis_max() -> float:
        out = 0.0
        for run, cols in [("run_20261103_004358_mesh_probe3d_s0", ("strip", "fill")),
                          ("run_20261103_030030_mesh_probe3d_s1", ("strip", "fill")),
                          ("run_20261103_004358_mc_probe3d_s0", ("solidify", "hollow")),
                          ("run_20261103_004358_mc_probe3d_s1", ("solidify", "hollow"))]:
            for k, r in _p3(run)["results"].items():
                if "_vis_" in k and not k.endswith("textscore_drop") \
                        and not k.startswith("cd_vox"):
                    out = max(out, *(abs(r["sep"][c]) for c in cols))
        return out
    C(Claim("every vis-sampled point metric within this many floors on interior probes",
            1.6, _vis_max, tol=0.05, unit="noise floors", appears_in="§6.2, app:probe3d"))

    # BrickGPT (T83), v1 card and validity summary.
    _bg = ROOT / "outputs" / "run_20261102_220228_brickgpt_lego"
    bgc = Card(_bg / "bench_v1" / "scorecard.json")
    for arm, m, v in [("brickgpt", "mv_dino_kid", 11.77), ("brickgpt", "geom_kid", 51.48),
                      ("real@solidify", "mv_dino_kid", -0.09),
                      ("real@solidify", "geom_kid", 36.41)]:
        C(Claim(f"BrickGPT suite {arm} {m}", v, lambda a=arm, m=m: bgc.sep(a, m),
                tol=0.006, unit="noise floors", appears_in="§6.3 tab:lego, app:brickgpt"))
    _vs = _load(_bg / "validity_summary.json")
    for side, key, v in [("gen", "valid", 0.988), ("gen", "stable_of_solved", 0.958),
                         ("gen", "mean_stability", 0.995), ("gen", "min_stability", 0.896),
                         ("ref", "mean_stability", 0.997), ("ref", "min_stability", 0.953)]:
        C(Claim(f"BrickGPT validity {side} {key}", v, lambda s_=side, k=key: _vs[s_][k],
                tol=0.0006, appears_in="app tab:brickgpt-validity"))
    C(Claim("BrickGPT geometric gap over render gap (abstract)", 4.4,
            lambda: bgc.sep("brickgpt", "geom_kid") / bgc.sep("brickgpt", "mv_dino_kid"),
            tol=0.05, appears_in="abstract"))
    C(Claim("BrickGPT structures scored", 253.0,
            lambda: float(_load(_bg / "gen_summary.json")["n_scored"]),
            appears_in="§6.3, app:brickgpt"))

    # Nearest-neighbour double dissociation (T84).
    _nn = _load(ROOT / "outputs" / "run_20261103_185719_nn_geom_vs_dino" / "nn.json")
    for name, f, v in [("solidify, DINO returns own source", lambda: _nn["real@solidify"]["dino_src_top1"], 1.0),
                       ("solidify, V-KID returns own source", lambda: _nn["real@solidify"]["geom_src_top1"], 0.26),
                       ("solidify, V-KID median source rank", lambda: _nn["real@solidify"]["geom_src_rank_med"], 17.5),
                       ("material shuffle, DINO returns own source", lambda: _nn["real_shuffled_materials"]["dino_src_top1"], 0.26),
                       ("material shuffle, V-KID returns own source", lambda: _nn["real_shuffled_materials"]["geom_src_top1"], 1.0),
                       ("real builds, same NN in both spaces", lambda: _nn["real_test"]["nn_same_house"], 0.039)]:
        C(Claim(f"NN: {name}", v, f, tol=0.005, appears_in="§6.4, fig:nn"))

    # Case-study table cells (protocol run, record role).
    for arm, gk in [("canon_unoriented", 19.570), ("patchwork", 19.976),
                    ("uniform_random", 37.729), ("real_test", -0.095),
                    ("train_verbatim", 0.960)]:
        C(Claim(f"case study raw V-KID {arm}", gk,
                lambda a=arm: card.realism(a, "geom_kid"), tol=0.0005,
                appears_in="app tab:casestudy"))
    C(Claim("protocol run, clean Voxel-AR raw render metric", 0.245,
            lambda: card.realism("canon_oriented", "mv_dino_kid"), tol=0.0005,
            appears_in="app:n256"))
    C(Claim("protocol run, clean Voxel-AR raw V-KID", 14.90,
            lambda: card.realism("canon_oriented", "geom_kid"), tol=0.005,
            appears_in="app:n256"))

    # Dream-Cubed yield and sealed interior (smoke run counts, leaderboard scalars).
    _smk = Card(ROOT / "outputs" / "run_20261102_233900_bench_dreamcubed-smoke"
                / "scorecard.json")
    C(Claim("Dream-Cubed extracted builds (yield)", 284.0,
            lambda: float(_smk.arms["dreamcubed"]["meta"]["n"]),
            appears_in="§6.4, app:dreamcubed"))
    C(Claim("Dream-Cubed chunks generated", 288.0,
            lambda: float(_smk.arms["dreamcubed_raw"]["meta"]["n"]),
            appears_in="§6.4, app:dreamcubed"))
    _lb1103 = Card(RUNS["1103_n256"] / "scorecard.json")
    C(Claim("Dream-Cubed sealed interior ratio", 0.016,
            lambda: _lb1103.arms["dreamcubed"]["geometry_scalars"]["interior_ratio_sealed"]["gen"]["mean"],
            tol=0.0005, appears_in="app:dreamcubed"))
    C(Claim("real sealed interior ratio (leaderboard reference)", 0.114,
            lambda: _lb1103.arms["dreamcubed"]["geometry_scalars"]["interior_ratio_sealed"]["real"]["mean"],
            tol=0.0005, appears_in="app:dreamcubed"))
    C(Claim("reference median blocks", 675.0,
            lambda: _load(ROOT / "outputs" / "figstage" / "dreamcubed_size.json")["ref_median_blocks"],
            tol=0.5, appears_in="app:dreamcubed"))

    # tab:seven (LEGO in-domain ladder; the only ladder of that run, see fig6).
    _l7 = _load(ROOT / "outputs" / "run_20261104_212706_bench_lego_ladder_postfix"
                / "ladder" / "dinov2b__v4_az45-135-225-315_el30-30-30-30_px224_ortho.json")
    for m, row in [("kid", (2.0, 12.3, 132.3)), ("mmd_rbf", (2.1, 13.9, 148.8)),
                   ("fd", (0.5, 10.8, 72.4)), ("one_minus_coverage", (0.2, 7.7, 19.8)),
                   ("one_minus_density", (0.1, 7.2, 14.0)),
                   ("one_minus_precision", (-0.4, 9.0, 28.0)),
                   ("one_minus_recall", (0.4, 1.4, 30.7))]:
        for rung, v in zip(("solidify", "chunk_delete_20", "material_shuffle"), row):
            C(Claim(f"tab:seven {m} @ {rung}", v,
                    lambda m=m, r=rung: (_l7["scores"][m][r] - _l7["noise_sd"][m]["mean"])
                    / _l7["noise_sd"][m]["sd"], tol=0.051, appears_in="app tab:seven"))

    # The n-scale decomposition: converse-probe separation over the protocol run's.
    C(Claim("separation ratio, dose run over protocol run", 1.42,
            lambda: _load(DOSE / "doseresponse.json")["axes"]["solidify"]["geom_kid"]["range_sd"]
            / card.sep("real@solidify", "geom_kid"), tol=0.005, appears_in="app:nscale"))

    # tab:probe3d, every printed cell at both seeds (pinned runs).
    _P3 = {"obj": ("run_20261103_004358_mesh_probe3d_s0", "run_20261103_030030_mesh_probe3d_s1",
                   ("strip", "fill", "noise", "wall", "decim")),
           "mc": ("run_20261103_004358_mc_probe3d_s0", "run_20261103_004358_mc_probe3d_s1",
                  ("solidify", "hollow", "canon16", "chunk_delete_40", "material_shuffle"))}
    for dom, (r0, r1, cols) in _P3.items():
        for si, run in enumerate((r0, r1)):
            res = _p3(run)["results"]
            for key in res:
                enc = key.split("_")[0]
                if not (key in ("dino4_kid", "dino_cut4_kid", "vkid_geom")
                        or (enc in ("uni3d", "ulip2") and key.endswith("_kid")
                            and "_vox" not in key)
                        or (enc in ("cd", "emd") and key.endswith("_mmd")
                            and key.split("_")[1] in ("all", "vis", "cut"))):
                    continue
                for c in cols:
                    v = round(res[key]["sep"][c], 1)
                    C(Claim(f"tab:probe3d {dom} s{si} {key} @ {c}", v,
                            lambda r=res, k=key, c=c: r[k]["sep"][c], tol=0.051,
                            appears_in="app tab:probe3d"))

    # Dose runs: the numbers the dose prose and tab:dose print (pinned files).
    d128 = _load(DOSE / "doseresponse.json")["axes"]
    d64 = _load(DOSE_N64 / "doseresponse.json")["axes"]
    for nm, d, ax, m, v, tol in [
            ("occ noise, render range n=128", d128, "occ_noise", "mv_dino_kid", 185.0, 0.5),
            ("mixture, render range n=128", d128, "mixture_uniform_random", "mv_dino_kid", 549.0, 0.5),
            ("occ noise, geometry range n=128", d128, "occ_noise", "geom_kid", 455.0, 0.5),
            ("mixture, geometry range n=128", d128, "mixture_uniform_random", "geom_kid", 444.0, 0.5),
            ("occ noise, render range n=64", d64, "occ_noise", "mv_dino_kid", 143.0, 0.5),
            ("mixture, render range n=64", d64, "mixture_uniform_random", "mv_dino_kid", 392.0, 0.5),
            ("occ noise, geometry range n=64", d64, "occ_noise", "geom_kid", 226.0, 0.5),
            ("mixture, geometry range n=64", d64, "mixture_uniform_random", "geom_kid", 211.0, 0.5)]:
        C(Claim(f"dose: {nm}", v, lambda d=d, ax=ax, m=m: d[ax][m]["range_sd"], tol=tol,
                unit="noise floors", appears_in="§6.1, app tab:dose"))
    C(Claim("dose: render Spearman on the interior axis, n=128", 0.700,
            lambda: d128["solidify"]["mv_dino_kid"]["spearman"], tol=0.0005,
            appears_in="app tab:dose"))
    C(Claim("dose: render Spearman on the interior axis, n=64", 0.700,
            lambda: d64["solidify"]["mv_dino_kid"]["spearman"], tol=0.0005,
            appears_in="app tab:dose"))
    for nm, d, v in [("n=64", d64, 0.00438), ("n=128", d128, 0.00316)]:
        C(Claim(f"dose: render noise floor {nm}", v,
                lambda d=d: d["solidify"]["mv_dino_kid"]["noise_sd"], tol=0.000005,
                appears_in="app:nscale"))
    mix = _load(DOSE / "doseresponse.json")["axes"]["mixture_uniform_random"]["mv_dino_kid"]["value"]
    C(Claim("mixture: 25%-bad arm raw render value", 0.105, lambda: mix[1], tol=0.0005,
            appears_in="app:failrate"))
    C(Claim("mixture: fully-bad arm raw render value", 1.740, lambda: mix[-1], tol=0.0005,
            appears_in="app:failrate"))

    # The rescore run's own floor, beside the protocol run's.
    _rs = _load(ROOT / "outputs" / "figstage" / "rescore_flat_opaque.json")["pipelines"]
    C(Claim("rescore: deployed renderer floor", 0.00337,
            lambda: _rs["paper"]["noise_floor_sd"], tol=0.000005, appears_in="app:hull"))
    return claims


#: The leaderboard re-scored with the within-set near-twin gate (notes §78) and
#: the vanilla-template control row scored through the GPU path (notes §78).
TWIN_RUN = ROOT / "outputs" / "run_20261104_214146_bench_twin_gate_rescore"
DC_AUDIT = ROOT / "outputs" / "analysis" / "dreamcubed_audit"
VANILLA_RUN = ROOT / "outputs" / "run_20261104_213125_bench_vanilla_template_control"


def added_claims_twin_gate() -> List[Claim]:
    """The self-duplication gate and the Dream-Cubed audit (notes §73, §78)."""
    tw = _load(TWIN_RUN / "twins.json")
    pa, gate, null = tw["per_arm"], tw["gate"], tw["null"]
    out = []
    for arm, v, tol, where in [
            ("dreamcubed", 0.598, 0.0005, "§6.4, Table 1, app:dreamcubed (as %)"),
            ("real_test", 0.008, 0.0005, "§6.4, Table 1, app:dreamcubed (as %)"),
            ("real@solidify", 0.066, 0.0005, "app:dreamcubed"),
            ("shell_box", 0.246, 0.0005, "app:dreamcubed"),
            ("gabled_house", 0.207, 0.0005, "app:dreamcubed")]:
        out.append(Claim(f"within-set near-twin rate {arm}", v,
                         lambda a=arm: pa[a]["twin_rate"], tol=tol,
                         appears_in=where))
    for arm in ("canon_oriented", "canon_unoriented", "native_oriented"):
        out.append(Claim(f"within-set near-twin rate {arm} is zero", 0.0,
                         lambda a=arm: pa[a]["twin_rate"], appears_in="app:dreamcubed",
                         printed=False))
    for arm, v in [("dreamcubed", 0.85), ("real_test", 0.42)]:
        out.append(Claim(f"median nearest-twin IoU {arm}", v,
                         lambda a=arm: pa[a]["nn_iou_median"], tol=0.005,
                         appears_in="app:dreamcubed"))
    out.append(Claim("near-twin gate bar at n=256", 0.068, lambda: gate["bar"],
                     tol=0.0005, appears_in="app:dreamcubed"))
    for part, v in [("test", 0.006), ("val", 0.008), ("train", 0.019)]:
        out.append(Claim(f"near-twin null mean, {part} draws", v,
                         lambda p=part: null[p]["mean"], tol=0.0005,
                         appears_in="app:dreamcubed"))
    st = tw["status"]
    for arm, v in [("dreamcubed", 29.5), ("shell_box", 11.9), ("gabled_house", 10.0)]:
        out.append(Claim(f"near-twin gate DQ spreads {arm}", v,
                         lambda a=arm: float(st[a]["disqualified"].rsplit("(", 1)[1]
                                             .split(" ")[0]), tol=0.05,
                         appears_in="Table 1, app:dreamcubed"))
    # The vanilla-template control, GPU render path, same protocol and noise floors.
    vc = _load(VANILLA_RUN / "control_ungated.json")
    vcard = _load(VANILLA_RUN / "scorecard.json")
    vh, va = vc["vanilla_village_houses"], vc["vanilla_village_all"]
    out += [
        Claim("vanilla templates (houses) ungated BlockScore", 202.95,
              lambda: vh["blockscore_ungated"], tol=0.005, appears_in="Table 1, §6.4, app:dreamcubed"),
        Claim("vanilla templates (houses) appearance", 110.21,
              lambda: vh["pillars"]["appearance"], tol=0.005, appears_in="Table 1"),
        Claim("vanilla templates (all village) ungated BlockScore", 194.98,
              lambda: va["blockscore_ungated"], tol=0.005, appears_in="app:dreamcubed"),
        Claim("vanilla templates (houses) twin rate", 0.672,
              lambda: vh["twin_rate"], tol=0.0005, appears_in="Table 1, app:dreamcubed"),
        Claim("vanilla templates (houses) DQ spreads", 33.2,
              lambda: float(vh["gated_status"]["disqualified"].rsplit("(", 1)[1].split(" ")[0]),
              tol=0.05, appears_in="app:dreamcubed"),
        Claim("vanilla templates (houses) raw mv_dino_kid", 0.178,
              lambda: vcard["arms"]["vanilla_village_houses"]["realism"]["mv_dino_kid"]["value"],
              tol=0.0005, appears_in="Table 1"),
        Claim("vanilla templates (houses) raw geom_kid", 10.67,
              lambda: vcard["arms"]["vanilla_village_houses"]["realism"]["geom_kid"]["value"],
              tol=0.005, appears_in="Table 1"),
        Claim("vanilla control run reproduces Dream-Cubed BlockScore", 165.71,
              lambda: vc["dreamcubed"]["blockscore_ungated"], tol=0.005,
              appears_in="app:dreamcubed", printed=False),
        Claim("vanilla house templates surviving extraction", 174.0,
              lambda: float(vh["n"]), appears_in="Table 1, app:dreamcubed"),
    ]
    # The peer audit's numbers (outputs/analysis/dreamcubed_audit, 284 builds).
    sm = _load(DC_AUDIT / "summary.json")
    q3 = _load(DC_AUDIT / "q3.json")["dreamcubed"]["kind_frac"]
    out += [
        Claim("Dream-Cubed median IoU to village templates", 0.52,
              lambda: sm["dc_vs_village_occ_iou"]["median"], tol=0.005,
              appears_in="app:dreamcubed"),
        Claim("held-out real median IoU to village templates", 0.32,
              lambda: sm["real_vs_village_occ_iou"]["median"], tol=0.005,
              appears_in="app:dreamcubed"),
        Claim("Dream-Cubed houses with a room", 0.45,
              lambda: q3["house (room)"], tol=0.005, appears_in="§6.4, app:dreamcubed"),
        Claim("Dream-Cubed roofless enclosures", 0.36,
              lambda: q3["roofless enclosure"], tol=0.005,
              appears_in="§6.4, app:dreamcubed"),
        Claim("Dream-Cubed fragments", 0.19,
              lambda: q3["fragment/other"], tol=0.005, appears_in="§6.4, app:dreamcubed"),
        Claim("Dream-Cubed size-matched near-twin rate", 0.73,
              lambda: _load(DC_AUDIT / "size_control.json")["dc"]["frac_nn_iou>=0.8"],
              tol=0.005, appears_in="app:dreamcubed"),
    ]
    return out


def added_claims_views25d() -> List[Claim]:
    """Supplement tab:views25d: facade and cross-section views at n=256 (notes §76, T90)."""
    j = _load(ROOT / "outputs" / "run_20261104_213620_view_facade_section" / "results.json")
    r, rho = j["results"]["256"], j["spearman"]["256"]
    P = ("corner (paper)", "A-sides", "A-render", "A-2.5D", "B-mid", "B-9", "B-cut")
    T = {"real@solidify": (0.3, 4.4, 3.6, 0, 71.1, 67.5, 14.3),
         "real@hollow": (0.0, 0.8, 0.6, 0, 39.7, 42.8, 5.3),
         "real_shuffled_materials": (346, 548, 610, 355, 226, 292, 415),
         "real@monochrome": (439, 1146, 1309, 314, 1080, 1314, 925),
         "real@rot90": (-0.2, 0.0, 0.1, 0.1, 0.4, 0.5, -0.2),
         "real@mirror": (-0.2, 0.1, 0.1, 0.0, -0.4, -0.3, 0.0)}
    out = []
    for arm, row in T.items():
        for prot, want in zip(P, row):
            tol = 0.51 if abs(want) >= 100 else 0.051
            out.append(Claim(f"tab:views25d {arm} {prot}", float(want),
                             lambda a=arm, q=prot: r[q][a]["sep"], tol=tol,
                             unit="noise floors", appears_in="app:views25d"))
    for prot, vc, vv in [("A-sides", 0.97, 0.10), ("A-2.5D", 0.96, 0.28),
                         ("B-mid", 0.93, 0.07), ("B-cut", 0.95, 0.05)]:
        out.append(Claim(f"tab:views25d rho vs corner {prot}", vc,
                         lambda q=prot: rho[q]["vs_corner"], tol=0.0051, appears_in="app:views25d"))
        out.append(Claim(f"tab:views25d rho vs vkid {prot}", vv,
                         lambda q=prot: rho[q]["vs_vkid"], tol=0.0051, appears_in="app:views25d"))
    return out



def added_claims_vkid_robust() -> List[Claim]:
    """§6 descriptor sentence and app:vkid-robust (notes §82, T105)."""
    j = _load(ROOT / "outputs" / "run_20261106_193358_vkid_robustness" / "results.json")
    meta, B, L = j["row_meta"], j["board"]["rows"], j["ladder"]["rows"]
    rows = lambda g: [k for k, m in meta.items() if m["group"] == g]
    sep = lambda k, a: B[k]["sep"][a]
    R = j["summary_random"]
    out = [
        Claim("every leave-one-family-out V-KID passes all gates", 1.0,
              lambda: float(all(L[k]["passed"] for k in rows("lofo"))), tol=0.0, appears_in="§6, app:vkid-robust"),
        Claim("random 50% subsets passing all gates (of 20)", 20,
              lambda: R["rand50"]["n_pass"], tol=0.0, appears_in="§6, app:vkid-robust"),
        Claim("random 25% subsets passing all gates (of 20)", 17,
              lambda: R["rand25"]["n_pass"], tol=0.0, appears_in="app:vkid-robust"),
        Claim("every generic descriptor passes all gates", 1.0,
              lambda: float(all(L[k]["passed"] for k in rows("generic"))), tol=0.0, appears_in="§6"),
        Claim("generic descriptors: min solidify sep", 257,
              lambda: min(sep(k, "real@solidify") for k in rows("generic")), tol=0.51,
              unit="noise floors", appears_in="§6"),
        Claim("generic descriptors: max solidify sep", 437,
              lambda: max(sep(k, "real@solidify") for k in rows("generic")), tol=0.51,
              unit="noise floors", appears_in="§6"),
        Claim("LOFO solidify min", 310, lambda: min(sep(k, "real@solidify") for k in rows("lofo") if "+" not in k),
              tol=0.51, unit="noise floors", appears_in="app:vkid-robust"),
        Claim("LOFO solidify max", 457, lambda: max(sep(k, "real@solidify") for k in rows("lofo") if "+" not in k),
              tol=0.51, unit="noise floors", appears_in="app:vkid-robust"),
        Claim("LOFO hollow min", 146, lambda: min(sep(k, "real@hollow") for k in rows("lofo") if "+" not in k),
              tol=0.51, unit="noise floors", appears_in="app:vkid-robust"),
        Claim("rand50 solidify median", 351, lambda: R["rand50"]["fill_board"][0], tol=0.51, appears_in="app:vkid-robust"),
        Claim("rand50 solidify min", 283, lambda: R["rand50"]["fill_board"][1], tol=0.51, appears_in="app:vkid-robust"),
        Claim("rand50 hollow median", 221, lambda: R["rand50"]["hollow_board"][0], tol=0.51, appears_in="app:vkid-robust"),
        Claim("rand50 hollow min", 133, lambda: R["rand50"]["hollow_board"][1], tol=0.51, appears_in="app:vkid-robust"),
        Claim("rand25 solidify min", 179, lambda: R["rand25"]["fill_board"][1], tol=0.51, appears_in="app:vkid-robust"),
    ]
    return out


def added_claims_rankflips() -> List[Claim]:
    """Abstract, §6.4 and app:rankflips: render vs geometric ranking of 40 generators (notes §83, T106)."""
    import ast
    j = _load(ROOT / "outputs" / "run_20261106_193513_render_blind_rank_flips" / "results.json")
    g = j["sets"]["generators"]
    rho = ast.literal_eval(g["spearman_mv_v1"]) if isinstance(g["spearman_mv_v1"], str) else g["spearman_mv_v1"]
    pr = ast.literal_eval(g["pairs_v1"]) if isinstance(g["pairs_v1"], str) else g["pairs_v1"]
    A = j["arms"]
    gens = g["arms"]
    ceil = A["real@facades6"]
    def rank(key, val):
        return 1 + sum(A[x][key] < val for x in gens)
    return [
        Claim("generators pooled", 40, lambda: len(gens), tol=0, appears_in="abstract, §6.4"),
        Claim("Spearman MV-KID vs V-KID over 40 generators", 0.50, lambda: rho["rho"], tol=0.0051, appears_in="abstract, §6.4"),
        Claim("facade ceiling rank under MV-KID (of 41)", 1, lambda: rank("mv_z", ceil["mv_z"]), tol=0, appears_in="§6.4"),
        Claim("facade ceiling rank under V-KID (of 41)", 19, lambda: rank("v1_z", ceil["v1_z"]), tol=0, appears_in="§6.4"),
    ]


def added_claims_effect_sizes() -> List[Claim]:
    """Effect sizes sqrt(MMD^2) relative to an anchor arm (notes §88, T115): §6.4 and abstract."""
    j = _load(ROOT / "outputs" / "run_20261108_205428_stats_fixes" / "results.json")["effects"]
    cm, P, B = j["cross_metric"], j["probes"], j["brickgpt_lego"]["brickgpt"]
    return [
        Claim("Fill-Interior effect size, appearance", 0.03, lambda: cm["Fill-Interior|Random-Voxels"]["e_mvkid"], tol=0.0051, appears_in="§6.4"),
        Claim("fresh-real 95th pct effect size, appearance", 0.03, lambda: P["Fill-Interior"]["mv"]["e_Random-Voxels_null95"], tol=0.0051, appears_in="§6.4"),
        Claim("Fill-Interior effect size, geometry", 0.78, lambda: cm["Fill-Interior|Random-Voxels"]["e_vkid"], tol=0.0051, appears_in="§6.4"),
        Claim("Patch-Stitch effect size, geometry", 0.71, lambda: cm["Patch-Stitch|Random-Voxels"]["e_vkid"], tol=0.0051, appears_in="§6.4"),
        Claim("BrickGPT geometric/render effect-size ratio", 2.2, lambda: B["ratio_e_v_over_mv"], tol=0.051, appears_in="abstract"),
    ]

def added_claims_unregistered_20261104() -> List[Claim]:
    """Main-text numbers the audit listed as unregistered on 2026-11-04."""
    import re as _re
    out = []
    ic = _load(ROOT / "outputs" / "figstage" / "interior_counts.json")["rows"]["draw_200"]
    out.append(Claim("solidify alters this share of examined builds", 0.94,
                     lambda: ic["altered_frac"], tol=0.005,
                     appears_in="§1 (printed 94%), app:interior"))
    lego = Card(ROOT / "outputs" / "run_20260925_005517_bench_lego_interior_blindness_replication"
                / "scorecard.json")
    out.append(Claim("LEGO render metric on monochrome", 685.8,
                     lambda: lego.sep("real@monochrome", "mv_dino_kid"), tol=0.06,
                     unit="noise floors", appears_in="§6.3 text and tab:lego",
                     note="post-fix LEGO run; identical to the 0918 run (wool is opaque)"))
    out.append(Claim("test-split views in the leak count", 1596.0,
                     lambda: 4.0 * len(__import__("blockgen.eval.bench.splits", fromlist=["x"])
                                       .load_split("houses_32", 0).test),
                     tol=0.0, appears_in="§6.2", printed=True,
                     note="399 held-out builds x 4 views"))
    ri = _load(TWIN_RUN / "rank_intervals.json")
    out.append(Claim("rank-interval subsamples", 1000.0, lambda: float(ri["n_boot"]),
                     tol=0.0, appears_in="Table 1 caption"))
    rows = {r["arm"]: r for r in _load(TWIN_RUN / "scorecard.json")["run"]["blockscore"]}
    def _spreads(arm):
        m = _re.search(r"\(([+-][0-9.]+) spreads\)", rows[arm]["disqualified"] or "")
        return float(m.group(1)) if m else float("nan")
    for arm, v in [("real@single_mode", 44.9), ("train_verbatim", 10.1),
                   ("dreamcubed_raw", 24.0)]:
        out.append(Claim(f"DQ spreads {arm}", v, lambda a=arm: _spreads(a), tol=0.051,
                         unit="spreads", appears_in="Table 1 DQ block"))
    return out


def added_claims_hollow_n256() -> List[Claim]:
    """The converse-probe rerun at n=256 under the post-fix renderer (notes §74).

    Replaces the unregistered pre-fix run_20260918_203947_bench_hollow. Pinned:
    the separations are this run's own (separation above its real_test draw).
    """
    h = _load(ROOT / "outputs" / "run_20261104_131806_bench_hollow_v1_n256" / "hollow.json")["realism"]
    out = []
    for arm, m, v, tol in [("real@hollow", "mv_dino_kid", -0.02, 0.005),
                           ("real@solidify", "mv_dino_kid", 0.21, 0.005),
                           ("real@hollow", "geom_kid", 231.1, 0.05),
                           ("real@solidify", "geom_kid", 401.7, 0.05),
                           ("real@hollow", "geom_kid", 231.0, 0.6),
                           ("real@solidify", "geom_kid", 402.0, 0.6)]:
        out.append(Claim(f"hollow run n=256 {arm} {m}", v,
                         lambda a=arm, m=m: h[m][a]["sep_from_real_sd"], tol=tol,
                         unit="noise floors", appears_in="§6.3, fig:gallery"))
    # tab:ablation, the post-fix view-protocol ablation (the run fig5/fig9/pipeline read).
    ab = {r["condition"]: r for r in _load(
        ROOT / "outputs" / "run_20260925_003516_view_protocol_ablation" / "ablation.json")["rows"]}
    for cond, v in [("dinov2b / 4 ortho (paper)", 0.2), ("clipB16 / 4 ortho", 0.1),
                    ("clipB / 4 ortho", 0.1), ("resnet50 / 4 ortho", 0.0),
                    ("inceptionv3 / 4 ortho", 0.2), ("dinov2b / 8 ortho", 0.2),
                    ("dinov2b / 16 ortho", 0.1), ("dinov2b / 4 perspective", 0.1),
                    ("cut: facing quadrant", 6.7), ("cut: vertical sections", 14.0),
                    ("cut: roof off (top half)", 17.1), ("cut: floorplan slabs", 55.8),
                    ("geom_kid (voxel descriptor)", 120.8)]:
        out.append(Claim(f"tab:ablation {cond}", v,
                         lambda c=cond: ab[c]["separation_sd"], tol=0.051,
                         unit="noise floors", appears_in="§6.1-6.2, app tab:ablation"))
    out.append(Claim("ablation: max exterior condition", 0.2,
                     lambda: max(r["separation_sd"] for r in ab.values()
                                 if not r["sees_interior"]), tol=0.051,
                     appears_in="§6.2, tab:ablation caption"))
    for nm, v, f in [("min cutaway", 6.7, min), ("max cutaway", 55.8, max)]:
        out.append(Claim(f"ablation: {nm}", v,
                         lambda f=f: f(r["separation_sd"] for r in ab.values()
                                       if r["family"] == "D cutaway"), tol=0.051,
                         appears_in="§6.2"))
    return out


def corpus_claims() -> List[Claim]:
    """Appendix Table `app:corpus`, cell by cell, from the cache manifests.

    The printed table was derived under a *three*-source assumption after a
    fourth (`text2mc_schem`) was added to the 48^3 build, so `pooled`, `drops`,
    `dups` and `final` were each short by that source's contribution. The
    identity claims below catch exactly that: a per-source column that stops
    summing to `final` means a source is missing from the table.
    """
    claims: List[Claim] = []
    C = claims.append
    #: (cache, pooled, drops, dups, final, {source: final count})
    SPEC = [
        ("houses_32", 3493, 826, 6, 2661,
         {"grabcraft": 1360, "3dcraft": 1267, "text2mc": 34}),
        ("houses_48", 4502, 1381, 33, 3088,
         {"grabcraft": 1410, "3dcraft": 1381, "text2mc": 180,
          "text2mc_schem": 117}),
    ]

    def report(cache: str) -> dict:
        return _load(CACHE / f"{cache}_manifest.json")["report"]

    for cache, pooled, drops, dups, final, by_src in SPEC:
        where = f"appendix table app:corpus, {cache} row"
        # The houses_48 row left the supplement in the 2026-11-07 restructure (no
        # reported run uses that cache); it is kept in SUPP_MOVED_TO_REPO.md.
        moved = cache == "houses_48"
        if moved:
            where = "SUPP_MOVED_TO_REPO.md (houses_48 row, cut from app:corpus 2026-11-07)"
        C(Claim(f"{cache}: pooled", float(pooled),
                lambda c=cache: float(report(c)["pooled"]), appears_in=where))
        C(Claim(f"{cache}: quality drops", float(drops),
                lambda c=cache: float(sum(report(c)["quality_drops"].values())),
                appears_in=where,
                note="the table prints the TOTAL; the per-reason breakdown in "
                     "the prose must sum to it"))
        C(Claim(f"{cache}: exact duplicates", float(dups),
                lambda c=cache: float(report(c)["exact_duplicates"]),
                appears_in=where))
        C(Claim(f"{cache}: final", float(final),
                lambda c=cache: float(report(c)["final"]), appears_in=where))
        C(Claim(f"{cache}: final = pooled - drops - dups", 0.0,
                lambda c=cache: float(
                    report(c)["pooled"] - sum(report(c)["quality_drops"].values())
                    - report(c)["exact_duplicates"] - report(c)["final"]),
                appears_in=where, note="arithmetic the printed table must satisfy"))
        C(Claim(f"{cache}: per-source columns sum to final", 0.0,
                lambda c=cache: float(sum(report(c)["final_by_corpus"].values())
                                      - report(c)["final"]),
                appears_in=where,
                note="fires if a source exists in the manifest but has no "
                     "column -- the defect that produced the 2,971 row"))
        C(Claim(f"{cache}: number of sources", float(len(by_src)),
                lambda c=cache: float(len(report(c)["final_by_corpus"])),
                appears_in=where + ", and the 'four public sources' prose"))
        for src, n in by_src.items():
            C(Claim(f"{cache}: {src} (post-gate, post-dedup)", float(n),
                    lambda c=cache, s=src: float(report(c)["final_by_corpus"][s]),
                    appears_in=where))
    for c in claims:
        if c.name.startswith("houses_48:"):
            c.printed = False
    return claims


def ladder_claims() -> List[Claim]:
    """Appendix Table `tab:ladder-render`, cell by cell, from the ladder artifact.

    Registered exhaustively because the printed table was a stale copy in which
    *every* value differed from the artifact while the gate column had been
    partly refreshed by hand -- the exact failure a spot check misses. The gate
    verdicts are registered as 1.0/0.0 so that a metric silently changing side
    fails the audit; `one_minus_recall` was printed as PASS while the artifact
    failed it on G8.
    """
    d = _load(LADDER)
    S, G, N = d["scores"], d["gates"], d["noise_sd"]
    #: metric -> (real, noise_10, occ_noise_10, chunk_delete_40,
    #:            material_shuffle, rot90_1, sd), rounded as printed
    RUNGS = ("real_heldout", "noise_10", "occ_noise_10", "chunk_delete_40",
             "material_shuffle", "rot90_1")
    TABLE = {
        "kid": (0.011, 0.165, 0.435, 0.106, 0.762, 0.010, 0.008),
        "mmd_rbf": (3.5, 49.6, 137.4, 32.2, 218.0, 3.0, 2.4),
        "one_minus_recall": (0.137, 0.331, 0.899, 0.371, 0.790, 0.105, 0.028),
        "fd": (0.107, 0.157, 0.250, 0.147, 0.373, 0.106, 0.005),
        "one_minus_coverage": (0.246, 0.508, 0.835, 0.488, 0.948, 0.242, 0.057),
        "one_minus_density": (0.000, 0.247, 0.637, 0.166, 0.897, 0.000, 0.073),
        "one_minus_precision": (0.125, 0.234, 0.234, 0.109, 0.641, 0.125, 0.036),
        "legacy_cmmd": (0.079, 0.468, 1.148, 0.317, 1.987, 0.075, 0.021),
    }
    #: which metrics the table prints as PASS. Only these two may rank anything.
    PASSING = {"kid", "mmd_rbf"}
    claims: List[Claim] = []
    C = claims.append
    for metric, row in TABLE.items():
        # Half of the last printed digit, plus float slack: MMD-RBF is printed
        # to one decimal, everything else to three. A tolerance of exactly half
        # the digit rejects an honestly-rounded 0.1875 -> 0.188.
        tol = 0.051 if metric == "mmd_rbf" else 0.00051
        for rung, want in zip(RUNGS, row):
            C(Claim(f"ladder/{metric} @ {rung}", want,
                    lambda m=metric, r=rung: S[m][r], tol=tol,
                    appears_in="appendix table tab:ladder-render"))
        C(Claim(f"ladder/{metric} noise floor sd", row[-1],
                lambda m=metric: N[m]["sd"], tol=tol,
                appears_in="appendix table tab:ladder-render"))
        C(Claim(f"ladder/{metric} passes every blocking gate",
                float(metric in PASSING),
                lambda m=metric: float(all(G[m].values())),
                appears_in="appendix table tab:ladder-render, §1, §3",
                note="1.0 means the table prints PASS. `one_minus_recall` was "
                     "printed PASS while failing G8_n_stability"))
    C(Claim("exactly two render-space metrics clear every gate", 2.0,
            lambda: float(len(d["passing"])),
            appears_in="§3 claim (3), §1 contribution list"))
    return claims


def baseline_claims() -> List[Claim]:
    """Appendix Table `tab:baselines`, recomputed from the generators themselves.

    Slower than the other claims (it builds 128 structures per arm and scores
    399 real ones), which is the price of a table with no artifact behind it --
    the baselines are built in memory and have no npz, so the code IS the
    provenance.

    The `interior` column previously mixed two statistics: `gabled_house` was
    quoted on the aperture-aware counter and `real` on the sealed one, making an
    empty shell with a doorway look like a furnished house. Both are registered
    separately here so that can never be collapsed into one column again.
    """
    from blockgen.eval.bench import baselines as bl, geometry, splits, topology
    import numpy as np

    cache: dict = {}

    def stats(arm: str) -> dict:
        if not cache:
            split = splits.load_split("houses_32")
            train = splits.split_structures(split, "train")
            built = bl.arms(train, 128, seed=0)
            built["real"] = splits.split_structures(split, "test")
            for name, structs in built.items():
                co = [topology.coherence(s) for s in structs]
                ge = [geometry.geometry_stats(s) for s in structs]
                cache[name] = {
                    "blocks": float(np.mean([c.n_blocks for c in co])),
                    "wall_frac": float(np.mean([g.wall_frac for g in ge])),
                    "sealed": float(np.mean([g.interior_ratio_sealed for g in ge])),
                    "open": float(np.mean([g.interior_ratio_open for g in ge])),
                    "lcc": float(np.mean([c.lcc_ratio for c in co])),
                }
        return cache[arm]

    #: arm -> (blocks, wall_frac, interior_sealed, interior_open, lcc)
    TABLE = {
        "uniform_random":   (1383, 0.049, 0.002, 0.003, 0.406),
        "shell_box":        (1345, 1.000, 0.609, 0.609, 1.000),
        "patchwork":        (1405, 0.624, 0.003, 0.009, 0.866),
        "gabled_house":     (1139, 0.969, 0.000, 0.562, 1.000),
        "train_copy_noise": (1057, 0.454, 0.002, 0.055, 0.967),
        "real":             ( 924, 0.745, 0.114, 0.129, 0.978),
    }
    COLS = ("blocks", "wall_frac", "sealed", "open", "lcc")
    claims: List[Claim] = []
    C = claims.append
    for arm, row in TABLE.items():
        for col, want in zip(COLS, row):
            C(Claim(f"baseline/{arm} {col}", float(want),
                    lambda a=arm, c=col: stats(a)[c],
                    tol=0.5 if col == "blocks" else 0.0005,
                    appears_in="appendix table tab:baselines"))
    C(Claim("gabled_house is an empty shell: sealed interior is exactly 0", 0.0,
            lambda: stats("gabled_house")["sealed"], tol=1e-12,
            appears_in="appendix table tab:baselines, §5, §7",
            note="it has a doorway, so NOTHING is sealed. Quoting its open "
                 "interior against real's sealed one is the error this catches"))
    return claims




# ---------------------------------------------------------------------------
# The TeX cross-check
#
# Everything above checks the registry against the data. Nothing above reads
# the paper, which is how the ICLR draft came to print the leaderboard of
# run_20260917 (+0.38, +220.83) while this registry was pinned to the re-scored
# run_20260925 (+0.14, +231.52) without a single FAIL. The functions below close
# that loop: every numeral the paper prints is parsed, and each one is either a
# registered value at its printed precision, a value of the same quantity under
# a DIFFERENT run (stale), a near-miss of a registered value (probably stale),
# or unregistered.
# ---------------------------------------------------------------------------

#: Integers that are protocol constants or prose, never claims: sample sizes,
#: grid resolutions. Keep this SHORT and never park a superseded value here --
#: an earlier version whitelisted 0.022 and 0.119, the two wrong numbers an
#: audit later found in the intro, which is exactly the failure this stops.
PROSE_INTS = {16, 32, 48, 64, 128, 256, 512}

#: Sections whose numerals must all be registered (the "unregistered" report).
HEADLINE_STEMS = ("abstract", "intro", "result", "conclusion")

#: command -> brace groups whose content is not prose (-1: all adjacent ones).
#: `com`, `TODO` and `UNMEASURED` are author notes: blanked whole, so a number
#: inside a margin note is never read as a claim the paper makes.
_MASK_ARGS = {
    **{k: 1 for k in ("ref", "cref", "Cref", "autoref", "eqref", "pageref",
                      "label", "cite", "citep", "citet", "citealp", "citeauthor",
                      "citeyear", "nocite", "includegraphics", "input", "include",
                      "url", "href", "vspace", "hspace", "end", "cline",
                      "scalebox", "raisebox", "textcolor", "color", "colorbox",
                      "cellcolor", "rowcolor", "bibliography",
                      "bibliographystyle", "hyperref", "linethickness")},
    **{k: 2 for k in ("multicolumn", "multirow", "resizebox", "fontsize")},
    **{k: -1 for k in ("begin", "cmidrule", "setlength", "addtolength", "rule",
                       "newcommand", "renewcommand", "providecommand",
                       "definecolor", "setcounter", "addtocounter", "usepackage",
                       "documentclass", "hypersetup", "graphicspath",
                       "titlespacing", "com", "TODO", "UNMEASURED")},
}
_DIM_RE = re.compile(
    r"-?(?:\d+\.?\d*|\.\d+)\s*(?:(?:pt|em|ex|cm|mm|bp|sp)(?![A-Za-z])"
    r"|\\(?:line|text|column|paper)(?:width|height))|-?(?:\d+\.?\d*|\.\d+)in(?![A-Za-z])")
_NUM_RE = re.compile(
    r"(?<![A-Za-z_\d.\\])(?:\d{1,3}(?:(?:\{,\}|\\,|,)\d{3})+(?:\.\d+)?(?!\d)"
    r"|\d+(?:\.\d+)?|\.\d+)")
_PCT_RE = re.compile(r"\s*\}?\s*\$?\s*(?:\\,|\\!|~)?\s*\\%")


def _group_end(s: str, j: int) -> Optional[int]:
    """Index just past the group opening at s[j] ('{', '[' or '(')."""
    close = {"{": "}", "[": "]", "(": ")"}[s[j]]
    depth, i = 0, j
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if (close == "}" and depth == 0) or (close != "}" and depth == 0 and ch == close):
            return i + 1
        i += 1
    return None


def mask_tex(text: str) -> str:
    """Blank (with spaces, newlines kept) everything that is not printed prose.

    Comments, author notes, cross-references, citations, lengths and table
    column specs all carry digits that are not claims. Offsets and line
    numbers are preserved so a hit can be reported as file:line.
    """
    chars = list(text)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if chars[k] != "\n":
                chars[k] = " "

    for m in re.finditer(r"(?<!\\)%[^\n]*", text):
        blank(*m.span())
    s = "".join(chars)
    for m in re.finditer(r"\\([A-Za-z]+)\*?", s):
        n = _MASK_ARGS.get(m.group(1))
        if n is None:
            continue
        i, groups = m.end(), 0
        while i < len(s):
            ch = s[i]
            if ch == "[" or (ch == "(" and m.group(1) == "cmidrule"):
                end = _group_end(s, i)
            elif ch == "{" and (n < 0 or groups < n):
                end = _group_end(s, i)
                groups += 1
            else:
                break
            if end is None:
                break
            i = end
        blank(m.start(), i)
    s = "".join(chars)
    for m in _DIM_RE.finditer(s):
        blank(*m.span())
    return "".join(chars)


@dataclass
class Num:
    """One numeral as the paper prints it."""
    value: float      # sign dropped; a percentage is divided by 100
    decimals: int     # decimal places of `value` (a percentage gains two)
    sig: int          # significant digits printed (trailing integer zeros excluded)
    raw: str
    file: str = ""
    line: int = 0
    percent: bool = False

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"

    @property
    def half(self) -> float:
        """Half the last printed unit: the window an honest rounding lands in."""
        h = 0.5 * 10.0 ** (-self.decimals)
        if self.decimals == 0 and not self.percent and self.value >= 100:
            digits = self.raw.replace(",", "")
            k = len(digits) - len(digits.rstrip("0"))
            if k:  # "530" may be 531 rounded to tens; never more than 5% off
                h = min(0.5 * 10.0 ** k, 0.05 * self.value)
        return h

    def matches(self, v: Optional[float]) -> bool:
        if v is None or v != v:
            return False
        return abs(abs(v) - self.value) <= self.half + 1e-9 * max(1.0, abs(v))

    @property
    def is_prose(self) -> bool:
        """Small integers, protocol constants and years: never claims."""
        if self.percent or self.decimals:
            return False
        v = int(self.value)
        return v <= 10 or v in PROSE_INTS or 1900 <= v <= 2100


def parse_numbers(text: str, file: str = "") -> List[Num]:
    """Every printed numeral in a TeX source, with its precision."""
    s = mask_tex(text)
    out: List[Num] = []
    for m in _NUM_RE.finditer(s):
        a, b = m.span()
        if a >= 2 and s[a - 1] == "-" and s[a - 2].isalpha():
            continue                      # ResNet-50, Fig-3: a name, not a value
        if b < len(s) and s[b].isalpha() and s[b] != "x":
            continue                      # 3D, 2AFC, 8B: a name, not a value
        raw = re.sub(r"\{,\}|\\,|,", "", m.group(0))
        pct = bool(_PCT_RE.match(s, b))
        intpart, _, frac = raw.partition(".")
        digits = (intpart + frac).lstrip("0")
        if not frac:
            digits = digits.rstrip("0")
        dec = len(frac) + (2 if pct else 0)
        val = float(raw) / (100.0 if pct else 1.0)
        out.append(Num(round(val, dec + 2), dec, len(digits), raw, file,
                       s.count("\n", 0, a) + 1, pct))
    return out


_INPUT_RE = re.compile(r"\\appendix\b|\\(?:input|include)\{([^}]+)\}")
_SKIP_INPUTS = {"math_commands", "preamble"}


def paper_files(paper: Path, appendix: bool = True) -> List[tuple]:
    """(path, is_appendix) for every section the paper inputs, in order.

    Follows `\\input`/`\\include` from main.tex (and a separate supplementary
    document if the venue keeps one); everything after `\\appendix`, or in the
    supplement, is appendix.
    """
    out: List[tuple] = []
    seen: set = set()

    def walk(f: Path, in_app: bool, depth: int) -> bool:
        text = re.sub(r"(?<!\\)%[^\n]*", "", f.read_text(errors="ignore"))
        for m in _INPUT_RE.finditer(text):
            if m.group(1) is None:
                in_app = True
                continue
            name = m.group(1).strip()
            p = paper / (name if name.endswith(".tex") else name + ".tex")
            if p.stem in _SKIP_INPUTS or not p.exists() or p in seen:
                continue
            seen.add(p)
            out.append((p, in_app))
            if depth < 3:
                in_app = walk(p, in_app, depth + 1)
        return in_app

    main = paper / "main.tex"
    if main.exists():
        walk(main, False, 0)
    for supp in ("supp.tex", "suppl.tex", "supplementary.tex"):
        if (paper / supp).exists():
            walk(paper / supp, True, 0)
    if not out:          # no main.tex yet: fall back to the section files
        for p in sorted((paper / "sec").glob("*.tex")):
            out.append((p, p.stem.startswith("X_") or "supp" in p.stem
                        or "appendix" in p.stem))
    return [(p, a) for p, a in out if appendix or not a]


def leaderboard_cells(bench: Path) -> dict:
    """Every number of the recorded leaderboard, registered or not.

    Lets the cross-check recognise an unregistered table cell (e.g. a
    submission's appearance pillar) as stale when it equals another run's
    value, which a registry of hand-picked cells cannot do.
    """
    try:
        lb = Leaderboard(bench / "scorecard.md")
    except (OSError, ValueError):
        return {}
    cells = {}
    for arm, row in lb.rows.items():
        for col, v in row.items():
            if col == "worst":
                m = re.search(r"\(\+?(-?\d+\.?\d*) spreads\)", v or "")
                if m:
                    cells[f"leaderboard {arm}/DQ spreads"] = float(m.group(1))
            elif v is not None:
                cells[f"leaderboard {arm}/{col}"] = v
    return cells


def _decimals(v: float) -> int:
    t = f"{v:g}"
    return len(t.partition(".")[2]) if "e" not in t else 6


def _trivial(v: Optional[float]) -> bool:
    return v is None or v != v or abs(v) in (0.0, 1.0)


@dataclass
class TexReport:
    files: List[str]
    found: dict           # claim name -> [file:line, ...]
    missing: List[str]    # printable claims the TeX never prints
    stale: List[dict]
    near: List[dict]
    unregistered: List[dict]


def tex_crosscheck(claims: List[Claim], got: dict, paper: Path,
                   alt: Optional[dict] = None, cells: Optional[dict] = None,
                   appendix: bool = True, near_rel: float = 0.05) -> TexReport:
    """Read the paper and classify every numeral it prints.

    `got` maps claim name -> value computed under the run of record, `alt`
    maps another run's key -> {name: value} for the run-dependent claims and
    leaderboard cells, and `cells` holds the record run's leaderboard cells.
    A numeral is:

    * **found** if it equals a registered claim (computed or registered value)
      at the precision it is printed to;
    * **stale** if it instead equals the same quantity under another run;
    * a **near-miss** if it is within `near_rel` of a registered value at a
      comparable precision -- most often a stale number of unknown origin;
    * **unregistered** if it is in a headline section and matches nothing.
    """
    alt, cells = alt or {}, cells or {}
    record: dict = dict(cells)
    targets: List[tuple] = []          # (name, value, printed decimals)
    by_name = {c.name: c for c in claims}
    for c in claims:
        if not c.printed:
            continue
        g = got.get(c.name)
        # A run-dependent claim's registered value is pinned to ONE run; under
        # any other run of record it is that run's (stale) value, so only the
        # computed value may count as printed correctly.
        vals = {g} if (c.run_dependent and g is not None) else {g, c.paper_value}
        for v in vals:
            if not _trivial(v):
                targets.append((c.name, v, _decimals(c.paper_value)))
        g = got.get(c.name)
        record[c.name] = g if g is not None and g == g else c.paper_value
    cell_targets = [(k, v, 2) for k, v in cells.items() if not _trivial(v)]
    #: printed precision per quantity: a claim's registered value, else the
    #: leaderboard's (two decimals; one for the DQ spreads)
    dec = {c.name: _decimals(c.paper_value) for c in claims}
    for vals in [cells, *alt.values()]:
        for k in vals:
            dec.setdefault(k, 1 if k.endswith("DQ spreads") else 2)

    files = paper_files(paper, appendix)
    found: dict = {}
    stale, near, unreg = [], [], []
    seen_at: set = set()
    for path, in_app in files:
        rel = str(path.relative_to(paper))
        headline = (not in_app) and any(h in path.stem for h in HEADLINE_STEMS)
        for n in parse_numbers(path.read_text(errors="ignore"), rel):
            if n.sig == 0:
                continue                          # 0, 0.00: never informative
            hits = [name for name, v, _ in targets if n.matches(v)]
            for name in dict.fromkeys(hits):
                found.setdefault(name, []).append(n.where)
            if hits:
                continue
            # a 2-digit or percentage hit on a table cell is chance, not a cell
            is_cell = (n.sig >= 3 and not n.percent
                       and any(n.matches(v) for _, v, _ in cell_targets))
            key_ = (n.where, n.raw)
            if key_ in seen_at:                 # BlockScore repeats its worst pillar
                continue
            seen_at.add(key_)
            reported = False
            if not reported and not is_cell and n.sig >= 2 and not n.is_prose:
                for key, vals in alt.items():
                    for name, v in vals.items():
                        if _trivial(v) or not n.matches(v):
                            continue
                        if n.percent and name.startswith("leaderboard "):
                            continue      # no pillar is ever printed as a percent
                        rv = record.get(name)
                        if rv is not None and n.matches(rv):
                            continue
                        if n.sig < 3 and (rv is None or n.decimals < dec.get(name, 2)):
                            continue      # a 2-digit hit at coarser precision is chance
                        stale.append({"where": n.where, "printed": n.raw
                                      + ("%" if n.percent else ""),
                                      "name": name, "record": rv,
                                      "alt_run": key, "alt": v,
                                      "registered": name in by_name})
                        reported = True
                        break
                    if reported:
                        break
            if not reported and not is_cell and n.sig >= 3 and not n.is_prose:
                best = None
                for name, v, d in targets:
                    if n.decimals != d or abs(v) < 1e-6:
                        continue
                    r = abs(n.value - abs(v)) / abs(v)
                    if r <= near_rel and (best is None or r < best[0]):
                        best = (r, name, v)
                if best:
                    near.append({"where": n.where, "printed": n.raw
                                 + ("%" if n.percent else ""), "name": best[1],
                                 "value": best[2], "rel": best[0]})
                    reported = True
            if headline and not reported and not n.is_prose:
                unreg.append({"where": n.where, "printed": n.raw
                              + ("%" if n.percent else ""),
                              "note": "recorded leaderboard cell, not registered"
                              if is_cell else ""})
    # A near-miss of a claim the paper never prints correctly anywhere is the
    # strong signal (the stale copy is the only copy); one printed correctly
    # elsewhere is usually a different quantity of similar size.
    for nm in near:
        nm["claim_printed_elsewhere"] = nm["name"] in found
    missing = [c.name for c in claims
               if c.printed and c.name not in found
               and not all(_trivial(v) for v in (got.get(c.name), c.paper_value))]
    return TexReport([str(p.relative_to(paper)) for p, _ in files],
                     found, missing, stale, near, unreg)


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def evaluate(claims: List[Claim], only_run_dependent: bool = False) -> dict:
    """name -> (ok, got, err) for every claim."""
    return {c.name: c.check() for c in claims
            if c.run_dependent or not only_run_dependent}


def audit(run: str, paper: Path, appendix: bool = True,
          leaderboard_run: Optional[str] = None) -> dict:
    """Registry check + TeX cross-check under one run of record per role.

    `run` binds the record-role claims, `leaderboard_run` (default
    LEADERBOARD_RUN) the leaderboard-role claims and the leaderboard cells.
    Every other registered run supplies the "alternative" values a printed
    number is stale against: for each role, only runs other than that role's
    own run count, so a leaderboard number is never stale merely for equalling
    the protocol run's value of a different quantity.
    """
    bench = resolve_run(run)
    lbench = resolve_run(leaderboard_run or LEADERBOARD_RUN)
    claims = build_claims(bench, lbench)
    results = evaluate(claims)
    got = {k: g for k, (ok, g, err) in results.items() if err is None}
    own = {"record": bench.resolve(), "leaderboard": lbench.resolve()}
    alt = {}
    for key, other in RUNS.items():
        if not (other / "scorecard.json").exists():
            continue
        o = other.resolve()
        if o == own["record"] and o == own["leaderboard"]:
            continue
        try:
            other_claims = build_claims(other, other)
        except (KeyError, OSError, ValueError):
            continue
        res = evaluate(other_claims, only_run_dependent=True)
        vals = {c.name: res[c.name][1] for c in other_claims
                if c.run_dependent and res[c.name][2] is None
                and own[c.role] != o}
        if o != own["leaderboard"]:
            vals.update(leaderboard_cells(other))
        if vals:
            alt[key] = vals
    tex = tex_crosscheck(claims, got, paper, alt, leaderboard_cells(lbench),
                         appendix=appendix)
    return {"run": run_key(bench), "bench": bench.name,
            "lb_run": run_key(lbench), "lb_bench": lbench.name,
            "paper": str(paper), "claims": claims, "results": results, "tex": tex}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def print_registry(rep: dict, verbose: bool) -> tuple:
    claims, results = rep["claims"], rep["results"]
    print(f"\n  auditing {len(claims)} registered claims against run of record "
          f"{rep['run']} (protocol) and {rep['lb_run']} (leaderboard)\n"
          f"  {DIM}{rep['bench']}\n  {rep['lb_bench']}{RESET}\n")
    failed = corpus = 0
    for c in claims:
        ok, got, err = results[c.name]
        if ok:
            mark, colour = "OK  ", GREEN
        elif needs_corpus(err):
            mark, colour = "CORP", YELLOW
            corpus += 1
        else:
            mark, colour = "FAIL", RED
            failed += 1
        if ok and not verbose:
            continue
        unit = f" {c.unit}" if c.unit else ""
        line = f"  {colour}{mark}{RESET}  {c.name}"
        if mark == "CORP":
            line += f"  {DIM}paper {c.paper_value:g}{unit}; needs the corpus on disk{RESET}"
        elif not ok:
            line += f"  {RED}paper {c.paper_value:g}{unit} vs computed {got:g}{unit}{RESET}"
            if err:
                line += f"  {DIM}({err}){RESET}"
        else:
            line += f"  {DIM}= {got:g}{unit} · {c.appears_in}{RESET}"
        print(line)
        if verbose and c.note:
            print(f"        {DIM}{c.note}{RESET}")
    if not verbose:
        print(f"  {DIM}{len(claims) - failed - corpus} OK rows hidden; "
              f"--verbose shows them{RESET}")
    return failed, corpus


def _fmt(v) -> str:
    return "absent" if v is None else f"{v:g}"


def print_tex(rep: dict, verbose: bool, unregistered: bool) -> None:
    tex: TexReport = rep["tex"]
    printable = sum(1 for c in rep["claims"] if c.printed
                    and not all(_trivial(v) for v in
                                (rep["results"][c.name][1], c.paper_value)))
    print(f"\n  TeX cross-check — {_rel(Path(rep['paper']))} · {len(tex.files)} "
          f"files · run of record {rep['run']}\n")
    for s in tex.stale:
        tag = "" if s["registered"] else f" {DIM}(unregistered cell){RESET}"
        print(f"  {RED}STALE{RESET} {s['where']}  prints {s['printed']} — "
              f"{s['name']} is {_fmt(s['record'])} under the run of record; "
              f"{s['printed']} is its {s['alt_run']} value{tag}")
    strong = [s for s in tex.near if not s["claim_printed_elsewhere"]]
    weak = [s for s in tex.near if s["claim_printed_elsewhere"]]
    for s in strong + (weak if verbose else []):
        also = f" {DIM}(printed correctly elsewhere){RESET}" \
            if s["claim_printed_elsewhere"] else ""
        print(f"  {YELLOW}NEAR {RESET} {s['where']}  prints {s['printed']} — "
              f"{s['name']} = {s['value']:g} ({100 * s['rel']:.1f}% off){also}")
    if weak and not verbose:
        print(f"  {DIM}{len(weak)} weaker near-miss(es) of claims printed correctly "
              f"elsewhere; --verbose lists them{RESET}")
    if not tex.stale and not strong:
        print(f"  {GREEN}no stale or near-miss numbers{RESET}")
    print(f"\n  {len(tex.found)} of {printable} printable claims found in the TeX; "
          f"{len(tex.missing)} not found"
          + ("" if verbose else f" {DIM}(--verbose lists them){RESET}"))
    if verbose:
        for name in tex.missing:
            print(f"         {DIM}{name}{RESET}")
    print(f"  {YELLOW if tex.unregistered else GREEN}{len(tex.unregistered)}"
          f"{RESET} unregistered numbers in "
          f"{'/'.join(HEADLINE_STEMS)}"
          + ("" if unregistered else f" {DIM}(--unregistered lists them){RESET}"))
    if unregistered:
        for u in tex.unregistered:
            note = f"  {DIM}{u['note']}{RESET}" if u["note"] else ""
            print(f"         {u['where']}  {u['printed']}{note}")
        print(f"  {DIM}a warning, not a failure: register each, or confirm it is prose{RESET}")


def report_json(rep: dict) -> dict:
    tex: TexReport = rep["tex"]
    return {
        "run": rep["run"], "bench": rep["bench"], "paper": rep["paper"],
        "lb_run": rep["lb_run"], "lb_bench": rep["lb_bench"],
        "claims": [{"name": c.name, "paper_value": c.paper_value,
                    "computed": None if g != g else g, "ok": ok, "error": err,
                    "needs_corpus": needs_corpus(err),
                    "run_dependent": c.run_dependent, "role": c.role,
                    "tex": tex.found.get(c.name, [])}
                   for c in rep["claims"]
                   for ok, g, err in [rep["results"][c.name]]],
        "failed": sum(1 for c in rep["claims"] if not rep["results"][c.name][0]
                      and not needs_corpus(rep["results"][c.name][2])),
        "tex": {"files": tex.files, "missing": tex.missing, "stale": tex.stale,
                "near": tex.near, "unregistered": tex.unregistered},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--paper", type=Path, default=None,
                    help="paper directory (default: deliverables/paper/cvpr27 "
                         "if it exists, else deliverables/paper/iclr)")
    ap.add_argument("--run", default=RUN_OF_RECORD,
                    help=f"run of record: one of {', '.join(RUNS)}, or a run "
                         f"directory (default {RUN_OF_RECORD})")
    ap.add_argument("--leaderboard-run", default=LEADERBOARD_RUN,
                    help=f"leaderboard run: one of {', '.join(RUNS)}, or a run "
                         f"directory (default {LEADERBOARD_RUN})")
    ap.add_argument("--no-appendix", action="store_true",
                    help="cross-check the main sections only")
    ap.add_argument("--no-tex", action="store_true",
                    help="skip the TeX cross-check (registry only)")
    ap.add_argument("--unregistered", action="store_true",
                    help="list numerals in the headline sections that match no claim")
    ap.add_argument("--compare-runs", action="store_true",
                    help="audit under every run in RUNS and summarise what changes")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write a machine-readable report here")
    args = ap.parse_args()
    paper = resolve_paper(args.paper)
    if not args.no_tex and not paper.is_dir():
        raise SystemExit(f"no paper directory at {paper}")

    if args.compare_runs:
        return compare_runs(paper, not args.no_appendix, args.run)

    rep = audit(args.run, paper, appendix=not args.no_appendix,
                leaderboard_run=args.leaderboard_run)
    failed, corpus = print_registry(rep, args.verbose)
    if not args.no_tex:
        print_tex(rep, args.verbose, args.unregistered)
    if args.json:
        args.json.write_text(json.dumps(report_json(rep), indent=1))

    claims = rep["claims"]
    stale = 0 if args.no_tex else len(rep["tex"].stale)
    print()
    if corpus:
        print(f"  {YELLOW}{corpus} claim(s) read the corpus, which is not on disk;"
              f" rebuild it to check them (see README).{RESET}")
    if failed:
        print(f"  {RED}{failed} claim(s) do not match the data.{RESET}")
    if stale:
        print(f"  {RED}{stale} number(s) in the TeX belong to a different run "
              f"than the run of record ({rep['run']}).{RESET}")
    if not failed and not stale:
        checked = len(claims) - corpus
        print(f"  {GREEN}Every checked claim ({checked} of {len(claims)}) matches "
              f"the run of record.{RESET}")
    print()
    return 1 if failed or stale else 0


def compare_runs(paper: Path, appendix: bool, record: str = RUN_OF_RECORD) -> int:
    """What the paper would have to change under each candidate LEADERBOARD run.

    The protocol run of record (`--run`) is held fixed; only the leaderboard
    role is swapped, since that is the choice the paper's table depends on.
    """
    print(f"\n  comparing leaderboard runs against {_rel(paper)} "
          f"(protocol run held at {record})\n")
    worst = 0
    for key in RUNS:
        rep = audit(record, paper, appendix, leaderboard_run=key)
        res, tex = rep["results"], rep["tex"]
        fails = [c for c in rep["claims"] if not res[c.name][0]
                 and not needs_corpus(res[c.name][2])]
        star = " (current LEADERBOARD_RUN)" if key == LEADERBOARD_RUN else ""
        print(f"  {key}{star}  {DIM}{rep['lb_bench']}{RESET}")
        print(f"    registry: {RED if fails else GREEN}{len(fails)} claim(s) "
              f"disagree with this run{RESET}")
        for c in fails[:12]:
            print(f"      {c.name}: registered {c.paper_value:g}, "
                  f"this run {res[c.name][1]:g}")
        print(f"    TeX: {RED if tex.stale else GREEN}{len(tex.stale)} stale{RESET}, "
              f"{len(tex.near)} near-miss, {len(tex.found)} claims found, "
              f"{len(tex.unregistered)} unregistered headline numbers")
        for s in tex.stale[:12]:
            print(f"      {s['where']}  {s['printed']}  ({s['name']}: "
                  f"{_fmt(s['record'])} here, {s['alt']:g} in {s['alt_run']})")
        if len(tex.stale) > 12:
            print(f"      ... and {len(tex.stale) - 12} more")
        print()
        worst = max(worst, int(bool(fails or tex.stale)))
    return worst


STATS_RUN = ROOT / "outputs" / "run_20261108_205428_stats_fixes"
FUNC_RUN = ROOT / "outputs" / "run_20261108_204723_functional_groundtruth"
REBASE_RUN = ROOT / "outputs" / "run_20261109_152704_rankflips_functional_rebase"
FLIP_RUN = ROOT / "outputs" / "run_20261106_193513_render_blind_rank_flips"


def added_claims_rankflips_exact() -> List[Claim]:
    """app:rankflips, tab:rankflips: the 40-generator comparison under the exact
    label-permutation test, Holm over 780 per metric (notes §88, T115)."""
    j = _load(STATS_RUN / "rankflips_exact.json")
    S = j["sets"]
    g, g256, ar, fac = S["generators"], S["generators_n256"], S["voxel_ar"], S["facade"]
    sp = g["spearman_v1"]
    rs = S["rank_shift_vs_hidden_blocks"]
    out = [
        Claim("exact: generators pooled", 40, lambda: g["k"], tol=0, appears_in="app:rankflips"),
        Claim("exact: arm pairs tested", 780, lambda: g["n_pairs"], tol=0, appears_in="app:rankflips"),
        Claim("exact: permutations per pair", 20000, lambda: j["n_perm"], tol=0, appears_in="app:rankflips"),
        Claim("exact: Spearman MV-KID vs V-KID, 40 generators", 0.50, lambda: sp["rho"], tol=0.0051,
              appears_in="tab:rankflips"),
        Claim("exact: Spearman half-sampling CI low", 0.37, lambda: sp["ci_sampling_halfsample"][0],
              tol=0.0051, appears_in="tab:rankflips", note="replaces 0.42 (subsampling)"),
        Claim("exact: Spearman half-sampling CI high", 0.58, lambda: sp["ci_sampling_halfsample"][1],
              tol=0.0051, appears_in="tab:rankflips", note="replaces 0.54"),
        Claim("exact: pairs separated by both metrics", 308, lambda: g["pairs_v1"]["both_separated"],
              tol=0, appears_in="tab:rankflips", note="replaces 388"),
        Claim("exact: opposite-order pairs", 40, lambda: g["pairs_v1"]["significant_flips"], tol=0,
              appears_in="tab:rankflips, app:functional, limitation line", note="replaces 76"),
        Claim("exact: n=256 Spearman", 0.60, lambda: g256["spearman_v1"]["rho"], tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: n=256 Spearman CI low", 0.46, lambda: g256["spearman_v1"]["ci_sampling_halfsample"][0],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: n=256 Spearman CI high", 0.67, lambda: g256["spearman_v1"]["ci_sampling_halfsample"][1],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: n=256 arms", 35, lambda: g256["k"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: n=256 both separated", 255, lambda: g256["pairs_v1"]["both_separated"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: n=256 opposite", 24, lambda: g256["pairs_v1"]["significant_flips"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: AR voxel arms", 24, lambda: ar["k"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: AR voxel Spearman", 0.37, lambda: ar["spearman_v1"]["rho"], tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: AR voxel Spearman CI low", 0.16, lambda: ar["spearman_v1"]["ci_sampling_halfsample"][0],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: AR voxel Spearman CI high", 0.51, lambda: ar["spearman_v1"]["ci_sampling_halfsample"][1],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: AR voxel both separated", 58, lambda: ar["pairs_v1"]["both_separated"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: AR voxel opposite", 7, lambda: ar["pairs_v1"]["significant_flips"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: facade Spearman", 0.99, lambda: fac["spearman_v1"]["rho"], tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: facade Spearman CI low", 0.88, lambda: fac["spearman_v1"]["ci_sampling_halfsample"][0],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: facade Spearman CI high", 1.00, lambda: fac["spearman_v1"]["ci_sampling_halfsample"][1],
              tol=0.0051, appears_in="tab:rankflips"),
        Claim("exact: facade both separated", 32, lambda: fac["pairs_v1"]["both_separated"], tol=0, appears_in="tab:rankflips"),
        Claim("exact: facade opposite", 0, lambda: fac["pairs_v1"]["significant_flips"], tol=0,
              appears_in="tab:rankflips", printed=False),
        Claim("exact: Spearman generator-bootstrap CI low", 0.20, lambda: sp["ci_generator_bootstrap"][0],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: Spearman generator-bootstrap CI high", 0.72, lambda: sp["ci_generator_bootstrap"][1],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: Spearman family-cluster CI low", 0.02, lambda: sp["ci_family_cluster_bootstrap"][0],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: Spearman family-cluster CI high", 0.91, lambda: sp["ci_family_cluster_bootstrap"][1],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: Spearman without the facade family", 0.28, lambda: sp["leave_one_family_out"]["facade"],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: Spearman without the voxel family", 0.72, lambda: sp["leave_one_family_out"]["voxel_ar"],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: reversed pairs where MV-KID prefers a facade arm", 22,
              lambda: g["pairs_v1"]["mv_prefers_family_in_flips"]["facade"], tol=0, appears_in="app:rankflips",
              note="replaces 38 of 76"),
        Claim("exact: rank shift vs hidden-block share, Spearman", -0.26, lambda: rs["spearman"], tol=0.0051,
              appears_in="app:rankflips"),
        Claim("exact: rank shift vs hidden-block share, p", 0.10, lambda: rs["p"], tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: 80-d descriptor, opposite pairs", 29, lambda: g["pairs_v2"]["significant_flips"], tol=0,
              appears_in="app:rankflips", note="replaces 46 of 405"),
        Claim("exact: 80-d descriptor, both separated", 308, lambda: g["pairs_v2"]["both_separated"], tol=0,
              appears_in="app:rankflips"),
        Claim("exact: 80-d descriptor Spearman", 0.65, lambda: g["spearman_v2"]["rho"], tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: 80-d Spearman CI low", 0.54, lambda: g["spearman_v2"]["ci_sampling_halfsample"][0],
              tol=0.0051, appears_in="app:rankflips"),
        Claim("exact: 80-d Spearman CI high", 0.71, lambda: g["spearman_v2"]["ci_sampling_halfsample"][1],
              tol=0.0051, appears_in="app:rankflips"),
    ]
    return out


def added_claims_functional_rebase() -> List[Claim]:
    """app:functional, tab:functional, app:rankflips design decisions and worked examples,
    and the main-text limitation sentence (notes §91, T123; FD from notes §87, T114)."""
    r = _load(REBASE_RUN / "results.json")
    f = _load(FUNC_RUN / "results.json")
    A, RA = f["arms"], f["rank_agreement"]["fd"]["generators"]
    P = r["reversed_pairs"]
    fd, nf, fc = P["fd"]["all"], P["fd"]["non_facade_pairs"], P["fd"]["facade_pairs"]
    sp, ent, dec, ex = r["spearman_fd"], r["enterability"], r["decisions"]["summary"]["v1"], r["worked_examples"]
    ind = f["independence_from_vkid"]["per_build"]
    r2 = [v["cv_r2_linear_on_vkid_v1_features"] for v in ind.values()]
    FB = {r["arm"]: r for r in f["fd_board"]}
    out_tab = [
        Claim("functional table: Fill-Interior FD", 1.21, lambda: FB["real@solidify"]["fd"], tol=0.0051, appears_in="tab:functional"),
        Claim("functional table: Fill-Interior FD CI low", 1.11, lambda: FB["real@solidify"]["fd_ci"][0], tol=0.0051, appears_in="tab:functional"),
        Claim("functional table: best voxel model FD", 1.32, lambda: FB["m2b_20m_sftmix_s1"]["fd"], tol=0.0051, appears_in="tab:functional"),
        Claim("functional table: Fill-Interior FD CI high", 1.32, lambda: FB["real@solidify"]["fd_ci"][1], tol=0.0051, appears_in="tab:functional"),
        Claim("functional table: best voxel model FD CI low", 1.20, lambda: FB["m2b_20m_sftmix_s1"]["fd_ci"][0], tol=0.0051, appears_in="tab:functional"),
        Claim("functional table: best voxel model FD CI high", 1.43, lambda: FB["m2b_20m_sftmix_s1"]["fd_ci"][1], tol=0.0051, appears_in="tab:functional"),
    ]
    out = [
        # pair split on the 40 exact-test reversals
        Claim("functional: reversed pairs, walk-through sides with MV-KID", 33, lambda: fd["sides_mv_kid"], tol=0,
              appears_in="app:functional, tab:functional, limitation line", note="replaces 60 of 76"),
        Claim("functional: reversed pairs, walk-through sides with V-KID", 7, lambda: fd["sides_v_kid"], tol=0,
              appears_in="app:functional, tab:functional", note="replaces 16 of 76"),
        Claim("functional: V-KID share (%)", 18, lambda: 100 * fd["frac_v_kid"], tol=0.51, appears_in="app:functional"),
        Claim("functional: V-KID share CP low (%)", 7, lambda: 100 * fd["frac_v_kid_ci"][0], tol=0.51, appears_in="app:functional"),
        Claim("functional: V-KID share CP high (%)", 33, lambda: 100 * fd["frac_v_kid_ci"][1], tol=0.51, appears_in="app:functional"),
        Claim("functional: every FD difference on the 40 pairs is significant", 40, lambda: fd["fd_significant"],
              tol=0, appears_in="app:functional", printed=False),
        Claim("functional: non-facade reversed pairs", 18, lambda: nf["n_pairs"], tol=0, appears_in="app:functional"),
        Claim("functional: non-facade pairs siding with V-KID", 0, lambda: nf["sides_v_kid"], tol=0,
              appears_in="app:functional", note="replaces 1 of 38", printed=False),
        Claim("functional: non-facade CP high (%)", 19, lambda: 100 * nf["frac_v_kid_ci"][1], tol=0.51, appears_in="app:functional"),
        Claim("functional: facade reversed pairs", 22, lambda: fc["n_pairs"], tol=0, appears_in="app:functional"),
        Claim("functional: facade pairs siding with V-KID", 7, lambda: fc["sides_v_kid"], tol=0, appears_in="app:functional"),
        Claim("functional: facade pairs where V-KID's pick is BrickNet and FD agrees", 5,
              lambda: r["by_family"]["MV-KID picks facade, V-KID picks external"]["fd_sides_v_kid"], tol=0,
              appears_in="app:functional"),
        Claim("functional: BrickNet enterable share (%)", 0.8, lambda: 100 * A["bricknet_mc_sft_base"]["means"]["enterable"],
              tol=0.051, appears_in="app:functional"),
        Claim("functional: noise-unit FD variant, V-KID side", 2, lambda: P["fd_noise"]["all"]["sides_v_kid"], tol=0, appears_in="app:functional"),
        Claim("functional: KID FD variant, V-KID side", 9, lambda: P["fd_kid"]["all"]["sides_v_kid"], tol=0, appears_in="app:functional"),
        Claim("functional: occupancy-only FD variant, V-KID side", 8, lambda: P["fd_occ"]["all"]["sides_v_kid"], tol=0, appears_in="app:functional"),
        Claim("functional: occupancy-only, non-facade V-KID side", 0, lambda: P["fd_occ"]["non_facade_pairs"]["sides_v_kid"],
              tol=0, appears_in="app:functional", printed=False),
        Claim("functional: leave-one-generator-out min (%)", 6, lambda: 100 * r["leave_one_generator_out"]["min"], tol=0.51, appears_in="app:functional"),
        Claim("functional: leave-one-generator-out max (%)", 24, lambda: 100 * r["leave_one_generator_out"]["max"], tol=0.51, appears_in="app:functional"),
        Claim("functional: enterable/floor/rooms each side with V-KID on 0 of 40", 0,
              lambda: max(r["per_measure"][m]["sides_v_kid"] for m in ("enterable", "floor_area", "rooms_reached")),
              tol=0, appears_in="app:functional", printed=False),
        # rank agreement over 40 generators
        Claim("functional: Spearman FD vs MV-KID", 0.67, lambda: RA["rho_mv"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: Spearman FD vs MV-KID CI low", 0.41, lambda: RA["rho_mv_ci"][0], tol=0.0051, appears_in="tab:functional"),
        Claim("functional: Spearman FD vs MV-KID CI high", 0.82, lambda: RA["rho_mv_ci"][1], tol=0.0051, appears_in="tab:functional"),
        Claim("functional: Spearman FD vs V-KID", 0.16, lambda: RA["rho_v1"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: Spearman FD vs V-KID CI low", -0.20, lambda: RA["rho_v1_ci"][0], tol=0.0051, appears_in="tab:functional"),
        Claim("functional: Spearman FD vs V-KID CI high", 0.45, lambda: RA["rho_v1_ci"][1], tol=0.0051, appears_in="tab:functional"),
        Claim("functional: Spearman FD vs max of the two pillars", 0.17, lambda: sp["rho_max"], tol=0.0051, appears_in="app:functional"),
        Claim("functional: max-pillar Spearman CI low", -0.18, lambda: sp["rho_max_ci"][0], tol=0.0051, appears_in="app:functional"),
        Claim("functional: max-pillar Spearman CI high", 0.45, lambda: sp["rho_max_ci"][1], tol=0.0051, appears_in="app:functional"),
        Claim("functional: generators whose worse pillar is geometry", 37, lambda: sp["max_pillar_is_v1"], tol=0, appears_in="app:functional"),
        # validity and FD board
        Claim("functional: real rank among 41 (all four variants)", 1,
              lambda: max(f["real_rank_among_40_plus_real"].values()), tol=0, appears_in="app:functional", printed=False),
        Claim("functional: FD held-out real", 0.09, lambda: A["real_test"]["fd"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: FD held-out real CI high", 0.19, lambda: A["real_test"]["fd_ci"][1], tol=0.0051, appears_in="tab:functional"),
        Claim("functional: FD Fill-Interior", 1.21, lambda: A["real@solidify"]["fd"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: Fill-Interior enterable", 0.20, lambda: A["real@solidify"]["means"]["enterable"], tol=0.0051, appears_in="app:functional"),
        Claim("functional: FD Hollow-Shell", 0.60, lambda: A["real_test@hollow"]["fd"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: FD best generator", 0.52, lambda: A["classic-matched"]["fd"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: FD best voxel model", 1.32, lambda: A["m2b_20m_sftmix_s1"]["fd"], tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: FD best voxel model, hidden blocks deleted", 0.90, lambda: A["m2b_20m_sftmix_s1@hollow"]["fd"],
              tol=0.0051, appears_in="app:functional, tab:functional"),
        Claim("functional: FD gain from deleting hidden blocks", 0.41,
              lambda: A["m2b_20m_sftmix_s1"]["fd"] - A["m2b_20m_sftmix_s1@hollow"]["fd"], tol=0.0051, appears_in="app:functional"),
        Claim("functional: FD 3D-BPE pieces arm", 1.34, lambda: A["wc5m_sftmix_p_s1"]["fd"], tol=0.0051, appears_in="app:functional"),
        Claim("functional: FD unoriented voxel baseline of that decision", 1.67, lambda: A["wc5m_sftmix_u_s1"]["fd"], tol=0.0051,
              appears_in="app:functional"),
        Claim("functional: AR voxel median enterable (%)", 22, lambda: 100 * ent["voxel_ar"]["median_enterable"], tol=0.51, appears_in="app:functional"),
        Claim("functional: AR voxel median reachable floor cells", 3.4, lambda: ent["voxel_ar"]["median_floor_cells"], tol=0.051, appears_in="app:functional"),
        Claim("functional: real enterable (%)", 94, lambda: 100 * ent["real_val"]["enterable"], tol=0.51, appears_in="app:functional"),
        Claim("functional: real floor cells", 80, lambda: ent["real_val"]["floor_cells"], tol=0.51, appears_in="app:functional"),
        Claim("functional: ridge R2 min", 0.29, lambda: min(r2), tol=0.0051, appears_in="app:functional"),
        Claim("functional: ridge R2 max", 0.62, lambda: max(r2), tol=0.0051, appears_in="app:functional"),
        # 34 design decisions under the exact test
        Claim("decisions (exact): agree", 6, lambda: dec["agree"], tol=0, appears_in="app:rankflips", note="replaces 13"),
        Claim("decisions (exact): opposite sign, both significant", 0, lambda: dec["FLIP"], tol=0,
              appears_in="app:rankflips", printed=False),
        Claim("decisions (exact): MV-KID only", 14, lambda: dec["mv_only"], tol=0, appears_in="app:rankflips", note="replaces 12"),
        Claim("decisions (exact): V-KID only", 1, lambda: dec["geom_only"], tol=0, appears_in="app:rankflips", note="replaces 3"),
        Claim("decisions (exact): neither", 13, lambda: dec["neither"], tol=0, appears_in="app:rankflips", note="replaces 6"),
        Claim("decisions (exact): pillar-dependent", 15, lambda: dec["mv_only"] + dec["geom_only"], tol=0, appears_in="app:rankflips"),
        Claim("decisions (exact): V-KID significant at all", 7, lambda: dec["agree"] + dec["geom_only"], tol=0, appears_in="app:rankflips"),
        # worked examples
        Claim("tied pair: MV-KID gap", 3.1, lambda: abs(ex["tied_pair"]["mv"]["delta_z"]), tol=0.051,
              unit="noise floors", appears_in="app:rankflips"),
        Claim("tied pair: V-KID gap", 292.9, lambda: abs(ex["tied_pair"]["v1"]["delta_z"]), tol=0.051,
              unit="noise floors", appears_in="app:rankflips"),
        Claim("tied pair: V-KID Holm p", 0.039, lambda: ex["tied_pair"]["v1"]["p_holm"], tol=0.0005, appears_in="app:rankflips"),
        Claim("flip pair: MV-KID gap", 55.1, lambda: abs(ex["flip_pair"]["mv"]["delta_z"]), tol=0.051, appears_in="app:rankflips"),
        Claim("flip pair: V-KID gap", 209.5, lambda: abs(ex["flip_pair"]["v1"]["delta_z"]), tol=0.051, appears_in="app:rankflips"),
    ]
    out += out_tab
    return out


def added_claims_stats_supp() -> List[Claim]:
    """stats.tex: tab:effect, the equivalence paragraph, fig:detection, the floor audit and
    tab:kernel (notes §88, T115). Fill-Interior MV 0.03 / V 0.78, Patch-Stitch V 0.71, the
    0.031 null and BrickGPT 2.2 are already in added_claims_effect_sizes()."""
    s = _load(STATS_RUN / "results.json")
    E, E2, K = s["effects"], s["effects2"]["probe_vs_source"], s["kernel"]
    P, T, TO, DN = E["probes"], E["tests"], E["tost"], E["detection_n"]
    cm = E["cross_metric"]
    sim, sm = s["floor_sim"]["board"], s["summary"]["task1_floor"]
    sd_mv = E["floors"]["mv"]["card_48rep"]
    out = []
    for arm, mv, v, lo_mv, hi_mv, lo_v, hi_v in [
            ("Hollow-Shell", 0.02, 0.60, 0.00, 0.08, 0.57, 0.63),
            ("Grow-Shell", 0.47, 0.80, 0.44, 0.50, 0.77, 0.84),
            ("Material-Shuffle", 0.62, 0.0, 0.60, 0.65, None, None),
            ("Monochrome", 0.70, 0.0, 0.68, 0.73, None, None),
            ("Patch-Stitch", 0.63, None, 0.61, 0.66, 0.69, 0.73)]:
        out.append(Claim(f"tab:effect {arm} MV-KID", mv, lambda a=arm: P[a]["mv"]["e_Random-Voxels"], tol=0.0051, appears_in="tab:effect"))
        out.append(Claim(f"tab:effect {arm} MV-KID CI", lo_mv, lambda a=arm: P[a]["mv"]["e_Random-Voxels_ci"][0], tol=0.0051, appears_in="tab:effect"))
        out.append(Claim(f"tab:effect {arm} MV-KID CI hi", hi_mv, lambda a=arm: P[a]["mv"]["e_Random-Voxels_ci"][1], tol=0.0051, appears_in="tab:effect"))
        if v is not None:
            out.append(Claim(f"tab:effect {arm} V-KID", v, lambda a=arm: P[a]["v1"]["e_Random-Voxels"], tol=0.0051, appears_in="tab:effect"))
        if lo_v is not None:
            out.append(Claim(f"tab:effect {arm} V-KID CI", lo_v, lambda a=arm: P[a]["v1"]["e_Random-Voxels_ci"][0], tol=0.0051, appears_in="tab:effect"))
            out.append(Claim(f"tab:effect {arm} V-KID CI hi", hi_v, lambda a=arm: P[a]["v1"]["e_Random-Voxels_ci"][1], tol=0.0051, appears_in="tab:effect"))
    out += [
        Claim("tab:effect Fill-Interior MV CI high", 0.08, lambda: P["Fill-Interior"]["mv"]["e_Random-Voxels_ci"][1], tol=0.0051, appears_in="tab:effect"),
        Claim("tab:effect Fill-Interior V CI low", 0.74, lambda: P["Fill-Interior"]["v1"]["e_Random-Voxels_ci"][0], tol=0.0051, appears_in="tab:effect"),
        Claim("tab:effect Fill-Interior V CI high", 0.83, lambda: P["Fill-Interior"]["v1"]["e_Random-Voxels_ci"][1], tol=0.0051, appears_in="tab:effect"),
        Claim("tab:effect Fill ratio", 27, lambda: cm["Fill-Interior|Random-Voxels"]["ratio_v_over_mv"], tol=0.51, appears_in="tab:effect"),
        Claim("tab:effect Hollow ratio", 26, lambda: cm["Hollow-Shell|Random-Voxels"]["ratio_v_over_mv"], tol=0.51, appears_in="tab:effect"),
        Claim("tab:effect Grow ratio", 1.7, lambda: cm["Grow-Shell|Random-Voxels"]["ratio_v_over_mv"], tol=0.051, appears_in="tab:effect"),
        Claim("tab:effect Patch-Stitch ratio", 1.1, lambda: P["Patch-Stitch"]["v1"]["e_Random-Voxels"] / P["Patch-Stitch"]["mv"]["e_Random-Voxels"],
              tol=0.051, appears_in="tab:effect"),
        Claim("tab:effect BrickGPT MV-KID", 0.17, lambda: E["brickgpt_lego"]["brickgpt"]["mv"]["e_Single-Mode"], tol=0.0051, appears_in="tab:effect"),
        Claim("tab:effect BrickGPT V-KID", 0.37, lambda: E["brickgpt_lego"]["brickgpt"]["v1"]["e_Single-Mode"], tol=0.0051, appears_in="tab:effect"),
        Claim("probe vs source: Fill MV-KID shift", 0.018, lambda: E2["mv"]["Fill-Interior"]["e_pair"], tol=0.0005, appears_in="app:failrate"),
        Claim("probe vs source: Fill MV-KID shift CI low", 0.014, lambda: E2["mv"]["Fill-Interior"]["ci"][0], tol=0.0005, appears_in="app:failrate"),
        Claim("probe vs source: Fill MV-KID shift CI high", 0.024, lambda: E2["mv"]["Fill-Interior"]["ci"][1], tol=0.0005, appears_in="app:failrate"),
        Claim("probe vs source: paired null", 0.005, lambda: E2["mv"]["Fill-Interior"]["null95"], tol=0.0005, appears_in="app:failrate"),
        Claim("probe vs source: Fill V-KID", 0.80, lambda: E2["v1"]["Fill-Interior"]["e_pair"], tol=0.0051, appears_in="app:failrate"),
        Claim("probe vs source: V/MV ratio", 44, lambda: E2["v1"]["Fill-Interior"]["e_pair"] / E2["mv"]["Fill-Interior"]["e_pair"],
              tol=0.51, appears_in="app:failrate"),
        # equivalence
        Claim("unpaired label test, Fill vs real, n=256", 0.87, lambda: T["Fill-Interior|mv"]["p_unpaired_label_perm"], tol=0.0051, appears_in="app:significance"),
        Claim("paired swap test, Fill", 0.055, lambda: T["Fill-Interior|mv"]["p_paired_swap_perm"], tol=0.0005, appears_in="app:significance"),
        Claim("paired swap test, Hollow", 0.82, lambda: T["Hollow-Shell|mv"]["p_paired_swap_perm"], tol=0.0051, appears_in="app:significance"),
        Claim("TOST Fill 90% CI low (floors)", -0.28, lambda: TO["Fill-Interior|mv"]["2_card_floors"]["ci90"][0] / sd_mv, tol=0.0051, appears_in="app:significance"),
        Claim("TOST Fill 90% CI high (floors)", 0.98, lambda: TO["Fill-Interior|mv"]["2_card_floors"]["ci90"][1] / sd_mv, tol=0.0051, appears_in="app:significance"),
        Claim("TOST Hollow 90% CI low (floors)", -0.25, lambda: TO["Hollow-Shell|mv"]["2_card_floors"]["ci90"][0] / sd_mv, tol=0.0051, appears_in="app:significance"),
        Claim("TOST Hollow 90% CI high (floors)", 0.33, lambda: TO["Hollow-Shell|mv"]["2_card_floors"]["ci90"][1] / sd_mv, tol=0.0051, appears_in="app:significance"),
        Claim("TOST Fill and Hollow equivalent at 2 floors, p<0.001", 1.0,
              lambda: float(all(TO[k]["2_card_floors"]["equivalent"] and TO[k]["2_card_floors"]["p_tost"] < 1e-3
                                for k in ("Fill-Interior|mv", "Hollow-Shell|mv"))), tol=0, unit="holds", appears_in="app:significance"),
        Claim("TOST effect-size margin in floors", 2.7, lambda: TO["Fill-Interior|mv"]["e_lt_0.05_of_anchor"]["margin_in_card_floors"],
              tol=0.051, appears_in="app:significance"),
        Claim("detection: MV-KID max power on Fill/Hollow", 0.20,
              lambda: max(c["power"] for k in ("Fill-Interior|mv", "Hollow-Shell|mv") for c in DN[k]["curve"].values()),
              tol=0.0051, appears_in="app:significance, fig:detection"),
        Claim("detection: V-KID detects Fill/Hollow/Grow at n", 8,
              lambda: max(DN[f"{p}|v1"]["n_detect_80"] for p in ("Fill-Interior", "Hollow-Shell", "Grow-Shell")),
              tol=0, appears_in="app:significance, fig:detection"),
        # floor audit
        Claim("floor sim: truth/bench min, n=256", 0.88,
              lambda: min(x["bench"]["ratio_truth_over_mean"] for x in sim if x["n"] == 256), tol=0.0051, appears_in="app:n256"),
        Claim("floor sim: truth/bench max, n=256", 0.95,
              lambda: max(x["bench"]["ratio_truth_over_mean"] for x in sim if x["n"] == 256), tol=0.0051, appears_in="app:n256"),
        Claim("floor sim: truth/bench min, n=128", 0.89,
              lambda: min(x["bench"]["ratio_truth_over_mean"] for x in sim if x["n"] == 128), tol=0.0051, appears_in="app:n256"),
        Claim("floor sim: truth/bench max, n=128", 0.93,
              lambda: max(x["bench"]["ratio_truth_over_mean"] for x in sim if x["n"] == 128), tol=0.0051, appears_in="app:n256"),
        Claim("floor sim: (1-f) overstatement factor at n=256", 1.8,
              lambda: 1 / (sum(x["degen"]["ratio_truth_over_mean"] for x in sim if x["n"] == 256)
                           / sum(1 for x in sim if x["n"] == 256)), tol=0.051, appears_in="app:n256"),
        Claim("floor sim: population size", 1464, lambda: sim[0]["pop_size"], tol=0, appears_in="app:n256"),
        Claim("floor audit: stored V-KID floor above true (%)", 11, lambda: 100 * (sm["card_floor_over_true"]["v1_n256"] - 1),
              tol=0.51, appears_in="app:n256"),
        Claim("floor audit: stored MV-KID floor below true (%)", 2.5, lambda: 100 * (1 - sm["card_floor_over_true"]["mv_n256"]),
              tol=0.051, appears_in="app:n256"),
        Claim("floor audit: 48-draw V-KID floor 95% low", 0.043, lambda: sm["bench48_mc_95pct_range_v1_n256"][0], tol=0.0005, appears_in="app:n256"),
        Claim("floor audit: 48-draw V-KID floor 95% high", 0.067, lambda: sm["bench48_mc_95pct_range_v1_n256"][1], tol=0.0005, appears_in="app:n256"),
        # kernel
        Claim("kernel: linear share of MV-KID min", 0.998,
              lambda: min(v["linear_share"] for v in K["decomposition_n256"]["mv"].values() if v.get("linear_share") is not None),
              tol=0.0005, appears_in="app:kernel"),
        Claim("kernel: every V-KID variant passes the geometry ladder at n=64 and 128", 1.0,
              lambda: float(all(v["passed"] for g in K["geometry_ladder"].values() for v in g.values())),
              tol=0, unit="holds", appears_in="app:kernel"),
    ]
    B = K["board_n256"]
    for var, row in [("vkid_base_cubic", (422, 252, 446, 0)), ("vkid_base_linear", (419, 251, 444, 0)),
                     ("vkid_base_rbf", (475, 266, 485, 0)), ("vkid_noclip_cubic", (288, 247, 409, 0)),
                     ("vkid_nowhiten_cubic", (1637, 213, 165, 0)), ("vkid_nowhiten_rbf", (692, 160, 180, 0)),
                     ("mvkid_base_cubic", (0.31, 0.03, 206, 365)), ("mvkid_linear", (0.31, 0.03, 206, 365)),
                     ("mvkid_rbf", (0.32, 0.03, 215, 351)), ("mvkid_whitened_cubic", (0.50, 0.06, 200, 277))]:
        for arm, want in zip(("Fill-Interior", "Hollow-Shell", "Grow-Shell", "Material-Shuffle"), row):
            out.append(Claim(f"tab:kernel {var} {arm}", float(want), lambda v=var, a=arm: B[v][a]["z_bench"],
                             tol=0.51 if abs(want) >= 10 else 0.0051, unit="noise floors", appears_in="tab:kernel",
                             printed=bool(want)))
    return out



def added_claims_round2_text() -> List[Claim]:
    """Numbers added by the round-2 text and layout pass (2026-11-11).

    The position-debiased GPT-4o reanalysis (run_20261111_015408_vlm_judge_debias, no
    new calls) behind §6.1, tab:vlm and app:vlm; the disqualification-gate facts of
    app:gatespec (diversity bar, the twin-gate margin of the probe exemption), read from
    the twin-gate rescore card, the gate code and the gate-provenance sweep; and the
    percentile level of the fresh-real effect-size null quoted in §6.4.
    """
    from blockgen.eval.bench import composite as _cp
    deb = _load(ROOT / "outputs" / "run_20261111_015408_vlm_judge_debias" / "results.json")
    Cd, Pd, Sy = deb["conditions"], deb["pooled"], deb["system"]
    card = _load(ROOT / "outputs" / "run_20261104_214146_bench_twin_gate_rescore"
                 / "scorecard.json")["arms"]
    nov = lambda a, k: card[a]["novelty"][k]["value"]
    gates = {g.key: g for g in _cp.GATES}
    sw = _load(ROOT / "outputs" / "run_20261106_193109_gate_provenance" / "sweep.json")
    divsw, corners = sw["part_b_novelty"]["sweep"], sw["part_a_ladder"]["corners"]
    shipped = {k for k, v in sw["part_b_novelty"]["shipped"].items() if v}
    eff = _load(STATS_RUN / "results.json")["effects"]["probes"]["Fill-Interior"]["mv"]

    def spreads(arm, key, direction):
        g = gates[key]
        sd = max(g.min_spread, _spread_of(card["real_test"]["novelty"][key]))
        d = (nov(arm, key) - nov("real_test", key)) / sd
        return d if direction == "above" else -d

    out = []
    for cond, v, lo, hi, where in [
            ("interior_ext", 0.61, 0.48, 0.78, "§6.1, tab:vlm"),
            ("interior_cut", 0.70, 0.65, 0.76, "§6.1, tab:vlm"),
            ("interior_cutq", 0.63, 0.53, 0.73, "tab:vlm"),
            ("hollow_ext", 0.57, 0.45, 0.71, "tab:vlm"),
            ("hollow_cut", 0.80, 0.72, 0.90, "tab:vlm"),
            ("gabled_patchwork", 0.28, 0.21, 0.35, "tab:vlm")]:
        out += [
            Claim(f"VLM debiased BT P(first) {cond}", v, lambda c=cond: Cd[c]["bt_p_real"],
                  tol=0.0051, appears_in=where),
            Claim(f"VLM debiased BT P(first) {cond} CI low", lo,
                  lambda c=cond: Cd[c]["ci"]["bt_p_real"][0], tol=0.0051, appears_in=where),
            Claim(f"VLM debiased BT P(first) {cond} CI high", hi,
                  lambda c=cond: Cd[c]["ci"]["bt_p_real"][1], tol=0.0051, appears_in=where)]
    PE, PR = Pd["pooled_exterior"], Pd["pooled_roof_off"]
    out += [
        Claim("VLM debiased BT P(first) catch arm", 1.0, lambda: Cd["catch"]["bt_p_real"],
              tol=0.0051, appears_in="tab:vlm"),
        Claim("VLM first-slot rate, Fill exterior (debias run)", 0.93,
              lambda: Cd["interior_ext"]["p_first"], tol=0.0051,
              appears_in="Fig. 1 caption, §6.1 (as 93%)"),
        Claim("VLM pooled exterior: order-consistent pairs", 31, lambda: PE["n_consistent"],
              appears_in="app:vlm"),
        Claim("VLM pooled exterior: P(real) on consistent pairs", 0.68,
              lambda: PE["p_real_consistent"], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled exterior: consistent CI low", 0.50,
              lambda: PE["ci"]["p_real_consistent"][0], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled exterior: consistent CI high", 0.84,
              lambda: PE["ci"]["p_real_consistent"][1], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled exterior: binomial p vs 0.5", 0.07,
              lambda: PE["binom_p_consistent_vs_half"], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled roof off: order-consistent pairs", 103, lambda: PR["n_consistent"],
              appears_in="app:vlm"),
        Claim("VLM pooled roof off: P(real) on consistent pairs", 0.89,
              lambda: PR["p_real_consistent"], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled roof off: consistent CI low", 0.83,
              lambda: PR["ci"]["p_real_consistent"][0], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled roof off: consistent CI high", 0.95,
              lambda: PR["ci"]["p_real_consistent"][1], tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled exterior BT P(real)", 0.59, lambda: PE["bt_p_real"], tol=0.0051,
              appears_in="app:vlm"),
        Claim("VLM pooled exterior BT CI low", 0.50, lambda: PE["ci"]["bt_p_real"][0],
              tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled exterior BT CI high", 0.70, lambda: PE["ci"]["bt_p_real"][1],
              tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled roof off BT P(real)", 0.73, lambda: PR["bt_p_real"], tol=0.0051,
              appears_in="app:vlm"),
        Claim("VLM pooled roof off BT CI low", 0.68, lambda: PR["ci"]["bt_p_real"][0],
              tol=0.0051, appears_in="app:vlm"),
        Claim("VLM pooled roof off BT CI high", 0.78, lambda: PR["ci"]["bt_p_real"][1],
              tol=0.0051, appears_in="app:vlm"),
        Claim("VLM system ranking unchanged by the position term", 1.0,
              lambda: float(Sy["rank"] == Sy["old_rank"]), unit="holds",
              appears_in="app:vlm"),
        # app:gatespec: disqualification bars and the probe exemption
        Claim("diversity gate bar (spreads)", 10.0, lambda: gates["voxel_diversity"].z,
              appears_in="app:gatespec", printed=False),
        Claim("Fill-Interior voxel diversity", 0.810, lambda: nov("real@solidify", "voxel_diversity"),
              tol=0.0005, appears_in="app:gatespec"),
        Claim("held-out real voxel diversity", 0.898, lambda: nov("real_test", "voxel_diversity"),
              tol=0.0005, appears_in="app:gatespec, app:dreamcubed"),
        Claim("Fill-Interior diversity gap (spreads)", 4.4,
              lambda: spreads("real@solidify", "voxel_diversity", "below"), tol=0.051,
              appears_in="app:gatespec"),
        Claim("Fill-Interior twin-rate gap (spreads)", 2.93,
              lambda: spreads("real@solidify", "voxel_twin_rate", "above"), tol=0.0051,
              appears_in="app:gatespec"),
        Claim("probes exempt from the twin gate", 1.0,
              lambda: float(not gates["voxel_twin_rate"].applies_to_probes), unit="holds",
              appears_in="app:gatespec"),
        Claim("diversity bar 5 and 20 give the shipped verdicts; 3 adds only Fill-Interior", 1.0,
              lambda: float(set(divsw["diversity z=5"]["dq"]) == shipped
                            and set(divsw["diversity z=20"]["dq"]) == shipped
                            and divsw["diversity z=3"]["changed"] == ["real@solidify"]),
              unit="holds", appears_in="app:gatespec"),
        Claim("fig:gatesweep joint corners as captioned", 1.0,
              lambda: float(corners["strict"]["params"] == {"order_tol": -0.5, "k6": 0.5, "k7": 1.0, "k8": 1.0}
                            and corners["lenient"]["params"] == {"order_tol": 1.0, "k6": 3.0, "k7": 3.0, "k8": 3.0}
                            and corners["very_lenient"]["params"] == {"order_tol": 2.0, "k6": 5.0, "k7": 5.0, "k8": 5.0}),
              unit="holds", appears_in="fig:gatesweep caption"),
        # §6.4: the fresh-real null quoted for the appearance effect size is its 95th percentile
        Claim("percentile level of the fresh-real effect-size null", 95,
              lambda: 95.0 if "e_Random-Voxels_null95" in eff else float("nan"),
              appears_in="§6.4", note="the stats run stores this null as e_Random-Voxels_null95"),
    ]
    return out


def _spread_of(metric: dict) -> float:
    """composite._spread on a stored metric dict: half the 95% interval over 1.96, or nan."""
    lo, hi = (metric.get("ci") or [None, None])[:2]
    if lo is None or hi is None or hi <= lo:
        return float("nan")
    return (hi - lo) / (2 * 1.96)


if __name__ == "__main__":
    raise SystemExit(main())
