"""Verify every numeric claim in the paper against the data that produced it.

A paper's numbers drift. A run gets re-scored, a probe changes, a table is
edited by hand, and the text quietly stops matching the artifact it cites. That
failure is invisible at build time -- LaTeX compiles a wrong number exactly as
happily as a right one -- and expensive at review time.

This is the registry that closes that gap. Each entry names a claim, where it
appears in the paper, and how to recompute it from a scorecard or a recorded
run. Run it before every submission:

    .venv/bin/python scripts/audit_claims.py
    .venv/bin/python scripts/audit_claims.py --verbose   # show the derivation

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
PAPER = ROOT / "deliverables" / "paper" / "iclr"
BENCH = ROOT / "outputs" / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull"
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
LADDER = (ROOT / "outputs" / "analysis" / "ladder"
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


def build_claims() -> List[Claim]:
    card = Card(BENCH / "scorecard.json")
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

    C(Claim("a 25%-bad arm reads this fraction of fully-bad", 0.064,
            bad_quarter_ratio, tol=0.004, appears_in="abstract, §5",
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
    lb = Leaderboard(BENCH / "scorecard.md")
    for arm, pillar, want, tol in [
            ("real@solidify", "appearance", 0.14, 0.005),
            ("real@solidify", "geometry", 201.80, 0.005),
            ("real_shuffled_materials", "appearance", 231.52, 0.005),
            ("real_shuffled_materials", "geometry", 0.00, 1e-9),
            ("real@monochrome", "appearance", 297.95, 0.005),
            ("real@monochrome", "palette", 79.63, 0.005),
            ("patchwork", "appearance", 246.15, 0.005),
            ("patchwork", "geometry", 183.03, 0.005),
            ("canon_oriented", "geometry", 136.73, 0.005),
            ("native_oriented", "geometry", 106.87, 0.005),
            ("llm_control", "appearance", 149.19, 0.005),
            ("llm_bias", "appearance", 191.88, 0.005),
            ("real_test", "appearance", 0.00, 1e-9),
    ]:
        C(Claim(f"leaderboard {arm}/{pillar}", want,
                lambda a=arm, q=pillar: lb.pillar(a, q), tol=tol,
                unit="noise floors", appears_in="Table 1",
                note="cell of the restored leaderboard"))
    C(Claim("pillar gap on the interior probe", 1441.0,
            lambda: lb.ratio("real@solidify", "geometry", "appearance"), tol=1.0,
            unit="x", appears_in="§6.4",
            note="the paper rounds this to 530; an earlier draft said 580, which "
                 "came from a run whose appearance pillar read +0.35 rather than "
                 "+0.38 -- a cross-run division of exactly the kind Card.sep exists "
                 "to prevent"))
    C(Claim("best attack on the geometric metric", 1.34,
            lambda: lb.ratio("patchwork", "appearance", "geometry"), tol=0.02,
            unit="x", appears_in="§6.4",
            note="Patch-Stitch, not Material-Shuffle; the paper named the wrong "
                 "arm for this number before the leaderboard was restored"))

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
            appears_in="intro bullet 4, §6.5",
            note="the intro said 0.022, which is the GEOMETRIC metric's value"))
    C(Claim("raw readout MAE, render metric", 0.128,
            lambda: _mf["mv_dino_kid"]["mae_raw"], tol=0.0015,
            appears_in="intro bullet 4, §6.5",
            note="the intro said 0.119, traceable only to an n=64 run whose bad "
                 "pool (canon8) this project disavowed as a quality reference"))
    C(Claim("sqrt readout MAE, geometric metric", 0.020,
            lambda: _mf["geom_kid"]["mae_sqrt"], tol=0.0006,
            appears_in="§6.5"))
    C(Claim("raw readout MAE, geometric metric", 0.131,
            lambda: _mf["geom_kid"]["mae_raw"], tol=0.0015,
            appears_in="§6.5"))
    C(Claim("mixture exponent, render metric", 2.05,
            lambda: _mf["mv_dino_kid"]["exponent_best_fit"], tol=0.05,
            appears_in="§6.5", note="quadratic response, as the identity predicts"))

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
    C(Claim("contiguous draw inflates the noise floor", 10.4,
            lambda: _cd["sd_inflation"], tol=0.15, unit="x",
            appears_in="§5",
            note="replaces an unbacked 'roughly 45%' that traced to no recorded run"))
    C(Claim("shuffled calibration draw reproduces the bench floor", 0.00305,
            lambda: _cd["shuffled"]["sd"], tol=0.0001,
            appears_in="§5",
            note="independent recomputation matching the run of record's "
                 "realism.mv_dino_kid floor of 0.003055, which is what makes the "
                 "contiguous comparison meaningful"))

    # --- per-build influence, the probe-free evidence (fig:influence) -------
    _inf = _load(ROOT / "outputs" / "figstage" / "influence_pair.json")
    C(Claim("render metric's per-build ranking vs sealed interior volume", 0.020,
            lambda: _inf["rho_render_sealed"], tol=0.01, unit="Spearman rho",
            appears_in="§6.4, fig:influence",
            note="not significantly different from zero; this is interior "
                 "blindness measured on real builds with no probe"))
    C(Claim("geometric metric's per-build ranking vs sealed interior volume", -0.343,
            lambda: _inf["rho_geom_sealed"], tol=0.01, unit="Spearman rho",
            appears_in="§6.4, fig:influence"))
    C(Claim("the two per-build rankings are nearly independent", 0.122,
            lambda: _inf["rho_render_geom"], tol=0.01, unit="Spearman rho",
            appears_in="§6.4, fig:influence"))

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

    # --- the rater study (run_20260923_human_v5) ---------------------------
    _hs = _load(ROOT / "outputs" / "run_20260923_human_v5" / "analysis.json")
    _hp = _hs["pairs"]
    C(Claim("raters on the exterior pair", 0.512,
            lambda: _hp["real|real@solidify"]["p"], tol=0.005,
            unit="fraction preferring real", appears_in="§6.7",
            note="interval contains chance: the corruption is invisible to "
                 "people as well as to the metric"))
    C(Claim("raters on the cutaway pair (positive control)", 0.767,
            lambda: _hp["real#cut|real@solidify#cut"]["p"], tol=0.005,
            unit="fraction preferring real", appears_in="§6.7",
            note="the same people, the same builds, a view that shows the "
                 "interior: the information is recoverable"))
    C(Claim("raters reverse the metrics on the pre-registered pair", 1.0,
            lambda: _hp["gabled_house|patchwork"]["p"], tol=0.0,
            unit="fraction preferring gabled_house", appears_in="§6.7",
            note="93/93 unanimous, against both metrics ranking patchwork "
                 "higher; the direction was pre-registered in the instrument"))
    C(Claim("catch-trial accuracy", 0.991,
            lambda: _hp["real|uniform_random"]["p"], tol=0.005,
            appears_in="§6.7", note="raters were attending"))

    return claims


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
        "kid": (0.013, 0.178, 0.447, 0.108, 0.782, 0.011, 0.008),
        "mmd_rbf": (3.8, 53.5, 141.7, 32.9, 226.5, 3.3, 2.5),
        "one_minus_recall": (0.113, 0.315, 0.919, 0.323, 0.750, 0.121, 0.027),
        "fd": (0.107, 0.161, 0.252, 0.147, 0.374, 0.105, 0.005),
        "one_minus_coverage": (0.258, 0.512, 0.839, 0.516, 0.952, 0.282, 0.052),
        "one_minus_density": (0.000, 0.259, 0.594, 0.216, 0.881, 0.000, 0.071),
        "one_minus_precision": (0.125, 0.188, 0.203, 0.141, 0.531, 0.109, 0.043),
        "legacy_cmmd": (0.082, 0.502, 1.178, 0.321, 2.037, 0.078, 0.022),
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


def check_paper_mentions(claims: List[Claim]) -> List[str]:
    """Numbers in the abstract that no claim covers.

    A weak check by design -- it cannot know which numerals are claims -- but it
    catches a number added to the abstract and never registered here.
    """
    abstract = (PAPER / "sec" / "0_abstract.tex")
    if not abstract.exists():
        return []
    text = abstract.read_text()
    # Model names carry digits that are not claims. Strip them before scanning,
    # or "ResNet-50" is reported forever as an unregistered "-50".
    text = re.sub(r"(?:ResNet|DINOv|CLIP|VGG|Inception)-?[A-Za-z]?\d+", " ", text)
    nums = set(re.findall(r"\$?(-?\d+\.?\d*)\\?%?\$?", text))

    known = set()
    for c in claims:
        for v in (c.paper_value, abs(c.paper_value)):
            known.add(f"{v:g}")
            # a claim registered as a fraction is often quoted as a percentage
            if abs(v) <= 1.0:
                known.add(f"{round(v * 100):g}")
                known.add(f"{v * 100:g}")

    # Numerals that are genuinely prose: sample sizes, counts, section numbers.
    # Keep this list SHORT and never park a superseded value here -- an earlier
    # version whitelisted 0.022 and 0.119, the two wrong numbers an audit later
    # found in the intro, which is exactly the failure this check exists to stop.
    # "200" is the abstract's rounded-down form of the 202 claim; the bound
    # itself is registered above, so the rounding is verified, not assumed.
    allowed = {"3", "0", "1", "2", "4", "16", "128", "256", "200"}
    return sorted(n for n in nums - known - allowed
                  if n not in {"", "-"} and float(n) != 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    claims = build_claims()
    print(f"\n  auditing {len(claims)} registered claims against\n"
          f"  {DIM}{BENCH.name}{RESET}\n")
    failed = 0
    corpus = 0
    for c in claims:
        ok, got, err = c.check()
        if ok:
            mark, colour = "OK  ", GREEN
        elif needs_corpus(err):
            mark, colour = "CORP", YELLOW
            corpus += 1
        else:
            mark, colour = "FAIL", RED
            failed += 1
        unit = f" {c.unit}" if c.unit else ""
        line = f"  {colour}{mark}{RESET}  {c.name}"
        if mark == "CORP":
            line += f"  {DIM}paper {c.paper_value:g}{unit}; needs the corpus on disk{RESET}"
        elif not ok:
            line += f"  {RED}paper {c.paper_value:g}{unit} vs computed {got:g}{unit}{RESET}"
            if err:
                line += f"  {DIM}({err}){RESET}"
        elif args.verbose:
            line += f"  {DIM}= {got:g}{unit} · {c.appears_in}{RESET}"
        print(line)
        if args.verbose and c.note:
            print(f"        {DIM}{c.note}{RESET}")

    stray = check_paper_mentions(claims)
    if stray:
        print(f"\n  {YELLOW}unregistered numbers in the abstract:{RESET} "
              f"{', '.join(stray)}")
        print(f"  {DIM}add them to build_claims() or confirm they are prose{RESET}")

    print()
    if corpus:
        print(f"  {YELLOW}{corpus} claim(s) read the corpus, which is not on disk;"
              f" rebuild it to check them (see README).{RESET}")
    if failed:
        print(f"  {RED}{failed} claim(s) do not match the data.{RESET}\n")
    else:
        checked = len(claims) - corpus
        print(f"  {GREEN}Every checked claim ({checked} of {len(claims)}) matches the run of record.{RESET}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
