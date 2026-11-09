"""Claim-registry additions for drafts_20261109 (rankflips.tex, stats.tex, limitation_line.tex).

Paste the three functions below into scripts/audit_claims.py (next to
added_claims_rankflips / added_claims_effect_sizes) and register them in
build_claims():

    claims += added_claims_rankflips_exact()
    claims += added_claims_functional_rebase()
    claims += added_claims_stats_supp()

and DELETE from added_claims_rankflips() the claims this text no longer prints
(see README.md, "Obsolete claims"). They read
`run_20261106_193513_render_blind_rank_flips/results.json`, whose pair counts
came from the subsampling test.

All three functions read pinned runs (no `card` / `lb` free variable), so
build_claims gives them role "" and they move with neither run of record.
This file imports Claim/_load/ROOT from audit_claims only so it can be
smoke-tested standalone:

    .venv/bin/python deliverables/paper/cvpr27/drafts_20261109/audit_claims_additions.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "scripts"))
from audit_claims import ROOT, Claim, _load  # noqa: E402

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


if __name__ == "__main__":
    bad = 0
    for fn in (added_claims_rankflips_exact, added_claims_functional_rebase, added_claims_stats_supp):
        cs = fn()
        for c in cs:
            ok, got, err = c.check()
            if not ok:
                bad += 1
                print(f"FAIL {fn.__name__}: {c.name}: paper {c.paper_value} got {got} {err or ''}")
        print(f"{fn.__name__}: {len(cs)} claims")
    print("all pass" if not bad else f"{bad} failing")
    sys.exit(1 if bad else 0)
