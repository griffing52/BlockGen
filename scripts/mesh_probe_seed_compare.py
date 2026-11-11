"""Objaverse 568-object probe: seed ranges (item 4 of the 2026-11-11 queue).

Reads the seed-0 run and the seed-1 CPU-only rerun. If seed 1 has its results.json, every
row is compared; if only embeddings.npz exists, the 3D-encoder KID rows are rescored from
it with `probe3d.score` (as seed 0's kid3d_from_embeddings.json). Writes results.json,
results.md, samples.png and item4 LaTeX rows into the seed-1 run dir.

    .venv/bin/python scripts/mesh_probe_seed_compare.py <seed-1 run dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blockgen.eval.bench import distances as D  # noqa: E402
from blockgen.eval.bench import probe3d as p3  # noqa: E402

S0 = Path("outputs/run_20261106_210006_mesh_probe3d_568_s0")
PROBES = ("strip", "fill", "noise", "wall", "decim")
ROWS = [("dino4_kid", "DINOv2, 4 exterior views, KID"), ("dino4_fd", "DINOv2, 4 exterior views, FD"),
        ("dino_cut4_kid", "DINOv2, 4 cutaways, KID"), ("vkid_geom", "V-KID"),
        ("uni3d_all_kid", "Uni3D KID, all"), ("uni3d_vis_kid", "Uni3D KID, vis"),
        ("uni3d_cut_kid", "Uni3D KID, cut"), ("ulip2_all_kid", "ULIP-2 KID, all"),
        ("ulip2_vis_kid", "ULIP-2 KID, vis"), ("ulip2_cut_kid", "ULIP-2 KID, cut"),
        ("cd_all_mmd", "chamfer MMD, all"), ("cd_vis_mmd", "chamfer MMD, vis"),
        ("emd_all_mmd", "EMD MMD, all"), ("emd_vis_mmd", "EMD MMD, vis"),
        ("iou_vox_mmd", "voxel IoU MMD")]


def rescore_from_embeddings(run: Path, seed: int) -> dict:
    z = np.load(run / "embeddings.npz")
    ref, null = z["ref_ix"], z["null_ix"]
    rung = {k[5:]: z[k] for k in z.files if k.startswith("rung_")}
    n = len(rung["real_heldout"])
    table = {}
    for k in z.files:
        if k.endswith("_kid"):
            X = z[k]
            table[k] = (lambda r, p: D.kid(p, r), X[ref], {a: X[v] for a, v in rung.items()},
                        X[null])
    return p3.score(table, PROBES, n, 48, 200, np.random.default_rng(seed + 2), log=lambda s: None)


def main():
    s1 = Path(sys.argv[1])
    shutil.copy(__file__, s1 / Path(__file__).name)
    r0 = json.loads((S0 / "results.json").read_text())["results"]
    if (s1 / "results.json").exists() and "results" in json.loads((s1 / "results.json").read_text()):
        r1 = json.loads((s1 / "results.json").read_text())["results"]
        src = "results.json"
    else:
        r1 = rescore_from_embeddings(s1, 1)
        src = "embeddings.npz (3D-encoder KID rows only)"
    out = {"seed0": str(S0), "seed1": str(s1), "seed1_source": src, "rows": {}}
    L = ["# Objaverse 568 strict multi-room objects: seeds 0 and 1 (n = 170, ref 284, pool 284)", "",
         f"Seed 1 source: {src}. Separation in each seed's own noise floors [95% paired subsample CI].",
         "Voxelization: each mesh centred and scaled to unit longest side, 32 cells per unit "
         "(mesh_probe.VOXEL_R), occupancy = surface cells by dense area-weighted sampling "
         "(about 40 samples per cell-face area) plus Fill's solid points; hidden/visible faces "
         "classified on a 48-per-unit grid (HIDDEN_R).", "",
         "| row | Strip s0 | Strip s1 | Fill s0 | Fill s1 | noise s0/s1 | wall s0/s1 | decim s0/s1 |",
         "|---|---|---|---|---|---|---|---|"]
    f = lambda r, v: (f"{r['sep'][v]:+.1f} [{r['sep_ci95'][v][0]:+.1f}, {r['sep_ci95'][v][1]:+.1f}]"
                      if r else "--")
    for key, lab in ROWS:
        a, b = r0.get(key), r1.get(key)
        if a is None:
            continue
        out["rows"][key] = {"label": lab, "s0": a["sep"], "s1": b["sep"] if b else None,
                            "s0_ci": a["sep_ci95"], "s1_ci": b["sep_ci95"] if b else None,
                            "range": {v: [min(a["sep"][v], b["sep"][v]), max(a["sep"][v], b["sep"][v])]
                                      for v in PROBES} if b else None}
        g = lambda v: f"{a['sep'][v]:+.1f} / {b['sep'][v]:+.1f}" if b else f"{a['sep'][v]:+.1f} / --"
        L.append(f"| {lab} | {f(a, 'strip')} | {f(b, 'strip')} | {f(a, 'fill')} | {f(b, 'fill')} | "
                 f"{g('noise')} | {g('wall')} | {g('decim')} |")
    (s1 / "seed_compare.json").write_text(json.dumps(out, indent=2) + "\n")
    (s1 / "seed_compare.md").write_text("\n".join(L) + "\n")
    tex = ["% auto-written by scripts/mesh_probe_seed_compare.py: rows 'seed 0 / seed 1'"]
    for key, d in out["rows"].items():
        if d["s1"] is None:
            continue
        cells = " & ".join(f"${d['s0'][v]:+.1f}$\\,/\\,${d['s1'][v]:+.1f}$" for v in PROBES)
        tex.append(f"{d['label']} & {cells} \\\\")
    Path("deliverables/paper/cvpr27/drafts_20261111_experiments/item4_objaverse_rows.tex").write_text(
        "\n".join(tex) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in out["rows"] if out["rows"][k]["s1"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (v, c) in enumerate((("strip", "#2b6cb0"), ("fill", "#c05621"))):
        x = np.arange(len(ks)) + (j - 0.5) * 0.35
        lo = [min(out["rows"][k]["s0"][v], out["rows"][k]["s1"][v]) for k in ks]
        hi = [max(out["rows"][k]["s0"][v], out["rows"][k]["s1"][v]) for k in ks]
        ax.vlines(x, np.maximum(np.array(lo), 0.1), np.maximum(np.array(hi), 0.1), color=c, lw=4,
                  label=f"{v}: seed 0 to seed 1")
    ax.axhline(2, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([out["rows"][k]["label"] for k in ks], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("separation, noise floors (floored at 0.1)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(s1 / "seed_compare.png", dpi=130)


if __name__ == "__main__":
    main()
