"""Blind labelling sheet: does KID's tail actually contain worse examples?

`scripts/build_influence.py` decomposes an arm's KID into an exact per-build
contribution. Builds with a high contribution are *atypical*; whether they are
also *bad* is a question about the corpus that the metric cannot answer about
itself. This asks a person.

PRE-REGISTERED DESIGN, fixed before any label is collected
----------------------------------------------------------
* **Hypothesis.** Builds in the top influence stratum are more likely to be
  labelled "not a building" or "building, not a house" than builds in the
  bottom stratum.
* **Sampling.** The arm's builds are split into four equal strata by influence
  and `--per-stratum` builds are drawn from each, so the sheet is balanced by
  construction and the labeller cannot infer a stratum from how often a label
  seems warranted.
* **Blinding.** The page shows renders only. No score, no rank, no stratum, no
  corpus, no title, and card ids are a shuffled hash. Presentation order is
  randomised independently of the sampling order. The key is written to a
  separate file the page never loads.
* **Labels.** house / building, not a house / not a building / unsure.
* **Primary analysis.** Cochran-Armitage trend test across the four strata on
  P(not a house) = P(building, not a house) + P(not a building), two-sided,
  alpha = 0.05. "unsure" is excluded and its rate reported.
* **Secondary.** Top vs bottom stratum as a 2x2 with Fisher's exact test.
* **What a null means.** If the trend is flat, influence does not track what a
  person calls a bad example, and the tail is atypical rather than bad. That is
  a publishable answer and it is the one that forbids using this for curation.

This sheet shows builds from every corpus on purpose: the question is whether
the quality gate let something through, and a filtered view cannot answer it.
The renders are therefore INTERNAL. Do not publish anything derived from them
without re-applying the 3D-Craft gate.

Usage
-----
    .venv/bin/python scripts/curation_label_sheet.py            # build the sheet
    open outputs/curation/label_sheet.html                      # label, then Export
    .venv/bin/python scripts/curation_label_sheet.py --analyze outputs/curation/labels.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "curation"
N_STRATA = 4
LABELS = ["house", "building, not a house", "not a building", "unsure"]


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img[..., :3]).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build_sheet(args) -> int:
    from blockgen.eval.bench import splits, features as ft
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures
    from scripts.build_influence import influence

    split = splits.load_split("houses_32")
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    ref_s = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]

    idx = np.random.default_rng(1).permutation(len(test))[:args.n]
    arm = [test[i] for i in idx]
    print(f"embedding {len(arm)} + {len(ref_s)} builds ...", flush=True)
    view = ft.ViewConfig()
    fa = ft.pooled(ft.embed_views(arm, view, verbose=False))
    fr = ft.pooled(ft.embed_views(ref_s, view, verbose=False))
    c = influence(fa, fr)

    order = np.argsort(c)                      # low (typical) -> high (atypical)
    strata = np.array_split(order, N_STRATA)
    rng = np.random.default_rng(args.seed)
    picked = []
    for s_i, st in enumerate(strata):
        take = rng.choice(st, size=min(args.per_stratum, len(st)), replace=False)
        picked += [(int(j), s_i) for j in take]
    rng.shuffle(picked)                        # presentation order independent of stratum

    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    rows = list(split.indices("test"))

    # The METRIC'S OWN renderer, with the vanilla texture pack, because the
    # labeller must judge the images the metric ranked. `features.render_views`
    # calls `load_face_textures()` with no arguments, so the influence scores
    # these strata are built from were computed on textured renders; showing
    # flat colours here would ask a person about different pictures. Textures
    # also carry most of the evidence for "is this a house": wood against
    # stone, a glass pane, a door.
    #
    # Minecraft textures are Mojang's copyright, which is why every figure in
    # the paper uses flat colours instead. That constraint is what makes this
    # sheet INTERNAL ONLY, as the module docstring says.
    tex = load_face_textures()

    cards, key = [], []
    for pos, (j, s_i) in enumerate(picked):
        cid = hashlib.blake2b(f"{args.seed}:{j}".encode(), digest_size=6).hexdigest()
        views = [_png_b64(render_structure(arm[j], px=300, azim_deg=az, elev_deg=28,
                                           ortho=True, face_textures=tex))
                 for az in (45, 225)]
        cards.append({"id": cid, "views": views})
        b = int(idx[j])
        key.append({"id": cid, "position": pos, "test_index": b, "stratum": s_i,
                    "influence": float(c[j]),
                    "deviation": float(c[j] - c.mean()),
                    "corpus": man["items"][rows[b]].get("corpus"),
                    "title": man["items"][rows[b]].get("title") or "",
                    "blocks": int(arm[j].occupied_mask.sum())})
        print(f"  rendered {pos+1}/{len(picked)}", end="\r", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answer_key.json").write_text(json.dumps(
        {"n_strata": N_STRATA, "seed": args.seed, "arm_n": args.n,
         "per_stratum": args.per_stratum, "cards": key}, indent=1))
    (OUT / "label_sheet.html").write_text(_html(cards))
    print(f"\n\nwrote {OUT/'label_sheet.html'}  ({len(cards)} builds)")
    print(f"wrote {OUT/'answer_key.json'}  (do NOT open before labelling)")
    print("\nOpen the HTML, label every card, press Export, save next to the key,")
    print("then: .venv/bin/python scripts/curation_label_sheet.py --analyze "
          f"{OUT/'labels.json'}")
    return 0


def _html(cards) -> str:
    opts = "".join(
        f'<button class="opt" data-v="{l}">{l}</button>' for l in LABELS)
    items = "".join(
        f'<div class="card" data-id="{c["id"]}">'
        f'<div class="imgs"><img src="data:image/png;base64,{c["views"][0]}">'
        f'<img src="data:image/png;base64,{c["views"][1]}"></div>'
        f'<div class="opts">{opts}</div></div>'
        for c in cards)
    return """<!doctype html><meta charset="utf-8"><title>Curation labelling</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f7f7f6;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
        padding:12px 20px;display:flex;gap:18px;align-items:center;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600}
 #prog{font-variant-numeric:tabular-nums;color:#555}
 button.exp{margin-left:auto;padding:7px 16px;border:1px solid #1a1a1a;
            background:#1a1a1a;color:#fff;border-radius:5px;cursor:pointer}
 button.exp[disabled]{opacity:.35;cursor:not-allowed}
 .wrap{padding:20px;display:grid;gap:16px;
       grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
 .card{background:#fff;border:1px solid #e2e2e0;border-radius:8px;padding:10px}
 .card.done{border-color:#9ac;background:#fbfdff}
 .imgs{display:flex;gap:6px}
 .imgs img{width:50%;border-radius:4px;background:#fff}
 .opts{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
 .opt{flex:1 1 auto;padding:6px 8px;font-size:12px;border:1px solid #ccc;
      background:#fafafa;border-radius:4px;cursor:pointer}
 .opt:hover{background:#eef}
 .opt.sel{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
 .note{padding:0 20px 8px;color:#666;font-size:12.5px;max-width:70ch}
</style>
<header><h1>Is this a house?</h1><span id="prog"></span>
<button class="exp" id="exp" disabled>Export labels</button></header>
<p class="note">Two views of the same build. Judge only what it <em>is</em>, not
how good it looks: a plain house is still a house, an ornate tower is a building
but not a house, and a tree or a boat is not a building. Order is random.</p>
<div class="wrap">""" + items + """</div>
<script>
const ans={},cards=document.querySelectorAll('.card'),prog=document.getElementById('prog');
function upd(){const n=Object.keys(ans).length;prog.textContent=n+' / '+cards.length;
 document.getElementById('exp').disabled = n<cards.length;}
cards.forEach(c=>c.querySelectorAll('.opt').forEach(b=>b.onclick=()=>{
 c.querySelectorAll('.opt').forEach(o=>o.classList.remove('sel'));
 b.classList.add('sel');c.classList.add('done');
 ans[c.dataset.id]=b.dataset.v;upd();}));
document.getElementById('exp').onclick=()=>{
 const blob=new Blob([JSON.stringify({labels:ans},null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='labels.json';a.click();};
upd();
</script>"""


def analyze(path: Path) -> int:
    key = json.loads((OUT / "answer_key.json").read_text())
    labels = json.loads(Path(path).read_text())["labels"]
    by_id = {c["id"]: c for c in key["cards"]}
    missing = set(by_id) - set(labels)
    if missing:
        print(f"{len(missing)} of {len(by_id)} cards unlabelled; finish the sheet first.")
        return 1

    n_s = key["n_strata"]
    tot = np.zeros(n_s, int); bad = np.zeros(n_s, int); unsure = np.zeros(n_s, int)
    for cid, lab in labels.items():
        s = by_id[cid]["stratum"]
        if lab == "unsure":
            unsure[s] += 1
            continue
        tot[s] += 1
        bad[s] += lab in ("building, not a house", "not a building")

    print("stratum  (1 = most typical by KID, "
          f"{n_s} = most atypical)\n")
    for s in range(n_s):
        r = bad[s] / tot[s] if tot[s] else float("nan")
        print(f"  {s+1}:  not-a-house {bad[s]:>2}/{tot[s]:<2} = {r:5.1%}"
              f"   unsure {unsure[s]}")

    # Cochran-Armitage trend
    x = np.arange(n_s, dtype=float)
    N, S = tot.sum(), bad.sum()
    if N == 0 or S in (0, N):
        print("\nno variation in labels; trend test not defined.")
        return 0
    p = S / N
    xbar = (tot * x).sum() / N
    T = (bad * (x - xbar)).sum()
    varT = p * (1 - p) * (tot * (x - xbar) ** 2).sum()
    z = T / np.sqrt(varT) if varT > 0 else 0.0
    from math import erfc, sqrt
    pval = erfc(abs(z) / sqrt(2))
    print(f"\nCochran-Armitage trend: z = {z:+.2f}, two-sided p = {pval:.4f}")

    # top vs bottom, Fisher exact
    from math import comb
    a, b = bad[-1], tot[-1] - bad[-1]
    c_, d = bad[0], tot[0] - bad[0]
    n1, n2, k = a + b, c_ + d, a + c_
    def hyp(i): return comb(n1, i) * comb(n2, k - i) / comb(n1 + n2, k)
    lo, hi = max(0, k - n2), min(k, n1)
    obs = hyp(a)
    pf = sum(hyp(i) for i in range(lo, hi + 1) if hyp(i) <= obs + 1e-12)
    print(f"top vs bottom stratum: {a}/{n1} against {c_}/{n2}, "
          f"Fisher two-sided p = {pf:.4f}")

    print("\nnot-a-house calls in the top stratum, for your review:")
    for cid, lab in labels.items():
        c = by_id[cid]
        if c["stratum"] == n_s - 1 and lab in ("building, not a house", "not a building"):
            print(f"  test[{c['test_index']:>3}]  dev={c['deviation']:+.3f}  "
                  f"{c['corpus']:<9} {lab:<22} {c['title'][:34]}")
    if pval >= 0.05:
        print("\nTrend is not significant: on this evidence KID's tail is atypical "
              "rather than worse, and must not be used to filter the corpus.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=128, help="arm size")
    ap.add_argument("--per-stratum", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--analyze", metavar="LABELS_JSON")
    args = ap.parse_args()
    return analyze(Path(args.analyze)) if args.analyze else build_sheet(args)


if __name__ == "__main__":
    raise SystemExit(main())
