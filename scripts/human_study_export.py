"""Export a two-alternative forced-choice study for validating the benchmark.

Why 2AFC over individual builds, when the metrics score *sets*
--------------------------------------------------------------
`geom_kid` and `mv_dino_kid` are two-sample statistics: they score a
distribution, not a build, so there is no per-build number to correlate with a
per-build human rating. The obvious fix -- show raters a grid from arm A beside a
grid from arm B and ask which *set* looks more real -- asks people to eyeball a
distribution, which they are poor at and which produces one datum per rater per
pair.

So: raters compare **individual builds**, one drawn from each arm, which is an
easy and reliable judgement; and the trials are then **aggregated to the arm
pair**, which is the level the metric speaks at. For each ordered pair (A, B) we
get a human preference rate p_AB, and the metric makes a binary prediction
(whichever arm it scores closer to real). Agreement over pairs is the headline;
`|p_AB - 0.5|` against the metric's margin says whether the metric's confidence
tracks human confidence, which is the more informative of the two.

Catch trials (real vs `uniform_random`) are interleaved. A rater who does not
get those right is not doing the task, and their data is dropped before any
analysis -- decided here, in advance, rather than after looking at the results.

    python scripts/human_study_export.py --per-arm 12 --out outputs/run_.../
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import baselines as bl
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

#: The catch pair. Anyone who cannot tell a real house from uniform noise is not
#: attending, and this is the cheapest possible way to find out.
CATCH = ("real", "uniform_random")


def render_one(s: Structure, px: int, azim: float = 45.0, elev: float = 30.0
               ) -> np.ndarray:
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures

    img = render_structure(s, px=px, azim_deg=azim, elev_deg=elev, ortho=True,
                           face_textures=load_face_textures())
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def to_jpeg(img: np.ndarray, quality: int = 82) -> str:
    """Base64 JPEG so the whole study is one self-contained file.

    A study that depends on a sibling image directory breaks the moment it is
    emailed, uploaded or opened from a different path, and a broken stimulus is
    indistinguishable from an ugly build in the data it produces.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def collect_arms(corpus: str, seed: int, per_arm: int, extra: Dict[str, str],
                 ) -> Dict[str, List[Structure]]:
    split = splits.load_split(corpus, seed)
    train = splits.split_structures(split, "train")
    test = splits.split_structures(split, "test")
    rng = np.random.default_rng(seed)

    pool = [s.crop_to_non_air() for s in test]
    pool = [pool[i] for i in rng.permutation(len(pool))]
    arms: Dict[str, List[Structure]] = {"real": pool[:per_arm]}

    for name in bl.ORDER:
        if name == "train_copy_noise":
            continue                      # too close to `real` to be informative
        arms[name] = bl.build(name, per_arm, train[:600], seed=seed)

    # A known-quality rung, so the study has an internal ordering to recover.
    arms["real@solidify"] = [pb.solidify(s) for s in pool[per_arm:2 * per_arm]]

    from blockgen.curation.houses import load_structures_from_cache
    for name, path in extra.items():
        structs, _ = load_structures_from_cache(path)
        structs = [s.crop_to_non_air() for s in structs]
        structs = [s for s in structs if s.occupied_mask.any()]
        if not structs:
            print(f"[study] skipping {name}: no non-empty structures")
            continue
        idx = np.random.default_rng(seed).permutation(len(structs))[:per_arm]
        arms[name] = [structs[i] for i in idx]
    return arms


def make_trials(arms: Dict[str, List[int]], rng: np.random.Generator,
                per_pair: int) -> List[dict]:
    """All unordered arm pairs, `per_pair` build pairs each, sides randomized."""
    names = list(arms)
    trials: List[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for _ in range(per_pair):
                ia = int(rng.choice(arms[a]))
                ib = int(rng.choice(arms[b]))
                # Randomize which side each arm appears on, or raters develop a
                # side bias that is indistinguishable from a preference.
                flip = bool(rng.integers(2))
                trials.append({
                    "left": ib if flip else ia,
                    "right": ia if flip else ib,
                    "left_arm": b if flip else a,
                    "right_arm": a if flip else b,
                    "catch": {a, b} == set(CATCH),
                })
    rng.shuffle(trials)
    for k, t in enumerate(trials):
        t["id"] = k
    return trials


#: The rater-facing page. Kept here rather than in a sibling file so the study
#: is reproducible from one script, and generated rather than hand-maintained so
#: the arm labels cannot drift back into the DOM (see `build_page`).
PAGE = """<title>Which Build Looks Real</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  /* Deliberately single-theme. A perceptual judgement made against a dark
     ground is not the same judgement made against a light one, and the stimuli
     are rendered on white; letting the viewer's OS decide the surround would
     put an uncontrolled variable in the middle of the experiment. Every colour
     is painted explicitly so the page holds on either host ground. */
  :root {
    --ground: #edeef0; --surface: #ffffff; --ink: #16181c; --muted: #6b7078;
    --accent: #2f5d50; --accent-soft: #dfe8e4; --rule: #d3d6db;
    --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: var(--sans); -webkit-font-smoothing: antialiased;
    min-height: 100vh; display: flex; flex-direction: column;
  }
  .bar { height: 3px; background: var(--rule); flex: 0 0 auto; }
  .bar > i { display: block; height: 100%; width: 0; background: var(--accent);
             transition: width .18s ease; }
  main { flex: 1 1 auto; display: flex; flex-direction: column;
         align-items: center; justify-content: center; gap: 22px;
         padding: 22px clamp(14px, 3vw, 40px) 28px; }
  .prose { max-width: 62ch; }
  h1 { font-size: clamp(23px, 3.4vw, 31px); font-weight: 600; margin: 0 0 14px;
       letter-spacing: -.015em; text-wrap: balance; }
  h2 { font-size: 15px; font-weight: 600; margin: 22px 0 7px; }
  p { line-height: 1.62; margin: 0 0 12px; color: #33373e; }
  .eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .13em;
             text-transform: uppercase; color: var(--muted); margin: 0 0 9px; }
  .ask { font-size: clamp(15px, 1.9vw, 18px); font-weight: 500; text-align: center;
         margin: 0; }
  .ask em { font-style: normal; color: var(--accent); font-weight: 600; }
  .pair { display: grid; grid-template-columns: 1fr 1fr;
          gap: clamp(12px, 2.4vw, 28px); width: 100%; max-width: 980px; }
  .card { background: var(--surface); border: 1px solid var(--rule); padding: 0;
          cursor: pointer; display: flex; flex-direction: column; gap: 0;
          font: inherit; color: inherit; transition: border-color .12s,
          box-shadow .12s, transform .12s; }
  .card img { width: 100%; height: auto; display: block; }
  .card .k { font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
             color: var(--muted); padding: 9px 0 11px; text-align: center;
             border-top: 1px solid var(--rule); }
  .card:hover, .card:focus-visible { border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft); outline: none; }
  .card:active { transform: translateY(1px); }
  .card[disabled] { cursor: default; opacity: .55; }
  .meta { font-family: var(--mono); font-size: 12px; color: var(--muted);
          display: flex; gap: 18px; justify-content: center;
          font-variant-numeric: tabular-nums; }
  button.go { font: 500 15px var(--sans); background: var(--accent); color: #fff;
              border: 0; padding: 12px 26px; cursor: pointer; }
  button.go:hover { background: #26493f; }
  button.go:focus-visible { outline: 3px solid var(--accent-soft);
                            outline-offset: 2px; }
  button.ghost { font: 400 13px var(--mono); background: transparent;
                 color: var(--muted); border: 1px solid var(--rule);
                 padding: 8px 14px; cursor: pointer; }
  .note { font-size: 13px; color: var(--muted); line-height: 1.55; }
  code { font-family: var(--mono); font-size: .92em; background: #e3e5e9;
         padding: 1px 5px; }
  textarea { width: 100%; height: 190px; font-family: var(--mono); font-size: 11px;
             border: 1px solid var(--rule); padding: 10px; background: var(--surface);
             color: var(--ink); }
  [hidden] { display: none !important; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  @media (max-width: 640px) { .pair { grid-template-columns: 1fr; } }
</style>

<div class="bar"><i id="bar"></i></div>
<main>
  <section id="intro" class="prose">
    <p class="eyebrow">Perceptual study &middot; about 4 minutes</p>
    <h1>Which build looks like a real Minecraft house?</h1>
    <p>You'll see two builds side by side, __N__ times. Each time, pick the one
      that looks more like a house a person actually built in Minecraft.</p>
    <p>Go with your first impression &mdash; there's no right answer on most
      pairs, and speed matters more than deliberation. Some builds are real,
      some were generated by a computer, and some are deliberately damaged.</p>
    <h2>Controls</h2>
    <p class="note">Click either image, or press <code>&larr;</code> /
      <code>&rarr;</code>. You can't go back to a previous pair.</p>
    <h2>What's recorded</h2>
    <p class="note">Which image you picked and how long you took. Nothing else,
      and nothing that identifies you.</p>
    <p><button class="go" id="begin">Begin</button></p>
  </section>

  <section id="trial" hidden style="display:contents">
    <p class="ask">Which looks more like a <em>real Minecraft house</em>?</p>
    <div class="pair">
      <button class="card" id="cardL"><img id="imgL" alt="Build option, left">
        <span class="k">&larr; LEFT</span></button>
      <button class="card" id="cardR"><img id="imgR" alt="Build option, right">
        <span class="k">RIGHT &rarr;</span></button>
    </div>
    <div class="meta"><span id="count"></span></div>
  </section>

  <section id="done" hidden class="prose">
    <p class="eyebrow">Complete</p>
    <h1>Thank you.</h1>
    <p id="status">Saving your responses&hellip;</p>
    <p class="note" id="fallback" hidden>Responses couldn't be saved
      automatically. Copy the text below and send it to whoever shared this
      link.</p>
    <textarea id="dump" hidden readonly></textarea>
    <p><button class="ghost" id="copy" hidden>Copy responses</button></p>
  </section>

  <section id="admin" hidden class="prose">
    <p class="eyebrow">Study administration</p>
    <h1>Collected responses</h1>
    <p id="admin-status" class="note">Loading&hellip;</p>
    <p><button class="go" id="download">Download all responses</button>
       <button class="ghost" id="refresh">Refresh</button></p>
    <textarea id="admin-dump" readonly></textarea>
  </section>
</main>

<script>
const STIM = __DATA__;
const $ = (id) => document.getElementById(id);
const sessionId = (crypto.randomUUID ? crypto.randomUUID()
                  : String(Date.now()) + Math.random().toString(16).slice(2));

/* Sample this session's trials without replacement. Every rater sees a
   different random subset, so coverage across the 36 arm pairs accumulates
   across raters rather than depending on any one of them finishing. */
function sample(arr, k) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, Math.min(k, a.length));
}
const trials = sample(STIM.trials, STIM.session_trials);
const responses = [];
let at = 0, shownAt = 0;

const src = (i) => "data:image/jpeg;base64," + STIM.images[i];

function preload(n) {
  const t = trials[n];
  if (!t) return;
  [t.left, t.right].forEach((i) => { const im = new Image(); im.src = src(i); });
}

function render() {
  const t = trials[at];
  $("imgL").src = src(t.left);
  $("imgR").src = src(t.right);
  $("count").textContent = `${at + 1} / ${trials.length}`;
  $("bar").style.width = (100 * at / trials.length) + "%";
  $("cardL").disabled = $("cardR").disabled = false;
  shownAt = performance.now();
  preload(at + 1);
}

function choose(side) {
  if (at >= trials.length || $("cardL").disabled) return;
  $("cardL").disabled = $("cardR").disabled = true;
  responses.push({
    trial: trials[at].id, chose: side,
    ms: Math.round(performance.now() - shownAt),
  });
  at += 1;
  if (at < trials.length) { render(); return; }
  $("bar").style.width = "100%";
  $("trial").hidden = true;
  $("done").hidden = false;
  submit();
}

$("cardL").addEventListener("click", () => choose("left"));
$("cardR").addEventListener("click", () => choose("right"));
addEventListener("keydown", (e) => {
  if ($("trial").hidden) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); choose("left"); }
  if (e.key === "ArrowRight") { e.preventDefault(); choose("right"); }
});
$("begin").addEventListener("click", () => {
  $("intro").hidden = true;
  $("trial").hidden = false;
  render();
});

const record = () => ({
  session: sessionId, stimuli_seed: STIM.meta.seed, git_sha: STIM.meta.git_sha,
  started: new Date().toISOString(), n: responses.length, responses,
});

async function submit() {
  const payload = record();
  let db = null;
  try { db = await claude.use("db"); } catch (e) { db = null; }
  if (!db) { offerManual(payload, "This page can't save responses here."); return; }
  try {
    await db.doc("responses/" + sessionId).set(payload);
    $("status").textContent =
      `Saved. ${responses.length} responses recorded. You can close this tab.`;
  } catch (e) {
    offerManual(payload, "Saving failed.");
  }
}

function offerManual(payload, why) {
  $("status").textContent = why;
  $("fallback").hidden = false;
  $("dump").hidden = false;
  $("copy").hidden = false;
  $("dump").value = JSON.stringify(payload);
}
$("copy").addEventListener("click", () => {
  $("dump").select();
  navigator.clipboard?.writeText($("dump").value);
  $("copy").textContent = "Copied";
});

/* Administration view: #admin in the URL. Only someone who can read the
   response collection sees anything, so this is a convenience rather than a
   control -- the permission lives in the capability rules. */
async function admin() {
  $("intro").hidden = true;
  $("admin").hidden = false;
  const db = await claude.use("db");
  if (!db) { $("admin-status").textContent = "Storage unavailable in this view."; return; }
  const load = async () => {
    try {
      const snap = await db.collection("responses").get();
      const docs = (snap.docs || []).map((d) => d.data ? d.data() : d);
      const total = docs.reduce((s, d) => s + (d.n || 0), 0);
      $("admin-status").textContent =
        `${docs.length} sessions, ${total} judgements.`;
      $("admin-dump").value = JSON.stringify(docs, null, 1);
      return docs;
    } catch (e) {
      $("admin-status").textContent = "Could not read responses: " + e.message;
      return [];
    }
  };
  let docs = await load();
  $("refresh").addEventListener("click", async () => { docs = await load(); });
  $("download").addEventListener("click", async () => {
    const dl = await claude.use("downloads");
    const data = JSON.stringify(docs, null, 1);
    if (!dl) { $("admin-dump").select(); return; }
    await dl.save({ filename: "human_study_responses.json", data });
  });
}
if (location.hash === "#admin") admin();
</script>
"""


def build_page(payload: dict) -> str:
    """Render the rater-facing page with the answer key stripped out.

    The trial list the browser receives carries only `id`, `left` and `right`.
    Arm names never reach the DOM, so a rater cannot read the condition off the
    page source, and a screenshot of the page cannot leak it either. The join
    back to arms happens offline against `answer_key.json`.
    """
    import json as _json

    blind = {
        "images": payload["images"],
        "session_trials": payload["session_trials"],
        "meta": {"seed": payload["meta"]["seed"],
                 "git_sha": payload["meta"]["git_sha"]},
        "trials": [{"id": t["id"], "left": t["left"], "right": t["right"]}
                   for t in payload["trials"]],
    }
    return (PAGE
            .replace("__N__", str(payload["session_trials"]))
            .replace("__DATA__", _json.dumps(blind, separators=(",", ":"))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-arm", type=int, default=12, help="builds rendered per arm")
    ap.add_argument("--per-pair", type=int, default=8, help="trials per arm pair")
    ap.add_argument("--px", type=int, default=288)
    ap.add_argument("--arms", nargs="*", default=[],
                    help="extra arms as name:path.npz")
    ap.add_argument("--session-trials", type=int, default=40,
                    help="how many trials one participant is shown")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    extra = dict(a.split(":", 1) for a in args.arms)
    arms = collect_arms(args.corpus, args.seed, args.per_arm, extra)
    print(f"[study] {len(arms)} arms: {', '.join(arms)}", flush=True)

    images: List[str] = []
    index: Dict[str, List[int]] = {}
    for name, structs in arms.items():
        index[name] = []
        for s in structs:
            index[name].append(len(images))
            images.append(to_jpeg(render_one(s, args.px)))
        print(f"  rendered {name}: {len(index[name])}", flush=True)

    rng = np.random.default_rng(args.seed)
    trials = make_trials(index, rng, args.per_pair)
    n_catch = sum(t["catch"] for t in trials)
    print(f"[study] {len(trials)} trials over {len(arms)*(len(arms)-1)//2} pairs "
          f"({n_catch} catch)", flush=True)

    run_dir = Path(args.out) if args.out else new_run_dir("human_study")
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "images": images, "trials": trials, "index": index,
        "session_trials": args.session_trials,
        "meta": {"corpus": args.corpus, "seed": args.seed, "px": args.px,
                 "per_arm": args.per_arm, "per_pair": args.per_pair,
                 "catch_pair": list(CATCH), "git_sha": sc.git_sha()},
    }
    (run_dir / "stimuli.json").write_text(json.dumps(payload))
    # The arm labels are the answer key and must not travel with the stimuli a
    # rater sees; the page is built from a copy with them stripped.
    key = {"trials": [{"id": t["id"], "left_arm": t["left_arm"],
                       "right_arm": t["right_arm"], "catch": t["catch"]}
                      for t in trials], "index": index, "meta": payload["meta"]}
    (run_dir / "answer_key.json").write_text(json.dumps(key, indent=2))

    page = build_page(payload)
    (run_dir / "study.html").write_text(page)

    total_mb = sum(len(i) for i in images) / 1e6
    print(f"[study] stimuli {total_mb:.1f} MB base64 across {len(images)} images")
    print(f"[study] page {len(page) / 1e6:.1f} MB (arm labels stripped)")
    print(f"-> {run_dir}/stimuli.json")
    print(f"-> {run_dir}/answer_key.json")
    print(f"-> {run_dir}/study.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
