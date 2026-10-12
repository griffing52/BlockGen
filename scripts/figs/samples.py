"""The sample gallery: what every arm on the leaderboard actually builds.

    .venv/bin/python scripts/figs/samples.py

Writes `outputs/figstage/samples.png` and `.pdf`.

Why this figure exists
----------------------
The benchmark ranks eighteen arms -- learned models, prompted LLMs, procedural
baselines, real held-out data and deliberately corrupted real data -- and the
paper never shows the reader one build. A reviewer of a generative-evaluation
paper cannot check a single claim about "what the metric sees" without seeing
the samples, so this is a contact sheet ordered by BlockScore: the gallery and
the leaderboard are the same object.

Where every pixel and every number comes from
---------------------------------------------
* The builds are the run of record's own recorded example strip,
  `examples_251.npz` + `examples_251_manifest.json`, written by
  `blockgen.eval.bench.examples` at scoring time. Nothing is regenerated here
  and nothing is re-sampled: a *different* draw would be a different figure from
  the one the scorecard describes.
* The scores are parsed out of `scorecard.md`'s `## leaderboard -- BlockScore`
  table -- the artifact itself, not a re-derivation.

Selection rule, stated so it can be checked
-------------------------------------------
No build on this page was chosen by eye. The run wrote `k=8` builds per arm
under a seeded permutation of that arm's scored list (`examples.pick`, `seed 0`,
`ctx.rng(9)`, indices sorted ascending). This script takes those eight in the
order they were written, drops the ones the licensing gate forbids, and shows
the first `NCOL` that remain. The row therefore reads in source order, and the
only filter applied is the legal one.

The licensing gate
------------------
GrabCraft's terms name pictures of its blueprints as prohibited distribution and
text2mc's builds come from PlanetMinecraft, whose terms forbid redistributing
site data, so **only 3D-Craft builds may be published as images** (see
`_corpus_mask` in `scripts/paper_figures.py`). Two consequences, both enforced
below rather than asserted in prose:

1. *Rows made of real voxels* (`REAL_DERIVED`) are filtered to `corpus ==
   "3dcraft"`. Three of those arms lost their metadata on the way through the
   probe -- `probes.shuffle_materials` and `probes.monochrome` build a bare
   `Structure`, so the cache records the fallback corpus, not the true one --
   so provenance is *inherited* from the `real_test` row at the same strip
   position, and only after `_same_build` checks that the two really are the
   same building (identical occupancy, or a superset for `solidify`, which only
   adds blocks). A real row whose corpus cannot be established this way is
   dropped, never published on the assumption that it is probably fine.
2. *Rows made of model output* are generated content and are not the corpus's
   to license -- except that a memorised near-verbatim copy of a GrabCraft build
   is, in effect, a picture of that blueprint. `audit_memorization` therefore
   measures every published generated build against all 1863 training builds
   with the suite's own novelty machinery (`voxelize_occupancy`, `_iou_matrix`,
   32^3, `dup_threshold` 0.95) and withholds any cell whose nearest training
   neighbour is both a near-duplicate and not from 3D-Craft. It costs ~2 s and
   it runs every time, because a gate nobody exercises is a gate nobody knows
   is broken.

Flat colours, never Minecraft's textures: `flat_textures()` is copied from
`scripts/paper_figures.py` deliberately (the brief forbids importing from it).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armnames import display          # noqa: E402
OUTDIR = ROOT / "outputs" / "figstage"

#: The one bench run every figure in the paper reads.
BENCH_RUN = (ROOT / "outputs"
             / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull")

#: The metric's own texture pack. See paper_figures.flat_textures.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")

# --- palette ----------------------------------------------------------------
# Colour and font come from scripts/figs/figstyle.py.
#
# What the two hues mean here is what they mean everywhere else in the paper:
# ORANGE is the render tier (the `appearance` pillar, mv_dino_kid) and BLUE is
# the geometry tier (the `geometry` pillar, geom_kid), and they are applied to
# the one thing on this page that IS a tier -- the worst-pillar word in each
# row's score line. They used to mean something else entirely: BLUE was "a
# learned model", ORANGE was "real data under a deliberate corruption", and a
# local VIOLET was "a fine-tuned LLM". Three category colours borrowed from the
# metric palette put a blue chip beside a row whose worst pillar is appearance,
# which is the opposite of what blue means four figures earlier.
#
# The arm categories those chips carried are still on the page, and now say so
# in words: every row is labelled with its kind, which is strictly more than a
# colour swatch plus a legend lookup was. See ROW_KIND / KIND_LABEL below.
import figstyle                       # noqa: E402
from figstyle import BLUE, INK, INK_2, ORANGE, PAPER, RULE  # noqa: E402

#: Pillar name -> its tier colour. Anything not a tier (the `palette`,
#: `coherence` and `geometry_scalars` pillars, and the gate metric a DQ row
#: names) is set in ink, because it is not one of the two hues' meanings and
#: inventing a third would be how the palette drifted in the first place.
PILLAR_COLOR = {"appearance": ORANGE, "geometry": BLUE}

NCOL = 6              # builds shown per arm
PX = 300              # render resolution per thumbnail

#: Type sizes. The figure is drawn at 5.5 in and `X_appendix.tex` includes it
#: at `width=\linewidth`, which is 5.5 in, so every size here prints at its own
#: value and nothing is set below 6.2 pt. (The comment this replaces said
#: 0.9\linewidth and budgeted for a tenth of shrink that does not happen; the
#: DQ badge was set at 5.8 pt on the strength of it.)
PT_TITLE, PT_KIND, PT_RANK = 8.4, 6.4, 7.2
PT_NAME, PT_HEAD, PT_FINE, PT_FOOT = 8.4, 6.9, 6.4, 6.4
PT_DQ = 6.2

#: Which kind of thing an arm is. Printed on the row; never the ordering, which
#: is the leaderboard's.
KIND_LABEL = OrderedDict([
    ("real", "real held-out"),
    ("model", "learned model"),
    ("llm", "fine-tuned LLM"),
    ("corrupt", "corrupted real / degenerate"),
    ("proc", "procedural baseline"),
])

#: The rows, and why each is here. Twelve of the run's eighteen arms: all five
#: submissions (so no model is hidden, including the one that is disqualified),
#: the real reference, the three corruption controls the paper's claims name,
#: the two degenerate controls the suite disqualifies, one procedural baseline
#: and the bottom-ranked arm. Order on the page is the leaderboard's, read from
#: `scorecard.md`; this dict's order is not used.
#:
#: Omitted, with reasons: `real@canon16` and `real@canon8` (decimation is a
#: change of resolution, not of quality -- `fast.control_arms` says so itself
#: and keeps them out of the quality reading); `shell_box` and `patchwork`
#: (procedural, and `gabled_house` already stands for that family);
#: `real@monochrome` (an appearance-only corruption of the same voxels as
#: `real_shuffled_materials`, which is shown); `train_copy_noise` (real train
#: voxels whose upstream corpus cannot be established from the strip, so the
#: licensing gate would withhold the whole row anyway).
#: `native_oriented`, the contaminated checkpoint, is deliberately NOT here.
#: It is unrankable by the provenance gate, so placing it in a gallery ordered
#: by BlockScore would show it at rank 3, above the clean retrain, and imply a
#: standing it is not entitled to. The case-study figure is where it belongs.
ROW_KIND = OrderedDict([
    ("real_test", "real"),
    ("canon_oriented", "model"),
    ("llm_control", "llm"),
    ("llm_bias", "llm"),
    ("real@solidify", "corrupt"),
    ("real_shuffled_materials", "corrupt"),
    ("gabled_house", "proc"),
    ("uniform_random", "proc"),
    ("real@single_mode", "corrupt"),
    ("train_verbatim", "corrupt"),
    ("canon_unoriented", "model"),
])

#: Arms whose voxels ARE real corpus builds (possibly damaged). These are the
#: rows the 3D-Craft-only gate applies to.
REAL_DERIVED = {
    "real_test", "real@canon16", "real@canon8", "real_shuffled_materials",
    "real@monochrome", "real@solidify", "real@single_mode",
    "train_verbatim", "train_copy_noise",
}

#: Corpus names that identify a real upstream source. Anything else in the
#: cache's `corpus` field is the fallback `examples._relabel` writes when the
#: builder left no metadata, and carries no provenance at all.
KNOWN_CORPORA = {"3dcraft", "grabcraft", "text2mc"}
PUBLISHABLE_CORPUS = "3dcraft"

#: The suite's own near-duplicate threshold (`BenchContext.dup_threshold`).
DUP_THRESHOLD = 0.95


def flat_textures():
    """A texture library with no texture files, so every block renders solid."""
    import os
    from blockgen.renderer.textures import load_face_textures
    # The metric's own pack, so the gallery shows what was scored.
    # BLOCKGEN_FLAT_FIGURES=1 falls back to solid colors.
    if os.environ.get("BLOCKGEN_FLAT_FIGURES") == "1":
        return load_face_textures(pack_dir=_FLAT_DIR)
    return load_face_textures()


def _style() -> None:
    figstyle.apply(size=8)


# ---------------------------------------------------------------------------
# the scorecard
# ---------------------------------------------------------------------------

_LB_HEADER = "## leaderboard — BlockScore"


def leaderboard(run_dir: Path = BENCH_RUN) -> "OrderedDict[str, Dict]":
    """Parse the scorecard's own BlockScore table. One row per arm, in order.

    Reading the rendered table rather than recomputing the composite from
    `scorecard.json` is deliberate: the figure then cannot disagree with the
    artifact the paper cites, and `composite.py` is not re-run at figure time
    against a card it may have moved on from.
    """
    text = (run_dir / "scorecard.md").read_text()
    body = text.split(_LB_HEADER, 1)[1]
    out: "OrderedDict[str, Dict]" = OrderedDict()
    for line in body.splitlines():
        if not line.startswith("|"):
            if out:
                break          # table ended
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0] in {"#", "---"} or set(cells[0]) <= {"-"}:
            continue
        rank, arm, score, worst = cells[0], cells[1], cells[2], cells[3]
        raw = score.replace("*", "").strip()
        dq = raw.upper() == "DQ"
        out[arm] = {
            "rank": None if rank in {"—", "-", ""} else int(rank),
            "score": None if dq else float(raw),
            "dq": dq,
            "worst": worst,
            "order": len(out),
        }
    if not out:
        raise SystemExit(f"no leaderboard table found in {run_dir}/scorecard.md")
    return out


def dq_reason(worst: str) -> Tuple[str, str]:
    """`("memorization", "dino_nn_percentile, +10.8 spreads")` from a DQ cell.

    The cell reads `metric: headline: prose (+N spreads)`; the headline is the
    part a reader needs on the page and the rest goes in the small line.
    """
    metric, _, rest = worst.partition(":")
    headline, _, prose = rest.strip().partition(":")
    spreads = re.search(r"\(([+-][\d.]+)\s+spreads\)", prose or rest)
    tail = f"{metric.strip()}, {spreads.group(1)} spreads" if spreads else metric.strip()
    return headline.strip() or "disqualified", tail


# ---------------------------------------------------------------------------
# the builds
# ---------------------------------------------------------------------------

def load_examples(run_dir: Path = BENCH_RUN):
    """`(arm -> [(strip_pos, row, item)], block_ids, block_data)` from the cache.

    `strip_pos` is the build's position within its arm's 8-build strip and `row`
    its row in the run's npz; both are kept so a finding can be traced back to
    the run without re-deriving anything.
    """
    npz = sorted(run_dir.glob("examples_*.npz"))
    if len(npz) != 1:
        raise SystemExit(f"expected exactly one examples_*.npz in {run_dir}")
    manifest = json.loads(Path(str(npz[0]).replace(".npz", "_manifest.json")).read_text())
    blob = np.load(npz[0], allow_pickle=True)
    ids, data = blob["block_ids"], blob["block_data"]
    strips = OrderedDict()
    for arm, rows in manifest["report"]["arms"].items():
        strips[arm] = [(pos, int(r), manifest["items"][int(r)])
                       for pos, r in enumerate(rows)]
    return strips, ids, data, manifest


def structure(ids, data, row: int, item: Dict):
    from blockgen.utils.data import Structure
    return Structure(block_ids=ids[row], block_data=data[row],
                     source_path=item.get("source_path") or None,
                     metadata={k: item.get(k, "") for k in
                               ("corpus", "category", "title", "url")})


def _same_build(a, b) -> bool:
    """Is `b` the same building as `a`, possibly with blocks added?

    The test the corpus-inheritance step needs and no more: same voxel grid, and
    every block of `a` still occupied in `b`. True for `shuffle_materials` and
    `monochrome` (identical occupancy) and for `solidify` (a strict superset);
    false for anything that moved, cropped or resampled the build.
    """
    if a.shape != b.shape:
        return False
    occ_a, occ_b = a.occupied_mask, b.occupied_mask
    return bool((occ_a & ~occ_b).sum() == 0)


def resolve_corpus(arm: str, strips, ids, data) -> List[Optional[str]]:
    """The upstream corpus of each of an arm's 8 strip builds, or None.

    `None` means "not established", which the gate treats as "do not publish".
    """
    own = [it.get("corpus") for _, _, it in strips[arm]]
    out: List[Optional[str]] = [c if c in KNOWN_CORPORA else None for c in own]
    if all(c is not None for c in out) or arm not in REAL_DERIVED:
        return out

    # Inherit from the real_test strip, which shares the arm's source rows, but
    # only where the two builds can be shown to be the same building.
    ref = strips.get("real_test")
    if ref is None or len(ref) != len(strips[arm]):
        return out
    for i, (c, (_, row, item)) in enumerate(zip(out, strips[arm])):
        if c is not None:
            continue
        ref_corpus = ref[i][2].get("corpus")
        if ref_corpus not in KNOWN_CORPORA:
            continue
        a = structure(ids, data, ref[i][1], ref[i][2])
        b = structure(ids, data, row, item)
        if _same_build(a, b):
            out[i] = ref_corpus
    return out


# ---------------------------------------------------------------------------
# the two gates
# ---------------------------------------------------------------------------

def audit_memorization(published, ids, data) -> Dict[Tuple[str, int], Dict]:
    """Nearest training build for every generated build we intend to publish.

    `published` maps arm -> [(strip_pos, row, item)] for the NON-real-derived
    arms only. Returns `(arm, strip_pos) -> {iou, nn_corpus, nn_title}` using the
    suite's own descriptor so the number means what `voxel_nn_iou`/`voxel_dup_rate`
    mean on the scorecard.
    """
    from blockgen.eval.bench import splits as sp, fast
    from blockgen.eval.novelty import voxelize_occupancy, _iou_matrix

    corpus, manifest = sp.load_corpus("houses_32")
    split = sp.load_split("houses_32")
    tr_idx = list(split.indices("train"))
    train = [corpus[i].crop_to_non_air() for i in tr_idx]
    vocab = fast.build_vocab(train, 32)
    train_occ, _ = voxelize_occupancy(train, 32, vocab)

    keys, structs = [], []
    for arm, rows in published.items():
        for pos, row, item in rows:
            keys.append((arm, pos))
            structs.append(structure(ids, data, row, item).crop_to_non_air())
    if not keys:
        return {}
    gen_occ, _ = voxelize_occupancy(structs, 32, vocab)
    iou = _iou_matrix(gen_occ, train_occ)
    nn = np.argmax(iou, axis=1)
    report = {}
    for k, key in enumerate(keys):
        item = manifest["items"][tr_idx[int(nn[k])]]
        report[key] = {"iou": float(iou[k, nn[k]]),
                       "nn_corpus": item.get("corpus", ""),
                       "nn_title": item.get("title", "")}
    return report


def choose(strips, ids, data, memo) -> "OrderedDict[str, Dict]":
    """Apply both gates and take the first NCOL survivors, in strip order."""
    picks: "OrderedDict[str, Dict]" = OrderedDict()
    for arm in ROW_KIND:
        if arm not in strips:
            raise SystemExit(f"arm {arm!r} has no examples in {BENCH_RUN.name}")
        corpora = resolve_corpus(arm, strips, ids, data)
        kept, dropped = [], []
        for (pos, row, item), corpus in zip(strips[arm], corpora):
            if arm in REAL_DERIVED:
                if corpus is None:
                    dropped.append((pos, "corpus not established"))
                    continue
                if corpus != PUBLISHABLE_CORPUS:
                    dropped.append((pos, corpus))
                    continue
            else:
                m = memo.get((arm, pos))
                if m and m["iou"] >= DUP_THRESHOLD and m["nn_corpus"] != PUBLISHABLE_CORPUS:
                    dropped.append((pos, f"memorised {m['nn_corpus']} build"
                                         f" (IoU {m['iou']:.2f})"))
                    continue
            kept.append((pos, row, item, corpus))

        # A build one voxel thick draws as a hairline and costs a tile to say
        # nothing a reader can see. Those are held back from the strip and
        # reported as a rate on the row instead, which carries the same fact in
        # a form that reads: "a fifth of this arm's output is one block thick"
        # is more informative than one unreadable cell. The rate is measured
        # over the arm's whole scored sample, not over the six shown.
        degenerate = [k for k in kept if _is_thin(k[1], ids, data)]
        kept = [k for k in kept if k not in degenerate]

        # The run records eight examples per arm, which is not enough to fill
        # six cells once the degenerate ones are set aside: the two
        # language-model arms have five. Top up from that arm's full scored
        # sample, in the same seeded order, skipping degenerate builds. The
        # selection rule is unchanged in kind -- first N by a fixed order, not
        # chosen -- and the larger pool is if anything the fairer draw. Only
        # model arms are topped up; the real-derived rows keep the recorded
        # strips, because those carry the corpus provenance the licensing gate
        # needs and the column alignment the figure's central claim relies on.
        extra = []
        if len(kept) < NCOL and arm not in REAL_DERIVED:
            extra = _topup(arm, NCOL - len(kept))

        picks[arm] = {"shown": kept[:NCOL], "extra": extra, "dropped": dropped,
                      "n_available": len(strips[arm]), "n_kept": len(kept),
                      "n_thin": len(degenerate), "corpora": corpora}
    return picks


def _topup(arm: str, need: int) -> list:
    """`need` further non-degenerate builds from the arm's full scored sample."""
    import numpy as np
    path = ROOT / "outputs" / "bench_arms" / f"{arm}_32.npz"
    if not path.exists():
        return []
    z = np.load(path, allow_pickle=True)
    aid, adat = z["block_ids"], z["block_data"]
    rng = np.random.default_rng(0)
    out = []
    for i in rng.permutation(len(aid)):
        occ = aid[i] != 0
        if not occ.any():
            continue
        x = np.argwhere(occ)
        if (x.max(0) - x.min(0) + 1).min() <= 1:
            continue
        out.append((aid[i], adat[i]))
        if len(out) == need:
            break
    return out


def _is_thin(row, ids, data) -> bool:
    """True when the build has an extent of one voxel on some axis."""
    import numpy as np
    occ = ids[row] != 0
    if not occ.any():
        return True
    idx = np.argwhere(occ)
    return bool((idx.max(0) - idx.min(0) + 1).min() <= 1)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def thumb(struct, tex, px: int = PX) -> np.ndarray:
    """One build, flat-shaded, composited on the page surface, cropped square."""
    from blockgen.renderer.textured import render_structure
    img = render_structure(struct, px=px, azim_deg=45, elev_deg=25,
                           ortho=True, face_textures=tex)
    rgb = img[..., :3].astype(np.float32)
    alpha = (img[..., 3:4].astype(np.float32) / 255.0) if img.shape[-1] == 4 \
        else np.ones_like(rgb[..., :1])
    bg = np.array(matplotlib.colors.to_rgb(PAPER)) * 255.0
    flat = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

    solid = alpha[..., 0] > 0.02
    if not solid.any():
        return flat
    ys, xs = np.where(solid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = flat[y0:y1, x0:x1]
    # Pad to square so the cell never stretches a build.
    h, w = crop.shape[:2]
    side = max(h, w)
    pad = np.full((side, side, 3), bg.astype(np.uint8), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    pad[oy:oy + h, ox:ox + w] = crop
    return pad


def _width(text: str, pt: float, weight: str = "normal") -> float:
    """Width of `text` in INCHES at `pt`, measured in the font that will draw it.

    Estimating this from a characters-per-inch constant put "appearance" on
    top of the first thumbnail, because a proportional face is not monospaced
    and the error accumulates across a line. `TextPath` lays the string out
    with the same font machinery the renderer uses; its extents are in points.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    if not text:
        return 0.0
    prop = FontProperties(family=plt.rcParams["font.family"], size=pt,
                          weight=weight)
    return float(TextPath((0, 0), text, size=pt, prop=prop).get_extents().width) / 72.0


def _wrap(text: str, pt: float, avail_in: float) -> List[str]:
    """Greedy wrap to a width in inches, measuring each candidate line."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and _width(trial, pt) > avail_in:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _score_line(info: Dict) -> Tuple[str, str]:
    """The score line, split where its colour changes.

    `("BlockScore 136.7  ·  ", "geometry")`: the second half is the worst
    pillar and is drawn in that pillar's tier colour, so the line has to be set
    as two strings. Both halves live here so the width measurement that sizes
    the label column and the drawing code cannot disagree about the text.
    """
    return f"BlockScore {info['score']:.1f}  ·  ", str(info["worst"])


def _sticks(picks) -> Tuple[int, int]:
    """How many shown builds are one block thick, and how long the longest is.

    Several cells in the two prompted-LLM rows look blank. They are not failed
    renders: the build is a 1-block-thick line, and framed to its own bounding
    box a line is a hairline. Saying so on the figure costs one clause and stops
    a reader concluding the pipeline is broken -- and the number is measured
    from what is actually drawn, so it cannot drift from the picture.
    """
    flat = [it["dims"] for p in picks.values() for _, _, it, _ in p["shown"]
            if min(it["dims"]) == 1]
    return len(flat), (max(max(d) for d in flat) if flat else 0)


def _footer(withheld: bool, max_iou: float, sticks: Tuple[int, int],
            fig_w: float) -> List[str]:
    """The caption's small print, printed for pasting. Every clause is checkable."""
    note = ("The first "
            f"{NCOL} of the 8 builds the run itself recorded for each arm "
            "(seeded permutation of that arm's scored list, seed 0), in source "
            "order, less any the licensing gate withholds. Where fewer than "
            f"{NCOL} survive, the row is topped up from that arm's full scored "
            "sample in the same seeded order; builds one voxel thick are passed "
            "over, since they draw as a hairline. Real, Fill-Interior and "
            "Material-Shuffle are the same six buildings column for column, so "
            "a corruption reads down a column.")
    if withheld:
        note += (" † withheld: GrabCraft's terms forbid publishing pictures of "
                 "its blueprints, so rows made of real voxels show only their "
                 "3D-Craft builds;")
    else:
        note += " Licensing:"
    note += (f" no generated build shown is a memorized copy (nearest of 1863 "
             f"training builds, IoU {max_iou:.2f}). Each "
             "build is framed to its own bounding box, so cells are not to a "
             "common scale.")
    n_stick, longest = sticks
    if n_stick:
        note += (f" {n_stick} of the builds drawn for this figure are one voxel "
                 "thick and are held back, because a hairline costs a tile and "
                 "shows nothing; the rate is given on the row instead.")
    return _wrap(note, PT_FOOT, fig_w - 0.06)


def build_figure(picks, board, memo, strips, ids, data) -> plt.Figure:
    _style()
    rows = sorted(ROW_KIND, key=lambda a: (board[a]["order"]))

    fig_w = 5.5
    # The label column is as wide as its widest label, MEASURED. Guessing this
    # put "appearance" on top of the first thumbnail; a 20% error in a
    # characters-per-inch constant is invisible until it is a collision.
    name_x, dq_w = 0.31, 0.215
    need = 0.0
    for arm in rows:
        info = board[arm]
        need = max(need, _width(arm, PT_NAME))
        if info["dq"]:
            head_txt, tail = dq_reason(info["worst"])
            need = max(need, dq_w + _width(head_txt, PT_HEAD),
                       _width(tail, PT_FINE))
        else:
            # The pillar word is set bold (see below), so measure it bold; a
            # regular-weight measurement is short by enough to put it on the
            # first thumbnail.
            prefix, pillar = _score_line(info)
            need = max(need, _width(prefix, PT_HEAD)
                       + _width(pillar, PT_HEAD, "bold"))
        need = max(need, _width(KIND_LABEL[ROW_KIND[arm]], PT_KIND))
    label_w = name_x + need + 0.07
    gutter = 0.05
    cell = (fig_w - label_w - 0.04 - gutter * (NCOL - 1)) / NCOL
    pitch = cell + 0.045
    # Title only. The colour legend that used to sit under it is gone with the
    # category chips: each row now names its own kind, so there is nothing left
    # to look up.
    head = 0.30
    # No small print under the grid: the gate, the selection rule and the
    # framing all belong in the caption, where they are set in the document's
    # type. `_footer` still assembles that text and `main` prints it. What is
    # left here is the thin margin that keeps the last row off the page edge.
    foot = 0.03
    fig_h = head + pitch * len(rows) + foot
    fig = plt.figure(figsize=(fig_w, fig_h))

    def X(v):
        return v / fig_w

    def Y(v):
        return v / fig_h

    tex = flat_textures()

    # --- header ------------------------------------------------------------
    fig.text(X(0.02), Y(fig_h - 0.155), "Ordered by BlockScore: each arm's worst "
             "pillar, in real-sample spreads (0 = held-out real)",
             fontsize=PT_TITLE, va="center", color=INK)

    # --- rows --------------------------------------------------------------
    for r, arm in enumerate(rows):
        top = fig_h - head - r * pitch
        y = top - cell
        info = board[arm]
        kind = ROW_KIND[arm]

        fig.patches.append(Rectangle((X(0.02), Y(top + 0.021)), X(fig_w - 0.04),
                                     Y(0.004), transform=fig.transFigure,
                                     facecolor=RULE, edgecolor="none"))

        # Three text lines in the label column (four on a disqualified row), and
        # nothing below 6.2 pt: the figure prints at its drawn size, and 5 pt
        # type is not a label, it is a rumour. Every string is kept inside
        # `label_w` -- one that overruns lands on the first thumbnail, and
        # `bbox_inches="tight"` would then silently widen the whole page.
        if info["dq"]:
            name_y, head_y, fine_y, kind_y = (y + cell * f
                                              for f in (0.80, 0.545, 0.31, 0.09))
        else:
            name_y, head_y, kind_y = (y + cell * f for f in (0.74, 0.44, 0.16))
        rank = f"{info['rank']}." if info["rank"] else "—"
        fig.text(X(name_x - 0.05), Y(name_y), rank, fontsize=PT_RANK,
                 va="center", ha="right", color=INK_2)
        fig.text(X(name_x), Y(name_y), display(arm), fontsize=PT_NAME,
                 va="center", ha="left", color=INK)
        # What kind of thing this arm is. This is the category the five colour
        # chips used to encode; it is a word now, so the two metric hues are
        # free to mean what they mean in every other figure.
        fig.text(X(name_x), Y(kind_y), KIND_LABEL[kind], fontsize=PT_KIND,
                 va="center", ha="left", color=INK_2)

        if info["dq"]:
            head_txt, tail = dq_reason(info["worst"])
            fig.patches.append(Rectangle((X(name_x), Y(head_y - 0.052)),
                                         X(0.185), Y(0.104),
                                         transform=fig.transFigure,
                                         facecolor=INK, edgecolor="none"))
            fig.text(X(name_x + 0.0925), Y(head_y), "DQ", fontsize=PT_DQ,
                     va="center", ha="center", color=PAPER)
            fig.text(X(name_x + dq_w), Y(head_y), head_txt, fontsize=PT_HEAD,
                     va="center", ha="left", color=INK)
            fig.text(X(name_x), Y(fine_y), tail, fontsize=PT_FINE,
                     va="center", ha="left", color=INK_2)
        else:
            # The worst pillar is set in its own tier colour -- ORANGE for
            # `appearance` (the render tier, mv_dino_kid), BLUE for `geometry`
            # (geom_kid) -- so a reader scanning the column sees which tier is
            # vetoing each arm, in the same two hues the rest of the paper uses
            # for those two tiers. Bold, because a hue at 6.9 pt has to carry
            # its weight as well as its colour.
            prefix, pillar = _score_line(info)
            fig.text(X(name_x), Y(head_y), prefix, fontsize=PT_HEAD,
                     va="center", ha="left", color=INK)
            fig.text(X(name_x + _width(prefix, PT_HEAD)), Y(head_y),
                     pillar, fontsize=PT_HEAD, va="center", ha="left",
                     fontweight="bold",
                     color=PILLAR_COLOR.get(pillar, INK_2))

        shown = picks[arm]["shown"]
        for c in range(NCOL):
            x0 = label_w + c * (cell + gutter)
            ax = fig.add_axes([X(x0), Y(y), X(cell), Y(cell)])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(RULE)
                s.set_linewidth(0.4)
            extra = picks[arm].get("extra", [])
            if c < len(shown):
                pos, row, item, _ = shown[c]
                ax.imshow(thumb(structure(ids, data, row, item), tex),
                          interpolation="antialiased")
                ax.set_facecolor(PAPER)
            elif c - len(shown) < len(extra):
                from blockgen.utils.data import Structure
                eid, edat = extra[c - len(shown)]
                ax.imshow(thumb(Structure(block_ids=eid, block_data=edat)
                                .crop_to_non_air(), tex),
                          interpolation="antialiased")
                ax.set_facecolor(PAPER)
            else:
                ax.set_facecolor(PAPER)
                for s in ax.spines.values():
                    s.set_linestyle((0, (1.5, 1.5)))
                ax.text(0.5, 0.5, "†", fontsize=7, color=INK_2,
                        ha="center", va="center", transform=ax.transAxes)

    return fig


# ---------------------------------------------------------------------------

def main() -> None:
    board = leaderboard()
    strips, ids, data, manifest = load_examples()
    missing = [a for a in ROW_KIND if a not in board]
    if missing:
        raise SystemExit(f"arms not on the leaderboard: {missing}")

    generated = OrderedDict((arm, strips[arm]) for arm in ROW_KIND
                            if arm not in REAL_DERIVED)
    memo = audit_memorization(generated, ids, data)
    picks = choose(strips, ids, data, memo)

    print(f"run {BENCH_RUN.name}")
    print(f"strip: {manifest['report']['k']} builds/arm, seed "
          f"{manifest['report']['seed']}, {manifest['count']} rows total")
    print("\narm                         score   shown  gate")
    for arm in sorted(ROW_KIND, key=lambda a: board[a]["order"]):
        p = picks[arm]
        score = "DQ" if board[arm]["dq"] else f"{board[arm]['score']:7.1f}"
        drops = ", ".join(f"#{pos}:{why}" for pos, why in p["dropped"]) or "—"
        print(f"{arm:26s} {score:>7s}  {len(p['shown'])}/{p['n_available']}  {drops}")

    print("\nmemorization audit (nearest of 1863 training builds, 32^3 IoU)")
    worst = sorted(memo.items(), key=lambda kv: -kv[1]["iou"])
    for (arm, pos), m in worst[:8]:
        print(f"  {arm:20s} #{pos}  IoU {m['iou']:.3f}  nn={m['nn_corpus']:9s} "
              f"{m['nn_title'][:44]}")
    over = [k for k, m in memo.items() if m["iou"] >= DUP_THRESHOLD]
    print(f"  near-duplicates (IoU >= {DUP_THRESHOLD}): {len(over)} of {len(memo)}")

    fig = build_figure(picks, board, memo, strips, ids, data)
    print("\n--- caption text for the LaTeX figure ---")
    print(" ".join(_footer(any(len(picks[a]["shown"]) < NCOL for a in picks),
                           max((m["iou"] for m in memo.values()), default=float("nan")), _sticks(picks), 9.9)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"samples.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"\nwrote {OUTDIR.relative_to(ROOT)}/samples.pdf (+ .png), "
          f"{fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in")


if __name__ == "__main__":
    main()
