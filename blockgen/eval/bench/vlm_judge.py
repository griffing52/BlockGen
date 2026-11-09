"""VLM-judge arm: a pairwise vision-language judge, run on the benchmark's own stimuli.

Why this exists
---------------
The paper's claim is that render-and-embed metrics (multi-view KID on DINOv2/CLIP
features of four exterior renders) cannot see a building's interior: `solidify`
(Fill-Interior) fills every sealed room of a real house and the four exterior
views do not change. The newest evaluator family for 3D generation is not a
feature distance but a *VLM judge* shown renders and asked which of two objects
is better (GPTEval3D, 3DGen-Bench). This module asks whether that family inherits
the same observation limit, and gives a system-level ranking of the leaderboard
arms that the metrics can be correlated against.

Design, each choice for a reason
--------------------------------
* **The metric's own pixels.** `ext4` is `features.render_views` with the default
  `ViewConfig` -- the four orthographic views at azimuth 45/135/225/315, 224 px,
  the textured renderer, white background -- tiled 2x2. A judge shown anything
  else answers a different question from the one the paper asks of the metric.
* **The leaderboard's own builds.** Arms are rebuilt exactly as the leaderboard run
  of record (`RUN_OF_RECORD`) built them, and every arm's `structures_sha` is
  checked against that scorecard before a single call, so the VLM ranking and the
  metric numbers it is correlated with are about the identical structure sets.
* **Paired probes.** `real_test[i]` is compared with `solidify(real_test[i])`, the
  same house, so a preference can only mean the judge saw the alteration (the
  human study learned this the hard way; see `scripts/human_study_export.py`).
* **Both orders.** Every build pair is shown A/B and B/A. Position bias is measured
  (P(first slot)), and the order-averaged preference cancels it.
* **No rater data.** The 2AFC rater study was removed from the paper (no IRB
  determination, 2026-11-02). Its `human1*` protocols remain in the code for
  replay of old runs only; the default battery neither sends its stimuli nor
  joins its results.
* **Pre-registration.** `prereg.json` (predictions + the sha of the trial plan) is
  written into the run dir before any call is made; `analyze` scores each
  prediction mechanically against it.
* **Cost and replay.** Calls go through `blockgen.agentic.providers` (disk cache
  keyed on the request, so a re-analysis or a crashed batch re-run is free, and a
  spend gate: a cache miss needs `BLOCKGEN_ALLOW_SPEND=1`).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[3]

#: The leaderboard run of record (Table `tab:leaderboard`, `scripts/audit_claims.py`).
RUN_OF_RECORD = _REPO / "outputs" / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull"
#: The removed rater study whose stimuli the `human1*` protocols reuse verbatim.
HUMAN_RUN = _REPO / "outputs" / "run_20260923_human_v5"
CACHE_DIR = _REPO / "outputs" / "vlm_judge_cache"

#: Submissions as loaded by the run of record (`--arms ...` on its card).
SUBMISSION_NPZ = {
    "canon_oriented": "outputs/bench_arms/canon_oriented_32.npz",
    "canon_unoriented": "outputs/bench_arms/canon_unoriented_32.npz",
    "native_oriented": "outputs/bench_arms/native_oriented_32.npz",
    "llm_control": "outputs/bench_arms/llm_control_32.npz",
    "llm_bias": "outputs/bench_arms/llm_bias_32.npz",
}

#: Arms the system-level run ranks: every ranked row of the leaderboard plus the
#: four probes/controls with known damage. DQ rows (memorisation, mode collapse)
#: are left out; they are refused a rank by the suite and a judge cannot see
#: memorisation from one build anyway.
ALL_SYSTEM_ARMS = ("real_test", "native_oriented", "canon_oriented", "llm_control",
                   "llm_bias", "patchwork", "gabled_house", "shell_box", "uniform_random",
                   "real@solidify", "real_shuffled_materials", "real@monochrome",
                   "train_copy_noise")
#: Excluded by default (user decision 2026-11-02): the LLM-FT arms' samples are
#: truncated one-block-thick sticks, not representative builds. `--exclude-arms`
#: in scripts/run_vlm_judge.py restores them.
DEFAULT_EXCLUDE = ("llm_control", "llm_bias")
SYSTEM_ARMS = tuple(a for a in ALL_SYSTEM_ARMS if a not in DEFAULT_EXCLUDE)
#: The leaderboard ranking (anchor + submissions + baselines); filtered by presence.
RANKED_ARMS = ("real_test", "native_oriented", "canon_oriented", "llm_control",
               "llm_bias", "patchwork", "gabled_house", "shell_box", "uniform_random")
#: Procedural baselines: the "obvious cases" a correlation can be driven by.
PROCEDURAL_ARMS = ("uniform_random", "shell_box", "gabled_house", "patchwork")
#: Paper display names, for tables.
DISPLAY = {"real_test": "Real", "native_oriented": "Voxel-AR (contaminated)",
           "canon_oriented": "Voxel-AR (clean)", "llm_control": "LLM-FT",
           "llm_bias": "LLM-FT-Geo", "patchwork": "Patch-Stitch",
           "gabled_house": "Gabled-House", "shell_box": "Hollow-Box",
           "uniform_random": "Random-Voxels", "real@solidify": "Fill-Interior",
           "real@hollow": "Hollow-Shell", "real_shuffled_materials": "Material-Shuffle",
           "real@monochrome": "Monochrome", "train_copy_noise": "Train-Copy+Noise"}

# --- view protocols ----------------------------------------------------------
#: Same mapping as `scripts/ablate_view_protocol.py`: each camera's facing quadrant.
QUAD_FOR_AZIM = {45.0: "quad:hi:hi", 135.0: "quad:hi:lo",
                 225.0: "quad:lo:lo", 315.0: "quad:lo:hi"}


def view_config(protocol: str):
    """`features.ViewConfig` for a rendered protocol (the paper's cutaway recipes)."""
    from blockgen.eval.bench import features as ft
    if protocol == "ext4":            # the metric's protocol, unchanged
        return ft.ViewConfig()
    if protocol == "cut4_roof":       # paper fig. 9 "roof off (top half)"
        return ft.ViewConfig(views=tuple((a, 60.0) for a, _ in ft.DEFAULT_VIEWS),
                             cuts=("half:y:lo",) * 4)
    if protocol == "cut4_quad":       # paper fig. 9 "facing quadrant"
        return ft.ViewConfig(cuts=tuple(QUAD_FOR_AZIM[a] for a, _ in ft.DEFAULT_VIEWS))
    raise ValueError(f"not a rendered protocol: {protocol!r}")


#: What the judge is told about the picture. Describes the stimulus only; the
#: question is identical across protocols.
VIEW_NOTES = {
    "ext4": "Each image shows one build from four viewing angles (a 2x2 grid of "
            "exterior views).",
    "cut4_roof": "Each image shows one build from four viewing angles (a 2x2 grid) "
                 "in cutaway: the upper half of the build has been removed so you "
                 "can see inside.",
    "cut4_quad": "Each image shows one build from four viewing angles (a 2x2 grid) "
                 "in cutaway: in each view the corner facing the camera has been "
                 "removed so you can see inside.",
    "human1": "Each image shows one build from a single exterior viewing angle.",
    "human1_cut": "Each image shows one build from a single viewing angle in "
                  "cutaway: the corner nearest the camera has been removed so you "
                  "can see inside.",
}
HUMAN_PROTOCOLS = ("human1", "human1_cut")

SYSTEM_PROMPT = ("You are an experienced Minecraft player judging builds for a "
                 "research benchmark. You always answer with a single JSON object.")
#: The human study's question ("Which looks more like a real Minecraft house?"),
#: forced choice, with a one-line reason for audit.
QUESTION = ("Two Minecraft builds, A and B, are shown below. {note} Which looks more "
            "like a real Minecraft house, as built by a player? You must pick one; "
            "\"equal\" is not allowed. Reply with only a JSON object: "
            "{{\"choice\": \"A\" or \"B\", \"reason\": \"<at most 20 words>\"}}")


def grid_2x2(views: Sequence[np.ndarray], gutter: int = 4) -> np.ndarray:
    """Four HxWx3 views -> one image, row-major (az 45, 135 / 225, 315)."""
    if len(views) != 4:
        raise ValueError(f"need 4 views, got {len(views)}")
    h, w = views[0].shape[:2]
    out = np.full((2 * h + gutter, 2 * w + gutter, 3), 200, np.uint8)  # grey gutter
    for k, v in enumerate(views):
        r, c = divmod(k, 2)
        out[r * (h + gutter): r * (h + gutter) + h, c * (w + gutter): c * (w + gutter) + w] = v[..., :3]
    return out


def png_bytes(img: np.ndarray) -> bytes:
    from blockgen.agentic.providers import encode_png
    return encode_png(img)


# --- builds ------------------------------------------------------------------
def leaderboard_arms(card_dir: Path = RUN_OF_RECORD, verify: bool = True,
                     with_hollow: bool = True) -> Dict[str, list]:
    """Rebuild every arm of the run of record, exactly, and check it by sha.

    Mirrors `runner.main`: controls from `fast.control_arms(test, train, n,
    rng=ctx.rng(1))`, baselines from `baselines.arms(train, n, seed)`, submissions
    from their npz, everything cropped (the scorer crops on load).
    `real@hollow` (Hollow-Shell) is not a leaderboard row, so it is derived from
    `real_test` here, paired index-for-index like `real@solidify`.
    """
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.eval.bench import baselines as bl
    from blockgen.eval.bench import fast, probes, splits
    from blockgen.eval.bench import features as ft

    card = json.loads((Path(card_dir) / "scorecard.json").read_text())
    ctx = card["context"]
    corpus, seed = ctx.get("corpus", "houses_32"), int(ctx.get("seed", 0))
    n = int(card["arms"]["real_test"]["meta"]["n"])
    split = splits.load_split(corpus, seed)
    train = splits.split_structures(split, "train")
    test = splits.split_structures(split, "test")
    arms: Dict[str, list] = {}
    for a in fast.control_arms(test, train=train, n=n, rng=np.random.default_rng(seed + 1),
                               corpus=corpus, seed=seed):
        arms[a.name] = [s.crop_to_non_air() for s in a.structures]
    for name, structs in bl.arms(train, n, seed=seed).items():
        arms[name] = [s.crop_to_non_air() for s in structs]
    for name, rel in SUBMISSION_NPZ.items():
        arms[name] = [s.crop_to_non_air() for s in load_structures_from_cache(str(_REPO / rel))[0]]
    if verify:
        bad = []
        for name, structs in arms.items():
            want = (card["arms"].get(name, {}).get("meta") or {}).get("structures_sha")
            if want and ft.structures_sha(structs) != want:
                bad.append(name)
        if bad:
            raise RuntimeError(f"rebuilt arms differ from {card_dir.name}: {bad}; the "
                               "VLM ranking would not be about the scored builds")
    if with_hollow:
        arms["real@hollow"] = [probes.hollow(s) for s in arms["real_test"]]
    return arms


def leaderboard_metrics(card_dir: Path = RUN_OF_RECORD) -> Dict[str, Dict[str, float]]:
    """Per arm: BlockScore (scorecard.md leaderboard block), MV-KID and V-KID.

    All three are lower-better. BlockScore is not stored in scorecard.json (the
    scorer derives it), so it is parsed from the recorded markdown, as
    `scripts/audit_claims.Leaderboard` does for the paper table.
    """
    card = json.loads((Path(card_dir) / "scorecard.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for name, arm in card["arms"].items():
        r = arm.get("realism") or {}
        out[name] = {"mv_kid": (r.get("mv_dino_kid") or {}).get("value"),
                     "v_kid": (r.get("geom_kid") or {}).get("value")}
    in_block = False
    for line in (Path(card_dir) / "scorecard.md").read_text().splitlines():
        if line.startswith("## leaderboard"):
            in_block = True
            continue
        if in_block and line.startswith("## "):
            break
        if in_block and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) > 2 and cells[1] in out:
                t = cells[2].replace("*", "").strip()
                out[cells[1]]["blockscore"] = None if t == "DQ" else float(t)
    return out


# --- stimuli -----------------------------------------------------------------
@dataclass(frozen=True)
class Build:
    """One stimulus image: which build, shown how."""
    arm: str
    idx: int
    protocol: str

    @property
    def key(self) -> str:
        return f"{self.arm}#{self.idx}@{self.protocol}"


class StimulusStore:
    """Renders (or, for `human1*`, decodes) each Build once, as PNG on disk.

    Rendering happens in the calling thread only (`prepare`), never in the API
    worker threads: pyrender's EGL context is not thread-safe.
    """

    def __init__(self, root: Path, arms: Optional[Dict[str, list]] = None,
                 human_run: Path = HUMAN_RUN):
        self.root = Path(root)
        self.arms = arms or {}
        self.human_run = Path(human_run)
        self._human: Optional[dict] = None
        self._lock = threading.Lock()

    def path(self, b: Build) -> Path:
        safe = b.arm.replace("@", "+").replace("#", "_")
        return self.root / b.protocol / f"{safe}_{b.idx:03d}.png"

    def human_stimuli(self) -> dict:
        if self._human is None:
            self._human = json.loads((self.human_run / "stimuli.json").read_text())
        return self._human

    def _make(self, b: Build) -> bytes:
        if b.protocol in HUMAN_PROTOCOLS:
            import base64
            from PIL import Image
            stim = self.human_stimuli()
            arm = HUMAN_ARM.get(b.arm, b.arm) + ("#cut" if b.protocol == "human1_cut" else "")
            jpeg = base64.b64decode(stim["images"][stim["index"][arm][b.idx]])
            return png_bytes(np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")))
        from blockgen.eval.bench import features as ft
        views = ft.render_views([self.arms[b.arm][b.idx]], view_config(b.protocol),
                                verbose=False)
        return png_bytes(grid_2x2(views))

    def get(self, b: Build) -> bytes:
        p = self.path(b)
        if not p.is_file():
            data = self._make(b)
            p.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: two worker threads may want the same build (both orders).
            tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
            return data
        return p.read_bytes()

    def prepare(self, builds: Iterable[Build], log: Callable[[str], None] = print) -> int:
        todo = sorted({b for b in builds if not self.path(b).is_file()}, key=lambda b: b.key)
        t0 = time.time()
        for k, b in enumerate(todo):
            self.get(b)
            if (k + 1) % 100 == 0:
                log(f"  stimuli {k + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
        return len(todo)


#: Benchmark arm name -> human-study arm name.
HUMAN_ARM = {"real_test": "real", "real3dc": "real"}


# --- conditions and trials ---------------------------------------------------
@dataclass
class Condition:
    """A pairwise comparison: arm_a vs arm_b, under one view protocol.

    `rate` in the analysis is P(judge prefers arm_a). `paired` compares
    arm_a[i] with arm_b[i] (a probe and its source house); otherwise build
    indices are drawn independently.
    """
    name: str
    arm_a: str
    arm_b: str
    protocol: str
    n_pairs: int
    paired: bool = False
    prediction: str = ""          # "chance" | "above" | "near_one"
    why: str = ""
    human: Optional[str] = None   # the human_v5 pair key to read beside it


def default_conditions(scale: float = 1.0) -> List[Condition]:
    """The pre-registered battery. n is in build pairs; each is judged twice."""
    def n(k: int) -> int:
        return max(1, int(round(k * scale)))
    return [
        Condition("interior_ext", "real3dc", "real3dc@solidify", "ext4", n(128), True,
                  "chance", "Fill-Interior on the metric's exterior views: no "
                  "information, so a judge should be at chance", "real|real@solidify"),
        Condition("interior_cut", "real3dc", "real3dc@solidify", "cut4_roof", n(128), True,
                  "above", "positive control: roof off, interiors visible"),
        Condition("interior_cutq", "real3dc", "real3dc@solidify", "cut4_quad", n(64), True,
                  "above", "the paper's facing-quadrant cutaway (weakest cut)"),
        Condition("hollow_ext", "real3dc", "real3dc@hollow", "ext4", n(128), True,
                  "chance", "converse probe, exterior views: Hollow-Shell keeps every "
                  "visible block", "real|real@hollow"),
        Condition("hollow_cut", "real3dc", "real3dc@hollow", "cut4_roof", n(64), True,
                  "above", "converse probe, roof off: furniture and floors removed"),
        Condition("catch", "real3dc", "uniform_random", "ext4", n(64), False,
                  "near_one", "judge sanity", "real|uniform_random"),
        Condition("gabled_patchwork", "gabled_house", "patchwork", "ext4", n(128), False,
                  "above", "both metrics rank Patch-Stitch above Gabled-House",
                  "gabled_house|patchwork"),
    ]


@dataclass
class Trial:
    id: str
    condition: str
    pair: str          # build-pair id; both orders share it (the bootstrap cluster)
    order: int         # 0: arm_a shown as A; 1: arm_a shown as B
    a: Build           # shown first, labelled "A"
    b: Build
    arm_a: str         # condition arm_a / arm_b (not slots)
    arm_b: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["a"], d["b"] = self.a.key, self.b.key
        return d


def _pool(arms_len: Dict[str, int], arm: str, protocol: str, cap: Optional[int]) -> int:
    if protocol in HUMAN_PROTOCOLS:
        return 24
    n = arms_len[arm]
    return min(n, cap) if cap else n


def make_trials(conditions: Sequence[Condition], arms_len: Dict[str, int],
                seed: int = 0, system_arms: Sequence[str] = (),
                system_k: int = 0, system_protocol: str = "ext4",
                system_pool: Optional[int] = 64) -> List[Trial]:
    """Both orders of every build pair, for every condition plus the system run."""
    rng = np.random.default_rng(seed)
    trials: List[Trial] = []

    def add(cond: str, arm_a: str, arm_b: str, ia: int, ib: int, proto: str, k: int):
        pa, pb = Build(arm_a, ia, proto), Build(arm_b, ib, proto)
        pair = f"{cond}/{k}"
        trials.append(Trial(f"{pair}/0", cond, pair, 0, pa, pb, arm_a, arm_b))
        trials.append(Trial(f"{pair}/1", cond, pair, 1, pb, pa, arm_a, arm_b))

    for c in conditions:
        na = _pool(arms_len, c.arm_a, c.protocol, None)
        nb = _pool(arms_len, c.arm_b, c.protocol, None)
        if c.paired:
            idx = rng.permutation(min(na, nb))[: c.n_pairs]
            pairs = [(int(i), int(i)) for i in idx]
        else:
            pairs = [(int(rng.integers(na)), int(rng.integers(nb))) for _ in range(c.n_pairs)]
        for k, (ia, ib) in enumerate(pairs):
            add(c.name, c.arm_a, c.arm_b, ia, ib, c.protocol, k)
    if system_k:
        for i, x in enumerate(system_arms):
            for y in system_arms[i + 1:]:
                nx = _pool(arms_len, x, system_protocol, system_pool)
                ny = _pool(arms_len, y, system_protocol, system_pool)
                for k in range(system_k):
                    add(f"system:{x}|{y}", x, y, int(rng.integers(nx)),
                        int(rng.integers(ny)), system_protocol, k)
    return trials


def plan_sha(trials: Sequence[Trial]) -> str:
    blob = json.dumps([t.to_dict() for t in trials], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --- the judge ---------------------------------------------------------------
#: USD per 1M tokens (input, output) for pinned snapshots that
#: `providers.PRICES` lists only under their alias. Unknown -> unpriced (None).
JUDGE_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-2024-11-20": (2.5, 10.0), "gpt-4o-2024-08-06": (2.5, 10.0),
    "gpt-4.1-2025-04-14": (2.0, 8.0), "gpt-4.1-mini-2025-04-14": (0.4, 1.6),
}


def price_of(model: str) -> Optional[Tuple[float, float]]:
    from blockgen.agentic.providers import PRICES
    return JUDGE_PRICES.get(model) or PRICES.get(model)


def judge_messages(t: Trial, store: StimulusStore):
    from blockgen.agentic.providers import Message, Part, text_message
    q = QUESTION.format(note=VIEW_NOTES[t.a.protocol])
    user = Message("user", [Part("text", text=q),
                            Part("text", text="Build A:"), Part("image", image_png=store.get(t.a)),
                            Part("text", text="Build B:"), Part("image", image_png=store.get(t.b))])
    return [text_message("system", SYSTEM_PROMPT), user]


_CHOICE_RE = re.compile(r'"choice"\s*:\s*"?\s*([AB])\b', re.I)


def parse_choice(text: str) -> Optional[str]:
    """'A' | 'B' | None. JSON first, then the choice field, then a bare letter."""
    s = (text or "").strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            v = str(json.loads(m.group(0)).get("choice", "")).strip().upper()
            if v in ("A", "B"):
                return v
        except (json.JSONDecodeError, AttributeError):
            pass
    m = _CHOICE_RE.search(s)
    if m:
        return m.group(1).upper()
    s2 = s.strip(" .\"'`*").upper()
    return s2 if s2 in ("A", "B") else None


def judge_one(provider, t: Trial, store: StimulusStore) -> Dict[str, Any]:
    t0 = time.time()
    rec = t.to_dict()
    try:
        resp = provider.complete(judge_messages(t, store))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        rec.update(error=f"{type(exc).__name__}: {exc}", choice=None, valid=False)
        return rec
    choice = parse_choice(resp.text)
    price = price_of(resp.model)
    cost = None if price is None else (resp.prompt_tokens * price[0]
                                       + resp.completion_tokens * price[1]) / 1e6
    # Which ARM the judge picked, so order 0 and order 1 are directly comparable.
    picked = None if choice is None else (t.a.arm if choice == "A" else t.b.arm)
    prefers_a = None if choice is None else ((choice == "A") == (t.order == 0))
    rec.update(raw=resp.text, choice=choice, valid=choice is not None,
               prefers_arm_a=prefers_a, picked_arm=picked, model=resp.model,
               provider=provider.describe(), prompt_tokens=resp.prompt_tokens,
               completion_tokens=resp.completion_tokens, cost_usd=cost,
               cached=resp.cached, finish_reason=resp.finish_reason,
               latency_s=round(time.time() - t0, 2), meta=resp.meta or None)
    return rec


def run_trials(provider, trials: Sequence[Trial], store: StimulusStore,
               out_jsonl: Path, workers: int = 8,
               log: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    """Judge every trial not already in `out_jsonl` (resumable), appending as it goes."""
    out_jsonl = Path(out_jsonl)
    done: Dict[str, Dict[str, Any]] = {}
    if out_jsonl.is_file():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("valid") is not None and not r.get("error"):
                    done[r["id"]] = r
    todo = [t for t in trials if t.id not in done]
    # Load the repo .env HERE, in one thread: `providers.load_env` sets its
    # loaded flag before it reads the file, so workers racing into it on first
    # use see "already loaded" and no key (observed in the first smoke run).
    from blockgen.agentic.providers import load_env
    load_env()
    log(f"[vlm] {len(done)} trials already judged, {len(todo)} to go "
        f"({provider.describe()}, {workers} workers)")
    lock = threading.Lock()
    spent = [0.0]
    t0 = time.time()
    with out_jsonl.open("a") as fh, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(judge_one, provider, t, store): t for t in todo}
        for k, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                done[rec["id"]] = rec
                spent[0] += rec.get("cost_usd") or 0.0
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                log(f"[vlm] {k + 1}/{len(todo)} judged, ${spent[0]:.2f} so far, "
                    f"{time.time() - t0:.0f}s")
    return [done[t.id] for t in trials if t.id in done]


# --- analysis ----------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """The rater study's Wilson interval (`scripts/human_study_analyze.wilson`)."""
    from scripts.human_study_analyze import wilson as _w
    lo, hi = _w(k, n, z)
    return float(lo), float(hi)


def bradley_terry(wins: np.ndarray, iters: int = 2000, prior: float = 0.5) -> np.ndarray:
    """Log-strengths; the rater study's estimator, vectorised for the bootstrap.

    Same MM fit and the same half-win smoothing on every observed pair as
    `scripts/human_study_analyze.bradley_terry` (a test pins the equality), with
    the geometric mean of strengths fixed at 1.
    """
    w = np.asarray(wins, float)
    seen = (w + w.T) > 0
    w = w + prior * seen
    total = w + w.T
    wi = w.sum(1)
    p = np.ones(len(w))
    for _ in range(iters):
        denom = (total / (p[:, None] + p[None, :])).sum(1)
        new = np.where(denom > 0, wi / np.maximum(denom, 1e-12), p)
        new = np.where(new > 0, new, 1e-12)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < 1e-10:
            p = new
            break
        p = new
    return np.log(p)


def _cond_stats(recs: List[Dict[str, Any]], rng: np.random.Generator,
                n_boot: int = 2000) -> Dict[str, Any]:
    n_err = sum(1 for r in recs if r.get("error"))
    recs = [r for r in recs if not r.get("error")]   # transport errors are re-run, not data
    valid = [r for r in recs if r.get("valid")]
    n = len(valid)
    k = sum(bool(r["prefers_arm_a"]) for r in valid)
    first = sum(r["choice"] == "A" for r in valid)
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in valid:
        by_pair.setdefault(r["pair"], []).append(r)
    both = [v for v in by_pair.values() if len(v) == 2]
    agree = sum(v[0]["prefers_arm_a"] == v[1]["prefers_arm_a"] for v in both)
    same_slot = sum(v[0]["choice"] == v[1]["choice"] for v in both)
    # Order-averaged score per build pair: 1, 0.5 (orders disagree) or 0.
    scores = np.array([np.mean([bool(r["prefers_arm_a"]) for r in v]) for v in by_pair.values()])
    boot = None
    if len(scores):
        idx = rng.integers(len(scores), size=(n_boot, len(scores)))
        means = scores[idx].mean(1)
        boot = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    from scipy.stats import binomtest
    return {
        "n_trials": len(recs), "n_valid": n, "n_invalid": len(recs) - n, "n_errors": n_err,
        "n_pairs": len(by_pair), "n_pairs_both_orders": len(both),
        "rate_a": k / n if n else None, "wilson": list(wilson(k, n)) if n else None,
        "cluster_boot_ci": boot,
        "order_averaged_rate_a": float(scores.mean()) if len(scores) else None,
        "p_binom_vs_half": float(binomtest(k, n, 0.5).pvalue) if n else None,
        "p_first_slot": first / n if n else None,
        "first_slot_wilson": list(wilson(first, n)) if n else None,
        "order_agreement": agree / len(both) if both else None,
        "position_driven_pairs": same_slot / len(both) if both else None,
        "cost_usd": sum(r.get("cost_usd") or 0.0 for r in recs),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in recs),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
    }


def _system_wins(recs: Sequence[Dict[str, Any]], arms: Sequence[str]) -> np.ndarray:
    ix = {a: i for i, a in enumerate(arms)}
    w = np.zeros((len(arms), len(arms)))
    for r in recs:
        if not r.get("valid") or r["arm_a"] not in ix or r["arm_b"] not in ix:
            continue
        a, b = ix[r["arm_a"]], ix[r["arm_b"]]
        if r["prefers_arm_a"]:
            w[a, b] += 1
        else:
            w[b, a] += 1
    return w


def system_analysis(recs: Sequence[Dict[str, Any]], metrics: Dict[str, Dict[str, float]],
                    arms: Sequence[str] = SYSTEM_ARMS, n_boot: int = 1000,
                    seed: int = 0) -> Dict[str, Any]:
    """Bradley-Terry over the arms, then Kendall/Spearman against each metric.

    The bootstrap resamples build pairs (both orders together) within each arm
    pair, refits BT, and recomputes every correlation, so the CI is the judge's
    sampling uncertainty at fixed metric values. Metrics are lower-better, so
    the correlation is taken against their negation: positive = agreement.
    """
    from scipy.stats import kendalltau, spearmanr
    present = [a for a in arms if any(r["arm_a"] == a or r["arm_b"] == a for r in recs)]
    if len(present) < 3:
        return {"error": f"only {len(present)} arms judged"}
    bt = bradley_terry(_system_wins(recs, present))
    # With and without the procedural baselines: with only two or three model
    # arms, a correlation over the full set is mostly carried by the obvious
    # cases (Random-Voxels, Hollow-Box), and the reader should see that.
    subsets = {"all": present, "ranked": [a for a in present if a in RANKED_ARMS]}
    subsets["all_no_procedural"] = [a for a in subsets["all"] if a not in PROCEDURAL_ARMS]
    subsets["ranked_no_procedural"] = [a for a in subsets["ranked"]
                                       if a not in PROCEDURAL_ARMS]

    def corr(strength: Dict[str, float]) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for sname, sub in subsets.items():
            for m in ("blockscore", "mv_kid", "v_kid"):
                xs = [(strength[a], -metrics[a][m]) for a in sub
                      if metrics.get(a, {}).get(m) is not None]
                if len(xs) < 3:
                    continue
                x, y = zip(*xs)
                out[f"{sname}/{m}"] = {"kendall": float(kendalltau(x, y)[0]),
                                       "spearman": float(spearmanr(x, y)[0]), "n_arms": len(xs)}
        return out

    point = corr(dict(zip(present, bt)))
    clusters: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in recs:
        clusters.setdefault(r["condition"], {}).setdefault(r["pair"], []).append(r)
    rng = np.random.default_rng(seed)
    boots: Dict[str, Dict[str, List[float]]] = {}
    bt_boot = []
    groups = [list(v.values()) for v in clusters.values()]
    for _ in range(n_boot):
        sample: List[Dict[str, Any]] = []
        for g in groups:
            for j in rng.integers(len(g), size=len(g)):
                sample.extend(g[j])
        b = bradley_terry(_system_wins(sample, present))
        bt_boot.append(b)
        for key, v in corr(dict(zip(present, b))).items():
            for stat in ("kendall", "spearman"):
                boots.setdefault(key, {}).setdefault(stat, []).append(v[stat])
    for key, v in point.items():
        for stat in ("kendall", "spearman"):
            arr = np.array(boots.get(key, {}).get(stat, []), float)
            arr = arr[np.isfinite(arr)]
            v[f"{stat}_ci"] = ([float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
                               if len(arr) else None)
    bt_boot = np.array(bt_boot)
    order = sorted(range(len(present)), key=lambda i: -bt[i])
    return {
        "arms": present,
        "bradley_terry": {present[i]: {"strength": float(bt[i]),
                                       "ci": [float(np.percentile(bt_boot[:, i], 2.5)),
                                              float(np.percentile(bt_boot[:, i], 97.5))],
                                       **{m: metrics.get(present[i], {}).get(m)
                                          for m in ("blockscore", "mv_kid", "v_kid")}}
                          for i in order},
        "order": [present[i] for i in order],
        "correlations": point, "n_boot": n_boot,
        "n_valid": sum(1 for r in recs if r.get("valid")),
    }


PREDICTION_TEXT = {
    "chance": "rate_a CI contains 0.5 and the order-averaged rate is within 0.5 +/- 0.1",
    "above": "rate_a Wilson lower bound > 0.5",
    "near_one": "rate_a >= 0.95",
}


def check_prediction(kind: str, s: Dict[str, Any]) -> Optional[bool]:
    if not s.get("n_valid"):
        return None
    lo, hi = s["wilson"]
    if kind == "chance":
        return bool(lo <= 0.5 <= hi and abs(s["order_averaged_rate_a"] - 0.5) <= 0.1)
    if kind == "above":
        return bool(lo > 0.5)
    if kind == "near_one":
        return bool(s["rate_a"] >= 0.95)
    return None


def prereg(conditions: Sequence[Condition], trials: Sequence[Trial], judge: str,
           extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The pre-registration record, written before any call."""
    return {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": judge, "plan_sha": plan_sha(trials), "n_trials": len(trials),
        "question": QUESTION, "system_prompt": SYSTEM_PROMPT, "view_notes": VIEW_NOTES,
        "predictions": [{"condition": c.name, "arm_a": c.arm_a, "arm_b": c.arm_b,
                         "protocol": c.protocol, "n_pairs": c.n_pairs, "paired": c.paired,
                         "prediction": c.prediction, "rule": PREDICTION_TEXT.get(c.prediction),
                         "why": c.why} for c in conditions],
        "system_predictions": {
            "catch_bt": "real_test above uniform_random with non-overlapping BT CIs",
            "fill_interior_bt": "Fill-Interior's BT CI overlaps real_test's (exterior views)",
            "kendall_blockscore_ranked": "> 0 (judge and BlockScore agree on the ranked arms)",
        },
        "analysis_rules": {
            "unit": "judgment; Wilson CI over judgments, plus a cluster bootstrap over "
                    "build pairs (both orders together)",
            "invalid": "unparseable replies are excluded and counted, never imputed",
            "position_bias": "P(first slot) with Wilson CI; order agreement = share of "
                             "build pairs whose two orders pick the same build",
        },
        **(extra or {}),
    }


def analyze(records: Sequence[Dict[str, Any]], conditions: Sequence[Condition],
            metrics: Optional[Dict[str, Dict[str, float]]] = None,
            human_analysis: Optional[Dict[str, Any]] = None,
            n_boot: int = 1000, seed: int = 0,
            system_arms: Sequence[str] = SYSTEM_ARMS) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_cond: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    out: Dict[str, Any] = {"conditions": {}}
    hp = (human_analysis or {}).get("pairs", {})
    for c in conditions:
        recs = by_cond.get(c.name, [])
        if not recs:
            continue
        s = _cond_stats(recs, rng)
        s.update(arm_a=c.arm_a, arm_b=c.arm_b, protocol=c.protocol,
                 prediction=c.prediction, prediction_met=check_prediction(c.prediction, s))
        if c.human and c.human in hp:
            h = hp[c.human]
            s["human_v5"] = {"pair": c.human, "rate_a": h["p"], "n": h["n"], "ci": h["ci"]}
        out["conditions"][c.name] = s
    sys_recs = [r for r in records if r["condition"].startswith("system:")]
    if sys_recs:
        sys_stats = _cond_stats(sys_recs, rng)
        out["system_position"] = {k: sys_stats[k] for k in
                                  ("n_valid", "n_invalid", "p_first_slot", "first_slot_wilson",
                                   "order_agreement", "position_driven_pairs", "cost_usd")}
        if metrics:
            out["system"] = system_analysis(sys_recs, metrics, arms=system_arms,
                                            n_boot=n_boot, seed=seed)
    allv = [r for r in records if r.get("valid")]
    out["overall"] = {
        "n_trials": len(records), "n_valid": len(allv),
        "p_first_slot": (sum(r["choice"] == "A" for r in allv) / len(allv)) if allv else None,
        "cost_usd": sum(r.get("cost_usd") or 0.0 for r in records),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
        "models": sorted({r.get("model") for r in records if r.get("model")}),
    }
    return out


def report_md(an: Dict[str, Any]) -> str:
    lines = ["# VLM judge", "", f"models: {an['overall']['models']}  ",
             f"valid {an['overall']['n_valid']}/{an['overall']['n_trials']}, "
             f"P(first slot) {_f(an['overall']['p_first_slot'])}, "
             f"cost ${an['overall']['cost_usd']:.2f}", "",
             "| condition | A vs B | view | n | P(A) | Wilson | pair-boot | order-avg | P(1st) | agree | pred | met | human |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in an["conditions"].items():
        h = s.get("human_v5")
        lines.append(
            f"| {name} | {s['arm_a']} vs {s['arm_b']} | {s['protocol']} | {s['n_valid']} | "
            f"{_f(s['rate_a'])} | {_ci(s['wilson'])} | {_ci(s['cluster_boot_ci'])} | "
            f"{_f(s['order_averaged_rate_a'])} | {_f(s['p_first_slot'])} | "
            f"{_f(s['order_agreement'])} | {s['prediction']} | {s['prediction_met']} | "
            f"{(_f(h['rate_a']) + ' (n=' + str(h['n']) + ')') if h else ''} |")
    for dname, dd in (an.get("door_diagnostic") or {}).items():
        lines += ["", f"## door diagnostic, flag = {dname} (Fill-Interior cells visible "
                  "from outside; `reachable`: through air or see-through blocks; "
                  "`behind_window`: directly behind an exterior see-through block)", "",
                  f"houses with any exterior-visible filled cell: "
                  f"{_f(dd.get('frac_houses_changed'))} of {dd.get('n_houses')} "
                  f"(any fill at all: {_f(dd.get('frac_houses_any_fill'))})", "",
                  "| condition | pairs changed | P(real) changed | n | P(real) unchanged | n |",
                  "|---|---|---|---|---|---|"]
        for k, v in dd.items():
            if isinstance(v, dict) and "changed" in v:
                lines.append(f"| {k} | {_f(v['frac_pairs_changed'])} | "
                             f"{_f(v['changed']['rate_real'])} {_ci(v['changed']['wilson'])} | "
                             f"{v['changed']['n']} | {_f(v['unchanged']['rate_real'])} "
                             f"{_ci(v['unchanged']['wilson'])} | {v['unchanged']['n']} |")
    sysa = an.get("system")
    if sysa and "bradley_terry" in sysa:
        lines += ["", "## system-level Bradley-Terry", "",
                  "**Reference-set mismatch.** The judge saw only 3D-Craft-derived builds "
                  "(real_test and its probes: the 3D-Craft members of the scored 128; "
                  "train_copy_noise: 3D-Craft-sourced members; patchwork: rebuilt from "
                  "3D-Craft train builds; model arms: near-copies of non-3D-Craft builds "
                  "dropped; see `restriction`). BlockScore, MV-KID and V-KID are the "
                  "leaderboard run's numbers over the full scored sets against the full "
                  "reference. The correlation compares rankings over different subsets.", "",
                  "| arm | BT | CI | BlockScore | MV-KID | V-KID |", "|---|---|---|---|---|---|"]
        for a, v in sysa["bradley_terry"].items():
            lines.append(f"| {DISPLAY.get(a, a)} | {v['strength']:+.2f} | {_ci(v['ci'])} | "
                         f"{_f(v['blockscore'])} | {_f(v['mv_kid'])} | {_f(v['v_kid'])} |")
        lines += ["", "| set/metric | Kendall tau | CI | Spearman | CI | arms |",
                  "|---|---|---|---|---|---|"]
        for k, v in sysa["correlations"].items():
            lines.append(f"| {k} | {_f(v['kendall'])} | {_ci(v.get('kendall_ci'))} | "
                         f"{_f(v['spearman'])} | {_ci(v.get('spearman_ci'))} | {v['n_arms']} |")
    return "\n".join(lines) + "\n"


def _f(x: Any) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _ci(ci: Any) -> str:
    return "" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def samples_png(store: StimulusStore, trials: Sequence[Trial], path: Path,
                per_condition: int = 1) -> None:
    """One example pair (order 0) per condition, labelled; into the run folder."""
    from PIL import Image, ImageDraw
    seen: Dict[str, int] = {}
    rows = []
    for t in trials:
        key = t.condition.split(":")[0] if t.condition.startswith("system") else t.condition
        if t.order != 0 or seen.get(key, 0) >= per_condition:
            continue
        seen[key] = seen.get(key, 0) + 1
        rows.append((f"{t.condition}  [{t.a.protocol}]  A={t.a.key}  B={t.b.key}",
                     store.get(t.a), store.get(t.b)))
    if not rows:
        return
    tile = 300
    W, H = 2 * tile + 30, len(rows) * (tile + 26) + 10
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for r, (label, pa, pb) in enumerate(rows):
        y = 10 + r * (tile + 26)
        d.text((10, y), label[:110], fill=(0, 0, 0))
        for c, png in enumerate((pa, pb)):
            im = Image.open(io.BytesIO(png)).convert("RGB")
            im.thumbnail((tile, tile))
            canvas.paste(im, (10 + c * (tile + 10), y + 16))
    canvas.save(path)


# --- licence restriction: only 3D-Craft content goes to the API ---------------
# The human study showed raters only 3D-Craft builds (`human_study_export.
# _publishable_rows`): 3D-Craft is an MIT-licensed research release, while
# GrabCraft's terms forbid distributing pictures of its blueprints and text2mc
# comes from PlanetMinecraft. A third-party API call is distribution enough to
# hold it to the same rule, so every stimulus that shows a real build, or a
# probe derived from one, is restricted to 3D-Craft here.
PUBLISHABLE = "3dcraft"
#: Arms whose i-th build is real_test[i], altered (control_arms builds them index-wise).
REAL_DERIVED = ("real_test", "real_shuffled_materials", "real@monochrome",
                "real@solidify", "real@hollow")
MODEL_ARMS = ("native_oriented", "canon_oriented", "llm_control", "llm_bias")
#: Condition arms: ALL 3D-Craft held-out test houses (more than the 79 of
#: real_test's 128 that are 3D-Craft), and their paired probes.
REAL3 = "real3dc"


def split_sources(corpus: str = "houses_32", seed: int = 0) -> Dict[str, List[str]]:
    """Upstream corpus ('3dcraft' | 'grabcraft' | 'text2mc') of every split row."""
    from blockgen.eval.bench import splits
    man = json.loads((_REPO / "data" / "minecraft" / "cache"
                      / f"{corpus}_manifest.json").read_text())["items"]
    split = splits.load_split(corpus, seed)
    return {w: [man[r].get("corpus") for r in split.indices(w)]
            for w in ("train", "val", "test")}


def _occ32(s, dim: int = 32) -> np.ndarray:
    occ = s.crop_to_non_air().occupied_mask[:dim, :dim, :dim]
    out = np.zeros((dim, dim, dim), np.float32)
    out[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    return out


def _d4(s) -> list:
    """The 8 y-axis rotations/mirrors, so a rotated copy still counts as a copy."""
    from blockgen.eval.bench import probes
    out = []
    for m in (s, probes.mirror_x(s)):
        for k in range(4):
            out.append(probes.rotate_y(m, k).crop_to_non_air())
    return out


def nn_iou(samples: Sequence, corpus: Sequence, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Per sample: max occupancy IoU over D4 x corpus, and the argmax corpus row."""
    B = np.stack([_occ32(c).ravel() for c in corpus])
    nb = B.sum(1)
    best = np.zeros(len(samples))
    arg = np.zeros(len(samples), int)
    for i0 in range(0, len(samples), chunk // 8):
        group = samples[i0:i0 + chunk // 8]
        A = np.stack([_occ32(v).ravel() for s in group for v in _d4(s)])
        inter = A @ B.T
        iou = inter / np.maximum(A.sum(1)[:, None] + nb[None, :] - inter, 1)
        iou = iou.reshape(len(group), 8, -1).max(1)
        best[i0:i0 + len(group)] = iou.max(1)
        arg[i0:i0 + len(group)] = iou.argmax(1)
    return best, arg


def restrict_to_3dcraft(arms: Dict[str, list], corpus: str = "houses_32", seed: int = 0,
                        log: Callable[[str], None] = print,
                        copy_iou: Optional[float] = None) -> Tuple[Dict[str, list], Dict[str, Any]]:
    """Arms with only 3D-Craft-derived content, and a provenance report.

    * real_test and its index-wise probes: the members whose source is 3D-Craft.
    * `real3dc` (+ `@solidify`, `@hollow`): every 3D-Craft test house, for the
      paired conditions, which need more than real_test's 79.
    * train_copy_noise: replayed to recover each build's training source; only
      3D-Craft sources kept (the same builds the leaderboard scored).
    * patchwork: 8^3 patches lifted verbatim from many train builds, so almost
      no leaderboard build is 3D-Craft-only; REBUILT with the same recipe and
      seed from the 3D-Craft train builds. Not the scored set (sha recorded).
    * model arms: a sample whose nearest corpus build (occupancy IoU, any D4
      pose) is non-3D-Craft at IoU >= `copy_iou` is a near-copy and dropped.
      `copy_iou` defaults to max(0.8, 99th pct of held-out-real 3D-Craft NN IoU
      to train), i.e. closer to the corpus than real unseen builds get.
    * uniform_random / shell_box / gabled_house: palette statistics only, kept.
    """
    from blockgen.eval.bench import baselines as bl
    from blockgen.eval.bench import fast, probes, splits
    from blockgen.eval.bench import features as ft

    src = split_sources(corpus, seed)
    split = splits.load_split(corpus, seed)
    train = splits.split_structures(split, "train")
    test = splits.split_structures(split, "test")
    n = len(arms["real_test"])
    perm = np.random.default_rng(seed + 1).permutation(len(test))[:n]  # control_arms' draw
    real_src = [src["test"][i] for i in perm]
    keep_real = [i for i, c in enumerate(real_src) if c == PUBLISHABLE]
    out: Dict[str, list] = {}
    rep: Dict[str, Any] = {"rule": f"only {PUBLISHABLE} content is sent to the API"}
    for a in REAL_DERIVED:
        if a in arms:
            out[a] = [arms[a][i] for i in keep_real]
            rep[a] = {"kept": len(keep_real), "of": len(arms[a]),
                      "leaderboard_rows": keep_real, "same_builds_as_scored": True}
    t3 = [s.crop_to_non_air() for s, c in zip(test, src["test"]) if c == PUBLISHABLE]
    out[REAL3] = t3
    out[REAL3 + "@solidify"] = [probes.solidify(s) for s in t3]
    out[REAL3 + "@hollow"] = [probes.hollow(s) for s in t3]
    rep[REAL3] = {"kept": len(t3), "of": len(test),
                  "note": "all 3D-Craft held-out test houses (cropped, split order)"}

    # train_copy_noise: replay `baselines.arms` with the source recorded.
    rec: List[int] = []
    orig = bl.train_copy_noise

    def spy(tr, rng, p=0.10):
        i = int(rng.integers(len(tr)))
        rec.append(i)
        return probes.occupancy_noise(tr[i], p, rng)
    bl.train_copy_noise = spy
    try:
        replay = bl.build("train_copy_noise", n, train,
                          seed=seed + bl.ORDER.index("train_copy_noise"))
    finally:
        bl.train_copy_noise = orig
    if ft.structures_sha([s.crop_to_non_air() for s in replay]) != ft.structures_sha(arms["train_copy_noise"]):
        raise RuntimeError("train_copy_noise replay does not reproduce the scored arm")
    keep = [i for i, r in enumerate(rec) if src["train"][r] == PUBLISHABLE]
    out["train_copy_noise"] = [arms["train_copy_noise"][i] for i in keep]
    rep["train_copy_noise"] = {"kept": len(keep), "of": n, "leaderboard_rows": keep,
                               "same_builds_as_scored": True}

    train3 = [s for s, c in zip(train, src["train"]) if c == PUBLISHABLE]
    pw = [s.crop_to_non_air() for s in
          bl.build("patchwork", n, train3, seed=seed + bl.ORDER.index("patchwork"))]
    out["patchwork"] = pw
    rep["patchwork"] = {"kept": len(pw), "of": n, "same_builds_as_scored": False,
                        "note": "rebuilt from the 3D-Craft train builds only (recipe "
                                "lifts real 8^3 patches verbatim)",
                        "structures_sha": ft.structures_sha(pw)}
    for a in ("uniform_random", "shell_box", "gabled_house"):
        out[a] = arms[a]
        rep[a] = {"kept": len(arms[a]), "of": len(arms[a]), "same_builds_as_scored": True,
                  "note": "procedural; no upstream build content"}

    # Model arms: near-copy screen against the whole corpus.
    corpus_structs = list(train) + splits.split_structures(split, "val") + list(test)
    corpus_src = src["train"] + src["val"] + src["test"]
    if copy_iou is None:
        ref_iou, _ = nn_iou(t3, train)
        copy_iou = max(0.8, float(np.percentile(ref_iou, 99)))
        rep["near_copy_reference"] = {
            "what": "NN occupancy IoU (D4) of held-out 3D-Craft test houses to train",
            "median": float(np.median(ref_iou)), "p99": float(np.percentile(ref_iou, 99))}
    rep["near_copy_iou"] = copy_iou
    for a in MODEL_ARMS:
        if a not in arms:
            continue
        best, arg = nn_iou(arms[a], corpus_structs)
        nn_src = [corpus_src[j] for j in arg]
        drop = [i for i in range(len(best)) if best[i] >= copy_iou and nn_src[i] != PUBLISHABLE]
        keep = [i for i in range(len(best)) if i not in set(drop)]
        out[a] = [arms[a][i] for i in keep]
        rep[a] = {"kept": len(keep), "of": len(best), "dropped_near_copies": drop,
                  "dropped_nn_iou": [round(float(best[i]), 3) for i in drop],
                  "median_nn_iou": float(np.median(best)),
                  "n_nn_iou_ge_threshold_any_source": int((best >= copy_iou).sum()),
                  "same_builds_as_scored": True, "leaderboard_rows": keep}
        log(f"[3dc] {a}: {len(drop)} near-copies of non-3D-Craft builds dropped "
            f"(IoU >= {copy_iou:.2f}); median NN IoU {np.median(best):.2f}")
    for a, v in rep.items():
        if isinstance(v, dict) and "kept" in v:
            log(f"[3dc] {a}: {v['kept']}/{v['of']} kept")
    return out, rep


# --- diagnostic: is Fill-Interior visible from outside? -----------------------
def visible_fill(s) -> Dict[str, int]:
    """Cells `solidify` fills that a camera outside can see.

    `solidify` fills only air unreachable from outside through AIR, so it never
    closes an open doorway. It can still change the exterior render through a
    transparent block (glass, panes, doors, fences, leaves): air behind a window
    is sealed by occupancy, yet visible. A filled cell counts as visible if it
    is reachable from outside by a 6-connected path through air or transparent
    blocks (the renderer's own transparency classes, `textured._cull_classes`).
    """
    from scipy import ndimage
    from blockgen.eval.bench import probes
    from blockgen.renderer.textured import _CULL_FIRST, _cull_classes
    from blockgen.renderer.textures import load_face_textures
    global _TEX
    try:
        tex = _TEX
    except NameError:
        tex = _TEX = load_face_textures()
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    filled = probes.solidify(c).occupied_mask
    add = filled & ~occ
    see = (~occ) | (_cull_classes(c, tex) >= _CULL_FIRST)
    padded = np.pad(see, 1, constant_values=True)
    lab, _ = ndimage.label(padded, structure=ndimage.generate_binary_structure(3, 1))
    reach = (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]
    # Stricter: filled cells directly behind a see-through block that itself
    # touches outside air (the first cell seen through a window or glazed door).
    st = ndimage.generate_binary_structure(3, 1)
    window = (see & occ) & ndimage.binary_dilation(probes.exterior_air(occ), st)
    behind = add & ndimage.binary_dilation(window, st)
    return {"added": int(add.sum()), "visible_added": int((add & reach).sum()),
            "behind_window": int(behind.sum()),
            "transparent_blocks": int((see & occ).sum())}


def door_diagnostic(records: Sequence[Dict[str, Any]], vis: Sequence[Dict[str, int]],
                    conditions: Sequence[str] = ("interior_ext", "interior_cut",
                                                 "interior_cutq")) -> Dict[str, Any]:
    """Judge preference for real over Fill-Interior, split by exterior-visible change.

    `vis[i]` is `visible_fill(real3dc[i])`. If the exterior-view preference
    lives entirely in the 'changed' pairs, the judge is reading a window, not
    the interior.
    """
    return {"reachable": _door_split(records, vis, "visible_added", conditions),
            "behind_window": _door_split(records, vis, "behind_window", conditions)}


def _door_split(records, vis, field_name, conditions):
    changed = [v.get(field_name, 0) > 0 for v in vis]
    out: Dict[str, Any] = {"flag": field_name,
        "n_houses": len(vis),
        "frac_houses_changed": float(np.mean(changed)) if vis else None,
        "frac_houses_any_fill": float(np.mean([v["added"] > 0 for v in vis])) if vis else None,
        "median_visible_added_when_changed": (float(np.median(
            [v[field_name] for v in vis if v[field_name] > 0]))
            if any(changed) else None)}
    for cond in conditions:
        recs = [r for r in records if r["condition"] == cond and r.get("valid")]
        if not recs:
            continue
        idx_of = lambda r: int(r["a"].split("#")[1].split("@")[0])  # paired: same idx
        pairs = {r["pair"] for r in recs}
        out_c: Dict[str, Any] = {"frac_pairs_changed": float(np.mean(
            [changed[idx_of(next(r for r in recs if r["pair"] == p))] for p in pairs]))}
        for flag, label in ((True, "changed"), (False, "unchanged")):
            sub = [r for r in recs if changed[idx_of(r)] == flag]
            k = sum(bool(r["prefers_arm_a"]) for r in sub)
            out_c[label] = {"n": len(sub), "rate_real": (k / len(sub)) if sub else None,
                            "wilson": list(wilson(k, len(sub))) if sub else None}
        out[cond] = out_c
    return out
