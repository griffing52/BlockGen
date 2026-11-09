"""Sampling, rendering and the end-of-run benchmark for the shared trainer."""
from __future__ import annotations

import glob
import inspect
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.utils.data import Structure
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN


_warned_uncached: set = set()


def supports_cache(model) -> bool:
    """Whether the model can be driven by ``forward(step, past=..., return_cache=True)``:
    its forward takes both keywords and ``supports_cache()``, when it has one, says yes
    (``pe=alibi`` needs the full bias and says no)."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    if not {"past", "return_cache"} <= set(params):
        return False
    supports = getattr(model, "supports_cache", None)
    return bool(supports()) if callable(supports) else True


class Decoder:
    """Next-token logits for a batch, with the KV cache when the model offers one.

    The path is decided up front by ``supports_cache``: the fast one is
    ``model.forward(step, past=past, return_cache=True)``; a model without it is
    re-run on the whole prefix every step. Slower, same tokens under the same seed,
    so any model that implements ``forward(input_ids, pad_mask=None)`` samples. The
    fallback is announced once per model class, and an error inside a real cached
    forward propagates rather than being mistaken for a missing cache.
    """

    def __init__(self, model):
        self.model = model
        self.past = None
        self.use_cache = supports_cache(model)
        self.fell_back = not self.use_cache
        if self.fell_back and type(model).__name__ not in _warned_uncached:
            _warned_uncached.add(type(model).__name__)
            print(f"[sample] {type(model).__name__} has no KV cache; sampling by full re-forward per step", flush=True)

    def next_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits for the token after ``tokens`` (``[B, L]``), shape ``[B, V]``."""
        if self.use_cache:
            step_in = tokens if self.past is None else tokens[:, -1:]
            logits, self.past = self.model.forward(step_in, past=self.past, return_cache=True)
            return logits[:, -1, :]
        return self.model(tokens)[:, -1, :]


def _pick(nl: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Sample one token per row from logits ``[B, V]``; returns ``[B, 1]``."""
    nl = nl.float() / max(temperature, 1e-5)
    if top_k and 0 < top_k < nl.shape[-1]:
        vals, idxs = torch.topk(nl, k=top_k, dim=-1)
        return idxs.gather(-1, torch.multinomial(torch.softmax(vals, dim=-1), 1))
    return torch.multinomial(torch.softmax(nl, dim=-1), 1)


def _cut_at_eos(tokens: torch.Tensor) -> List[np.ndarray]:
    out = []
    for row in tokens.cpu().numpy():
        eos = np.flatnonzero(row == EOS_TOKEN)
        out.append(row[: int(eos[0]) + 1].astype(np.uint16) if len(eos) else row.astype(np.uint16))
    return out


@torch.no_grad()
def generate(model, tokenizer: CorpusTokenizer, tags: Dict[str, str], n: int, max_blocks: int,
             temperature: float, top_k: int, seed: int = 0, batch: int = 64,
             prompts: Optional[List[List[int]]] = None, max_new: Optional[int] = None) -> List[np.ndarray]:
    """``n`` token sequences from the tag prefix, ``batch`` at a time with the KV cache.

    The model's own ``generate`` is one sequence per call; at 5k tokens a sample that is
    minutes per build, and a scorecard wants 256 of them. Rows that have emitted EOS keep
    being fed PAD until the batch finishes and are cut at their EOS afterwards.

    ``prompts`` replaces the shared tag prefix with one prompt per sequence (all of one
    length, so a batch is a rectangle); the facade sampler uses it to name the face and
    its size. ``max_new`` caps the tokens generated after the prompt. A model without a
    KV cache is driven by ``Decoder``'s full re-forward path.
    """
    from blockgen.utils.serialize import PAD_TOKEN
    device = next(model.parameters()).device
    if prompts is None:
        prompts = [tokenizer.prefix(tags)] * n
    assert len(prompts) == n and len({len(p) for p in prompts}) == 1, "prompts must be n of one length"
    torch.manual_seed(seed)
    model.eval()
    budget = min(max_new or (4 * max_blocks + 1), model.max_seq_len - len(prompts[0]) - 1)
    out: List[np.ndarray] = []
    for b0 in range(0, n, batch):
        B = min(batch, n - b0)
        tokens = torch.tensor(prompts[b0:b0 + B], dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        dec = Decoder(model)
        for _ in range(budget):
            nxt = _pick(dec.next_logits(tokens), temperature, top_k)
            nxt = torch.where(done[:, None], torch.full_like(nxt, PAD_TOKEN), nxt)
            tokens = torch.cat([tokens, nxt], dim=1)
            done |= nxt[:, 0] == EOS_TOKEN
            if bool(done.all()):
                break
        out.extend(_cut_at_eos(tokens))
    model.train()
    return out


def to_structures(seqs: List[np.ndarray], tokenizer: CorpusTokenizer, corpus: str = "generated") -> List[Structure]:
    """Token sequences to structures by ``tokenizer.decode``: blocks for voxels, each
    piece's pattern placed at its anchor for pieces (``cluster_tokens_to_structure``)."""
    structs = []
    for s in seqs:
        ids, data, tags = tokenizer.decode(s)
        structs.append(Structure(block_ids=ids.astype(np.int32), block_data=data.astype(np.int32),
                                 metadata={"corpus": corpus, "tags": tags}))
    return structs


@torch.no_grad()
def generate_faces(model, tokenizer: CorpusTokenizer, tags: Dict[str, str], plans: List[List[Tuple[str, int, int]]],
                   temperature: float, top_k: int, seed: int = 0, batch: int = 32) -> List[np.ndarray]:
    """One joint-layout sequence per plan, the faces generated in sight of each other.

    A plan is the faces of one build with their sizes, ``[(face, U, V), ...]``. The
    model writes the cells; every face header (tag, U, V) is forced, so the six
    faces of a build always describe one box, and a face the model ends early (an
    EOS or a face tag before its ``U*V`` cells are done) simply hands over to the
    next header. The last face is followed by a forced EOS.
    """
    from blockgen.utils.serialize import PAD_TOKEN
    device = next(model.parameters()).device
    vocab = tokenizer.vocab
    empty = tokenizer.tag_id("cell", "empty")
    torch.manual_seed(seed)
    model.eval()

    def header(face, U, V):
        return [tokenizer.tag_id("face", face), vocab.coord_id(U), vocab.coord_id(V)]

    out: List[np.ndarray] = []
    for b0 in range(0, len(plans), batch):
        chunk = plans[b0:b0 + batch]
        B = len(chunk)
        prompts = [tokenizer.prefix(tags) + header(*p[0]) for p in chunk]
        assert len({len(p) for p in prompts}) == 1
        tokens = torch.tensor(prompts, dtype=torch.long, device=device)
        k = [0] * B                                             # face index per row
        left = [p[0][1] * p[0][2] for p in chunk]               # cells still owed by the current face
        pending: List[List[int]] = [[] for _ in range(B)]
        done = [False] * B
        budget = min(max(sum(3 + 2 * U * V for _, U, V in p) for p in chunk) + 1,
                     model.max_seq_len - len(prompts[0]) - 1)
        dec = Decoder(model)
        for _ in range(budget):
            nxt = _pick(dec.next_logits(tokens), temperature, top_k)
            sampled = nxt[:, 0].tolist()
            forced = []
            for r in range(B):
                if done[r]:
                    forced.append(PAD_TOKEN); continue
                if pending[r]:
                    tok = pending[r].pop(0)
                else:
                    tok = sampled[r]
                    if tok == empty or vocab.is_block(tok):
                        left[r] -= 1
                    elif vocab.is_coord(tok):
                        pass
                    else:                                       # EOS, a face tag, anything else: face over
                        left[r] = 0
                        tok = None
                    if left[r] <= 0 and not pending[r]:
                        k[r] += 1
                        if k[r] < len(chunk[r]):
                            face, U, V = chunk[r][k[r]]
                            pending[r] = header(face, U, V); left[r] = U * V
                        else:
                            pending[r] = [EOS_TOKEN]
                    if tok is None:
                        tok = pending[r].pop(0)
                if tok == EOS_TOKEN:
                    done[r] = True
                forced.append(tok)
            nxt = torch.tensor(forced, dtype=torch.long, device=device)[:, None]
            tokens = torch.cat([tokens, nxt], dim=1)
            if all(done):
                break
        out.extend(_cut_at_eos(tokens))
    model.train()
    return out


def facade_builds(model, tokenizer: CorpusTokenizer, tags: Dict[str, str], n: int, dims: np.ndarray,
                  faces: List[str], temperature: float, top_k: int, seed: int = 0, corpus: str = "generated",
                  batch: int = 64) -> Tuple[List[Structure], List[Dict[str, "fc.Facade"]]]:
    """``n`` builds from a facade model: every face of a build is sampled with the same
    bounding box (a real one, drawn from ``dims``), then the faces are lifted into it.

    One build is ``len(faces)`` sequences; the prompts carry the face and its ``(U, V)``,
    so the model never has to agree with itself on size, only on content. Returns the
    lifted builds and the faces themselves (for the facade sheet).
    """
    from blockgen.corpus import facade as fc
    rng = np.random.default_rng(seed)
    boxes = dims[rng.integers(len(dims), size=n)]

    def size(face, X, Y, Z):
        axis = face[1]
        return (int(X), int(Y)) if axis == "z" else (int(Z), int(Y)) if axis == "x" else (int(X), int(Z))

    per_build: List[Dict[str, "fc.Facade"]] = [{} for _ in range(n)]
    if tokenizer.layout == "joint":
        plans = [[(face, *size(face, X, Y, Z)) for face in faces] for X, Y, Z in boxes]
        seqs = generate_faces(model, tokenizer, tags, plans, temperature, top_k, seed, batch=min(batch, 32))
        for i, seq in enumerate(seqs):
            for f in tokenizer.decode_faces(seq)[0]:
                per_build[i].setdefault(f.face, f)
    elif tokenizer.layout == "pooled4":
        from .sample_constrained import generate_faces_constrained
        plans, shapes, who = [], [], []
        for i, (X, Y, Z) in enumerate(boxes):
            for face in faces:
                U, V = size(face, X, Y, Z)
                plans.append([(face, -(-U // 4), -(-V // 4)), (face, U, V)])     # the level f=4, then the face
                shapes.append((int(X), int(Y), int(Z)))
                who.append((i, face))
        seqs, _ = generate_faces_constrained(model, tokenizer, tags, plans, shapes, temperature, top_k, seed,
                                             batch=batch, constrain=False)
        for (i, face), seq in zip(who, seqs):
            f, _ = tokenizer.decode_face(seq)
            if f is not None:
                per_build[i][face] = f
    else:
        prompts, who = [], []
        for i, (X, Y, Z) in enumerate(boxes):
            for face in faces:
                prompts.append(tokenizer.prefix(tags, face, size(face, X, Y, Z)))
                who.append((i, face))
        max_cells = int(max(p[-1] - tokenizer.vocab.coord_offset for p in prompts) *
                        max(p[-2] - tokenizer.vocab.coord_offset for p in prompts))
        seqs = generate(model, tokenizer, tags, len(prompts), 0, temperature, top_k, seed, batch=batch,
                        prompts=prompts, max_new=2 * max_cells + 1)
        for (i, face), seq in zip(who, seqs):
            f, _ = tokenizer.decode_face(seq)
            if f is not None:
                per_build[i][face] = f
    builds = []
    for i, (X, Y, Z) in enumerate(boxes):
        shape = (int(X), int(Y), int(Z))
        s = fc.lift(per_build[i], shape) if per_build[i] else Structure(
            block_ids=np.zeros(shape, np.int32), block_data=np.zeros(shape, np.int32))
        s.metadata.update({"corpus": corpus, "faces": ",".join(sorted(per_build[i]))})
        builds.append(s)
    return builds, per_build


def _train_dims(cfg, tokenizer: CorpusTokenizer) -> Tuple[np.ndarray, List[str]]:
    """Real bounding boxes and the face set from the first training shard's meta."""
    from blockgen.corpus.shards import Shard, find
    shard = Shard(find(cfg.train_specs()[0][0], tokenizer.hash, cfg.kind))
    m = shard.meta()
    front = m[m.face.isin(["-z", "joint"])]          # per-face shards: the front; joint shards: the box
    dims = np.stack([front.u.values, front.v.values, front.extent.values], 1).astype(int)
    return dims, list(shard.spec.get("faces") or [])


def sample_structures(model, tokenizer: CorpusTokenizer, cfg, n: int, seed: Optional[int] = None) -> Tuple[List[Structure], dict]:
    """``n`` draws from the model as cropped, non-empty structures, plus what the draw
    knew that a structure cannot carry (``extra``).

    Dispatches on ``tokenizer.kind``: voxel models sample token sequences under the
    config's tags, facade models sample every face of a real bounding box and lift
    them; pieces models sample like voxel ones and decode each piece at its anchor.
    ``extra`` holds ``n``, ``n_empty`` (draws replaced), ``seed``, ``lengths`` (tokens per draw,
    cells for facades), ``topup_rounds`` and, for facades, ``per_build`` (the faces, for the sheet).
    Empty draws are redrawn under a shifted seed, up to ``TOPUP_ROUNDS`` times, so the
    result has exactly ``n`` builds unless the model keeps emitting nothing.
    The stable entry point for post-training (best-of-N, rejection sampling) and for
    ``sample_and_render``.
    """
    seed = cfg.seed if seed is None else seed
    extra: dict = {"n": n, "seed": seed, "n_empty": 0, "lengths": []}
    keep: List[Structure] = []
    # Empty draws are replaced, so a card always gets exactly ``n`` builds (BlockScore is comparable only at
    # equal n). Round 0 is the old single draw under ``seed``, so earlier caches stay a prefix of new ones.
    for rnd in range(TOPUP_ROUNDS):
        want = n - len(keep)
        if want <= 0:
            break
        got, lengths, per_build = _draw(model, tokenizer, cfg, want, seed + rnd * TOPUP_SEED_STRIDE)
        extra["lengths"] += lengths
        if per_build is not None:
            extra.setdefault("per_build", []).extend(per_build)
        ok = [s for s in got if s.occupied_mask.any()]
        extra["n_empty"] += len(got) - len(ok)
        keep += ok
        if rnd:
            print(f"[sample] top-up round {rnd}: {len(ok)} of {want} non-empty", flush=True)
    extra["topup_rounds"] = rnd
    return keep[:n], extra


#: How many draws ``sample_structures`` makes at most to replace empty builds, and the seed step between them.
TOPUP_ROUNDS = 4
TOPUP_SEED_STRIDE = 1_000_003


def _draw(model, tokenizer: CorpusTokenizer, cfg, n: int, seed: int):
    """One draw of ``n``: cropped structures, tokens (cells for facades) per draw, and facade faces or None."""
    per_build = None
    if tokenizer.kind == "facades":
        dims, faces = _train_dims(cfg, tokenizer)
        structs, per_build = facade_builds(model, tokenizer, cfg.sample_tags, n, dims, faces,
                                           cfg.temperature, cfg.top_k, seed, corpus=cfg.name, batch=cfg.sample_batch)
        lengths = [sum(len(f.depth.ravel()) for f in fb.values()) for fb in per_build]
    else:
        seqs = generate(model, tokenizer, cfg.sample_tags, n, cfg.sample_max_blocks, cfg.temperature, cfg.top_k, seed,
                        batch=cfg.sample_batch)
        structs = to_structures(seqs, tokenizer, corpus=cfg.name)
        lengths = [len(q) for q in seqs]
    return [s.crop_to_non_air() for s in structs], lengths, per_build


def _facade_sheet(per_build, run_dir: str) -> None:
    """``facades.png``: the first eight builds' faces as the model drew them, for a reader."""
    try:
        from PIL import Image
        from blockgen.corpus import facade as fc
        sheets = [fc.sheet(f, scale=4) for f in per_build[:8] if f]
        if sheets:
            W = max(s.shape[1] for s in sheets)
            canvas = np.full((sum(s.shape[0] + 4 for s in sheets), W, 3), 255, np.uint8)
            y = 0
            for s in sheets:
                canvas[y:y + s.shape[0], :s.shape[1]] = s; y += s.shape[0] + 4
            Image.fromarray(canvas).save(os.path.join(run_dir, "facades.png"))
    except Exception as e:
        print(f"[sample] facade sheet failed: {e}", flush=True)


def sample_and_render(model, tokenizer: CorpusTokenizer, cfg, run_dir: str, step: int, n: int,
                      ckpt: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
    """Write ``samples.png`` and a structure cache the benchmark can score; returns the npz path.

    The draw is the arch's own ``sample`` (``archs.get(cfg.arch).sample``; for ``ar2``
    that is ``sample_structures``). ``step`` is the step the weights are from and
    ``ckpt`` the checkpoint file they were read from, both for the cache report; the
    trainer passes them after checkpoint selection, ``--sample`` what it loaded.
    ``name`` is the cache file's stem (default ``samples_step<step>``).
    """
    from blockgen.curation.houses import save_house_cache
    from blockgen.renderer.grid import save_grid
    from . import archs
    t0 = time.time()
    keep, extra = archs.get(cfg.arch).sample(model, tokenizer, cfg, n)
    if "per_build" in extra:
        _facade_sheet(extra["per_build"], run_dir)
    occ = [int(s.occupied_mask.sum()) for s in keep]
    print(f"[sample] step {step}: {n} samples in {(time.time() - t0) / 60:.1f} min, "
          f"{extra['n_empty']} empty, median {int(np.median(occ)) if occ else 0} blocks, "
          f"median length {int(np.median(extra['lengths'])) if extra['lengths'] else 0} tokens"
          + (f", {tokenizer.kind}" if tokenizer.kind != "voxels" else ""), flush=True)
    for attempt in (1, 2):                      # rendering must never kill a run; one retry rides out a GPU memory spike
        try:
            save_grid(keep[:min(len(keep), 64)], os.path.join(run_dir, "samples.png"), cols=8)
            break
        except Exception as e:
            print(f"[sample] render failed (attempt {attempt}): {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not keep:
        return None
    report = {"run": run_dir, "step": step, "n": n, "n_empty": extra["n_empty"], "seed": cfg.seed,
              "temperature": cfg.temperature, "top_k": cfg.top_k, "tags": cfg.sample_tags,
              "model": cfg.name, "arch": cfg.arch, "kind": tokenizer.kind,
              "checkpoint": {"file": ckpt, "step": step},
              "trained_on_signals": list(cfg.trained_on_signals or []),
              "train_split": cfg.split_name()}
    return save_house_cache(keep, tokenizer.max_dim, cache_dir=run_dir, name=name or f"samples_step{step}", report=report)


def bench(npz: str, cfg, run_dir: str) -> dict:
    """BlockScore on the samples at ``cfg.bench_tier``, in a subprocess so a bench failure stays a warning."""
    from .archs import ARCHS
    out = os.path.join(run_dir, "bench")
    track = ARCHS[cfg.arch].track if cfg.arch in ARCHS else "ar"      # the benchmark arm's track prefix
    cmd = [sys.executable, "-m", "blockgen.eval.bench", "--arms", f"{track}/{cfg.name}:{npz}", "--tier", cfg.bench_tier,
           "--n", str(cfg.bench_n), "--corpus", cfg.bench_corpus, "--out", out, "--name", f"train_{cfg.name}",
           "--no-head-to-head"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(run_dir, "bench.log"), "w").write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        print(f"[bench] failed (rc={r.returncode}); see bench.log", flush=True)
        return {"ok": False}
    summary = {"ok": True, "seconds": int(time.time() - t0), "dir": out, "tier": cfg.bench_tier}
    for p in sorted(glob.glob(os.path.join(out, "**", "*.json"), recursive=True), key=os.path.getmtime):
        try:
            blob = json.load(open(p))
        except Exception:
            continue
        found = _find_scores(blob, cfg.name)
        if found:
            summary.update(found); summary["card"] = p
    print(f"[bench] {json.dumps({k: v for k, v in summary.items() if k not in ('dir', 'card')})}", flush=True)
    return summary


def _find_scores(blob, arm: str) -> dict:
    """The arm's BlockScore entry from ``run.blockscore`` of a scorecard JSON."""
    run = blob.get("run") if isinstance(blob, dict) else None
    if not isinstance(run, dict):
        return {}
    for e in run.get("blockscore") or []:
        if isinstance(e, dict) and e.get("arm") == arm:
            return {"blockscore": e.get("score"), "status": e.get("status"), "worst_pillar": e.get("worst_pillar"),
                    "pillars": e.get("pillars"), "disqualified": e.get("disqualified"), "unranked": e.get("unranked")}
    return {}


# --------------------------------------------------------------------------- per-class sampling

def class_size_mix(cfg, tokenizer: CorpusTokenizer) -> Dict[str, Dict[str, float]]:
    """``{class: {size tag: share}}`` of the whole builds in the first training shard, so a
    class is prompted at the sizes it was trained at (a tower is a 48 more often than a farm)."""
    from blockgen.corpus.shards import Shard, find
    shard = Shard(find(cfg.train_specs()[0][0], tokenizer.hash, cfg.kind))
    m = shard.meta()
    if "cat" not in m:
        return {}
    m = m[m.win == "whole"]
    tab = m.groupby(["cat", "size"]).size()
    out: Dict[str, Dict[str, float]] = {}
    for (c, size), k in tab.items():
        out.setdefault(str(c), {})[str(size)] = float(k)
    return {c: {s: v / sum(d.values()) for s, v in d.items()} for c, d in out.items()}


def _split_n(n: int, mix: Dict[str, float]) -> List[Tuple[str, int]]:
    """``n`` draws over the sizes of ``mix``, largest-remainder rounding, so the counts sum to ``n``."""
    sizes = sorted(mix)
    raw = [n * mix[s] for s in sizes]
    k = [int(x) for x in raw]
    for i in sorted(range(len(sizes)), key=lambda i: raw[i] - k[i], reverse=True)[: n - sum(k)]:
        k[i] += 1
    return [(s, c) for s, c in zip(sizes, k) if c > 0]


def sample_classes(model, tokenizer: CorpusTokenizer, cfg, out_dir: str, n: int, cats: List[str],
                   step: int, ckpt: Optional[str] = None, sizes: str = "auto") -> Dict[str, str]:
    """``n`` builds per prompted class, each class its own structure cache and sheet.

    Every class ``c`` is drawn under ``cfg.sample_tags`` with ``cat:c``; ``sizes="auto"``
    splits its ``n`` over the size tags in the class's training share
    (``class_size_mix``), ``sizes="tags"`` keeps the one size in ``sample_tags``.
    Writes ``samples_cat-<c>_n<n>_<max_dim>.npz`` (``save_house_cache``),
    ``samples_cat-<c>.png`` and ``cat_samples.json`` (``{class: npz}``, the input of
    ``blockgen.eval.conditioned``) under ``out_dir``; returns the index.
    """
    import copy
    from blockgen.curation.houses import save_house_cache
    from blockgen.renderer.grid import save_grid
    from . import archs
    if not tokenizer.cats:
        raise ValueError(f"tokenizer {tokenizer.hash} has no category group; per-class sampling needs a cat model")
    bad = [c for c in cats if c not in tokenizer.cats]
    if bad:
        raise ValueError(f"unknown class(es) {bad}; the tokenizer has {tokenizer.cats}")
    mix = class_size_mix(cfg, tokenizer) if sizes == "auto" else {}
    index_path = os.path.join(out_dir, "cat_samples.json")
    try:
        index = json.load(open(index_path))
    except (OSError, ValueError):
        index = {}
    for c in cats:
        t0 = time.time()
        plan = _split_n(n, mix[c]) if c in mix else [(str(cfg.sample_tags.get("size", "32")), n)]
        keep, n_empty, lengths = [], 0, []
        for size, k in plan:
            sub = copy.copy(cfg)
            sub.sample_tags = {**cfg.sample_tags, "cat": c, "size": size}
            structs, extra = archs.get(cfg.arch).sample(model, tokenizer, sub, k)
            for s in structs:
                s.metadata["category"] = c
            keep += structs; n_empty += extra["n_empty"]; lengths += extra["lengths"]
        print(f"[sample] cat:{c}: {n} samples ({', '.join(f'size {s} x{k}' for s, k in plan)}) in "
              f"{(time.time() - t0) / 60:.1f} min, {n_empty} empty, median length "
              f"{int(np.median(lengths)) if lengths else 0} tokens", flush=True)
        try:
            save_grid(keep[:min(len(keep), 64)], os.path.join(out_dir, f"samples_cat-{c}.png"), cols=8)
        except Exception as e:
            print(f"[sample] render failed for cat:{c}: {e}", flush=True)
        if not keep:
            continue
        report = {"run": out_dir, "step": step, "n": n, "n_empty": n_empty, "seed": cfg.seed,
                  "temperature": cfg.temperature, "top_k": cfg.top_k, "tags": {**cfg.sample_tags, "cat": c},
                  "sizes": dict(plan), "model": cfg.name, "arch": cfg.arch, "kind": tokenizer.kind,
                  "checkpoint": {"file": ckpt, "step": step},
                  "trained_on_signals": list(cfg.trained_on_signals or []), "train_split": cfg.split_name()}
        index[c] = save_house_cache(keep, tokenizer.max_dim, cache_dir=out_dir, name=f"samples_cat-{c}_n{n}", report=report)
    json.dump(index, open(index_path, "w"), indent=1)
    return index
