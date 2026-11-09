"""Train and score the fixed-pooling face cascade on one face type.

Step 2 of the 2D next-scale plan (research.md H.7, notes §63): a bidirectional
transformer (`blockgen.models.face_cascade`) draws level ``f`` of a face from its
pooled level ``2f`` (`blockgen.corpus.facade_levels`), five levels from the box to
the relief, all cells of a level in one pass. No learned tokenizer: the levels are
``facade.pool`` under a rule (``any`` by default, ``majority`` as the control).

``train`` fits the model on the side faces of the train split (every side face is
a facade; ``--faces front`` restricts to ``-z``), mirrored along ``u`` as augmentation,
with NCM-style corruption of the parent level and MaskGIT-style partial masking of
the level being drawn, selects the checkpoint by validation NLL and then runs
``eval``. ``eval`` scores a run on the test split's front faces:

* teacher-forced NLL per level in bits per in-box cell;
* box-conditioned samples from the real test boxes, in one pass per level and
  with MaskGIT refinement, at temperature 1.0 and 0.7, scored with the face
  statistics the facade sheets report (jump rate, components, holes, lone cells,
  fill) against the real test faces and the front faces of the existing arms
  (patchwork, markov, the per-face and joint AR models); and
* oracle-coarse completion: the real level f=4 (and f=2) given, the finer levels
  drawn, agreement with the real face against the naive upsample of T75.

Writes ``samples.png`` (one box per row, the cascade's levels as columns, the real
face of the box last), ``completion.png``, ``stats.json``, ``stats.md`` into the run
dir and appends to ``outputs/registry.jsonl``.

    PYTHONPATH=. python scripts/facade_cascade.py train --name facade_cascade_5m
    PYTHONPATH=. python scripts/facade_cascade.py eval --run outputs/run_<stamp>_facade_cascade_5m
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from blockgen.corpus import facade as fc
from blockgen.corpus import facade_levels as fl
from blockgen.eval.bench import splits
from blockgen.eval.facade_stats import face_agreement, one_face_stats
from blockgen.models import face_cascade as mc
from blockgen.train import registry
from blockgen.utils.runs import new_run_dir

FACE_SETS = {"sides": fc.SIDES, "front": ("-z",), "all": fc.SIDES_TOP}
#: Existing arms whose front faces sit beside the cascade's on the stats table.
ARMS = {
    "patchwork": "outputs/bench_arms/facade_patchwork_32.npz",
    "markov": "outputs/bench_arms/facade_markov_32.npz",
    "per-face 5M AR": "outputs/run_20261026_153838_houses32_facades_5m/samples399/samples_step399_48.npz",
    "joint 20M D4 AR": "outputs/run_20261028_125240_houses32_facades_joint_d4_20m/samples399/samples_step399_48.npz",
    "joint 20M D4 RFT t0.7": "outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07/samples399",
}
STAT_COLS = ["jumps", "comps", "holes", "lone", "fill_frac", "n_depths", "n_blocks"]
SAMPLE_SETTINGS = [(1, 1.0), (4, 1.0), (1, 0.7), (4, 0.7)]       # (steps per level, temperature)
PRESETS = {"5m": dict(dim=256, layers=6, heads=8), "20m": dict(dim=512, layers=8, heads=8)}


# --- data --------------------------------------------------------------------
def load_faces(corpus: str, seed: int, split_name: str, faces: Sequence[str]) -> List[fc.Facade]:
    split = splits.load_split(corpus, seed)
    return fl.side_faces(splits.split_structures(split, split_name), faces)


class Levels:
    """The stacked arrays of a face set on a device, with the training corruptions."""

    def __init__(self, faces: List[fc.Facade], vocab: fl.BlockVocab, rule: str, device, flip: bool = False):
        if flip:
            faces = faces + [fl.flip_u(f) for f in faces]
        arr = fl.stack_arrays(faces, vocab, rule)
        self.n = len(faces)
        self.factors = fl.FACTORS
        self.depth = {f: torch.as_tensor(arr["depth"][f].astype(np.int64), device=device) for f in self.factors}
        self.block = {f: torch.as_tensor(arr["block"][f].astype(np.int64), device=device) for f in self.factors}
        self.inbox = {f: torch.as_tensor(arr["inbox"][f], device=device) for f in self.factors}
        self.box = torch.as_tensor(arr["box"].astype(np.int64), device=device)
        self.device = device

    def parent_of(self, f: int, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.factors.index(f)
        if k == 0:
            return None, None
        pf = self.factors[k - 1]
        return mc.repeat2(self.depth[pf][idx], pf // f), mc.repeat2(self.block[pf][idx], pf // f)

    def batch(self, model: mc.FaceCascade, f: int, idx: torch.Tensor, corrupt_max: float, p_full: float,
              generator: Optional[torch.Generator] = None) -> dict:
        """A training batch of level ``f``: the parent level, corrupted; the level's
        own cells, partly revealed; targets and the loss mask."""
        B = len(idx)
        G = fl.grid_size(f)
        dev = self.device
        td, tb, inbox, box = self.depth[f][idx], self.block[f][idx], self.inbox[f][idx], self.box[idx]
        pd, pb = self.parent_of(f, idx)
        if pd is None:
            pd, pb = model.none_parent(B, G, dev)
        elif corrupt_max > 0:
            rate = torch.rand((B, 1, 1), device=dev, generator=generator) * corrupt_max
            hit = (torch.rand((B, G, G), device=dev, generator=generator) < rate) & inbox
            rand_d = (torch.rand((B, G, G), device=dev, generator=generator) * (box[:, 2].view(B, 1, 1) + 1)).long()
            rand_b = torch.randint(1, model.n_blocks, (B, G, G), device=dev, generator=generator)
            rand_b = torch.where(rand_d > 0, rand_b, torch.zeros_like(rand_b))
            pd = torch.where(hit, rand_d, pd)
            pb = torch.where(hit, rand_b, pb)
        full = torch.rand((B, 1, 1), device=dev, generator=generator) < p_full
        ratio = torch.where(full, torch.ones_like(full, dtype=torch.float),
                            torch.rand((B, 1, 1), device=dev, generator=generator))
        masked = (torch.rand((B, G, G), device=dev, generator=generator) < ratio) & inbox
        # a sample with nothing masked contributes nothing; mask one in-box cell at least
        none = ~masked.reshape(B, -1).any(1)
        if none.any():
            first = inbox.reshape(B, -1).float().argmax(1)
            fix = torch.zeros((B, G * G), dtype=torch.bool, device=dev)
            fix[torch.arange(B, device=dev), first] = True
            masked = masked | (fix.reshape(B, G, G) & none.view(B, 1, 1))
        od = torch.where(masked, torch.full_like(td, model.depth_mask_id), td)
        ob = torch.where(masked, torch.full_like(tb, model.block_mask_id), tb)
        od = torch.where(inbox, od, torch.zeros_like(od))
        ob = torch.where(inbox, ob, torch.zeros_like(ob))
        return dict(parent_depth=pd, parent_block=pb, own_depth=od, own_block=ob, inbox=inbox, box=box,
                    target_depth=td, target_block=tb, loss_mask=masked)

    def teacher_forced(self, model: mc.FaceCascade, f: int, idx: torch.Tensor) -> dict:
        """Everything masked, the true parent: the NLL of a level given the one above."""
        B = len(idx)
        G = fl.grid_size(f)
        pd, pb = self.parent_of(f, idx)
        if pd is None:
            pd, pb = model.none_parent(B, G, self.device)
        od, ob = model.masked_own(B, G, self.device)
        inbox = self.inbox[f][idx]
        return dict(parent_depth=pd, parent_block=pb, own_depth=od, own_block=ob, inbox=inbox, box=self.box[idx],
                    target_depth=self.depth[f][idx], target_block=self.block[f][idx], loss_mask=inbox)


@torch.no_grad()
def nll_bits(model: mc.FaceCascade, data: Levels, batch_size: int, amp: bool) -> Dict[str, float]:
    """Teacher-forced NLL per level in bits per in-box cell (depth + block, the block
    term over filled cells spread onto all cells), and the total over levels."""
    model.eval()
    out = {}
    total, cells = 0.0, 0.0
    for f in data.factors:
        sd = sb = nd = 0.0
        for s in range(0, data.n, batch_size):
            idx = torch.arange(s, min(s + batch_size, data.n), device=data.device)
            b = data.teacher_forced(model, f, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _, parts = mc.level_loss(model, f, **b)
            sd += parts["sum_depth"].item(); sb += parts["sum_block"].item(); nd += parts["n_depth"].item()
        bits = (sd + sb) / max(nd, 1) / math.log(2)
        out[f"f{f}"] = bits
        total += sd + sb
        cells += nd
    out["total_bits_per_cell"] = total / max(cells, 1) / math.log(2)
    out["bits_per_face"] = total / math.log(2) / max(data.n, 1)
    model.train()
    return out


# --- faces from the model ----------------------------------------------------
def levels_to_faces(levels: Dict[int, Tuple[torch.Tensor, torch.Tensor]], boxes: np.ndarray, face: str,
                    vocab: fl.BlockVocab) -> Dict[int, List[fc.Facade]]:
    out = {}
    for f, (d, b) in levels.items():
        d, b = d.cpu().numpy(), b.cpu().numpy()
        out[f] = [fl.face_from_arrays(d[i], b[i], f, (int(boxes[i, 0]), int(boxes[i, 1])), int(boxes[i, 2]),
                                      face, vocab) for i in range(len(boxes))]
    return out


@torch.no_grad()
def draw(model: mc.FaceCascade, boxes: np.ndarray, face: str, vocab: fl.BlockVocab, device, steps: int,
         temperature: float, seed: int, batch_size: int = 128,
         start: Optional[Tuple[int, np.ndarray, np.ndarray]] = None,
         per_level: Optional[Dict[int, mc.FaceCascade]] = None) -> Dict[int, List[fc.Facade]]:
    """Faces for every box (``[N, 3]``), optionally completing from a given level."""
    model.eval()
    for m in (per_level or {}).values():
        m.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    out: Dict[int, List[fc.Facade]] = {f: [] for f in model.factors}
    for s in range(0, len(boxes), batch_size):
        bx = boxes[s:s + batch_size]
        box_t = torch.as_tensor(bx.astype(np.int64), device=device)
        inbox = {f: torch.as_tensor(np.stack([fl.inbox_grid((int(b[0]), int(b[1])), f) for b in bx]), device=device)
                 for f in model.factors}
        st = None
        if start is not None:
            st = (start[0], torch.as_tensor(start[1][s:s + batch_size].astype(np.int64), device=device),
                  torch.as_tensor(start[2][s:s + batch_size].astype(np.int64), device=device))
        lv = mc.cascade(model, box_t, inbox, steps, temperature, gen, start=st, per_level=per_level)
        for f, faces in levels_to_faces(lv, bx, face, vocab).items():
            out[f].extend(faces)
    return {f: v for f, v in out.items() if v}


def arm_faces(path: str, face: str) -> List[fc.Facade]:
    """Front faces of an arm's builds (a structure cache or a directory holding one)."""
    from blockgen.curation.houses import load_structures_from_cache
    if os.path.isdir(path):
        cands = [p for p in sorted(os.listdir(path)) if p.endswith(".npz")]
        if not cands:
            return []
        path = os.path.join(path, cands[0])
    structs, _ = load_structures_from_cache(path)
    return fl.side_faces([s.crop_to_non_air() for s in structs if s.occupied_mask.any()], (face,))


def stats_table(groups: Dict[str, List[fc.Facade]], real_key: str = "real") -> Dict[str, dict]:
    """Mean face statistics per group and the 1D Wasserstein distance of each
    statistic's distribution to the real group's."""
    from scipy.stats import wasserstein_distance
    per = {k: [one_face_stats(f) for f in v] for k, v in groups.items()}
    out = {}
    for k, st in per.items():
        row = {"n": len(st)}
        for c in STAT_COLS:
            vals = np.array([s[c] for s in st])
            row[c] = float(vals.mean()) if len(vals) else float("nan")
            if k != real_key and len(vals) and real_key in per:
                ref = np.array([s[c] for s in per[real_key]])
                row[f"w1_{c}"] = float(wasserstein_distance(ref, vals))
        row["empty_faces"] = float(np.mean([s["filled"] == 0 for s in st])) if st else float("nan")
        out[k] = row
    return out


def agreement_table(real: List[fc.Facade], others: Dict[str, List[fc.Facade]]) -> Dict[str, dict]:
    out = {}
    for k, faces in others.items():
        rows = [face_agreement(r, o) for r, o in zip(real, faces)]
        out[k] = {c: float(np.nanmean([r[c] for r in rows])) for c in rows[0]}
        out[k]["n"] = len(rows)
    return out


# --- sheets ------------------------------------------------------------------
def draw_sheet(rows: List[List[fc.Facade]], labels: List[str], scale: int = 5, gap: int = 6,
               header: int = 14) -> np.ndarray:
    pics = [[np.repeat(np.repeat(fc.face_picture(f), scale, 0), scale, 1) for f in r] for r in rows]
    col_w = [max(p[c].shape[1] for p in pics) for c in range(len(labels))]
    row_h = [max(p.shape[0] for p in r) for r in pics]
    W = sum(col_w) + gap * (len(labels) + 1)
    H = header + sum(row_h) + gap * (len(rows) + 1)
    out = np.full((H, W, 3), 255, np.uint8)
    y = header + gap
    for r, h in zip(pics, row_h):
        x = gap
        for c, p in enumerate(r):
            out[y + h - p.shape[0]:y + h, x:x + p.shape[1]] = p
            x += col_w[c] + gap
        y += h + gap
    img = Image.fromarray(out)
    dr = ImageDraw.Draw(img)
    x = gap
    for c, lab in enumerate(labels):
        dr.text((x, 1), lab, fill=(0, 0, 0))
        x += col_w[c] + gap
    return np.asarray(img)


# --- the model on disk -------------------------------------------------------
def save_model(path: Path, model: mc.FaceCascade, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(run: Path, ckpt: str, device) -> Tuple[mc.FaceCascade, dict, fl.BlockVocab]:
    ck = torch.load(run / ckpt, map_location=device, weights_only=False)
    meta = ck["meta"]
    vocab = fl.BlockVocab.from_dict(json.loads((run / "vocab.json").read_text()))
    model = mc.FaceCascade(vocab.size, **meta["arch"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, meta, vocab


def load_per_level(run: Path, device) -> Dict[int, mc.FaceCascade]:
    """``best_f<f>.pt`` per level where present: the checkpoint each level's own
    validation NLL selects."""
    out = {}
    for f in fl.FACTORS:
        if (run / f"best_f{f}.pt").exists():
            out[f] = load_model(run, f"best_f{f}.pt", device)[0]
    return out


# --- eval --------------------------------------------------------------------
def evaluate(run: Path, model: mc.FaceCascade, meta: dict, vocab: fl.BlockVocab, device, args) -> dict:
    face = args.eval_face
    rule = meta["rule"]
    per_level = load_per_level(run, device)
    test_faces = load_faces(meta["corpus"], meta["seed"], "test", (face,))
    if args.n and args.n < len(test_faces):
        test_faces = test_faces[:args.n]
    print(f"[eval] {len(test_faces)} {face} test faces", flush=True)
    amp = device.type == "cuda"
    test = Levels(test_faces, vocab, rule, device)
    report: dict = {"run": str(run), "face": face, "rule": rule, "n_test": len(test_faces), "meta": meta}
    report["test_nll_bits"] = nll_bits(model, test, 256, amp)
    print("[eval] test NLL bits/cell per level:", {k: round(v, 3) for k, v in report["test_nll_bits"].items()},
          flush=True)
    if per_level:
        steps_of = {f: torch.load(run / f"best_f{f}.pt", map_location="cpu", weights_only=False)["meta"]["step"]
                    for f in per_level}
        report["per_level_steps"] = steps_of
        per_bits = {f: nll_bits(m_, test, 256, amp)[f"f{f}"] for f, m_ in per_level.items()}
        report["test_nll_bits_per_level_best"] = per_bits
        print("[eval] per-level-best checkpoints:", steps_of, {f: round(v, 3) for f, v in per_bits.items()}, flush=True)

    boxes = test.box.cpu().numpy()
    groups: Dict[str, List[fc.Facade]] = {"real": test_faces}
    sheets: Dict[str, Dict[int, List[fc.Facade]]] = {}
    for steps, temp in SAMPLE_SETTINGS:
        for sel, pl in (("", None), (" lvbest", per_level)):
            if pl is not None and not pl:
                continue
            tag = f"cascade s{steps} T{temp}{sel}"
            t0 = time.time()
            lv = draw(model, boxes, face, vocab, device, steps, temp, args.seed, per_level=pl)
            groups[tag] = lv[1]
            sheets[tag] = lv
            print(f"[eval] {tag}: {len(lv[1])} faces in {time.time() - t0:.1f}s", flush=True)
    for name, path in ARMS.items():
        if os.path.exists(path):
            faces = arm_faces(path, face)
            if faces:
                groups[name] = faces
    report["face_stats"] = stats_table(groups)

    # oracle-coarse completion from the real level f0, against the naive upsample
    comp: Dict[str, List[fc.Facade]] = {}
    for f0 in (4, 2):
        start = (f0, test.depth[f0].cpu().numpy(), test.block[f0].cpu().numpy())
        lv = draw(model, boxes, face, vocab, device, 1, 1.0, args.seed, start=start)
        comp[f"complete from f{f0}"] = lv[1]
        comp[f"mode from f{f0}"] = draw(model, boxes, face, vocab, device, 1, 0.0, args.seed, start=start)[1]
        comp[f"upsample f{f0}"] = [fc.upsample(fc.pool(f, f0, rule), f0, f.depth.shape) for f in test_faces]
        if f0 == 4:
            completion_sheet = lv
    report["completion"] = agreement_table(test_faces, comp)
    report["completion_stats"] = stats_table({"real": test_faces, **comp})

    # sheets: the cascade's levels for twelve boxes, with the real face of the box last
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(test_faces), min(12, len(test_faces)), replace=False)
    lv = sheets["cascade s1 T1.0"]
    rows = [[lv[f][i] for f in model.factors] + [test_faces[i]] for i in pick]
    img = draw_sheet(rows, [f"f={f}" for f in model.factors] + ["real (same box)"])
    Image.fromarray(img).save(run / "samples.png")
    lv4 = sheets["cascade s4 T1.0"]
    rows = [[lv4[f][i] for f in model.factors] + [test_faces[i]] for i in pick]
    Image.fromarray(draw_sheet(rows, [f"f={f}" for f in model.factors] + ["real (same box)"])).save(
        run / "samples_s4.png")
    rows = [[test_faces[i], comp["upsample f4"][i], completion_sheet[1][i], comp["mode from f4"][i],
             comp["complete from f2"][i], sheets["cascade s1 T1.0"][1][i]] for i in pick]
    Image.fromarray(draw_sheet(rows, ["real", "its f=4", "completed from f=4", "mode from f=4",
                                      "completed from f=2", "from the box"])).save(run / "completion.png")
    if per_level:
        lv = sheets["cascade s1 T1.0 lvbest"]
        rows = [[lv[f][i] for f in model.factors] + [test_faces[i]] for i in pick]
        Image.fromarray(draw_sheet(rows, [f"f={f}" for f in model.factors] + ["real (same box)"])).save(
            run / "samples_lvbest.png")

    (run / "stats.json").write_text(json.dumps(report, indent=2, default=float))
    (run / "stats.md").write_text(report_md(report))
    print(report_md(report), flush=True)
    return report


def report_md(r: dict) -> str:
    L = [f"# Face cascade: {r['run']}", "",
         f"{r['n_test']} {r['face']} test faces, pooling rule `{r['rule']}`, arch {r['meta']['arch']}, "
         f"{r['meta'].get('params', '?')} params, best step {r['meta'].get('step', '?')}.", "",
         "## Teacher-forced NLL, bits per in-box cell", "",
         "| " + " | ".join(r["test_nll_bits"]) + " |", "|" + "--:|" * len(r["test_nll_bits"]),
         "| " + " | ".join(f"{v:.3f}" for v in r["test_nll_bits"].values()) + " |", ""]
    if r.get("per_level_steps"):
        L += ["Per-level-best checkpoints (step: test bits): " + ", ".join(
            f"f{f} {r['per_level_steps'][f]}: {r['test_nll_bits_per_level_best'][f]:.3f}" for f in r["per_level_steps"]), ""]
    L += [
         "## Face statistics of front faces (means; W1 = distance of the per-face distribution to real)", "",
         "| arm | n | " + " | ".join(STAT_COLS) + " | empty | W1 jumps | W1 comps | W1 fill |",
         "|---|--:|" + "--:|" * (len(STAT_COLS) + 4)]
    for k, row in r["face_stats"].items():
        L.append(f"| {k} | {row['n']} | " + " | ".join(f"{row[c]:.3f}" for c in STAT_COLS)
                 + f" | {row['empty_faces']:.3f} | " + " | ".join(
                     f"{row.get('w1_' + c, float('nan')):.3f}" for c in ("jumps", "comps", "fill_frac")) + " |")
    L += ["", "## Oracle-coarse completion: agreement with the real face", "",
          "| from | mask IoU | recall | precision | depth exact | depth MAE | block agree |", "|---|--:|--:|--:|--:|--:|--:|"]
    for k, row in r["completion"].items():
        L.append(f"| {k} | {row['mask_iou']:.3f} | {row['mask_recall']:.3f} | {row['mask_precision']:.3f} | "
                 f"{row['depth_exact']:.3f} | {row['depth_mae']:.2f} | {row['block_agree']:.3f} |")
    L += ["", "| from | jumps | comps | holes | lone | fill |", "|---|--:|--:|--:|--:|--:|"]
    for k, row in r["completion_stats"].items():
        L.append(f"| {k} | {row['jumps']:.3f} | {row['comps']:.2f} | {row['holes']:.3f} | {row['lone']:.4f} | "
                 f"{row['fill_frac']:.3f} |")
    return "\n".join(L) + "\n"


# --- train -------------------------------------------------------------------
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    run = Path(args.out) if args.out else new_run_dir(args.name)
    run.mkdir(parents=True, exist_ok=True)
    faces = FACE_SETS[args.faces]
    train_faces = load_faces(args.corpus, args.seed, "train", faces)
    val_faces = load_faces(args.corpus, args.seed, "val", (args.eval_face,))
    if args.limit:
        train_faces, val_faces = train_faces[:args.limit], val_faces[:max(args.limit // 4, 4)]
    vocab = fl.BlockVocab.fit(train_faces, max_size=args.vocab)
    (run / "vocab.json").write_text(json.dumps(vocab.to_dict()))
    t0 = time.time()
    data = Levels(train_faces, vocab, args.rule, device, flip=not args.no_flip)
    val = Levels(val_faces, vocab, args.rule, device)
    print(f"[train] {data.n} train faces ({len(train_faces)} before flips), {val.n} val {args.eval_face} faces, "
          f"vocab {vocab.size}, levels built in {time.time() - t0:.0f}s on {device}", flush=True)

    arch = dict(PRESETS[args.preset], dropout=args.dropout)
    if args.dim: arch["dim"] = args.dim
    if args.layers: arch["layers"] = args.layers
    model = mc.FaceCascade(vocab.size, **arch).to(device)
    params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    meta = dict(name=args.name, corpus=args.corpus, seed=args.seed, faces=list(faces), eval_face=args.eval_face,
                rule=args.rule, arch=arch, params=params, steps=args.steps, batch=args.batch, lr=args.lr,
                corrupt_max=args.corrupt, p_full=args.p_full, flip=not args.no_flip, vocab=vocab.size,
                git=git_sha(), device=str(device))
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] {params / 1e6:.2f}M params, {args.steps} steps, batch {args.batch}; run {run}", flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = open(run / "metrics.jsonl", "a")
    best = float("inf")
    best_step = 0
    since_best = 0
    best_level = {f: float("inf") for f in data.factors}
    model.train()
    t0 = time.time()
    acc = {f: [0.0, 0.0] for f in data.factors}
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            ev = nll_bits(model, val, 256, amp)
            rec = {"step": step, "val_bits": ev, "lr": lr_at(step), "elapsed": time.time() - t0,
                   "train_bits": {f"f{f}": (a[0] / max(a[1], 1)) / math.log(2) for f, a in acc.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            acc = {f: [0.0, 0.0] for f in data.factors}
            print(f"[train] step {step} val {ev['total_bits_per_cell']:.4f} bits/cell "
                  f"({' '.join(f'f{f}={ev[f'f{f}']:.3f}' for f in data.factors)}) {rec['elapsed']:.0f}s", flush=True)
            save_model(run / "last.pt", model, {**meta, "step": step})
            for f in data.factors:                 # the checkpoint each level would pick for itself
                if ev[f"f{f}"] < best_level[f]:
                    best_level[f] = ev[f"f{f}"]
                    save_model(run / f"best_f{f}.pt", model, {**meta, "step": step, "val_bits": ev})
            if ev["total_bits_per_cell"] < best:
                best, best_step, since_best = ev["total_bits_per_cell"], step, 0
                save_model(run / "best.pt", model, {**meta, "step": step, "val_bits": ev})
            else:
                since_best += 1
                if args.patience and since_best >= args.patience:
                    print(f"[train] early stop at {step}: no val gain for {args.patience} evals", flush=True)
                    break
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        f = data.factors[int(torch.randint(len(data.factors), (1,), generator=gen, device=device).item())]
        idx = torch.randint(data.n, (args.batch,), generator=gen, device=device)
        b = data.batch(model, f, idx, args.corrupt, args.p_full, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, parts = mc.level_loss(model, f, **b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc[f][0] += (parts["sum_depth"] + parts["sum_block"]).item()
        acc[f][1] += parts["n_depth"].item()
    log.close()
    summary = {"best_val_bits_per_cell": best, "best_step": best_step, "train_seconds": time.time() - t0}
    (run / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    print(f"[train] done: best val {best:.4f} bits/cell at step {best_step}", flush=True)
    registry.append({"run_dir": str(run), "kind": "facade_cascade", "name": args.name, **summary,
                     "params": params, "rule": args.rule, "arch": arch, "git": meta["git"]})
    if not args.no_eval:
        model, meta, vocab = load_model(run, "best.pt", device)
        evaluate(run, model, meta, vocab, device, args)
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--name", default="facade_cascade_5m")
    tr.add_argument("--out", default=None, help="run dir; default outputs/run_<stamp>_<name>")
    tr.add_argument("--corpus", default="houses_32")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--faces", choices=sorted(FACE_SETS), default="sides", help="train faces")
    tr.add_argument("--eval-face", default="-z")
    tr.add_argument("--rule", choices=fl.fc.POOL_RULES, default="any")
    tr.add_argument("--preset", choices=sorted(PRESETS), default="5m")
    tr.add_argument("--dim", type=int, default=0)
    tr.add_argument("--layers", type=int, default=0)
    tr.add_argument("--vocab", type=int, default=512, help="block vocabulary cap")
    tr.add_argument("--steps", type=int, default=30000)
    tr.add_argument("--batch", type=int, default=96)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--wd", type=float, default=0.1)
    tr.add_argument("--warmup", type=int, default=500)
    tr.add_argument("--eval-every", type=int, default=1000)
    tr.add_argument("--patience", type=int, default=8, help="evals without val gain before stopping; 0 = never")
    tr.add_argument("--corrupt", type=float, default=0.3, help="max fraction of parent cells randomised")
    tr.add_argument("--dropout", type=float, default=0.0)
    tr.add_argument("--p-full", type=float, default=0.5, help="share of batches with the whole level masked")
    tr.add_argument("--no-flip", action="store_true")
    tr.add_argument("--no-eval", action="store_true")
    tr.add_argument("--limit", type=int, default=0, help="train faces to use (smoke tests)")
    tr.add_argument("--n", type=int, default=0, help="test faces for eval (0 = all)")
    tr.add_argument("--cpu", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("--run", required=True)
    ev.add_argument("--ckpt", default="best.pt")
    ev.add_argument("--eval-face", default="-z")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--n", type=int, default=0)
    ev.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        run = Path(args.run)
        model, meta, vocab = load_model(run, args.ckpt, device)
        evaluate(run, model, meta, vocab, device, args)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ".")
    sys.exit(main())
