"""Does a pretrained LLM's attention carry 3D adjacency? (probe, not a generator)

The proposal this tests: take a pretrained LLM, read its self-attention matrix `A`
while it processes a serialized build, and treat `A[i, j]` as the probability that
block `i` connects to block `j` -- "attention as an adjacency matrix", so the
structural prior comes for free from pretraining instead of from training.

The test is direct. For every held-out house we know the true answer: piece `i` is
6-adjacent to piece `j`, or it is not. So for each attention head we ask how well
`A[i, j]` ranks the truly-adjacent previous pieces above the non-adjacent ones,
scored as AUC over the causal candidate set (`j < i`) -- exactly the pointer-network
question "which already-placed piece does this one attach to".

Three reference points, because "AUC 0.7" alone means nothing:

* **recency** -- score `-(i - j)`. Lines are emitted in `(y, z, x)` raster order, so
  sequence distance already predicts adjacency. Any head must beat this to be
  carrying structure rather than position.
* **base vs finetuned** -- the same probe with the LoRA adapter off (the literal
  "steal it from a pretrained model" claim) and on (does the Track D finetune
  *create* adjacency-carrying heads?).
* **hidden-state probe** -- a low-rank bilinear head trained on the same layer's
  hidden states. This is the supervised ceiling for "the information is in there
  somewhere", and the thing raw attention has to be compared against: the
  attention-probing literature's standing result is that a trained probe wins.

Head selection is honest: the best head is chosen on a *train* half of the held-out
builds and reported on the *test* half, with the oracle-on-test number quoted
separately as the upper bound.

Run::

    .venv/bin/python scripts/probe_llm_attention.py                 # full
    .venv/bin/python scripts/probe_llm_attention.py --n-builds 8    # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import load_piece_vocab
from blockgen.tokenizers.cluster_bpe import _apply_merge, _atomic_labeling
from blockgen.utils.runs import new_run_dir
from train_llm_pieces import MODEL_NAME, CAPTIONS_PATH, PIECE_VOCAB, _base_vocab, build_piece_names

ADAPTER = "outputs/run_20260722_072022_llm_pieces/adapter"
NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


# --- ground truth ----------------------------------------------------------
def piece_instances(s, cv, id_to_name) -> List[dict] | None:
    """Ordered piece instances, one per emitted line: `{name, anchor, vox}`.

    Replays the learned merges exactly like `train_llm_pieces.serialize_pieces`, but
    keeps the instance records (we need each piece's voxels, not just its anchor).
    """
    c = s.crop_to_non_air()
    if max(c.shape) > cv.max_dim:
        return None
    inst, owner, nid = _atomic_labeling(c, _base_vocab(cv))
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)
    order = sorted(inst.values(), key=lambda d: (d["anchor"][1], d["anchor"][2], d["anchor"][0]))
    return [{"name": id_to_name[d["piece"]], "anchor": d["anchor"], "vox": d["vox"]} for d in order]


def adjacency_matrix(instances: List[dict]) -> np.ndarray:
    """`[L, L]` bool: piece i has a voxel 6-adjacent to a voxel of piece j."""
    owner: Dict[Tuple[int, int, int], int] = {}
    for i, d in enumerate(instances):
        for v in d["vox"]:
            owner[tuple(v)] = i
    L = len(instances)
    adj = np.zeros((L, L), dtype=bool)
    for i, d in enumerate(instances):
        for (x, y, z) in d["vox"]:
            for dx, dy, dz in NEIGHBORS:
                j = owner.get((x + dx, y + dy, z + dz))
                if j is not None and j != i:
                    adj[i, j] = adj[j, i] = True
    return adj


# --- tokenization with line spans -----------------------------------------
def line_spans(tokenizer, prompt: str, lines: List[str], max_tokens: int):
    """Tokenize `prompt + "\\n".join(lines)`, keeping only whole lines that fit.

    Returns `(input_ids, spans, n_lines)` where `spans[i]` is the `[start, end)`
    token range of line `i`. Truncation keeps a *prefix* of the build, which in
    raster order is a valid partial structure -- so the adjacency ground truth over
    the retained pieces stays exactly correct.
    """
    completion = "\n".join(lines)
    text = prompt + completion
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    # char range of each line inside `text`
    bounds, pos = [], len(prompt)
    for ln in lines:
        bounds.append((pos, pos + len(ln)))
        pos += len(ln) + 1  # + "\n"

    spans: List[Tuple[int, int]] = []
    li = 0
    starts = [b[0] for b in bounds]
    tok_line = np.full(len(ids), -1, dtype=np.int64)
    for t, (a, b) in enumerate(offsets):
        if a < len(prompt):
            continue
        # line whose char range contains this token's first char
        while li + 1 < len(bounds) and a >= starts[li + 1]:
            li += 1
        tok_line[t] = li
    for i in range(len(lines)):
        idx = np.flatnonzero(tok_line == i)
        spans.append((int(idx[0]), int(idx[-1]) + 1) if idx.size else (-1, -1))

    keep = len(lines)
    for i in range(len(lines)):
        if spans[i][0] < 0 or spans[i][1] > max_tokens:
            keep = i
            break
    if keep < len(lines):
        spans = spans[:keep]
        ids = ids[: spans[-1][1]] if keep else ids[: len(prompt)]
    return ids, spans, keep


# --- pooling + AUC ---------------------------------------------------------
def pool_attention(attn: torch.Tensor, spans: List[Tuple[int, int]]) -> torch.Tensor:
    """`[H, T, T]` token attention -> `[H, L, L]` line attention.

    Mean over the query line's tokens, sum over the key line's tokens: the average
    attention mass line `i` places on line `j`. Prompt tokens (which include the BOS
    attention sink, typically 30-60% of the mass) are dropped and each row is
    renormalized over the previous lines, so rows are comparable across builds.
    """
    H, T, _ = attn.shape
    L = len(spans)
    q_idx = torch.zeros(T, dtype=torch.long, device=attn.device)
    k_sel = torch.zeros((T, L), dtype=attn.dtype, device=attn.device)
    q_cnt = torch.zeros(L, device=attn.device)
    for i, (a, b) in enumerate(spans):
        q_idx[a:b] = i
        k_sel[a:b, i] = 1.0
        q_cnt[i] = b - a
    keep = torch.zeros(T, dtype=torch.bool, device=attn.device)
    keep[spans[0][0]: spans[-1][1]] = True

    a = attn[:, keep][:, :, :]                       # [H, Tq, T]
    m = torch.einsum("hqt,tl->hql", a, k_sel)        # [H, Tq, L]
    rows = q_idx[keep]
    out = torch.zeros((H, L, L), dtype=torch.float32, device=attn.device)
    out.index_add_(1, rows, m.float())
    out = out / q_cnt.clamp(min=1)[None, :, None]
    return out


def auc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUC for each row of `scores` `[H, P]` against shared binary `labels` `[P]`.

    Mann-Whitney U with average ranks (ties matter: the recency baseline is heavily
    tied). Returns `[H]`, NaN when a build has no positives or no negatives.
    """
    H, P = scores.shape
    n_pos = int(labels.sum())
    n_neg = P - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(H, np.nan)
    order = np.argsort(scores, axis=1, kind="mergesort")
    ss = np.take_along_axis(scores, order, axis=1)
    newgrp = np.empty(ss.shape, dtype=bool)
    newgrp[:, 0] = True
    np.not_equal(ss[:, 1:], ss[:, :-1], out=newgrp[:, 1:])
    grp = np.cumsum(newgrp, axis=1) - 1 + (np.arange(H)[:, None] * P)
    flat = grp.ravel()
    cnt = np.bincount(flat, minlength=H * P)
    pos = np.tile(np.arange(1, P + 1, dtype=np.float64), (H, 1)).ravel()
    tot = np.bincount(flat, weights=pos, minlength=H * P)
    ranks_sorted = (tot / np.maximum(cnt, 1))[grp]
    ranks = np.empty_like(ranks_sorted)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)
    r_pos = ranks[:, labels].sum(axis=1)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


DIST_BINS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10**9))


def auc_rows_stratified(scores: np.ndarray, labels: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """AUC computed *within* sequence-distance bins, then pooled by bin weight.

    Lines are emitted in `(y, z, x)` raster order, so `i - j` alone predicts adjacency
    well (the `recency` baseline). Holding distance roughly fixed removes that, and
    asks the only question that matters: does the head know anything about neighbours
    that position does not already say? Recency itself scores ~0.5 here by
    construction, which is the point -- this view has a floor of chance.
    """
    H = scores.shape[0]
    num = np.zeros(H)
    den = 0.0
    for lo, hi in DIST_BINS:
        m = (dist >= lo) & (dist <= hi)
        if m.sum() < 2:
            continue
        lab = labels[m]
        n_pos, n_neg = int(lab.sum()), int((~lab).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        w = float(n_pos * n_neg)
        num += auc_rows(scores[:, m], lab) * w
        den += w
    return num / den if den > 0 else np.full(H, np.nan)


def candidate_pairs(L: int, adj: np.ndarray, max_pairs: int, rng: random.Random):
    """Causal candidates `(i, j), j < i` with labels; uniformly subsampled if huge."""
    ii, jj = np.tril_indices(L, k=-1)
    lab = adj[ii, jj]
    if ii.size > max_pairs:
        sel = np.array(rng.sample(range(ii.size), max_pairs))
        ii, jj, lab = ii[sel], jj[sel], lab[sel]
    return ii, jj, lab


# --- bilinear hidden-state probe ------------------------------------------
class BilinearProbe(torch.nn.Module):
    """`score(i, j) = (W h_i) . (V h_j) + b` -- the supervised ceiling."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.W = torch.nn.Linear(dim, rank, bias=False)
        self.V = torch.nn.Linear(dim, rank, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hi, hj):
        return (self.W(hi) * self.V(hj)).sum(-1) + self.b


def train_probe(builds_tr, builds_te, key: str, device, epochs=12, rank=64, seed=0):
    """Train on `builds_tr`, return mean per-build AUC on `builds_te`."""
    torch.manual_seed(seed)
    dim = builds_tr[0][key].shape[1]
    probe = BilinearProbe(dim, rank).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(builds_tr)))
        rng.shuffle(order)
        for bi in order:
            b = builds_tr[bi]
            h = torch.as_tensor(b[key], dtype=torch.float32, device=device)
            ii, jj, lab = b["pairs"]
            n = min(len(ii), 4096)
            sel = np.array(rng.sample(range(len(ii)), n)) if len(ii) > n else np.arange(len(ii))
            s = probe(h[ii[sel]], h[jj[sel]])
            y = torch.as_tensor(lab[sel].astype(np.float32), device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s, y, pos_weight=torch.tensor(float((~lab).sum()) / max(lab.sum(), 1), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    aucs, aucs_s = [], []
    with torch.no_grad():
        for b in builds_te:
            h = torch.as_tensor(b[key], dtype=torch.float32, device=device)
            ii, jj, lab = b["pairs"]
            s = probe(h[ii], h[jj]).cpu().numpy()[None, :]
            aucs.append(float(auc_rows(s, lab)[0]))
            aucs_s.append(float(auc_rows_stratified(s, lab, (ii - jj).astype(np.int64))[0]))
    return float(np.nanmean(aucs)), aucs, float(np.nanmean(aucs_s)), aucs_s


def boot_ci(vals, n=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    m = r.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --- main ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--n-builds", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-pairs", type=int, default=40000)
    ap.add_argument("--probe-layers", type=int, nargs="+", default=[14, 28])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines-only", default=None,
                    help="skip the model; add the surface-text baselines to an existing run")
    args = ap.parse_args()

    device = "cuda"
    rng = random.Random(args.seed)
    run = Path(args.baselines_only) if args.baselines_only else new_run_dir("attn_adjacency_probe")
    print(f"[run] {run}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    structures, _ = load_house_structures(max_dim=32)
    captions = json.loads(Path(CAPTIONS_PATH).read_text())
    cv = load_piece_vocab(PIECE_VOCAB)
    id_to_name, _ = build_piece_names(cv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- recover the exact Track D split so "held out" means held out -------
    from train_llm_pieces import build_examples
    random.seed(args.seed)
    examples, _ = build_examples(structures, captions, tokenizer, cv, id_to_name, 2048, None)
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, int(round(len(examples) * 0.10)))
    val_idx = {examples[i]["idx"] for i in order[:n_val]}
    train_idx = {examples[i]["idx"] for i in order[n_val:]}
    print(f"[split] track-D train {len(train_idx)} builds, val {len(val_idx)} builds", flush=True)

    # Held-out pool: the 30 val builds, then builds the finetune never saw at all
    # (they blew the 2048-token cap) -- truncated to a whole-line prefix here.
    never = [i for i in range(len(structures)) if i not in train_idx and i not in val_idx]
    rng.shuffle(never)
    pool = sorted(val_idx) + never
    print(f"[pool] {len(val_idx)} val + {len(never)} never-trained candidates", flush=True)

    builds = []
    for idx in pool:
        if len(builds) >= args.n_builds:
            break
        inst = piece_instances(structures[idx], cv, id_to_name)
        if inst is None or len(inst) < 20:
            continue
        lines = [f"{d['name']} {d['anchor'][0]} {d['anchor'][1]} {d['anchor'][2]}" for d in inst]
        prompt = f"Build: {captions[f'h{idx:05d}'][0]}\n"
        ids, spans, keep = line_spans(tokenizer, prompt, lines, args.max_tokens)
        if keep < 20:
            continue
        adj = adjacency_matrix(inst[:keep])
        ii, jj, lab = candidate_pairs(keep, adj, args.max_pairs, rng)
        if lab.sum() == 0 or (~lab).sum() == 0:
            continue
        builds.append({"idx": idx, "ids": ids, "spans": spans, "L": keep, "adj": adj,
                       "anchors": np.array([d["anchor"] for d in inst[:keep]], dtype=np.int64),
                       "pairs": (ii, jj, lab), "held_out": "val" if idx in val_idx else "never",
                       "n_lines_full": len(lines)})
    print(f"[builds] {len(builds)} usable  "
          f"(val {sum(b['held_out'] == 'val' for b in builds)}, "
          f"never-trained {sum(b['held_out'] == 'never' for b in builds)}); "
          f"lines median {int(np.median([b['L'] for b in builds]))}, "
          f"positive rate {np.mean([b['pairs'][2].mean() for b in builds]):.4f}", flush=True)

    # --- surface-text baselines --------------------------------------------
    # The serialization literally prints each piece's coordinates, so before asking
    # what a 1.5B model's attention knows, ask what the *characters on the page*
    # already say. `anchor_L1` is the distance between the two anchors the lines
    # spell out; `coord_match` counts how many of x/y/z are the identical token.
    # Anything a copying head can do, these do -- with no model at all.
    surf: Dict[str, List[float]] = {k: [] for k in ("anchor_L1", "coord_match")}
    surf_s: Dict[str, List[float]] = {k: [] for k in ("anchor_L1", "coord_match")}
    for b in builds:
        ii, jj, lab = b["pairs"]
        a = b["anchors"]
        d = np.abs(a[ii] - a[jj])
        sc = {"anchor_L1": -d.sum(1).astype(np.float64),
              "coord_match": (d == 0).sum(1).astype(np.float64)}
        dist = (ii - jj).astype(np.int64)
        for k, v in sc.items():
            surf[k].append(float(auc_rows(v[None, :], lab)[0]))
            surf_s[k].append(float(auc_rows_stratified(v[None, :], lab, dist)[0]))
    if args.baselines_only:
        S = json.loads((run / "summary.json").read_text())
        perm0 = list(range(len(builds)))
        random.Random(args.seed + 1).shuffle(perm0)
        te0 = perm0[len(builds) // 2:]
        S["surface_baselines"] = {
            k: {"auc_on_test": float(np.nanmean([surf[k][i] for i in te0])),
                "auc_ci": boot_ci([surf[k][i] for i in te0]),
                "strat_auc_on_test": float(np.nanmean([surf_s[k][i] for i in te0])),
                "strat_auc_ci": boot_ci([surf_s[k][i] for i in te0])}
            for k in surf}
        (run / "summary.json").write_text(json.dumps(S, indent=1))
        np.savez_compressed(run / "surface_baselines.npz",
                            **{k: np.array(v) for k, v in surf.items()},
                            **{f"{k}_strat": np.array(v) for k, v in surf_s.items()})
        print(json.dumps(S["surface_baselines"], indent=1), flush=True)
        print(f"[done] baselines added to {run}", flush=True)
        return

    print(f"[model] {MODEL_NAME} (eager attention)", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[model] {n_layers} layers x {n_heads} heads = {n_layers * n_heads} candidate maps",
          flush=True)

    results = {arm: {"head_auc": [], "head_auc_strat": [], "hidden": {}}
               for arm in ("base", "finetuned")}
    recency_auc: List[float] = []
    recency_auc_strat: List[float] = []
    example_pack = None

    for arm in ("base", "finetuned"):
        print(f"[probe] arm={arm}", flush=True)
        for bn, b in enumerate(builds):
            ids = torch.tensor([b["ids"]], device=device)
            with torch.no_grad():
                if arm == "base":
                    with model.disable_adapter():
                        out = model(ids, output_attentions=True, output_hidden_states=True)
                else:
                    out = model(ids, output_attentions=True, output_hidden_states=True)
                mats = []
                for a in out.attentions:
                    mats.append(pool_attention(a[0], b["spans"]))
                A = torch.cat(mats, dim=0)                       # [n_layers*n_heads, L, L]
                A = A / A.tril(-1).sum(-1, keepdim=True).clamp(min=1e-9)
                ii, jj, lab = b["pairs"]
                scores = A[:, ii, jj].float().cpu().numpy()
                hid = {}
                for li in args.probe_layers:
                    h = out.hidden_states[li][0]
                    per_line = torch.stack([h[a:c].mean(0) for a, c in b["spans"]])
                    hid[f"L{li}"] = per_line.float().cpu().numpy()
            del out
            torch.cuda.empty_cache()

            dist = (ii - jj).astype(np.int64)
            results[arm]["head_auc"].append(auc_rows(scores, lab))
            results[arm]["head_auc_strat"].append(auc_rows_stratified(scores, lab, dist))
            for k, v in hid.items():
                b[f"{arm}:{k}"] = v
            if arm == "finetuned" and bn == 0:
                example_pack = {"adj": b["adj"], "A": A.float().cpu().numpy(), "L": b["L"],
                                "idx": b["idx"]}
            if arm == "base":
                rec = (-(ii - jj)).astype(np.float64)[None, :]
                recency_auc.append(float(auc_rows(rec, lab)[0]))
                recency_auc_strat.append(float(auc_rows_stratified(rec, lab, dist)[0]))
            if (bn + 1) % 10 == 0:
                print(f"  [{arm}] {bn + 1}/{len(builds)} builds", flush=True)

        results[arm]["head_auc"] = np.stack(results[arm]["head_auc"])            # [B, H]
        results[arm]["head_auc_strat"] = np.stack(results[arm]["head_auc_strat"])  # [B, H]

    # --- honest head selection: pick on train half, report on test half -----
    B = len(builds)
    perm = list(range(B))
    random.Random(args.seed + 1).shuffle(perm)
    tr, te = perm[: B // 2], perm[B // 2:]
    summary: Dict[str, dict] = {}
    for arm in ("base", "finetuned"):
        M = results[arm]["head_auc"]
        Ms = results[arm]["head_auc_strat"]
        best = int(np.nanmean(M[tr], axis=0).argmax())
        sel = M[te, best]
        oracle_head = int(np.nanmean(M[te], axis=0).argmax())
        best_s = int(np.nanmean(Ms[tr], axis=0).argmax())
        summary[arm] = {
            "best_head_on_train": {"layer": best // n_heads, "head": best % n_heads},
            "auc_selected_on_test": float(np.nanmean(sel)),
            "auc_selected_ci": boot_ci(sel),
            "auc_oracle_on_test": float(np.nanmean(M[te, oracle_head])),
            "oracle_head": {"layer": oracle_head // n_heads, "head": oracle_head % n_heads},
            "auc_mean_over_all_heads": float(np.nanmean(M[te])),
            # distance-stratified: what the head knows beyond raster position
            "strat_best_head_on_train": {"layer": best_s // n_heads, "head": best_s % n_heads},
            "strat_auc_selected_on_test": float(np.nanmean(Ms[te, best_s])),
            "strat_auc_selected_ci": boot_ci(Ms[te, best_s]),
            "strat_auc_oracle_on_test": float(np.nanmax(np.nanmean(Ms[te], axis=0))),
        }

    summary["surface_baselines"] = {
        k: {"auc_on_test": float(np.nanmean([surf[k][i] for i in te])),
            "auc_ci": boot_ci([surf[k][i] for i in te]),
            "strat_auc_on_test": float(np.nanmean([surf_s[k][i] for i in te])),
            "strat_auc_ci": boot_ci([surf_s[k][i] for i in te])}
        for k in surf}

    summary["recency"] = {
        "auc_on_test": float(np.nanmean([recency_auc[i] for i in te])),
        "auc_ci": boot_ci([recency_auc[i] for i in te]),
        "strat_auc_on_test": float(np.nanmean([recency_auc_strat[i] for i in te])),
        "strat_auc_ci": boot_ci([recency_auc_strat[i] for i in te]),
    }

    tr_b = [builds[i] for i in tr]
    te_b = [builds[i] for i in te]
    for arm in ("base", "finetuned"):
        for li in args.probe_layers:
            key = f"{arm}:L{li}"
            auc, per, auc_s, per_s = train_probe(tr_b, te_b, key, device, seed=args.seed)
            summary.setdefault("hidden_probe", {})[key] = {
                "auc_on_test": auc, "auc_ci": boot_ci(per),
                "strat_auc_on_test": auc_s, "strat_auc_ci": boot_ci(per_s)}
            print(f"[hidden probe] {key}: AUC {auc:.4f}  stratified {auc_s:.4f}", flush=True)

    summary["setup"] = {
        "model": MODEL_NAME, "adapter": os.path.abspath(args.adapter),
        "n_builds": B, "n_train_builds": len(tr), "n_test_builds": len(te),
        "n_layers": n_layers, "n_heads": n_heads,
        "median_lines": int(np.median([b["L"] for b in builds])),
        "positive_rate": float(np.mean([b["pairs"][2].mean() for b in builds])),
        "held_out_val": sum(b["held_out"] == "val" for b in builds),
        "held_out_never_trained": sum(b["held_out"] == "never" for b in builds),
        "max_tokens": args.max_tokens, "seed": args.seed,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(run / "head_auc.npz",
                        base=results["base"]["head_auc"],
                        finetuned=results["finetuned"]["head_auc"],
                        base_strat=results["base"]["head_auc_strat"],
                        finetuned_strat=results["finetuned"]["head_auc_strat"],
                        recency=np.array(recency_auc),
                        recency_strat=np.array(recency_auc_strat),
                        test_builds=np.array(te), train_builds=np.array(tr))
    np.savez_compressed(run / "example_build.npz", **{k: v for k, v in example_pack.items()})
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[done] {run}", flush=True)


if __name__ == "__main__":
    main()
