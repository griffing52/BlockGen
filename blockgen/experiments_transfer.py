"""Pool-pretrain → labeled finetune: the cross-medium transfer evidence (roadmap Phase 0).

Pretrains the phase4 AR on the FULL cross-corpus Minecraft pool (labeled crawl +
legacy raw + GrabCraft + 3D-Craft + remapped text2mc — everything, not just
houses), then finetunes on the curated ``houses_32`` set, against a from-scratch
control at the same finetune budget. Honest protocol: the houses VAL split is
excluded from the pool by source_path, all corpora share the legacy vocab, and
every arm is evaluated with both plain and adjacency-constrained sampling.

Arms (each resumable via <run>/novelty.json):
  scratch          — houses-only training, E_ft epochs (control)
  pretrain         — pool pretraining, E_pt epochs (+ zero-shot eval on houses)
  finetune         — pretrain checkpoint + E_ft epochs on houses (lr x0.3)

    .venv/bin/python -m blockgen.experiments_transfer --stamp $(date +%Y%m%d_%H%M%S)
    .venv/bin/python -m blockgen.experiments_transfer --stamp smoke --quick
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blockgen.curation.curate import Curator  # noqa: E402
from blockgen.curation.houses import load_house_structures  # noqa: E402
from blockgen.experiments_gen import ar_feasible, canonicalize  # noqa: E402
from blockgen.experiments_overnight import OUTPUTS, _eval_row, _log  # noqa: E402
from blockgen.training.constrained_decode import sample_constrained_structures  # noqa: E402
from blockgen.training.train_ar import ARTrainConfig, build_sequences  # noqa: E402
from blockgen.training.train_ar import sample_structures as ar_sample  # noqa: E402
from blockgen.training.train_ar_ext import train_from_sequences  # noqa: E402
from blockgen.utils.augment import augment_with_labels  # noqa: E402
from blockgen.utils.data import Structure  # noqa: E402
from blockgen.utils.serialize import build_block_vocab  # noqa: E402

DS = "transfer-houses32"


# --------------------------------------------------------------------------- #
def load_pool(max_dim: int = 48, min_blocks: int = 60,
              exclude_sources: Optional[set] = None,
              text2mc_limit: Optional[int] = None) -> List[Structure]:
    """Everything we have, on the shared legacy vocab, minus excluded sources."""
    from blockgen.data.build_cache import load_cached_structures
    from blockgen.data.grabcraft_dataset import load_grabcraft_structures
    from blockgen.data.tfrecord_dataset import load_labeled_structures
    from blockgen.utils import corpora
    from blockgen.utils.block_remap import build_token_lut, remap_token_array
    from blockgen.curation.houses import _strip_ground

    exclude_sources = exclude_sources or set()
    pool: List[Structure] = []

    try:
        labeled, _ = load_labeled_structures(max_dim=24)
        pool += labeled
        _log(DS, f"pool += labeled crawl: {len(labeled)}")
    except Exception as e:  # noqa: BLE001
        _log(DS, f"labeled crawl unavailable ({e}); skipping")
    try:
        legacy = load_cached_structures(max_dim=24)
        pool += legacy
        _log(DS, f"pool += legacy raw: {len(legacy)}")
    except Exception as e:  # noqa: BLE001
        _log(DS, f"legacy raw unavailable ({e}); skipping")
    gc, _ = load_grabcraft_structures(max_dim=48)
    pool += gc
    _log(DS, f"pool += grabcraft: {len(gc)}")
    dc = corpora.load_3dcraft(min_blocks=min_blocks, max_dim=max_dim, verbose=False)
    pool += dc
    _log(DS, f"pool += 3dcraft: {len(dc)}")

    # text2mc: generic (NOT house-filtered), remapped to legacy ids, ground-stripped.
    import h5py
    index = json.loads((Path(__file__).resolve().parents[1] / "data" / "minecraft"
                        / "text2mc_index.json").read_text())
    files = {p.name: p for p in (Path(__file__).resolve().parents[1] / "data"
                                 / "minecraft" / "text2mc" / "processed_builds").rglob("*.h5")}
    lut = build_token_lut()
    n_t2m = 0
    for fname, info in index.items():
        if info["occ"] < min_blocks or max(info["shape"]) > max_dim + 24:
            continue
        path = files.get(fname)
        if path is None:
            continue
        with h5py.File(path, "r") as h:
            arr = np.asarray(h[list(h.keys())[0]])
        ids, data = remap_token_array(arr, lut)
        s = Structure(block_ids=ids, block_data=data, source_path=str(path),
                      metadata={"corpus": "text2mc"})
        s, _cut = _strip_ground(s)
        if max(s.shape) > max_dim or int(s.occupied_mask.sum()) < min_blocks:
            continue
        pool.append(s)
        n_t2m += 1
        if text2mc_limit and n_t2m >= text2mc_limit:
            break
    _log(DS, f"pool += text2mc (remapped, ground-stripped): {n_t2m}")

    before = len(pool)
    pool = [s for s in pool if (s.source_path or "") not in exclude_sources]
    _log(DS, f"pool: {before} -> {len(pool)} after excluding houses-val sources")
    return pool


def _split_houses(seed: int, val_frac: float) -> Tuple[List[Structure], List[Structure]]:
    houses, _ = load_house_structures(max_dim=32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(houses))
    n_val = max(8, int(round(val_frac * len(houses))))
    val = [houses[i] for i in perm[:n_val]]
    train = [houses[i] for i in perm[n_val:]]
    return train, val


def _make_model(vocab_size: int, cfg: ARTrainConfig, state_path: Optional[Path] = None):
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    m = VoxelTransformerAR2(pe="phase4", vocab_size=vocab_size,
                            max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
                            nhead=cfg.nhead, num_layers=cfg.num_layers,
                            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
    if state_path is not None:
        m.load_state_dict(torch.load(state_path, map_location="cpu"))
    return m.to("cuda" if torch.cuda.is_available() else "cpu")


def _train(model, seqs, vocab_size, cfg: ARTrainConfig):
    """train_from_sequences builds its own model; mirror its loop for a given model."""
    from torch.utils.data import DataLoader
    from blockgen.training.train_ar import _TokenDataset, _collate  # type: ignore[attr-defined]
    device = next(model.parameters()).device
    loader = DataLoader(_TokenDataset(seqs), batch_size=cfg.batch_size,
                        shuffle=True, collate_fn=_collate,
                        num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory and str(device) != "cpu",
                        persistent_workers=cfg.num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    hist = {"loss": []}
    use_amp = cfg.amp and str(device) != "cpu" and torch.cuda.is_bf16_supported()
    model.train()
    for ep in range(cfg.epochs):
        # On-device accumulation: float(loss)/int(...) per step syncs the GPU.
        tot = torch.zeros((), device=device, dtype=torch.float32)
        n = torch.zeros((), device=device, dtype=torch.float32)
        for input_ids, targets, pad_mask in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids, pad_mask=pad_mask)
                loss = crit(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            k = (targets != 0).sum()
            tot += loss.detach().float() * k; n += k
        hist["loss"].append(float(tot / n.clamp(min=1)))
        if (ep + 1) % 5 == 0 or ep == 0:
            _log(DS, f"  epoch {ep + 1}/{cfg.epochs} loss {hist['loss'][-1]:.4f}")
    return hist


def _eval_both(tag: str, model, vocab, ref, val, dim: int, run: Path,
               n_samples: int) -> List[dict]:
    rows = []
    s = ar_sample(model, vocab, num_samples=n_samples, temperature=1.0, top_k=40)
    rows.append(_eval_row(tag, s, ref, val, vocab, dim, run))
    run_c = run.parent / f"{run.name}_constrained"
    run_c.mkdir(parents=True, exist_ok=True)
    sc = sample_constrained_structures(model, vocab, num_samples=n_samples,
                                       temperature=1.0, top_k=40)
    rows.append(_eval_row(f"{tag}_constrained", sc, ref, val, vocab, dim, run_c,
                          gated=False))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--canon-dim", type=int, default=16)
    ap.add_argument("--ar-seq", type=int, default=1600)
    ap.add_argument("--ar-batch", type=int, default=8)
    ap.add_argument("--epochs-pretrain", type=int, default=30)
    ap.add_argument("--epochs-finetune", type=int, default=60)
    ap.add_argument("--finetune-lr-scale", type=float, default=0.3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs_pretrain = args.epochs_finetune = 2
        args.samples = 4
    torch.manual_seed(args.seed)

    out = OUTPUTS / f"run_{args.stamp}_transfer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmd.txt").write_text(json.dumps(vars(args), indent=2))
    t0 = time.time()
    dim = args.canon_dim

    # --- houses split + canon ------------------------------------------------
    h_train, h_val = _split_houses(args.seed, args.val_frac)
    _log(DS, f"houses_32 split: {len(h_train)} train / {len(h_val)} val")
    ref_c = canonicalize(h_train, dim)
    val_c = canonicalize(h_val, dim)
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))
    fit_c = canonicalize(h_aug, dim)

    # --- pool (excluding val sources), canon, dedup, augment -----------------
    pool = load_pool(exclude_sources={s.source_path or "" for s in h_val},
                     text2mc_limit=None if not args.quick else 50)
    cur = Curator.from_structures(pool, max_dim=dim)
    cur.decisions.clear()
    n_dup = cur.dedupe_keep_variants(iou_threshold=0.95)
    pool = cur.apply()
    _log(DS, f"pool deduped -{n_dup} -> {len(pool)}")
    p_aug, _ = augment_with_labels(pool, ["p"] * len(pool))
    pool_c = canonicalize(p_aug, dim)

    # --- shared vocab ---------------------------------------------------------
    vocab = build_block_vocab(pool_c + ref_c + val_c, max_dim=dim)
    _log(DS, f"shared vocab: {vocab.vocab_size}")
    cfg_ft = ARTrainConfig(max_seq_len=args.ar_seq, epochs=args.epochs_finetune,
                           batch_size=args.ar_batch, device="cuda")
    house_seqs = build_sequences(ar_feasible(fit_c, vocab, args.ar_seq), vocab, args.ar_seq)
    _log(DS, f"houses fit sequences: {len(house_seqs)}")

    rows: List[dict] = []

    def done(tag: str) -> Optional[dict]:
        f = out / tag / "novelty.json"
        return json.loads(f.read_text()) if f.exists() else None

    # --- arm 1: scratch control ----------------------------------------------
    tag = "scratch"
    if done(tag) and done(f"{tag}_constrained"):
        rows += [done(tag), done(f"{tag}_constrained")]
        _log(DS, "scratch: already done")
    else:
        try:
            run = out / tag; run.mkdir(exist_ok=True)
            model = _make_model(vocab.vocab_size, cfg_ft)
            hist = _train(model, house_seqs, vocab.vocab_size, cfg_ft)
            torch.save(model.state_dict(), run / "model.pt")
            r = _eval_both(tag, model, vocab, ref_c, val_c, dim, run, args.samples)
            for x in r: x["final_loss"] = round(hist["loss"][-1], 4)
            rows += r
            del model; torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            (out / tag / "ERROR.txt").write_text(traceback.format_exc())
            _log(DS, "scratch FAILED")

    # --- arm 2: pool pretraining ----------------------------------------------
    tag = "pretrain"
    run_pt = out / tag; run_pt.mkdir(exist_ok=True)
    pt_ckpt = run_pt / "model.pt"
    if not pt_ckpt.exists():
        try:
            pool_seqs = build_sequences(ar_feasible(pool_c, vocab, args.ar_seq),
                                        vocab, args.ar_seq)
            _log(DS, f"pool fit sequences: {len(pool_seqs)}")
            cfg_pt = ARTrainConfig(max_seq_len=args.ar_seq, epochs=args.epochs_pretrain,
                                   batch_size=args.ar_batch, device="cuda")
            model = _make_model(vocab.vocab_size, cfg_pt)
            hist = _train(model, pool_seqs, vocab.vocab_size, cfg_pt)
            torch.save(model.state_dict(), pt_ckpt)
            (run_pt / "history.json").write_text(json.dumps(hist))
        except Exception:  # noqa: BLE001
            (run_pt / "ERROR.txt").write_text(traceback.format_exc())
            _log(DS, "pretrain FAILED"); model = None
    else:
        _log(DS, "pretrain: checkpoint exists, reloading")
        model = _make_model(vocab.vocab_size, cfg_ft, pt_ckpt)
    if pt_ckpt.exists():
        if model is None:
            model = _make_model(vocab.vocab_size, cfg_ft, pt_ckpt)
        if not done("pretrain_zeroshot"):
            runz = out / "pretrain_zeroshot"; runz.mkdir(exist_ok=True)
            rows += _eval_both("pretrain_zeroshot", model, vocab, ref_c, val_c, dim,
                               runz, args.samples)
        else:
            rows += [done("pretrain_zeroshot"), done("pretrain_zeroshot_constrained")]

        # --- arm 3: finetune ---------------------------------------------------
        tag = "finetune"
        if done(tag) and done(f"{tag}_constrained"):
            rows += [done(tag), done(f"{tag}_constrained")]
            _log(DS, "finetune: already done")
        else:
            try:
                run = out / tag; run.mkdir(exist_ok=True)
                cfg_f = ARTrainConfig(max_seq_len=args.ar_seq, epochs=args.epochs_finetune,
                                      batch_size=args.ar_batch, device="cuda",
                                      lr=cfg_ft.lr * args.finetune_lr_scale)
                hist = _train(model, house_seqs, vocab.vocab_size, cfg_f)
                torch.save(model.state_dict(), run / "model.pt")
                r = _eval_both(tag, model, vocab, ref_c, val_c, dim, run, args.samples)
                for x in r: x["final_loss"] = round(hist["loss"][-1], 4)
                rows += r
            except Exception:  # noqa: BLE001
                (out / tag / "ERROR.txt").write_text(traceback.format_exc())
                _log(DS, "finetune FAILED")

    rows = [r for r in rows if r]
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    cols = ["tag", "final_loss", "mean_nn_iou", "duplicate_rate", "val_nn_iou",
            "diversity", "validity_rate", "validity_gated", "median_sample_occ"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out / "leaderboard.md").write_text("\n".join(md) + "\n")
    _log(DS, f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
