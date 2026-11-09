"""Tokenised shards: a tier, through a tokenizer, as windows the trainer can stream.

Why shards. The corpus is 10.7B blocks across 649k builds; the old trainers
re-serialise every build from npz at the start of each run, which is fine for
2,661 houses and not for this. A shard is built once per (tier, tokenizer) and
memory-mapped by every run that trains on it.

Windows (docs/pipeline.md §3.6). A build whose tight box fits 48³ and has at most
``max_blocks`` blocks is one sequence tagged ``win:whole`` with ``size:32`` or
``size:48``. Anything larger yields ``k = ceil(n_blocks / max_blocks)`` windows
(capped at ``max_windows``), each centred on a random occupied voxel, tagged
``win:window``; a window with more than ``max_blocks`` blocks is retried at edge
32, 24, 16 and dropped if still too dense (the sparse raster arm cannot afford a
dense 32³, that is what the diffusion arm is for). Window choice is seeded by the
build id, so a shard is reproducible.

Layout of ``data/minecraft/corpus/shards/<tier>/<tokenizer hash>/``::

    tokens.u16      every sequence back to back, uint16
    offsets.npy     int64 [n+1]: sequence i is tokens[offsets[i]:offsets[i+1]]
    meta.parquet    seq, id, source, src, q, size, win, ox, oy, oz, n_blocks, n_tokens
    tokenizer.json  the tokenizer artifact
    spec.json       tier name + sha, tokenizer hash, parameters, counts

The ``facades`` kind (``--kind facades [--faces all|sides] [--layout per_face|joint|pooled4]``)
writes one sequence per face of every whole build or window (``per_face``), or
every face of the volume in one sequence (``joint``, each face conditioned on the
ones before it), through the facade tokenizer, into its own hash directory; ``find`` tells the kinds apart by
``spec.json``, so a voxel run never picks up a facade shard by being newest.

The ``pieces`` kind (3D-BPE) has a learned vocabulary, so it is built from a
tokenizer artifact (``--tokenizer <hash|path>``, written by ``python -m
blockgen.corpus.tokenizer learn-pieces``); windows and the ``max_blocks`` rule
are the voxel kind's, only the body is pieces instead of blocks.

Category conditioning (``--cat``) shards a tier through the same tokenizer plus the
``cat`` group (``CorpusTokenizer.default(cats=True)``; with ``--tokenizer`` the named
artifact plus the group), reading each row's class from the labels sidecar
(``blockgen.corpus.labels``; ``none`` when the row has none). ``meta.parquet``
gains a ``cat`` column and ``spec.json`` ``cats: true``; ``find`` without a hash
prefers a shard without the group, so a tier that has both keeps resolving to
the unconditioned one it always did.

Derived tiers. ``from_structures`` writes the same five files for a list of
``Structure`` objects that never passed through the manifest (model samples a
reward kept, for rejection-sampling fine-tuning). Each structure is tokenised
exactly as a whole manifest build would be, with the same tags, and the tier
json under ``tiers/<tier>.json`` carries ``kind: derived`` and the sha256 of the
structures' content, so a checkpoint's ``tiers`` record names the data it saw.
``find`` and ``blockgen.train.data.open_shards`` need no change to read one.

Staleness. ``Shard`` compares the ``tier_sha256`` it was built from with the
current ``tiers/<tier>.json`` and warns (``StaleShardWarning``) when the tier
has been rebuilt since; the shard still opens, the run should be rebuilt.

CLI::

    python -m blockgen.corpus.shards build <tier> [--max-blocks 2048] [--workers 6] [--oriented]
                                                  [--kind voxels|facades] [--faces all|sides]
                                                  [--tokenizer <hash|tokenizer.json>]   (pieces)
                                                  [--cat]                               (class prefix)
    python -m blockgen.corpus.shards show <tier> [--hash H]
    python -m blockgen.corpus.shards from-npz <npz> --tier <name> [--kind voxels|facades]
                                                  [--layout per_face|joint|pooled4] [--tokenizer <hash>]
                                                  [--tags k=v ...] [--source NAME] [--max-blocks N]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.augment import _d4_arrays
from blockgen.utils.data import Structure

from . import facade as fc
from .manifest import CORPUS_DIR, ROOT, load
from . import tiers as _tiers
from .tiers import TIER_DIR, _ids_hash
from .tokenizer import TOKENIZER_DIR, CorpusTokenizer

AUGMENTS = ("", "d4")
FACE_SETS = {"all": fc.ALL_FACES, "sides": fc.SIDES, "sides_top": fc.SIDES_TOP}

SHARD_DIR = os.path.join(CORPUS_DIR, "shards")
KINDS = os.path.join(CORPUS_DIR, "kinds.parquet")
EDGES = (48, 32, 24, 16)
MIN_BLOCKS = 16


# --------------------------------------------------------------------------- loading builds

def _load_build(path: str, cache: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    """(ids, data) XYZ for a manifest path; non-member voxels zeroed."""
    if "#" in path:
        npz, idx = path.split("#")
        if npz not in cache:
            z = np.load(os.path.join(ROOT, npz), allow_pickle=True)
            cache.clear()
            cache[npz] = (z["block_ids"], z["block_data"])
        ids_all, data_all = cache[npz]
        i = int(idx)
        return np.asarray(ids_all[i]), np.asarray(data_all[i])
    z = np.load(os.path.join(ROOT, path))
    ids = z["ids"]; data = z["data"] if "data" in z else np.zeros_like(ids, np.uint8)
    if "mask" in z:
        m = z["mask"].astype(bool)
        ids = ids * m; data = data * m
    return ids, data


def _tight(ids: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    occ = ids != 0
    if not occ.any():
        return ids[:0, :0, :0], data[:0, :0, :0]
    xs, ys, zs = np.nonzero(occ)
    sl = np.s_[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]
    return ids[sl], data[sl]


def _src_tag(source: str, kind: str) -> str:
    if source == "templates":
        return "templates"
    if source == "worldcrop":
        return "vanilla" if kind in ("village", "mineshaft") else "archive"
    return "curated"


def _q_tag(p_built) -> str:
    if p_built is None or (isinstance(p_built, float) and math.isnan(p_built)):
        return "4"
    return str(min(4, max(0, int(float(p_built) * 5))))


def _windows(ids: np.ndarray, data: np.ndarray, max_blocks: int, max_windows: int, rng: np.random.Generator):
    """Yield (ox, oy, oz, edge, ids_w, data_w) windows of a build too big to be whole."""
    occ = np.argwhere(ids != 0)
    n = len(occ)
    k = min(max_windows, max(1, math.ceil(n / max_blocks)))
    shape = np.array(ids.shape)
    for c in occ[rng.choice(n, size=k, replace=n < k)]:
        for edge in EDGES:
            o = np.clip(c - edge // 2 + rng.integers(-edge // 4, edge // 4 + 1, size=3), 0, np.maximum(shape - edge, 0))
            sl = tuple(slice(int(o[a]), int(o[a]) + edge) for a in range(3))
            w = ids[sl]
            nb = int(np.count_nonzero(w))
            if nb <= max_blocks:
                if nb >= MIN_BLOCKS:
                    yield int(o[0]), int(o[1]), int(o[2]), edge, w, data[sl]
                break


def _encode_one(tok, lut, ids, data, tags, faces, augment=""):
    """``[(sequence, extra meta)]`` for one volume: one entry for voxels, one per face for facades.

    ``augment="d4"`` encodes the eight horizontal symmetries of the volume (four
    rotations about y, each with and without a mirror; ``blockgen.utils.augment``
    remaps stair and log facings), duplicates of a symmetric build collapsed, and
    tags each entry ``aug`` 0..7 with 0 the identity. Houses in this corpus have
    no preferred facing, so the orbit is eight free samples of the same
    distribution; the holdout tiers are never augmented.
    """
    if augment == "d4":
        out, seen = [], set()
        for g, (bi, bd) in enumerate(_d4_arrays(ids, data)):
            key = bi.shape + (bi.tobytes(), bd.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.extend((seq, {**extra, "aug": g}) for seq, extra in _encode_one(tok, lut, bi, bd, tags, faces))
        return out
    if augment:
        raise ValueError(f"unknown augment {augment!r}; one of {AUGMENTS}")
    shared = {k: tags[k] for k in ("src", "q", "size", "win") + (("cat",) if "cat" in tags else ())}
    if tok.kind != "facades":
        return [(tok.encode(ids, data, shared, lut), {})]
    out = []
    s = Structure(block_ids=ids, block_data=data)
    proj = fc.project_all(s, faces)
    if tok.layout == "joint":
        seq = tok.encode_faces(list(proj.values()), shared, lut)
        X, Y, Z = ids.shape                      # the box, in the columns the sampler reads
        return [(seq, {"face": "joint", "u": int(X), "v": int(Y), "extent": int(Z),
                       "n_cells": int(sum(f.mask.sum() for f in proj.values()))})]
    for name, f in proj.items():
        seq = tok.encode_face(f, shared, lut)
        out.append((seq, {"face": name, "u": int(f.depth.shape[0]), "v": int(f.depth.shape[1]),
                          "extent": int(f.extent), "n_cells": int(f.mask.sum())}))
    return out


def _encode_rows(args):
    rows, tok_json, max_blocks, max_windows, faces, augment = args
    tok = CorpusTokenizer.load(tok_json)
    lut = tok.block_lut()
    cache: Dict[str, object] = {}
    seqs: List[np.ndarray] = []
    meta: List[dict] = []
    for r in rows:
        try:
            ids, data = _load_build(r["path"], cache)
        except Exception as e:  # unreadable file: skip, count
            meta.append({"id": r["id"], "error": str(e)[:80]})
            continue
        ids, data = _tight(ids, data)
        if ids.size == 0:
            continue
        n = int(np.count_nonzero(ids))
        base = {"id": r["id"], "source": r["source"], "src": _src_tag(r["source"], r.get("kind", "build")),
                "q": _q_tag(r.get("p_built"))}
        if tok.cats:
            base["cat"] = r.get("cat") or "none"
        if max(ids.shape) <= 48 and n <= max_blocks:
            if n < MIN_BLOCKS:
                continue
            size = "32" if max(ids.shape) <= 32 else "48"
            tags = {**base, "size": size, "win": "whole"}
            for s, extra in _encode_one(tok, lut, ids, data, tags, faces, augment):
                seqs.append(s)
                meta.append({**tags, "ox": 0, "oy": 0, "oz": 0, "n_blocks": n, "n_tokens": len(s), **extra})
            continue
        rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(r["id"].encode(), digest_size=4).digest(), "big"))
        for ox, oy, oz, edge, w_ids, w_data in _windows(ids, data, max_blocks, max_windows, rng):
            tags = {**base, "size": str(edge), "win": "window"}
            for s, extra in _encode_one(tok, lut, w_ids, w_data, tags, faces, augment):
                seqs.append(s)
                meta.append({**tags, "ox": ox, "oy": oy, "oz": oz, "n_blocks": int(np.count_nonzero(w_ids)),
                             "n_tokens": len(s), **extra})
    return seqs, meta


# --------------------------------------------------------------------------- build

def _tier_rows(tier: str):
    """The tier's manifest rows (its auxiliary manifests included) with ``kind`` and, when the
    labels sidecar exists, ``cat``."""
    import pandas as pd
    with open(os.path.join(TIER_DIR, f"{tier}.json")) as fh:
        spec = json.load(fh)
    with open(os.path.join(TIER_DIR, f"{tier}.ids.txt")) as fh:
        ids = set(fh.read().split())
    df = load(columns=["id", "source", "path", "n_blocks", "p_built"], aux=spec.get("aux"))
    df = df[df.id.isin(ids)]
    if os.path.exists(_tiers.LABELS):
        lab = pd.read_parquet(_tiers.LABELS, columns=["id", "cat"])
        df = df.merge(lab, on="id", how="left")
        df["cat"] = df["cat"].fillna("none")
    else:
        df["cat"] = "none"
    if os.path.exists(KINDS):
        k = pd.read_parquet(KINDS, columns=["id", "kind"])
        df = df.merge(k, on="id", how="left")
        df["kind"] = df["kind"].fillna("build")
    else:
        df["kind"] = "build"
    # cache rows share one 50 MB npz: keep them adjacent so a worker loads it once
    df = df.sort_values(["path"]).reset_index(drop=True)
    return spec, df


def build(tier: str, max_blocks: int = 2048, max_windows: int = 32, workers: int = 6,
          oriented: bool = False, chunk: int = 500, limit: int = 0,
          kind: str = "voxels", faces: str = "all", layout: str = "per_face", augment: str = "",
          tokenizer: Optional[str] = None, cats: bool = False) -> str:
    """Shard ``tier`` through the default tokenizer of ``kind``, or the artifact ``tokenizer`` names
    (a hash or a path; its kind wins). The pieces kind needs one: its merges are learned.
    ``cats`` adds the category group to whichever tokenizer that is."""
    import pandas as pd
    t0 = time.time()
    if tokenizer is not None:
        tok = resolve_tokenizer(tokenizer, kind, layout, oriented)
        kind, layout, oriented = tok.kind, tok.layout, tok.vocab.oriented
    elif kind == "pieces":
        raise ValueError("the pieces kind needs --tokenizer <hash|path>; learn one with "
                         "python -m blockgen.corpus.tokenizer learn-pieces")
    else:
        tok = CorpusTokenizer.default(oriented=oriented, kind=kind, layout=layout)
    if cats and not tok.cats:
        tok = tok.with_cats()
    face_set = FACE_SETS[faces] if kind == "facades" else ()
    out = os.path.join(SHARD_DIR, tier, tok.hash + (f"-{augment}" if augment else ""))
    os.makedirs(out, exist_ok=True)
    tok_json = tok.save(os.path.join(out, "tokenizer.json"))
    spec, df = _tier_rows(tier)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=0).sort_values("path")
    rows = df.to_dict("records")
    jobs = [(rows[i:i + chunk], tok_json, max_blocks, max_windows, face_set, augment)
            for i in range(0, len(rows), chunk)]
    print(f"{tier}: {len(rows)} builds, {len(jobs)} chunks, tokenizer {tok.hash}, vocab {tok.vocab_size}", flush=True)
    tokens_path = os.path.join(out, "tokens.u16")
    offsets = [0]
    metas: List[dict] = []
    errors = 0
    with open(tokens_path, "wb") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        for j, (seqs, meta) in enumerate(ex.map(_encode_rows, jobs)):
            for s in seqs:
                f.write(s.tobytes()); offsets.append(offsets[-1] + len(s))
            for m in meta:
                if "error" in m:
                    errors += 1
                else:
                    metas.append(m)
            if (j + 1) % 20 == 0 or j + 1 == len(jobs):
                print(f"  [{j + 1}/{len(jobs)}] seqs={len(metas)} tokens={offsets[-1] / 1e6:.1f}M "
                      f"errors={errors} {time.time() - t0:.0f}s", flush=True)
    header = {"tier": tier, "tier_sha256": spec["sha256"], "tokenizer": tok.hash, "kind": kind,
              "faces": list(face_set), "layout": tok.layout, "augment": augment, "max_blocks": max_blocks,
              "max_windows": max_windows, "oriented": oriented, "limit": limit}
    if tok.cats:
        header["cats"] = True
    counts = _finish(out, offsets, metas, errors, len(rows), header, t0)
    print(f"{tier}: {counts['n_seqs']} sequences, {counts['n_tokens'] / 1e6:.1f}M tokens, "
          f"{counts['n_blocks'] / 1e6:.1f}M blocks, {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def _finish(out: str, offsets: List[int], metas: List[dict], errors: int, n_builds: int,
            header: dict, t0: float) -> dict:
    """Write ``offsets.npy``, ``meta.parquet`` and ``spec.json`` once ``tokens.u16`` is on disk.

    Shared by ``build`` and ``from_structures`` so a derived shard has the same
    files, columns and spec counts as a manifest one.
    """
    import pandas as pd
    np.save(os.path.join(out, "offsets.npy"), np.asarray(offsets, np.int64))
    meta_df = pd.DataFrame(metas)
    meta_df.insert(0, "seq", np.arange(len(meta_df)))
    meta_df.to_parquet(os.path.join(out, "meta.parquet"), index=False)
    counts = {
        "n_builds": int(n_builds), "n_seqs": int(len(metas)), "n_tokens": int(offsets[-1]), "errors": errors,
        "n_blocks": int(meta_df.n_blocks.sum()) if len(meta_df) else 0,
        "by_src": meta_df.groupby("src").size().to_dict() if len(meta_df) else {},
        "by_win": meta_df.groupby(["win", "size"]).size().to_dict() if len(meta_df) else {},
        **({"by_cat": meta_df.groupby("cat").size().to_dict()} if len(meta_df) and "cat" in meta_df else {}),
        "seq_len": {q: int(meta_df.n_tokens.quantile(q)) for q in (0.5, 0.9, 0.99)} if len(meta_df) else {},
    }
    counts["by_win"] = {f"{k[0]}:{k[1]}": int(v) for k, v in counts["by_win"].items()}
    with open(os.path.join(out, "spec.json"), "w") as fh:
        json.dump({**header, "built": time.strftime("%Y-%m-%d %H:%M"), "seconds": int(time.time() - t0), **counts},
                  fh, indent=1)
    return counts


# --------------------------------------------------------------------------- derived tiers

DEFAULT_TAGS = {"src": "curated", "q": "4", "win": "whole"}
SKIP_REASONS = ("empty", "small", "large", "dense")


def _content_hash(ids: np.ndarray, data: np.ndarray) -> str:
    """sha256 of a tight volume's shape and blocks, dtype-independent: the id of a derived build."""
    h = hashlib.sha256()
    h.update(np.asarray(ids.shape, np.int64).tobytes())
    h.update(np.clip(ids, 0, 65535).astype(np.uint16).tobytes())
    h.update(np.clip(data, 0, 255).astype(np.uint8).tobytes())
    return h.hexdigest()


def _check_derived_name(tier: str) -> None:
    """A derived tier may not shadow a configured one: that would silently retarget every run on it."""
    if not tier or tier != os.path.basename(tier) or tier.startswith("."):
        raise ValueError(f"tier name must be a plain directory name, got {tier!r}")
    if os.path.exists(os.path.join(_tiers.CONFIG_DIR, f"{tier}.yaml")):
        raise ValueError(f"{tier} is a configured tier (configs/corpus/{tier}.yaml); a derived tier needs its own name")
    p = os.path.join(TIER_DIR, f"{tier}.json")
    if os.path.exists(p):
        with open(p) as fh:
            kind = json.load(fh).get("kind")
        if kind != "derived":
            raise ValueError(f"{p} exists and is not a derived tier; refusing to overwrite it")


def _check_tags(tok: CorpusTokenizer, tags: Dict[str, str]) -> None:
    for g, v in tags.items():
        if g == "cat" and tok.cats:
            if str(v) not in tok.cats:
                raise ValueError(f"tag cat={v!r} is not one of {tok.cats}")
            continue
        if g not in tok.tags:
            raise ValueError(f"unknown tag group {g!r}; the tokenizer has {list(tok.tags)}")
        if str(v) not in tok.tags[g]:
            raise ValueError(f"tag {g}={v!r} is not one of {tok.tags[g]}")


def from_structures(structures: Sequence[Structure], tier: str, tok: CorpusTokenizer, *,
                    tags: Optional[Dict[str, str]] = None, source: Optional[str] = None,
                    max_blocks: Optional[int] = None, faces: str = "all", augment: str = "") -> str:
    """Write a derived tier and its shard from structures that never passed through the manifest.

    Every structure is cropped to its occupied box and encoded by ``_encode_one``
    with the tags a whole manifest build of the same size would carry
    (``src:curated q:4 win:whole`` and ``size`` from the largest edge); ``tags``
    overrides any of them for all structures. A structure is skipped and counted
    in ``spec.skipped`` when it is empty, has fewer than ``MIN_BLOCKS`` blocks,
    does not fit the tokenizer's ``max_dim`` edge, or has more than ``max_blocks``
    blocks when that is given. Nothing is windowed: a derived tier holds whole
    samples or nothing. When every structure is skipped the shard directory is
    removed, no tier json is written and ``ValueError`` names the skip counts,
    since an empty ``tokens.u16`` cannot be memory-mapped and a tier of nothing
    must not be trainable.

    The shard lands in ``SHARD_DIR/<tier>/<tok.hash>[-<augment>]/`` with the five
    files ``build`` writes; ``TIER_DIR/<tier>.json`` records ``kind: derived``,
    ``source`` and ``sha256`` (the sorted content hashes of the kept structures),
    which is also ``spec.tier_sha256`` and ``spec.source_sha256``. Returns the
    shard directory.
    """
    t0 = time.time()
    _check_derived_name(tier)
    if tok.kind == "facades":
        face_set: Tuple[str, ...] = tuple(FACE_SETS[faces])
    else:
        face_set = ()
    if augment and augment not in AUGMENTS:
        raise ValueError(f"unknown augment {augment!r}; one of {AUGMENTS}")
    over = {k: str(v) for k, v in (tags or {}).items()}
    _check_tags(tok, over)
    src = source or "generated"
    lut = tok.block_lut()
    out = os.path.join(SHARD_DIR, tier, tok.hash + (f"-{augment}" if augment else ""))
    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"))
    offsets = [0]
    metas: List[dict] = []
    ids_kept: List[str] = []
    skipped = {k: 0 for k in SKIP_REASONS}
    n_in = 0
    with open(os.path.join(out, "tokens.u16"), "wb") as f:
        for i, s in enumerate(structures):
            n_in += 1
            ids, data = _tight(np.asarray(s.block_ids), np.asarray(s.block_data))
            if ids.size == 0:
                skipped["empty"] += 1; continue
            n = int(np.count_nonzero(ids))
            if n < MIN_BLOCKS:
                skipped["small"] += 1; continue
            if max(ids.shape) > tok.max_dim:
                skipped["large"] += 1; continue
            if max_blocks is not None and n > max_blocks:
                skipped["dense"] += 1; continue
            bid = (s.metadata or {}).get("id") or _content_hash(ids, data)
            size = "32" if max(ids.shape) <= 32 else "48"
            tg = {**DEFAULT_TAGS, "size": size, **({"cat": "none"} if tok.cats else {}), **over}
            for seq, extra in _encode_one(tok, lut, ids, data, tg, face_set, augment):
                f.write(seq.tobytes()); offsets.append(offsets[-1] + len(seq))
                metas.append({"id": bid, "source": src, **{k: tg[k] for k in ("src", "q", "size", "win", "cat") if k in tg},
                              "ox": 0, "oy": 0, "oz": 0, "n_blocks": n, "n_tokens": len(seq), **extra})
            ids_kept.append(bid)
    if not ids_kept:
        shutil.rmtree(out, ignore_errors=True)
        parent = os.path.dirname(out)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
        raise ValueError(f"{tier}: none of {n_in} structures kept: {skipped}")
    sha = _ids_hash(ids_kept)
    header = {"tier": tier, "tier_sha256": sha, "tokenizer": tok.hash, "kind": tok.kind, "faces": list(face_set),
              "layout": tok.layout, "augment": augment, "source": src, "source_sha256": sha, "n_structures": n_in,
              "skipped": skipped, "max_blocks": max_blocks, "max_windows": 0, "oriented": tok.vocab.oriented, "limit": 0}
    if tok.cats:
        header["cats"] = True
    counts = _finish(out, offsets, metas, 0, len(ids_kept), header, t0)
    os.makedirs(TIER_DIR, exist_ok=True)
    rec = {"name": tier, "kind": "derived", "source": src, "sha256": sha, "n": len(ids_kept),
           "description": f"derived from {src}: {len(ids_kept)} of {n_in} structures kept",
           "n_structures": n_in, "skipped": skipped, "n_blocks_total": counts["n_blocks"],
           "built": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(TIER_DIR, f"{tier}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    with open(os.path.join(TIER_DIR, f"{tier}.ids.txt"), "w") as fh:
        fh.write("\n".join(ids_kept))
    print(f"{tier}: {len(ids_kept)}/{n_in} structures, {counts['n_seqs']} sequences, "
          f"{counts['n_tokens'] / 1e6:.2f}M tokens, skipped {skipped}, sha {sha[:12]} -> {out}", flush=True)
    return out


def structures_from_npz(path: str) -> List[Structure]:
    """Structures from a structure-cache npz (``block_ids``/``block_data``, stacked or
    object arrays) or a keyed one (``s0``, ``s1``, ... block-id grids)."""
    z = np.load(path, allow_pickle=True)
    if "block_ids" in z.files:
        ids = z["block_ids"]
        data = z["block_data"] if "block_data" in z.files else [np.zeros_like(np.asarray(b)) for b in ids]
        return [Structure(block_ids=np.asarray(b), block_data=np.asarray(d)) for b, d in zip(ids, data)]
    keys = sorted((k for k in z.files if k.startswith("s") and k[1:].isdigit()), key=lambda k: int(k[1:]))
    if not keys:
        raise ValueError(f"{path}: no block_ids array and no s0, s1, ... grids")
    return [Structure(block_ids=z[k], block_data=np.zeros_like(z[k])) for k in keys]


def resolve_tokenizer(spec: Optional[str], kind: str = "voxels", layout: str = "per_face",
                      oriented: bool = False) -> CorpusTokenizer:
    """The tokenizer a hash names: a ``tokenizer.json`` path, a learned tokenizer under
    ``corpus/tokenizers/<hash>.json``, the hash of an existing shard's artifact, or the
    default for ``kind``/``layout``/``oriented`` when it matches."""
    tok = CorpusTokenizer.default(oriented=oriented, kind=kind, layout=layout)
    if spec is None or spec == tok.hash:
        return tok
    if spec == tok.with_cats().hash:                              # the same default plus the category group
        return tok.with_cats()
    if os.path.isfile(spec):
        return CorpusTokenizer.load(spec)
    learned = os.path.join(TOKENIZER_DIR, f"{spec}.json")         # learn-pieces output
    for p in ([learned] if os.path.isfile(learned) else []) + sorted(glob.glob(os.path.join(SHARD_DIR, "*", spec, "tokenizer.json"))):
        return CorpusTokenizer.load(p)
    raise ValueError(f"no tokenizer with hash {spec}: the default for kind={kind} layout={layout} "
                     f"oriented={oriented} is {tok.hash}, and no shard under {SHARD_DIR} carries it")


# --------------------------------------------------------------------------- reading

class StaleShardWarning(UserWarning):
    """The tier a shard was built from has been rebuilt since; the shard still opens."""


def _check_fresh(path: str, spec: dict) -> None:
    """Warn when ``spec.tier_sha256`` is not the sha the tier json holds now."""
    tier, was = spec.get("tier"), spec.get("tier_sha256")
    p = os.path.join(TIER_DIR, f"{tier}.json")
    if not tier or not was or not os.path.exists(p):
        return
    try:
        with open(p) as fh:
            now = json.load(fh).get("sha256")
    except (OSError, ValueError):
        return
    if now and now != was:
        warnings.warn(f"shard {path} was built from tier {tier} at sha {was[:12]} but {p} is now sha {now[:12]}; "
                      f"rebuild with: python -m blockgen.corpus.shards build {tier}", StaleShardWarning, stacklevel=3)


class Shard:
    """Memory-mapped sequences; ``shard[i]`` is a uint16 array."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "spec.json")) as fh:
            self.spec = json.load(fh)
        _check_fresh(path, self.spec)
        self.tokenizer = CorpusTokenizer.load(os.path.join(path, "tokenizer.json"))
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.tokens = np.memmap(os.path.join(path, "tokens.u16"), dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[self.offsets[i]:self.offsets[i + 1]])

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def meta(self):
        import pandas as pd
        return pd.read_parquet(os.path.join(self.path, "meta.parquet"))


def _matches(spec_path: str, kind: str, augment: str) -> bool:
    with open(spec_path) as fh:
        spec = json.load(fh)
    return spec.get("kind", "voxels") == kind and spec.get("augment", "") == augment


def find(tier: str, tok_hash: Optional[str] = None, kind: str = "voxels", augment: str = "") -> str:
    """The shard directory for a tier: by tokenizer hash, else the newest of ``kind``.
    An augmented shard lives beside the plain one as ``<hash>-<augment>``."""
    base = os.path.join(SHARD_DIR, tier)
    suffix = f"-{augment}" if augment else ""
    if tok_hash:
        return os.path.join(base, tok_hash + suffix)
    cands = sorted(glob.glob(os.path.join(base, "*", "spec.json")), key=os.path.getmtime)
    cands = [c for c in cands if _matches(c, kind, augment)]
    plain = [c for c in cands if not json.load(open(c)).get("cats")]
    cands = plain or cands                     # an unconditioned shard wins when the tier has both
    if not cands:
        raise FileNotFoundError(f"no {kind} shard for tier {tier}; run python -m blockgen.corpus.shards "
                                f"build {tier}" + (f" --kind {kind}" if kind != "voxels" else ""))
    return os.path.dirname(cands[-1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("tier")
    b.add_argument("--max-blocks", type=int, default=2048); b.add_argument("--max-windows", type=int, default=32)
    b.add_argument("--workers", type=int, default=6); b.add_argument("--oriented", action="store_true")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--kind", choices=("voxels", "facades", "pieces"), default="voxels")
    b.add_argument("--tokenizer", default=None, metavar="HASH",
                   help="tokenizer hash or tokenizer.json path (required for pieces); its kind overrides --kind")
    b.add_argument("--faces", choices=sorted(FACE_SETS), default="all", help="facades kind only")
    b.add_argument("--layout", choices=("per_face", "joint", "pooled4"), default="per_face",
                   help="facades kind only: one face per sequence, or every face of a build in one")
    b.add_argument("--augment", choices=AUGMENTS, default="",
                   help="d4: every horizontal symmetry of each build, eight sequences for one (train tiers only)")
    b.add_argument("--cat", action="store_true",
                   help="add the category group: each sequence starts with its cat:<class> (blockgen.corpus.labels)")
    s = sub.add_parser("show"); s.add_argument("tier"); s.add_argument("--hash", default=None)
    s.add_argument("--kind", choices=("voxels", "facades", "pieces"), default="voxels")
    n = sub.add_parser("from-npz", help="a derived tier + shard from a structure npz (model samples a reward kept)")
    n.add_argument("npz"); n.add_argument("--tier", required=True)
    n.add_argument("--kind", choices=("voxels", "facades", "pieces"), default="voxels")
    n.add_argument("--layout", choices=("per_face", "joint", "pooled4"), default="per_face", help="facades kind only")
    n.add_argument("--faces", choices=sorted(FACE_SETS), default="all", help="facades kind only")
    n.add_argument("--oriented", action="store_true")
    n.add_argument("--tokenizer", default=None, metavar="HASH",
                   help="tokenizer hash (or tokenizer.json path); default: the one --kind/--layout/--oriented name")
    n.add_argument("--tags", nargs="*", default=[], metavar="k=v", help="override src/q/size/win for every structure")
    n.add_argument("--source", default=None, help="meta.source and the tier's source; default 'generated'")
    n.add_argument("--max-blocks", type=int, default=None, help="skip structures denser than this (none by default)")
    n.add_argument("--augment", choices=AUGMENTS, default="")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.tier, a.max_blocks, a.max_windows, a.workers, a.oriented, limit=a.limit, kind=a.kind, faces=a.faces,
              layout=a.layout, augment=a.augment, tokenizer=a.tokenizer, cats=a.cat)
    elif a.cmd == "from-npz":
        tags = dict(kv.split("=", 1) for kv in a.tags)
        if a.kind == "pieces" and a.tokenizer is None:
            raise SystemExit("--kind pieces needs --tokenizer <hash|path>: the piece vocabulary is learned")
        tok = resolve_tokenizer(a.tokenizer, a.kind, a.layout, a.oriented)
        from_structures(structures_from_npz(a.npz), a.tier, tok, tags=tags or None, source=a.source,
                        max_blocks=a.max_blocks, faces=a.faces, augment=a.augment)
    else:
        with open(os.path.join(find(a.tier, a.hash, a.kind), "spec.json")) as fh:
            print(json.dumps(json.load(fh), indent=1))


if __name__ == "__main__":
    main()
