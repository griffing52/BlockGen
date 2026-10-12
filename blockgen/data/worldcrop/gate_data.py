"""Natural-terrain negatives for the gate: patches cut from chunks with no built block.

Streams a strided sample of one archive month, extracts each world, scans chunks with
the detector's own seed rule, and samples boxes from chunk groups where *no* chunk has a
strong (built) block. Trees, lakes, snow, cliffs and caves are all in; anything a player
placed is (by the detector's definition) not. Output: ``gate/terrain_<month>.npz`` with
object arrays ``ids``, ``data``, ``mask`` (mask = every non-air voxel) and ``meta``.

    python -m blockgen.data.worldcrop.gate_data --month 201407 --worlds 40 --per-world 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time

import numpy as np

from . import anvil, detect, remote, run
from .gate import GATE_DIR


def sample_world(world_dir: str, rng: random.Random, per_world: int, cfg: detect.DetectConfig):
    level = anvil.read_level_dat(os.path.join(world_dir, "level.dat"))
    if level.is_flat:
        return []
    rule = detect.SeedRule(level)
    stats, _ = detect.scan_chunks(world_dir, level, rule, cfg)
    natural = {k for k, s in stats.items() if s.n_seed == 0 and s.n_blocks > 500}
    if len(natural) < 4:
        return []
    out = []
    tries = 0
    keys = list(natural)
    while len(out) < per_world and tries < per_world * 6:
        tries += 1
        cx, cz = rng.choice(keys)
        k = rng.choice((1, 1, 2, 2, 3, 4))
        group = [(cx + i, cz + j) for i in range(k) for j in range(k)]
        if any(g not in natural for g in group):
            continue
        # natural chunks carry no seed y-range (y_min = y_max = -1): load the full column
        ids, data, _ = detect._load_tile(world_dir, cfg, cx, cz, cx + k - 1, cz + k - 1, 0, 15, stats)
        occ = ids != 0
        if occ.sum() < 200:
            continue
        # surface band: from a few blocks under the lowest surface to the top
        top = np.where(occ.any(1), ids.shape[1] - 1 - np.argmax(occ[:, ::-1, :], axis=1), -1)
        tv = top[top >= 0]
        if tv.size < 16:
            continue
        y1 = int(tv.max()) + 1
        y0 = max(int(tv.min()) - rng.randint(1, 10), 0)
        side = rng.choice((8, 12, 16, 20, 24, 32, 40, 48, 64))
        side = min(side, ids.shape[0])
        x0 = rng.randint(0, ids.shape[0] - side); z0 = rng.randint(0, ids.shape[2] - side)
        sub = ids[x0:x0 + side, y0:y1, z0:z0 + side]
        sdat = data[x0:x0 + side, y0:y1, z0:z0 + side]
        m = sub != 0
        n = int(m.sum())
        if n < 24 or n == m.size:            # empty or a solid slab of stone: keep it varied
            if rng.random() < 0.9:
                continue
        # tight crop
        xs, yy, zs = np.nonzero(m)
        if xs.size == 0:
            continue
        sl = np.s_[xs.min():xs.max() + 1, yy.min():yy.max() + 1, zs.min():zs.max() + 1]
        out.append((sub[sl].astype(np.uint16), sdat[sl].astype(np.uint8), m[sl],
                    {"world": os.path.basename(world_dir), "chunk": (cx, cz), "k": k, "side": side}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="201407")
    ap.add_argument("--worlds", type=int, default=40)
    ap.add_argument("--per-world", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    cfg = detect.DetectConfig()
    os.makedirs(GATE_DIR, exist_ok=True)
    arch = run._archive(args.month, None)
    entries = [e for e in arch.worlds() if 50_000 <= e.size <= 150_000_000]
    rng.shuffle(entries)
    allp = []
    t0 = time.time()
    for e in entries:
        tmp = tempfile.mkdtemp(prefix="gate_", dir=run.SCRATCH)
        try:
            ap_ = os.path.join(tmp, "a" + os.path.splitext(e.member)[1])
            arch.fetch(e, ap_)
            dest = os.path.join(tmp, "x")
            if not run.extract_archive(ap_, dest):
                continue
            dirs = anvil.find_world_dirs(dest)[:2]
            got = 0
            for wd in dirs:
                try:
                    ps = sample_world(wd, rng, args.per_world, cfg)
                except Exception as ex:
                    print("skip", e.member, type(ex).__name__, ex, flush=True)
                    continue
                for ids, dat, m, meta in ps:
                    meta["world"] = e.world_id; meta["title"] = e.title
                    allp.append((ids, dat, m, meta))
                got += len(ps)
            n_worlds = len({p[3]["world"] for p in allp})
            print(f"{e.world_id} {e.title[:40]!r}: +{got} patches, total {len(allp)} from {n_worlds} worlds, {time.time()-t0:.0f}s", flush=True)
            if n_worlds >= args.worlds:
                break
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    out = os.path.join(GATE_DIR, f"terrain_{args.month}.npz")
    np.savez_compressed(out, ids=np.array([p[0] for p in allp], dtype=object), data=np.array([p[1] for p in allp], dtype=object),
                        mask=np.array([p[2] for p in allp], dtype=object), meta=np.array([json.dumps(p[3]) for p in allp]))
    print("wrote", out, len(allp))




# --------------------------------------------------------------------------- #
# Synthetic edited-terrain negatives
# --------------------------------------------------------------------------- #
# Natural chunks never contain the shapes that actually leak through the detector:
# superflat plates, WorldPainter mounds of snow block / clay / stone, and bare dirt
# pillars left by flat-world edits. Those are cheap to synthesise exactly.
_SURFACE_STACKS = [          # (top ids..., filler) drawn top-down
    ((2,), 3), ((2,), 3), ((80,), 80), ((80, 80), 1), ((12,), 24), ((12, 12), 12), ((1,), 1),
    ((172,), 172), ((159,), 159), ((3,), 3), ((13,), 1), ((78, 2), 3), ((79,), 79), ((110,), 3),
    ((43,), 43), ((1,), 7), ((98,), 98), ((82,), 82), ((174,), 174),
]


def _heightfield(rng, w, d, amp, smooth):
    h = np.zeros((w, d))
    for _ in range(rng.randint(1, 5)):
        cx, cz = rng.uniform(0, w), rng.uniform(0, d)
        sx, sz = rng.uniform(smooth * 0.5, smooth * 2), rng.uniform(smooth * 0.5, smooth * 2)
        a = rng.uniform(0.2, 1.0) * amp
        xs, zs = np.arange(w)[:, None], np.arange(d)[None, :]
        h += a * np.exp(-((xs - cx) ** 2 / (2 * sx ** 2) + (zs - cz) ** 2 / (2 * sz ** 2)))
    return h


def synth_terrain(n: int, rng: random.Random):
    out = []
    while len(out) < n:
        kind = rng.choice(("plate", "plate", "mound", "mound", "ridge", "pillar", "slab_stack"))
        top, filler = rng.choice(_SURFACE_STACKS)
        w, d = rng.randint(6, 64), rng.randint(6, 64)
        if kind == "plate":
            h = np.full((w, d), rng.randint(1, 6), float)
            if rng.random() < 0.5:   # bite a few holes / ragged edge
                h[rng.randint(0, w - 1), :] = 0
        elif kind == "mound":
            h = 1 + _heightfield(rng, w, d, rng.uniform(3, 40), rng.uniform(4, 24))
        elif kind == "ridge":
            h = 1 + _heightfield(rng, w, 3, rng.uniform(4, 30), rng.uniform(3, 20))
            h = np.repeat(h, max(1, d // 3), axis=1)[:, :d]
            if h.shape[1] < d:
                h = np.pad(h, ((0, 0), (0, d - h.shape[1])), mode="edge")
        elif kind == "pillar":
            w, d = rng.randint(4, 24), rng.randint(4, 24)
            h = np.full((w, d), rng.randint(6, 48), float)
        else:  # slab_stack: a few flat layers of different materials
            h = np.full((w, d), rng.randint(2, 8), float)
        H = int(np.ceil(h.max()))
        ids = np.zeros((w, H, d), np.uint16)
        hi = np.clip(np.round(h).astype(int), 0, H)
        for x in range(w):
            for z in range(d):
                k = hi[x, z]
                if k <= 0:
                    continue
                ids[x, :k, z] = filler
                for j, t in enumerate(top):
                    if k - 1 - j >= 0:
                        ids[x, k - 1 - j, z] = t
        if kind == "slab_stack":
            for y in range(H):
                layer = ids[:, y, :]
                layer[layer > 0] = rng.choice((1, 3, 2, 12, 24, 43, 80, 159, 172))
        m = ids != 0
        if m.sum() < 24:
            continue
        out.append((ids, np.zeros_like(ids, np.uint8), m, {"synth": kind, "top": list(top), "filler": filler}))
    return out


def main_synth(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("solid", "skin"), default="solid")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    ps = synth_terrain(args.n, rng) if args.kind == "solid" else synth_skins(args.n, rng)
    os.makedirs(GATE_DIR, exist_ok=True)
    out = os.path.join(GATE_DIR, f"synth_{'terrain' if args.kind == 'solid' else 'skin'}_{args.seed}.npz")
    np.savez_compressed(out, ids=np.array([p[0] for p in ps], dtype=object), data=np.array([p[1] for p in ps], dtype=object),
                        mask=np.array([p[2] for p in ps], dtype=object), meta=np.array([json.dumps(p[3]) for p in ps]))
    print("wrote", out, len(ps))


# The detector masks a crop to its *component*: on edited terrain that is the surface
# skin of seed blocks (snow block, clay, sandstone, stone bricks...) over unmasked fill,
# never the solid mound. The missed WorldPainter crops have fill 0.01-0.26 and exposed
# faces 0.4-0.5 for exactly that reason, so negatives must be skins too.
_SKIN_MATERIALS = (80, 80, 159, 172, 24, 12, 2, 1, 13, 82, 78, 110, 174, 79, 98, 4, 48, 3, 43, 155, 35)


def synth_skins(n: int, rng: random.Random):
    out = []
    while len(out) < n:
        kind = rng.choice(("mound", "mound", "mound", "ridge", "plate", "cliff"))
        w, d = rng.randint(8, 96), rng.randint(8, 96)
        if kind == "mound":
            h = 1 + _heightfield(rng, w, d, rng.uniform(4, 50), rng.uniform(4, 30))
        elif kind == "ridge":
            h = 1 + _heightfield(rng, w, 3, rng.uniform(6, 40), rng.uniform(3, 20))
            h = np.repeat(h, max(1, d // 3), axis=1)[:, :d]
            if h.shape[1] < d:
                h = np.pad(h, ((0, 0), (0, d - h.shape[1])), mode="edge")
        elif kind == "plate":
            h = np.full((w, d), rng.randint(2, 12), float) + _heightfield(rng, w, d, rng.uniform(0, 3), rng.uniform(6, 30))
        else:  # cliff: a step in height with a sloped face
            h = np.full((w, d), rng.randint(2, 6), float)
            step = rng.randint(1, max(2, w - 2))
            h[step:, :] += rng.uniform(6, 40)
            h = np.asarray(np.maximum(h, 1))
            from scipy import ndimage as _nd
            h = _nd.gaussian_filter(h, rng.uniform(0.5, 3.0))
        H = int(np.ceil(h.max())) + 1
        hi = np.clip(np.round(h).astype(int), 1, H - 1)
        # real WorldPainter crops: the seed skin covers 10-60% of the columns in patches
        # (snow on the upper slopes, clay bands) and is 1-8 blocks thick
        k_mean = rng.choice((1, 1, 2, 3, 4, 6, 7, 8))
        top, under = rng.choice(_SKIN_MATERIALS), rng.choice(_SKIN_MATERIALS)
        ids = np.zeros((w, H, d), np.uint16)
        mask = np.zeros((w, H, d), bool)
        foot = np.ones((w, d), bool)
        mode = rng.choice(("all", "disc", "patches", "patches", "upper"))
        xs, zs = np.arange(w)[:, None], np.arange(d)[None, :]
        if mode == "disc":
            cx, cz, r = rng.uniform(0, w), rng.uniform(0, d), rng.uniform(0.4, 0.9) * max(w, d)
            foot &= ((xs - cx) ** 2 + (zs - cz) ** 2) < r ** 2
        elif mode == "patches":
            noise = _heightfield(rng, w, d, 1.0, rng.uniform(3, 15)) + _heightfield(rng, w, d, 1.0, rng.uniform(2, 8))
            foot &= noise > np.quantile(noise, rng.uniform(0.4, 0.9))
        elif mode == "upper":
            foot &= h > np.quantile(h, rng.uniform(0.4, 0.9))
        if foot.sum() < 16:
            continue
        for x in range(w):
            for z in range(d):
                if not foot[x, z]:
                    continue
                t = hi[x, z]
                k = max(1, int(round(rng.gauss(k_mean, 1.0))))
                lo = max(t - k, 0)
                ids[x, lo:t, z] = top
                mask[x, lo:t, z] = True
                if lo > 0 and rng.random() < 0.3:
                    ids[x, lo - 1, z] = under          # context voxel, unmasked
        # a few trees / debris that weak growth would have pulled in
        for _ in range(rng.randint(0, 6)):
            x, z = rng.randrange(w), rng.randrange(d)
            if foot[x, z]:
                t = hi[x, z]
                hgt = rng.randint(1, 6)
                ids[x, t:t + hgt, z] = 17
                mask[x, t:t + hgt, z] = True
        if mask.sum() < 24:
            continue
        xs, ys, zs = np.nonzero(mask)
        sl = np.s_[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]
        out.append((ids[sl], np.zeros_like(ids[sl], np.uint8), mask[sl], {"synth": "skin_" + kind, "top": int(top), "k": int(k)}))
    return out


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "synth":
        main_synth(_sys.argv[2:])
    else:
        main()
