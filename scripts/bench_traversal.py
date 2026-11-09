"""Can a player walk into these houses? The traversability probe, run as a bench.

`blockgen.eval.traversal` sets a two-block player down outside a build and
floods everywhere it can walk: through doors, up stairs, down into cellars.
This script scores that on the bench's pool, on probes whose answer is known,
and on every generated arm with a structure cache, then renders the walk.

Arms:

* `real_test`                 -- the test-split pool, the floor.
* `real@seal_openings`        -- every doorway and window bricked up. Interior
                                 volume unchanged, nobody gets in.
* `real@lower_ceilings`       -- the top layer of every room filled. Rooms that
                                 were two high are gone to a player, the
                                 exterior is untouched.
* `real@solidify`, `real@hollow` -- the bench's interior probes.
* generated arms              -- any `--arm name=path.npz` (defaults below:
                                 the leaderboard's cached arms).

Two numbers per arm matter most. `enterable` is the share of builds a player
can get into at all. `walkable` is the share of each build's interior standing
cells a player can reach, with no interior counting as zero. `floors reached`
says whether the stairs work, and `headroom3` whether the rooms are built to
the player's scale.

    python scripts/bench_traversal.py --n 128 --gif 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from blockgen.eval import traversal as tv
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import splits
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

PROBES: Dict[str, Callable[[Structure], Structure]] = {
    "real_test": lambda s: s,
    "real@seal_openings": tv.seal_openings,
    "real@lower_ceilings": tv.lower_ceilings,
    "real@solidify": pb.solidify,
    "real@hollow": pb.hollow,
}

#: Cached generated arms, as the leaderboard scores them.
DEFAULT_ARMS = {
    "canon_oriented": "outputs/bench_arms/canon_oriented_32.npz",
    "native_oriented": "outputs/bench_arms/native_oriented_32.npz",
    "llm_control": "outputs/bench_arms/llm_control_32.npz",
    "llm_bias": "outputs/bench_arms/llm_bias_32.npz",
    "dream-caption": "outputs/run_20261008_004051_bench_houses32_dream_caption133/dream-caption_32.npz",
    "dream": "outputs/run_20261007_092654_bench_houses32_dream/dream_32.npz",
    "classic-matched": "outputs/run_20261007_092654_bench_houses32_classic/classic-matched_32.npz",
}

COLUMNS = (
    ("enterable", "enterable"),
    ("walkable", "walkable (mean)"),
    ("walkable_median", "walkable (median)"),
    ("no_interior", "no interior"),
    ("floors", "floors"),
    ("floors_reached", "reached"),
    ("all_floors", "all floors"),
    ("headroom3", "headroom3"),
    ("entrances", "entrances"),
    ("blocks", "blocks (median)"),
)


def load_arm(path: str) -> List[Structure]:
    from blockgen.curation.houses import load_structures_from_cache
    structs, _ = load_structures_from_cache(path)
    return [s.crop_to_non_air() for s in structs if s.occupied_mask.any()]


def score_arm(structs: Sequence[Structure], rng: np.random.Generator,
              n_rep: int = 200) -> dict:
    rows = [tv.traversal_stats(s).as_dict() for s in structs]
    walk = np.array([r["interior_standing_reached"] for r in rows], dtype=float)
    enter = np.array([r["enterable"] for r in rows], dtype=float)
    head = np.array([r["headroom3"] for r in rows], dtype=float)
    floors = np.array([r["floors"] for r in rows], dtype=float)
    reached = np.array([r["floors_reached"] for r in rows], dtype=float)
    blocks = np.array([int(s.occupied_mask.sum()) for s in structs], dtype=float)
    with_floors = floors > 0

    def ci(v: np.ndarray, fn=np.mean) -> List[float]:
        """Percentile interval of `fn` over 80% subsamples, the bench's habit."""
        if v.size < 2:
            return [float("nan"), float("nan")]
        k = max(1, int(round(0.8 * v.size)))
        vals = [fn(v[rng.permutation(v.size)[:k]]) for _ in range(n_rep)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    summary = {
        "n": len(rows),
        "enterable": float(enter.mean()), "enterable_ci": ci(enter),
        "walkable": float(walk.mean()), "walkable_ci": ci(walk),
        "walkable_median": float(np.median(walk)),
        "no_interior": float(np.mean([r["interior_standing"] == 0 for r in rows])),
        "floors": float(floors.mean()),
        "floors_reached": float(reached.mean()),
        "all_floors": (float((reached[with_floors] == floors[with_floors]).mean())
                       if with_floors.any() else float("nan")),
        "headroom3": float(np.nanmedian(head)) if np.isfinite(head).any() else float("nan"),
        "entrances": float(np.mean([r["entrances"] for r in rows])),
        "blocks": float(np.median(blocks)),
    }
    return {"summary": summary, "rows": rows}


def render_table(results: Dict[str, dict], real_key: str = "real_test") -> str:
    head = "| arm | n | " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|---|--:|" + "|".join("--:" for _ in COLUMNS) + "|"
    out = [head, sep]
    for name, res in results.items():
        s = res["summary"]
        cells = []
        for key, _ in COLUMNS:
            v = s[key]
            if key in ("enterable", "walkable", "walkable_median", "no_interior",
                       "all_floors", "headroom3"):
                cells.append("nan" if not np.isfinite(v) else f"{v:.2f}")
            elif key == "blocks":
                cells.append(f"{v:.0f}")
            else:
                cells.append(f"{v:.2f}")
        out.append(f"| `{name}` | {s['n']} | " + " | ".join(cells) + " |")
    lines = ["\n".join(out), ""]
    real = results.get(real_key)
    if real:
        r = real["summary"]
        lines.append(f"Real floor: enterable {r['enterable']:.2f} "
                     f"[{r['enterable_ci'][0]:.2f}, {r['enterable_ci'][1]:.2f}], "
                     f"walkable {r['walkable']:.2f} "
                     f"[{r['walkable_ci'][0]:.2f}, {r['walkable_ci'][1]:.2f}] "
                     f"(80% subsample intervals).")
    return "\n".join(lines)


# --- pictures ---------------------------------------------------------------
def _camera_for_cut(cut: str) -> float:
    """Azimuth that looks into the quadrant a `quad:` cut removes."""
    if cut.startswith("quad:"):
        _, xs, zs = cut.split(":")
        return {("lo", "lo"): 225.0, ("lo", "hi"): 315.0,
                ("hi", "lo"): 135.0, ("hi", "hi"): 45.0}[(xs, zs)]
    return 45.0


def render_frames(frames: Sequence[Structure], cut: str, px: int,
                  elev: float = 30.0) -> List[np.ndarray]:
    from blockgen.eval.bench import features as ft
    from blockgen.renderer.textured import load_face_textures, render_structure
    tex = load_face_textures()
    az = _camera_for_cut(cut)
    out = []
    for f in frames:
        g = ft.apply_cut(f, cut) if cut and cut != "none" else f
        out.append(render_structure(g, px=px, azim_deg=az, elev_deg=elev,
                                    face_textures=tex, bg=(1.0, 1.0, 1.0))[..., :3])
    return out


def write_walkthrough(s: Structure, run_dir: Path, tag: str, cut: str, px: int,
                      max_frames: int = 60, ms: int = 120) -> dict:
    """GIF of the flood fill plus a four-frame strip for a figure."""
    from PIL import Image
    w = tv.walk(s)
    step = max(1, math.ceil((w.max_dist + 1) / max_frames))
    frames = list(tv.walkthrough_frames(s, step=step))
    imgs = render_frames(frames, cut, px)
    pil = [Image.fromarray(im) for im in imgs]
    gif = run_dir / f"walkthrough_{tag}.gif"
    pil[0].save(gif, save_all=True, append_images=pil[1:] + [pil[-1]] * 8,
                duration=ms, loop=0, optimize=False)
    picks = sorted({0, len(pil) // 3, 2 * len(pil) // 3, len(pil) - 1})
    strip = Image.new("RGB", (px * len(picks), px), "white")
    for k, i in enumerate(picks):
        strip.paste(pil[i], (k * px, 0))
    strip.save(run_dir / f"walkthrough_{tag}_strip.png")
    return {"gif": gif.name, "frames": len(pil), "bfs_steps": w.max_dist,
            "stats": tv.traversal_stats(s).as_dict(),
            "title": s.metadata.get("title", "")}


def write_samples(pool: Sequence[Structure], run_dir: Path, cut: str, px: int,
                  k: int = 4) -> None:
    """samples.png: k real houses, uncut beside their finished walk, cut open."""
    from PIL import Image
    from blockgen.renderer.textured import load_face_textures, render_structure
    tex = load_face_textures()
    rows = []
    for s in pool[:k]:
        final = list(tv.walkthrough_frames(s, step=max(1, tv.walk(s).max_dist)))[-1]
        plain = render_structure(s, px=px, azim_deg=_camera_for_cut(cut), elev_deg=30,
                                 face_textures=tex, bg=(1.0, 1.0, 1.0))[..., :3]
        walked = render_frames([final], cut, px)[0]
        rows.append(np.concatenate([plain, walked], axis=1))
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(run_dir / "samples.png")


# --- main -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=128, help="real builds per probe arm")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=PATH.npz",
                    help="a generated arm's structure cache (repeatable); default: "
                         "the leaderboard's cached arms that exist on disk")
    ap.add_argument("--no-default-arms", action="store_true")
    ap.add_argument("--gif", type=int, default=2,
                    help="walkthrough GIFs for this many real houses (0 to skip)")
    ap.add_argument("--gif-index", type=int, action="append", default=None,
                    help="pool index to animate instead of the automatic picks")
    ap.add_argument("--cut", default="quad:lo:lo",
                    help="features.apply_cut spec for the renders; 'none' for uncut")
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    split = splits.load_split(args.corpus, args.seed)
    test = splits.split_structures(split, "test")
    pool = [s.crop_to_non_air() for s in test]
    pool = [pool[i] for i in rng.permutation(len(pool))[:args.n]]
    print(f"[traversal] pool {len(pool)} of {len(test)} test builds", flush=True)

    arms: Dict[str, List[Structure]] = {name: [fn(s) for s in pool]
                                        for name, fn in PROBES.items()}
    sources: Dict[str, str] = {name: f"probe on {args.corpus} test" for name in PROBES}
    gen = {} if args.no_default_arms else dict(DEFAULT_ARMS)
    for spec in args.arm or []:
        name, _, path = spec.partition("=")
        gen[name] = path
    for name, path in gen.items():
        if not Path(path).exists():
            print(f"[traversal] skip {name}: {path} not found", flush=True)
            continue
        arms[name] = load_arm(path)
        sources[name] = path

    results: Dict[str, dict] = {}
    for name, structs in arms.items():
        results[name] = score_arm(structs, np.random.default_rng(args.seed + 1))
        s = results[name]["summary"]
        print(f"  {name:<22} n={s['n']:<4} enterable {s['enterable']:.2f}  "
              f"walkable {s['walkable']:.2f} (median {s['walkable_median']:.2f})  "
              f"floors {s['floors_reached']:.2f}/{s['floors']:.2f}  "
              f"headroom3 {s['headroom3']:.2f}", flush=True)

    run_dir = Path(args.out) if args.out else new_run_dir("bench_traversal")
    run_dir.mkdir(parents=True, exist_ok=True)

    pictures: Dict[str, dict] = {}
    if args.gif:
        # Houses worth watching: a titled (catalogued) build, fully walkable,
        # at least two floors reached, with the most floor to walk.
        picks = args.gif_index
        if not picks:
            stats = results["real_test"]["rows"]
            order = sorted(range(len(pool)), key=lambda i: (
                -(stats[i]["floors_reached"] >= 2 and stats[i]["interior_standing_reached"] >= 0.95),
                -bool(pool[i].metadata.get("title")),
                -min(stats[i]["floors_reached"], 3), -stats[i]["interior_standing"]))
            picks = order[:args.gif]
        for i in picks:
            tag = f"{i:03d}"
            pictures[tag] = write_walkthrough(pool[i], run_dir, tag, args.cut, args.px)
            print(f"  [gif] pool {i} {pictures[tag]['title']!r}: "
                  f"{pictures[tag]['frames']} frames -> {pictures[tag]['gif']}", flush=True)
        write_samples([pool[i] for i in picks] + pool[:4], run_dir, args.cut, args.px)

    payload = {
        "context": {"corpus": args.corpus, "seed": args.seed, "n": len(pool),
                    "split_key": split.key(), "cut": args.cut, "sources": sources},
        "summary": {k: v["summary"] for k, v in results.items()},
        "rows": {k: v["rows"] for k, v in results.items()},
        "pictures": pictures,
    }
    (run_dir / "traversal.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = render_table(results)
    (run_dir / "traversal.md").write_text(md + "\n")
    print()
    print(md)
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
