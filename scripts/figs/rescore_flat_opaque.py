"""How much of MV-KID's residual +0.38 on Fill-Interior is our renderer's artefact?

`app:hull` proves that filling sealed air cannot change an exterior view under an
opaque renderer, and measures that our deployed renderer nevertheless alters 91%
of views. Two leaks explain that: Minecraft glass is translucent (a real channel)
and `renderer.textured.build_mesh` repacks its texture atlas when the interior
faces it also meshes stop existing (an artefact).

The open question this answers: the benchmark reports MV-KID placing the probe
+0.38 noise floors from real. If most of that comes from the atlas artefact, the
*measured* blindness is deeper than the paper reports -- so we rescore the same
arms, the same draw and the same estimator through four render pipelines that
switch each leak off.

The prediction, from the lemma: `flat_opaque` must give a separation of EXACTLY
zero, because the renders are bit-identical and therefore so are the features.
That is a falsifiable end-to-end check of the proof through the real bench path,
not a restatement of it.

Reproduces the bench's own draw: pool `default_rng(seed+1)`, floor
`default_rng(seed+991)` with reps=48 against the full test set, `D.kid(arm, ref)`.
Run: `.venv/bin/python scripts/figs/rescore_flat_opaque.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEED = 0
N = 128
REPS = 48
OUT = ROOT / "outputs" / "figstage" / "rescore_flat_opaque.json"

_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def _opaque_library(pack_dir):
    """A texture library with every alpha channel forced to 255."""
    from blockgen.renderer.textures import FaceTextureLibrary

    class _Opaque(FaceTextureLibrary):
        def __missing__(self, key):
            faces = super().__missing__(key)
            out = {}
            for k, arr in faces.items():
                a = np.asarray(arr).copy()
                a[..., 3] = 255
                out[k] = a
            self[key] = out
            return out

    return _Opaque(pack_dir=pack_dir)


def pipelines():
    """The deployed path and three counterfactuals isolating its two leaks.

    `flat` removes the atlas artefact (every tile is one 2x2 solid colour, so no
    UV shift can move a texel); `opaque` removes transmission; `flat_opaque`
    removes both and is the setting the occlusion lemma assumes.
    """
    from blockgen.renderer.textures import load_face_textures, DEFAULT_PACK_DIR
    return {
        "paper":       lambda: load_face_textures(),
        "opaque":      lambda: _opaque_library(DEFAULT_PACK_DIR),
        "flat":        lambda: load_face_textures(pack_dir=_FLAT_DIR),
        "flat_opaque": lambda: _opaque_library(_FLAT_DIR),
    }


def main() -> int:
    import blockgen.renderer.textures as tx
    from blockgen.eval.bench import splits, probes, features as ft, ladder as ld
    from blockgen.eval.bench import distances as D

    split = splits.load_split("houses_32")
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    ref_structs = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]
    print(f"test {len(test)}  ref {len(ref_structs)}", flush=True)

    # the bench's own pool draw
    idx = np.random.default_rng(SEED + 1).permutation(len(test))[:N]
    real_arm = [test[i] for i in idx]
    solid_arm = [probes.solidify(s) for s in real_arm]

    view = ft.ViewConfig()
    orig_loader = tx.load_face_textures
    results = {}

    for name, make in pipelines().items():
        t0 = time.time()
        lib = make()
        tx.load_face_textures = lambda *a, **k: lib
        try:
            f_ref = ft.pooled(ft.embed_views(ref_structs, view, verbose=False))
            f_all = ft.pooled(ft.embed_views(test, view, verbose=False))
            f_real = ft.pooled(ft.embed_views(real_arm, view, verbose=False))
            f_sol = ft.pooled(ft.embed_views(solid_arm, view, verbose=False))
        finally:
            tx.load_face_textures = orig_loader

        feats_identical = bool(np.array_equal(f_real, f_sol))
        max_feat_diff = float(np.abs(f_real - f_sol).max())

        kid_real = float(D.kid(f_real, f_ref))
        kid_sol = float(D.kid(f_sol, f_ref))
        _, sd, _ = ld.noise_floor(lambda r, p: D.kid(p, r), f_ref, f_all, N,
                                  reps=REPS,
                                  rng=np.random.default_rng(SEED + 991))
        sep = (kid_sol - kid_real) / sd if sd > 0 else float("nan")
        results[name] = {
            "kid_real": kid_real, "kid_solidify": kid_sol,
            "noise_floor_sd": float(sd),
            "separation_floors": float(sep),
            "features_bit_identical": feats_identical,
            "max_abs_feature_diff": max_feat_diff,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name:12s} kid_real={kid_real:+.6f} kid_sol={kid_sol:+.6f} "
              f"sd={sd:.6f} sep={sep:+.3f} feats_identical={feats_identical} "
              f"({results[name]['seconds']}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {"n": N, "reps": REPS, "seed": SEED,
                 "n_ref": len(ref_structs), "n_test": len(test),
                 "split_key": str(getattr(split, "key", None))},
        "pipelines": results,
    }, indent=1))
    print(f"\nwrote {OUT}")

    fo = results["flat_opaque"]
    if not fo["features_bit_identical"]:
        print("\n*** LEMMA CHECK FAILED: flat_opaque features are NOT identical "
              f"(max diff {fo['max_abs_feature_diff']:.3e}). The proof in "
              "app:hull predicts they must be. Investigate before trusting "
              "anything else here. ***")
        return 1
    print("\nlemma check PASSED: flat_opaque features bit-identical, "
          f"separation {fo['separation_floors']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
