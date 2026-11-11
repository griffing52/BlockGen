"""V-KID with real *exteriors* as the reference set (CPU).

The bench scores every arm against whole held-out houses, so a face set (which has no
interior) is penalised for what it cannot represent and Real-Shell itself sits far from
zero. This script repeats the bench's V-KID (same v2 descriptor, same standardizer
fitted on the reference, same unbiased KID at scale 1000) with the reference replaced by
the six-face visible shells of the 399 validation houses. Real test shells give the floor.
A percentile bootstrap over builds gives 95% intervals.

    PYTHONPATH=. python scripts/facade_exterior_vkid.py name=path.npz [name=path.npz ...] --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from blockgen.corpus import facade as fc  # noqa: E402
from blockgen.curation.houses import load_structures_from_cache  # noqa: E402
from blockgen.eval.bench import distances as D  # noqa: E402
from blockgen.eval.bench import geometry as geom  # noqa: E402
from blockgen.eval.bench import splits  # noqa: E402


def shells(structs):
    return [fc.shell(s, fc.ALL_FACES) for s in structs if s.occupied_mask.any()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=path.npz (builds are scored as given; pass shells for voxel arms)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-split", choices=("val", "test"), default="val",
                    help="reference split (its shells); the other split's shells give the real floor")
    a = ap.parse_args(argv)
    geom.set_descriptor_version("v2")
    split = splits.load_split("houses_32", 0)
    other = "test" if a.ref_split == "val" else "val"
    ref = shells(splits.split_structures(split, a.ref_split))
    std = geom.Standardizer.fit(geom.descriptor(ref))
    ref_f = geom.geom_features(ref, std)
    rng = np.random.default_rng(0)
    rows = {}
    arms = [(f"real_{other}@shell6", None)] + [tuple(x.split("=", 1)) for x in a.arms]
    for name, path in arms:
        if path is None:
            s = shells(splits.split_structures(split, other))
        else:
            s, _ = load_structures_from_cache(path)
            s = [x for x in s if x.occupied_mask.any()]
        f = geom.geom_features(s, std)
        v = D.kid(f, ref_f)
        boots = [D.kid(f[rng.integers(len(f), size=len(f))], ref_f) for _ in range(a.n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows[name] = {"n": len(s), "vkid_ext": float(v), "ci": [float(lo), float(hi)]}
        print(f"{name:55s} n={len(s):4d}  V-KID(ext) {v:7.2f}  [{lo:6.2f}, {hi:6.2f}]", flush=True)
    if a.out:
        json.dump({"reference": f"houses_32 {a.ref_split}, six-face shells", "descriptor": "v2", "arms": rows},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
