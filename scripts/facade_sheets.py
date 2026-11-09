"""Real vs generated face sheets, and where the fragmentation lives: in the faces or at the edges."""
import os, sys, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
sys.path.insert(0, '.')
from blockgen.corpus import facade as fc
from blockgen.curation.houses import load_structures_from_cache
from blockgen.eval.facade_stats import face_stats

ARMS = {
    "real (test shell)": "outputs/bench_arms/real_facades6_32.npz",
    "patchwork": "outputs/bench_arms/facade_patchwork_32.npz",
    "joint 20M plain": "outputs/run_20261028_110415_houses32_facades_joint_20m/samples399/samples_step399_48.npz",
    "joint 20M D4": "outputs/run_20261028_125240_houses32_facades_joint_d4_20m/samples399/samples_step399_48.npz",
}
OUT = "outputs/analysis/facade_sheets"
rng = np.random.default_rng(0)

def sheet_of(structs, k=8, scale=4):
    idx = rng.choice(len(structs), k, replace=False)
    sheets = [fc.sheet(fc.project_all(structs[i].crop_to_non_air(), fc.ALL_FACES), scale=scale) for i in idx]
    W = max(s.shape[1] for s in sheets)
    canvas = np.full((sum(s.shape[0] + 4 for s in sheets), W, 3), 255, np.uint8)
    y = 0
    for s in sheets:
        canvas[y:y + s.shape[0], :s.shape[1]] = s; y += s.shape[0] + 4
    return canvas


def main():
    arms = dict(a.split("=", 1) for a in sys.argv[1:]) if len(sys.argv) > 1 else ARMS   # name=path pairs replace the defaults
    rows = []
    for name, path in arms.items():
        structs, _ = load_structures_from_cache(path)
        structs = [s for s in structs if s.occupied_mask.any()]
        img = sheet_of(structs)
        Image.fromarray(img).save(os.path.join(OUT, name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"))
        st = [face_stats(s) for s in structs]
        agg = {k: float(np.median([d[k] for d in st])) for k in st[0]}
        agg_mean = {k: float(np.mean([d[k] for d in st])) for k in st[0]}
        rows.append((name, agg, agg_mean))

    print(f"{'arm':20s} {'2D comps/face':>13s} {'hole frac':>10s} {'jump frac':>10s} {'lone frac':>10s} | {'3D comps':>9s} {'lcc':>6s} {'dust(<5)':>9s} {'dust blk%':>9s}")
    for name, a, m in rows:
        print(f"{name:20s} {m['face_comps']:13.2f} {m['face_holes']:10.3f} {m['face_jumps']:10.3f} {m['face_lone']:10.4f} | "
              f"{m['comps3d']:9.1f} {m['lcc']:6.3f} {m['dust']:9.1f} {m['dust_blocks']*100:9.2f}")
    print("(means over 399 builds; 2D components use 8-connectivity on the face mask, 3D components 26-connectivity)")


if __name__ == "__main__":
    main()
