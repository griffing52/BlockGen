import argparse
import json
import os
import sys

from .config import apply_overrides, load
from .trainer import train


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="python -m blockgen.train --config configs/train/x.yaml [key=value ...]  |  "
                    "python -m blockgen.train --sample <run_dir> --n 399 [--out-dir D] [--ckpt best.pt] "
                    "[--cat house,castle|all [--sizes auto|tags]]")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", default=None, metavar="RUN_DIR",
                    help="sample a finished run into a structure cache the bench can score, instead of training")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ckpt", default=None, help="checkpoint file in the run dir; default the one the run's select chose "
                    "(summary.json selected_ckpt), else best.pt if it exists, else last.pt")
    ap.add_argument("--out-dir", default=None, help="where samples.png and the npz go; default the run dir")
    ap.add_argument("--cat", default=None, metavar="CLASSES",
                    help="category model: sample each class separately (comma list, or 'all' for every class "
                         "but none); writes samples_cat-<c>_n<N>_48.npz per class and cat_samples.json")
    ap.add_argument("--sizes", choices=("auto", "tags"), default="auto",
                    help="with --cat: split each class's n over its training size tags (auto), or use sample_tags")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    if a.sample:
        from .sample import sample_and_render
        from .trainer import load_run_checkpoint
        ckpt = a.ckpt or selected_ckpt(a.sample)
        model, tokenizer, cfg, info = load_run_checkpoint(a.sample, ckpt)
        apply_overrides(cfg, a.overrides)         # e.g. temperature=0.9 top_k=20 seed=1, coerced like a YAML key
        out = a.out_dir or a.sample
        os.makedirs(out, exist_ok=True)
        if a.cat:
            from .sample import sample_classes
            cats = [c for c in tokenizer.cats or [] if c != "none"] if a.cat == "all" else a.cat.split(",")
            index = sample_classes(model, tokenizer, cfg, out, a.n, cats, step=info["step"], ckpt=ckpt, sizes=a.sizes)
            print(f"[sample] {ckpt} of {a.sample}: {len(index)} classes -> {os.path.join(out, 'cat_samples.json')}",
                  flush=True)
            return 0
        # The report carries the checkpoint's real step; the cache keeps its sample-count name
        # (samples_step<n>, the path existing recipes and scripts point at) so it never
        # overwrites the trainer's own end-of-run cache of the same step.
        path = sample_and_render(model, tokenizer, cfg, out, step=info["step"], n=a.n, ckpt=ckpt, name=f"samples_step{a.n}")
        print(f"[sample] {ckpt} of {a.sample} -> {path}", flush=True)
        return 0
    cfg = load(a.config, a.overrides)
    train(cfg)
    return 0


def selected_ckpt(run_dir: str) -> str:
    """The checkpoint the run's ``select`` kept, which is what its card scored; ``best.pt``
    is only the lowest-nll one, and under ``select=reward`` or ``select=last`` it can differ."""
    try:
        with open(os.path.join(run_dir, "summary.json")) as f:
            ckpt = json.load(f).get("selected_ckpt")
        if ckpt and os.path.exists(os.path.join(run_dir, ckpt)):
            return ckpt
    except (OSError, ValueError):
        pass
    return "best.pt" if os.path.exists(os.path.join(run_dir, "best.pt")) else "last.pt"


if __name__ == "__main__":
    sys.exit(main())
