"""Standard sample rendering for conditioned runs: TEXTURED builds + input prompts.

Every conditioned run should render its held-out-val samples with the real-texture EGL
renderer (not the matplotlib voxel grid) and title each build with the exact caption it
was conditioned on, so prompt -> output is visible at a glance. Both the trainer
(``scripts/train_cond_resampler.py``) and the standalone re-render
(``scripts/render_cond_samples.py``) call ``render_cond_run`` here, so the format stays
consistent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blockgen.export.minecraftace import piece_records_to_structure
from blockgen.training.train_conditioned import BOS, EOS, NUM_SPECIAL, PAD
from blockgen.utils.data import Structure


def decode_tokens(tokens: Sequence[int], cv) -> Structure:
    """[BOS,(X,Y,Z,PIECE)*,EOS] piece tokens -> Structure (mirrors sample_conditioned)."""
    coord_off, piece_off = NUM_SPECIAL, NUM_SPECIAL + cv.max_dim
    recs, buf = [], []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t in (BOS, PAD):
            continue
        if len(buf) < 3:
            buf.append(t - coord_off + 1) if coord_off <= t < piece_off else buf.clear()
        else:
            if piece_off <= t:
                recs.append(buf + [t - piece_off])
            buf = []
    return piece_records_to_structure(np.array(recs or [[1, 1, 1, 0]]), cv)


def textured_prompt_grid(samples: Sequence[Structure], prompts: Sequence[str],
                         out_path, *, cols: int = 4, px: int = 384,
                         suptitle: str = "") -> Path:
    """Textured render of each build in a grid, titled with its input prompt."""
    from blockgen.renderer.textured import load_face_textures, render_structure
    lib = load_face_textures()
    imgs = []
    for s in samples:
        try:
            imgs.append(render_structure(s, px=px, bg=(1, 1, 1), face_textures=lib))
        except Exception as e:  # noqa: BLE001 - a bad decode must not kill the sheet
            imgs.append(None)
            print(f"textured render failed: {e}", flush=True)

    n = len(imgs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, img in enumerate(imgs):
        ax = axes[k]
        if img is not None:
            ax.imshow(img)
        blocks = int(samples[k].occupied_mask.sum())
        title = textwrap.fill(f'"{prompts[k]}"', width=34) + f"\n[{blocks} blocks]"
        ax.set_title(title, fontsize=7.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def render_cond_run(model, cv, picks: Sequence[int], tok_embeds, tok_mask,
                    captions: dict, out_path, *, cfg_scale: float = 3.0,
                    caption_idx: int = 0, device: str = "cuda", max_new: int = 2500,
                    cols: int = 4, px: int = 384, suptitle: str = "") -> Path:
    """Generate from val captions, decode, and textured-render with prompts.

    ``picks`` are structure indices; each is conditioned on its ``caption_idx`` caption
    (0 = the short tag). ``tok_embeds`` [N,4,L,D] + ``tok_mask`` [N,4,L] are the
    sequence-level condition embeddings; ``captions`` maps ``h{i:05d}`` -> caption list.
    """
    model.eval()
    conds = [tok_embeds[i, caption_idx, : int(tok_mask[i, caption_idx].sum())]
             .astype(np.float32) for i in picks]
    S = max(c.shape[0] for c in conds)
    cond = torch.zeros(len(conds), S, conds[0].shape[1], device=device)
    cmask = torch.ones(len(conds), S, dtype=torch.bool, device=device)  # True = pad
    for k, c in enumerate(conds):
        cond[k, : c.shape[0]] = torch.from_numpy(c)
        cmask[k, : c.shape[0]] = False

    toks = model.generate_cond(cond=cond, bos_token_id=BOS, eos_token_id=EOS,
                               max_new_tokens=max_new, temperature=1.0, top_k=32,
                               cfg_scale=cfg_scale, cond_mask=cmask)
    samples = [decode_tokens(row.cpu().numpy(), cv) for row in toks]
    prompts = [captions[f"h{i:05d}"][caption_idx] for i in picks]
    return textured_prompt_grid(samples, prompts, out_path, cols=cols, px=px,
                                suptitle=suptitle)
