"""Precompute frozen condition embeddings for conditioned AR training.

LegoACE conditions its decoder on frozen-encoder embeddings (DINOv2 for 4-view
images, CLIP for text) projected and prepended as prefix tokens. We do the same
but precompute the embeddings once, so training never loads the encoders:

* images: DINOv2-base CLS token per view  -> ``image_embeds`` [N, 4, 768]
* text:   CLIP ViT-B/32 pooled embedding  -> ``text_embeds``  [N, 4, 512]
          (one per caption, 4 captions per structure)

Output: ``data/minecraft/labels/houses_32_cond_embeds.npz`` keyed by structure
index (row i == houses cache index i == id h{i:05d}).

Usage:
    python -m blockgen.labeling.embed_conditions \
        --renders outputs/renders/houses_32 \
        --captions data/minecraft/labels/houses_32_captions.json \
        --out data/minecraft/labels/houses_32_cond_embeds.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

N_VIEWS = 4
BATCH = 64


@torch.no_grad()
def embed_images(renders_dir: str, n_structures: int, device: str) -> np.ndarray:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    out = np.zeros((n_structures, N_VIEWS, 768), dtype=np.float32)
    paths, owners = [], []
    for i in range(n_structures):
        for v in range(N_VIEWS):
            p = os.path.join(renders_dir, f"h{i:05d}_view{v}.png")
            if os.path.exists(p):
                paths.append(p)
                owners.append((i, v))
    for start in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0]  # (B, 768)
        for (i, v), e in zip(owners[start:start + BATCH], cls.float().cpu().numpy()):
            out[i, v] = e
        if (start // BATCH) % 20 == 0:
            print(f"images {start + len(imgs)}/{len(paths)}", flush=True)
    return out


@torch.no_grad()
def embed_texts(captions: dict, n_structures: int, device: str) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    out = np.zeros((n_structures, 4, 512), dtype=np.float32)
    flat, owners = [], []
    for i in range(n_structures):
        caps = captions.get(f"h{i:05d}", [])
        for j, c in enumerate(caps[:4]):
            flat.append(c)
            owners.append((i, j))
    for start in range(0, len(flat), BATCH):
        inputs = processor(text=flat[start:start + BATCH], return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        emb = model.get_text_features(**inputs)  # (B, 512)
        for (i, j), e in zip(owners[start:start + BATCH], emb.float().cpu().numpy()):
            out[i, j] = e
        if (start // BATCH) % 20 == 0:
            print(f"texts {start + len(inputs['input_ids'])}/{len(flat)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dim", type=int, default=32)
    args = parser.parse_args()

    from blockgen.curation.houses import load_house_structures
    structures, _ = load_house_structures(max_dim=args.max_dim)
    n = len(structures)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.captions) as f:
        captions = json.load(f)

    image_embeds = embed_images(args.renders, n, device)
    text_embeds = embed_texts(captions, n, device)
    np.savez_compressed(args.out, image_embeds=image_embeds, text_embeds=text_embeds)
    print(f"saved image {image_embeds.shape} + text {text_embeds.shape} -> {args.out}")


if __name__ == "__main__":
    main()
