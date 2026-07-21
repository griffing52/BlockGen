"""Precompute frozen condition embeddings for conditioned AR training.

LegoACE conditions its decoder on frozen-encoder embeddings (DINOv2 for 4-view
images, CLIP for text) projected and prepended as prefix tokens. We do the same
but precompute the embeddings once, so training never loads the encoders:

* images: DINOv2-base CLS token per view  -> ``image_embeds`` [N, 4, 768]
* text:   CLIP ViT-B/32 pooled embedding  -> ``text_embeds``  [N, 4, 512]
          (one per caption, 4 captions per structure)

**Sequence-level embeddings (idea #6, ideas.md).** The pooled ``text_embeds`` is a
single vector per caption — the textbook weak conditioning channel. For the
``ResampledCondVoxelAR2`` conditioning-channel upgrade we also emit the per-token
sequence so a resampler can cross-attend the whole prompt:

* text sequence: CLIP text ``last_hidden_state`` -> ``text_token_embeds``
  [N, 4, L, 512] (float16) + ``text_token_mask`` [N, 4, L] (1 = real token, 0 = pad;
  HF attention-mask convention). Enabled by default; ``--no-text-sequence`` skips it.
* image patches (optional, ``--image-patches``, storage-heavy): DINOv2 patch tokens
  -> ``image_patch_embeds`` [N, 4, P, 768] (float16). Off by default; the 4 view CLS
  tokens in ``image_embeds`` already serve as a short image sequence for the resampler.

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
def embed_image_patches(renders_dir: str, n_structures: int, device: str,
                        n_patches: int = 256) -> np.ndarray:
    """DINOv2 patch tokens per view (opt-in; storage-heavy).

    Returns ``[N, 4, n_patches, 768]`` float16 — the per-patch sequence a resampler can
    cross-attend for richer image conditioning than the 4 CLS tokens.
    """
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    out = np.zeros((n_structures, N_VIEWS, n_patches, 768), dtype=np.float16)
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
        patches = model(**inputs).last_hidden_state[:, 1:]      # drop CLS -> (B, P, 768)
        P = min(n_patches, patches.shape[1])
        for (i, v), e in zip(owners[start:start + BATCH], patches.float().cpu().numpy()):
            out[i, v, :P] = e[:P].astype(np.float16)
        if (start // BATCH) % 20 == 0:
            print(f"image-patches {start + len(imgs)}/{len(paths)}", flush=True)
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


def _load_text_encoder(encoder: str, device: str, model_name: str = None):
    """Return (processor, model, forward_fn) for CLIP or SigLIP.

    forward_fn(inputs) -> (last_hidden_state [B,L,H], attention_mask [B,L]). SigLIP
    (google/siglip-*) is a stronger text tower than CLIP ViT-B/32 for long/compositional
    captions (idea #1/#6, ideas.md); it pads to a fixed 64 tokens, so we derive the real
    length from non-zero token ids rather than an attention_mask it may not return.
    """
    from transformers import AutoModel, AutoProcessor
    if encoder == "clip":
        name = model_name or "openai/clip-vit-base-patch32"
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()

        def fwd(inputs):
            hs = model.text_model(**inputs).last_hidden_state
            return hs, inputs["attention_mask"]
        return proc, model, fwd
    if encoder == "siglip":
        name = model_name or "google/siglip-base-patch16-224"
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()

        def fwd(inputs):
            hs = model.text_model(**inputs).last_hidden_state
            am = inputs.get("attention_mask")
            if am is None:  # SigLIP pads to 64; treat non-pad token ids as real
                am = (inputs["input_ids"] != 1).long()
            return hs, am
        return proc, model, fwd
    raise ValueError(f"unknown text encoder {encoder!r} (clip|siglip)")


@torch.no_grad()
def embed_texts_sequence(captions: dict, n_structures: int, device: str,
                         n_captions: int = 4, encoder: str = "clip",
                         model_name: str = None):
    """Per-token text features + attention mask, for the #6 resampler.

    Returns ``(embeds [N, n_captions, L, H] float16, mask [N, n_captions, L] int8)``
    where ``L`` is the max real token length, ``mask`` is 1 for real tokens and 0 for
    padding (convert to a resampler key-padding mask with ``mask == 0``). ``encoder`` is
    ``clip`` (H=512) or ``siglip`` (H=768 for base). Unused caption slots are all-pad.
    """
    proc, _, fwd = _load_text_encoder(encoder, device, model_name)
    pad_kw = dict(padding="max_length", max_length=64) if encoder == "siglip" \
        else dict(padding=True, truncation=True, max_length=77)

    flat, owners = [], []
    for i in range(n_structures):
        for j, c in enumerate(captions.get(f"h{i:05d}", [])[:n_captions]):
            flat.append(c)
            owners.append((i, j))

    seqs, masks = [], []
    hidden = 512
    for start in range(0, len(flat), BATCH):
        inputs = proc(text=flat[start:start + BATCH], return_tensors="pt",
                      **pad_kw).to(device)
        hs, am = fwd(inputs)
        hidden = hs.shape[-1]
        for k in range(hs.shape[0]):
            length = max(1, int(am[k].sum()))
            seqs.append(hs[k, :length].float().cpu().numpy())
            masks.append(length)
        if (start // BATCH) % 20 == 0:
            print(f"text-seq[{encoder}] {start + hs.shape[0]}/{len(flat)}", flush=True)

    L = max(masks) if masks else 1
    embeds = np.zeros((n_structures, n_captions, L, hidden), dtype=np.float16)
    mask = np.zeros((n_structures, n_captions, L), dtype=np.int8)
    for (i, j), seq, length in zip(owners, seqs, masks):
        embeds[i, j, :length] = seq.astype(np.float16)
        mask[i, j, :length] = 1
    return embeds, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--cache", default=None,
                        help="structure cache .npz (e.g. data/minecraft/all_32.npz); "
                             "default: the houses_<max_dim> cache. Sets N and its order.")
    parser.add_argument("--text-encoder", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--text-model", default=None, help="override encoder checkpoint")
    parser.add_argument("--no-text-sequence", action="store_true",
                        help="skip the per-token text embeddings (idea #6 resampler)")
    parser.add_argument("--no-pooled", action="store_true",
                        help="skip the legacy pooled image/text embeds (seq-only run)")
    parser.add_argument("--image-patches", action="store_true",
                        help="also store DINOv2 patch tokens (storage-heavy)")
    args = parser.parse_args()

    if args.cache:
        from blockgen.curation.houses import load_structures_from_cache
        structures, _ = load_structures_from_cache(args.cache)
    else:
        from blockgen.curation.houses import load_house_structures
        structures, _ = load_house_structures(max_dim=args.max_dim)
    n = len(structures)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.captions) as f:
        captions = json.load(f)

    arrays = {}
    if not args.no_pooled:
        arrays["image_embeds"] = embed_images(args.renders, n, device)
        arrays["text_embeds"] = embed_texts(captions, n, device)
    if not args.no_text_sequence:
        tok, msk = embed_texts_sequence(captions, n, device, encoder=args.text_encoder,
                                        model_name=args.text_model)
        arrays["text_token_embeds"] = tok
        arrays["text_token_mask"] = msk
    if args.image_patches:
        arrays["image_patch_embeds"] = embed_image_patches(args.renders, n, device)

    np.savez_compressed(args.out, **arrays)
    shapes = ", ".join(f"{k} {v.shape}" for k, v in arrays.items())
    print(f"saved {shapes} -> {args.out}")


if __name__ == "__main__":
    main()
