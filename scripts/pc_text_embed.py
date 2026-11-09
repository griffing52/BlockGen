"""Text embeddings for the Uni3D and ULIP-2 text-shape scores (notes §65).

Uni3D-B is aligned to OpenCLIP EVA02-E-14-plus (laion2b_s9b_b144k) and ULIP-2
to OpenCLIP ViT-bigG-14 (laion2b_s39b_b160k). Both checkpoints are ~10 GB, of
which the text tower is 1.4 GB (bf16) and 2.8 GB (fp32). This script reads the
safetensors header and fetches only the text tensors with HTTP range requests,
runs the standard OpenCLIP text transformer on the CPU (causal mask, GELU MLP,
EOT pooling, `text_projection`), writes the L2-normalised prompt embeddings to
data/external/weights/pc_text_emb.npz, and keeps no weights.

    .venv/bin/python scripts/pc_text_embed.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

OUT = Path("data/external/weights/pc_text_emb.npz")
MODELS = {"uni3d": ("timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k", "text."),
          "ulip2": ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "")}
PROMPTS = ["a 3D model of a building", "a 3D model of a house", "a 3D model of a chair",
           "a 3D model of a car", "a 3D model of an animal", "a 3D model of a tree"]
DT = {"F32": np.float32, "F16": np.float16, "BF16": None}


def fetch_text(repo: str, prefix: str):
    import requests
    import torch
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo, "open_clip_model.safetensors")
    n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
    hd = json.loads(requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=60).content)
    base = 8 + n
    sd = {}
    for k, v in hd.items():
        if k == "__metadata__" or not k.startswith(prefix) or k[len(prefix):].startswith("visual"):
            continue
        a, b = v["data_offsets"]
        buf = requests.get(url, headers={"Range": f"bytes={base + a}-{base + b - 1}"},
                           timeout=600).content
        assert len(buf) == b - a, (k, len(buf), b - a)
        if v["dtype"] == "BF16":
            t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).float()
        else:
            t = torch.from_numpy(np.frombuffer(buf, dtype=DT[v["dtype"]]).copy()).float()
        sd[k[len(prefix):]] = t.reshape(v["shape"])
    return sd


def encode(sd, tokens):
    import torch
    import torch.nn.functional as F
    x = sd["token_embedding.weight"][tokens] + sd["positional_embedding"]
    L = x.shape[1]
    mask = torch.full((L, L), float("-inf")).triu(1)
    layers = len({k.split(".")[2] for k in sd if k.startswith("transformer.resblocks.")})
    W = x.shape[-1]
    H = W // 64
    for i in range(layers):
        p = f"transformer.resblocks.{i}."
        h = F.layer_norm(x, (W,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"])
        qkv = h @ sd[p + "attn.in_proj_weight"].T + sd[p + "attn.in_proj_bias"]
        q, k, v = qkv.reshape(x.shape[0], L, 3, H, W // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).reshape(x.shape[0], L, W)
        x = x + a @ sd[p + "attn.out_proj.weight"].T + sd[p + "attn.out_proj.bias"]
        h = F.layer_norm(x, (W,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"])
        h = F.gelu(h @ sd[p + "mlp.c_fc.weight"].T + sd[p + "mlp.c_fc.bias"])
        x = x + h @ sd[p + "mlp.c_proj.weight"].T + sd[p + "mlp.c_proj.bias"]
    x = F.layer_norm(x, (W,), sd["ln_final.weight"], sd["ln_final.bias"])
    x = x[torch.arange(x.shape[0]), tokens.argmax(-1)] @ sd["text_projection"]
    return F.normalize(x, dim=-1).numpy()


def main() -> None:
    import torch
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    ids = tok(PROMPTS, padding="max_length", max_length=77, truncation=True)["input_ids"]
    tokens = torch.tensor(ids)
    out = {"prompts": np.array(PROMPTS)}
    with torch.no_grad():
        for name, (repo, prefix) in MODELS.items():
            sd = fetch_text(repo, prefix)
            out[name] = encode(sd, tokens)
            print(name, out[name].shape, np.round(out[name] @ out[name].T, 2)[0], flush=True)
            del sd
    np.savez(OUT, **out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
