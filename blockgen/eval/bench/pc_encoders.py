"""Point-cloud foundation encoders and EMD for the 3D-native metric rows.

Two CLIP-aligned point encoders, loaded from their released checkpoints and run
in pure torch (no pointnet2_ops / CUDA extensions, so they run on sm_120):

    uni3d   Uni3D-B (Zhou et al. 2023, arXiv 2310.06773): EVA02-B transformer
            over 512 FPS groups of 64 kNN points, xyz+rgb, 10k points, output
            the 1,024-d embedding aligned to EVA02-E/14-plus CLIP.
            Checkpoint BAAI/Uni3D modelzoo/uni3d-b/model.pt (178 MB).
    ulip2   ULIP-2 PointBERT (Xue et al. 2023, arXiv 2305.08275): 18-layer,
            384-wide transformer over 512 FPS groups of 32 kNN points, xyz+rgb,
            10k points, [cls, max-pool] through the 768x1280 projection into
            OpenCLIP ViT-bigG/14 space. Checkpoint SFXX/ulip ULIP-2/pretrained_models/
            ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt.

Preprocessing follows each repo's evaluation path: centre on the centroid and
scale to the unit sphere (`pc_normalize`), rgb in [0, 1]; Uni3D's OpenShape
setting swaps y and z (its training data is z-up; ours is y-up), ULIP-2 keeps
the native frame. FPS is the exact greedy farthest-point sampler (first index
0, as pointnet2_ops), kNN is `topk` on squared distances, as both repos.

EMD is exact: equal-size clouds, optimal one-to-one assignment
(`scipy.optimize.linear_sum_assignment`) on Euclidean distance, reported as the
mean matched distance (the PointFlow definition; PointFlow's own CUDA kernel is
an auction approximation of the same quantity). Exact assignment costs
O(k^3), so the clouds are 512-point subsamples; at 2,048 points neither the
exact solver nor a converged Sinkhorn fits the shared GPU's budget for the
~10^6 pairs a ladder needs (benchmarked: Sinkhorn 2,048 at 50 eps-scaling
iterations is 47 ms/pair and still leaves marginals off by 30-50%).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

WEIGHTS = Path("data/external/weights")
UNI3D_CKPT = WEIGHTS / "uni3d/modelzoo/uni3d-b/model.pt"
ULIP2_CKPT = WEIGHTS / ("ulip2/ULIP-2/pretrained_models/"
                        "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
#: timm is installed out of tree (`pip --target`) so the project venv is untouched.
PYLIBS = Path("data/external/pylibs")
NPOINTS = 10_000


def _timm():
    try:
        import timm  # noqa: F401
    except ImportError:
        sys.path.append(str(PYLIBS.resolve()))
        import timm  # noqa: F401
    return sys.modules["timm"]


# --- grouping (pure torch) ------------------------------------------------------
def fps(xyz, m: int):
    """`(B, m)` indices, greedy farthest-point sampling starting at index 0."""
    import torch
    B, N, _ = xyz.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        idx[:, i] = far
        c = xyz[ar, far][:, None, :]
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(1)
    return idx


def group(xyz, rgb, num_group: int, group_size: int):
    """`(neigh_xyz - centre, neigh_rgb, centre)` as Point-BERT/Uni3D `Group`."""
    import torch
    B = xyz.shape[0]
    ar = torch.arange(B, device=xyz.device)[:, None]
    centre = xyz[ar, fps(xyz, num_group)]                               # B G 3
    d = torch.cdist(centre, xyz) ** 2                                   # B G N
    nn = d.topk(group_size, dim=-1, largest=False, sorted=False).indices
    nb = xyz[ar[:, :, None], nn]                                        # B G M 3
    nc = rgb[ar[:, :, None], nn]
    return nb - centre[:, :, None], nc, centre


class _MiniPointNet:
    """Point-BERT's two-stage shared MLP + max-pool group encoder."""

    @staticmethod
    def build(out: int):
        import torch.nn as nn

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.first_conv = nn.Sequential(nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128),
                                                nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
                self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512),
                                                 nn.ReLU(inplace=True), nn.Conv1d(512, out, 1))

            def forward(self, g):                                       # B G M 6
                import torch
                bs, G, M, c = g.shape
                g = g.reshape(bs * G, M, c).transpose(2, 1)
                f = self.first_conv(g)
                f = torch.cat([f.max(2, keepdim=True)[0].expand(-1, -1, M), f], 1)
                f = self.second_conv(f).max(2)[0]
                return f.reshape(bs, G, -1)
        return Enc()


def _pc_normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz - xyz.mean(0, keepdims=True)
    return xyz / max(float(np.sqrt((xyz ** 2).sum(1)).max()), 1e-9)


# --- Uni3D --------------------------------------------------------------------------
class Uni3D:
    name = "uni3d"
    dim = 1024

    def __init__(self, device: str = "cuda", ckpt: Path = UNI3D_CKPT):
        import torch
        import torch.nn as nn
        timm = _timm()
        self.device = device
        visual = timm.create_model("eva02_base_patch14_448", pretrained=False,
                                   drop_path_rate=0.0)

        class Enc(nn.Module):
            def __init__(s):
                super().__init__()
                s.encoder = _MiniPointNet.build(512)
                s.encoder2trans = nn.Linear(512, 768)
                s.trans2embed = nn.Linear(768, 1024)
                s.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
                s.cls_pos = nn.Parameter(torch.zeros(1, 1, 768))
                s.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 768))
                s.visual = visual

        self.m = Enc()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["module"]
        sd = {k[len("point_encoder."):]: v for k, v in sd.items()
              if k.startswith("point_encoder.")}
        missing, unexpected = self.m.load_state_dict(sd, strict=False)
        # timm's EVA carries rope / patch-embed buffers Uni3D never uses
        bad = [k for k in missing if not k.startswith("visual.")
               or any(t in k for t in ("blocks.", "fc_norm", "cls_token"))]
        if bad:
            raise RuntimeError(f"Uni3D load: missing {bad[:8]}")
        self.m.to(device).eval()

    def prep(self, xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        p = _pc_normalize(xyz.astype(np.float32))[:, [0, 2, 1]]       # y-up -> z-up
        return np.concatenate([p, rgb.astype(np.float32)], 1)

    def forward(self, pc):
        import torch
        e = self.m
        nb, nc, centre = group(pc[..., :3].contiguous(), pc[..., 3:].contiguous(), 512, 64)
        tok = e.encoder2trans(e.encoder(torch.cat([nb, nc], -1)))
        B = tok.shape[0]
        x = torch.cat([e.cls_token.expand(B, -1, -1), tok], 1)
        pos = torch.cat([e.cls_pos.expand(B, -1, -1), e.pos_embed(centre)], 1)
        x = e.visual.pos_drop(x + pos)
        for blk in e.visual.blocks:
            x = blk(x)
        x = e.visual.fc_norm(e.visual.norm(x[:, 0, :]))
        return e.trans2embed(x)


# --- ULIP-2 PointBERT -------------------------------------------------------------
class ULIP2:
    name = "ulip2"
    dim = 1280

    def __init__(self, device: str = "cuda", ckpt: Path = ULIP2_CKPT):
        import torch
        import torch.nn as nn
        self.device = device

        class Attn(nn.Module):
            def __init__(s, dim, heads):
                super().__init__()
                s.h = heads
                s.qkv = nn.Linear(dim, dim * 3, bias=False)
                s.proj = nn.Linear(dim, dim)

            def forward(s, x):
                B, N, C = x.shape
                q, k, v = s.qkv(x).reshape(B, N, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return s.proj(x.transpose(1, 2).reshape(B, N, C))

        class Block(nn.Module):
            def __init__(s, dim, heads):
                super().__init__()
                s.norm1, s.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
                s.attn = Attn(dim, heads)
                s.mlp = nn.Sequential()
                s.mlp.fc1 = nn.Linear(dim, dim * 4)
                s.mlp.act = nn.GELU()
                s.mlp.fc2 = nn.Linear(dim * 4, dim)

            def forward(s, x):
                x = x + s.attn(s.norm1(x))
                return x + s.mlp(s.norm2(x))

        class Enc(nn.Module):
            def __init__(s):
                super().__init__()
                s.encoder = _MiniPointNet.build(256)
                s.reduce_dim = nn.Linear(256, 384)
                s.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
                s.cls_pos = nn.Parameter(torch.zeros(1, 1, 384))
                s.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 384))
                s.blocks = nn.Module()
                s.blocks.blocks = nn.ModuleList([Block(384, 6) for _ in range(18)])
                s.norm = nn.LayerNorm(384)

        self.m = Enc()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        self.proj = sd["module.pc_projection"].float().to(device)
        sd = {k[len("module.point_encoder."):]: v for k, v in sd.items()
              if k.startswith("module.point_encoder.")}
        self.m.load_state_dict(sd, strict=True)
        self.m.to(device).eval()

    def prep(self, xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        return np.concatenate([_pc_normalize(xyz.astype(np.float32)),
                               rgb.astype(np.float32)], 1)

    def forward(self, pc):
        import torch
        e = self.m
        nb, nc, centre = group(pc[..., :3].contiguous(), pc[..., 3:].contiguous(), 512, 32)
        tok = e.reduce_dim(e.encoder(torch.cat([nb, nc], -1)))
        B = tok.shape[0]
        x = torch.cat([e.cls_token.expand(B, -1, -1), tok], 1)
        pos = torch.cat([e.cls_pos.expand(B, -1, -1), e.pos_embed(centre)], 1)
        for blk in e.blocks.blocks:
            x = blk(x + pos)
        x = e.norm(x)
        return torch.cat([x[:, 0], x[:, 1:].max(1)[0]], -1) @ self.proj


ENCODERS = {"uni3d": Uni3D, "ulip2": ULIP2}


def embed(enc, clouds: Sequence[np.ndarray], batch: int = 4) -> np.ndarray:
    """`(N, dim)` L2-normalised embeddings of `(k, 6)` xyz+rgb clouds."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(clouds), batch):
            pc = np.stack([enc.prep(c[:, :3], c[:, 3:6]) for c in clouds[i:i + batch]])
            z = enc.forward(torch.as_tensor(pc, device=enc.device)).float()
            out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# --- EMD -----------------------------------------------------------------------------
_CLOUDS = None


def _emd_init(clouds):
    global _CLOUDS
    _CLOUDS = clouds


def _emd_rows(pairs):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    clouds = _CLOUDS
    out = np.empty(len(pairs), np.float32)
    for t, (i, j) in enumerate(pairs):
        D = cdist(clouds[i], clouds[j])
        r, c = linear_sum_assignment(D)
        out[t] = D[r, c].mean()
    return out


def emd_matrix(clouds: np.ndarray, need: Optional[np.ndarray] = None, workers: int = 16,
               chunk: int = 400) -> np.ndarray:
    """`(N, N)` exact EMD (mean matched Euclidean distance) between equal-size
    clouds. `need` is an `(N, N)` bool mask of the entries the set statistics
    will read (symmetrised here); others are NaN, so a statistic that reads one
    it did not ask for fails loudly instead of silently using a zero."""
    from concurrent.futures import ProcessPoolExecutor
    N = len(clouds)
    need = np.ones((N, N), bool) if need is None else (need | need.T)
    iu = np.argwhere(np.triu(need, 1))
    out = np.full((N, N), np.nan, np.float32)
    np.fill_diagonal(out, 0.0)
    jobs = [iu[s:s + chunk] for s in range(0, len(iu), chunk)]
    with ProcessPoolExecutor(workers, initializer=_emd_init,
                             initargs=(np.asarray(clouds, np.float64),)) as ex:
        for pr, vals in zip(jobs, ex.map(_emd_rows, jobs)):
            out[pr[:, 0], pr[:, 1]] = vals
            out[pr[:, 1], pr[:, 0]] = vals
    return out


def needed_pairs(N: int, ref: np.ndarray, sets: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """Entries `pointcloud.mmd/one_minus_cov/one_nna` read for every set in
    `sets` (rungs and the null pool) against `ref`: ref x set, set x set, and
    the first `n` reference shapes among themselves."""
    m = np.zeros((N, N), bool)
    r = np.asarray(ref).reshape(-1)
    for s in sets.values():
        s = np.asarray(s).reshape(-1)
        m[np.ix_(r, s)] = True
        m[np.ix_(s, s)] = True
    m[np.ix_(r[:n], r[:n])] = True
    return m
