"""Two-stage factored generation: occupancy first, materials second.

The Scaffold-Diffusion recipe (research.md §B.1) adapted to our 3D-UNet stack:

  * **Stage 1 (shape)** — masked discrete diffusion over a *binary* grid
    (air / occupied). A binary target removes the many-way material competition
    that scrambles block types, and the class imbalance is only 2-way.
  * **Stage 2 (materials)** — masked discrete diffusion over block classes where
    ONLY occupied voxels are ever masked; air stays visible as class 0 everywhere.
    The material model therefore always sees the full occupancy footprint and
    solves a local labeling problem ("walls get wall blocks") instead of jointly
    inventing shape + palette. Loss is computed on masked occupied voxels only,
    so there is no air-class imbalance pressure at all.

Scaffold showed the material half on ground-truth occupancy; generating the
occupancy too (stage 1) is their stated future work — both stages here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from blockgen.models.voxel_diffusion import VoxelUNet3D, mask_schedule, sample_grids
from blockgen.training.train_diffusion import build_grids
from blockgen.utils.data import Structure
from blockgen.utils.serialize import BlockVocab, grid_to_structure, num_grid_classes


@dataclass
class TwoStageConfig:
    grid: int = 24
    base_channels: int = 48
    epochs_occ: int = 60
    epochs_mat: int = 60
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda"
    sample_steps: int = 18
    log_every: int = 5
    air_weight: float = 0.05     # stage-1 air down-weighting (2-class)


def train_occupancy(structures: List[Structure], vocab: BlockVocab,
                    config: TwoStageConfig) -> Tuple[VoxelUNet3D, dict]:
    """Stage 1: binary occupancy diffusion (classes: 0=air, 1=occupied)."""
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    grids = build_grids(structures, vocab, config.grid)
    data = torch.tensor((grids > 0).astype(np.int64))
    loader = DataLoader(TensorDataset(data), batch_size=config.batch_size, shuffle=True)

    model = VoxelUNet3D(num_classes=2, base_channels=config.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    w = torch.tensor([config.air_weight, 1.0], device=device)
    history = {"loss": []}
    model.train()
    for epoch in range(config.epochs_occ):
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            ratio = torch.rand(batch.size(0), device=device)
            mask = torch.rand_like(batch.float()) < mask_schedule(ratio)[:, None, None, None]
            corrupted = batch.clone()
            corrupted[mask] = model.mask_token
            logits = model(corrupted, t=1.0 - ratio)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 4, 1).reshape(-1, 2),
                                   batch.reshape(-1), weight=w, reduction="none")
            mf = mask.reshape(-1).float()
            loss = (loss * mf).sum() / mf.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * batch.size(0); n += batch.size(0)
        history["loss"].append(total / max(n, 1))
        if epoch % config.log_every == 0 or epoch == config.epochs_occ - 1:
            print(f"[OCC] epoch {epoch:3d}  loss {history['loss'][-1]:.4f}", flush=True)
    return model, history


def train_materials(structures: List[Structure], vocab: BlockVocab,
                    config: TwoStageConfig) -> Tuple[VoxelUNet3D, dict]:
    """Stage 2: material diffusion clamped to the occupancy footprint."""
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    num_classes = num_grid_classes(vocab)
    grids = build_grids(structures, vocab, config.grid)
    data = torch.tensor(grids, dtype=torch.long)
    loader = DataLoader(TensorDataset(data), batch_size=config.batch_size, shuffle=True)

    model = VoxelUNet3D(num_classes=num_classes, base_channels=config.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history = {"loss": []}
    model.train()
    for epoch in range(config.epochs_mat):
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            occ = batch > 0
            ratio = torch.rand(batch.size(0), device=device)
            # mask only occupied voxels; air is always visible context
            mask = (torch.rand_like(batch.float()) < mask_schedule(ratio)[:, None, None, None]) & occ
            corrupted = batch.clone()
            corrupted[mask] = model.mask_token
            logits = model(corrupted, t=1.0 - ratio)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 4, 1).reshape(-1, num_classes),
                                   batch.reshape(-1), reduction="none")
            mf = mask.reshape(-1).float()
            loss = (loss * mf).sum() / mf.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * batch.size(0); n += batch.size(0)
        history["loss"].append(total / max(n, 1))
        if epoch % config.log_every == 0 or epoch == config.epochs_mat - 1:
            print(f"[MAT] epoch {epoch:3d}  loss {history['loss'][-1]:.4f}", flush=True)
    return model, history


@torch.no_grad()
def sample_twostage(occ_model: VoxelUNet3D, mat_model: VoxelUNet3D, vocab: BlockVocab,
                    grid: int, num_samples: int, steps: int = 18,
                    occ_air_bias: float = 0.0, device: str = "cuda",
                    temperature: float = 1.0) -> List[Structure]:
    """Sample occupancy (stage 1), then paint materials onto it (stage 2)."""
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    occ = sample_grids(occ_model, grid, num_samples, steps=steps, device=device,
                       temperature=temperature, air_bias=occ_air_bias)  # (B,G,G,G) in {0,1}

    mat_model.eval()
    x = torch.where(occ > 0,
                    torch.full_like(occ, mat_model.mask_token),
                    torch.zeros_like(occ))
    n_occ_max = int((occ > 0).reshape(num_samples, -1).sum(1).max().item())
    from blockgen.models.voxel_diffusion import _keep_masked_fraction
    for step in range(steps):
        progress = (step + 1) / steps
        t = torch.full((num_samples,), 1.0 - progress, device=device)
        logits = mat_model(x, t) / max(temperature, 1e-5)
        logits[:, 0] = -1e9  # air is not a material choice; occupancy is fixed
        probs = torch.softmax(logits, dim=1)
        flat_probs = probs.permute(0, 2, 3, 4, 1).reshape(-1, probs.size(1))
        sampled = torch.multinomial(flat_probs, 1).reshape(num_samples, grid, grid, grid)
        logp = torch.log(torch.gather(probs, 1, sampled.unsqueeze(1)).squeeze(1) + 1e-9)
        u = torch.rand_like(logp).clamp(1e-9, 1 - 1e-9)
        conf = logp + (1.0 - progress) * (-torch.log(-torch.log(u)))

        keep_masked = 0 if step == steps - 1 else int(_keep_masked_fraction(progress) * n_occ_max)
        for b in range(num_samples):
            flat_conf = conf[b].reshape(-1).clone()
            flat_pred = sampled[b].reshape(-1)
            flat_x = x[b].reshape(-1)
            masked_pos = flat_x == mat_model.mask_token
            n_to_unmask = int(masked_pos.sum().item()) - keep_masked
            if n_to_unmask <= 0:
                continue
            flat_conf[~masked_pos] = -float("inf")
            chosen = torch.topk(flat_conf, n_to_unmask).indices
            flat_x[chosen] = flat_pred[chosen]
            x[b] = flat_x.reshape(grid, grid, grid)

    x[x == mat_model.mask_token] = 0
    return [grid_to_structure(g.cpu().numpy(), vocab) for g in x]
