"""Track B training: masked discrete diffusion over the fixed voxel grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from blockgen.models.voxel_diffusion import VoxelUNet3D, sample_grids, mask_schedule
from blockgen.utils.data import Structure
from blockgen.utils.serialize import (
    BlockVocab,
    grid_to_structure,
    num_grid_classes,
    structure_to_grid,
)


@dataclass
class DiffusionTrainConfig:
    grid: int = 24
    base_channels: int = 48
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda"
    sample_steps: int = 18
    log_every: int = 5
    # Down-weight the dominant air class. Voxel grids are ~99% air, so this must
    # be small to avoid collapsing to all-air; ~0.05 calibrates occupancy on the
    # small subset. Tune up if samples are too sparse, down if too dense.
    air_weight: float = 0.05


def build_grids(structures: List[Structure], vocab: BlockVocab, grid: int) -> np.ndarray:
    arr = np.stack([structure_to_grid(s, grid, vocab) for s in structures], axis=0)
    return arr  # (N, grid, grid, grid) int64


def train_diffusion(
    structures: List[Structure],
    vocab: BlockVocab,
    config: Optional[DiffusionTrainConfig] = None,
):
    """Train the masked-diffusion 3D-UNet. Returns (model, history)."""
    config = config or DiffusionTrainConfig()
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"

    num_classes = num_grid_classes(vocab)
    grids = build_grids(structures, vocab, config.grid)
    data = torch.tensor(grids, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(data), batch_size=config.batch_size, shuffle=True
    )

    model = VoxelUNet3D(num_classes=num_classes, base_channels=config.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history = {"loss": []}

    # Down-weight air (class 0); voxel grids are very sparse, so an unweighted
    # objective collapses to predicting all-air.
    class_weights = torch.ones(num_classes, device=device)
    class_weights[0] = config.air_weight

    model.train()
    for epoch in range(config.epochs):
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)  # (B, G,G,G)
            bsz = batch.size(0)

            # Sample a per-example mask ratio and build the corrupted input.
            ratio = torch.rand(bsz, device=device)
            keep_mask_frac = mask_schedule(ratio)  # (B,)
            rand = torch.rand_like(batch.float())
            mask = rand < keep_mask_frac[:, None, None, None]
            corrupted = batch.clone()
            corrupted[mask] = model.mask_token

            logits = model(corrupted, t=1.0 - ratio)  # (B, C, G,G,G)
            # Loss only on masked positions, with air down-weighted.
            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 4, 1).reshape(-1, num_classes),
                batch.reshape(-1),
                weight=class_weights,
                reduction="none",
            )
            mask_flat = mask.reshape(-1).float()
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item() * bsz
            n += bsz

        avg = total / max(n, 1)
        history["loss"].append(avg)
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            print(f"[DIFF] epoch {epoch:3d}  loss {avg:.4f}  (n={len(structures)})")

    return model, history


@torch.no_grad()
def calibrate_air_bias(
    model: VoxelUNet3D,
    vocab: BlockVocab,
    target_occupancy: float,
    grid: int = 24,
    steps: int = 18,
    candidates: Optional[List[float]] = None,
    probe_samples: int = 6,
    device: str = "cuda",
) -> float:
    """Pick the ``air_bias`` whose median sampled occupancy best matches a target.

    Diffusion occupancy is sensitive to training randomness, so rather than hard-code
    a bias we sample a few structures at several biases (cheap) and choose the one
    closest to the training median block count. Returns the chosen bias.
    """
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    if candidates is None:
        candidates = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    best_bias, best_err = 0.0, float("inf")
    for bias in candidates:
        grids = sample_grids(
            model, grid, probe_samples, steps=steps, device=device, air_bias=bias
        )
        occ = [(g > 0).sum().item() for g in grids]
        med = float(np.median(occ))
        err = abs(med - target_occupancy)
        if err < best_err:
            best_err, best_bias = err, bias
    return best_bias


@torch.no_grad()
def sample_structures(
    model: VoxelUNet3D,
    vocab: BlockVocab,
    num_samples: int = 16,
    grid: int = 24,
    steps: int = 12,
    temperature: float = 1.0,
    air_bias: float = 0.0,
    device: str = "cuda",
) -> List[Structure]:
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    grids = sample_grids(
        model, grid, num_samples, steps=steps, device=device,
        temperature=temperature, air_bias=air_bias,
    )
    return [grid_to_structure(g.cpu().numpy(), vocab) for g in grids]
