"""Track B: discrete (absorbing/masked) diffusion over a fixed voxel grid.

A small 3D-UNet operates on a ``grid^3`` volume of block-class indices (0 = air).
Training follows the MaskGIT / D3PM-absorbing recipe: a fraction of voxels are
replaced by a special ``MASK`` class and the network predicts their true class
via cross-entropy. Sampling starts from an all-``MASK`` grid and iteratively
unmasks the most confident predictions over a fixed number of steps.

This is the simplest discrete-diffusion formulation (no Gaussian noise on
categorical voxels) and gives a parallel, non-autoregressive counterpoint to the
token transformer (Track A) and the graph VAE (Track C).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _conv_block(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class VoxelUNet3D(nn.Module):
    """Compact 3D-UNet for masked-voxel class prediction.

    ``num_classes`` covers air + block classes. An extra embedding row is added
    for the ``MASK`` token (index ``num_classes``); the output layer predicts
    only over the real classes.
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 48,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mask_token = num_classes  # absorbing state

        self.embed = nn.Embedding(num_classes + 1, base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels)
        )
        self.time_dim = time_dim

        c = base_channels
        self.enc1 = _conv_block(c, c)
        self.down1 = nn.Conv3d(c, c * 2, 4, stride=2, padding=1)
        self.enc2 = _conv_block(c * 2, c * 2)
        self.down2 = nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1)

        self.mid = _conv_block(c * 4, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = _conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = _conv_block(c * 2, c)

        self.head = nn.Conv3d(c, num_classes, 1)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding of the scalar mask-ratio / timestep in [0,1].
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W) long class indices (may include mask_token). t: (B,) in [0,1].
        h = self.embed(x).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        temb = self.time_mlp(self._time_embedding(t))  # (B, C)
        h = h + temb[:, :, None, None, None]

        e1 = self.enc1(h)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))

        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # (B, num_classes, D, H, W)


def mask_schedule(ratio: torch.Tensor) -> torch.Tensor:
    """Training corruption level: fraction of voxels masked at sampled `ratio`.

    Monotonic in `ratio ~ U(0,1)`, covering (0, 1]. Used to draw a random
    masking fraction during training.
    """
    return torch.cos(0.5 * math.pi * (1.0 - ratio)).clamp(1e-3, 1.0)


def _keep_masked_fraction(progress: float) -> float:
    """Fraction of voxels to *keep masked* after reaching `progress` in [0,1].

    Starts near 1 (almost everything masked) and decays to 0 (fully revealed),
    so each MaskGIT step commits only the most confident few voxels.
    """
    return math.cos(0.5 * math.pi * progress)


@torch.no_grad()
def sample_grids(
    model: VoxelUNet3D,
    grid: int,
    num_samples: int,
    steps: int = 12,
    device: str = "cuda",
    temperature: float = 1.0,
    gumbel_scale: float = 1.0,
    air_bias: float = 0.0,
) -> torch.Tensor:
    """Iterative MaskGIT-style *stochastic* decoding from an all-MASK grid.

    At each step we sample a class per masked voxel from the predicted
    categorical (not argmax — argmax from an identical all-mask start gives
    identical, deterministic samples), then commit the most confident ones using
    Gumbel-perturbed log-probabilities (the standard MaskGIT confidence trick).
    The Gumbel scale is annealed to 0 over the schedule.

    ``air_bias`` subtracts from the air-class (index 0) logit at sampling time to
    directly control output density — raise it if samples come out near-empty,
    lower it (or go negative) if they come out as dense blobs.

    Returns a (num_samples, grid, grid, grid) long tensor of class indices.
    """
    model.eval()
    n_vox = grid ** 3
    x = torch.full((num_samples, grid, grid, grid), model.mask_token, dtype=torch.long, device=device)

    for step in range(steps):
        progress = (step + 1) / steps
        t = torch.full((num_samples,), 1.0 - progress, device=device)
        logits = model(x, t) / max(temperature, 1e-5)  # (B, C, D,H,W)
        if air_bias != 0.0:
            logits[:, 0] = logits[:, 0] - air_bias
        probs = torch.softmax(logits, dim=1)

        # Sample a class per voxel from the categorical distribution.
        flat_probs = probs.permute(0, 2, 3, 4, 1).reshape(-1, probs.size(1))
        sampled = torch.multinomial(flat_probs, 1).reshape(num_samples, grid, grid, grid)
        logp = torch.log(torch.gather(probs, 1, sampled.unsqueeze(1)).squeeze(1) + 1e-9)

        # Gumbel-perturbed confidence, annealed over steps.
        u = torch.rand_like(logp).clamp(1e-9, 1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        anneal = gumbel_scale * (1.0 - progress)
        conf = logp + anneal * gumbel

        keep_masked = 0 if step == steps - 1 else int(_keep_masked_fraction(progress) * n_vox)

        for b in range(num_samples):
            flat_conf = conf[b].reshape(-1).clone()
            flat_pred = sampled[b].reshape(-1)
            flat_x = x[b].reshape(-1)
            masked_pos = flat_x == model.mask_token
            n_masked = int(masked_pos.sum().item())
            n_to_unmask = n_masked - keep_masked
            if n_to_unmask <= 0:
                continue
            flat_conf[~masked_pos] = -float("inf")
            chosen = torch.topk(flat_conf, n_to_unmask).indices
            flat_x[chosen] = flat_pred[chosen]
            x[b] = flat_x.reshape(grid, grid, grid)

    x[x == model.mask_token] = 0  # any leftover masks -> air
    return x
