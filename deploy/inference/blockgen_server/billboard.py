"""Dream image -> a wall of solid blocks next to the build.

The dream recipes start by *imagining* the house (one generated image) and then try
to build what they imagined. Showing that image in the world is what makes the loop
legible: you see the target, then watch each attempt get closer to it. A map item
would be sharper but needs custom map state in the mod; a block wall needs nothing
new on the client, because it is just more blocks.

Colours are matched against the mean texture colour of full, untextured-looking
blocks (concrete, wool, terracotta) plus a few common building blocks. The image
background (estimated from its border) is dropped, so the dream reads as a cut-out
rather than a white rectangle.
"""

from __future__ import annotations

import io
from typing import Dict, Sequence, Tuple

import numpy as np

Pos = Tuple[int, int, int]

# Approximate mean texture colours (sRGB). Exactness does not matter here: the
# wall is a 40-block-wide preview, not map art.
PALETTE: Dict[str, Tuple[int, int, int]] = {
    # concrete
    "white_concrete": (207, 213, 214), "orange_concrete": (224, 97, 0),
    "magenta_concrete": (169, 48, 159), "light_blue_concrete": (35, 137, 198),
    "yellow_concrete": (241, 175, 21), "lime_concrete": (94, 169, 24),
    "pink_concrete": (213, 101, 142), "gray_concrete": (54, 57, 61),
    "light_gray_concrete": (125, 125, 115), "cyan_concrete": (21, 119, 136),
    "purple_concrete": (100, 31, 156), "blue_concrete": (44, 46, 143),
    "brown_concrete": (96, 59, 31), "green_concrete": (73, 91, 36),
    "red_concrete": (142, 32, 32), "black_concrete": (8, 10, 15),
    # wool
    "white_wool": (233, 236, 236), "orange_wool": (240, 118, 19),
    "magenta_wool": (189, 68, 179), "light_blue_wool": (58, 175, 217),
    "yellow_wool": (248, 197, 39), "lime_wool": (112, 185, 25),
    "pink_wool": (237, 141, 172), "gray_wool": (62, 68, 71),
    "light_gray_wool": (142, 142, 134), "cyan_wool": (21, 137, 145),
    "purple_wool": (121, 42, 172), "blue_wool": (53, 57, 157),
    "brown_wool": (114, 71, 40), "green_wool": (84, 109, 27),
    "red_wool": (160, 39, 34), "black_wool": (20, 21, 25),
    # terracotta
    "terracotta": (152, 94, 67), "white_terracotta": (209, 178, 161),
    "orange_terracotta": (161, 83, 37), "magenta_terracotta": (149, 88, 108),
    "light_blue_terracotta": (113, 108, 137), "yellow_terracotta": (186, 133, 35),
    "lime_terracotta": (103, 117, 52), "pink_terracotta": (161, 78, 78),
    "gray_terracotta": (57, 42, 35), "light_gray_terracotta": (135, 106, 97),
    "cyan_terracotta": (86, 91, 91), "purple_terracotta": (118, 70, 86),
    "blue_terracotta": (74, 59, 91), "brown_terracotta": (77, 51, 35),
    "green_terracotta": (76, 83, 42), "red_terracotta": (143, 61, 46),
    "black_terracotta": (37, 22, 16),
    # building blocks, so wood and stone read as wood and stone
    "oak_planks": (162, 130, 78), "spruce_planks": (114, 84, 48),
    "birch_planks": (192, 175, 121), "dark_oak_planks": (66, 43, 20),
    "stone": (125, 125, 125), "stone_bricks": (122, 121, 122),
    "sandstone": (216, 203, 155), "bricks": (150, 97, 83),
    "quartz_block": (235, 229, 222), "snow_block": (249, 254, 254),
}
_NAMES = list(PALETTE)
_RGB = np.array([PALETTE[n] for n in _NAMES], dtype=np.float32)


def _nearest(rgb: np.ndarray) -> np.ndarray:
    """Index of the nearest palette colour per pixel ("redmean" weighted RGB)."""
    p = rgb.reshape(-1, 1, 3).astype(np.float32)
    rmean = (p[..., 0] + _RGB[None, :, 0]) / 2.0
    d = p - _RGB[None]
    dist = ((2 + rmean / 256) * d[..., 0] ** 2 + 4 * d[..., 1] ** 2
            + (2 + (255 - rmean) / 256) * d[..., 2] ** 2)
    return dist.argmin(axis=1).reshape(rgb.shape[:2])


def billboard_blocks(png: bytes, canvas: Sequence[int], *, size: int = 40,
                     gap: int = 3, bg_tol: float = 40.0) -> Dict[Pos, str]:
    """Blocks for a ``size``-wide picture of ``png``, standing on the build's north side.

    The wall is the x-y plane at ``z = -gap - 1``, centred on the canvas in x, with
    image row 0 at the top. Seen from the build (looking north) it reads the right
    way round. Pixels within ``bg_tol`` of the border colour are left as air.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    sw = max(1, int(size))
    sh = max(1, round(sw * h / w))
    small = np.asarray(img.resize((sw, sh), Image.LANCZOS), dtype=np.float32)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    bg = np.median(border, axis=0)
    keep = np.linalg.norm(small - bg, axis=2) > bg_tol
    idx = _nearest(small)
    x0 = int(canvas[0]) // 2 - sw // 2
    z = -int(gap) - 1
    out: Dict[Pos, str] = {}
    for r in range(sh):
        for c in range(sw):
            if keep[r, c]:
                out[(x0 + c, sh - 1 - r, z)] = "minecraft:" + _NAMES[idx[r, c]]
    return out
