"""What a block *looks like*, measured from the shipped textures.

The colour of oak planks is not a matter of opinion and it is not worth an API
call: ``data/textures/vanilla/`` holds the real 16x16 vanilla PNGs (fetched by
``python -m blockgen.renderer.textures --fetch``), and
``renderer.textures.resolve_face_names`` already maps a legacy ``(id, data)``
pair to the texture basenames the renderer draws. So every appearance attribute
here is measured from the same pixels the eval renders, which is the strongest
form of "the ontology and the picture agree" available to us.

Three attributes come out:

``primary_color``  mean RGB over non-transparent pixels of the *side* face, as
                   hex plus a short English descriptor. The side is what you see
                   in a wall; the top face of grass would say "green" about a
                   block used as brown ground. The renderer's foliage/water tints
                   are applied, so the catalog and the render agree.
``transparency``   ``opaque`` / ``cutout`` / ``translucent``, from the alpha
                   channel. The distinction is real for a builder: a cutout
                   (fence, ladder, pane) does not seal a wall.
``texture``        ``flat`` / ``subtle`` / ``busy``, from the per-channel
                   standard deviation. This is what separates concrete from
                   cobblestone at the same colour, and it is the attribute a
                   model most often gets wrong from the name alone.

Everything degrades to ``None`` when a texture is missing rather than guessing,
because a fabricated colour in a table labelled "measured" is worse than a gap: a
fresh clone has no ``data/textures/`` at all and must still build a catalog.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_PACK_DIR = Path("data/textures/vanilla")

#: Hue sectors, in degrees, longest name last so the table reads as a lookup.
_HUES = ((15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
         (200, "cyan"), (255, "blue"), (290, "purple"), (335, "pink"), (360, "red"))


@dataclass(frozen=True)
class Appearance:
    """Measured appearance of one block, plus the texture it was measured from."""

    hex_color: str
    rgb: Tuple[int, int, int]
    color_name: str
    transparency: str
    texture: str
    texture_name: str = ""

    def to_attrs(self) -> Dict[str, object]:
        return {"appearance.color": f"{self.hex_color} {self.color_name}",
                "appearance.transparency": self.transparency,
                "appearance.texture": self.texture}


def color_name(rgb: Tuple[int, int, int]) -> str:
    """A short English description of an RGB triple.

    Hue/saturation/value rules rather than nearest-neighbour against a named
    palette: a fixed palette forces every brown into "olive" or "maroon", while
    "pale warm grey" and "dark brown" are the words a builder actually uses and
    they degrade gracefully for colours no palette anticipated.
    """
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360
    # Below this the hue is real but nobody sees it: black wool is #15151a and
    # black concrete #080a0f, and calling either "dark blue" is worse than useless
    # in a table a model uses to match materials.
    if v < 0.12:
        return "near-black"
    if s < 0.10:
        if v > 0.86:
            return "white"
        if v > 0.62:
            return "light grey"
        if v > 0.32:
            return "grey"
        return "near-black"
    base = next(name for edge, name in _HUES if hue < edge)
    # Warm hues read as brown/tan rather than orange/red unless they are both
    # saturated and bright: spruce log is #3b2611, which is "dark brown" to every
    # builder alive and "dark orange" only to a hue lookup.
    if base in ("orange", "red", "yellow") and (s < 0.55 or v < 0.45):
        base = "tan" if v > 0.62 else "brown"
    if v < 0.28:
        return f"dark {base}"
    if v > 0.85 and s < 0.35:
        return f"pale {base}"
    return base


@lru_cache(maxsize=1024)
def _load(name: str, pack_dir: str) -> Optional[np.ndarray]:
    path = Path(pack_dir) / f"{name}.png"
    if not path.is_file():
        return None
    try:
        from PIL import Image

        image = Image.open(path).convert("RGBA")
        # Animated textures (water, fire) are a vertical strip of frames; the
        # first frame is the representative one, and averaging the strip would
        # mix in frames a still render never shows.
        width, height = image.size
        if height > width and height % width == 0:
            image = image.crop((0, 0, width, width))
        return np.asarray(image, dtype=np.uint8)
    except Exception:                                    # noqa: BLE001
        return None


def measure(block_id: int, block_data: int,
            pack_dir: Path | str = DEFAULT_PACK_DIR) -> Optional[Appearance]:
    """Appearance of one legacy block, or ``None`` if its texture is not on disk."""
    from blockgen.renderer.textures import TINTS, resolve_face_names

    faces = resolve_face_names(int(block_id), int(block_data))
    if not faces:
        return None
    name = faces.get("side") or faces.get("top") or faces.get("bottom")
    if not name:
        return None
    pixels = _load(name, str(pack_dir))
    if pixels is None:
        return None

    rgb, alpha = pixels[..., :3].astype(np.float32), pixels[..., 3]
    visible = alpha > 16
    if not visible.any():
        return None
    # Grass, leaves and water ship greyscale and are tinted at draw time. Reading
    # the raw pixels would report oak leaves as "grey", contradicting every render
    # in the repo -- so apply the renderer's own multipliers.
    tint = TINTS.get(name)
    if tint:
        rgb = rgb * np.asarray(tint, dtype=np.float32)
    mean = rgb[visible].mean(axis=0).clip(0, 255)
    triple = tuple(int(round(float(c))) for c in mean)

    zero_frac = float((alpha <= 16).mean())
    solid_alpha = float(alpha[visible].mean())
    if solid_alpha < 250:
        transparency = "translucent"
    elif zero_frac > 0.02:
        transparency = "cutout"
    else:
        transparency = "opaque"

    spread = float(rgb[visible].std(axis=0).mean())
    texture = "flat" if spread < 6 else ("subtle" if spread < 16 else "busy")
    return Appearance(hex_color="#%02x%02x%02x" % triple, rgb=triple,
                      color_name=color_name(triple), transparency=transparency,
                      texture=texture, texture_name=name)


def texture_path(block_id: int, block_data: int,
                 pack_dir: Path | str = DEFAULT_PACK_DIR) -> Optional[Path]:
    """Where the side texture lives, for anything that wants to show the pixels
    (the lab serves these as swatches)."""
    from blockgen.renderer.textures import resolve_face_names

    faces = resolve_face_names(int(block_id), int(block_data)) or {}
    name = faces.get("side") or faces.get("top") or faces.get("bottom")
    if not name:
        return None
    path = Path(pack_dir) / f"{name}.png"
    return path if path.is_file() else None


__all__ = ["Appearance", "DEFAULT_PACK_DIR", "color_name", "measure", "texture_path"]
