"""The representations a build can be read in, by name.

Every track reads the same `Structure` and writes it back; what differs is the
intermediate. A `Representation` is that intermediate with its three verbs:
`encode` a build into it, `decode` it back into a build, and `picture` it for a
person. Code that wants to switch representations (a tokenizer kind, a lab
panel, a figure script) asks `get(name)` and uses the verbs, so adding a
representation is one entry here and nothing elsewhere changes.

`voxels` is the identity, pictured through the textured renderer; `facades` is
the 2.5D reading in `blockgen.corpus.facade`. The token representations in
`blockgen.utils.serialize` need a vocabulary to encode and are not registered
until they are rewritten around one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.utils.data import Structure


@dataclass(frozen=True)
class Representation:
    name: str
    doc: str
    encode: Callable[[Structure], Any]
    decode: Callable[[Any], Structure]
    #: `(encoded, px) -> RGB uint8`; `px` is the width the caller wants.
    picture: Callable[[Any, int], np.ndarray]
    #: True when `picture` draws through `blockgen.renderer`, which holds one
    #: EGL context and must be called from its own thread (the lab serialises
    #: those); False for pure numpy pictures.
    needs_renderer: bool = False


def _voxel_picture(s: Structure, px: int) -> np.ndarray:
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures
    img = np.asarray(render_structure(s, px=px, face_textures=load_face_textures()))
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def _facade_picture(faces: Dict[str, fc.Facade], px: int) -> np.ndarray:
    sides = sum(f.depth.shape[0] for f in faces.values() if f.axis != 1)
    return fc.sheet(faces, scale=max(1, px // max(sides + 5, 1)))


REPRESENTATIONS: Dict[str, Representation] = {
    "voxels": Representation(
        "voxels", "the block volume itself, cropped",
        encode=lambda s: s.crop_to_non_air(), decode=lambda s: s,
        picture=_voxel_picture, needs_renderer=True),
    "facades": Representation(
        "facades", "six face images of (depth, block); sides in walking order, "
                   "then roof plan and underside",
        encode=lambda s: fc.project_all(s.crop_to_non_air(), fc.ALL_FACES),
        decode=fc.lift, picture=_facade_picture),
}


def names() -> Tuple[str, ...]:
    return tuple(REPRESENTATIONS)


def get(name: str) -> Representation:
    try:
        return REPRESENTATIONS[name]
    except KeyError:
        raise KeyError(f"no representation {name!r}; have {names()}") from None


def picture(name: str, s: Structure, px: int = 512) -> np.ndarray:
    """Encode `s` and draw it, in one call."""
    r = get(name)
    return r.picture(r.encode(s), px)
