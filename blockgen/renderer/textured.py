"""Headless textured rendering of Structures with real Minecraft textures.

Builds an exposed-face quad mesh with a stitched texture atlas (Minecraft-style
face shading baked into pre-darkened atlas tiles) and renders it offscreen via
pyrender/EGL with flat (unlit) shading and an orthographic camera.

Usage:
    python -m blockgen.renderer.textured \
        --cache data/cache/gc_small_32.npz --index 0 \
        --out outputs/figures/test_textured.png
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import atexit
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyrender
from PIL import Image

from blockgen.renderer.textures import DEFAULT_PACK_DIR, FaceTextureLibrary, load_face_textures
from blockgen.utils.data import Structure

GL_NEAREST = 9728
GL_CLAMP_TO_EDGE = 33071

# Minecraft-style directional face shading, baked into atlas tiles.
# (name, neighbor offset, face texture kind, brightness)
_DIRECTIONS: List[Tuple[str, Tuple[int, int, int], str, float]] = [
    ("top", (0, 1, 0), "top", 1.0),
    ("bottom", (0, -1, 0), "bottom", 0.55),
    ("north", (0, 0, -1), "side", 0.85),
    ("south", (0, 0, 1), "side", 0.85),
    ("west", (-1, 0, 0), "side", 0.70),
    ("east", (1, 0, 0), "side", 0.70),
]

# Quad corners (voxel-relative, CCW from outside). Sides ordered
# bottom-left, bottom-right, top-right, top-left so the standard UV pattern
# keeps textures upright (pyrender flips images: v=0 is the image bottom).
_FACE_CORNERS: Dict[str, np.ndarray] = {
    "top": np.array([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)], dtype=np.float32),
    "bottom": np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], dtype=np.float32),
    "south": np.array([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.float32),
    "north": np.array([(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=np.float32),
    "east": np.array([(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)], dtype=np.float32),
    "west": np.array([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], dtype=np.float32),
}
_FACE_NORMALS: Dict[str, Tuple[float, float, float]] = {
    "top": (0, 1, 0), "bottom": (0, -1, 0), "south": (0, 0, 1),
    "north": (0, 0, -1), "east": (1, 0, 0), "west": (-1, 0, 0),
}
_UV_PATTERN = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

_RENDERER_CACHE: Dict[int, "pyrender.OffscreenRenderer"] = {}


@atexit.register
def _cleanup_renderers() -> None:
    for renderer in _RENDERER_CACHE.values():
        try:
            renderer.delete()
        except Exception:
            pass
    _RENDERER_CACHE.clear()


def _exposed_masks(occupied: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-direction masks of voxels whose face in that direction is exposed."""
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    masks: Dict[str, np.ndarray] = {}
    for name, (dx, dy, dz), _, _ in _DIRECTIONS:
        neighbor = padded[
            1 + dx : padded.shape[0] - 1 + dx,
            1 + dy : padded.shape[1] - 1 + dy,
            1 + dz : padded.shape[2] - 1 + dz,
        ]
        masks[name] = occupied & ~neighbor
    return masks


def _shade_tile(tile: np.ndarray, brightness: float) -> np.ndarray:
    out = tile.copy()
    out[..., :3] = (tile[..., :3].astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)
    return out


def build_mesh(structure: Structure, face_textures: FaceTextureLibrary) -> "pyrender.Mesh":
    """Culled quad mesh with a stitched atlas; face shading baked into tiles.

    Returns one pyrender.Mesh with at most two primitives (opaque + blended).
    """
    occupied = structure.occupied_mask
    masks = _exposed_masks(occupied)
    ids = structure.block_ids
    data = structure.block_data

    # Collect faces and assign atlas tiles keyed by (texture identity, shade).
    tile_index: Dict[Tuple[int, float], int] = {}
    tile_specs: List[Tuple[np.ndarray, float]] = []
    # faces[i] = (x, y, z, direction_name, tile_idx)
    faces: List[Tuple[int, int, int, str, int]] = []
    for name, _, kind, brightness in _DIRECTIONS:
        coords = np.argwhere(masks[name])
        for x, y, z in coords:
            tex = face_textures[(int(ids[x, y, z]), int(data[x, y, z]))][kind]
            key = (id(tex), brightness)
            idx = tile_index.get(key)
            if idx is None:
                idx = len(tile_specs)
                tile_index[key] = idx
                tile_specs.append((tex, brightness))
            faces.append((int(x), int(y), int(z), name, idx))

    if not faces:
        raise ValueError("structure has no exposed faces (all air?)")

    # Stitch atlas (uniform tile size, nearest-neighbor resize for odd sizes).
    tile_px = max(max(t.shape[0] for t, _ in tile_specs), 2)
    n_tiles = len(tile_specs)
    cols = int(math.ceil(math.sqrt(n_tiles)))
    rows = int(math.ceil(n_tiles / cols))
    atlas = np.zeros((rows * tile_px, cols * tile_px, 4), dtype=np.uint8)
    tile_transparent: List[bool] = []
    for idx, (tex, brightness) in enumerate(tile_specs):
        if tex.shape[0] != tile_px or tex.shape[1] != tile_px:
            tex = np.asarray(
                Image.fromarray(tex).resize((tile_px, tile_px), Image.NEAREST),
                dtype=np.uint8,
            )
        tile = _shade_tile(tex, brightness)
        r, c = divmod(idx, cols)
        atlas[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = tile
        tile_transparent.append(bool((tile[..., 3] < 255).any()))

    # Tile UV rects with a half-texel inset against bleeding.
    inset = 0.5 / tile_px
    uv_rects = np.empty((n_tiles, 4), dtype=np.float32)  # u0, v0, u1, v1
    for idx in range(n_tiles):
        r, c = divmod(idx, cols)
        u0, u1 = (c + inset) / cols, (c + 1 - inset) / cols
        # pyrender uploads images flipped, so v=0 is the atlas bottom row.
        v_top = 1.0 - (r + inset) / rows
        v_bot = 1.0 - (r + 1 - inset) / rows
        uv_rects[idx] = (u0, v_bot, u1, v_top)

    def _primitive(face_subset: Sequence[Tuple[int, int, int, str, int]], blend: bool):
        n = len(face_subset)
        positions = np.empty((n * 4, 3), dtype=np.float32)
        normals = np.empty((n * 4, 3), dtype=np.float32)
        uvs = np.empty((n * 4, 2), dtype=np.float32)
        indices = np.empty((n * 2, 3), dtype=np.uint32)
        for i, (x, y, z, name, idx) in enumerate(face_subset):
            v = i * 4
            positions[v : v + 4] = _FACE_CORNERS[name] + np.array([x, y, z], dtype=np.float32)
            normals[v : v + 4] = _FACE_NORMALS[name]
            u0, v0, u1, v1 = uv_rects[idx]
            uvs[v : v + 4, 0] = u0 + _UV_PATTERN[:, 0] * (u1 - u0)
            uvs[v : v + 4, 1] = v0 + _UV_PATTERN[:, 1] * (v1 - v0)
            indices[i * 2] = (v, v + 1, v + 2)
            indices[i * 2 + 1] = (v, v + 2, v + 3)
        sampler = pyrender.Sampler(
            magFilter=GL_NEAREST, minFilter=GL_NEAREST,
            wrapS=GL_CLAMP_TO_EDGE, wrapT=GL_CLAMP_TO_EDGE,
        )
        texture = pyrender.Texture(source=atlas, source_channels="RGBA", sampler=sampler)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=texture,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND" if blend else "OPAQUE",
            doubleSided=False,
        )
        return pyrender.Primitive(
            positions=positions, normals=normals, texcoord_0=uvs,
            indices=indices, material=material, mode=4,
        )

    opaque = [f for f in faces if not tile_transparent[f[4]]]
    blended = [f for f in faces if tile_transparent[f[4]]]
    primitives = []
    if opaque:
        primitives.append(_primitive(opaque, blend=False))
    if blended:
        primitives.append(_primitive(blended, blend=True))
    return pyrender.Mesh(primitives=primitives)


def _look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _get_renderer(px: int) -> "pyrender.OffscreenRenderer":
    renderer = _RENDERER_CACHE.get(px)
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(viewport_width=px, viewport_height=px)
        _RENDERER_CACHE[px] = renderer
    return renderer


def render_structure(
    structure: Structure,
    px: int = 512,
    azim_deg: float = 45.0,
    elev_deg: float = 30.0,
    pack_dir: str = DEFAULT_PACK_DIR,
    ortho: bool = True,
    bg: Optional[Tuple[float, float, float]] = None,
    face_textures: Optional[FaceTextureLibrary] = None,
) -> np.ndarray:
    """Render a Structure to a (px, px, 4) uint8 RGBA array (Y up, EGL offscreen).

    bg=None gives a transparent background; otherwise pass an RGB tuple in [0, 1].
    Pass a shared `face_textures` library when rendering in batches.
    """
    structure = structure.crop_to_non_air()
    if face_textures is None:
        face_textures = load_face_textures(pack_dir)
    mesh = build_mesh(structure, face_textures)

    bg_color = (0.0, 0.0, 0.0, 0.0) if bg is None else (*bg[:3], 1.0)
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(1.0, 1.0, 1.0))
    scene.add(mesh)

    dims = np.asarray(structure.shape, dtype=np.float64)
    center = dims / 2.0
    radius = 0.5 * float(np.linalg.norm(dims))
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    direction = np.array([
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
        math.cos(elev) * math.cos(azim),
    ])
    distance = 2.5 * radius + 1.0
    eye = center + direction * distance
    pose = _look_at(eye, center)

    margin = 1.05
    if ortho:
        # Frame the bbox extents projected onto the camera axes.
        corners = np.array([
            (x, y, z)
            for x in (0.0, dims[0]) for y in (0.0, dims[1]) for z in (0.0, dims[2])
        ]) - center
        right, true_up = pose[:3, 0], pose[:3, 1]
        xmag = float(np.abs(corners @ right).max()) * margin
        ymag = float(np.abs(corners @ true_up).max()) * margin
        mag = max(xmag, ymag, 1e-3)
        camera = pyrender.OrthographicCamera(
            xmag=mag, ymag=mag, znear=0.01, zfar=distance + 2.0 * radius,
        )
    else:
        yfov = 2.0 * math.atan2(radius * margin, distance)
        camera = pyrender.PerspectiveCamera(
            yfov=max(yfov, 0.05), znear=0.01, zfar=distance + 2.0 * radius,
        )
    scene.add(camera, pose=pose)

    renderer = _get_renderer(px)
    flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
    color, _ = renderer.render(scene, flags=flags)
    color = np.asarray(color, dtype=np.uint8)
    if color.shape[-1] == 3:  # some drivers ignore RGBA
        alpha = np.full(color.shape[:2] + (1,), 255, dtype=np.uint8)
        color = np.concatenate([color, alpha], axis=-1)
    return color


def _load_structure_from_cache(cache: str, index: int) -> Structure:
    archive = np.load(cache, allow_pickle=True)
    return Structure(
        block_ids=archive["block_ids"][index],
        block_data=archive["block_data"][index],
        source_path=f"{cache}[{index}]",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a cached structure with real textures")
    parser.add_argument("--cache", type=str, required=True, help="npz cache path")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="output png path")
    parser.add_argument("--px", type=int, default=512)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--pack-dir", type=str, default=DEFAULT_PACK_DIR)
    parser.add_argument("--perspective", action="store_true", help="use perspective camera")
    args = parser.parse_args()

    structure = _load_structure_from_cache(args.cache, args.index)
    image = render_structure(
        structure, px=args.px, azim_deg=args.azim, elev_deg=args.elev,
        pack_dir=args.pack_dir, ortho=not args.perspective,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
    print(f"Wrote {out_path} ({image.shape[0]}x{image.shape[1]})")


if __name__ == "__main__":
    main()
