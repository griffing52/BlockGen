"""Real Minecraft block textures for legacy (1.12 numeric) block ids.

Provides a CLI to fetch vanilla textures from Mojang's servers (or extract a
resource pack), a `FACE_TEXTURES` mapping from legacy `(block_id, block_data)`
to modern texture basenames, and `load_face_textures` which returns per-face
RGBA arrays with solid-color fallbacks so every non-air block renders.

Usage:
    python -m blockgen.renderer.textures --fetch
    python -m blockgen.renderer.textures --pack Faithful.zip --name faithful
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from blockgen.utils.data import _color_for, _resource_location_for

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
BLOCK_TEXTURE_PREFIX = "assets/minecraft/textures/block/"
DEFAULT_PACK_DIR = "data/textures/vanilla"

FaceNames = Dict[str, str]
FaceArrays = Dict[str, np.ndarray]

_COLORS = [
    "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink",
    "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red",
    "black",
]
_WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak"]

# Color multipliers applied to specific textures at load time.
FOLIAGE_TINT = (0.55, 0.75, 0.35)
WATER_TINT = (0.25, 0.46, 0.90)
TINTS: Dict[str, Tuple[float, float, float]] = {
    "grass_block_top": FOLIAGE_TINT,
    "water_still": WATER_TINT,
    **{f"{w}_leaves": FOLIAGE_TINT for w in _WOODS},
    **{n: FOLIAGE_TINT for n in ("short_grass", "fern", "vine", "lily_pad")},
}

# Rendered fully opaque ("fast graphics" style) to avoid alpha-blend artifacts.
FORCE_OPAQUE = {f"{w}_leaves" for w in _WOODS}


def _cube(name: str) -> FaceNames:
    return {"top": name, "side": name, "bottom": name}


def _tsb(top: str, side: str, bottom: str) -> FaceNames:
    return {"top": top, "side": side, "bottom": bottom}


def _pillar(name: str) -> FaceNames:
    return {"top": f"{name}_top", "side": name, "bottom": f"{name}_top"}


def _build_face_textures() -> Dict[Tuple[int, int], FaceNames]:
    ft: Dict[Tuple[int, int], FaceNames] = {}

    # Stone variants 1:0-6.
    for d, n in enumerate([
        "stone", "granite", "polished_granite", "diorite",
        "polished_diorite", "andesite", "polished_andesite",
    ]):
        ft[(1, d)] = _cube(n)

    ft[(2, 0)] = _tsb("grass_block_top", "grass_block_side", "dirt")
    ft[(3, 0)] = _cube("dirt")
    ft[(3, 1)] = _cube("coarse_dirt")
    ft[(3, 2)] = _tsb("podzol_top", "podzol_side", "dirt")
    ft[(4, 0)] = _cube("cobblestone")
    for d, w in enumerate(_WOODS):
        ft[(5, d)] = _cube(f"{w}_planks")
    ft[(7, 0)] = _cube("bedrock")
    ft[(8, 0)] = _cube("water_still")
    ft[(9, 0)] = _cube("water_still")
    ft[(10, 0)] = _cube("lava_still")
    ft[(11, 0)] = _cube("lava_still")
    ft[(12, 0)] = _cube("sand")
    ft[(12, 1)] = _cube("red_sand")
    ft[(13, 0)] = _cube("gravel")
    ft[(14, 0)] = _cube("gold_ore")
    ft[(15, 0)] = _cube("iron_ore")
    ft[(16, 0)] = _cube("coal_ore")

    # Logs: species = data & 0x3 (orientation bits handled by lookup masking).
    for d, w in enumerate(_WOODS[:4]):
        ft[(17, d)] = _pillar(f"{w}_log")
    for d, w in enumerate(_WOODS[4:]):
        ft[(162, d)] = _pillar(f"{w}_log")
    for d, w in enumerate(_WOODS[:4]):
        ft[(18, d)] = _cube(f"{w}_leaves")
    for d, w in enumerate(_WOODS[4:]):
        ft[(161, d)] = _cube(f"{w}_leaves")

    ft[(19, 0)] = _cube("sponge")
    ft[(19, 1)] = _cube("wet_sponge")
    ft[(20, 0)] = _cube("glass")
    ft[(21, 0)] = _cube("lapis_ore")
    ft[(22, 0)] = _cube("lapis_block")
    ft[(24, 0)] = _tsb("sandstone_top", "sandstone", "sandstone_bottom")
    ft[(24, 1)] = _tsb("sandstone_top", "chiseled_sandstone", "sandstone_bottom")
    ft[(24, 2)] = _tsb("sandstone_top", "cut_sandstone", "sandstone_bottom")

    for d, c in enumerate(_COLORS):
        ft[(35, d)] = _cube(f"{c}_wool")       # wool
        ft[(171, d)] = _cube(f"{c}_wool")      # carpet
        ft[(95, d)] = _cube(f"{c}_stained_glass")
        ft[(160, d)] = _cube(f"{c}_stained_glass")  # stained pane
        ft[(159, d)] = _cube(f"{c}_terracotta")
        ft[(251, d)] = _cube(f"{c}_concrete")
        ft[(252, d)] = _cube(f"{c}_concrete_powder")

    ft[(41, 0)] = _cube("gold_block")
    ft[(42, 0)] = _cube("iron_block")
    ft[(45, 0)] = _cube("bricks")
    ft[(46, 0)] = _tsb("tnt_top", "tnt_side", "tnt_bottom")
    ft[(47, 0)] = _tsb("oak_planks", "bookshelf", "oak_planks")
    ft[(48, 0)] = _cube("mossy_cobblestone")
    ft[(49, 0)] = _cube("obsidian")
    ft[(50, 0)] = _cube("torch")
    ft[(54, 0)] = _cube("oak_planks")          # chest (entity texture; use planks)
    ft[(56, 0)] = _cube("diamond_ore")
    ft[(57, 0)] = _cube("diamond_block")
    ft[(58, 0)] = _tsb("crafting_table_top", "crafting_table_side", "oak_planks")
    ft[(61, 0)] = _tsb("furnace_top", "furnace_side", "furnace_top")
    ft[(62, 0)] = _tsb("furnace_top", "furnace_side", "furnace_top")
    ft[(65, 0)] = _cube("ladder")
    ft[(73, 0)] = _cube("redstone_ore")
    ft[(74, 0)] = _cube("redstone_ore")
    ft[(78, 0)] = _cube("snow")                # snow layer
    ft[(79, 0)] = _cube("ice")
    ft[(80, 0)] = _cube("snow")
    ft[(82, 0)] = _cube("clay")
    ft[(86, 0)] = _tsb("pumpkin_top", "pumpkin_side", "pumpkin_top")
    ft[(87, 0)] = _cube("netherrack")
    ft[(88, 0)] = _cube("soul_sand")
    ft[(89, 0)] = _cube("glowstone")
    ft[(91, 0)] = _tsb("pumpkin_top", "jack_o_lantern", "pumpkin_top")
    ft[(98, 0)] = _cube("stone_bricks")
    ft[(98, 1)] = _cube("mossy_stone_bricks")
    ft[(98, 2)] = _cube("cracked_stone_bricks")
    ft[(98, 3)] = _cube("chiseled_stone_bricks")
    ft[(101, 0)] = _cube("iron_bars")
    ft[(102, 0)] = _cube("glass")              # glass pane
    ft[(103, 0)] = _tsb("melon_top", "melon_side", "melon_top")
    ft[(110, 0)] = _tsb("mycelium_top", "mycelium_side", "dirt")
    ft[(112, 0)] = _cube("nether_bricks")
    ft[(113, 0)] = _cube("nether_bricks")      # nether brick fence
    ft[(121, 0)] = _cube("end_stone")
    ft[(153, 0)] = _cube("nether_quartz_ore")
    ft[(155, 0)] = _tsb("quartz_block_top", "quartz_block_side", "quartz_block_bottom")
    ft[(155, 1)] = _tsb("chiseled_quartz_block_top", "chiseled_quartz_block", "chiseled_quartz_block_top")
    ft[(155, 2)] = _tsb("quartz_pillar_top", "quartz_pillar_side", "quartz_pillar_top")
    ft[(129, 0)] = _cube("emerald_ore")
    ft[(133, 0)] = _cube("emerald_block")
    ft[(168, 0)] = _cube("prismarine")
    ft[(168, 1)] = _cube("prismarine_bricks")
    ft[(168, 2)] = _cube("dark_prismarine")
    ft[(169, 0)] = _cube("sea_lantern")
    ft[(170, 0)] = _tsb("hay_block_top", "hay_block_side", "hay_block_top")
    ft[(172, 0)] = _cube("terracotta")
    ft[(173, 0)] = _cube("coal_block")
    ft[(174, 0)] = _cube("packed_ice")
    ft[(179, 0)] = _tsb("red_sandstone_top", "red_sandstone", "red_sandstone_bottom")
    ft[(179, 1)] = _tsb("red_sandstone_top", "chiseled_red_sandstone", "red_sandstone_bottom")
    ft[(179, 2)] = _tsb("red_sandstone_top", "cut_red_sandstone", "red_sandstone_bottom")
    ft[(201, 0)] = _cube("purpur_block")
    ft[(202, 0)] = _tsb("purpur_pillar_top", "purpur_pillar_side", "purpur_pillar_top")
    ft[(206, 0)] = _cube("end_stone_bricks")
    ft[(213, 0)] = _cube("magma")
    ft[(214, 0)] = _cube("nether_wart_block")
    ft[(215, 0)] = _cube("red_nether_bricks")
    ft[(216, 0)] = _tsb("bone_block_top", "bone_block_side", "bone_block_top")

    # Doors: door bottom texture on all faces.
    for bid, w in [(64, "oak"), (193, "spruce"), (194, "birch"),
                   (195, "jungle"), (196, "acacia"), (197, "dark_oak")]:
        ft[(bid, 0)] = _cube(f"{w}_door_bottom")
    ft[(71, 0)] = _cube("iron_door_bottom")
    ft[(96, 0)] = _cube("oak_trapdoor")
    ft[(167, 0)] = _cube("iron_trapdoor")

    # Slabs / double slabs rendered as full cubes of their material.
    stone_slab_mats = [
        "smooth_stone", "sandstone", "oak_planks", "cobblestone",
        "bricks", "stone_bricks", "nether_bricks", "quartz_block_side",
    ]
    for d, n in enumerate(stone_slab_mats):
        ft[(43, d)] = _cube(n)
        ft[(44, d)] = _cube(n)
    for d, w in enumerate(_WOODS):
        ft[(125, d)] = _cube(f"{w}_planks")
        ft[(126, d)] = _cube(f"{w}_planks")
    ft[(181, 0)] = _cube("red_sandstone")
    ft[(182, 0)] = _cube("red_sandstone")
    ft[(204, 0)] = _cube("purpur_block")
    ft[(205, 0)] = _cube("purpur_block")

    # Stairs rendered as full cubes of their material.
    for bid, n in [
        (53, "oak_planks"), (67, "cobblestone"), (108, "bricks"),
        (109, "stone_bricks"), (114, "nether_bricks"), (128, "sandstone"),
        (134, "spruce_planks"), (135, "birch_planks"), (136, "jungle_planks"),
        (156, "quartz_block_side"), (163, "acacia_planks"),
        (164, "dark_oak_planks"), (180, "red_sandstone"), (203, "purpur_block"),
    ]:
        ft[(bid, 0)] = _cube(n)

    # Fences / gates rendered as full cubes of their plank material.
    for bid, w in [(85, "oak"), (188, "spruce"), (189, "birch"),
                   (190, "jungle"), (191, "dark_oak"), (192, "acacia"),
                   (107, "oak"), (183, "spruce"), (184, "birch"),
                   (185, "jungle"), (186, "dark_oak"), (187, "acacia")]:
        ft[(bid, 0)] = _cube(f"{w}_planks")

    # Walls.
    ft[(139, 0)] = _cube("cobblestone")
    ft[(139, 1)] = _cube("mossy_cobblestone")

    # Decoration/plant blocks (cross-models rendered as translucent cubes; far
    # better than the solid-color fallback).
    ft[(30, 0)] = _cube("cobweb")
    ft[(31, 1)] = _cube("short_grass")   # renamed from "grass" in modern packs
    ft[(31, 2)] = _cube("fern")
    ft[(32, 0)] = _cube("dead_bush")
    ft[(37, 0)] = _cube("dandelion")
    ft[(38, 0)] = _cube("poppy")
    ft[(39, 0)] = _cube("brown_mushroom")
    ft[(40, 0)] = _cube("red_mushroom")
    ft[(83, 0)] = _cube("sugar_cane")
    ft[(106, 0)] = _cube("vine")
    ft[(111, 0)] = _cube("lily_pad")
    ft[(140, 0)] = _cube("flower_pot")
    ft[(175, 0)] = _cube("sunflower_front")
    ft[(175, 1)] = _cube("lilac_top")
    ft[(175, 4)] = _cube("rose_bush_top")
    ft[(175, 5)] = _cube("peony_top")
    # Infested / monster-egg blocks look like their host material.
    for d, n in enumerate(["stone", "cobblestone", "stone_bricks",
                           "mossy_stone_bricks", "cracked_stone_bricks",
                           "chiseled_stone_bricks"]):
        ft[(97, d)] = _cube(n)

    return ft


FACE_TEXTURES: Dict[Tuple[int, int], FaceNames] = _build_face_textures()


def resolve_face_names(block_id: int, block_data: int) -> Optional[FaceNames]:
    """Look up face texture names, masking orientation/upper-slab data bits."""
    for key in (
        (block_id, block_data),
        (block_id, block_data & 0x7),
        (block_id, block_data & 0x3),
        (block_id, 0),
    ):
        if key in FACE_TEXTURES:
            return FACE_TEXTURES[key]
    return None


def _solid_texture(block_id: int, block_data: int) -> np.ndarray:
    resource = _resource_location_for(block_id, block_data)
    r, g, b, a = _color_for(resource, block_data)
    tile = np.empty((2, 2, 4), dtype=np.uint8)
    tile[..., 0] = int(round(r * 255))
    tile[..., 1] = int(round(g * 255))
    tile[..., 2] = int(round(b * 255))
    tile[..., 3] = int(round(a * 255))
    return tile


class FaceTextureLibrary(dict):
    """dict[(block_id, block_data)] -> {"top"/"side"/"bottom": RGBA uint8 array}.

    Lazily loads pngs from `pack_dir` (first frame of animated strips,
    tints applied) and falls back to a solid-color 2x2 tile for unmapped
    blocks so every non-air block renders.
    """

    def __init__(self, pack_dir: str = DEFAULT_PACK_DIR) -> None:
        super().__init__()
        self.pack_dir = Path(pack_dir)
        self._by_name: Dict[str, Optional[np.ndarray]] = {}
        self.fallback_keys: set = set()

    def _load_png(self, name: str) -> Optional[np.ndarray]:
        if name in self._by_name:
            return self._by_name[name]
        path = self.pack_dir / f"{name}.png"
        arr: Optional[np.ndarray] = None
        if path.exists():
            img = Image.open(path).convert("RGBA")
            w, h = img.size
            if h > w:  # animated strip: keep first square frame
                img = img.crop((0, 0, w, w))
            arr = np.asarray(img, dtype=np.uint8)
            tint = TINTS.get(name)
            if tint is not None:
                rgb = arr[..., :3].astype(np.float32) * np.asarray(tint, dtype=np.float32)
                arr = arr.copy()
                arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
            if name in FORCE_OPAQUE:
                arr = arr.copy()
                arr[..., 3] = 255
        self._by_name[name] = arr
        return arr

    def __missing__(self, key: Tuple[int, int]) -> FaceArrays:
        block_id, block_data = key
        names = resolve_face_names(block_id, block_data)
        faces: FaceArrays = {}
        if names is not None:
            loaded = {face: self._load_png(name) for face, name in names.items()}
            if all(a is not None for a in loaded.values()):
                faces = loaded  # type: ignore[assignment]
        if not faces:
            solid = _solid_texture(block_id, block_data)
            faces = {"top": solid, "side": solid, "bottom": solid}
            self.fallback_keys.add(key)
        self[key] = faces
        return faces


def load_face_textures(pack_dir: str = DEFAULT_PACK_DIR) -> FaceTextureLibrary:
    """Load per-face textures for legacy blocks from an extracted pack dir."""
    return FaceTextureLibrary(pack_dir=pack_dir)


# ---------------------------------------------------------------------------
# Texture acquisition
# ---------------------------------------------------------------------------

def _extract_block_textures(jar_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(jar_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(BLOCK_TEXTURE_PREFIX) or not name.endswith(".png"):
                continue
            rel = name[len(BLOCK_TEXTURE_PREFIX):]
            if "/" in rel:  # keep the flat block/ subtree only
                continue
            (dest / rel).write_bytes(zf.read(info))
            count += 1
    return count


def fetch_vanilla(dest: str = DEFAULT_PACK_DIR, force: bool = False) -> int:
    """Download the latest-release client jar and extract block textures."""
    dest_path = Path(dest)
    existing = list(dest_path.glob("*.png")) if dest_path.exists() else []
    if existing and not force:
        print(f"{dest_path} already has {len(existing)} textures; skipping fetch")
        return len(existing)

    with urllib.request.urlopen(MANIFEST_URL) as resp:
        manifest = json.load(resp)
    release = manifest["latest"]["release"]
    version_url = next(v["url"] for v in manifest["versions"] if v["id"] == release)
    with urllib.request.urlopen(version_url) as resp:
        version_json = json.load(resp)
    client_url = version_json["downloads"]["client"]["url"]

    dest_path.mkdir(parents=True, exist_ok=True)
    jar_path = dest_path.parent / f"client_{release}.jar"
    print(f"Downloading Minecraft {release} client jar...")
    urllib.request.urlretrieve(client_url, jar_path)
    try:
        count = _extract_block_textures(jar_path, dest_path)
    finally:
        jar_path.unlink(missing_ok=True)
    print(f"Extracted {count} block textures to {dest_path}")
    return count


def extract_pack(pack_zip: str, name: str, textures_root: str = "data/textures") -> int:
    """Extract block textures from a resource-pack zip into data/textures/<name>."""
    dest = Path(textures_root) / name
    count = _extract_block_textures(Path(pack_zip), dest)
    print(f"Extracted {count} block textures to {dest}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch/extract Minecraft block textures")
    parser.add_argument("--fetch", action="store_true", help="download vanilla textures")
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument("--pack", type=str, default=None, help="resource-pack zip to extract")
    parser.add_argument("--name", type=str, default="pack", help="pack name (dest subdir)")
    args = parser.parse_args()

    if args.fetch:
        fetch_vanilla(force=args.force)
    if args.pack:
        extract_pack(args.pack, args.name)
    if not args.fetch and not args.pack:
        parser.print_help()


if __name__ == "__main__":
    main()
