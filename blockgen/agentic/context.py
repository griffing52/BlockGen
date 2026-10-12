"""Context modules: what the model is told (or shown) besides the request.

A context is a registered function ``(ep, **params) -> ContextParts``. Text goes
into the system prompt; images go into the first build turn (Anthropic and Claude
Code system prompts cannot carry images). The Episode calls each context of the
recipe once when it builds the system message, in recipe order, so a context may
leave a result in ``ep.notes`` for a later one (``dream_palette`` ->
``texture_sheet{blocks: dream_palette}``).

Built-ins, each the suite form of an existing experiment axis:

``palette``        the legacy palette list, byte-identical to ``prompts._palette_block``
``ontology``       the legacy ontology table, byte-identical to BuildAgent's
``examples``       k demonstrations IN THE ACTION SPACE'S FORMAT, as prior turns
``texture_sheet``  a labelled image of block faces (texsheet.py), + shuffled control
``dream_palette``  the blocks whose measured colour is nearest the dream's colours
``reference``      the user's own reference images
``scale``          player-scale rules in blocks (scale.py): doors, ceilings, rooms

Why byte-identical legacy text: published ``ont_*`` and scaffolding numbers, and
the response cache, are keyed on those exact bytes. A context that "improves" the
wording would silently make the suite's legacy arms a different experiment.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.core import ContextParts, parse_spec, register
from blockgen.agentic.providers import Message, Part, text_message

#: The note legacy ``prompts.build_prompt`` puts on reference images (kept verbatim).
REFERENCE_NOTE = ("Reference image(s) of what to build. Match the massing, "
                  "proportions and materials as closely as the block palette allows.")


def _event(ep: Any, kind: str, **data: Any) -> None:
    trace = getattr(ep, "trace", None)
    if trace is not None:
        trace.event(kind, **data)


def _notes(ep: Any) -> Dict[str, Any]:
    notes = getattr(ep, "notes", None)
    if notes is None:
        notes = {}
        try:
            ep.notes = notes
        except AttributeError:
            pass
    return notes


# ===========================================================================
# palette / ontology (legacy text)
# ===========================================================================
def _active_ontology(ep: Any) -> bool:
    """Whether the recipe also shows an ontology table that is not 'none'."""
    spec = getattr(ep, "spec", None)
    for c in getattr(spec, "contexts", None) or []:
        try:
            name, _, params = parse_spec(c)
        except ValueError:
            continue
        if name == "ontology" and params.get("variant", "mined") not in ("none", "", None):
            return True
    return False


@register("context", "palette")
def palette(ep: Any, *, skip_if_ontology: bool = True) -> ContextParts:
    """The legacy palette list (``prompts._palette_block``), verbatim.

    With an active ontology context in the same recipe it yields nothing, exactly
    as legacy ``system_prompt`` does (the ontology's first column IS the palette;
    printing both would make the ontology arm pay ~300 tokens for a second copy of
    the vocabulary). ``skip_if_ontology=False`` prints both on purpose.
    """
    from blockgen.agentic.prompts import _palette_block

    if skip_if_ontology and _active_ontology(ep):
        return ContextParts()
    return ContextParts(system_text=_palette_block())


def ontology_text(variant: str = "mined", path: Optional[str] = None, seed: int = 0) -> str:
    """The text BuildAgent._ontology_block builds for this variant ('' for 'none').

    Same calls, same failure: a missing catalog raises here instead of degrading,
    because an ontology arm that silently ran without its ontology lands in the
    results table as a null result.
    """
    if variant in ("none", "", None):
        return ""
    from blockgen.ontology import default_path, prompt_block, variant_catalog

    catalog = variant_catalog(variant, path, seed=seed)
    if catalog is None:
        raise FileNotFoundError(
            f"ontology arm '{variant}' needs a catalog at {path or default_path()}; "
            f"build it with `python -m blockgen.ontology`")
    return prompt_block(catalog)


@register("context", "ontology")
def ontology(ep: Any, *, variant: str = "mined", path: Optional[str] = None,
             seed: int = 0) -> ContextParts:
    """The block-ontology table (mined / shuffled control / stats / none)."""
    text = ontology_text(variant, path, seed)
    _notes(ep)["ontology"] = {"variant": variant, "seed": seed, "chars": len(text)}
    return ContextParts(system_text=text)


# ===========================================================================
# examples
# ===========================================================================
@dataclass
class ExampleParts(ContextParts):
    """ContextParts plus demonstration turns.

    ``turns`` alternate user ``Build: <caption>`` / assistant ``<answer>``, the
    same shape legacy ``prompts.build_prompt`` uses, and are meant to be inserted
    as prior turns before the first build request. They are also stored in
    ``ep.notes['example_turns']`` for an Episode that reads notes instead. While
    the Episode does not set ``inserts_example_turns``, ``user_parts`` carries the
    same demonstrations as text (see :func:`examples`).
    """

    turns: List[Message] = field(default_factory=list)


def prior_turns(parts: Sequence[ContextParts]) -> List[Message]:
    """All demonstration turns contributed by a list of context results."""
    out: List[Message] = []
    for p in parts:
        out.extend(getattr(p, "turns", None) or [])
    return out


def _convert_examples(ep: Any, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """DSL example programs -> the active action space's format."""
    acts = ep.kit.actions
    kind = getattr(acts, "name", "dsl")
    if kind == "dsl":
        return pairs
    if kind in ("python", "blocks"):
        from blockgen.agentic import actions as act_mod   # WP3 converters, lazy

        if kind == "python":
            return [(q, act_mod.dsl_to_python(a)) for q, a in pairs]
        size = tuple(getattr(ep.spec, "canvas", (32, 32, 32)))
        return [(q, act_mod.dsl_to_blocks(a, size)) for q, a in pairs]
    # Any other space (``mixed``, a future one) converts through its own
    # ``from_dsl``. Not pass-through for ``mixed``: with ``parts: [python, blocks]``
    # unfenced DSL is IGNORED by its lowering, so a raw-DSL example would teach
    # the model a format its replies cannot use.
    conv = getattr(acts, "from_dsl", None)
    if callable(conv):
        return [(q, conv(a)) for q, a in pairs]
    raise ValueError(f"examples: no DSL converter for action space '{kind}'")


@register("context", "examples")
def examples(ep: Any, *, k: int = 1, pool: str = "builtin") -> ExampleParts:
    """k demonstrations in the action space's own format, as prior turns.

    ``pool='builtin'`` asks the action space (``kit.actions.examples``), which
    already speaks its format. Any other value is a JSON file for
    ``examples.load_examples`` (DSL programs, converted to the action space); such
    a pool may come from the training corpus, which is why provenance marks it
    ``train_only``.
    """
    if k <= 0:
        return ExampleParts()
    if pool == "builtin":
        pairs = list(ep.kit.actions.examples(ep.prompt, k))
    else:
        from blockgen.agentic.examples import load_examples, select_examples

        if not Path(pool).is_file():
            raise FileNotFoundError(f"examples pool {pool!r}: not 'builtin' and not a file")
        chosen = select_examples(ep.prompt, k, load_examples(pool))
        pairs = _convert_examples(ep, [(ex.caption, ex.program) for ex in chosen])
    turns: List[Message] = []
    for request, answer in pairs:
        turns.append(text_message("user", f"Build: {request}"))
        turns.append(text_message("assistant", str(answer).strip()))
    _notes(ep)["example_turns"] = turns
    if not turns or getattr(ep, "inserts_example_turns", False):
        return ExampleParts(turns=turns)
    # The current Episode reads only ``system_text`` and ``user_parts``, so turns
    # alone would be dropped SILENTLY (an `examples` arm identical to no examples).
    # Until an Episode declares ``inserts_example_turns = True`` (and then inserts
    # ``turns`` as real prior turns), the same demonstrations ride in the first
    # build turn as text, which every Episode already delivers.
    return ExampleParts(user_parts=[Part("text", text=examples_text(pairs))], turns=turns)


def examples_text(pairs: Sequence[Tuple[str, str]]) -> str:
    """Demonstrations as one text block (the first-build-turn form of ``examples``)."""
    out = ["Example builds in the required output format (for reference):"]
    for i, (request, answer) in enumerate(pairs, start=1):
        out.append(f"--- Example {i} ---\nBuild: {request}\nAnswer:\n{str(answer).strip()}")
    out.append("--- End of examples ---")
    return "\n\n".join(out)


# ===========================================================================
# colours: sRGB <-> Lab, measured block colours
# ===========================================================================
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8/float (..., 3) in 0..255 -> CIE Lab (D65).

    Lab rather than RGB because the nearest-block match should be perceptual: in
    RGB, dark browns are all "close" to each other and to black, while a builder
    sees spruce and dark oak as clearly different.
    """
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_DYES = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
         "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")
#: Extra full blocks beyond PALETTE, so the dream palette can name colours the
#: curated palette lacks (every dye colour, the stone variants, nether/end blocks).
_EXTRA_CANDIDATES = tuple(f"{c}_{t}" for t in ("wool", "concrete", "terracotta")
                          for c in _DYES) + (
    "polished_andesite", "granite", "polished_granite", "diorite", "polished_diorite",
    "red_sandstone", "mossy_stone_bricks", "cracked_stone_bricks", "nether_bricks",
    "red_nether_bricks", "prismarine", "prismarine_bricks", "dark_prismarine",
    "snow_block", "purpur_block", "end_stone_bricks", "end_stone", "mycelium",
    "brown_mushroom_block", "red_mushroom_block", "iron_block", "gold_block",
    "emerald_block", "lapis_block", "redstone_block", "coal_block", "melon",
    "pumpkin", "sponge", "soul_sand", "terracotta")
#: Name endings / names that are not full cubes. Matched by name because the
#: measured texture of a stair is just its plank face.
_NOT_FULL_SUFFIX = ("_stairs", "_slab", "_fence", "_fence_gate", "_door", "_trapdoor",
                    "_wall", "_pane", "_carpet", "_button", "_pressure_plate", "_sign",
                    "_sapling", "_leaves", "_glass")
_NOT_FULL = frozenset({"ladder", "torch", "lantern", "glass", "water", "lava", "chest",
                       "rail", "cobweb", "vine"})


@dataclass(frozen=True)
class BlockColor:
    name: str
    block: Tuple[int, int]
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]


def _is_full(name: str) -> bool:
    return name not in _NOT_FULL and not name.endswith(_NOT_FULL_SUFFIX)


@lru_cache(maxsize=4)
def block_colors(pack_dir: Optional[str] = None) -> Tuple[BlockColor, ...]:
    """Opaque full blocks with a measured side-texture colour (PALETTE first).

    Blocks whose texture is missing are left out rather than guessed (the
    ontology's rule: a fabricated colour in a "measured" list is worse than a gap).
    Names that resolve to an already-listed legacy block are dropped too, so the
    list never offers two names for one rendered material.
    """
    from blockgen.agentic.blockstate import PALETTE, try_resolve_block
    from blockgen.agentic import texsheet
    from blockgen.ontology.appearance import measure

    pack = pack_dir or str(texsheet.PACK_DIR)
    out: List[BlockColor] = []
    seen = set()
    for name in tuple(PALETTE) + _EXTRA_CANDIDATES:
        if not _is_full(name):
            continue
        pair = try_resolve_block(name)
        if pair is None or pair == (0, 0) or pair in seen:
            continue
        app = measure(pair[0], pair[1], pack)
        if app is None or app.transparency != "opaque":
            continue
        seen.add(pair)
        lab = tuple(float(v) for v in rgb_to_lab(np.asarray(app.rgb)))
        out.append(BlockColor(name, pair, tuple(app.rgb), lab))  # type: ignore[arg-type]
    return tuple(out)


# ===========================================================================
# dream_palette
# ===========================================================================
#: Brightness the renderers give each visible face (renderer/textured.py
#: ``_DIRECTIONS``: top 1.0, north/south 0.85, east/west 0.70), plus the midpoints a
#: block-scale cell straddling two faces averages to. Candidate block colours are
#: matched at every shade, because two thirds of an iso view is shaded side faces:
#: unshaded, magenta_terracotta x 0.79 IS purple_terracotta's mean colour.
FACE_SHADES: Tuple[float, ...] = (1.0, 0.925, 0.85, 0.775, 0.70)

#: dE added per unit of shade away from 1.0: on a tie, the lit reading wins.
_SHADE_PENALTY = 4.0

#: Long side (px) the dream is box-filtered to before clustering. Roughly one cell
#: per block for a typical framed house, so a cluster is a BLOCK colour, not a
#: texel colour (raw texels split bricks into mortar + brick, and each half
#: names an unrelated block).
PALETTE_SIDE = 48


def _decode(png: bytes, max_side: int = PALETTE_SIDE) -> np.ndarray:
    """RGB float array, box-filtered so the long side is at most ``max_side``."""
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGBA")
    if max(img.size) > max_side:
        img = img.resize((max(1, img.width * max_side // max(img.size)),
                          max(1, img.height * max_side // max(img.size))), Image.BOX)
    arr = np.asarray(img, dtype=np.float32)
    a = arr[..., 3:4] / 255.0
    return arr[..., :3] * a + 255.0 * (1 - a)            # alpha -> white, like renders


def foreground_lab(png: bytes, *, thresh: float = 12.0,
                   max_side: int = PALETTE_SIDE) -> np.ndarray:
    """Lab pixels that differ from the background (border median) by > ``thresh`` dE.

    The border median rather than "white": dream prompts ask for a white
    background, but an image model does not always comply, and a sky-blue
    backdrop must not become the first colour of the palette. The mask is eroded
    by one cell (when that keeps at least a quarter of it): after box filtering,
    silhouette cells are part background and read as a paler, unrelated block.
    """
    from scipy import ndimage

    rgb = _decode(png, max_side)
    lab = rgb_to_lab(rgb)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(lab - bg, axis=-1) > thresh
    inner = ndimage.binary_erosion(fg)
    if inner.sum() >= 0.25 * fg.sum() and inner.any():
        fg = inner
    return lab[fg]


def _shaded_lab(cands: Sequence["BlockColor"]) -> np.ndarray:
    """(B, S, 3) Lab of every candidate at every :data:`FACE_SHADES` brightness."""
    rgb = np.asarray([b.rgb for b in cands], np.float64)
    return np.stack([rgb_to_lab(np.clip(rgb * s, 0, 255)) for s in FACE_SHADES], 1)


def kmeans(x: np.ndarray, k: int, *, iters: int = 25, seed: int = 0
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic Lloyd's k-means (k-means++ init). -> (centres, labels).

    Hand-rolled (20 lines) rather than sklearn: it is not a dependency, and the
    result must be identical across runs because it ends up in the prompt.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(x.round(3), axis=0)
    k = max(1, min(int(k), len(uniq)))
    centres = [uniq[rng.integers(len(uniq))]]
    for _ in range(1, k):
        d2 = np.min(((uniq[:, None] - np.asarray(centres)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 0:
            break
        centres.append(uniq[rng.choice(len(uniq), p=d2 / d2.sum())])
    c = np.asarray(centres, dtype=np.float64)
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        labels = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[labels == j].mean(0) if (labels == j).any() else c[j]
                        for j in range(len(c))])
        if np.allclose(new, c):
            break
        c = new
    return c, labels


@dataclass
class PaletteEntry:
    name: str
    share: float          # fraction of foreground pixels whose cluster maps here (0 = fill)
    de: float             # Lab distance from the cluster centre to the block colour
    rgb: Tuple[int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "share": round(self.share, 4), "de": round(self.de, 2),
                "rgb": list(self.rgb)}


def match_palette(png: bytes, k: int = 12, *, fill_slack: float = 15.0,
                  candidates: Optional[Sequence[BlockColor]] = None,
                  max_pixels: int = 4096, min_clusters: int = 8) -> List[PaletteEntry]:
    """Up to ``k`` blocks for the dream's foreground colours, most prominent first.

    The dream is box-filtered to about block scale (:data:`PALETTE_SIDE`), and its
    foreground is clustered into ``max(k, min_clusters)`` Lab centres (not ``k``:
    with k=1, one centre is the AVERAGE colour, a purple for a red-and-blue house,
    which names a block that is in neither). Each centre maps to the block whose
    colour at some renderer face shade (:data:`FACE_SHADES`) is nearest; shares
    add up per block and the top ``k`` blocks are kept.
    If that yields fewer than ``k`` distinct blocks, runners-up are added round
    robin over the clusters (largest first), but only blocks within
    ``fill_slack`` dE of the cluster's best match, so a two-colour image never
    gets padded with unrelated colours.
    """
    if candidates is None:
        from blockgen.agentic import texsheet

        # Keyed on the CURRENT pack dir: block_colors() is lru-cached, and a bare
        # call would keep serving the colours of whatever pack was active first.
        candidates = block_colors(str(texsheet.PACK_DIR))
    cands = list(candidates)
    if not cands:
        return []
    px = foreground_lab(png)
    if len(px) == 0:
        return []
    if len(px) > max_pixels:                         # deterministic subsample
        px = px[np.linspace(0, len(px) - 1, max_pixels).astype(np.int64)]
    centres, labels = kmeans(px, max(int(k), int(min_clusters)))
    shares = np.bincount(labels, minlength=len(centres)) / float(len(labels))
    shaded = _shaded_lab(cands)                                         # (B, S, 3)
    pen = _SHADE_PENALTY * np.abs(1.0 - np.asarray(FACE_SHADES))
    dist = (np.linalg.norm(centres[:, None, None] - shaded[None], axis=-1)
            + pen).min(-1)                                              # (C, B)
    order = np.argsort(-shares, kind="stable")
    chosen: Dict[int, PaletteEntry] = {}
    for ci in order:
        bi = int(dist[ci].argmin())
        e = chosen.get(bi)
        if e is None:
            chosen[bi] = PaletteEntry(cands[bi].name, float(shares[ci]), float(dist[ci, bi]),
                                      cands[bi].rgb)
        else:
            e.share += float(shares[ci])
    entries = sorted(chosen.values(), key=lambda e: -e.share)
    ranked = {ci: list(np.argsort(dist[ci], kind="stable")) for ci in order}
    used = set(chosen)
    while len(entries) < k:
        added = False
        for ci in order:
            best = float(dist[ci].min())
            while ranked[ci]:
                bi = int(ranked[ci].pop(0))
                if bi in used:
                    continue
                if dist[ci, bi] <= best + fill_slack:
                    used.add(bi)
                    entries.append(PaletteEntry(cands[bi].name, 0.0, float(dist[ci, bi]),
                                                cands[bi].rgb))
                    added = True
                break
            if len(entries) >= k:
                break
        if not added:
            break
    return entries[:k]


def _latest_dream(ep: Any, key: str):
    """(key, DreamVersion) for ``key``, else the first dream key with a version."""
    dreams = getattr(ep, "dreams", None) or {}
    getter = getattr(ep, "latest_dream", None)
    if callable(getter):
        dv = getter(key)
    else:
        dv = (dreams.get(key) or [None])[-1]
    if dv is not None:
        return key, dv
    for k2, versions in dreams.items():
        if versions:
            return k2, versions[-1]
    return key, None


@register("context", "dream_palette", needs_dream=True)
def dream_palette(ep: Any, *, k: int = 12, key: str = "iso") -> ContextParts:
    """Blocks whose measured texture colour is nearest the dream's main colours.

    Foreground pixels of the latest ``key`` dream go through k-means in Lab; each
    centre is matched to the nearest opaque full block by measured RGB at any of
    the renderer's face shades. Sets
    ``ep.notes['dream_palette']`` (names) for ``texture_sheet{blocks: dream_palette}``.
    """
    used_key, dv = _latest_dream(ep, key)
    notes = _notes(ep)
    if dv is None:
        _event(ep, "dream_palette_skipped", reason=f"no dream '{key}' yet")
        notes["dream_palette"] = []
        return ContextParts()
    if used_key != key:
        _event(ep, "dream_palette_key", wanted=key, used=used_key)
    entries = match_palette(dv.png, k)
    notes["dream_palette"] = [e.name for e in entries]
    notes["dream_palette_detail"] = [e.to_dict() for e in entries]
    if not entries:
        _event(ep, "dream_palette_skipped", reason="no foreground colours or no measured "
                                                   "block colours (textures missing?)")
        return ContextParts()
    listed = ", ".join(f"{e.name} ({e.share:.0%})" if e.share >= 0.005 else e.name
                       for e in entries)
    text = ("Dream palette: the blocks whose real texture colour is closest to the main "
            "colours of the target image, most prominent first (share of the image in "
            f"brackets). Prefer these materials:\n  {listed}")
    return ContextParts(system_text=text)


# ===========================================================================
# texture_sheet
# ===========================================================================
def _used_names(ep: Any) -> List[str]:
    """Block names in the current world, PALETTE spellings preferred.

    A pair is named only by a name that renders with the SAME face textures
    (renderer ``resolve_face_names``). The lookup tries the pair, then the pair
    with its orientation bits masked the way the renderer masks them (data & 7,
    data & 3, 0), then any named block with identical textures. A plain
    ``(id, 0)`` fallback would call a quartz slab (44, 7) 'stone_slab' and an
    oriented spruce log (17, 5) 'oak_log', and show the wrong tile under it. A pair
    with no same-texture name is left out.
    """
    from blockgen.agentic.blockstate import PALETTE, try_resolve_block
    from blockgen.renderer.textures import resolve_face_names

    world = getattr(ep, "world", None)
    if world is None or world.block_count() == 0:
        return []
    s = world.to_structure(crop=True)
    occ = s.occupied_mask
    pairs = sorted(set(zip(s.block_ids[occ].tolist(), s.block_data[occ].tolist())))
    names: Dict[Tuple[int, int], str] = {}
    for n in tuple(PALETTE) + _EXTRA_CANDIDATES:
        p = try_resolve_block(n)
        if p is not None and p not in names:
            names[p] = n

    def faces(p: Tuple[int, int]) -> Optional[Tuple[Tuple[str, str], ...]]:
        f = resolve_face_names(int(p[0]), int(p[1]))
        return None if f is None else tuple(sorted(f.items()))

    by_faces: Dict[Any, str] = {}
    for p, n in names.items():
        by_faces.setdefault(faces(p), n)
    out: List[str] = []
    for p in pairs:
        want = faces(p)
        n = None
        for key in (p, (p[0], p[1] & 7), (p[0], p[1] & 3), (p[0], 0)):
            if key in names and (want is None or faces(key) == want):
                n = names[key]
                break
        if n is None and want is not None:
            n = by_faces.get(want)
        if n is not None and n not in out:
            out.append(n)
    return out


def _sheet_names(ep: Any, blocks: Any) -> List[str]:
    from blockgen.agentic.blockstate import PALETTE

    if isinstance(blocks, (list, tuple)):
        return [str(b) for b in blocks]
    if blocks == "palette":
        return list(PALETTE)
    if blocks == "used":
        world = getattr(ep, "world", None)
        if world is None or world.block_count() == 0:
            # The Episode runs contexts once, before the first build turn, so on a
            # fresh episode 'used' would silently show nothing every time.
            raise ValueError("texture_sheet blocks='used' needs a world that already "
                             "has blocks; contexts run once before the first build "
                             "turn, so in a recipe's contexts it is always empty. Use "
                             "'palette', 'dream_palette' or a list of names there.")
        return _used_names(ep)
    if blocks == "dream_palette":
        notes = _notes(ep)
        if "dream_palette" not in notes:            # recipe did not list it first
            dream_palette(ep)
        return list(notes.get("dream_palette") or [])
    raise ValueError(f"texture_sheet blocks must be 'palette', 'used', 'dream_palette' "
                     f"or a list of names, got {blocks!r}")


@register("context", "texture_sheet", images=True)
def texture_sheet(ep: Any, *, blocks: Any = "palette", tile: int = 64, cols: int = 10,
                  shuffled: bool = False, seed: int = 0, face: str = "side") -> ContextParts:
    """A labelled image of block faces (one tile per block, name underneath).

    ``shuffled=True`` is the token-matched control: identical tiles and label
    tokens, every label on the wrong tile. The caption text is identical for both
    arms, so the only difference the model can use is the picture-name binding.
    """
    from blockgen.agentic import texsheet

    names = _sheet_names(ep, blocks)
    if not names:
        _event(ep, "texture_sheet_skipped", blocks=str(blocks), reason="no blocks")
        return ContextParts()
    sheet = texsheet.build_sheet(names, tile=tile, cols=cols, face=face,
                                 shuffled=shuffled, seed=seed)
    if sheet.missing or sheet.unknown:
        msg = (f"texture_sheet: {len(sheet.missing)} solid-colour fallback tile(s) "
               f"{sheet.missing[:6]}, {len(sheet.unknown)} unknown name(s) {sheet.unknown[:6]}")
        warnings.warn(msg)
        _event(ep, "texture_sheet_degraded", missing=sheet.missing, unknown=sheet.unknown)
    _notes(ep)["texture_sheet"] = {"key": sheet.key, "pages": len(sheet.pages),
                                   "n": len(sheet.names), "shuffled": sheet.shuffled}
    n = len(sheet.pages)
    caption = (f"Texture sheet{f' ({n} pages)' if n > 1 else ''}: each tile shows the "
               f"{face} face of a block, with its block name underneath. Use it to choose "
               f"materials by how they look.")
    parts = [Part("text", text=caption)] + [Part("image", image_png=p) for p in sheet.pages]
    return ContextParts(user_parts=parts)


# ===========================================================================
# scale
# ===========================================================================
@register("context", "scale")
def scale(ep: Any, *, strict: bool = False, canvas: bool = True,
          dream_cue: bool = True) -> ContextParts:
    """Player-scale rules in blocks (a player is 2 blocks tall), as system text.

    System text, so every role with the builder system prompt (builder, planner,
    critic) gets it in any action space and method. ``canvas`` adds what the
    canvas holds at this scale and the rule to simplify rather than shrink;
    ``strict`` states the rules as hard constraints. The canvas note offers no
    more storeys than the method's own cap (``scale.method_max_storeys``: 2 for
    blueprint and floors). ``dream_cue`` is read by
    ``dreams.render_prompt`` (straight from the recipe, since dreams come before
    the system prompt): it adds :data:`scale.DREAM_CUE` after the dream style.
    """
    from blockgen.agentic import scale as scale_mod

    cap = scale_mod.method_max_storeys(getattr(ep, "spec", None))
    text = scale_mod.scale_rules(tuple(getattr(ep, "size", None) or (32, 32, 32)),
                                 strict=bool(strict), canvas=bool(canvas), max_storeys=cap)
    _notes(ep)["scale"] = {"strict": bool(strict), "canvas": bool(canvas),
                           "dream_cue": bool(dream_cue), "chars": len(text)}
    return ContextParts(system_text=text)


# ===========================================================================
# grid
# ===========================================================================
#: Which side of the build each named camera shows on the left and right of the
#: image. Measured, not derived: rendered a marker build (x=max wall, z=max wall,
#: z=0 row, pillar at the origin) through the textured renderer at each view on
#: 2026-10-05; tests/agentic/test_grid.py re-checks the 'iso' row on GPU machines.
VIEW_SIDES: Dict[str, Tuple[str, str]] = {
    "iso": ("z=max", "x=max"),
    "iso_back": ("z=0", "x=0"),
    "iso_left": ("x=max", "z=0"),
    "iso_right": ("x=0", "z=max"),
}

GRID_RULES = """\
Coordinate grid: x and z are horizontal, y is up, and (0,0,0) is a bottom corner of
the canvas. The ground is the x-z plane at y=0.
- A flat HORIZONTAL part (floor, slab roof, carpet, path) spans x and z at one y.
- A flat VERTICAL part (wall, sign, clock face, banner, windmill sails) spans y and ONE
  horizontal axis, with the other axis fixed: sails on the z=max face span x and y at a
  single z; sails on the x=max face span z and y at a single x. Before writing a flat
  shape, decide which axis it holds fixed.
- Anything that sticks out of a face (sail hub, balcony, awning) extends along the axis
  that the face holds fixed."""


def grid_text(views: Sequence[str] = ("iso",)) -> str:
    """:data:`GRID_RULES` plus where each named camera's left and right faces are."""
    lines = [GRID_RULES]
    known = [v for v in views if v in VIEW_SIDES]
    if known:
        lines.append("Cameras: TARGET images and renders of your build use these cameras.")
        for v in known:
            left, right = VIEW_SIDES[v]
            lines.append(f"- '{v}': the face on the LEFT of the image is the {left} side, "
                         f"the face on the RIGHT is the {right} side; the opposite two sides "
                         f"are hidden.")
        if "iso" in known:
            left, right = VIEW_SIDES["iso"]
            lines.append(f"Put the front (main door, facade, sails) on a side the main 'iso' "
                         f"camera sees ({left} or {right}), and copy the TARGET literally: a "
                         f"feature on the image's left face goes on the {left} side.")
    return "\n".join(lines)


@register("context", "grid")
def grid(ep: Any, *, views: Sequence[str] = ("iso",)) -> ContextParts:
    """Axis and camera orientation, as system text.

    Why: the system prompt only said "y is up". Builds made of floors, walls and
    roofs came out right anyway, but every windmill in pilot_diverse_v0 whose build
    did not copy a dream closely laid its sails flat in the x-z plane. Vertical flat
    features need the fixed axis spelled out, and dream methods need to know which
    side of the build each TARGET face is (the iso camera hides x=0 and z=0, the
    side the in-context example puts its door on).
    """
    text = grid_text(tuple(views))
    _notes(ep)["grid"] = {"views": list(views), "chars": len(text)}
    return ContextParts(system_text=text)


# ===========================================================================
# reference
# ===========================================================================
@register("context", "reference", images=True)
def reference(ep: Any, *, note: str = REFERENCE_NOTE) -> ContextParts:
    """The user's reference images (``ep.reference_images``), with the legacy note."""
    images = list(getattr(ep, "reference_images", None) or [])
    if not images:
        return ContextParts()
    return ContextParts(user_parts=[Part("text", text=note)]
                        + [Part("image", image_png=b) for b in images])


__all__ = ["BlockColor", "ExampleParts", "FACE_SHADES", "PALETTE_SIDE", "PaletteEntry", "REFERENCE_NOTE", "block_colors",
           "dream_palette", "examples", "examples_text", "foreground_lab", "kmeans",
           "match_palette", "ontology", "ontology_text", "palette", "prior_turns",
           "reference", "rgb_to_lab", "scale", "texture_sheet"]
