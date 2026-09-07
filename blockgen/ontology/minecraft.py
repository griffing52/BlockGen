"""Compose the Minecraft block catalog: mined statistics + textures + rules.

This is the module that turns measurements into an ontology. The three sources
(:mod:`~blockgen.ontology.mine`, :mod:`~blockgen.ontology.appearance`,
:mod:`~blockgen.ontology.authored`) each answer a different question, and this
file is where they are joined onto the ~70 names the agentic track actually
advertises to the model.

Two decisions here are worth defending.

**The vocabulary is closed.** Every block named in the table -- including inside a
``pairs with`` list -- is a palette entry the DSL can resolve. A neighbour list
that mentions "Double Oak Wood Slab" would teach the model a name that fails to
parse, converting knowledge into per-line errors. Neighbours outside the palette
are dropped, not renamed.

**The thresholds that turn numbers into words live here, in one place, visible.**
``height_mean = 0.61`` becomes ``roof`` because of a cutoff written on line ~90,
not because of a rule buried in the miner. A reader who disagrees with "post"
should be able to find the number and the comparison in one screen, and an
ablation that wants raw numbers instead of words can ask for the ``stats.*``
fields, which carry them unrounded.

Palette collisions are recorded, not hidden: ``stripped_oak_log`` and ``oak_log``
remap to the same legacy pair, so they carry identical mined statistics and say
so in ``notes``. That is a real property of the family-level representation
(``docs/agentic.md``, "Known limits") and the ontology should not pretend
otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from blockgen.agentic.blockstate import PALETTE, resolve_block
from blockgen.eval.bench.palette import block_family
from blockgen.ontology import appearance as app
from blockgen.ontology import authored
from blockgen.ontology.mine import BlockStats, CorpusStats, mine_corpus, symbol_for
from blockgen.ontology.schema import AttributeSpec, Catalog, Part

DEFAULT_CORPUS = "data/minecraft/cache/houses_32.npz"

#: Corpus categories that carry no style information. ``house`` is the catch-all
#: label on half the corpus, so every block is "over-represented" in it by
#: accident of arithmetic; printing it would spend tokens saying nothing.
GENERIC_CATEGORIES = frozenset({"house", "houses", "building", "other", ""})
DEFAULT_OUT = Path("data/ontology")

#: Minimum placements before a block's mined tags are trusted. Below this the
#: numbers are still reported (in ``stats.*``) but the words are withheld -- a
#: role inferred from 40 placements is noise wearing a label.
MIN_SUPPORT = 200

SCHEMA: Tuple[AttributeSpec, ...] = (
    # Not in the prompt: the block name already says "oak_planks is wood", and a
    # 70-row column that restates the name is the cheapest thing to cut.
    AttributeSpec("category", "text", "derived", "material family", header="kind",
                  prompt=False),
    AttributeSpec("appearance.color", "color", "asset",
                  "mean texture colour (hex + name)", header="colour"),
    AttributeSpec("appearance.look", "text", "asset",
                  "surface: flat/subtle/busy, and cutout/translucent if not solid",
                  header="look"),
    AttributeSpec("build.layer", "text", "mined",
                  "where it sits in real builds: ground/low/mid/upper/roof",
                  header="layer"),
    AttributeSpec("build.form", "text", "mined",
                  "how it is laid: post (vertical runs), plate (flat courses), "
                  "mass (bulk), trim (mostly exposed faces)", header="form"),
    AttributeSpec("build.frequency", "number", "mined",
                  "share of all blocks placed in the corpus, %", header="freq%"),
    AttributeSpec("build.pairs_with", "refs", "mined",
                  "blocks it touches far more often than chance", header="pairs with"),
    AttributeSpec("build.styles", "tags", "mined",
                  "build categories it is over-represented in", header="styles"),
    AttributeSpec("physical.notes", "tags", "authored",
                  "flammable / falls / lightN / not-full (leaves gaps in a wall)",
                  header="physical"),
    # Three families take state properties; a 70-row column to say so is waste,
    # so this rides in the prompt as a one-line footnote instead (`prompt_block`).
    AttributeSpec("physical.states", "text", "derived",
                  "state properties the executor honours, e.g. oak_stairs[facing=north]",
                  header="states", prompt=False),
    # Raw measurements: kept out of the prompt (they cost tokens and say the same
    # thing as the words) but carried in the JSON for the lab and for an arm that
    # wants to test numbers-instead-of-words.
    AttributeSpec("stats.count", "number", "mined", "placements in the corpus",
                  prompt=False),
    AttributeSpec("stats.builds_frac", "number", "mined",
                  "fraction of builds containing it", prompt=False),
    AttributeSpec("stats.height_mean", "number", "mined",
                  "mean normalized height within its build", prompt=False),
    AttributeSpec("stats.vertical_run", "number", "mined",
                  "mean contiguous run length along y", prompt=False),
    AttributeSpec("stats.horizontal_run", "number", "mined",
                  "mean contiguous run length along x/z", prompt=False),
    AttributeSpec("stats.anisotropy", "number", "mined",
                  "vertical run / horizontal run", prompt=False),
    AttributeSpec("stats.support_frac", "number", "mined",
                  "fraction with something solid directly below", prompt=False),
    AttributeSpec("stats.exposure", "number", "mined",
                  "mean air-facing sides, 0-6", prompt=False),
    AttributeSpec("stats.self_affinity", "number", "mined",
                  "fraction of its faces touching its own kind", prompt=False),
    AttributeSpec("stats.neighbors", "refs", "mined",
                  "top neighbours as (block, npmi, share, pairs)", prompt=False),
)


# --- numbers -> words ------------------------------------------------------
def layer_word(height_mean: float) -> str:
    """Where a block sits, as a word. Cutoffs are even fifths of build height
    except for the first, pulled down to 0.20 so that "ground" means foundation
    and landscaping rather than the whole lower storey."""
    if height_mean < 0.20:
        return "ground"
    if height_mean < 0.35:
        return "low"
    if height_mean < 0.50:
        return "mid"
    if height_mean < 0.62:
        return "upper"
    return "roof"


def form_word(stats: BlockStats) -> str:
    """How a block is laid, from run-length anisotropy and face exposure.

    Ordered rules, first match wins:

    * ``trim``  exposure >= 2.4 of 6 faces open -- torches, fences, trapdoors,
      walls: detail hung on a surface rather than part of it.
    * ``post``  vertical runs >= 1.3x horizontal -- logs, ladders, doors: the
      vertical members.
    * ``plate`` horizontal runs >= 2.3 and clearly flatter than tall -- floors,
      terraces, flat roofs.
    * ``mass``  everything else: the bulk material a wall is made of.

    ``trim`` is tested before ``post`` because an exposed fence is a fence, not a
    column, and the exposure signal is the one a builder can act on.
    """
    if stats.exposure >= 2.4:
        return "trim"
    if stats.anisotropy >= 1.3:
        return "post"
    if stats.horizontal_run >= 2.3 and stats.anisotropy <= 0.7:
        return "plate"
    return "mass"


def look_word(a: Optional[app.Appearance]) -> Optional[str]:
    if a is None:
        return None
    return a.texture if a.transparency == "opaque" else f"{a.texture} {a.transparency}"


# --- palette <-> corpus symbols --------------------------------------------
def palette_symbols(palette: Sequence[str] = PALETTE) -> Dict[str, str]:
    """``palette name -> corpus symbol``."""
    return {name: symbol_for(*resolve_block(name)) for name in palette}


def _symbol_to_palette(palette: Sequence[str]) -> Dict[str, str]:
    """``corpus symbol -> the palette name to print for it``.

    First name wins, so ``oak_log`` is printed rather than ``stripped_oak_log``:
    the palette is ordered with the canonical member of each family first.
    """
    out: Dict[str, str] = {}
    for name in palette:
        out.setdefault(symbol_for(*resolve_block(name)), name)
    return out


def build_catalog(stats: CorpusStats, *, palette: Sequence[str] = PALETTE,
                  pack_dir: Path | str = app.DEFAULT_PACK_DIR,
                  min_support: int = MIN_SUPPORT, n_pairs: int = 3,
                  n_styles: int = 2, name: str = "") -> Catalog:
    """Join mined stats, textures and rules onto the palette."""
    to_palette = _symbol_to_palette(palette)
    by_symbol: Dict[str, List[str]] = {}
    for entry in palette:
        by_symbol.setdefault(symbol_for(*resolve_block(entry)), []).append(entry)

    parts: Dict[str, Part] = {}
    for entry in palette:
        pair = resolve_block(entry)
        symbol = symbol_for(*pair)
        block = stats.get(symbol)
        look = app.measure(*pair, pack_dir=pack_dir)
        attrs: Dict[str, object] = {"category": block_family(*pair)}
        if look is not None:
            attrs["appearance.color"] = f"{look.hex_color} {look.color_name}"
            attrs["appearance.look"] = look_word(look)
        attrs.update(authored.physical_attrs(entry))

        support = int(block.count) if block else 0
        if block is not None:
            attrs.update({
                "stats.count": block.count,
                "stats.builds_frac": round(block.builds_frac, 4),
                "stats.height_mean": round(block.height_mean, 4),
                "stats.vertical_run": round(block.vertical_run, 3),
                "stats.horizontal_run": round(block.horizontal_run, 3),
                "stats.anisotropy": round(block.anisotropy, 3),
                "stats.support_frac": round(block.support_frac, 4),
                "stats.exposure": round(block.exposure, 3),
                "stats.self_affinity": round(block.self_affinity, 4),
                "stats.neighbors": [[n[0], round(n[1], 3), round(n[2], 4), n[3]]
                                    for n in block.neighbors],
            })
        if block is not None and block.count >= min_support:
            pairs: List[str] = []
            for partner, _npmi, _share, _n in block.neighbors:
                mapped = to_palette.get(partner)
                if mapped and mapped != entry and mapped not in pairs:
                    pairs.append(mapped)
                if len(pairs) >= n_pairs:
                    break
            attrs.update({
                "build.layer": layer_word(block.height_mean),
                "build.form": form_word(block),
                "build.frequency": round(100.0 * block.share, 2),
                "build.pairs_with": pairs,
                "build.styles": [c for c, lift, _ in block.category_lift
                                 if lift >= 1.3
                                 and c not in GENERIC_CATEGORIES][:n_styles],
            })

        siblings = [s for s in by_symbol.get(symbol, []) if s != entry]
        notes = ""
        if siblings:
            notes = ("same legacy block as " + ", ".join(siblings) +
                     " after family remap; their statistics are identical")
        elif block is None:
            notes = "not present in the mined corpus"
        elif block.count < min_support:
            notes = f"only {block.count} placements; tags withheld"
        parts[entry] = Part(id=entry, category=str(attrs["category"]), attrs=attrs,
                            support=support, notes=notes)

    return Catalog(
        domain="minecraft", name=name or f"minecraft/{stats.corpus or 'corpus'}",
        schema=SCHEMA, parts=parts,
        meta={"corpus": stats.corpus, "n_builds": stats.n_builds,
              "n_placements": stats.n_placements, "min_support": min_support,
              "palette_size": len(palette),
              "categories": stats.categories,
              "n_with_tags": sum(1 for p in parts.values()
                                 if "build.layer" in p.attrs)})


def mine_and_build(corpus_path: str = DEFAULT_CORPUS, *,
                   palette: Sequence[str] = PALETTE,
                   limit: int = 0, min_support: int = MIN_SUPPORT,
                   pack_dir: Path | str = app.DEFAULT_PACK_DIR,
                   progress_every: int = 0) -> Catalog:
    """Load a structure cache, mine it, and return the catalog. The one-call path
    used by the CLI, the tests and the lab."""
    from blockgen.curation.houses import load_structures_from_cache

    structures, _ = load_structures_from_cache(corpus_path)
    if limit:
        structures = structures[:limit]
    corpus = Path(corpus_path).stem + (f"[:{limit}]" if limit else "")
    stats = mine_corpus(structures, corpus=corpus, min_support=min_support,
                        top_neighbors=25, progress_every=progress_every)
    return build_catalog(stats, palette=palette, pack_dir=pack_dir,
                         min_support=min_support)


#: Widths that keep the table inside a sane token budget. The two list columns
#: are the ones that blow up (three block names is 40 characters), and a truncated
#: third neighbour costs less than the tokens it would take to print in full.
PROMPT_WIDTHS = 34


def prompt_block(catalog: Catalog, fields: Sequence[str] = (),
                 part_ids: Sequence[str] = ()) -> str:
    """The catalog as it appears in the system prompt: heading, table, footnote.

    The footnote carries the state-property grammar for the three families that
    have one -- as one line rather than a mostly-empty 70-row column.
    """
    table = catalog.render_table(fields, part_ids, max_width=PROMPT_WIDTHS)
    if not table:
        return ""
    corpus = catalog.meta.get("corpus") or "the corpus"
    n_builds = catalog.meta.get("n_builds") or 0
    head = (f"Block reference — every column below is MEASURED from {n_builds} real "
            f"builds ({corpus}) or from the block textures, not from taste. Use it "
            f"to pick materials that go together and to put them where they belong; "
            f"you may still use any block name you know.")
    foot = ("Orientable blocks: <wood>_stairs[facing=north|south|east|west,half=top], "
            "<wood>_log[axis=x|y|z], <stone>_slab[type=top|bottom].")
    return f"{head}\n\n{table}\n{foot}"


def default_path(corpus: str = "houses_32", out_dir: Path | str = DEFAULT_OUT) -> Path:
    return Path(out_dir) / f"minecraft_{corpus}.json"


__all__ = ["DEFAULT_CORPUS", "DEFAULT_OUT", "GENERIC_CATEGORIES", "MIN_SUPPORT", "PROMPT_WIDTHS", "SCHEMA",
           "build_catalog", "default_path", "form_word", "layer_word",
           "mine_and_build", "palette_symbols", "prompt_block"]
