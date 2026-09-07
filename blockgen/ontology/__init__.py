"""Block ontology: what each material *is*, measured rather than asserted.

The agentic track (``docs/agentic.md``) hands a frontier model a list of ~70
block names and asks for a build program. This package attaches knowledge to
those names -- colour and surface from the shipped textures, placement behaviour
and material affinity mined from the corpus, physical rules written down once --
and renders it as a table the model reads in its system prompt.

The experiment it exists to run: **does a grounded ontology beat the model's own
priors?** So provenance is tracked per field (``mined`` / ``asset`` /
``authored`` / ``derived``), and :meth:`Catalog.shuffled` ships a token-matched
control whose attributes are attached to the wrong blocks. An arm that scores the
same on the shuffled catalog was helped by tokens, not knowledge.

    from blockgen.ontology import load_catalog, prompt_block
    catalog = load_catalog()                 # data/ontology/minecraft_houses_32.json
    print(prompt_block(catalog))             # what the model sees

Build or rebuild it:

    python -m blockgen.ontology --corpus data/minecraft/cache/houses_32.npz

The container (:mod:`blockgen.ontology.schema`) is domain-agnostic on purpose:
LEGO parts and electronic components are the next catalogs, and they should be a
new miner plus a new schema rather than a fork of this one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from blockgen.ontology.minecraft import (DEFAULT_CORPUS, SCHEMA, build_catalog,
                                         default_path, mine_and_build,
                                         prompt_block)
from blockgen.ontology.schema import AttributeSpec, Catalog, Part

#: Ontology variants an experiment arm can ask for by name.
#:
#: ``none``      no ontology (the baseline: today's bare palette list)
#: ``mined``     the full catalog -- mined + asset + authored + derived
#: ``shuffled``  the control: same table, attributes permuted across blocks
#: ``stats``     mined numbers only, no authored or asset fields
VARIANTS = ("none", "mined", "shuffled", "stats")


@lru_cache(maxsize=8)
def load_catalog(path: Optional[str] = None) -> Optional[Catalog]:
    """Load the saved catalog, or ``None`` if it has not been built.

    ``None`` rather than an exception: an ontology arm should fail loudly at the
    point that *asks* for it, while everything else in the repo (the lab's index
    page, an unrelated agentic run) must work on a clone that never ran the miner.
    """
    target = Path(path) if path else default_path()
    if not target.is_file():
        return None
    return Catalog.load(target)


def variant_catalog(variant: str = "mined", path: Optional[str] = None,
                    seed: int = 0) -> Optional[Catalog]:
    """The catalog an arm named ``variant`` should see."""
    if variant in ("none", "", None):
        return None
    catalog = load_catalog(path)
    if catalog is None:
        return None
    if variant == "mined":
        return catalog
    if variant == "shuffled":
        return catalog.shuffled(seed)
    if variant == "stats":
        return catalog.subset_by_source(("mined",))
    raise ValueError(f"unknown ontology variant {variant!r}; expected one of {VARIANTS}")


__all__ = ["AttributeSpec", "Catalog", "DEFAULT_CORPUS", "Part", "SCHEMA", "VARIANTS",
           "build_catalog", "default_path", "load_catalog", "mine_and_build",
           "prompt_block", "variant_catalog"]
