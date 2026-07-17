"""Free category labels from PlanetMinecraft TAGS (text2mc metadata).

Every text2mc build ships a ``TAGS`` string in ``processed_build_dataframe.csv``
(a Python-list repr like ``"['Medieval', 'City', 'Land Structure']"``) scraped
from its PlanetMinecraft page. Those tags are a coarse, human-assigned
categorization we get for *free* — no VLM call. This module normalizes them into
a build *type* (house / castle / church / tower / …) and a *style* (medieval /
modern / fantasy / …), and composes a caption-ready category phrase.

Used by :func:`blockgen.curation.houses.load_text2mc_schem_houses` to set each
structure's ``category`` metadata, which then flows into template captions and
conditioning embeddings alongside (or ahead of) any VLM caption.
"""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple

# Build type: first match wins (order = specificity). Keyword -> canonical type.
_TYPE_RULES: List[Tuple[str, str]] = [
    ("castle", "castle"), ("palace", "castle"), ("fortress", "castle"),
    ("keep", "castle"), ("citadel", "castle"),
    ("church", "church"), ("cathedral", "church"), ("chapel", "church"),
    ("temple", "temple"), ("shrine", "temple"), ("pagoda", "temple"),
    ("tower", "tower"), ("lighthouse", "tower"), ("windmill", "tower"),
    ("barn", "farm"), ("farmhouse", "house"), ("farm", "farm"),
    ("mansion", "mansion"), ("manor", "mansion"), ("villa", "mansion"),
    ("cottage", "house"), ("cabin", "house"), ("hut", "house"),
    ("house", "house"), ("home", "house"), ("residence", "house"),
    ("shop", "shop"), ("tavern", "shop"), ("inn", "shop"), ("market", "shop"),
    ("bridge", "bridge"), ("statue", "statue"), ("fountain", "fountain"),
    ("ship", "ship"), ("boat", "ship"), ("garden", "garden"),
]

# Style modifiers (adjective kept as-is in the phrase). Keyword -> canonical style.
_STYLE_RULES: List[Tuple[str, str]] = [
    ("medieval", "medieval"), ("fantasy", "fantasy"), ("modern", "modern"),
    ("futuristic", "futuristic"), ("sci-fi", "futuristic"), ("scifi", "futuristic"),
    ("victorian", "victorian"), ("rustic", "rustic"), ("japanese", "japanese"),
    ("asian", "asian"), ("desert", "desert"), ("nordic", "nordic"),
    ("viking", "nordic"), ("gothic", "gothic"), ("steampunk", "steampunk"),
    ("survival", "survival"), ("suburban", "suburban"),
]


def parse_tags(tags: str | List[str] | None) -> List[str]:
    """Parse a raw TAGS field (list-repr string or list) into a list of strings."""
    if not tags:
        return []
    if isinstance(tags, list):
        return [str(t) for t in tags]
    try:
        val = ast.literal_eval(tags)
        if isinstance(val, (list, tuple)):
            return [str(t) for t in val]
    except (ValueError, SyntaxError):
        pass
    return [tags]


def _match(text: str, rules: List[Tuple[str, str]]) -> Optional[str]:
    for keyword, canonical in rules:
        if keyword in text:
            return canonical
    return None


def classify_tags(tags: str | List[str] | None,
                  url: str = "") -> Tuple[str, Optional[str]]:
    """Return ``(build_type, style)`` from TAGS (+ URL as a fallback text source).

    ``build_type`` defaults to ``"house"`` (these are house-filtered builds);
    ``style`` is ``None`` when no known style keyword is present.
    """
    text = " ".join(parse_tags(tags)).lower() + " " + (url or "").lower()
    return (_match(text, _TYPE_RULES) or "house", _match(text, _STYLE_RULES))


def category_phrase(tags: str | List[str] | None, url: str = "") -> str:
    """Caption-ready category, e.g. ``"medieval castle"`` or ``"house"``."""
    build_type, style = classify_tags(tags, url)
    return f"{style} {build_type}" if style else build_type
