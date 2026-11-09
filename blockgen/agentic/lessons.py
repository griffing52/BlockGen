"""The ``lessons`` context: a curated knowledge base of building rules, retrieved
per request.

ldraw-nova's agent reads a curated knowledge base ("construction lessons") before
it plans; its authors keep research and lessons apart on purpose ("source
research, not automatically admitted construction lessons"). This is the same
device for the Minecraft builder: a YAML file of short rules, each with tags, and
a retriever that puts the ``always`` lessons plus the ``k`` best tag matches for
the request in the system prompt. The default file is
``configs/agentic/lessons.yaml``; a recipe may point at its own.

Retrieval is deliberately lexical (tag words against request words, with a plural
and a hyphen fold), the same device ``examples.select_examples`` uses, so the
arm's difference is the knowledge, not an embedding model. The chosen ids land in
``ep.notes['lessons']`` and the trace, so a run can tell which rules each build
saw. A control with the SAME token count and no knowledge is the ``shuffled``
flag: every lesson's text is replaced by the text of a lesson that does not match
the request (the ``texsheet-shuffled`` and ``dream-caption-shuffled`` pattern).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from blockgen.agentic.core import ContextParts, register

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LESSONS = _REPO / "configs" / "agentic" / "lessons.yaml"
HEADER = "BUILDING LESSONS (rules from inspected builds; follow the ones that apply):"


@dataclass(frozen=True)
class Lesson:
    id: str
    tags: Tuple[str, ...]
    text: str

    @property
    def always(self) -> bool:
        return "always" in self.tags


def _words(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z][a-z\-]+", text.lower()):
        out.add(w)
        out.add(w.replace("-", ""))
        out.update(part for part in w.split("-") if len(part) > 2)
        if w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
        if w.endswith("ies") and len(w) > 4:
            out.add(w[:-3] + "y")
    return out


@lru_cache(maxsize=8)
def load_lessons(path: Optional[str] = None) -> Tuple[Lesson, ...]:
    """Lessons from a YAML file ``{lessons: [{id, tags, text}, ...]}``."""
    import yaml

    p = Path(path) if path else DEFAULT_LESSONS
    if not p.is_file():
        raise FileNotFoundError(f"lessons file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    items = data.get("lessons") if isinstance(data, dict) else data
    out: List[Lesson] = []
    for i, it in enumerate(items or []):
        if not isinstance(it, dict) or not str(it.get("text", "")).strip():
            raise ValueError(f"{p}: lesson #{i} needs a 'text'")
        tags = tuple(str(t).strip().lower() for t in (it.get("tags") or []))
        out.append(Lesson(str(it.get("id") or f"lesson{i}"), tags, " ".join(str(it["text"]).split())))
    ids = [l.id for l in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{p}: duplicate lesson ids")
    return tuple(out)


def select_lessons(prompt: str, lessons: Sequence[Lesson], k: int = 6,
                   *, always: bool = True) -> List[Lesson]:
    """The ``always`` lessons (in file order) plus the ``k`` best by tag overlap with
    the request, ties broken by file order. A lesson with no overlap is never
    chosen, so a request a lesson does not fit costs no tokens."""
    words = _words(prompt)
    chosen = [l for l in lessons if l.always] if always else []
    rest = [l for l in lessons if l not in chosen]
    scored = []
    for i, l in enumerate(rest):
        hits = sum(1 for t in l.tags if t != "always" and (t in words or t.replace("-", "") in words))
        if hits:
            scored.append((-hits, i, l))
    scored.sort()
    return chosen + [l for _, _, l in scored[:max(0, int(k))]]


def shuffled_texts(chosen: Sequence[Lesson], pool: Sequence[Lesson], prompt: str,
                   seed: int) -> List[str]:
    """Token-matched control: the ``always`` lessons stay (both arms share them);
    each RETRIEVED lesson's text is swapped for the text of a lesson that does not
    match the request and is not shown. Non-matching lessons come first; if they
    run out, unshown matching ones follow, then the list cycles."""
    words = _words(prompt)
    shown = {l.id for l in chosen}
    unshown = [l for l in pool if l.id not in shown]
    non_match = [l for l in unshown if not any(t in words for t in l.tags if t != "always")]
    rest = [l for l in unshown if l not in non_match]
    rng = random.Random(seed)
    rng.shuffle(non_match)
    rng.shuffle(rest)
    others = non_match + rest
    out, j = [], 0
    for l in chosen:
        if l.always or not others:
            out.append(l.text)
        else:
            out.append(others[j % len(others)].text)
            j += 1
    return out


def lessons_text(texts: Sequence[str]) -> str:
    return "\n".join([HEADER] + [f"- {t}" for t in texts])


def _notes(ep: Any) -> Dict[str, Any]:
    notes = getattr(ep, "notes", None)
    if notes is None:
        notes = {}
        try:
            ep.notes = notes
        except AttributeError:
            pass
    return notes


@register("context", "lessons")
def lessons(ep: Any, *, k: int = 6, path: Optional[str] = None, always: bool = True,
            shuffled: bool = False, seed: int = 0) -> ContextParts:
    """Curated building rules retrieved for the request, as system text.

    ``k`` tag-matched lessons on top of the ``always`` ones; ``path`` another
    YAML file; ``shuffled`` the token-matched control (same count of lessons,
    texts of non-matching lessons).
    """
    pool = load_lessons(path)
    chosen = select_lessons(ep.prompt, pool, k=k, always=always)
    if not chosen:
        _notes(ep)["lessons"] = {"ids": [], "shuffled": bool(shuffled)}
        return ContextParts()
    texts = (shuffled_texts(chosen, pool, ep.prompt, int(seed) + int(getattr(ep, "seed", 0) or 0))
             if shuffled else [l.text for l in chosen])
    _notes(ep)["lessons"] = {"ids": [l.id for l in chosen], "shuffled": bool(shuffled)}
    trace = getattr(ep, "trace", None)
    if trace is not None:
        trace.event("lessons", ids=[l.id for l in chosen], shuffled=bool(shuffled))
    return ContextParts(system_text=lessons_text(texts))


__all__ = ["DEFAULT_LESSONS", "HEADER", "Lesson", "lessons", "lessons_text", "load_lessons",
           "select_lessons", "shuffled_texts"]
