"""A domain-agnostic parts catalog: what a build is made of, and what we know
about each material.

Why this is not a Minecraft module
----------------------------------
The agentic track's thesis is that an LLM builds better when it writes a
*program* over a *known vocabulary*. The vocabulary today is
``blockgen.agentic.blockstate.PALETTE`` -- 70 bare strings. This module is the
same list with knowledge attached, and the end goal is LEGO parts and electronic
schematics, where the catalog is thousands of entries and the model cannot be
assumed to know any of them. So the container is written once, generically:

* :class:`Part`   -- one entry, a flat ``dotted.key -> value`` attribute bag.
* :class:`Catalog` -- a set of parts, plus the :class:`AttributeSpec` schema that
  says what each field means, where it came from, and whether it is worth its
  tokens in a prompt.

``MinecraftCatalog`` (in :mod:`blockgen.ontology.minecraft`) is the first
backend. A LEGO backend is a new miner plus a new schema, not a rewrite.

Provenance is a first-class field, not a comment
------------------------------------------------
Every attribute declares a ``source``:

``mined``     measured from the corpus (:mod:`blockgen.ontology.mine`) -- the
              only fields that can carry information the model does not already
              have.
``asset``     measured from the shipped textures (:mod:`blockgen.ontology.appearance`).
``authored``  written by a human from domain knowledge.
``derived``   computed from repo code (e.g. which state properties a block accepts).

This matters because the experiment is "does a *grounded* ontology beat the
model's own priors": an ontology that is 90% authored is mostly the model being
told what it already believes, and the arms have to be able to say which fields
they are actually testing. :meth:`Catalog.subset_by_source` is that seam.

The shuffled control
--------------------
:meth:`Catalog.shuffled` permutes each field's values across parts
independently. Marginal distributions, table size and token count are *identical*
to the real catalog; only the block-to-attribute association is destroyed. If an
arm scores the same on a shuffled ontology as on the real one, the gain was extra
tokens rather than knowledge -- which is the first question any reader will ask,
so the control ships with the catalog rather than being bolted on later.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SOURCES = ("mined", "asset", "authored", "derived")

#: Rendered when a part has no value for a field. One character, because it
#: appears in a table the model reads and "unknown" costs a token per cell.
MISSING = "-"


@dataclass(frozen=True)
class AttributeSpec:
    """One column of the catalog: its meaning, provenance and prompt budget."""

    name: str                 # dotted path, e.g. "appearance.primary_color"
    kind: str                 # text | number | tags | color | refs | bool
    source: str               # one of SOURCES
    describe: str             # one line, shown to the model as a column legend
    prompt: bool = True       # include in render_table() by default
    header: str = ""          # short column header; defaults to the leaf name

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: source must be one of {SOURCES}, "
                             f"got {self.source!r}")

    @property
    def label(self) -> str:
        return self.header or self.name.rsplit(".", 1)[-1]

    def to_json(self) -> dict:
        return {"name": self.name, "kind": self.kind, "source": self.source,
                "describe": self.describe, "prompt": self.prompt,
                "header": self.header}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "AttributeSpec":
        return cls(name=str(d["name"]), kind=str(d.get("kind", "text")),
                   source=str(d.get("source", "authored")),
                   describe=str(d.get("describe", "")),
                   prompt=bool(d.get("prompt", True)),
                   header=str(d.get("header", "")))


@dataclass
class Part:
    """One catalog entry.

    ``attrs`` is flat with dotted keys (``"positional.role"``) rather than nested,
    because every consumer -- the prompt table, the lab's column picker, the
    shuffle -- wants to address a field by one string. :meth:`nested` puts the
    tree back for display.

    ``support`` is the evidence count behind the mined fields (placements in the
    corpus). It is carried explicitly so a reader can tell "oak planks, measured
    over 400k placements" from "podzol, measured over 12", and so the miner can
    withhold tags it does not have the data for.
    """

    id: str
    category: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)
    support: int = 0
    notes: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        return self.attrs.get(name, default)

    def nested(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.attrs.items():
            node = out
            *parents, leaf = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return out

    def to_json(self) -> dict:
        return {"id": self.id, "category": self.category, "attrs": self.attrs,
                "support": self.support, "notes": self.notes}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Part":
        return cls(id=str(d["id"]), category=str(d.get("category", "")),
                   attrs=dict(d.get("attrs") or {}),
                   support=int(d.get("support", 0)),
                   notes=str(d.get("notes", "")))


def _fmt(value: Any) -> str:
    """One cell, as the model will read it. Lists become dot-joined, floats get
    two decimals, ``None`` becomes :data:`MISSING`."""
    if value is None or value == "" or value == []:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return "·".join(_fmt(v) for v in value) or MISSING
    return str(value)


@dataclass
class Catalog:
    """A set of parts over one schema, plus the provenance of the whole thing."""

    domain: str
    name: str
    schema: Tuple[AttributeSpec, ...] = ()
    parts: Dict[str, Part] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- basics -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.parts)

    def __contains__(self, part_id: str) -> bool:
        return part_id in self.parts

    def __iter__(self):
        return iter(self.parts.values())

    def get(self, part_id: str) -> Optional[Part]:
        return self.parts.get(part_id)

    def ids(self) -> List[str]:
        return list(self.parts)

    def spec(self, name: str) -> Optional[AttributeSpec]:
        return next((s for s in self.schema if s.name == name), None)

    def field_names(self, *, prompt_only: bool = False,
                    sources: Sequence[str] = ()) -> List[str]:
        return [s.name for s in self.schema
                if (not prompt_only or s.prompt)
                and (not sources or s.source in sources)]

    # --- selection --------------------------------------------------------
    def subset(self, part_ids: Iterable[str]) -> "Catalog":
        keep = [p for p in part_ids if p in self.parts]
        return Catalog(domain=self.domain, name=self.name, schema=self.schema,
                       parts={p: self.parts[p] for p in keep},
                       meta={**self.meta, "subset_of": self.name, "n": len(keep)})

    def subset_by_source(self, sources: Sequence[str]) -> "Catalog":
        """Drop every attribute whose spec is not in ``sources``.

        This is what makes "mined only" vs "authored only" an ablation you can
        run rather than a claim you make: the two catalogs then differ in
        content, not in which blocks exist.
        """
        keep = {s.name for s in self.schema if s.source in sources}
        parts = {pid: Part(id=p.id, category=p.category, support=p.support,
                           notes=p.notes,
                           attrs={k: v for k, v in p.attrs.items() if k in keep})
                 for pid, p in self.parts.items()}
        return Catalog(domain=self.domain, name=f"{self.name}:{'+'.join(sources)}",
                       schema=tuple(s for s in self.schema if s.name in keep),
                       parts=parts,
                       meta={**self.meta, "sources": list(sources)})

    def shuffled(self, seed: int = 0) -> "Catalog":
        """The control arm: same values, wrong blocks.

        Each field is permuted independently, so every marginal (the set of
        colours, the set of role tags, the neighbour lists and their lengths) is
        preserved exactly and the rendered table has the same shape and very
        nearly the same token count. ``category`` and ``support`` ride along with
        the part id -- they are identity, not knowledge.

        The one repair applied afterwards: a permuted ``refs`` list can land a
        block in its own neighbour list ("oak_log pairs with oak_log"), which is
        a visible tell that the table is scrambled. Self-references are dropped so
        the control looks exactly as trustworthy as the real thing.
        """
        rng = random.Random(seed)
        ids = sorted(self.parts)
        parts = {pid: Part(id=pid, category=self.parts[pid].category,
                           support=self.parts[pid].support,
                           notes=self.parts[pid].notes, attrs={})
                 for pid in ids}
        for spec in self.schema:
            values = [self.parts[pid].attrs.get(spec.name) for pid in ids]
            order = list(range(len(ids)))
            rng.shuffle(order)
            for pid, j in zip(ids, order):
                value = values[j]
                if value is None:
                    continue
                if spec.kind == "refs" and isinstance(value, list):
                    value = [v for v in value if v != pid]
                parts[pid].attrs[spec.name] = value
        return Catalog(domain=self.domain, name=f"{self.name}:shuffled{seed}",
                       schema=self.schema, parts=parts,
                       meta={**self.meta, "shuffled_seed": seed,
                             "shuffled_from": self.name})

    # --- prompt rendering -------------------------------------------------
    def render_table(self, fields: Sequence[str] = (), part_ids: Sequence[str] = (),
                     *, legend: bool = True, max_width: int = 0) -> str:
        """The catalog as the model sees it: a fixed-width table plus a legend.

        A table, not JSON. The same content as JSON costs ~5x the tokens in
        braces, quotes and repeated key names -- and the key names are exactly
        what the legend says once. 70 blocks x 6 fields lands around 1.4k tokens,
        which is affordable in a system prompt that is re-sent every round.
        """
        names = list(fields) if fields else self.field_names(prompt_only=True)
        specs = [self.spec(n) for n in names]
        specs = [s for s in specs if s is not None]
        ids = list(part_ids) if part_ids else sorted(self.parts)
        if not ids or not specs:
            return ""

        headers = ["block"] + [s.label for s in specs]
        rows: List[List[str]] = []
        for pid in ids:
            part = self.parts[pid]
            rows.append([pid] + [_fmt(part.attrs.get(s.name)) for s in specs])
        widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
        if max_width:
            widths = [min(w, max_width) for w in widths]

        def line(cells: Sequence[str]) -> str:
            out = []
            for cell, width in zip(cells, widths):
                cell = cell if len(cell) <= width else cell[:width - 1] + "…"
                out.append(cell.ljust(width))
            return "  ".join(out).rstrip()

        body = [line(headers), line(["-" * w for w in widths])]
        body += [line(r) for r in rows]
        if not legend:
            return "\n".join(body)
        legend_lines = [f"  {s.label}: {s.describe}" for s in specs if s.describe]
        return "\n".join(body + ["", "Columns:"] + legend_lines)

    # --- serialization ----------------------------------------------------
    def to_json(self) -> dict:
        return {"domain": self.domain, "name": self.name, "meta": self.meta,
                "schema": [s.to_json() for s in self.schema],
                "parts": [p.to_json() for p in self.parts.values()]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Catalog":
        parts = [Part.from_json(p) for p in d.get("parts") or []]
        return cls(domain=str(d.get("domain", "")), name=str(d.get("name", "")),
                   schema=tuple(AttributeSpec.from_json(s) for s in d.get("schema") or []),
                   parts={p.id: p for p in parts},
                   meta=dict(d.get("meta") or {}))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=False))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Catalog":
        return cls.from_json(json.loads(Path(path).read_text()))


__all__ = ["AttributeSpec", "Catalog", "MISSING", "Part", "SOURCES"]
