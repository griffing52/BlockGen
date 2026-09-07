"""The set of servable models, read from ``models.json``.

Backends are constructed eagerly (cheap: just a spec) but their weights load lazily
on first use, so a broken or half-trained entry costs nothing until someone asks for
it and `/model` can still list everything. ``available()`` reports why an entry is
unusable rather than hiding it — a model whose vocab was never saved should be
visible and explained, not silently missing.

**Groups.** One entry can serve many models. The agentic entry is a provider, not a
checkpoint, so enumerating ``agentic:gpt-5-mini``, ``agentic:gemini-3.5-flash``, …
as separate registry rows would bury the four trained models under a wall of API
names in ``/model``. Instead an entry may declare ``"models": [...]``; the group
shows as **one** row, and ``get("agentic:gemini-3.5-flash")`` derives a backend for
that member on demand (cached after the first request). ``describe(group="agentic")``
lists the members — that is what ``/model agentic list`` calls.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from blockgen_server.backends import Backend, ModelSpec, build_backend

# Separates a group entry from the member it serves: "agentic:gpt-5-mini".
MEMBER_SEP = ":"


class Registry:
    def __init__(self, config_path: Path, repo_root: Path) -> None:
        blob = json.loads(Path(config_path).read_text())
        self.repo_root = repo_root
        self.default = blob["default"]
        self.entries: Dict[str, dict] = {}
        self.backends: Dict[str, Backend] = {}
        self._derived: Dict[str, Backend] = {}
        for name, entry in blob["models"].items():
            self.entries[name] = entry
            self.backends[name] = self._build(name, entry)
        if self.resolve(self.default) is None:
            raise ValueError(f"default model {self.default!r} is not in models.json")

    def _build(self, name: str, entry: dict) -> Backend:
        known = ModelSpec.__dataclass_fields__.keys()
        spec = ModelSpec(name=name, **{k: v for k, v in entry.items() if k in known})
        spec.extra = {k: v for k, v in entry.items() if k not in known}
        return build_backend(spec, self.repo_root)

    # --- groups -----------------------------------------------------------
    def groups(self) -> Dict[str, List[str]]:
        """``{group name: [member, …]}`` for every entry that serves many models."""
        return {name: b.group_members() for name, b in self.backends.items()
                if b.group_members()}

    def _derive(self, group: str, member: str) -> Backend:
        """Build (and cache) the backend for one member of a group."""
        full = f"{group}{MEMBER_SEP}{member}"
        if full in self._derived:
            return self._derived[full]
        base = self.backends[group]
        allowed = base.group_members()
        if allowed and member not in allowed and self.entries[group].get(
                "strict_models", True):
            raise KeyError(
                f"unknown model {full!r}; {group} serves {sorted(allowed)}. "
                f'Set "strict_models": false on the {group} entry in models.json '
                f"to allow any model name through.")
        entry = copy.deepcopy(self.entries[group])
        entry["provider"] = member
        entry["description"] = f"{base.spec.description} [{member}]"
        backend = self._build(full, entry)
        self._derived[full] = backend
        return backend

    # --- lookup -----------------------------------------------------------
    def names(self) -> List[str]:
        return list(self.backends)

    def resolve(self, name: Optional[str]) -> Optional[Backend]:
        """Name -> backend without loading it. ``None`` if the name is unknown."""
        name = name or self.default
        if name in self.backends:
            return self.backends[name]
        if MEMBER_SEP in name:
            group, _, member = name.partition(MEMBER_SEP)
            if group in self.backends and self.backends[group].group_members():
                return self._derive(group, member)
        return None

    def get(self, name: Optional[str]) -> Backend:
        b = self.resolve(name)
        if b is None:
            raise KeyError(f"unknown model {name!r}; have {sorted(self.backends)} "
                           f"(for a group, use <group>{MEMBER_SEP}<model>)")
        if not b.is_loaded():
            b.load()
        return b

    # --- description ------------------------------------------------------
    def missing_files(self, b: Backend) -> List[str]:
        """Which declared artifacts are absent — the usual reason a model can't run."""
        out = []
        for rel in (b.spec.checkpoint, b.spec.piece_vocab, b.spec.block_vocab):
            if rel and not b.path(rel).exists():
                out.append(str(rel))
        return out

    def _row(self, name: str, b: Backend) -> dict:
        info = b.info()
        info["name"] = name
        info["default"] = (name == self.default)
        missing = self.missing_files(b)
        reason = f"missing: {', '.join(missing)}" if missing else b.unavailable_reason()
        info["available"] = reason is None
        if reason:
            info["unavailable_reason"] = reason
        members = b.group_members()
        if members:
            info["group_size"] = len(members)
            info["members"] = members
        return info

    def describe(self, group: Optional[str] = None) -> List[dict]:
        """Rows for ``/model``. With ``group`` set, rows for that group's members."""
        if group:
            base = self.backends.get(group)
            if base is None or not base.group_members():
                raise KeyError(f"{group!r} is not a model group; groups: "
                               f"{sorted(self.groups())}")
            return [self._row(f"{group}{MEMBER_SEP}{m}", self._derive(group, m))
                    for m in base.group_members()]
        return [self._row(name, b) for name, b in self.backends.items()]
