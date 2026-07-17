"""The set of servable models, read from ``models.json``.

Backends are constructed eagerly (cheap: just a spec) but their weights load lazily
on first use, so a broken or half-trained entry costs nothing until someone asks for
it and `/model` can still list everything. ``available()`` reports why an entry is
unusable rather than hiding it — a model whose vocab was never saved should be
visible and explained, not silently missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from blockgen_server.backends import Backend, ModelSpec, build_backend


class Registry:
    def __init__(self, config_path: Path, repo_root: Path) -> None:
        blob = json.loads(Path(config_path).read_text())
        self.repo_root = repo_root
        self.default = blob["default"]
        self.backends: Dict[str, Backend] = {}
        for name, entry in blob["models"].items():
            known = ModelSpec.__dataclass_fields__.keys()
            spec = ModelSpec(name=name, **{k: v for k, v in entry.items() if k in known})
            spec.extra = {k: v for k, v in entry.items() if k not in known}
            self.backends[name] = build_backend(spec, repo_root)
        if self.default not in self.backends:
            raise ValueError(f"default model {self.default!r} is not in models.json")

    def names(self) -> List[str]:
        return list(self.backends)

    def get(self, name: Optional[str]) -> Backend:
        name = name or self.default
        if name not in self.backends:
            raise KeyError(f"unknown model {name!r}; have {sorted(self.backends)}")
        b = self.backends[name]
        if not b.is_loaded():
            b.load()
        return b

    def missing_files(self, b: Backend) -> List[str]:
        """Which declared artifacts are absent — the usual reason a model can't run."""
        out = []
        for rel in (b.spec.checkpoint, b.spec.piece_vocab, b.spec.block_vocab):
            if rel and not b.path(rel).exists():
                out.append(str(rel))
        return out

    def describe(self) -> List[dict]:
        rows = []
        for name, b in self.backends.items():
            info = b.info()
            info["default"] = (name == self.default)
            missing = self.missing_files(b)
            info["available"] = not missing
            if missing:
                info["unavailable_reason"] = f"missing: {', '.join(missing)}"
            rows.append(info)
        return rows
