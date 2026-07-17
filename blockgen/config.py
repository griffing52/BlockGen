"""Layered YAML experiment configs.

The experiment entry points (``experiments_ideas``, ``experiments_overnight``)
take a ``--config path.yaml`` flag. The YAML supplies values for the argparse
*dest* names (``epochs_ar``, ``diff_grid``, ``pe_arms``, ...); those become the
parser's defaults, so anything passed explicitly on the command line still wins.
That keeps one reproducible file per run while leaving room for one-off tweaks.

Composition: a config may list ``extends: [a.yaml, b.yaml]`` (paths relative to
the file itself, then to the ``configs/`` root). Bases are merged left-to-right
and the current file's own keys override them, so a small experiment file can
pull in shared dataset/training/model fragments and change just a knob or two.

The merged mapping is flat: keys are argparse dests. Fragments group related
keys only by convention (which folder they live in), not by YAML nesting, so
there is never any ambiguity about which flag a value maps to.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"

# argparse-style booleans and Nones survive a YAML round-trip fine; we only need
# to guard against a config that fat-fingers a section-style nested dict.
_RESERVED = {"extends", "description"}


def _resolve(ref: str, relative_to: Path) -> Path:
    """Resolve a config reference against the parent file, the root, then the
    standard subfolders — so a bare name like ``ideas-full`` finds
    ``configs/experiments/ideas-full.yaml``."""
    bases = [relative_to.parent, CONFIG_ROOT]
    bases += [d for d in sorted(CONFIG_ROOT.glob("*")) if d.is_dir()]
    for base in bases:
        cand = (base / ref).resolve()
        if cand.suffix == "":
            cand = cand.with_suffix(".yaml")
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"config '{ref}' (referenced from {relative_to}) not found under "
        f"{relative_to.parent}, {CONFIG_ROOT}, or its subfolders"
    )


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in _RESERVED:
            continue
        out[k] = v
    return out


def _load_file(path: Path, _seen: List[Path]) -> Dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in _seen + [path])
        raise ValueError(f"circular config extends: {chain}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping at top level")

    merged: Dict[str, Any] = {}
    extends = raw.get("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    for ref in extends:
        base_path = _resolve(ref, path)
        merged = _merge(merged, _load_file(base_path, _seen + [path]))
    return _merge(merged, raw)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a config file, resolving ``extends``, into a flat dest->value dict.

    A bare name (``ideas-full``) or a name under ``configs/`` is accepted as well
    as an explicit path, so ``--config ideas-full`` works from anywhere.
    """
    p = Path(path)
    if not p.is_file():
        p = _resolve(str(path), CONFIG_ROOT / "_")
    cfg = _load_file(p, [])
    for k in _RESERVED:
        cfg.pop(k, None)
    return cfg


def apply_to_parser(parser, config: Dict[str, Any]) -> None:
    """Override an argparse parser's defaults with config values.

    Unknown keys (not a dest on the parser) raise, so a typo'd knob fails loudly
    instead of being silently ignored.
    """
    dests = {a.dest for a in parser._actions}
    unknown = set(config) - dests
    if unknown:
        raise ValueError(
            f"config keys not recognized as flags: {sorted(unknown)}; "
            f"valid dests: {sorted(dests - {'help'})}"
        )
    parser.set_defaults(**config)
