"""Standard run-directory naming: ``outputs/run_<YYYYMMDD_HHMMSS>_<name>``.

Every training run should write to a timestamped, name-suffixed directory so the most
recent runs sort last (and to the top of a reverse listing) and are findable at a
glance -- matching the ``run_20260715_065938_native`` format the experiment batteries
already use. Training scripts call ``new_run_dir(name)`` instead of taking a raw path.
"""

from __future__ import annotations

import time
from pathlib import Path


def run_stamp() -> str:
    """Sortable timestamp ``YYYYMMDD_HHMMSS`` (local time)."""
    return time.strftime("%Y%m%d_%H%M%S")


def new_run_dir(name: str, base: str = "outputs", stamp: str | None = None) -> Path:
    """Create and return ``<base>/run_<stamp>_<name>``.

    ``name`` is the experiment name (e.g. ``all32_text_resampler``). Pass an explicit
    ``stamp`` to group sibling runs (e.g. a text + image arm) under one timestamp.
    """
    stamp = stamp or run_stamp()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    path = Path(base) / f"run_{stamp}_{safe}"
    path.mkdir(parents=True, exist_ok=True)
    return path
