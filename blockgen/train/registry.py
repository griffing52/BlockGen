"""``outputs/registry.jsonl``: one line per finished run, the index every report reads."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

PATH = os.path.join("outputs", "registry.jsonl")


def append(record: dict, path: Optional[str] = None) -> None:
    """``path`` defaults to ``PATH`` at call time, so redirecting the module attribute
    (a test, a scratch registry) is honoured by the trainer's call."""
    path = path or PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load(path: Optional[str] = None):
    path = path or PATH
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]
