"""Scene -> text, one part per line, with the token span of every line.

Two formats:

* ``absolute``: ``x y z block`` (LEGO: ``x layer z brick rotation``). Coordinates are
  printed, so a head can find neighbours by matching coordinate strings (T28's
  diagnosis). Orders: ``raster`` (y, z, x; spatial neighbours are often textual
  neighbours) and ``shuffle`` (random permutation; breaks that link).
* ``relative``: coordinate-free growth. Line ``k`` reads ``k block on p dir``: the block
  touches the ``dir`` face of earlier line ``p``. Only the spanning-tree edge to the
  parent is written down; every other contact (a block resting on two others, a wall
  closing a loop) has to be inferred by composing moves. Orders: ``bfs`` (breadth-first
  from the lowest block) and ``random`` (random growth: any frontier block next).

Lines are tokenised one at a time and concatenated, so each span is exact by
construction. Every line ends in ``\\n``, which keeps BPE boundaries stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .scenes import Scene

HEADER = "Minecraft build. One block per line as `x y z block`:\n"
LEGO_HEADER = "LEGO build. One brick per line as `x layer z brick rotation`:\n"
REL_HEADER = ("Minecraft build. Line `k block on p dir` places block k touching the dir "
              "face of earlier block p:\n")
ORDERS = {"absolute": ("raster", "shuffle"), "relative": ("bfs", "random")}
DIRS = {(1, 0, 0): "east", (-1, 0, 0): "west", (0, 1, 0): "up", (0, -1, 0): "down",
        (0, 0, 1): "south", (0, 0, -1): "north"}


@dataclass
class Serialized:
    scene: Scene                      # parts in serialised order
    text: str
    input_ids: List[int]
    spans: List[Tuple[int, int]]      # token [start, end) of each line, aligned to scene
    parents: Optional[List[int]] = None   # relative format: stated parent of each line

    @property
    def lines(self) -> List[str]:
        return self.text.split("\n")[1:-1]


def raster_index(scene: Scene) -> np.ndarray:
    c = scene.coords
    return np.lexsort((c[:, 0], c[:, 2], c[:, 1]))


def growth_order(scene: Scene, randomized: bool, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Visit order and parent (index into the *visit order*) of a connected scene."""
    rng = np.random.default_rng(seed)
    root = int(raster_index(scene)[0])
    order, parent_of = [root], {root: -1}
    frontier: List[Tuple[int, int]] = []          # (node, discovering node)

    def expand(u: int) -> None:
        for v in np.flatnonzero(scene.adjacency[u]):
            if int(v) not in parent_of:
                frontier.append((int(v), u))

    expand(root)
    while frontier:
        k = int(rng.integers(len(frontier))) if randomized else 0
        v, u = frontier.pop(k)
        if v in parent_of:
            continue
        parent_of[v] = u
        order.append(v)
        expand(v)
    if len(order) != len(scene):
        raise ValueError("relative format needs a connected scene")
    pos = {v: i for i, v in enumerate(order)}
    return order, [-1 if parent_of[v] < 0 else pos[parent_of[v]] for v in order]


def order_scene(scene: Scene, order: str, seed: int = 0) -> Tuple[Scene, Optional[List[int]]]:
    """Re-order the scene; returns ``(scene, parents)`` (parents only for growth orders)."""
    if order == "raster":
        return scene.subset(raster_index(scene)), None
    if order == "shuffle":
        return scene.subset(np.random.default_rng(seed).permutation(len(scene))), None
    if order in ("bfs", "random"):
        idx, parents = growth_order(scene, randomized=order == "random", seed=seed)
        return scene.subset(idx), parents
    raise ValueError(f"unknown order {order!r}")


def absolute_line(scene: Scene, i: int) -> str:
    x, y, z = (int(v) for v in scene.coords[i])
    parts = [str(x), str(y), str(z), scene.names[i]] + ([scene.extra[i]] if scene.extra else [])
    return " ".join(parts) + "\n"


def relative_line(scene: Scene, i: int, parent: int) -> str:
    if parent < 0:
        return f"{i} {scene.names[i]} root\n"
    d = tuple(int(v) for v in scene.coords[i] - scene.coords[parent])
    return f"{i} {scene.names[i]} on {parent} {DIRS[d]}\n"


def serialize(scene: Scene, tokenizer, order: str = "raster", seed: int = 0,
              fmt: str = "absolute", header: Optional[str] = None) -> Serialized:
    if order not in ORDERS[fmt]:
        raise ValueError(f"{fmt} format takes orders {ORDERS[fmt]}, got {order!r}")
    scene, parents = order_scene(scene, order, seed)
    if header is None:
        header = REL_HEADER if fmt == "relative" else (LEGO_HEADER if scene.extra else HEADER)
    ids = list(tokenizer(header, add_special_tokens=True)["input_ids"])
    text, spans = header, []
    for i in range(len(scene)):
        line = relative_line(scene, i, parents[i]) if fmt == "relative" else absolute_line(scene, i)
        piece = tokenizer(line, add_special_tokens=False)["input_ids"]
        spans.append((len(ids), len(ids) + len(piece)))
        ids.extend(piece)
        text += line
    return Serialized(scene=scene, text=text, input_ids=ids, spans=spans, parents=parents)
