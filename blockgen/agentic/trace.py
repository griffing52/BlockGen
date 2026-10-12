"""Per-build traces and the contact sheets drawn from them.

A trace is the replay and offline-rescore substrate of one build: every dream
version, every executed attempt (source, canonical DSL, report, score, the voxel
structure, its renders), every LLM request in full (images by content address),
every event and charge. Legacy runs kept only round block counts and the text of
each ask, so a run could not be re-scored, audited, or turned into an iteration
strip; this module is the fix, and ``suite.rescore`` reads nothing else.

Layout (docs/agentic-suite.md, spec 6.1)::

    <root>/
      dreams/<key>_v<k>.png  dreams/<key>_v<k>.json
      iters/<idx>_<stage>/source.txt program.dsl delta.dsl report.json score.json
                          structure.npz render_<view>.png
      messages.jsonl  events.jsonl  charges.jsonl
      assets/<sha16>.png
      contact.png     result.json

Writes are append-and-flush per record (JSONL) or whole-file (everything else),
so a build that crashes mid-way still leaves a readable trace of what happened up
to the crash. ``close`` rewrites the parts the Episode may have changed after the
fact (re-anchored scores after a re-dream, a dream's critique verdict, renders
added by ``rescore_all``), so the files on disk always match ``result.json``.

Contact sheets read ONLY the files on disk, never live objects: the same
function draws a sheet for a build that just finished and for one from last week.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blockgen.agentic.core import VIEWS, Attempt, DreamVersion, MethodResult
from blockgen.agentic.providers import LLMResponse, Message
from blockgen.utils.data import Structure

PathLike = Union[str, Path]

#: Contact-sheet geometry (px). Exported so tests and callers can predict sizes.
TILE = 160
HEADER = 48          # per-column header: stage, "0.71 acc", "sel 0.62" | "unit 0.41"
TITLE = 22           # per-build title bar in arm_contact_sheet / variants rows
LABEL_W = 220        # prompt column in variants_sheet
_DIM = 0.45          # blend towards white for superseded dream versions


def _slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")[:n] or "x"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=_json_default, **kw)


def save_structure(s: Structure, path: PathLike) -> None:
    """Uncropped voxel arrays (``World.from_structure`` rebuilds the exact canvas)."""
    np.savez_compressed(path, block_ids=s.block_ids.astype(np.int32),
                        block_data=s.block_data.astype(np.int32),
                        air_block_id=np.int32(s.air_block_id))


def load_structure(path: PathLike) -> Structure:
    """Inverse of :func:`save_structure` (used by ``suite.rescore``)."""
    with np.load(path) as z:
        return Structure(z["block_ids"], z["block_data"],
                         air_block_id=int(z["air_block_id"]) if "air_block_id" in z else 0)


# ===========================================================================
# Trace writer
# ===========================================================================
class Trace:
    """Writes one build's trace under ``root`` (implements ``core.TraceLike``).

    ``dream_views`` maps dream keys to the VIEWS entry each is scored against, so
    the contact sheet can put the dream next to the matching render. Without it
    the mapping comes from ``dreams.DREAMS`` (when that module exists) or from the
    key itself ('iso' -> 'iso', 'back' -> 'iso_back').
    """

    def __init__(self, root: PathLike, *, dream_views: Optional[Dict[str, Optional[str]]] = None,
                 contact: bool = True) -> None:
        self.root: Optional[Path] = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dream_views: Dict[str, Optional[str]] = dict(dream_views or {})
        self.contact = contact
        self._lock = threading.RLock()
        self._iter_dirs: Dict[int, Path] = {}      # id(attempt) -> its iters/ dir
        # The attempts themselves, kept alive: ids are only unique among LIVE
        # objects, and a discarded attempt's id reused by a later one would make
        # the later attempt overwrite the earlier one's directory.
        self._attempts: List[Attempt] = []
        self._n_requests = 0
        self._dream_seq: Dict[Tuple[str, int], int] = {}   # write order of dream versions
        self._t0 = time.time()

    @classmethod
    def for_kit(cls, root: PathLike, kit: Any, **kw: Any) -> "Trace":
        """A trace whose dream -> view map comes from the recipe's own DreamSpecs.

        Use this rather than ``Trace(root)`` in the suite: an inline dream such as
        ``{key: back, view: iso_back}`` is not a ``dreams.DREAMS`` preset (or is
        one with a different view), and the fallback mapping would put it next to
        the wrong render on the contact sheet.
        """
        views = {d.key: d.view for d in getattr(kit, "dreams", None) or []}
        views.update(kw.pop("dream_views", None) or {})
        return cls(root, dream_views=views, **kw)

    # --- primitives --------------------------------------------------------
    @property
    def _root(self) -> Path:
        assert self.root is not None
        return self.root

    def _append(self, name: str, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(self._root / name, "a", encoding="utf-8") as fh:
                fh.write(_dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

    def _write(self, path: Path, data: Union[str, bytes]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")

    def asset(self, png: bytes) -> str:
        """Store ``png`` once under assets/<sha16>.png; return the relative path."""
        rel = f"assets/{hashlib.sha256(png).hexdigest()[:16]}.png"
        path = self._root / rel
        with self._lock:
            if not path.exists():
                self._write(path, png)
        return rel

    # --- TraceLike ----------------------------------------------------------
    def event(self, kind: str, **data: Any) -> None:
        self._append("events.jsonl", {"t": round(time.time() - self._t0, 3),
                                      "kind": kind, **data})

    def request(self, role: str, messages: Sequence[Message], response: LLMResponse,
                stage: str) -> None:
        """One LLM call in full; image parts become asset references."""
        msgs = []
        for m in messages:
            parts = []
            for p in m.parts:
                if p.kind == "image" and p.image_png is not None:
                    parts.append({"image": self.asset(p.image_png)})
                else:
                    parts.append({"text": p.text})
            msgs.append({"role": m.role, "parts": parts})
        with self._lock:
            n = self._n_requests
            self._n_requests += 1
        self._append("messages.jsonl", {
            "n": n, "t": round(time.time() - self._t0, 3), "role": role, "stage": stage,
            "messages": msgs, "response": response.text, "usage": response.to_dict()})

    def dream(self, d: DreamVersion) -> None:
        base = self._root / "dreams" / f"{_slug(d.key)}_v{d.version}"
        self._write(base.with_suffix(".png"), d.png)
        self._write(base.with_suffix(".json"), _dumps(self._dream_meta(d), indent=1))

    def _dream_meta(self, d: DreamVersion) -> Dict[str, Any]:
        with self._lock:
            seq = self._dream_seq.setdefault((d.key, d.version), len(self._dream_seq))
        return {"key": d.key, "version": d.version, "seq": seq, "prompt": d.prompt,
                "generator": d.generator, "seed": d.seed, "critique": d.critique,
                "cost_usd": d.cost_usd, "cached": d.cached, "digest": d.digest,
                "view": self._view_of(d.key)}

    def _view_of(self, key: str) -> Optional[str]:
        return dream_view(key, self.dream_views)

    def attempt(self, a: Attempt) -> None:
        self._write_attempt(a)

    def _attempt_dir(self, a: Attempt) -> Path:
        with self._lock:
            d = self._iter_dirs.get(id(a))
            if d is None:
                d = self._root / "iters" / f"{int(a.idx):02d}_{_slug(a.stage)}"
                base, k = d, 1
                while d in self._iter_dirs.values():
                    # A second attempt with the same idx and stage (e.g. a wrapped
                    # legacy run): keep both rather than overwrite.
                    d = base.with_name(f"{base.name}~{k}")
                    k += 1
                self._iter_dirs[id(a)] = d
                self._attempts.append(a)
            return d

    def _write_attempt(self, a: Attempt, *, final: bool = False) -> None:
        d = self._attempt_dir(a)
        d.mkdir(parents=True, exist_ok=True)
        if not final or not (d / "source.txt").exists():
            self._write(d / "source.txt", a.source_text or "")
            self._write(d / "program.dsl", a.program_text or "")
            self._write(d / "delta.dsl", a.delta_text or "")
            self._write(d / "report.json", _dumps(a.report.to_dict(), indent=1))
        if a.world is not None and not (d / "structure.npz").exists():
            try:
                save_structure(a.world.to_structure(crop=False), d / "structure.npz")
            except Exception as exc:                          # noqa: BLE001
                self.event("trace_error", what="structure", idx=a.idx, error=repr(exc))
        for view, png in (a.renders or {}).items():
            path = d / f"render_{_slug(view)}.png"
            if not path.exists():
                self._write(path, png)
        score = {"idx": a.idx, "stage": a.stage, "blocks": a.blocks, "patch": a.patch,
                 "accepted": a.accepted, "reason": a.reason,
                 "score": a.score.to_dict() if a.score else None,
                 "selection": getattr(a, "selection", None),
                 "extra": dict(getattr(a, "extra", None) or {})}
        self._write(d / "score.json", _dumps(score, indent=1))

    def close(self, result: MethodResult) -> None:
        """Refresh scores/dreams/renders, then write charges, result.json, contact.png."""
        if not result.trace_dir:
            result.trace_dir = str(self._root)
        for vs in (result.dreams or {}).values():
            for d in vs:                     # critique verdicts arrive after dream()
                self.dream(d)
        for a in result.attempts:            # scores re-anchored by rescore_all
            self._write_attempt(a, final=True)
        with self._lock:
            with open(self._root / "charges.jsonl", "w", encoding="utf-8") as fh:
                for c in result.charges:
                    fh.write(_dumps(c.to_dict()) + "\n")
        try:
            blob = result.to_dict()
        except Exception as exc:                              # noqa: BLE001
            # metrics() may need optional eval deps; the trace must not die on it.
            blob = {"description": result.description, "error": result.error,
                    "to_dict_error": repr(exc)}
        blob["best_idx"] = result.best_idx
        # best_idx is the attempt's ``idx`` (not necessarily its list position, e.g.
        # for attempts adopted from a wrapped run), so look it up by idx.
        best = next((a for a in result.attempts if a.idx == result.best_idx), None)
        blob["best_dir"] = (self._iter_dirs[id(best)].name
                            if best is not None and id(best) in self._iter_dirs else None)
        blob["dream_views"] = {k: self._view_of(k) for k in (result.dreams or {})}
        self._write(self._root / "result.json", _dumps(blob, indent=1))
        if self.contact:
            try:
                contact_sheet(self._root)
            except Exception:                                 # noqa: BLE001
                self.event("contact_error", traceback=traceback.format_exc(limit=4))
                warnings.warn(f"contact sheet failed for {self._root}")


def dream_view(key: str, explicit: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    """The VIEWS entry dream ``key`` is scored against (None = context-only/unknown)."""
    if explicit and key in explicit:
        return explicit[key]
    from blockgen.agentic.dreams import DREAMS          # the shipped dream presets

    spec = DREAMS.get(key)
    if spec is not None:
        return spec.view
    if key in VIEWS:
        return key
    if f"iso_{key}" in VIEWS:
        return f"iso_{key}"
    return None


# ===========================================================================
# Drawing helpers
# ===========================================================================
def _font(size: int = 12):
    from blockgen.agentic.texsheet import load_font

    return load_font(size)


def _fit(png_or_path: Union[bytes, Path, None], px: int, *, dim: float = 0.0):
    """An image scaled to fit (px, px), centred on white. None -> blank tile."""
    from PIL import Image

    tile = Image.new("RGB", (px, px), (255, 255, 255))
    if png_or_path is None:
        return tile
    try:
        data = png_or_path.read_bytes() if isinstance(png_or_path, Path) else png_or_path
        img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:                                         # noqa: BLE001
        return tile
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    s = px / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                     Image.LANCZOS if s < 1 else Image.NEAREST)
    if dim:
        img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), dim)
    tile.paste(img, ((px - img.width) // 2, (px - img.height) // 2))
    return tile


def _text(draw, xy: Tuple[float, float], text: str, *, size: int = 12, width: int = 0,
          fill=(0, 0, 0), box: bool = False) -> None:
    """Draw one line, truncated with '~' to ``width``; ``box`` = white backing so a
    label stays legible over a dark image."""
    from blockgen.agentic.texsheet import text_width

    font = _font(size)
    if width:
        while text and text_width(text, font) > width:
            text = text[:-2] + "~" if len(text) > 2 else ""
    if box and text:
        draw.rectangle([xy[0] - 2, xy[1] - 1, xy[0] + text_width(text, font) + 2,
                        xy[1] + size + 2], fill=(255, 255, 255))
    draw.text(xy, text, fill=fill, font=font)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _iter_dirs(root: Path) -> List[Path]:
    d = root / "iters"
    if not d.is_dir():
        return []

    def key(p: Path) -> Tuple[int, str]:
        m = re.match(r"(\d+)_", p.name)
        return (int(m.group(1)) if m else 10 ** 6, p.name)
    return sorted((p for p in d.iterdir() if p.is_dir()), key=key)


def _dream_versions(root: Path) -> List[Dict[str, Any]]:
    """[{key, version, png: Path, meta}] sorted by (key order of first version, version)."""
    d = root / "dreams"
    out = []
    if not d.is_dir():
        return out
    for js in d.glob("*.json"):
        meta = _read_json(js)
        png = js.with_suffix(".png")
        if png.exists():
            out.append({"key": str(meta.get("key", js.stem.rsplit("_v", 1)[0])),
                        "version": int(meta.get("version", 0)), "png": png, "meta": meta})
    # Keys in the order they were first dreamed (iso before the views conditioned
    # on it), versions in order within a key.
    first: Dict[str, Tuple[int, str]] = {}
    for v in out:
        seq = v["meta"].get("seq")
        rank = (int(seq) if isinstance(seq, int) else 10 ** 6, v["key"])
        first[v["key"]] = min(first.get(v["key"], rank), rank)
    return sorted(out, key=lambda v: (first[v["key"]], v["version"]))


def _view_order(names: Sequence[str]) -> List[str]:
    known = list(VIEWS)
    return sorted(set(names), key=lambda n: (known.index(n) if n in known else len(known), n))


# ===========================================================================
# Contact sheets
# ===========================================================================
def contact_layout(root: PathLike) -> Dict[str, Any]:
    """What contact_sheet will draw: rows (views), columns (iters), dream strip.

    Exposed separately so callers (and tests) can predict the sheet's size:
    width = cols * TILE, height = HEADER + rows * TILE.
    """
    root = Path(root)
    iters = _iter_dirs(root)
    views = _view_order([p.name[len("render_"):-len(".png")]
                         for it in iters for p in it.glob("render_*.png")])
    dreams = _dream_versions(root)
    n_rows = len(views) + (1 if dreams else 0)
    n_cols = max(1 + len(iters), len(dreams), 1)
    return {"iters": iters, "views": views, "dreams": dreams, "rows": max(1, n_rows),
            "cols": n_cols, "width": n_cols * TILE, "height": HEADER + max(1, n_rows) * TILE}


def contact_sheet(trace_root: PathLike, out: Optional[PathLike] = None) -> Path:
    """Draw ``contact.png`` for one build from its trace files.

    Rows are the rendered views (plus a bottom strip of every dream version, older
    versions dimmed); columns are 'dream (latest)' then one per attempt in order.
    Each attempt's header reads '<stage>', '<score> acc|rej' and, on a third line,
    'sel <value>' (the value selection ranked it by, 'sel gated' below 0) or, for a
    unit attempt of a staged method, 'unit <unit score>'; the selected (best)
    attempt is boxed in red. Why the third line: with a final_score the selection
    value differs from the score, and a boxed column that looks worse than its
    neighbours is otherwise unexplained. The dream cell of a row is the latest version
    of the dream scored against that row's view.
    """
    from PIL import Image, ImageDraw

    root = Path(trace_root)
    lay = contact_layout(root)
    result = _read_json(root / "result.json")
    best_dir = result.get("best_dir")
    views_of = result.get("dream_views") or {}
    latest: Dict[str, Dict[str, Any]] = {}
    for v in lay["dreams"]:
        latest[v["key"]] = v                         # sorted by version: last wins
    dream_for_view: Dict[str, Dict[str, Any]] = {}
    for key, v in latest.items():
        view = views_of.get(key, v["meta"].get("view", dream_view(key)))
        if view and view not in dream_for_view:
            dream_for_view[view] = v

    img = Image.new("RGB", (lay["width"], lay["height"]), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _text(draw, (4, 3), "dream", size=12)
    for c, it in enumerate(lay["iters"], start=1):
        sc = _read_json(it / "score.json")
        stage = str(sc.get("stage") or it.name.split("_", 1)[-1])
        val = (sc.get("score") or {}).get("value")
        verdict = "acc" if sc.get("accepted") else "rej"
        head = f"{val:.2f} {verdict}" if isinstance(val, (int, float)) else f"- {verdict}"
        x = c * TILE
        _text(draw, (x + 4, 3), stage, size=12, width=TILE - 8)
        _text(draw, (x + 4, 17), head, size=12, width=TILE - 8,
              fill=(0, 110, 0) if sc.get("accepted") else (170, 0, 0))
        sel = sc.get("selection")
        unit = (sc.get("extra") or {}).get("unit_score")
        third = ("sel gated" if isinstance(sel, (int, float)) and sel < 0
                 else f"sel {sel:.2f}" if isinstance(sel, (int, float))
                 else f"unit {unit:.2f}" if isinstance(unit, (int, float)) else "")
        if third:
            _text(draw, (x + 4, 31), third, size=12, width=TILE - 8, fill=(40, 40, 140))

    for r, view in enumerate(lay["views"]):
        y = HEADER + r * TILE
        dv = dream_for_view.get(view)
        img.paste(_fit(dv["png"] if dv else None, TILE), (0, y))
        _text(draw, (4, y + 3), f"{view}" + (f"  [{dv['key']} v{dv['version']}]" if dv else ""),
              size=12, width=TILE - 8, fill=(60, 60, 60), box=True)
        for c, it in enumerate(lay["iters"], start=1):
            png = it / f"render_{_slug(view)}.png"
            img.paste(_fit(png if png.exists() else None, TILE), (c * TILE, y))
            if not png.exists():
                _text(draw, (c * TILE + 4, y + TILE // 2 - 6), "(no render)", size=12,
                      fill=(150, 150, 150))

    if lay["dreams"]:
        y = HEADER + len(lay["views"]) * TILE
        for c, v in enumerate(lay["dreams"]):
            is_latest = latest[v["key"]] is v
            img.paste(_fit(v["png"], TILE, dim=0.0 if is_latest else _DIM), (c * TILE, y))
            crit = v["meta"].get("critique") or {}
            tag = f" s={crit['score']}" if isinstance(crit, dict) and "score" in crit else ""
            _text(draw, (c * TILE + 4, y + 3), f"{v['key']} v{v['version']}{tag}", size=12,
                  width=TILE - 8, fill=(60, 60, 60) if is_latest else (150, 150, 150),
                  box=True)

    if best_dir:
        for c, it in enumerate(lay["iters"], start=1):
            if it.name == best_dir:
                x0, x1 = c * TILE, (c + 1) * TILE - 1
                y1 = HEADER + len(lay["views"]) * TILE - 1
                draw.rectangle([x0, 0, x1, max(HEADER, y1)], outline=(220, 0, 0), width=3)
    if not lay["iters"] and not lay["dreams"]:
        _text(draw, (4, HEADER + 4), "(no attempts)", size=12, fill=(150, 150, 150))

    dest = Path(out) if out else root / "contact.png"
    img.save(dest, format="PNG")
    return dest


def _title(root: Path) -> str:
    res = _read_json(root / "result.json")
    return str(res.get("description") or "")


def arm_contact_sheet(build_roots: Sequence[PathLike], out: PathLike, k: int = 8) -> Path:
    """Stack the contact sheets of the first ``k`` builds (one title bar each)."""
    from PIL import Image, ImageDraw

    roots = [Path(r) for r in list(build_roots)[:k]]
    sheets = []
    for r in roots:
        cp = r / "contact.png"
        if not cp.exists():
            try:
                contact_sheet(r)
            except Exception as exc:                          # noqa: BLE001
                warnings.warn(f"arm_contact_sheet: no contact sheet for {r} ({exc!r})")
                continue
        if cp.exists():
            sheets.append((r, Image.open(cp).convert("RGB")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not sheets:
        img = Image.new("RGB", (TILE * 2, TITLE + 24), (255, 255, 255))
        _text(ImageDraw.Draw(img), (4, 4), "(no builds)", size=12)
        img.save(out, format="PNG")
        return out
    width = max(s.width for _, s in sheets)
    height = sum(TITLE + s.height for _, s in sheets)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 0
    for r, s in sheets:
        draw.rectangle([0, y, width, y + TITLE - 1], fill=(235, 235, 235))
        _text(draw, (4, y + 4), f"{r.name}: {_title(r)}", size=12, width=width - 8)
        img.paste(s, (0, y + TITLE))
        y += TITLE + s.height
    img.save(out, format="PNG")
    return out


def _final_render(root: Path) -> Optional[Path]:
    """The selected attempt's iso render (else its first render, else the last attempt's)."""
    res = _read_json(root / "result.json")
    iters = _iter_dirs(root)
    best = next((it for it in iters if it.name == res.get("best_dir")), None)
    for it in ([best] if best else []) + list(reversed(iters)):
        renders = sorted(it.glob("render_*.png"))
        if renders:
            iso = it / "render_iso.png"
            return iso if iso.exists() else renders[0]
    return None


def _primary_dream(root: Path) -> Optional[Path]:
    versions = _dream_versions(root)
    if not versions:
        return None
    latest: Dict[str, Path] = {}
    for v in versions:
        latest[v["key"]] = v["png"]
    return latest.get("iso") or next(iter(latest.values()))


def variants_sheet(run_dir: PathLike, arms: Sequence[str], k: int = 12,
                   out: Optional[PathLike] = None) -> Path:
    """rows = prompts, cols = arms, each cell 'dream | final' (dream blank if none).

    Builds are aligned by index (``builds/<arm>/NNN``): every arm of a run gets
    the same prompt list in the same order, so row i compares the arms on one
    prompt. The row label is the prompt from the first arm that has it.
    """
    from PIL import Image, ImageDraw

    run_dir = Path(run_dir)
    arms = list(arms)
    per_arm: Dict[str, Dict[str, Path]] = {}
    for arm in arms:
        d = run_dir / "builds" / arm
        per_arm[arm] = ({p.name: p for p in d.iterdir() if p.is_dir() and p.name.isdigit()}
                        if d.is_dir() else {})
    rows = sorted(set().union(*[set(v) for v in per_arm.values()]) if per_arm else set())[:k]
    cell_w = 2 * TILE
    width = LABEL_W + max(1, len(arms)) * cell_w
    height = TITLE + max(1, len(rows)) * TILE
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for c, arm in enumerate(arms):
        _text(draw, (LABEL_W + c * cell_w + 4, 4), f"{arm}  (dream | final)", size=12,
              width=cell_w - 8)
    for r, name in enumerate(rows):
        y = TITLE + r * TILE
        prompt = next((_title(per_arm[a][name]) for a in arms if name in per_arm[a]), "")
        _wrap_text(draw, (4, y + 4), f"{name}: {prompt}", LABEL_W - 8, max_lines=TILE // 15)
        for c, arm in enumerate(arms):
            x = LABEL_W + c * cell_w
            root = per_arm[arm].get(name)
            if root is None:
                _text(draw, (x + 4, y + TILE // 2), "(missing)", size=12, fill=(150, 150, 150))
                continue
            img.paste(_fit(_primary_dream(root), TILE), (x, y))
            fin = _final_render(root)
            img.paste(_fit(fin, TILE), (x + TILE, y))
            if fin is None:
                _text(draw, (x + TILE + 4, y + TILE // 2), "(no render)", size=12,
                      fill=(150, 150, 150))
            draw.line([x + TILE, y, x + TILE, y + TILE - 1], fill=(220, 220, 220))
    dest = Path(out) if out else run_dir / "variants.png"
    img.save(dest, format="PNG")
    return dest


def _wrap_text(draw, xy: Tuple[int, int], text: str, width: int, *, max_lines: int = 8,
               size: int = 12) -> None:
    from blockgen.agentic.texsheet import text_width

    font = _font(size)
    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, font) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines[:max_lines]):
        _text(draw, (xy[0], xy[1] + i * 15), line, size=size, width=width)


__all__ = ["HEADER", "LABEL_W", "TILE", "TITLE", "Trace", "arm_contact_sheet",
           "contact_layout", "contact_sheet", "dream_view", "load_structure",
           "save_structure", "variants_sheet"]
