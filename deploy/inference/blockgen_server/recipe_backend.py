"""Serve an agentic *recipe* (dream, dream-caption, classic-matched, ...) live.

The ``agentic`` backend runs the legacy one-shot loop and replays the final program.
Recipes are different: the benchmarked ones (results.md T51) imagine the house as an
image first, build, render the build, compare, and try again. Most of the time is
spent thinking, so replaying only the final program would hide the interesting part.

This backend runs ``blockgen.agentic.episode.run_episode`` on a background thread
with a :class:`LiveTrace`, which turns the episode's trace hooks into a stream:

* **chat lines** for each stage, each LLM call (when it starts, and a short
  summary of what came back: the builder's program size, the critic's verdict),
  each dream and each scored attempt, with the running cost;
* **the dream image as a block wall** (``billboard.py``) next to the build,
  replaced if the loop re-dreams;
* **each attempt as a block diff** against what is already standing, so you watch
  the build change between rounds. At the end the attempt the recipe *selected*
  is restored, which is not always the last one.

It is a group like ``agentic``: ``recipe:dream-caption`` names one recipe in
``configs/agentic/recipes``. Member ``mock`` runs the ``mock_recipe`` (default
``dream``) on the offline mock backends: free, used by the tests and for checking
the mod without spending anything.

**Spend.** Each recipe keeps its own ``budget.usd_per_build``. On top of that every
recipe build charges one server-wide ledger capped at ``session_usd_cap`` from
``models.json``; a build that would cross it stops early and keeps its best attempt,
exactly like a per-build budget stop. Builds are serialized per backend, so the
cost reported for a build is that build's alone.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from blockgen_server.backends import Backend, BlockBatch, GenerateRequest, ModelSpec
from blockgen_server.decode import Block

Pos = Tuple[int, int, int]

AIR = "minecraft:air"
_DONE = object()


class Cancelled(Exception):
    """Raised inside the episode thread to stop a cancelled build at its next call."""


# --- the "thinking..." hook ------------------------------------------------
# Trace hooks fire only AFTER an LLM call returns, and a call can take a minute.
# Episode.ask is the one place every call passes through, so it is wrapped once
# per process; the wrapper only does anything on a thread that registered a sink.
_local = threading.local()
_ask_patched = False
_patch_lock = threading.Lock()


def _install_ask_hook() -> None:
    global _ask_patched
    with _patch_lock:
        if _ask_patched:
            return
        from blockgen.agentic.episode import Episode
        orig = Episode.ask

        def ask(self, turn, *args, **kw):
            sink = getattr(_local, "sink", None)
            if sink is not None:
                sink(kw.get("role", "builder"), kw.get("stage", ""))
            return orig(self, turn, *args, **kw)

        ask.__doc__ = orig.__doc__
        Episode.ask = ask
        _ask_patched = True


# --- narration helpers -----------------------------------------------------
def _line(text: str, style: str = "info") -> dict:
    return {"text": text, "style": style}


def _clip(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


# Reviewer replies are JSON; these are the fields worth a line of chat, in order.
_REVIEW_KEYS = ("verdict", "summary", "critique", "feedback", "issues", "problems",
                "fix", "fixes", "reason", "notes")


def summarize_reply(role: str, text: str) -> Optional[str]:
    """One short line for an LLM reply, or None if it says nothing worth showing."""
    from blockgen.agentic.episode import extract_json
    data = extract_json(text) if "{" in text else {}
    if not data and role in ("builder", "critic"):
        # A program (the critic rewrites the program in the dream match turns).
        # The first comment line is usually the model's own one-line intent.
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cmds = [l for l in lines if not l.startswith(("#", "```"))]
        note = next((l.lstrip("# ") for l in lines if l.startswith("#")), "")
        verb = "wrote" if role == "builder" else "rewrote"
        return f"{verb} a {len(cmds)}-command program" + (f": {_clip(note, 90)}" if note else "")
    if data:
        bits = []
        for k in _REVIEW_KEYS:
            v = data.get(k)
            if v in (None, "", [], {}):
                continue
            if isinstance(v, list):
                v = "; ".join(str(x if not isinstance(x, dict) else
                                  x.get("issue") or x.get("text") or x) for x in v[:3])
            bits.append(f"{k}: {v}" if k not in ("summary", "critique", "feedback") else str(v))
            if len(bits) == 2:
                break
        score = next((data[k] for k in ("score", "similarity", "match") if
                      isinstance(data.get(k), (int, float))), None)
        if score is not None:
            bits.insert(0, f"score {score}")
        return _clip(" | ".join(bits)) if bits else None
    return _clip(text) if text.strip() else None


def _score_text(a: Any) -> str:
    sc = getattr(a, "score", None)
    if sc is None:
        return ""
    parts = ", ".join(f"{k} {v:.2f}" for k, v in list(sc.parts.items())[:3]
                      if isinstance(v, (int, float)))
    gated = f" gated by {sc.gated}" if sc.gated else ""
    return f"score {sc.value:.2f}" + (f" ({parts})" if parts else "") + gated


# --- the live trace ----------------------------------------------------------
class LiveTrace:
    """TraceLike that forwards to a disk Trace and narrates into a queue.

    Hooks run on the episode thread; they only snapshot data and enqueue it.
    Diffing against the blocks already standing happens on the stream thread, which
    owns the placed-block state.
    """

    def __init__(self, inner: Any, out: "queue.Queue", ledger: Any, abort: threading.Event,
                 billboard: bool) -> None:
        self.inner = inner
        self.root = getattr(inner, "root", None)
        self.out = out
        self.ledger = ledger
        self.spent0 = ledger.spent_usd
        self.n0 = len(ledger.charges)
        self.abort = abort
        self.billboard = billboard
        self._dream_key: Optional[str] = None

    def spent(self) -> float:
        return self.ledger.spent_usd - self.spent0

    def charges(self) -> list:
        """This build's charges (builds are serialized, so nothing interleaves)."""
        return list(self.ledger.charges[self.n0:])

    def _check(self) -> None:
        if self.abort.is_set():
            raise Cancelled("cancelled from Minecraft")

    def say(self, text: str, style: str = "info") -> None:
        self.out.put(("log", [_line(text, style)]))

    # Episode.ask hook (see _install_ask_hook)
    def thinking(self, role: str, stage: str) -> None:
        self._check()
        where = f" ({stage})" if stage else ""
        self.say(f"{role} thinking{where}...", "think")

    # --- TraceLike -------------------------------------------------------
    def asset(self, png: bytes) -> str:
        return self.inner.asset(png)

    def event(self, kind: str, **data: Any) -> None:
        self.inner.event(kind, **data)
        if kind == "stage_start":
            self.say(f"== stage: {data.get('stage')}", "stage")
        elif kind == "stop":
            why = data.get("rule", "")
            val = data.get("value")
            self.say(f"stopping: {why}" + (f" (score {val:.2f})" if isinstance(val, float) else ""),
                     "stage")
        elif kind == "redream":
            self.say(f"re-imagining {data.get('key')}: {_clip(data.get('desc') or '', 120)}",
                     "dream")
        elif kind.endswith("budget_stop") or kind == "budget_stop":
            self.say(f"budget stop: {_clip(data.get('error', ''), 120)}", "warn")
        elif kind in ("error", "schematic_error"):
            self.say(f"error: {_clip(data.get('error', ''), 160)}", "error")
        elif kind == "plan_verify":
            self.say(f"plan check: {'ok' if data.get('ok') else 'needs work'}", "critique")
        elif kind == "critique_skipped":
            self.say(f"critique skipped ({data.get('reason')})", "warn")

    def request(self, role: str, messages: Any, response: Any, stage: str) -> None:
        self.inner.request(role, messages, response, stage)
        summary = summarize_reply(role, response.text or "")
        cost = "cached" if getattr(response, "cached", False) else f"${self.spent():.3f} so far"
        tok = getattr(response, "completion_tokens", 0) or 0
        self.say(f"{role}: {summary or 'replied'}  [{tok} tok, {cost}]",
                 "build" if role == "builder" else "critique")
        self._check()

    def dream(self, d: Any) -> None:
        self.inner.dream(d)
        src = "cached" if d.cached else f"${d.cost_usd:.3f}"
        self.say(f"dreamed '{d.key}' v{d.version} with {d.generator.split('?')[0]} ({src})",
                 "dream")
        if self._dream_key is None:
            self._dream_key = d.key       # the first dream is the primary target
        if self.billboard and d.key == self._dream_key:
            self.out.put(("billboard", d.png))
        self._check()

    def attempt(self, a: Any) -> None:
        self.inner.attempt(a)
        world = getattr(a, "world", None)
        verdict = "kept" if a.accepted else "rejected"
        reason = f": {_clip(a.reason, 80)}" if a.reason else ""
        self.say(f"attempt {a.idx} [{a.stage}] {a.blocks} blocks, {_score_text(a)}, "
                 f"{verdict}{reason}  (${self.spent():.3f})", "score")
        if world is not None:
            self.out.put(("world", (a.idx, _world_blocks(world))))

    def close(self, result: Any) -> None:
        self.inner.close(result)


def _world_blocks(world: Any) -> Dict[Pos, str]:
    """Every non-air voxel of a canvas/world as a modern block state string."""
    import numpy as np
    from blockgen_server.blockmap import modern_state

    s = world.to_structure(crop=False)
    ids, data = s.block_ids, s.block_data
    out: Dict[Pos, str] = {}
    cache: Dict[Tuple[int, int], str] = {}
    for x, y, z in zip(*np.nonzero(ids != s.air_block_id)):
        key = (int(ids[x, y, z]), int(data[x, y, z]))
        st = cache.get(key)
        if st is None:
            st = cache[key] = modern_state(key[0], key[1], oriented=True)
        out[(int(x), int(y), int(z))] = st
    return out


def diff_blocks(have: Dict[Pos, str], want: Dict[Pos, str]) -> List[Block]:
    """Placements that turn ``have`` into ``want``: removals first, then new
    blocks bottom-up, so a rebuild reads as clearing and then going up."""
    gone = [Block(x, y, z, AIR) for (x, y, z) in have if (x, y, z) not in want]
    gone.sort(key=lambda b: -b.y)
    put = [Block(x, y, z, s) for (x, y, z), s in want.items() if have.get((x, y, z)) != s]
    put.sort(key=lambda b: (b.y, b.x, b.z))
    return gone + put


# --- the backend -------------------------------------------------------------
class RecipeBackend(Backend):
    """Group backend: ``recipe:<name>`` runs ``configs/agentic/recipes/<name>.yaml``."""

    supports_text = True
    DEFAULT_RECIPE = "dream-caption"

    # One ledger for every recipe build in this server process (session cap).
    _session_ledger = None
    _session_lock = threading.Lock()

    def __init__(self, spec: ModelSpec, repo_root: Path) -> None:
        super().__init__(spec, repo_root)
        e = spec.extra
        self.recipe: str = e.get("provider") or e.get("recipe") or self.DEFAULT_RECIPE
        self.models: List[str] = list(e.get("models") or [self.recipe])
        self.mock = self.recipe == "mock"
        self._kit = None
        self._build_lock = threading.Lock()

    # --- config ----------------------------------------------------------
    def method_spec(self):
        from blockgen.agentic.recipe import load_recipe
        e = self.spec.extra
        name = e.get("mock_recipe", "dream") if self.mock else self.recipe
        return load_recipe(name, overrides=e.get("overrides") or None)

    @classmethod
    def session_ledger(cls, cap: Optional[float]):
        from blockgen.agentic.core import Budget, Ledger
        with cls._session_lock:
            if cls._session_ledger is None:
                cls._session_ledger = Ledger(Budget(usd=cap), name="deploy-session")
            return cls._session_ledger

    def group_members(self) -> List[str]:
        return list(self.models)

    def unavailable_reason(self) -> Optional[str]:
        """Missing API keys for the recipe's LLM or image backends, up front."""
        if self.mock:
            return None
        import os
        from blockgen.agentic.providers import load_env
        try:
            spec = self.method_spec()
        except Exception as exc:  # noqa: BLE001 - unknown recipe name, bad yaml
            return f"cannot load recipe {self.recipe!r}: {exc}"
        load_env()
        backends = list(spec.roles().values()) + ([spec.image] if spec.image else [])
        need = {"openai": ["OPENAI_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                "anthropic": ["ANTHROPIC_API_KEY"]}
        for b in backends:
            keys = need.get(str(b).split(":", 1)[0].lower())
            if keys and not any(os.environ.get(k) for k in keys):
                return f"no API key for {b}: set {' or '.join(keys)} in the repo .env"
        return None

    def info(self) -> dict:
        out = super().info()
        out.update({"group": self.spec.name.split(":")[0], "members": self.group_members(),
                    "recipe": self.recipe})
        try:
            spec = self.method_spec()
            out.update({"method": spec.method, "llm": spec.roles().get("builder"),
                        "image": spec.image, "canvas": list(spec.canvas),
                        "usd_per_build": (spec.budget or {}).get("usd_per_build")})
            if not self.mock:
                out["description"] = f"{self.spec.description.split(' [')[0]} [{self.recipe}: " \
                                     f"{spec.description}]"
        except Exception:  # noqa: BLE001 - listing must never fail on one bad recipe
            pass
        led = self._session_ledger
        if led is not None:
            out["session_spent_usd"] = round(led.spent_usd, 4)
        return out

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        from blockgen.agentic.core import load_builtins
        from blockgen.agentic.kit import build_kit, validate_spec

        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.spec.name}: {reason}")
        load_builtins()
        spec = self.method_spec()
        validate_spec(spec)
        e = self.spec.extra
        ledger = self.session_ledger(e.get("session_usd_cap"))
        # The server is the user's explicit opt-in to spend (it is what /gen is for),
        # so the BLOCKGEN_ALLOW_SPEND gate is off unless the entry turns it back on.
        self._kit = build_kit(spec, mock=self.mock, ledger=ledger,
                              spend_gate=bool(e.get("spend_gate", False)))
        _install_ask_hook()
        self.model = self._kit

    def _prompt_for(self, req: GenerateRequest) -> str:
        if req.prompt:
            return req.prompt
        from blockgen.agentic.tasks import SHORT_PROMPTS
        import random
        return random.Random(req.seed).choice(SHORT_PROMPTS)

    def _trace_dir(self, prompt: str) -> Optional[Path]:
        base = self.spec.extra.get("trace_dir", "outputs/deploy_traces")
        if not base:
            return None
        from blockgen.agentic.trace import _slug
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return self.path(base) / f"{stamp}_{_slug(self.recipe, 24)}_{_slug(prompt, 32)}"

    # --- generation ------------------------------------------------------
    def stream(self, req: GenerateRequest) -> Iterator[BlockBatch]:
        with self._build_lock:
            yield from self._stream(req)

    def _stream(self, req: GenerateRequest) -> Iterator[BlockBatch]:
        from blockgen.agentic.core import NullTrace
        from blockgen.agentic.episode import run_episode
        from blockgen.agentic.trace import Trace
        from blockgen_server.billboard import billboard_blocks

        kit, e = self._kit, self.spec.extra
        spec = kit.spec
        prompt = self._prompt_for(req)
        tdir = self._trace_dir(prompt)
        inner = Trace.for_kit(tdir, kit) if tdir else NullTrace()
        out: "queue.Queue" = queue.Queue()
        abort = threading.Event()
        live = LiveTrace(inner, out, kit.ledger, abort, billboard=bool(e.get("billboard", True)))
        seed = req.seed if e.get("vary_seed", True) else None
        result: Dict[str, Any] = {}

        def run() -> None:
            _local.sink = live.thinking
            try:
                result["res"] = run_episode(kit, prompt, seed=seed, trace=live)
            except BaseException as exc:  # noqa: BLE001 - reported on the stream
                result["exc"] = exc
            finally:
                _local.sink = None
                out.put((_DONE, None))

        budget = (spec.budget or {}).get("usd_per_build")
        cap = e.get("session_usd_cap")
        head = [_line(f"recipe {self.recipe} ({spec.method}, builder "
                      f"{spec.roles().get('builder')}"
                      + (f", dreams via {spec.image.split('?')[0]}" if spec.image else "")
                      + ")", "stage"),
                _line(f"prompt: {prompt}", "info")]
        if budget is not None or cap is not None:
            spent = kit.ledger.parent.spent_usd if kit.ledger.parent is not None else 0.0
            head.append(_line(
                (f"budget ${budget:.2f}/build" if budget is not None else "")
                + (f", session ${spent:.2f} of ${cap:.2f}" if cap is not None else ""), "info"))
        yield BlockBatch(log=head)

        placed: Dict[Pos, str] = {}       # the build region as it stands in world
        board: Dict[Pos, str] = {}        # the dream wall
        thread = threading.Thread(target=run, daemon=True, name=f"recipe-{self.recipe}")
        thread.start()
        try:
            while True:
                if req.cancel is not None and req.cancel.is_set():
                    abort.set()
                    yield BlockBatch()    # lets the server see the cancel and stop
                    return
                try:
                    kind, payload = out.get(timeout=0.25)
                except queue.Empty:
                    continue
                if kind is _DONE:
                    break
                if kind == "log":
                    yield BlockBatch(log=payload)
                elif kind == "billboard":
                    try:
                        want = billboard_blocks(payload, spec.canvas,
                                                size=int(e.get("billboard_size", 40)))
                    except Exception as exc:  # noqa: BLE001 - a bad image is cosmetic
                        yield BlockBatch(log=[_line(f"billboard failed: {exc}", "warn")])
                        continue
                    blocks = diff_blocks(board, want)
                    board = want
                    yield BlockBatch(blocks=blocks, log=[_line(
                        "the dream is on the wall north of the build", "dream")])
                elif kind == "world":
                    idx, want = payload
                    blocks = diff_blocks(placed, want)
                    placed = want
                    if blocks:
                        yield BlockBatch(blocks=blocks, log=[_line(
                            f"placing attempt {idx}: {len(blocks)} changes", "info")])
        finally:
            abort.set()       # generator closed early (server cancel): stop paying

        if "exc" in result:
            raise result["exc"]
        res = result["res"]
        best = next((a for a in res.attempts if a.idx == res.best_idx), None)
        world = getattr(best, "world", None) if best is not None else None
        if world is None:
            world = getattr(res, "canvas", None)
        log: List[dict] = []
        blocks: List[Block] = []
        if world is not None:
            want = _world_blocks(world)
            blocks = diff_blocks(placed, want)
            if blocks and best is not None:
                log.append(_line(f"restoring the selected attempt {best.idx} "
                                 f"({_score_text(best)})", "score"))
        if res.truncated:
            log.append(_line(f"stopped early ({res.truncated}): {_clip(res.error or '', 140)}",
                             "warn"))
        if blocks or log:
            yield BlockBatch(blocks=blocks, log=log)

        led = kit.ledger
        yield BlockBatch(stats={
            "provider": f"recipe:{self.recipe} ({spec.roles().get('builder')})",
            "prompt": prompt,
            "cost_usd": round(live.spent(), 6),
            "cost_known": all(c.priced for c in live.charges()),
            "commands": best.report.n_commands if best is not None else 0,
            "failed_commands": best.report.n_failed if best is not None else 0,
            "attempts": len(res.attempts),
            "best_attempt": res.best_idx,
            "best_score": (round(best.score.value, 3) if best is not None and best.score
                           else None),
            "session_usd": round(led.parent.spent_usd, 4) if led.parent is not None else None,
            "trace": str(tdir.relative_to(self.root)) if tdir else None,
            "llm_seconds": round(res.elapsed_s, 1)})
