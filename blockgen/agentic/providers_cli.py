"""LLM backends that are not HTTP APIs: the Claude Code CLI, and an offline auto-mock.

Both plug into :func:`blockgen.agentic.providers.get_provider` (vendor keys
``claude-code`` / ``cc`` and ``mock:auto``), so a recipe swaps them in with one
string and every suite feature (response cache, spend gate, ledger) applies.

**ClaudeCodeProvider** runs ``claude -p --output-format json`` as a subprocess. Why
a CLI and not the Anthropic SDK: the CLI bills the user's *subscription* quota,
which is the cheapest way to run Claude models here (and ``anthropic`` is not
installed). The CLI cannot take image bytes on stdin, so each image part is
written to a content-addressed file in a private temp workspace and referenced by
absolute path; the model opens it with the ``Read`` tool, which is the only tool it
gets. Cost is quota, not dollars, so responses are unpriced (``priced=False``) and
the CLI's notional ``total_cost_usd`` goes into ``LLMResponse.meta`` only.

**AutoMockProvider** is the offline LLM that makes ANY recipe run end to end in CI:
it recognises a build turn by the action space's output rule and answers with that
space's own valid program, and a JSON turn by ``turns.JSON_RULE`` and answers with a
verdict that satisfies every rubric. It needs no script, unlike ScriptedProvider,
because recipes differ in which turns they send and in what order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from blockgen.agentic.providers import (_DEMO_PROGRAM, LLMProvider, LLMResponse,
                                        Message, _retry)

# ===========================================================================
# Claude Code CLI
# ===========================================================================
#: Appended to the flattened transcript. The model must Read the image files (it
#: cannot see a path otherwise) and must not wrap the answer in commentary, since
#: the reply is parsed exactly like an API reply.
CLOSING_INSTRUCTION = ("Open every referenced image with the Read tool before answering. "
                       "Reply with only the answer.")

#: Used when a request has no system message. An empty ``--system-prompt`` is not
#: guaranteed to replace Claude Code's own (coding-agent) system prompt.
FALLBACK_SYSTEM = "You are a helpful assistant. Answer the user's request directly."

#: Environment variables removed from the child's environment.
#: ``ANTHROPIC_API_KEY``: with it set the CLI bills API credit instead of the
#: subscription. The ``CLAUDE_CODE_*`` session variables: when the suite itself runs
#: inside a Claude Code session they would attach the child to the parent session
#: (messaging socket, IDE port) instead of running it standalone.
_STRIP_ENV = ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")
_KEEP_CLAUDE_ENV = frozenset({"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                              "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                              "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CONFIG_DIR"})

#: System prompts longer than this (bytes) go through --system-prompt-file.
SYSTEM_ARG_MAX = 96 * 1024

_VERSIONS: Dict[str, str] = {}
_VERSION_LOCK = threading.Lock()


def cli_version(binary: str = "claude") -> str:
    """``claude --version`` once per process and binary ('missing' if absent).

    It is part of the provider params, hence of the response-cache key: a CLI
    update can change the model's harness, so it must never serve stale replies.
    """
    with _VERSION_LOCK:
        if binary not in _VERSIONS:
            try:
                out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                     timeout=30)
                _VERSIONS[binary] = (out.stdout.strip().split() or ["unknown"])[0] \
                    if out.returncode == 0 else "unknown"
            except (OSError, subprocess.SubprocessError):
                _VERSIONS[binary] = "missing"
        return _VERSIONS[binary]


def child_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for the ``claude`` subprocess (see ``_STRIP_ENV``)."""
    env = dict(os.environ if base is None else base)
    for k in list(env):
        if k in _STRIP_ENV or (k.startswith("CLAUDE_CODE_") and k not in _KEEP_CLAUDE_ENV):
            env.pop(k)
    return env


class ClaudeCodeProvider(LLMProvider):
    """The local Claude Code CLI (``claude -p``) as an LLM backend, on subscription quota.

    Params that change the answer (``max_turns``, ``safe_mode``, ``bare``,
    ``effort``, ``cli_version`` and any gen params such as ``max_tokens``) live in
    ``self.params`` and therefore in the cache key; operational knobs (binary,
    timeout, concurrency) do not, so changing a timeout keeps the cache.

    ``safe_mode=True`` (``--safe-mode``) disables the user's hooks, plugins,
    CLAUDE.md and MCP servers while keeping OAuth subscription auth. ``--bare`` would
    do the same but reads only ``ANTHROPIC_API_KEY`` (never OAuth), i.e. it would
    bill API credit; it is therefore off by default and only for API-key setups.
    """

    name = "claude-code"
    spends = True               # uses subscription quota: cache misses are gated
    billing = "subscription"
    #: Seconds before the single retry of a transient failure (tests set 0).
    retry_delay = 2.0
    _SEMS: Dict[str, threading.Semaphore] = {}
    _SEM_LOCK = threading.Lock()

    def __init__(self, model: str = "sonnet", *, binary: str = "claude", timeout_s: float = 600,
                 max_turns: int = 4, concurrency: int = 2, safe_mode: bool = True,
                 bare: bool = False, effort: Optional[str] = None,
                 keep_workspace: bool = False, **params: Any) -> None:
        super().__init__(model, max_turns=int(max_turns), safe_mode=bool(safe_mode),
                         bare=bool(bare), effort=effort, cli_version=cli_version(binary),
                         **params)
        self.binary = binary
        self.timeout_s = float(timeout_s)
        self.keep_workspace = keep_workspace
        # Class-level, per binary: every provider instance in the process shares the
        # limit, because the quota (and the machine) is shared, not the instance.
        with self._SEM_LOCK:
            self._sem = self._SEMS.setdefault(binary, threading.Semaphore(max(1, int(concurrency))))
        self.last_argv: List[str] = []

    # --- request assembly ------------------------------------------------------
    @staticmethod
    def _workspace() -> Path:
        root = os.environ.get("BLOCKGEN_SCRATCH") or None
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        # Absolute: the child runs with cwd=td, so a relative --add-dir (from a
        # relative $BLOCKGEN_SCRATCH) would resolve to td/td and every Read would fail.
        return Path(tempfile.mkdtemp(prefix="blockgen_cc_", dir=root)).resolve()

    @staticmethod
    def _write_image(td: Path, png: bytes) -> Path:
        # Same digest as Part.fingerprint, so a file name identifies the image
        # across the request log and the cache key.
        path = td / f"img_{hashlib.sha256(png).hexdigest()[:32]}.png"
        if not path.exists():
            path.write_bytes(png)
        return path.resolve()

    def build_request(self, messages: Sequence[Message], td: Path) -> Tuple[str, str, int]:
        """(system prompt, stdin transcript, number of distinct images).

        System text goes to ``--system-prompt``. Every other turn is flattened into
        ``<turn role="...">`` blocks with ``[image: /abs/path]`` markers in part
        order. Images in a system message (which no vendor accepts there) open the
        transcript as a ``system`` turn rather than being dropped.
        """
        system = "\n\n".join(m.text for m in messages if m.role == "system" and m.text)
        images: Dict[Path, None] = {}
        blocks: List[str] = []
        for m in messages:
            if m.role == "system" and all(p.kind == "text" for p in m.parts):
                continue
            chunks: List[str] = []
            for p in m.parts:
                if p.kind == "text":
                    if m.role != "system" and p.text:
                        chunks.append(p.text)
                else:
                    path = self._write_image(td, p.image_png or b"")
                    images[path] = None
                    chunks.append(f"[image: {path}]")
            blocks.append(f'<turn role="{m.role}">\n' + "\n".join(chunks) + "\n</turn>")
        body = "\n\n".join(blocks + [CLOSING_INSTRUCTION]) + "\n"
        return system or FALLBACK_SYSTEM, body, len(images)

    def argv(self, system: str, td: Path, n_images: int = 0) -> List[str]:
        """The CLI invocation. Flags verified against ``claude --help`` (2.1.x);
        ``--max-turns`` is not in the help text but is accepted by the parser.

        Each image Read is a tool turn, so ``max_turns`` is raised to
        ``n_images + 2`` when needed rather than letting a many-view critique fail
        with ``error_max_turns``.
        """
        p = self.params
        turns = max(int(p["max_turns"]), n_images + 2)
        sys_flag, sys_val = "--system-prompt", system
        if len(system.encode()) > SYSTEM_ARG_MAX:
            # Linux caps ONE argv string at 128 KiB (MAX_ARG_STRLEN); a long palette
            # or ontology table would make exec fail with E2BIG. The CLI's
            # --system-prompt-file (accepted by 2.1.x, documented under --bare)
            # carries it instead; the file lives in the private workspace.
            path = Path(td) / "system_prompt.txt"
            path.write_text(system)
            sys_flag, sys_val = "--system-prompt-file", str(path)
        argv = [self.binary, "-p", "--output-format", "json", "--model", self.model,
                sys_flag, sys_val, "--tools", "Read", "--allowedTools", "Read",
                "--add-dir", str(td), "--max-turns", str(turns), "--strict-mcp-config",
                "--setting-sources", "", "--no-session-persistence"]
        if p.get("safe_mode"):
            argv.append("--safe-mode")
        if p.get("bare"):
            argv.append("--bare")
        if p.get("effort"):
            argv += ["--effort", str(p["effort"])]
        return argv

    # --- call ----------------------------------------------------------------
    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        if self.params.get("cli_version") == "missing":
            raise RuntimeError(f"claude-code: binary {self.binary!r} not found on PATH "
                               f"(install Claude Code or pass binary=...)")
        td = self._workspace()
        try:
            system, body, n_images = self.build_request(messages, td)
            argv = self.argv(system, td, n_images)
            self.last_argv = argv
            env = child_env()

            def call() -> Tuple[Dict[str, Any], float]:
                t0 = time.time()
                with self._sem:
                    try:
                        proc = subprocess.run(argv, input=body, cwd=str(td), env=env,
                                              timeout=self.timeout_s, capture_output=True,
                                              text=True)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError(f"claude-code timed out after {self.timeout_s:g}s") \
                            from exc
                return self._parse(proc), time.time() - t0

            blob, latency = _retry(call, attempts=2, base_delay=self.retry_delay)
        finally:
            if not self.keep_workspace:
                shutil.rmtree(td, ignore_errors=True)
        usage = blob.get("usage") or {}
        prompt_tokens = sum(int(usage.get(k) or 0) for k in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        meta = {k: blob[k] for k in ("total_cost_usd", "num_turns", "duration_ms",
                                     "session_id") if k in blob}
        meta["n_images"] = n_images
        return LLMResponse(text=str(blob.get("result") or ""),
                           model=f"claude-code/{self.model}",
                           prompt_tokens=prompt_tokens,
                           completion_tokens=int(usage.get("output_tokens") or 0),
                           latency_s=latency, finish_reason=str(blob.get("subtype") or ""),
                           meta=meta)

    @staticmethod
    def _parse(proc: subprocess.CompletedProcess) -> Dict[str, Any]:
        """stdout JSON -> dict; non-zero exit, bad JSON or ``is_error`` raise.

        The error text carries the stderr tail (and the result text for
        ``is_error``) so that _retry can recognise transient failures such as
        overload or rate limits by their wording.
        """
        tail = (proc.stderr or "").strip()[-800:]
        if proc.returncode != 0:
            raise RuntimeError(f"claude-code exited {proc.returncode}: {tail or proc.stdout[-800:]}")
        try:
            blob = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude-code: stdout is not JSON ({exc}): "
                               f"{proc.stdout[:300]!r} {tail}") from exc
        if isinstance(blob, list):  # some versions emit the event list; take the result
            blob = next((e for e in reversed(blob) if isinstance(e, dict)
                         and e.get("type") == "result"), {})
        if not isinstance(blob, dict) or "result" not in blob and not blob.get("is_error"):
            raise RuntimeError(f"claude-code: no result in output: {proc.stdout[:300]!r}")
        if blob.get("is_error"):
            raise RuntimeError(f"claude-code error ({blob.get('subtype', '')}): "
                               f"{str(blob.get('result', ''))[:500]} {tail}")
        return blob

    def describe(self) -> str:
        return f"claude-code:{self.model}"


# ===========================================================================
# Offline auto-mock
# ===========================================================================
#: One verdict that satisfies every rubric the suite asks for (dream review,
#: reachability, judge, pick, compare-as-JSON): a pass, no issues, index 0.
AUTO_JSON: Dict[str, Any] = {
    "score": 7, "buildable": True, "on_prompt": True, "single_structure": True,
    "view_ok": True, "consistent_with_refs": True, "keep_composition": True,
    "issues": [], "revised_prompt": "", "index": 0, "unreachable": False,
    "differences": [], "missing": [], "wrong": [], "fix_hint": "",
}

#: Used only if turns.py (WP2) is not importable: a JSON turn is then recognised
#: by "json" in the last few hundred characters of the turn.
_JSON_TAIL_CHARS = 300

_SPACES: Optional[List[Tuple[str, str, str]]] = None
_SPACES_LOCK = threading.Lock()


def _action_spaces() -> List[Tuple[str, str, str]]:
    """(name, output_rule, mock_reply) for every registered ActionSpace, once.

    A space that cannot be built with default params is skipped: the mock must
    not fail because an unrelated component needs configuration.
    """
    global _SPACES
    with _SPACES_LOCK:
        if _SPACES is None:
            from blockgen.agentic import core
            out = []
            for name in core.names("actions"):
                try:
                    space = core.make("actions", name)
                    out.append((name, str(space.output_rule()), str(space.mock_reply())))
                except Exception:  # noqa: BLE001 - see docstring
                    continue
            _SPACES = out
        return list(_SPACES)


def reset_action_space_cache() -> None:
    """Forget the scanned action spaces (tests that register new ones)."""
    global _SPACES
    with _SPACES_LOCK:
        _SPACES = None
        _SPACE_OBJS.clear()


def _json_rule() -> Optional[str]:
    try:
        from blockgen.agentic import turns  # WP2
    except ImportError:
        return None
    rule = getattr(turns, "JSON_RULE", None)
    return str(rule).strip() if rule else None


def _detect_space(turn: str, system: str) -> Optional[Tuple[str, str]]:
    """(name, mock_reply) of the ActionSpace whose output rule appears in ``turn``
    (else in ``system``); the one appearing LAST wins, then the longest rule."""
    for text in (turn, system):
        best: Optional[Tuple[int, int, str, str]] = None
        for name, out_rule, reply in _action_spaces():
            if not out_rule:
                continue
            pos = text.rfind(out_rule)
            if pos >= 0:
                cand = (pos, len(out_rule), name, reply)
                best = cand if best is None or cand[:2] > best[:2] else best
        if best is not None:
            return best[2], best[3]
    return None


# --- staged methods (blueprint / floors / turnaround) ------------------------
_CANVAS_RE = re.compile(r"Canvas: (\d+) x (\d+) x (\d+) blocks")
_MAX_STOREYS_RE = re.compile(r"at most (\d+) storey")
_MAX_ROOMS_RE = re.compile(r"At most (\d+) rooms in total")


def _fit_mock_rooms(bp: Any, max_rooms: int) -> None:
    """Make the mock plan obey the turn's room limit, as a model is asked to.

    Why: without this, repair truncates the mock's 5-room plan to the default 4 and
    a door into the dropped upper room turns into an exterior door on storey 1, a
    door into thin air. Rooms are dropped from the top storey down, then openings
    that named them; a storey left with one room keeps it as the whole floor.
    """
    from blockgen.agentic.blueprint import OUTSIDE
    keep = list(bp.rooms)
    while len(keep) > max(0, max_rooms):
        keep.pop()
    names = {r.name for r in keep}
    bp.rooms = keep
    if not keep:
        # One open space per storey: keep the exterior openings, unnamed.
        bp.openings = [o for o in bp.openings if OUTSIDE in o.between]
        for o in bp.openings:
            o.between = ("", OUTSIDE)
        return
    bp.openings = [o for o in bp.openings
                   if all(t in names or t == OUTSIDE for t in o.between)]
    for st in {r.storey for r in keep}:
        on = [r for r in keep if r.storey == st]
        if len(on) == 1 and st > 0:
            on[0].rect = bp.footprint_of(st)
_REGION_RE = re.compile(r"ALLOWED REGION: x (-?\d+)\.\.(-?\d+), y (-?\d+)\.\.(-?\d+), "
                        r"z (-?\d+)\.\.(-?\d+)")
_BOX_RE = re.compile(r"x (-?\d+)\.\.(-?\d+), y (-?\d+)\.\.(-?\d+), z (-?\d+)\.\.(-?\d+)")
#: turns_staged.region_block's first line ("UNIT BUILD: room 'hall' (...)") and
#: its keep-clear line, which the mock furnishing avoids.
_UNIT_KIND_RE = re.compile(r"UNIT BUILD: (room|storey) ")
_KEEP_CLEAR = "Keep these cells clear"
_STOREY_WALLS = "Build the floor slab, outer walls"
_FRONT_DOOR = "The front door is on the +z side"
_SHELL_TURN = "Build the SHELL of this building"
_SHELL_JSON = "neighbouring rooms share a wall line):"
_ROOF_RE = re.compile(r"the roof starts at y=(-?\d+) over the footprint x (-?\d+)\.\.(-?\d+), "
                      r"z (-?\d+)\.\.(-?\d+)")
_REPLACES = "Your program REPLACES everything in the region"
#: Literal of sheets.SHEET_REVIEW_TAG, used if that module is not importable.
_SHEET_REVIEW_TAG = "TURNAROUND SHEET REVIEW"
_SPACE_OBJS: Dict[str, Any] = {}


def _space_from_dsl(name: str, dsl_text: str) -> str:
    """``dsl_text`` in action space ``name``'s own format (identity if the space
    has no ``from_dsl``, e.g. a test fake that speaks DSL)."""
    space = _SPACE_OBJS.get(name)
    if space is None:
        from blockgen.agentic import core
        try:
            space = core.make("actions", name)
        except Exception:  # noqa: BLE001 - fall back to plain DSL
            return dsl_text
        _SPACE_OBJS[name] = space
    conv = getattr(space, "from_dsl", None)
    return conv(dsl_text) if callable(conv) else dsl_text


def mock_furnishing(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, *,
                    on_floor: bool = False,
                    keep_clear: Sequence[Tuple[int, int, int, int, int, int]] = (),
                    walls: bool = False, front_door: bool = False) -> str:
    """A small furnishing INSIDE the box (canonical DSL): a carpet over the lowest
    free layer, a crafting table, a lantern and a stair 'chair', each clamped to it.

    ``on_floor``: the box's lowest layer is an existing floor (a room unit), so the
    carpet lies ON it (y0 + 1) instead of replacing it. ``keep_clear``: boxes
    ``(x0, x1, y0, y1, z0, z1)`` that must stay air (doorways, stairs); the carpet
    is cleared there again and no item is put in one, so a mock unit turn does not
    trip the region's revert-and-report path on every room with a door.
    ``walls``: also a ring of outer walls on the box border (a storey turn that
    asks for them), with a doorway on the +z side if ``front_door``.
    """
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    base = min(y0 + 1, y1) if on_floor else y0
    up = min(base + 1, y1) if on_floor else min(y0 + 1, y1)

    def blocked(x: int, y: int, z: int) -> bool:
        return any(a <= x <= b and c <= y <= d and e <= z <= f
                   for (a, b, c, d, e, f) in keep_clear)

    if not any(c <= base <= d for (_a, _b, c, d, _e, _f) in keep_clear):
        lines = [f"fill white_carpet {x0} {base} {z0} {x1} {base} {z1}"]
    else:
        # One fill per free run of each row: a keep-clear cell may hold a stair,
        # which neither a carpet nor a clear may touch.
        lines = []
        for z in range(z0, z1 + 1):
            x = x0
            while x <= x1:
                if blocked(x, base, z):
                    x += 1
                    continue
                start = x
                while x + 1 <= x1 and not blocked(x + 1, base, z):
                    x += 1
                lines.append(f"fill white_carpet {start} {base} {z} {x} {base} {z}")
                x += 1
    if walls and y1 > base:
        # A storey turn under ``shell: none`` asks for the outer walls too: one
        # cobblestone ring on the region border above the slab, split into runs
        # around keep-clear cells, with a doorway gap on the +z side (the front).
        door_x = (x0 + x1) // 2
        ring = ([(x, z0) for x in range(x0, x1 + 1)] + [(x, z1) for x in range(x0, x1 + 1)]
                + [(x0, z) for z in range(z0 + 1, z1)] + [(x1, z) for z in range(z0 + 1, z1)])
        for y in range(base + 1, y1 + 1):
            for (x, z) in ring:
                if blocked(x, y, z) or (front_door and z == z1 and x == door_x
                                        and y <= base + 2):
                    continue
                lines.append(f"set cobblestone {x} {y} {z}")
        x0, x1, z0, z1 = x0 + 1, x1 - 1, z0 + 1, z1 - 1   # items go inside the ring
    item_y = base if on_floor else up
    corners = [(x0, z0), (x1, z1), (x0, z1), (x1, z0)]
    free = [(x, z) for (x, z) in corners if not blocked(x, item_y, z)]
    for block, (x, z) in zip(("crafting_table", "lantern", "oak_stairs"), free):
        lines.append(f"set {block} {x} {item_y} {z}")
    return "\n".join(lines) + "\n"


def mock_roof(y: int, x0: int, x1: int, z0: int, z1: int, H: int) -> str:
    """A stepped hip roof over the footprint (canonical DSL): one-block-thick filled
    layers from ``y`` up, each one cell smaller on every side, below the canvas top.

    Why: the roof turn (``roof: llm``) otherwise gets the generic house reply, which
    lands beside the building instead of on it.
    """
    lines = []
    k = 0
    while x0 + k <= x1 - k and z0 + k <= z1 - k and y + k < H:
        lines.append(f"fill spruce_planks {x0 + k} {y + k} {z0 + k} {x1 - k} {y + k} {z1 - k}")
        k += 1
    return "\n".join(lines) + "\n"


_FIT_RE = re.compile(r"must fit (\d+)x(\d+)x(\d+)")


def mock_plan(turn: str) -> str:
    """A plain-text plan for the text plan turn (turns.plan), sized to its canvas.

    Prose, not a program: the plan is pasted into the build turn as "Your plan",
    and a mock that answered it with commands would hide a prompt that forgot to
    ask for plain text.
    """
    m = _FIT_RE.search(turn)
    w, h, d = (int(v) for v in m.groups()) if m else (32, 32, 32)
    x0, z0 = max(0, w // 2 - 6), max(0, d // 2 - 5)
    x1, z1 = min(w - 1, x0 + 11), min(d - 1, z0 + 9)
    seen = ("- TARGET: a one-storey cottage, gable roof, door at the front.\n"
            if "TARGET" in turn else "")
    return (f"{seen}- Footprint x {x0}..{x1}, z {z0}..{z1}; walls y 1..4, roof from y 5.\n"
            "- Palette: cobblestone foundation, oak_planks walls, oak_log trim, "
            "spruce_stairs roof, glass_pane windows.\n"
            f"- Parts: foundation y 0 over the footprint; walls x {x0}..{x1}, y 1..4, "
            f"z {z0}..{z1}; door x {(x0 + x1) // 2}, y 1..2, z {z1}; roof gable from y 5.\n"
            "- Player scale: door 1 x 2, 3 blocks of clear ceiling, rooms 3 x 3 or more.\n"
            "- Details: log corner posts; a chimney at the back.")


def mock_brief(turn: str) -> str:
    """A prose design brief for the plan_source 'text' turn, describing the same
    house :func:`blueprint.mock_blueprint` draws, so the JSON turn that follows
    (answered with that blueprint) transcribes it faithfully."""
    m = _CANVAS_RE.search(turn)
    size = tuple(int(v) for v in m.groups()) if m else (32, 32, 32)
    two = size[1] >= 24 and not re.search(r"at most 1 storey", turn)
    lines = ["Footprint: 20 x 16 blocks, centred on the canvas, front door on the +z side.",
             "Storeys: " + ("two, each with 4 blocks of clear height."
                            if two else "one, with 4 blocks of clear height."),
             "Rooms (sizes include walls): storey 0 has an entrance hall 10 x 10 at the "
             "front left, a kitchen 11 x 10 at the front right sharing its wall, and a "
             "bedroom 20 x 7 across the back.",
             "Circulation: the front door opens into the hall; the hall has a door to the "
             "kitchen and one to the bedroom" + ("; stairs climb from the hall to the "
                                                 "study above." if two else "."),
             "Openings: front door 1 x 2 on the hall's front wall; two-wide windows on the "
             "kitchen and bedroom outer walls, one block above the floor.",
             "Roof and materials: gable along x; stone_bricks walls, oak_planks floor, "
             "oak_log trim, spruce_stairs roof."]
    if two:
        lines.insert(3, "Storey 1: a study 10 x 16 and a loft bedroom 11 x 16, sharing a "
                        "wall, with a door between them.")
    return "\n".join(lines)


def _plan_head() -> str:
    try:
        from blockgen.agentic.turns import PLAN_HEAD
    except ImportError:
        return ""
    return str(PLAN_HEAD)


def _describe_reply(turn: str) -> Optional[Tuple[str, str]]:
    """A design spec or an image caption for the describe stage's turns
    (``describe.SPEC_TAG`` / ``CAPTION_TAG``), as JSON or prose like the turn asks."""
    try:
        from blockgen.agentic.describe import mock_reply
    except ImportError:
        return None
    return mock_reply(turn)


def _staged_reply(turn: str, system: str) -> Optional[Tuple[str, str]]:
    """Replies for the staged methods' turns, keyed on their literal tags.

    Why the mock needs this: a unit turn's region reverts any write outside it, so
    the generic house reply would be clipped away and the accept path never run;
    a blueprint turn needs a plan, not a verdict. ``None`` = not a staged turn (or
    the staged modules are not importable): the generic rules apply.
    """
    try:
        from blockgen.agentic import turns_staged as ts
    except ImportError:
        return None
    if ts.VERIFY_TAG in turn:
        return "staged:verify", json.dumps({"ok": True, "issues": [], "blueprint": None})
    try:
        from blockgen.agentic import sheets
        sheet_tag = str(getattr(sheets, "SHEET_REVIEW_TAG", _SHEET_REVIEW_TAG))
    except ImportError:
        sheet_tag = _SHEET_REVIEW_TAG
    if sheet_tag in turn:
        return "staged:sheet_review", json.dumps(
            {"consistent": True, "score": 8, "issues": [], "revised_prompt": ""})
    brief_tag = getattr(ts, "BRIEF_TAG", None)
    if brief_tag and brief_tag in turn and ts.BLUEPRINT_TAG not in turn:
        return "staged:brief", mock_brief(turn)
    if ts.BLUEPRINT_TAG in turn:
        try:
            from blockgen.agentic.blueprint import mock_blueprint
        except ImportError:
            return None
        m = _CANVAS_RE.search(turn)
        size = tuple(int(v) for v in m.groups()) if m else (32, 32, 32)
        m_st = _MAX_STOREYS_RE.search(turn)
        storeys = 2 if size[1] >= 24 else 1
        if m_st:
            storeys = max(1, min(storeys, int(m_st.group(1))))
        bp = mock_blueprint(size, storeys=storeys)
        if "No rooms:" in turn:
            _fit_mock_rooms(bp, 0)
        else:
            m_r = _MAX_ROOMS_RE.search(turn)
            if m_r:
                _fit_mock_rooms(bp, int(m_r.group(1)))
        return "staged:blueprint", bp.to_json()
    if _SHELL_TURN in turn:
        # shell: llm. Answer with the plan's own deterministic shell, so the arm's
        # later stages run on a real building (the generic reply is a small hut
        # in a canvas corner, which every room region then misses).
        start = turn.find(_SHELL_JSON)
        if start >= 0:
            line = turn[start + len(_SHELL_JSON):].lstrip("\n").split("\n", 1)[0]
            try:
                from blockgen.agentic.blueprint import Blueprint
                dsl_text = Blueprint.from_json(line).shell_dsl()
            except Exception:  # noqa: BLE001 - fall through to the generic reply
                dsl_text = ""
            if dsl_text.strip():
                space = _detect_space(turn, system)
                name = space[0] if space is not None else "dsl"
                return f"staged:shell:{name}", _space_from_dsl(name, dsl_text)
    m_roof = _ROOF_RE.search(turn)
    if m_roof and ts.UNIT_TAG not in turn:
        y, x0, x1, z0, z1 = (int(v) for v in m_roof.groups())
        m = _CANVAS_RE.search(turn)
        H = int(m.group(2)) if m else 32
        space = _detect_space(turn, system)
        name = space[0] if space is not None else "dsl"
        return f"staged:roof:{name}", _space_from_dsl(name, mock_roof(y, x0, x1, z0, z1, H))
    if ts.UNIT_TAG in turn:
        m = _REGION_RE.search(turn)
        if m is None:
            return None
        x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
        kind = _UNIT_KIND_RE.search(turn)
        keep: List[Tuple[int, int, int, int, int, int]] = []
        kc = turn.find(_KEEP_CLEAR)
        if kc >= 0:
            line = turn[kc:].split("\n", 1)[0]
            keep = [tuple(int(v) for v in b.groups())  # type: ignore[misc]
                    for b in _BOX_RE.finditer(line)]
        dsl_text = mock_furnishing(x0, x1, y0, y1, z0, z1,
                                   # a room's floor exists, unless the turn says the
                                   # program replaces it (rewrite mode)
                                   on_floor=bool(kind and kind.group(1) == "room"
                                                 and _REPLACES not in turn),
                                   keep_clear=keep,
                                   walls=bool(kind and kind.group(1) == "storey"
                                              and _STOREY_WALLS in turn),
                                   front_door=_FRONT_DOOR in turn)
        space = _detect_space(turn, system)
        name = space[0] if space is not None else "dsl"
        return f"staged:unit:{name}", _space_from_dsl(name, dsl_text)
    return None


class AutoMockProvider(LLMProvider):
    """Offline LLM that answers any suite turn plausibly (``mock:auto``).

    Decision order for each call, on the LAST user message (``turn``) and the
    system message:

    1. ``script``: ``{substring: reply}`` rules matched against ``turn`` (a dict
       reply is sent as JSON). This is how a test forces e.g. ``REDREAM: ...``.
    1a. Describe-stage turn (``describe.SPEC_TAG`` / ``CAPTION_TAG``): a design
        spec or an image caption, JSON or prose as the turn asks.
    1b. Staged turn (a literal tag of ``turns_staged`` / ``sheets``): a blueprint,
       a prose design brief, a passing check, or a small furnishing inside the
       turn's ALLOWED REGION in the detected action space's format (see
       :func:`_staged_reply`).
    1c. Text plan turn (``turns.PLAN_HEAD`` and "Plain text only."): a prose plan
       sized to the turn's canvas (:func:`mock_plan`).
    2. JSON turn (``turn`` ends with ``turns.JSON_RULE``): ``AUTO_JSON`` updated
       with ``json_overrides``.
    3. Build turn: the registered ActionSpace whose ``output_rule()`` appears in
       ``turn`` (else in the system message) answers with its ``mock_reply()``. If
       several match, the one appearing LAST wins (the rule ends a turn), then the
       longest.
    4. Otherwise the built-in demo program.

    Token counts are estimates (chars/4, +800 per image) so ``estimate`` mode has
    something to price; the model name is unpriced, so nothing is ever charged.
    """

    name = "mock"
    spends = False
    billing = "mock"

    def __init__(self, model: str = "auto", *,
                 script: Optional[Mapping[str, Union[str, Mapping[str, Any]]]] = None,
                 json_overrides: Optional[Mapping[str, Any]] = None, **params: Any) -> None:
        super().__init__(model, **params)
        self.script = dict(script or {})
        self.json_overrides = dict(json_overrides or {})
        self.calls: List[List[Message]] = []
        #: 'script:<k>' | 'describe:<kind>' | 'staged:<kind>' | 'plan' | 'json' |
        #: 'actions:<name>' | 'demo'
        self.decisions: List[str] = []

    def _decide(self, messages: Sequence[Message]) -> Tuple[str, str]:
        users = [m for m in messages if m.role == "user"]
        turn = users[-1].text if users else (messages[-1].text if messages else "")
        system = "\n".join(m.text for m in messages if m.role == "system")
        for needle, reply in self.script.items():
            if needle in turn:
                text = json.dumps(reply) if isinstance(reply, Mapping) else str(reply)
                return f"script:{needle}", text
        described = _describe_reply(turn)
        if described is not None:
            return described
        staged = _staged_reply(turn, system)
        if staged is not None:
            return staged
        plan_head = _plan_head()
        if plan_head and plan_head in turn and "Plain text only." in turn:
            return "plan", mock_plan(turn)
        rule = _json_rule()
        tail = turn.rstrip()
        if (tail.endswith(rule) if rule else "json" in tail[-_JSON_TAIL_CHARS:].lower()):
            return "json", json.dumps({**AUTO_JSON, **self.json_overrides})
        space = _detect_space(turn, system)
        if space is not None:
            return f"actions:{space[0]}", space[1]
        return "demo", _DEMO_PROGRAM

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        self.calls.append(list(messages))
        decision, text = self._decide(messages)
        self.decisions.append(decision)
        n_chars = sum(len(p.text) for m in messages for p in m.parts if p.kind == "text")
        n_images = sum(1 for m in messages for p in m.parts if p.kind == "image")
        return LLMResponse(text=text, model=self.model,
                           prompt_tokens=n_chars // 4 + 800 * n_images,
                           completion_tokens=len(text) // 4, finish_reason="stop",
                           meta={"decision": decision})

    def describe(self) -> str:
        return "mock:auto"


__all__ = ["AUTO_JSON", "AutoMockProvider", "CLOSING_INSTRUCTION", "ClaudeCodeProvider",
           "FALLBACK_SYSTEM", "child_env", "cli_version", "mock_brief", "mock_furnishing",
           "mock_plan", "reset_action_space_cache"]
