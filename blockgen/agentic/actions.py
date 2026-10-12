"""Action spaces: the ways a model can express a build, all lowered to one IR.

An *action space* is the representation the model writes in. The suite compares
several of them on the same loop, so each one is a registered component
(``actions: dsl`` / ``blocks`` / ``python`` / ``mixed`` in a recipe):

``dsl``     the WorldEdit-style command language of :mod:`blockgen.agentic.dsl`
``blocks``  one ``<block[state]> x y z`` line per voxel (the BrickGPT format of
            ``scripts/train_llm_brickgpt.py``, but validated by blockstate)
``python``  a sandboxed Python program calling ``fill/walls/...`` (pyapi.py)
``mixed``   fenced ```` ```python ```` / ```` ```dsl ```` / ```` ```blocks ```` sections

Why one IR: every space *lowers* to canonical DSL text that the existing
:func:`blockgen.agentic.dsl.run_program` executes. Op counts, the clipboard,
``ExecutionReport`` feedback and the deploy replay (``ProgramRunner``) then stay a
single implementation, and a build is replayable whatever the model wrote. A
second executor per representation would drift, quietly, exactly the way the
streaming and batch token decoders did in deploy.

Line numbers: lowering issues always carry the SOURCE line (what the model can
act on). ``dsl`` lowers by identity (byte-parity with ``BuildAgent``, whose cache
keys and repair prompts depend on it) and ``blocks`` lowers line-aligned (lowered
line i is source line i), so even execution warnings point at the model's own
lines. ``python`` cannot be line-aligned (a loop emits many ops), so each op is
annotated ``# L<n>`` with the source line that produced it.

Every space ships its own grammar text, its own one-sentence output rule (the
offline ``mock:auto`` LLM recognises a space by finding that rule in the system
message, so the rules are distinct and no space's instructions contain another
space's rule), in-format examples and a ``mock_reply`` that is a small valid house.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.blockstate import (_LOG_AXIS, _LOG_IDS, _SLAB_IDS, _STAIR_FACING,
                                         PALETTE, STAIRS_IDS, UnknownBlockError,
                                         parse_block, resolve_block)
from blockgen.agentic.core import Lowered, PromptCtx, _suggest, register
from blockgen.agentic.dsl import (COMMANDS, CommandSpec, Issue, Param, command_reference,
                                  lookup, parse_line, parse_program, run_program)
from blockgen.agentic.examples import EXAMPLES, Example, select_examples
from blockgen.agentic.prompts import METHOD, RULES, scale_hint

#: Canvas used to convert example programs (big enough that nothing clips).
CONVERT_SIZE: Tuple[int, int, int] = (64, 64, 64)

#: The per-block format explodes with build size; past this many voxels an example
#: costs more context than it teaches (and BrickGPT-style training data sat below it).
MAX_BLOCK_EXAMPLE = 400

_FENCES = ("```", "~~~")

# ---------------------------------------------------------------------------
# The offline mock house, shared by every space.
# ---------------------------------------------------------------------------
# Split into massing + detail so ``mixed`` can show one part per sub-language.
# DETAIL is additive only (``set`` overwrites, no ``clear``): every format,
# including per-block lines run on an empty canvas, can then express it and the
# four spaces' mock replies build the IDENTICAL 7x8x7 voxels (an acceptance test).
MOCK_MASSING = """\
# foundation, then the shell standing on it
fill cobblestone 0 0 0 6 0 6
walls oak_planks 0 1 0 6 3 6
# timber corner posts
line oak_log 0 1 0 0 3 0
line oak_log 6 1 0 6 3 0
line oak_log 0 1 6 0 3 6
line oak_log 6 1 6 6 3 6
# pitched roof sitting on the wall tops
gable oak_stairs 0 4 0 6 4 6 axis=x"""

MOCK_DETAIL = """\
# door (two blocks tall), windows and a light
set oak_door 3 1 0
set oak_door 3 2 0
set glass_pane 1 2 0
set glass_pane 5 2 0
set glass_pane 0 2 3
set glass_pane 6 2 3
set lantern 3 1 3"""

MOCK_DSL = MOCK_MASSING + "\n" + MOCK_DETAIL + "\n"

#: The mock house as an Example (the per-block space's only example under the cap).
MOCK_EXAMPLE = Example(
    caption="Small oak hut with a cobblestone floor, log corner posts, a pitched "
            "roof, a door and glass windows.",
    program=MOCK_DSL, tags=("hut", "small", "oak", "house", "cabin", "simple"))


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------
def _canvas_text(size: Sequence[int]) -> str:
    sx, sy, sz = (int(v) for v in size)
    return (f"Canvas: {sx} wide (x) x {sy} tall (y) x {sz} deep (z). Valid coordinates are\n"
            f"x in 0..{sx - 1}, y in 0..{sy - 1}, z in 0..{sz - 1}.")


def _tail(ctx: PromptCtx) -> str:
    """Optional size nudge, shared with the legacy prompt (``prompts.scale_hint``)."""
    hint = scale_hint(ctx.target_blocks)
    return f"\n\n{hint}" if hint else ""


def _is_fence(stripped: str) -> bool:
    return stripped.startswith(_FENCES)


@functools.lru_cache(maxsize=4096)
def _resolve(spec: str) -> Optional[Tuple[int, int]]:
    """Cached resolve_block (a python program can emit 10^5 ops over ~10 names)."""
    try:
        return resolve_block(spec)
    except UnknownBlockError:
        return None


#: The state keys/values ``blockstate._apply_props`` actually folds into the
#: legacy data nibble, per family. Anything else is silently dropped there.
_STAIR_STATES: Dict[str, Tuple[str, ...]] = {
    "facing": tuple(_STAIR_FACING), "half": ("top", "bottom"),
    "upside_down": ("true", "false")}
_LOG_STATES: Dict[str, Tuple[str, ...]] = {"axis": tuple(_LOG_AXIS)}
_SLAB_STATES: Dict[str, Tuple[str, ...]] = {
    "type": ("top", "bottom", "double"), "half": ("top", "bottom")}


@functools.lru_cache(maxsize=4096)
def state_warning(spec: str) -> Optional[str]:
    """Why some of ``spec``'s bracket states will be IGNORED (None = all honoured).

    ``resolve_block`` folds only stair facing/half, log axis and slab type into
    the legacy data value and drops every other key or value without a word, so
    ``oak_stairs[facng=north]`` would quietly place a default-facing stair. This
    names what was dropped so the model can learn it. Unknown block names are
    not judged here (that is an error reported elsewhere).
    """
    ref = parse_block(spec)
    if not ref.props:
        return None
    try:
        block_id = resolve_block(ref.name)[0]
    except UnknownBlockError:
        return None
    allowed: Dict[str, Tuple[str, ...]] = (
        _STAIR_STATES if block_id in STAIRS_IDS else _LOG_STATES if block_id in _LOG_IDS
        else _SLAB_STATES if block_id in _SLAB_IDS else {})
    bad = []
    for k, v in ref.props.items():
        if k not in allowed:
            bad.append(f"'{k}'")
        elif v not in allowed[k]:
            bad.append(f"'{k}={v}' (use {'|'.join(allowed[k])})")
    if not bad:
        return None
    known = (", ".join(sorted(allowed)) if allowed else "none")
    return (f"block state {', '.join(bad)} ignored for '{ref.name}' "
            f"(states it understands: {known}); placed with the default state")


def _canonical_block(spec: str) -> str:
    """Whitespace-free, namespace-free block spec that resolves identically.

    ``"minecraft:oak_stairs[facing=north, half=top]"`` -> ``"oak_stairs[facing=north,half=top]"``:
    the DSL tokenizes on whitespace, so a spaced state list must be closed up.
    """
    return str(parse_block(spec))


_L_TAG = re.compile(r"# L(\d+)$", re.MULTILINE)


def _tag_line(line: str, src_line: int) -> str:
    """Suffix a lowered command line with its source line (blank/comment lines as-is)."""
    s = line.strip()
    if not s or s.startswith("#") or _is_fence(s):
        return line
    return f"{line}  # L{src_line}"


def remap_issues(lowered: Lowered, issues: Sequence[Issue]) -> List[Issue]:
    """Execution issues re-addressed to SOURCE lines, where the lowering knows them.

    ``dsl.run_program`` numbers issues by LOWERED line and strips comments from
    ``Issue.text``. ``dsl`` and ``blocks`` are line-aligned, so nothing changes;
    ``python`` ops (and every command of a ``mixed`` reply) carry a ``# L<n>``
    tag, which this reads back. Lowering
    issues (already source-addressed) must not be passed here.
    """
    lines = lowered.program_text.split("\n")
    out: List[Issue] = []
    tagged = False
    for iss in issues:
        ln = iss.line_no
        m = _L_TAG.search(lines[ln - 1]) if 0 < ln <= len(lines) else None
        tagged = tagged or m is not None
        out.append(Issue(int(m.group(1)), iss.text, iss.message, iss.severity) if m else iss)
    # A loop line emits many ops; one repeated problem is one issue (with a
    # count), not 20 copies crowding every other error out of the repair prompt.
    return dedupe_issues(out) if tagged else out


def dedupe_issues(issues: Sequence[Issue]) -> List[Issue]:
    """Collapse issues with the same (line, message, severity), first-seen order,
    appending ``(xN calls)`` to the message of any issue that repeated."""
    counts: Dict[Tuple[int, str, str], int] = {}
    first: Dict[Tuple[int, str, str], Issue] = {}
    for iss in issues:
        key = (iss.line_no, iss.message, iss.severity)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, iss)
    return [iss if counts[k] == 1 else
            Issue(iss.line_no, iss.text, f"{iss.message} (x{counts[k]} calls)", iss.severity)
            for k, iss in first.items()]


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------
#: Python names for DSL commands whose name shadows a builtin the sandbox keeps.
PY_NAMES: Dict[str, str] = {"set": "place"}
#: Extra Python names (aliases) per DSL command. ``block`` is the DSL alias of set.
PY_ALIASES: Dict[str, Tuple[str, ...]] = {"set": ("block", "set_block")}

_PY_KEYWORDS_OR_BUILTINS = frozenset({
    "set", "list", "dict", "range", "len", "min", "max", "sum", "abs", "round", "map",
    "filter", "zip", "any", "all", "int", "float", "str", "bool", "tuple", "print",
    "sorted", "reversed", "enumerate", "type", "id", "hash", "iter", "next", "open",
    "lambda", "def", "class", "for", "while", "if", "else", "in", "is", "not", "and",
    "or", "return", "pass", "break", "continue", "import", "from", "as", "with",
    "global", "nonlocal", "try", "except", "finally", "raise", "del", "yield", "assert",
})


def py_name(dsl_name: str) -> str:
    """The Python API name of a DSL command (``set`` -> ``place``)."""
    return PY_NAMES.get(dsl_name, dsl_name)


def py_names(spec: CommandSpec) -> List[str]:
    """Every Python name bound to ``spec``: the main name, then usable aliases.

    DSL aliases that would shadow a Python builtin/keyword the sandbox keeps are
    dropped (``set`` must stay the set type for ordinary Python).
    """
    out = [py_name(spec.name)]
    for a in (*PY_ALIASES.get(spec.name, ()), *spec.aliases):
        if a.isidentifier() and a not in _PY_KEYWORDS_OR_BUILTINS and a not in out:
            out.append(a)
    return out


def render_dsl_value(param: Param, value: Any) -> str:
    """One argument value as canonical DSL text (the inverse of ``Param.coerce``)."""
    if param.kind == "bool":
        return "true" if bool(value) else "false"
    if param.kind == "int":
        return str(int(value))
    if param.kind == "block":
        return _canonical_block(str(value))
    return str(value)


def canonical_call(spec: CommandSpec, args: Mapping[str, Any]) -> str:
    """Canonical DSL line: required params positional, optional ones as ``k=v``
    only when they differ from the default. One spelling per meaning, so two
    lowerings of the same build diff cleanly."""
    parts = [spec.name]
    for p in spec.params:
        if p.name not in args:
            continue
        v = args[p.name]
        if p.required:
            parts.append(render_dsl_value(p, v))
        elif v != p.default:
            parts.append(f"{p.name}={render_dsl_value(p, v)}")
    return " ".join(parts)


def _py_value(param: Param, value: Any) -> str:
    if param.kind == "bool":
        return "True" if bool(value) else "False"
    if param.kind == "int":
        return str(int(value))
    return repr(str(value))


def python_call(spec: CommandSpec, args: Mapping[str, Any]) -> str:
    """A parsed DSL call as one Python API call (required positional, rest kw)."""
    pos: List[str] = []
    kw: List[str] = []
    for p in spec.params:
        if p.name not in args:
            continue
        v = args[p.name]
        if p.required:
            pos.append(_py_value(p, v))
        elif v != p.default:
            kw.append(f"{p.name}={_py_value(p, v)}")
    return f"{py_name(spec.name)}({', '.join(pos + kw)})"


def dsl_to_python(text: str) -> str:
    """Convert a DSL program line-wise into Python API calls.

    Line i of the output is line i of the input (comments kept, prose commented
    out, fences blanked), so the converted examples read like the originals and
    an error line in one maps to the other.
    """
    out: List[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or _is_fence(stripped):
            out.append("")
            continue
        if stripped.startswith("#"):
            out.append(stripped)
            continue
        _code, sep, comment = raw.partition("#")
        call, _issue = parse_line(raw, i)
        if call is None:
            out.append(f"# {stripped}")
            continue
        line = python_call(call.spec, call.args)
        if sep:
            line += f"  # {comment.strip()}"
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


#: State variants tried when naming a voxel back from its legacy pair.
_STATE_VARIANTS: Tuple[str, ...] = tuple(
    [f"facing={f}" for f in ("east", "west", "south", "north")]
    + [f"facing={f},half=top" for f in ("east", "west", "south", "north")]
    + ["half=top", "axis=x", "axis=z", "axis=none", "type=top", "type=double"])


def _block_specs(text: str) -> List[str]:
    """Block-spec arguments a DSL program uses, in order of first appearance."""
    seen: Dict[str, None] = {}
    for call in parse_program(text).calls:
        for p in call.spec.params:
            if p.kind == "block" and p.name in call.args:
                v = str(call.args[p.name]).strip().lower()
                if v not in ("any", "*", "all"):
                    seen.setdefault(parse_block(v).name, None)
    return list(seen)


def _namer(bases: Iterable[str]) -> Callable[[Tuple[int, int]], Optional[str]]:
    """Map a legacy (id, data) pair to a block spec that resolves to EXACTLY it.

    The program's own block names are tried first (so ``oak_planks`` stays
    ``oak_planks``), then the curated palette; each base is tried bare and with
    the state variants blockstate can fold (stair facing/half, log axis, slab
    type). Exact equality is checked, so the round trip is voxel-identical.
    """
    order = list(dict.fromkeys([*bases, *PALETTE]))

    @functools.lru_cache(maxsize=None)
    def name(pair: Tuple[int, int]) -> Optional[str]:
        for base in order:
            for cand in (base, *(f"{base}[{v}]" for v in _STATE_VARIANTS)):
                if _resolve(cand) == pair:
                    return cand
        return None
    return name


def dsl_to_blocks(text: str, size: Sequence[int] = CONVERT_SIZE, *,
                  max_blocks: Optional[int] = None) -> str:
    """Run a DSL program and serialize the result as ``<block> x y z`` lines.

    Lines are ordered bottom-up (y, then z, then x) like BrickGPT's serializer, so
    the format teaches building from the ground. Raises ``ValueError`` when the
    build exceeds ``max_blocks`` or a voxel has no name that resolves back to the
    same legacy pair (the conversion would not be lossless).
    """
    canvas, _report, _ = run_program(text, size=size)
    ids, data = canvas.block_ids, canvas.block_data
    coords = np.argwhere(ids != canvas.air_block_id)
    if max_blocks is not None and len(coords) > max_blocks:
        raise ValueError(f"build has {len(coords)} blocks > max_blocks={max_blocks}")
    order = np.lexsort((coords[:, 0], coords[:, 2], coords[:, 1]))
    namer = _namer(_block_specs(text))
    lines: List[str] = []
    for x, y, z in coords[order].tolist():
        pair = (int(ids[x, y, z]), int(data[x, y, z]))
        nm = namer(pair)
        if nm is None:
            raise ValueError(f"no block name resolves to legacy pair {pair} at {(x, y, z)}")
        lines.append(f"{nm} {x} {y} {z}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# dsl
# ---------------------------------------------------------------------------
@register("actions", "dsl")
class DslActions:
    """WorldEdit-style command language (the legacy agentic DSL), optionally a verb subset.

    ``commands`` restricts the verbs per INSTANCE: a disallowed verb becomes an
    Issue and its line is commented out at lowering; the global ``dsl.COMMANDS``
    registry is never touched, so two recipes with different subsets can run in
    one process. With ``commands=None`` lowering is the identity, which keeps the
    ExecutionReport (and so repair prompts and cache keys) byte-identical to
    ``BuildAgent``.
    """

    name = "dsl"
    patchable = True
    OUTPUT_RULE = "Output only command lines."   # the legacy build-turn sentence

    def __init__(self, commands: Optional[Sequence[str] | str] = None) -> None:
        self.allowed: Optional[Tuple[str, ...]] = None
        if commands is not None:
            if isinstance(commands, str):
                commands = [c for c in re.split(r"[,\s]+", commands) if c]
            allowed: List[str] = []
            for c in commands:
                spec = lookup(str(c))
                if spec is None:
                    raise ValueError(f"dsl: unknown command {c!r} in commands="
                                     f"{_suggest(str(c), list(COMMANDS))}; known: {list(COMMANDS)}")
                if spec.name not in allowed:
                    allowed.append(spec.name)
            if not allowed:
                raise ValueError("dsl: commands= must name at least one command")
            # Keep registry order so the reference reads the same as the full one.
            self.allowed = tuple(n for n in COMMANDS if n in allowed)

    # -- text -------------------------------------------------------------
    def specs(self) -> List[CommandSpec]:
        return [COMMANDS[n] for n in (self.allowed or tuple(COMMANDS))]

    def reference(self) -> str:
        """The command reference; the legacy generator when unrestricted."""
        if self.allowed is None:
            return command_reference()
        lines: List[str] = []
        for spec in self.specs():
            lines.append(spec.help_block())
            if spec.aliases:
                lines.append(f"      aliases: {', '.join(spec.aliases)}")
        return "\n".join(lines)

    def grammar(self, ctx: PromptCtx) -> str:
        return f"Commands:\n{self.reference()}"

    def instructions(self, ctx: PromptCtx) -> str:
        # Same pieces as prompts.system_prompt (command reference, RULES, METHOD),
        # imported not copied. The palette list is NOT here: the `palette` context
        # module supplies it, and RULES' "palette below" then reads correctly.
        return (
            "You are a master Minecraft builder. You do not place blocks by hand: you write a\n"
            "short program in the BlockGen build language, and the program is executed to\n"
            "produce the structure.\n\n"
            f"{_canvas_text(ctx.size)}\n\n{self.grammar(ctx)}\n\n{RULES}\n\n{METHOD}"
            f"{_tail(ctx)}\n\n{self.OUTPUT_RULE}")

    def output_rule(self) -> str:
        return self.OUTPUT_RULE

    # -- examples / mock --------------------------------------------------
    def _uses_only_allowed(self, program: str) -> bool:
        if self.allowed is None:
            return True
        return all(c.spec.name in self.allowed for c in parse_program(program).calls)

    def examples(self, query: str, k: int) -> List[Tuple[str, str]]:
        if k <= 0:
            return []
        if self.allowed is None:   # exactly the legacy selection
            return [(ex.caption, ex.program.strip()) for ex in select_examples(query, k)]
        ranked = select_examples(query, len(EXAMPLES))
        return [(ex.caption, ex.program.strip()) for ex in ranked
                if self._uses_only_allowed(ex.program)][:k]

    def from_dsl(self, text: str) -> str:
        return text

    def mock_reply(self) -> str:
        if self._uses_only_allowed(MOCK_DSL):
            return MOCK_DSL
        # A subset without the house's verbs still gets a valid (smaller) reply.
        allowed = self.allowed or ()
        if "fill" in allowed:
            return "fill cobblestone 0 0 0 4 0 4\n"
        if "set" in allowed:
            return "".join(f"set cobblestone {x} 0 {z}\n" for x in range(3) for z in range(3))
        return COMMANDS[allowed[0]].example + "\n"

    # -- lowering ---------------------------------------------------------
    def lower(self, text: str) -> Lowered:
        text = text or ""
        n = len(text.splitlines())
        if self.allowed is None:
            return Lowered(source=text, program_text=text, issues=[], n_source_lines=n)
        out: List[str] = []
        issues: List[Issue] = []
        for i, raw in enumerate(text.splitlines(), start=1):
            body = raw.split("#", 1)[0].strip()
            spec = lookup(body.split()[0]) if body else None
            if spec is not None and spec.name not in self.allowed:
                issues.append(Issue(i, raw, f"command '{spec.name}' is not available here; "
                                            f"use only: {', '.join(self.allowed)}"))
                out.append(f"# disallowed: {raw.strip()}")   # keeps lines aligned
            else:
                out.append(raw)
        return Lowered(source=text, program_text="\n".join(out), issues=issues,
                       n_source_lines=n)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
# '<block[state]> x y z' with tolerated decoration: list bullets or numbering
# ("- ", "3. "), a colon after the name, commas or parentheses around the
# coordinates, a trailing '# comment'. The block token is an identifier with an
# optional bracketed state list (which may contain spaces).
_BLOCK_HEAD = re.compile(r"^(?:[-*+•]\s+|\d+[.)]\s+)?"
                         r"(?P<block>[A-Za-z_][\w:.\-]*(?:\[[^\]]*\])?)(?P<rest>.*)$")
_NUM = re.compile(r"^-?\d+(?:\.0*)?$")


@register("actions", "blocks", "block_lines")
class BlockActions:
    """One '<block[state]> x y z' line per voxel (BrickGPT-style, validated by blockstate).

    Lowers to ``set <block> x y z`` line-aligned with the source, so every issue
    (lowering or execution) names the model's own line. Unknown blocks and
    malformed lines are Issues and dropped; prose lines are ignored. With
    ``allow_states=False`` a bracketed state is stripped (with a warning) rather
    than failing the voxel.
    """

    name = "blocks"
    patchable = True
    OUTPUT_RULE = "Output only block lines, one `<block> x y z` per line."

    def __init__(self, allow_states: bool = True) -> None:
        self.allow_states = bool(allow_states)

    # -- text -------------------------------------------------------------
    def grammar(self, ctx: PromptCtx) -> str:
        state = ("\nA block may carry state in brackets, e.g. `oak_stairs[facing=north,half=top] 4 6 0`,\n"
                 "`oak_log[axis=x] 1 3 0`, `oak_slab[type=top] 2 4 2`."
                 if self.allow_states else "\nBlock states are not supported: write the plain block name.")
        return ("Block-line format: one placed block per line, written `<block> x y z`,\n"
                "e.g. `oak_planks 3 1 0` places oak planks at x=3, y=1, z=0. Every block of the\n"
                "build needs its own line (there are no region commands). A later line at the\n"
                "same coordinates overwrites an earlier one, and `air x y z` removes a block."
                + state)

    def instructions(self, ctx: PromptCtx) -> str:
        return (
            "You are a master Minecraft builder. You describe the build block by block, and\n"
            "the lines are placed exactly as written to produce the structure.\n\n"
            f"{_canvas_text(ctx.size)}\n\n{self.grammar(ctx)}\n\n"
            "Rules:\n"
            "- No prose, no explanation, no code fences. `#` starts a comment.\n"
            "- Coordinates are integers, y is up, origin (0,0,0) is the corner of the canvas.\n"
            "  Blocks outside the canvas are dropped.\n"
            "- Build ON the ground: the lowest solid part of the build should sit at y=0.\n"
            "- Use the block names from the palette below; unknown names waste the line.\n\n"
            "Method: lay the floor first so nothing floats, then raise closed walls layer by\n"
            "layer, leave gaps only for doors and windows, put the roof ON the wall tops, and\n"
            "add trim and details last. Keep the build one connected piece."
            f"{_tail(ctx)}\n\n{self.OUTPUT_RULE}")

    def output_rule(self) -> str:
        return self.OUTPUT_RULE

    # -- examples / mock --------------------------------------------------
    def _own_format(self, lines: str) -> str:
        """Block lines as this instance accepts them: with ``allow_states=False``
        states are removed, so examples and the mock reply never show (or lower
        with warnings about) a syntax the instructions say is unsupported."""
        if self.allow_states:
            return lines
        return re.sub(r"\[[^\]]*\]", "", lines)

    def from_dsl(self, text: str) -> str:
        return self._own_format(dsl_to_blocks(text, CONVERT_SIZE))

    def examples(self, query: str, k: int) -> List[Tuple[str, str]]:
        """DSL examples converted to block lines, only those of <= 400 blocks.

        Ranked by the same keyword retrieval as the DSL examples. Every stock
        example is 595-964 blocks, over the cap, so the pool also holds the
        206-block mock house: without it this space would never show the model
        its own format.
        """
        if k <= 0:
            return []
        out: List[Tuple[str, str]] = []
        pool = (*EXAMPLES, MOCK_EXAMPLE)
        for ex in select_examples(query, len(pool), pool=pool):
            conv = _blocks_example(ex)
            if conv is not None:
                out.append((ex.caption, self._own_format(conv)))
            if len(out) >= k:
                break
        return out

    def mock_reply(self) -> str:
        return self._own_format(_mock_blocks())

    # -- lowering ---------------------------------------------------------
    def lower_line(self, raw: str, line_no: int) -> Tuple[str, List[Issue]]:
        """One source line -> (lowered line, issues). Lowered '' means 'nothing'."""
        code, sep, _comment = raw.partition("#")
        stripped = code.strip()
        if not stripped:
            return (raw.strip() if sep else ""), []
        if _is_fence(raw.strip()):
            return "", []
        m = _BLOCK_HEAD.match(stripped)
        if m is None:
            return "", []                      # prose ("1. walls first")
        rest = re.sub(r"[(),;]", " ", m.group("rest")).split()
        if not any(ch.isdigit() for ch in m.group("rest")):
            return "", []                      # prose ("Here is the build:")
        spec = m.group("block")
        if "[" not in spec:
            spec = spec.rstrip(":.-")          # "stone: 1, 2, 3" (not the namespace colon)
        if len(rest) != 3 or not all(_NUM.match(t) for t in rest):
            # Only a line that is recognisably an attempted block line is an
            # error; "Layer 1 (y=0):" or "Here are 3 walls" is prose, and turning
            # prose into errors would feed the repair turn noise.
            if _resolve(spec) is not None:
                return "", [Issue(line_no, raw, "expected `<block> x y z` with three "
                                                "integer coordinates")]
            call, _ = parse_line(stripped, line_no)
            if call is not None:
                return "", [Issue(line_no, raw, f"'{call.spec.name}' is a region command; "
                                                "this format takes one `<block> x y z` per line")]
            return "", []
        try:
            x, y, z = (int(round(float(t))) for t in rest)
        except (ValueError, OverflowError):    # '9'*400 -> float inf
            return "", [Issue(line_no, raw, "coordinate is too large to be a block position")]
        issues: List[Issue] = []
        ref = parse_block(spec)
        if ref.props and not self.allow_states:
            issues.append(Issue(line_no, raw, f"block states are disabled; placed plain "
                                              f"'{ref.name}'", severity="warning"))
            spec = ref.name
        if _resolve(spec) is None:
            try:
                resolve_block(spec)
                msg = f"unknown block '{spec}'"
            except UnknownBlockError as exc:
                msg = str(exc)
            return "", issues + [Issue(line_no, raw, msg)]
        warn = state_warning(spec)
        if warn is not None:
            issues.append(Issue(line_no, raw, warn, severity="warning"))
        return f"set {_canonical_block(spec)} {x} {y} {z}", issues

    def lower(self, text: str) -> Lowered:
        text = text or ""
        out: List[str] = []
        issues: List[Issue] = []
        lines = text.splitlines()
        for i, raw in enumerate(lines, start=1):
            low, iss = self.lower_line(raw, i)
            out.append(low)
            issues.extend(iss)
        return Lowered(source=text, program_text="\n".join(out), issues=issues,
                       n_source_lines=len(lines))


@functools.lru_cache(maxsize=1)
def _mock_blocks() -> str:
    return dsl_to_blocks(MOCK_DSL, CONVERT_SIZE)


@functools.lru_cache(maxsize=None)
def _blocks_example(ex: Example) -> Optional[str]:
    try:
        return dsl_to_blocks(ex.program, CONVERT_SIZE, max_blocks=MAX_BLOCK_EXAMPLE)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# mixed
# ---------------------------------------------------------------------------
#: Fence tag -> sub-space. Unknown tags are treated as dsl with a warning.
FENCE_TAGS: Dict[str, str] = {
    "python": "python", "py": "python", "python3": "python",
    "dsl": "dsl", "blockgen": "dsl", "worldedit": "dsl", "commands": "dsl",
    "": "dsl", "text": "dsl", "txt": "dsl", "plaintext": "dsl",
    "blocks": "blocks", "block": "blocks", "voxels": "blocks",
}
MIXED_PARTS: Tuple[str, ...] = ("python", "dsl", "blocks")


def _make_sub(part: str, params: Mapping[str, Any]) -> Any:
    if part == "dsl":
        return DslActions(**params)
    if part == "blocks":
        return BlockActions(**params)
    if part == "python":
        from blockgen.agentic.pyapi import PythonActions   # lazy: pyapi imports us
        return PythonActions(**params)
    raise ValueError(f"mixed: unknown part {part!r}{_suggest(part, MIXED_PARTS)}; "
                     f"parts are {MIXED_PARTS}")


@register("actions", "mixed")
class MixedActions:
    """Fenced ```python / ```dsl / ```blocks sections, each lowered by its sub-space, in order.

    Use it to let a model do massing in one language and touch-ups in another
    (``parts: [python, blocks]``). Sections are concatenated in source order;
    lowering issues are shifted to whole-reply line numbers. Unfenced text is
    dsl; when ``dsl`` is not an enabled part, unfenced prose is ignored but
    unfenced commands are reported. ``sub`` passes params to a part, e.g.
    ``sub: {python: {timeout_s: 5}}``.
    """

    name = "mixed"
    patchable = True
    OUTPUT_RULE = "Output only fenced ```python, ```dsl or ```blocks sections."

    def __init__(self, parts: Sequence[str] | str = MIXED_PARTS,
                 sub: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 size: Optional[Sequence[int]] = None) -> None:
        if isinstance(parts, str):
            parts = [p for p in re.split(r"[,+\s]+", parts) if p]
        self.parts: Tuple[str, ...] = tuple(dict.fromkeys(str(p) for p in parts))
        if not self.parts:
            raise ValueError("mixed: parts must name at least one of " + str(MIXED_PARTS))
        sub = dict(sub or {})
        for k in sub:
            if k not in self.parts:
                raise ValueError(f"mixed: sub params for {k!r}, which is not in parts {self.parts}")
        if size is not None and "python" in self.parts:   # the canvas the python SIZE names
            sub["python"] = {"size": size, **sub.get("python", {})}
        self.spaces: Dict[str, Any] = {p: _make_sub(p, sub.get(p, {})) for p in self.parts}

    # -- text -------------------------------------------------------------
    def instructions(self, ctx: PromptCtx) -> str:
        # Each sub-space contributes its GRAMMAR only (never its rules or output
        # rule: mock:auto identifies the space by the output rule it finds).
        blocks = []
        for p in self.parts:
            blocks.append(f"## ```{p} sections\n{self.spaces[p].grammar(ctx)}")
        sections = "\n\n".join(blocks)
        return (
            "You are a master Minecraft builder. You write the build in fenced sections, each\n"
            f"in one of these languages: {', '.join(self.parts)}. The sections run in order on\n"
            "the same canvas, so use each language for what it does best (e.g. code or\n"
            "region commands for massing, single blocks for touch-ups).\n\n"
            f"{_canvas_text(ctx.size)}\n\n{sections}\n\n"
            "Rules:\n"
            "- Start every section with a fence naming its language (```" + self.parts[0]
            + ") and close it with ```.\n"
            "- Text outside fences is ignored. `#` starts a comment in every language.\n"
            + ("- Each ```python section runs on its own: variables and functions do not carry\n"
               "  over to a later section.\n" if "python" in self.parts else "")
            + "- Coordinates are integers, inclusive, y is up, origin (0,0,0) is the canvas\n"
            "  corner. Build ON the ground (lowest solid part at y=0).\n"
            "- Use the block names from the palette below.\n\n"
            f"{METHOD}{_tail(ctx)}\n\n{self.OUTPUT_RULE}")

    def output_rule(self) -> str:
        return self.OUTPUT_RULE

    # -- examples / mock --------------------------------------------------
    def examples(self, query: str, k: int) -> List[Tuple[str, str]]:
        """Each DSL example as one section in the first part that can carry it."""
        if k <= 0:
            return []
        out: List[Tuple[str, str]] = []
        pool = (*EXAMPLES, MOCK_EXAMPLE)    # the hut: a blocks-only mix still gets one
        for ex in select_examples(query, len(pool), pool=pool):
            body = None
            for p in self.parts:
                try:
                    if p == "blocks":
                        conv = _blocks_example(ex)
                        body = self.spaces[p]._own_format(conv) if conv else None
                    else:
                        body = self.spaces[p].from_dsl(ex.program)
                except ValueError:
                    body = None
                if body:
                    out.append((ex.caption, f"```{p}\n{body.strip()}\n```"))
                    break
            if len(out) >= k:
                break
        return out

    def mock_reply(self) -> str:
        first = self.parts[0]
        second = self.parts[1] if len(self.parts) > 1 else first
        massing = self.spaces[first].from_dsl(MOCK_MASSING + "\n").strip()
        detail = self.spaces[second].from_dsl(MOCK_DETAIL + "\n").strip()
        return (f"Massing:\n```{first}\n{massing}\n```\n\n"
                f"Details:\n```{second}\n{detail}\n```\n")

    def from_dsl(self, text: str) -> str:
        p = self.parts[0]
        return f"```{p}\n{self.spaces[p].from_dsl(text).strip()}\n```\n"

    # -- lowering ---------------------------------------------------------
    def sections(self, text: str) -> List["Section"]:
        """Split a reply into sections, in source order (fenced or not)."""
        out: List[Section] = []
        cur: List[str] = []
        start, fence_line, tag, fenced = 1, 0, "", False
        for i, raw in enumerate(text.splitlines(), start=1):
            s = raw.strip()
            if not _is_fence(s):
                cur.append(raw)
                continue
            info = s.lstrip("`~").strip().lower()
            if fenced:
                out.append(Section(FENCE_TAGS.get(tag, "dsl"), tag, start, fence_line, True, cur))
                fenced, start, cur = False, i + 1, []
                if not info:
                    continue
                # A closing fence never carries a language tag, so "```blocks" while
                # a section is open means the model forgot the closing fence: close
                # the open section AND open the new one (otherwise its body would
                # be read as unfenced text and silently dropped).
            elif any(x.strip() for x in cur):
                out.append(Section("dsl", "", start, 0, False, cur))
            tag = info.split()[0] if info else ""
            fenced, fence_line, start, cur = True, i, i + 1, []
        if any(x.strip() for x in cur):   # trailing text, or a fence never closed
            out.append(Section(FENCE_TAGS.get(tag, "dsl") if fenced else "dsl",
                               tag if fenced else "", start, fence_line if fenced else 0,
                               fenced, cur))
        return out

    def lower(self, text: str) -> Lowered:
        text = text or ""
        src_lines = text.splitlines()
        programs: List[str] = []
        issues: List[Issue] = []
        for sec in self.sections(text):
            anchor = sec.fence_line or sec.first
            anchor_text = src_lines[anchor - 1] if 0 < anchor <= len(src_lines) else ""
            if sec.fenced and sec.tag not in FENCE_TAGS:
                issues.append(Issue(anchor, anchor_text, f"unknown section language "
                                    f"'{sec.tag}'; treated as dsl", severity="warning"))
            if sec.part not in self.spaces:
                if not sec.fenced:
                    # Prose between fences is normal; commands there are a mistake.
                    calls = parse_program(sec.body).calls
                    if calls:
                        ln = calls[0].line_no + sec.first - 1
                        issues.append(Issue(ln, src_lines[ln - 1], "commands outside a fenced "
                                            f"section are ignored; enabled: {', '.join(self.parts)}"))
                    continue
                issues.append(Issue(anchor, anchor_text, f"```{sec.part} sections are not "
                                    f"enabled here; use: {', '.join(self.parts)}"))
                continue
            low = self.spaces[sec.part].lower(sec.body)
            if not sec.fenced and not low.issues and not parse_program(low.program_text).calls:
                continue    # unfenced prose ("Massing:") contributes nothing
            for iss in low.issues:
                line = iss.line_no + sec.first - 1 if iss.line_no > 0 else anchor
                issues.append(Issue(line, iss.text, f"[{sec.part}] {iss.message}", iss.severity))
            body = low.program_text.rstrip()
            if sec.part == "python":
                # python's '# L<n>' op annotations -> whole-reply lines
                body = _L_TAG.sub(lambda m: f"# L{int(m.group(1)) + sec.first - 1}", body)
            else:
                # dsl/blocks lower line-aligned WITHIN the section, but the section
                # header and offset break alignment with the reply; tag each line
                # so remap_issues maps execution issues back to the reply too.
                body = "\n".join(_tag_line(ln, k + sec.first)
                                 for k, ln in enumerate(body.split("\n")))
            programs.append(f"# --- {sec.part} (source lines {sec.first}-{sec.last}) ---\n{body}")
        return Lowered(source=text, program_text="\n".join(programs) + ("\n" if programs else ""),
                       issues=issues, n_source_lines=len(src_lines))


@dataclass
class Section:
    """One section of a mixed reply. ``first`` is the 1-based source line of its
    first body line, so a sub-space issue on body line k is source line
    ``first + k - 1``; ``fence_line`` is the opening fence (0 when unfenced)."""

    part: str
    tag: str
    first: int
    fence_line: int
    fenced: bool
    lines: List[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines)

    @property
    def last(self) -> int:
        return self.first + len(self.lines) - 1


__all__ = ["BlockActions", "CONVERT_SIZE", "DslActions", "FENCE_TAGS", "MAX_BLOCK_EXAMPLE",
           "MIXED_PARTS", "MOCK_DETAIL", "MOCK_DSL", "MOCK_EXAMPLE", "MOCK_MASSING", "MixedActions", "Section",
           "PY_ALIASES", "PY_NAMES", "canonical_call", "dsl_to_blocks", "dsl_to_python",
           "py_name", "py_names", "python_call", "remap_issues", "render_dsl_value",
           "dedupe_issues", "state_warning"]
