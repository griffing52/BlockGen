"""The ``python`` action space: the model writes a Python program against a build API.

Why Python at all: a DSL has no loops, variables or arithmetic, so "a row of
twelve windows every three blocks" or "a spiral staircase" costs the model one
line per repetition and one arithmetic slip per line. Code-as-policy work
(CaP, CaP-X, VoxelCodeBench) finds frontier models are far more fluent in
Python than in any bespoke grammar. The API is generated from the DSL registry
(``dsl.COMMANDS``): ``fill(block, x1, y1, z1, x2, y2, z2)``, ``walls(...)``,
``gable(...)``, ... with ``set`` renamed ``place`` (``set`` must stay the
Python type; ``block``/``set_block`` are aliases). Each API call appends ONE
canonical DSL line, so the program lowers to the same IR as every other action
space and replays through the same executor.

Execution model
---------------
1. The parent parses the code and checks it against an AST allow-list (no
   import, no names or attributes starting with ``_``, no ``open``/``exec``/
   ``eval``/``getattr``..., no ``global``/``with``/``class``, no decorators, no
   calls in default values, no ``str.format``, no frame/traceback attributes).
   Violations are Issues with line numbers and nothing runs.
2. The code runs in a CHILD process: ``python -I -c RUNNER`` with an empty
   environment, a fresh empty working directory, the payload on stdin and the
   op log as JSON on stdout. The runner is stdlib-only and sets resource limits
   on itself before any user code runs: CPU seconds, address space, file size 0
   (no file can be written), a small open-file cap, no child processes. The
   parent also enforces a wall-clock ``timeout_s``.
3. The child exposes only a whitelisted ``__builtins__`` (no ``open``,
   ``__import__``, ``getattr``, ...), the API functions, ``math`` and a seeded
   ``random`` facade. ``print`` is a no-op (stdout is the result channel).
4. ``max_ops`` caps the op log; past it every API call raises, and a program
   that swallows that exception still ends truncated and reported.
5. The child caps every argument: integers at ``MAX_INT`` in magnitude, strings
   at ``MAX_STR`` ASCII characters, and the whole op log at ``max_output_mb``.
   The parent reads at most about twice that from the child's stdout.
6. The parent validates every op: block names (``blockstate``), ignored block
   states (a warning), coordinates within one canvas width of the canvas, and
   sizes (radius, height, count, ...) at most one canvas width. It also adds up
   an estimate of the voxel work each op will cost the executor and drops the
   ops past ``max_work``. Issues map back to the source line that made the call
   (``# L<n>`` annotations in the lowered text carry the same mapping into
   execution warnings), and one bad call in a loop is one issue with a count.

Threat model (honest version)
-----------------------------
This protects against what an LLM plausibly writes: an ``import os``, an
attempt to read a file "for reference", an infinite loop, a 10^9-iteration
nest, a memory blow-up. Those are rejected statically or bounded by the
timeout and rlimits. The subprocess bounds only the program's OWN execution:
the ops it emits run later in the PARENT (``dsl.run_program``), where neither
the timeout nor the rlimits apply. Step 6 (argument bounds and the work budget)
is what keeps a computed ``line(..., 10**8, 0, 0)`` or a loop of 5000 big
spheres from stalling the parent or exhausting its memory. It is NOT a security boundary against a determined
adversary: CPython-level sandboxes have a long history of escapes, and the
child runs as the same user with the same filesystem and network visibility
(no namespaces, no seccomp; the rlimits make writing files and spawning
processes fail, but reading a world-readable file or opening a socket after a
successful escape is possible). For untrusted prompts or a public deployment,
run the child inside a container / bubblewrap / ``unshare -n`` jail; the
subprocess boundary is designed so that is a change to ``argv`` only.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.actions import (CONVERT_SIZE, _canvas_text, _tail, dedupe_issues,
                                      dsl_to_python, py_names, render_dsl_value,
                                      state_warning)
from blockgen.agentic.blockstate import UnknownBlockError, resolve_block
from blockgen.agentic.core import Lowered, PromptCtx, register
from blockgen.agentic.dsl import COMMANDS, Call, CommandSpec, Issue, Param, parse_line
from blockgen.agentic.examples import select_examples
from blockgen.agentic.prompts import METHOD

# ---------------------------------------------------------------------------
# Static checks (parent side)
# ---------------------------------------------------------------------------
_ALLOWED_NODES: Tuple[type, ...] = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While,
    ast.If, ast.Break, ast.Continue, ast.Pass, ast.FunctionDef, ast.Return, ast.Lambda,
    ast.arguments, ast.arg, ast.keyword, ast.Call, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.Delete, ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Starred,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.NamedExpr, ast.JoinedStr, ast.FormattedValue, ast.Assert, ast.Raise,
    ast.Try, ast.ExceptHandler, ast.Nonlocal, ast.Yield, ast.YieldFrom,
    ast.alias,   # only reachable under Import, which is itself rejected (one issue, not two)
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context,
)

#: Builtins/names that reach outside the program (or reflect into the runner).
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "breakpoint", "input", "help",
    "memoryview", "type", "object", "super", "classmethod", "staticmethod",
    "property", "exit", "quit", "dir", "id", "sys", "os", "builtins",
})
#: Attributes that are not underscore-prefixed but still escape: string
#: formatting can walk attributes ("{0.__class__}"), frames reach globals.
BANNED_ATTRS = frozenset({
    "format", "format_map", "mro", "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame",
    "cr_code", "cr_await", "ag_frame", "ag_code", "f_globals", "f_locals", "f_builtins",
    "f_back", "f_code", "tb_frame", "tb_next", "co_code", "func_globals", "func_code",
})

_NODE_LABEL = {ast.Import: "import", ast.ImportFrom: "import", ast.Global: "global",
               ast.With: "with", ast.AsyncWith: "with", ast.ClassDef: "class",
               ast.AsyncFunctionDef: "async def", ast.Await: "await"}


def _has_call(node: Optional[ast.AST]) -> bool:
    return node is not None and any(isinstance(n, ast.Call) for n in ast.walk(node))


def check_code(code: str) -> List[Issue]:
    """Static allow-list check. Returns Issues (empty = OK to run).

    Defense in depth only: the child's restricted builtins and rlimits apply
    regardless, but a static rejection gives the model a precise, line-numbered
    reason instead of a NameError from deep inside the runner.
    """
    lines = code.splitlines()

    def issue(node: ast.AST, msg: str) -> Issue:
        ln = int(getattr(node, "lineno", 0) or 0)
        return Issue(ln, lines[ln - 1] if 0 < ln <= len(lines) else "", msg)

    try:
        tree = ast.parse(code, filename="<build>", mode="exec")
    except SyntaxError as exc:
        ln = int(exc.lineno or 0)
        return [Issue(ln, lines[ln - 1] if 0 < ln <= len(lines) else "",
                      f"SyntaxError: {exc.msg}")]
    except (ValueError, RecursionError, MemoryError) as exc:   # NUL bytes, deep nesting
        return [Issue(0, "", f"cannot parse program: {type(exc).__name__}: {exc}")]

    out: List[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            label = _NODE_LABEL.get(type(node), type(node).__name__)
            out.append(issue(node, f"'{label}' is not allowed in build programs "
                                   f"(no imports, classes, with, global; the API is pre-loaded)"))
            continue
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BANNED_NAMES):
            out.append(issue(node, f"name '{node.id}' is not available in build programs"))
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_")
                                                  or node.attr in BANNED_ATTRS):
            out.append(issue(node, f"attribute '.{node.attr}' is not allowed "
                                   f"(use f-strings instead of .format)"
                                   if node.attr in ("format", "format_map") else
                                   f"attribute '.{node.attr}' is not allowed"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            out.append(issue(node, "string constants may not contain '__'"))
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            args = node.args
            if any(_has_call(d) for d in (*args.defaults, *args.kw_defaults)):
                out.append(issue(node, "default argument values may not call functions"))
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    out.append(issue(node, "decorators are not allowed"))
                anns = [node.returns] + [a.annotation for a in
                                         (*args.posonlyargs, *args.args, *args.kwonlyargs)]
                if any(_has_call(a) for a in anns):
                    out.append(issue(node, "annotations may not call functions"))
        elif isinstance(node, ast.AnnAssign) and _has_call(node.annotation):
            out.append(issue(node, "annotations may not call functions"))
    out.sort(key=lambda i: i.line_no)
    return out


# ---------------------------------------------------------------------------
# The child runner (stdlib only; runs as `python -I -c RUNNER`)
# ---------------------------------------------------------------------------
RUNNER = r'''
import json, math, sys, traceback, types
import random as _random_mod

def _set_limits(cfg):
    try:
        import resource
    except ImportError:          # non-POSIX: timeout is the only bound
        return
    def lim(name, value):
        r = getattr(resource, name, None)
        if r is None or value is None:
            return
        try:
            resource.setrlimit(r, (value, value))
        except (ValueError, OSError):
            pass
    lim("RLIMIT_CPU", cfg["cpu_s"])
    lim("RLIMIT_AS", cfg["mem_bytes"])
    lim("RLIMIT_FSIZE", 0)
    lim("RLIMIT_CORE", 0)
    lim("RLIMIT_NPROC", 0)
    lim("RLIMIT_NOFILE", 8)

payload = json.loads(sys.stdin.read())
out = sys.stdout
ops = []
state = {"capped": False, "bytes": 0}
MAX_OPS = int(payload["max_ops"])
MAX_INT = int(payload["max_int"])
MAX_STR = int(payload["max_str"])
MAX_BYTES = int(payload["max_bytes"])

class OpLimit(BaseException):
    pass

def _user_line():
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename != "<build>":
        f = f.f_back
    return f.f_lineno if f is not None else 0

def _coerce(fname, p, v):
    kind = p["kind"]
    if kind == "int":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("%s(): '%s' must be a number, got %s" % (fname, p["name"], type(v).__name__))
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("%s(): '%s' must be finite" % (fname, p["name"]))
        if abs(v) > MAX_INT:
            raise ValueError("%s(): '%s' = %s is far outside any canvas (limit +-%d)"
                             % (fname, p["name"], ("%.3g" % v) if abs(v) < 1e300 else "huge",
                                MAX_INT))
        return str(int(round(v)))
    if kind == "bool":
        if isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "1", "yes", "y", "on", "hollow"):
                return "true"
            if low in ("false", "0", "no", "n", "off", "solid"):
                return "false"
            raise ValueError("%s(): '%s' must be True/False" % (fname, p["name"]))
        return "true" if v else "false"
    if not isinstance(v, str):
        raise TypeError("%s(): '%s' must be a string, got %s" % (fname, p["name"], type(v).__name__))
    if len(v) > MAX_STR:
        raise ValueError("%s(): '%s' is %d characters long (limit %d)"
                         % (fname, p["name"], len(v), MAX_STR))
    if not v.isascii():
        raise ValueError("%s(): '%s' must be plain ASCII" % (fname, p["name"]))
    s = "".join(v.split())
    if not s:
        raise ValueError("%s(): '%s' is empty" % (fname, p["name"]))
    if kind == "choice":
        s = s.lower()
        if p["choices"] and s not in p["choices"]:
            raise ValueError("%s(): '%s' must be one of %s, got %r"
                             % (fname, p["name"], "|".join(p["choices"]), v))
    return s

def _make(spec, fname):
    params = spec["params"]
    names = [p["name"] for p in params]
    def api(*args, **kwargs):
        if len(args) > len(params):
            raise TypeError("%s() takes at most %d arguments (%d given); usage: %s"
                            % (fname, len(params), len(args), spec["usage"]))
        vals = dict(zip(names, args))
        for k, v in kwargs.items():
            if k not in names:
                raise TypeError("%s() got an unexpected keyword argument '%s'; usage: %s"
                                % (fname, k, spec["usage"]))
            if k in vals:
                raise TypeError("%s() got multiple values for argument '%s'" % (fname, k))
            vals[k] = v
        parts = [spec["name"]]
        for p in params:
            if p["name"] not in vals:
                if p["required"]:
                    raise TypeError("%s() missing required argument '%s'; usage: %s"
                                    % (fname, p["name"], spec["usage"]))
                continue
            s = _coerce(fname, p, vals[p["name"]])
            if p["required"]:
                parts.append(s)
            elif s != p["default"]:
                parts.append("%s=%s" % (p["name"], s))
        if len(ops) >= MAX_OPS:
            state["capped"] = True
            raise OpLimit("op cap of %d calls reached" % MAX_OPS)
        text = " ".join(parts)
        state["bytes"] += len(text) + 16          # + JSON framing and the line number
        if state["bytes"] > MAX_BYTES:
            state["capped"] = True
            raise OpLimit("output cap of %d bytes reached" % MAX_BYTES)
        ops.append((text, _user_line()))
    api.__name__ = fname
    return api

ns = {}
for spec in payload["specs"]:
    for fname in spec["py_names"]:
        ns[fname] = _make(spec, fname)

_rng = _random_mod.Random(payload["seed"])
ns["random"] = types.SimpleNamespace(
    random=_rng.random, randint=_rng.randint, uniform=_rng.uniform, choice=_rng.choice,
    shuffle=_rng.shuffle, sample=_rng.sample, randrange=_rng.randrange)
ns["math"] = math
ns["SIZE"] = tuple(payload["size"])
import builtins as _b
SAFE = {k: getattr(_b, k) for k in payload["builtins"] if hasattr(_b, k)}
SAFE["print"] = lambda *a, **k: None
ns["__builtins__"] = SAFE
ns["__name__"] = "build"

error = None
try:
    code = compile(payload["code"], "<build>", "exec")
    _set_limits(payload)
    exec(code, ns)
except BaseException as exc:      # incl. OpLimit, MemoryError, RecursionError
    line = 0
    for fr, ln in traceback.walk_tb(exc.__traceback__):
        if fr.f_code.co_filename == "<build>":
            line = ln
    error = {"line": line, "type": type(exc).__name__, "message": str(exc)[:500]}
ops = ops[:MAX_OPS]
json.dump({"ops": ops, "error": error, "capped": state["capped"]}, out)
out.flush()
'''

#: Largest integer argument the child accepts (bigger is never a block position,
#: and ``float(10**400)`` would overflow the parent's parser).
MAX_INT = 10 ** 6
#: Longest string argument (block specs with states are ~40 characters).
MAX_STR = 200

#: Builtins the child exposes. Pure functions and types; nothing that does I/O,
#: introspects, or imports.
SAFE_BUILTINS: Tuple[str, ...] = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "int", "isinstance", "iter", "len", "list", "map",
    "max", "min", "next", "ord", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ArithmeticError", "AssertionError", "IndexError", "KeyError",
    "LookupError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError",
)


def _py_signature(spec: CommandSpec, name: Optional[str] = None) -> str:
    parts = []
    for p in spec.params:
        if p.required:
            parts.append(p.name)
        elif p.kind == "bool":
            parts.append(f"{p.name}={bool(p.default)}")
        elif p.kind == "int":
            parts.append(f"{p.name}={int(p.default)}")
        else:
            parts.append(f"{p.name}={str(p.default)!r}")
    return f"{name or py_names(spec)[0]}({', '.join(parts)})"


def _spec_payload(spec: CommandSpec) -> Dict[str, Any]:
    return {
        "name": spec.name, "py_names": py_names(spec), "usage": _py_signature(spec),
        "params": [{"name": p.name, "kind": p.kind, "required": p.required,
                    "default": None if p.required else render_dsl_value(p, p.default),
                    "choices": list(p.choices)} for p in spec.params],
    }


def api_reference() -> str:
    """The Python API, generated from ``dsl.COMMANDS`` like the DSL reference."""
    lines = []
    for spec in COMMANDS.values():
        example = dsl_to_python(spec.example).strip()
        lines.append(f"  {_py_signature(spec)}\n      {spec.summary}\n      e.g. {example}")
        extra = py_names(spec)[1:]
        if extra:
            lines.append(f"      aliases: {', '.join(extra)}")
    return "\n".join(lines)


@dataclass
class RunResult:
    """What the child reported: ``ops`` are (canonical DSL line, source line)."""

    ops: List[Tuple[str, int]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    capped: bool = False
    timed_out: bool = False
    crashed: str = ""


# ---------------------------------------------------------------------------
# The action space
# ---------------------------------------------------------------------------
@register("actions", "python", "pyapi")
class PythonActions:
    """Sandboxed Python calling fill/walls/gable/... (child process, timeout, op cap).

    ``timeout_s`` bounds wall-clock time, ``max_ops`` the number of API calls,
    ``mem_mb`` the child's address space, ``max_output_mb`` the op log it sends
    back, and ``max_work`` the estimated voxel writes the emitted ops cost the
    PARENT's executor (the child's limits do not cover that). ``annotate`` suffixes every lowered op
    with ``# L<n>`` (its source line) so execution warnings point back at the
    code. ``seed`` seeds the ``random`` facade, so a program is deterministic.

    ``size`` is the canvas the program's ``SIZE`` constant names. ``lower(text)``
    has no canvas argument (the ActionSpace protocol), so when ``size`` is not
    given the space adopts the ``ctx.size`` its instructions were last written
    for: the ``SIZE`` the model was told is then the ``SIZE`` it runs with (a
    model centring on ``SIZE[0] // 2`` would otherwise build off a 32^3 canvas).
    """

    name = "python"
    patchable = True
    OUTPUT_RULE = "Output only the Python program, in one ```python block."

    def __init__(self, timeout_s: float = 10, max_ops: int = 200000, mem_mb: int = 1024,
                 seed: int = 0, annotate: bool = True,
                 size: Optional[Sequence[int]] = None, max_work: int = 2_000_000,
                 max_output_mb: float = 32) -> None:
        if timeout_s <= 0 or max_ops <= 0 or mem_mb <= 0 or max_work <= 0 or max_output_mb <= 0:
            raise ValueError("python: timeout_s, max_ops, mem_mb, max_work and max_output_mb "
                             "must be positive")
        self.timeout_s = float(timeout_s)
        self.max_ops = int(max_ops)
        self.mem_mb = int(mem_mb)
        self.max_work = int(max_work)
        self.max_output_bytes = int(max_output_mb * 1024 * 1024)
        self.seed = int(seed)
        self.annotate = bool(annotate)
        self.size: Optional[Tuple[int, int, int]] = (
            tuple(int(v) for v in size) if size is not None else None)  # type: ignore[assignment]
        self._ctx_size: Optional[Tuple[int, int, int]] = None

    # -- text -------------------------------------------------------------
    def canvas_size(self) -> Tuple[int, int, int]:
        """The SIZE a lowered program runs with (explicit > last prompt > CONVERT_SIZE)."""
        return self.size or self._ctx_size or CONVERT_SIZE  # type: ignore[return-value]

    def grammar(self, ctx: PromptCtx) -> str:
        if self.size is None:
            self._ctx_size = tuple(int(v) for v in ctx.size)  # type: ignore[assignment]
        sx, sy, sz = self.size or ctx.size
        return (
            "Python build API (already loaded; every call places blocks immediately):\n"
            f"{api_reference()}\n\n"
            "Also available: `math`, `random` (seeded: random.randint/choice/uniform/...),\n"
            f"`SIZE` = ({sx}, {sy}, {sz}), and plain Python: variables, arithmetic, for/while,\n"
            "if, functions (def/lambda), lists, dicts, comprehensions, f-strings.")

    def instructions(self, ctx: PromptCtx) -> str:
        return (
            "You are a master Minecraft builder. You do not place blocks by hand: you write a\n"
            "short Python program that calls a build API, and the program is executed to\n"
            "produce the structure. Use loops and variables for anything repeated (window\n"
            "rows, floors, fences, columns) instead of writing each call out.\n\n"
            f"{_canvas_text(ctx.size)}\n\n{self.grammar(ctx)}\n\n"
            "Rules:\n"
            "- No imports, no files, no classes, no `with`/`global`, no names starting with `_`;\n"
            "  `print` does nothing. Anything else fails the program at that line.\n"
            "- Coordinates are integers, INCLUSIVE at both ends, y is up, origin (0,0,0) is the\n"
            "  corner of the canvas. Writes outside the canvas are dropped.\n"
            "- Build ON the ground: the lowest solid part of the build should sit at y=0.\n"
            "- Block names are strings from the palette below, e.g. \"oak_planks\" or\n"
            "  \"oak_stairs[facing=north]\"; unknown names fail that call.\n"
            f"- At most {self.max_ops} API calls and {self.timeout_s:g} seconds; prefer region calls\n"
            "  (`fill`, `walls`, `box`, `gable`, `stack`) over single `place` calls.\n\n"
            f"{METHOD}{_tail(ctx)}\n\n{self.OUTPUT_RULE}")

    def output_rule(self) -> str:
        return self.OUTPUT_RULE

    # -- examples / mock --------------------------------------------------
    def from_dsl(self, text: str) -> str:
        return dsl_to_python(text)

    def examples(self, query: str, k: int) -> List[Tuple[str, str]]:
        """The DSL examples, converted line-wise to API calls (``dsl_to_python``)."""
        return [(ex.caption, dsl_to_python(ex.program).strip())
                for ex in select_examples(query, k)]

    def mock_reply(self) -> str:
        return MOCK_PYTHON

    # -- execution --------------------------------------------------------
    def run(self, code: str, size: Optional[Sequence[int]] = None) -> RunResult:
        """Execute already-checked code in the sandboxed child; never raises."""
        size = self.canvas_size() if size is None else size
        payload = json.dumps({
            "code": code, "specs": [_spec_payload(s) for s in COMMANDS.values()],
            "max_ops": self.max_ops, "seed": self.seed, "size": list(size),
            "builtins": list(SAFE_BUILTINS),
            "cpu_s": int(math.ceil(self.timeout_s)) + 1,
            "mem_bytes": self.mem_mb * 1024 * 1024,
            "max_int": MAX_INT, "max_str": MAX_STR, "max_bytes": self.max_output_bytes,
        })
        # The child polices its own output size; this is the backstop, so a
        # broken or escaped child cannot make the parent buffer unbounded text.
        limit = 2 * self.max_output_bytes + (1 << 20)
        with tempfile.TemporaryDirectory(prefix="blockgen-pyapi-") as td:
            try:
                code_rc, out, err, timed_out, overflow = _bounded_run(
                    [sys.executable, "-I", "-S", "-c", RUNNER], payload.encode("utf-8"),
                    cwd=td, timeout=self.timeout_s, max_stdout=limit)
            except OSError as exc:
                return RunResult(crashed=f"could not start the sandbox: {exc}")
        if timed_out:
            return RunResult(timed_out=True)
        if overflow:
            return RunResult(crashed=f"sandbox output exceeded {limit} bytes")
        if code_rc != 0:
            why = (f"killed by signal {-code_rc}" if code_rc < 0 else f"exit code {code_rc}")
            tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            return RunResult(crashed=f"{why} (CPU or memory limit?) {tail[0][:200]}".strip())
        try:
            blob = json.loads(out.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return RunResult(crashed="sandbox returned no result")
        return RunResult(ops=[(str(t), int(ln)) for t, ln in blob.get("ops", [])],
                         error=blob.get("error"), capped=bool(blob.get("capped")))

    def lower(self, text: str) -> Lowered:
        text = text or ""
        code = extract_code(text)
        lines = code.splitlines()
        n = len(text.splitlines())

        def at(ln: int) -> str:
            return lines[ln - 1] if 0 < ln <= len(lines) else ""

        issues = check_code(code)
        if issues:
            return Lowered(source=text, program_text="", issues=issues, n_source_lines=n)
        size = tuple(int(v) for v in self.canvas_size())
        res = self.run(code, size)
        out: List[str] = []
        work = 0
        for op, ln in res.ops:
            call, bad, warn = _validate_op(op, ln, size)   # type: ignore[arg-type]
            if bad is not None:
                issues.append(Issue(ln, at(ln), bad))
                continue
            work += _op_work(call, size)                   # type: ignore[arg-type]
            if work > self.max_work:
                issues.append(Issue(ln, at(ln), (
                    f"the program asks for more than {self.max_work} voxel writes "
                    f"(oversized shapes or too many calls); this call and every later "
                    f"one were dropped")))
                break
            if warn is not None:
                issues.append(Issue(ln, at(ln), warn, severity="warning"))
            out.append(f"{op}  # L{ln}" if self.annotate and ln else op)
        issues = dedupe_issues(issues)
        if res.timed_out:
            issues.append(Issue(0, "", f"program timed out after {self.timeout_s:g}s "
                                       f"(an endless loop?); nothing was built"))
        if res.crashed:
            issues.append(Issue(0, "", f"program crashed: {res.crashed}; nothing was built"))
        err = res.error
        if res.capped or (err and err.get("type") == "OpLimit"):
            ln = int((err or {}).get("line") or 0)
            issues.append(Issue(ln, at(ln), f"op cap of {self.max_ops} API calls reached; "
                                            f"the build was truncated there"))
        elif err:
            ln = int(err.get("line") or 0)
            issues.append(Issue(ln, at(ln), f"{err.get('type')}: {err.get('message')} "
                                            f"(the program stopped here; earlier calls were kept)"))
        return Lowered(source=text, program_text="\n".join(out) + ("\n" if out else ""),
                       issues=issues, n_source_lines=n)


def extract_code(text: str) -> str:
    """The code to run, LINE-ALIGNED with ``text``.

    With fences, every line outside a fenced block (prose, the fence lines
    themselves) becomes empty, so line k of the code is line k of the reply and
    issue line numbers point at what the model wrote. Without fences, the text
    is the code.
    """
    lines = text.splitlines()
    if not any(l.strip().startswith(("```", "~~~")) for l in lines):
        return text
    out: List[str] = []
    inside = False
    for raw in lines:
        if raw.strip().startswith(("```", "~~~")):
            inside = not inside
            out.append("")
        else:
            out.append(raw if inside else "")
    return "\n".join(out)


_BLOCK_PARAMS: Dict[str, Tuple[Param, ...]] = {
    name: tuple(p for p in spec.params if p.kind == "block") for name, spec in COMMANDS.items()}


#: Int params that are positions, by axis (0=x, 1=y, 2=z).
_COORD_AXIS: Dict[str, int] = {
    "x": 0, "y": 1, "z": 2, "x1": 0, "y1": 1, "z1": 2, "x2": 0, "y2": 1, "z2": 2,
    "cx": 0, "cy": 1, "cz": 2}
#: Int params that are neither positions nor sizes (taken modulo 360 or so).
_FREE_INTS = frozenset({"rotate"})


def _validate_op(op: str, line_no: int, size: Sequence[int] = CONVERT_SIZE
                 ) -> Tuple[Optional[Call], Optional[str], Optional[str]]:
    """``(call, error, warning)`` for one child-emitted op (error None = keep it).

    Catches at lowering, where the issue can still name the SOURCE line: unknown
    blocks, ignored block states (a warning), positions more than one canvas
    width outside the canvas, and sizes (radius, height, count, thickness, ...)
    above one canvas width. The bounds are what keep the parent's executor,
    which no child limit covers, from a ``line`` to 10**8.
    """
    try:
        call, issue = parse_line(op, line_no)
    except (ValueError, OverflowError) as exc:
        return None, f"{op[:120]}: {exc}", None
    if issue is not None:
        return None, f"{op[:120]}: {issue.message}", None
    assert call is not None
    fname = py_names(call.spec)[0]
    m = max(int(v) for v in size)
    for p in call.spec.params:
        if p.kind != "int" or p.name not in call.args or p.name in _FREE_INTS:
            continue
        v = int(call.args[p.name])
        axis = _COORD_AXIS.get(p.name)
        if axis is not None:
            lo, hi = -m, int(size[axis]) - 1 + m
            if not lo <= v <= hi:
                return call, (f"{fname}(): {p.name}={v} is far outside the "
                              f"{'x'.join(str(int(s)) for s in size)} canvas "
                              f"(allowed {lo}..{hi}); the call was skipped"), None
        elif abs(v) > m:
            return call, (f"{fname}(): {p.name}={v} is larger than the canvas "
                          f"(limit {m}); the call was skipped"), None
    warn = None
    for p in _BLOCK_PARAMS.get(call.spec.name, ()):
        v = str(call.args[p.name])
        if v.strip().lower() in ("any", "*", "all"):
            continue
        try:
            resolve_block(v)
        except UnknownBlockError as exc:
            return call, f"{fname}(): {exc}", None
        warn = warn or state_warning(v)
    return call, None, (f"{fname}(): {warn}" if warn else None)


def _clipped_volume(call: Call, size: Sequence[int]) -> int:
    a = call.args
    vol = 1
    for i, (p, q) in enumerate((("x1", "x2"), ("y1", "y2"), ("z1", "z2"))):
        lo, hi = sorted((int(a[p]), int(a[q])))
        vol *= max(0, min(hi, int(size[i]) - 1) - max(lo, 0) + 1)
    return vol


def _op_work(call: Call, size: Sequence[int]) -> int:
    """Rough count of the voxel writes (python-level work) ``call`` costs the
    executor. Numpy box fills are cheap and count by clipped volume / 64; point
    sets built in Python (line, sphere, cylinder) count every point."""
    a, name = call.args, call.spec.name
    canvas = int(np.prod([int(v) for v in size]))
    if name in ("fill", "clear", "box", "walls"):
        return 1 + _clipped_volume(call, size) // 64
    if name == "replace":
        if str(a["from_block"]).strip().lower() in ("any", "*", "all"):
            return 1 + canvas
        return 1 + _clipped_volume(call, size) // 64
    if name == "line":
        steps = 1 + max(abs(int(a["x2"]) - int(a["x1"])), abs(int(a["y2"]) - int(a["y1"])),
                        abs(int(a["z2"]) - int(a["z1"])))
        t = max(1, int(a["thickness"]))
        return steps * (2 * t - 1) ** 3
    if name == "sphere":
        return (2 * max(0, int(a["radius"])) + 1) ** 3
    if name == "cylinder":
        return (2 * max(0, int(a["radius"])) + 1) ** 2 * max(1, int(a["height"]))
    if name == "pyramid":
        return 1 + max(0, int(a["size"]))
    if name == "gable":
        return 1 + max(abs(int(a["x2"]) - int(a["x1"])), abs(int(a["z2"]) - int(a["z1"])))
    if name in ("copy", "paste"):
        return 1 + canvas // 64
    if name == "stack":
        return 1 + max(0, int(a["count"])) * (1 + canvas // 64)
    return 1


def _bounded_run(argv: List[str], stdin: bytes, *, cwd: str, timeout: float,
                 max_stdout: int) -> Tuple[int, bytes, bytes, bool, bool]:
    """``subprocess.run`` with a byte cap on stdout (and a small one on stderr).

    Returns ``(returncode, stdout, stderr, timed_out, overflowed)``; the child is
    killed the moment either the deadline passes or stdout outgrows the cap.
    """
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd, env={}, close_fds=True)
    state = {"overflow": False}

    def feed() -> None:
        try:
            proc.stdin.write(stdin)       # type: ignore[union-attr]
            proc.stdin.close()            # type: ignore[union-attr]
        except (BrokenPipeError, OSError, ValueError):
            pass

    def drain(stream: Any, chunks: List[bytes], cap: int, kill: bool) -> None:
        total = 0
        while True:
            buf = stream.read(65536)
            if not buf:
                return
            if total + len(buf) > cap:
                if kill:
                    state["overflow"] = True
                    proc.kill()
                    return
                continue                  # stderr: keep the head, discard the rest
            chunks.append(buf)
            total += len(buf)

    out_chunks: List[bytes] = []
    err_chunks: List[bytes] = []
    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=drain, args=(proc.stdout, out_chunks, max_stdout, True),
                                daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, err_chunks, 1 << 16, False),
                                daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for t in threads:
        t.join(timeout=5)
    for s in (proc.stdout, proc.stderr):
        try:
            s.close()                     # type: ignore[union-attr]
        except OSError:
            pass
    return (proc.returncode, b"".join(out_chunks), b"".join(err_chunks), timed_out,
            state["overflow"])


#: The python mock reply: the same house as actions.MOCK_DSL (identical voxels,
#: asserted in tests), written the way we want models to write Python.
MOCK_PYTHON = '''\
```python
# foundation, then the shell standing on it
fill("cobblestone", 0, 0, 0, 6, 0, 6)
walls("oak_planks", 0, 1, 0, 6, 3, 6)
# timber corner posts
for x in (0, 6):
    for z in (0, 6):
        line("oak_log", x, 1, z, x, 3, z)
# pitched roof sitting on the wall tops
gable("oak_stairs", 0, 4, 0, 6, 4, 6, axis="x")
# door (two blocks tall), windows and a light
for y in (1, 2):
    place("oak_door", 3, y, 0)
for x, z in [(1, 0), (5, 0), (0, 3), (6, 3)]:
    place("glass_pane", x, 2, z)
place("lantern", 3, 1, 3)
```
'''

__all__ = ["BANNED_ATTRS", "BANNED_NAMES", "MOCK_PYTHON", "PythonActions", "RUNNER",
           "RunResult", "SAFE_BUILTINS", "api_reference", "check_code", "extract_code"]
