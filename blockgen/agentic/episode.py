"""The Episode: one prompt, one build, driven by a method through a few verbs.

A method (``recipes/*.py``) owns the control flow; this module owns the plumbing
that BuildAgent inlines today and every method would otherwise re-implement:

    ask      metered, traced LLM call per role, with the system prompt, context
             images and TARGET dreams put where each vendor can see them
    try_     lower -> execute on a fresh (or cloned, for patches) world -> render
             -> score -> accept or roll back; never raises on bad model text
    render   a world/attempt to PNGs per named view (cached on the attempt)
    dream    one image-generator call, metered, versioned, traced
    score    weighted scorer terms with gates, stamped with the dream set used
    best     the attempt the build returns (``select: last|best`` + final_score)

Two invariants matter for the paper's numbers:

* **Like with like.** A score is only comparable to scores measured against the
  same dreams. Every Score carries ``dream_id``; after a re-dream
  :meth:`Episode.rescore_all` re-anchors every attempt, so ``select: best`` never
  compares an attempt scored against dream v0 with one scored against v1.
* **Selection is not acceptance.** The accept policy decides which state the loop
  continues from; :meth:`Episode.best` may still return a rejected attempt (the
  argmax), which is what keep-best means.

:func:`run_episode` is the only entry point the suite uses: it builds the
Episode, runs the method, turns budget/cache/error exits into a truncated result
(keeping the best build so far), and closes the trace.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from blockgen.agentic.agent import AgentConfig
from blockgen.agentic.blockstate import PALETTE
from blockgen.agentic.canvas import Canvas, structure_to_canvas
from blockgen.agentic.core import (AcceptFn, Attempt, BudgetExceeded, CacheMiss, Charge,
                                   ContextParts, DreamSpec, DreamVersion, EmptyBuild,
                                   ImageRequest, Kit, Ledger, MethodResult,
                                   MethodSpec, NullTrace, PromptCtx, Score, ScoreTerm,
                                   SpendNotAllowed, TraceLike, World, billing_of,
                                   get_view, lookup)
from blockgen.agentic.core import Lowered
from blockgen.agentic.dsl import ExecutionReport, Issue
from blockgen.agentic.providers import (LLMResponse, Message, Part, encode_png,
                                        text_message)
from blockgen.utils.data import Structure

#: Roles whose (turn, reply) pair joins the accepted dialogue when try_ accepts:
#: the ones that write programs. Judges and dream critics never do.
PROGRAM_ROLES = frozenset({"builder", "critic"})
#: Roles that see the full builder system prompt (grammar, palette, contexts).
#: Reviewers get the short REVIEWER_SYSTEM: the grammar is irrelevant to judging
#: an image, costs tokens, and building it early would evaluate context modules
#: (e.g. dream_palette) before the dreams are final.
BUILDER_SYSTEM_ROLES = frozenset({"builder", "planner", "critic"})
REVIEWER_SYSTEM = (
    "You are a careful, terse reviewer of Minecraft builds and of images meant as "
    "targets for Minecraft builds. Judge only what you can see. Follow the requested "
    "reply format exactly.")

#: Share of ``max_tokens`` a reply is assumed to use when a usd cap is checked
#: BEFORE a call (the same default as ``suite.estimate``'s completion_frac).
EST_COMPLETION_FRAC = 0.3


def estimate_llm_usd(provider: Any, messages: Sequence[Message], gen: Dict[str, Any]) -> float:
    """Pre-call price estimate of one LLM request, for the ledger's usd caps.

    0 for a request already in the response cache (``peek``) or an unpriced model;
    else prompt chars/4 (+800 tokens per image) plus ``EST_COMPLETION_FRAC`` of
    ``max_tokens`` at the PRICES row. Never raises (a bad estimate only loosens the
    cap back to "stop after it was crossed").
    """
    from blockgen.agentic.providers import PRICES
    try:
        price = PRICES.get(str(getattr(provider, "model", "") or ""))
        if price is None:
            return 0.0
        peek = getattr(provider, "peek", None)
        if callable(peek) and peek(messages, **gen):
            return 0.0
        chars = imgs = 0
        for m in messages:
            for p in m.parts:
                if p.kind == "image":
                    imgs += 1
                else:
                    chars += len(p.text or "")
        prompt_tokens = chars / 4.0 + 800 * imgs
        completion = float(gen.get("max_tokens") or 16000) * EST_COMPLETION_FRAC
        return (prompt_tokens * price[0] + completion * price[1]) / 1e6
    except Exception:  # noqa: BLE001
        return 0.0


#: View names used for scoring/feedback when a recipe has no scored dreams.
DEFAULT_SCORE_VIEWS: Tuple[str, ...] = ("iso", "iso_back")


# ===========================================================================
# JSON replies
# ===========================================================================
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Dict[str, Any]:
    """Tolerant JSON-object extraction from a model reply; ``{}`` on failure.

    Models wrap JSON in code fences, prefix it with prose, or append a remark.
    Strategy: try fenced blocks, then the whole text, then every balanced
    ``{...}`` span from the first brace. A JSON list or scalar is not a verdict,
    so it also yields ``{}``.
    """
    if not text:
        return {}
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            v = json.loads(cand[start:i + 1])
                            if isinstance(v, dict):
                                return v
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            start = cand.find("{", start + 1)
    return {}


# ===========================================================================
# Component meta
# ===========================================================================
def _salt_int(salt: str) -> int:
    """A stable positive integer from a salt string (seed offset of a salted arm)."""
    return int(hashlib.sha256(str(salt).encode()).hexdigest()[:8], 16) % 1_000_003 + 1


def _scorer_flag(fn: Any, key: str, default: bool) -> bool:
    """A scorer's boolean meta flag (``needs_dream``).

    An attribute on the function wins (lets a test or an ad-hoc scorer declare it
    without registering), then the registry meta of its spec name, then ``default``.
    """
    target = getattr(fn, "func", fn)
    if hasattr(target, key):
        return bool(getattr(target, key))
    name = getattr(fn, "spec_name", None) or getattr(target, "__name__", "")
    try:
        return bool(lookup("scorer", name).meta.get(key, default))
    except (KeyError, ValueError):
        return default


# ===========================================================================
# Episode
# ===========================================================================
class Episode:
    """Implements :class:`~blockgen.agentic.core.EpisodeLike` for one build."""

    def __init__(self, prompt: str, kit: Kit, *, ledger: Optional[Ledger] = None,
                 seed: Optional[int] = None, images: Sequence[bytes] = (),
                 trace: Optional[TraceLike] = None) -> None:
        self.prompt = prompt
        self.kit = kit
        self.spec: MethodSpec = kit.spec
        # ``variation`` is the EXPLICIT seed (None when the caller gave none). The
        # legacy build turn adds a "Design variation #n" rider only for an explicit
        # seed, so classic must know the difference to stay byte-identical.
        self.variation: Optional[int] = seed
        #: ``spec.salt``: a salted arm shifts its seed, so its image requests get
        #: fresh cache keys (API backends ignore the seed itself, so it is simply a
        #: new sample), and its LLM requests carry a '[sample <salt>]' line.
        self.salt: str = str(getattr(kit.spec, "salt", "") or "")
        self.seed: int = (int(seed if seed is not None else kit.spec.seed)
                          + (_salt_int(self.salt) if self.salt else 0))
        #: :meth:`best` ranks only ``attempts[select_from:]`` (falling back to all
        #: when that holds no non-empty attempt). A staged method sets it when the
        #: building is first complete, so selection never returns a half-built house.
        self.select_from: int = 0
        self.size: Tuple[int, int, int] = tuple(int(v) for v in kit.spec.canvas)  # type: ignore
        self.world: World = kit.world_factory(self.size)
        self.dreams: Dict[str, List[DreamVersion]] = {}
        #: Current building description per dream key (a dream critic's or REDREAM
        #: rewrite). Every later (re)generation of that key uses it instead of the
        #: original prompt, so a rejected description never comes back.
        self.dream_desc: Dict[str, str] = {}
        self.reference_images: List[bytes] = list(images)
        self.attempts: List[Attempt] = []
        self.history: List[Message] = []
        self.plan: str = ""
        self.notes: Dict[str, Any] = {}
        self.ledger: Ledger = ledger if ledger is not None else kit.ledger.child(name=prompt[:40])
        self.trace: TraceLike = trace if trace is not None else NullTrace()
        self.transcript: List[Dict[str, Any]] = []
        self.select: str = kit.spec.select          # methods may override (best_of_n)
        self.truncated: str = ""
        self.error: str = ""
        self.config: Optional[AgentConfig] = None   # set by the legacy method
        #: False = do not prepend context parts / TARGET dreams to the first builder
        #: turn (classic's legacy-prompt mode must send BuildAgent's exact bytes).
        self.inject_first_turn: bool = True
        #: Context modules that return demonstration ``turns`` (context.ExampleParts)
        #: send them as real prior user/assistant turns when this is True, else as
        #: text in the first build turn (their fallback). Off by default until the
        #: suite switches over as a whole: flipping it changes request shapes, i.e.
        #: response-cache keys, for every recipe with an ``examples`` context.
        self.inserts_example_turns: bool = False
        self.last_response: Optional[LLMResponse] = None
        self._system: Optional[Message] = None
        self._ctx: Optional[List[Tuple[str, ContextParts]]] = None
        self._first_build_done = False
        self._pending: Optional[Tuple[Message, Message]] = None
        #: Set by a re-dream (keep_history only): the dialogue still shows the old
        #: TARGET images, so the next program turn is prefixed with the new ones
        #: until a turn carrying them is committed to the history.
        self._target_note: List[Part] = []
        self._branch_start = 0
        self._final_cache: Dict[int, float] = {}
        self._dream_specs: Dict[str, DreamSpec] = {d.key: d for d in kit.dreams}
        #: Keys added by :meth:`register_dream` (never scored, never in dream_id).
        self._runtime_dreams: Dict[str, DreamSpec] = {}
        self._needs_dream = any(_scorer_flag(fn, "needs_dream", False)
                                for fn, _ in kit.scorers.values())
        # final_score is meant to be dream-independent; if a recipe puts a
        # dream-dependent term there anyway, its cached values must be dropped
        # whenever the dreams change.
        self._final_needs_dream = any(_scorer_flag(fn, "needs_dream", False)
                                      for fn, _ in kit.final_scorers.values())
        # Render the score views of every non-empty attempt only when some scorer
        # reads pixels (meta ``needs_renders``, default True for an undeclared
        # scorer). A validity-only score skips the renders; the suite still renders
        # the trace views after the episode (suite._ensure_trace_renders).
        self._needs_renders = any(_scorer_flag(fn, "needs_renders", True)
                                  for fn, _ in [*kit.scorers.values(),
                                                *kit.final_scorers.values()])

    # ------------------------------------------------------------------ prompts
    def contexts(self) -> List[Tuple[str, ContextParts]]:
        """Every context module's output, computed once per episode."""
        if self._ctx is None:
            out = []
            for name, fn in self.kit.contexts:
                parts = fn(self)
                out.append((name, parts if isinstance(parts, ContextParts)
                            else ContextParts(system_text=str(parts or ""))))
            self._ctx = out
        return self._ctx

    def prompt_ctx(self) -> PromptCtx:
        return PromptCtx(size=self.size, palette=PALETTE, prompt=self.prompt,
                         target_blocks=self.spec.target_blocks)

    def system(self) -> Message:
        """Builder system message: action-space grammar + each context's text.

        Built lazily on first use so context modules that read dreams (e.g. a
        dream palette) see the final dream set. A method may pin it first with
        :meth:`set_system` (classic does, to reuse the legacy system prompt).
        """
        if self._system is None:
            blocks = [self.kit.actions.instructions(self.prompt_ctx())]
            blocks += [c.system_text for _, c in self.contexts() if c.system_text]
            self._system = text_message("system", "\n\n".join(b.strip("\n") for b in blocks if b))
        return self._system

    def set_system(self, msg: Message) -> None:
        self._system = msg

    def _first_turn_prefix(self) -> List[Part]:
        """Context images, then every builder-visible dream, labelled."""
        from blockgen.agentic.turns import builder_dream_parts

        parts: List[Part] = []
        for _, c in self.contexts():
            parts.extend(c.user_parts)
        return parts + builder_dream_parts(self)

    # ---------------------------------------------------------------------- ask
    def ask(self, turn: Union[str, Message], *, images: Sequence[bytes] = (),
            role: str = "builder", history: Union[bool, Sequence[Message]] = True,
            json_reply: bool = False, stage: str = "") -> Any:
        """One metered, traced LLM call. Returns the text, or a dict if ``json_reply``.

        ``history`` may be True (the accepted dialogue), False (none), or an
        explicit message list (the legacy keep_history=False request shapes).
        """
        msg = text_message("user", turn) if isinstance(turn, str) else turn
        extra = [Part("image", image_png=b) for b in images]
        prefix: List[Part] = []
        demos: List[Message] = []
        if role == "builder" and self.inject_first_turn and not self._first_build_done:
            prefix = self._first_turn_prefix()
            if self.inserts_example_turns:
                demos = [t for _, c in self.contexts() for t in (getattr(c, "turns", None) or [])]
        if role == "builder":
            self._first_build_done = True
        if role in PROGRAM_ROLES and self._target_note:
            # Skip images the turn already carries (raw feedback shows the new dreams).
            have = {p.image_png for p in msg.parts if p.kind == "image"}
            prefix = prefix + [p for p in self._target_note
                               if not (p.kind == "image" and p.image_png in have)]
        if prefix or extra:
            msg = Message(msg.role, prefix + list(msg.parts) + extra)
        system = (self.system() if role in BUILDER_SYSTEM_ROLES
                  else text_message("system", REVIEWER_SYSTEM))
        if self.salt:
            # A different request means a different cache key: the salted arm draws
            # its own LLM samples instead of sharing its twin's cached replies.
            system = Message(system.role, list(system.parts)
                             + [Part("text", text=f"[sample {self.salt}]")])
        if demos and history is True and self.spec.keep_history:
            self.history.extend(demos)        # permanent, like legacy example turns
            demos = []
        hist = (list(self.history) if history is True
                else [] if history is False else list(history))
        msgs = [system] + hist + demos + [msg]

        provider = self.kit.llm(role)
        gen = {"max_tokens": 16000, "temperature": None, "reasoning_effort": "medium",
               **(self.spec.gen or {})}
        # Checked against this call's estimated price, so a usd cap stops the call
        # that would cross it rather than the one after.
        self.ledger.check("llm", estimate_llm_usd(provider, msgs, gen))
        resp = provider.complete(msgs, **gen)
        self.ledger.charge(Charge.from_llm(role, resp, billing_of(provider)))
        self.trace.request(role, msgs, resp, stage)
        self.transcript.append({"stage": stage, "role": role, "request": msg.text,
                                "response": resp.text, "usage": resp.to_dict()})
        if role in PROGRAM_ROLES:
            self.last_response = resp
            self._pending = (msg, text_message("assistant", resp.text))
        if not json_reply:
            return resp.text
        out = extract_json(resp.text)
        if not out:
            self.trace.event("json_parse_fail", role=role, stage=stage, text=resp.text[:500])
        return out

    # --------------------------------------------------------------------- try_
    def try_(self, stage: str, text: str, *, patch: bool = False,
             region: Optional[Any] = None, offset: Optional[Tuple[int, int, int]] = None,
             ir: bool = False, accept: Optional[AcceptFn] = None) -> Attempt:
        """Execute one candidate and decide whether the loop continues from it.

        ``ir=True``: ``text`` is canonical DSL written by the method (a blueprint
        shell, a roof), so it is not lowered through the action space, and it never
        commits an LLM (turn, reply) pair to the dialogue. ``offset``: the text is
        in region-local coordinates; it is moved by ``offset`` before running.
        ``region`` (a :class:`~blockgen.agentic.region.Region`): after running,
        every voxel outside it is restored from the CURRENT accepted world, so a
        rewrite inside a region replaces that room and a patch adds to it; the
        discarded writes come back as warning issues on the model's own lines, and
        ``extra['reverted']`` counts them. The traced structure.npz is then the
        authoritative replay (``extra['replay_exact'] = False``): the cumulative
        program no longer rebuilds the world by itself. ``accept`` overrides the
        recipe's accept policy for this call (a room stage compares room scores).
        """
        actions = self.kit.actions
        if ir:
            src = text or ""
            low = Lowered(src, src, [], len(src.splitlines()))
            self._pending = None
        else:
            low = actions.lower(text or "")
        moved = offset is not None and any(int(v) for v in offset)
        if moved:
            from blockgen.agentic.region import translate_lowered
            low = translate_lowered(low, tuple(int(v) for v in offset))  # type: ignore[arg-type]
        patch = bool(patch and (ir or getattr(actions, "patchable", False)))
        cur = self.current()
        base = self.world.clone() if patch else self.kit.world_factory(self.size)
        start = base.clone() if region is not None else None
        report = base.run(low)
        extra: Dict[str, Any] = {}
        if region is not None:
            from blockgen.agentic.region import changed_mask, clipped_issues, restore_outside
            # What THIS program wrote outside the region (on a fresh world, the
            # current world's other rooms are not "writes"), then put them back.
            wrote = changed_mask(base, start) & ~region.mask(self.size)
            restore_outside(base, self.world, region)
            extra["region"] = region.name
            if wrote.any():
                extra["reverted"] = int(wrote.sum())
                extra["replay_exact"] = False
        if not ir and low.program_text != low.source and report.issues:
            # Execution issues are numbered by LOWERED line; python / mixed lowering
            # tags each op '# L<n>', so map them back to lines the model wrote.
            # Identity lowerings (dsl) skip this, keeping legacy repair bytes exact.
            from blockgen.agentic.actions import remap_issues
            n_low = len(low.issues)
            report.issues = (list(report.issues[:n_low])
                             + remap_issues(low, report.issues[n_low:]))
        if moved and report.issues:
            # Room-local coordinates: quote the model's own (untranslated) line, so
            # the repair turn does not show coordinates it never wrote. Only for
            # line-aligned programs (dsl, blocks, ir), where the numbers match.
            src_lines = low.source.split("\n")
            if len(src_lines) == len(low.program_text.split("\n")):
                report.issues = [
                    Issue(i.line_no, (src_lines[i.line_no - 1].split("#", 1)[0].strip()
                                      or i.text), i.message, i.severity)
                    if 0 < i.line_no <= len(src_lines) else i
                    for i in report.issues]
        if region is not None and extra.get("reverted"):
            report.issues = list(report.issues) + clipped_issues(start, low, region)
        if region is not None and cur is not None:
            # Outside the region the world is the current one, so the cumulative
            # program is too (exact when nothing was reverted).
            program_text = (cur.program_text + f"\n# --- region {region.name} ---\n"
                            + low.program_text)
        elif patch and cur is not None:
            program_text = cur.program_text + "\n# --- patch ---\n" + low.program_text
        else:
            program_text = low.program_text
        if region is not None and not patch and cur is not None:
            # A rewrite inside the region dropped what the region held before, which
            # the replayed cumulative program would still build.
            extra["replay_exact"] = False
        resp = (LLMResponse(text=text or "", model="") if ir
                else self.last_response or LLMResponse(text=text or "", model=""))
        att = Attempt(stage, program_text, report, resp, base.block_count(),
                      idx=len(self.attempts), source_text=text or "",
                      delta_text=low.program_text, patch=patch, world=base, extra=extra)
        try:
            self._evaluate(att)
        except Exception:
            # A scorer or renderer crash (CUDA OOM, missing weights) must not throw
            # away a build the model already paid for: keep it, ranked below every
            # scored attempt, then let run_episode record the error.
            att.score = Score(0.0, gated="score_error")
            att.accepted, att.reason = False, "score_error"
            self._pending = None
            self.last_response = None
            self.attempts.append(att)
            self.trace.attempt(att)
            raise
        ok, reason = (accept or self.kit.accept)(cur, att)
        att.accepted, att.reason = bool(ok), str(reason)
        if att.accepted:
            self.world = base
            if self.spec.keep_history and self._pending is not None:
                self.history += list(self._pending)
                self._target_note = []
        self._pending = None
        self.last_response = None
        self.attempts.append(att)
        self.trace.attempt(att)
        return att

    def _render_for_score(self, att: Attempt) -> None:
        """Make sure the scored views of ``att`` are rendered (if anything needs them)."""
        if att.blocks > 0 and self.kit.renderer is not None and self._needs_renders:
            try:
                self.render(att, "score")
            except EmptyBuild:
                pass

    def _evaluate(self, att: Attempt) -> None:
        """Render the scored views (if anything needs them) and score."""
        self._render_for_score(att)
        att.score = self.score(att)

    def adopt(self, att: Attempt, *, accepted: bool, reason: str = "") -> Attempt:
        """Record an attempt executed OUTSIDE try_ (the legacy wrapper's rounds):
        render, score, trace, but no accept policy and no history."""
        att.idx = len(self.attempts)
        self._evaluate(att)
        att.accepted, att.reason = accepted, reason
        if accepted and att.world is not None:
            self.world = att.world
        self.attempts.append(att)
        self.trace.attempt(att)
        return att

    # ------------------------------------------------------------------- render
    def _view_names(self, views: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(views, str):
            if views != "score":
                return [views]
            names: List[str] = []
            for ds in self.kit.dreams:
                if ds.view:
                    for v in (ds.view,) + tuple(ds.view_search):
                        if v not in names:
                            names.append(v)
            return names or list(DEFAULT_SCORE_VIEWS)
        return list(views)

    def render(self, target: Union[World, Attempt, Structure],
               views: Union[str, Sequence[str]] = "score") -> Dict[str, bytes]:
        """PNG per view name. Attempts cache their renders; an empty build and a
        recipe without a renderer give ``{}``. RenderError propagates (fail loud)."""
        names = self._view_names(views)
        if self.kit.renderer is None:
            return {}
        cache: Optional[Dict[str, bytes]] = None
        if isinstance(target, Attempt):
            cache = target.renders
            if target.world is None:
                return {n: cache[n] for n in names if n in cache}
            structure = target.world.to_structure()
        elif isinstance(target, Structure):
            structure = target
        else:
            structure = target.to_structure()
        missing = [n for n in names if cache is None or n not in cache]
        out: Dict[str, bytes] = {}
        if missing:
            if int(structure.occupied_mask.sum()) == 0:
                return {}
            imgs = self.kit.renderer.render(structure, [get_view(n) for n in missing])
            for n, img in zip(missing, imgs):
                png = img if isinstance(img, (bytes, bytearray)) else encode_png(img)
                out[n] = bytes(png)
            if cache is not None:
                cache.update(out)
        if cache is not None:
            return {n: cache[n] for n in names if n in cache}
        return {n: out[n] for n in names if n in out}

    # -------------------------------------------------------------------- dream
    def dream_spec(self, key: str) -> DreamSpec:
        if key not in self._dream_specs:
            raise KeyError(f"unknown dream '{key}'; this recipe has {sorted(self._dream_specs)}")
        return self._dream_specs[key]

    def register_dream(self, ds: DreamSpec, *, trace_view: Optional[str] = None) -> DreamSpec:
        """Add a RUNTIME dream: one a method creates while it runs (a room's interior,
        a turnaround sheet, the blueprint schematic).

        It works with :meth:`dream`, :meth:`latest_dream` and :meth:`put_dream`, and
        is traced and metered like any dream, but it is never in ``kit.dreams``: so
        it is never scored, never part of :meth:`dream_id` and never shown in the
        builder's TARGET prefix. Why: a per-room target scored globally would make
        every exterior score depend on which rooms were dreamed so far.

        Re-registering a key with an equal spec is a no-op; a different spec for a
        RECIPE dream raises ValueError (a runtime spec may be replaced).
        ``trace_view`` names the render row the dream pairs with on the contact sheet.
        """
        recipe = {d.key: d for d in self.kit.dreams}
        if ds.key in recipe:
            if recipe[ds.key] != ds:
                raise ValueError(f"dream '{ds.key}' is a recipe dream with a different "
                                 "spec; a runtime dream needs its own key")
            return recipe[ds.key]
        old = self._runtime_dreams.get(ds.key)
        if old == ds:
            return old
        self._runtime_dreams[ds.key] = ds
        self._dream_specs[ds.key] = ds
        views = getattr(self.trace, "dream_views", None)
        if isinstance(views, dict):
            views[ds.key] = trace_view
        self.trace.event("register_dream", key=ds.key, view=ds.view, trace_view=trace_view)
        return ds

    def put_dream(self, key: str, png: bytes, *, prompt: str = "", generator: str = "derived",
                  critique: Optional[Dict[str, Any]] = None) -> DreamVersion:
        """Append a version of ``key`` WITHOUT calling the generator (no charge): a
        tile split from a turnaround sheet, the blueprint schematic. Traced.

        If ``key`` is a scored recipe dream, the dream set changed: the caller must
        :meth:`rescore_all` before comparing scores (like with like).
        """
        self.dream_spec(key)                 # KeyError naming the known dreams
        versions = self.dreams.setdefault(key, [])
        dv = DreamVersion(key, len(versions), bytes(png), prompt, generator,
                          seed=self.seed, critique=dict(critique or {}))
        versions.append(dv)
        self.trace.dream(dv)
        return dv

    def dream(self, key: str, *, prompt: Optional[str] = None, refs: Sequence[bytes] = (),
              seed: Optional[int] = None, store: bool = True,
              view_hint: Optional[str] = None) -> DreamVersion:
        """Generate one version of dream ``key`` (metered, cached by the generator).

        Default refs are the latest versions of the dreams it is conditioned on.
        ``view_hint`` replaces the spec's view in the request (mock generators draw
        it, e.g. 'room' or 'grid2x2:iso,...'; API backends ignore it).
        """
        from blockgen.agentic import dreams as dreams_mod

        ds = self.dream_spec(key)
        image = self.kit.image
        if image is None:
            raise RuntimeError(f"recipe '{self.spec.name}' has dreams but no image generator")
        text = prompt or dreams_mod.render_prompt(ds, self)
        if not refs:
            refs = [dv.png for dv in (self.latest_dream(k) for k in ds.condition_on) if dv]
        refs = list(refs)
        if refs and not image.supports_refs():
            self.trace.event("refs_dropped", key=key, n=len(refs), generator=image.describe())
            refs = []
        req = ImageRequest(text, tuple(refs), size=str(getattr(image, "size", "1024x1024")),
                           quality=str(getattr(image, "quality", "low")),
                           seed=int(self.seed if seed is None else seed),
                           view=(view_hint if view_hint is not None else ds.view) or "")
        self.ledger.check("image", image.estimate_usd(req))
        res = image.generate(req)
        self.ledger.charge(Charge.from_image("image", res, billing_of(image)))
        versions = self.dreams.setdefault(key, []) if store else self.dreams.get(key, [])
        dv = DreamVersion(key, len(versions), res.images[0], text, image.describe(),
                          seed=req.seed, cost_usd=res.cost_usd, cached=res.cached)
        if store:
            versions.append(dv)
            self.trace.dream(dv)
        return dv

    def note_target_changed(self, desc: str = "") -> None:
        """Tell the builder/critic dialogue that the TARGET changed (after a re-dream).

        The first build turn (and earlier compare turns) in the history show the
        old dream; without this note a score/vdm critic would be measured against
        an image it never sees. No-op without ``keep_history`` (nothing stale).
        """
        from blockgen.agentic.turns import builder_dream_parts

        if not self.spec.keep_history:
            return
        head = "The TARGET has changed: it was re-imagined"
        head += f" as: {desc.strip()}." if desc and desc.strip() else "."
        head += " These NEW TARGET images replace every earlier TARGET image:"
        self._target_note = [Part("text", text=head)] + builder_dream_parts(self)

    def latest_dream(self, key: str) -> Optional[DreamVersion]:
        vs = self.dreams.get(key)
        return vs[-1] if vs else None

    def dream_id(self) -> str:
        """Digest over the latest version of every SCORED dream ('' if none)."""
        parts = [f"{d.key}:{dv.digest}" for d in self.kit.dreams
                 if d.view and (dv := self.latest_dream(d.key)) is not None]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12] if parts else ""

    # -------------------------------------------------------------------- score
    def score(self, att: Attempt) -> Score:
        """Weighted composite of the recipe's scorer terms, after gates.

        An empty build is 0 without calling any scorer (nothing to render).
        """
        if att.blocks <= 0:
            return Score(0.0, gated="empty", dream_id=self.dream_id() if self._needs_dream else "")
        parts: Dict[str, float] = {}
        per_view: Dict[str, Dict[str, float]] = {}
        notes: Dict[str, str] = {}
        num = den = 0.0
        for name, (fn, w) in self.kit.scorers.items():
            term = fn(self, att)
            if not isinstance(term, ScoreTerm):
                term = ScoreTerm(float(term))
            parts[name] = float(term.value)
            if term.per_view:
                per_view[name] = dict(term.per_view)
            if term.note:
                notes[name] = term.note
            num += float(w) * float(term.value)
            den += float(w)
        value = num / den if den > 0 else 0.0
        gated = ""
        for g, thr in (self.spec.gates or {}).items():
            if g in parts and parts[g] < float(thr):
                value, gated = 0.0, g
                break
        return Score(float(value), parts, per_view, notes,
                     dream_id=self.dream_id() if self._needs_dream else "", gated=gated)

    def _outdated(self, att: Attempt, did: str) -> bool:
        return att.score is None or (self._needs_dream and att.score.dream_id != did)

    def rescore_all(self) -> None:
        """Re-anchor every attempt to the current dreams (missing views re-rendered).

        Only attempts scored against another dream set (or not at all) are
        recomputed: a dream-dependent scorer may be an LLM judge, and re-asking it
        about an already current score costs money for nothing. Empty attempts are
        re-stamped too (free: no scorer runs), else they would keep the episode
        :meth:`stale` forever and force a second full rescore at the end.

        Atomic: every new score is computed before any is assigned, so a budget
        stop half-way never leaves a mix of dream sets for :meth:`best` to compare.
        """
        did = self.dream_id()
        todo = [a for a in self.attempts if self._outdated(a, did)]
        new: List[Score] = []
        for att in todo:
            self._render_for_score(att)
            new.append(self.score(att))
        for att, sc in zip(todo, new):
            att.score = sc
        if self._final_needs_dream:
            self._final_cache.clear()
        self.trace.event("rescore_all", dream_id=did, n=len(todo))

    def stale(self) -> bool:
        """True if some attempt was scored against an older dream set."""
        if not self._needs_dream:
            return False
        did = self.dream_id()
        return any(a.score is not None and a.score.dream_id != did for a in self.attempts)

    def final_value(self, att: Attempt) -> float:
        """The selection value: ``final_score`` terms if configured, else the score.

        A gated attempt ranks below every ungated one (-1), so a dream-independent
        tie-break can never resurrect a build the gates rejected.
        """
        if att.score is not None and att.score.gated:
            return -2.0 if att.score.gated == "score_error" else -1.0
        if not self.kit.final_scorers:
            return float(att.score.value) if att.score else 0.0
        if att.idx not in self._final_cache:
            num = den = 0.0
            for fn, w in self.kit.final_scorers.values():
                term = fn(self, att)
                num += float(w) * float(term.value if isinstance(term, ScoreTerm) else term)
                den += float(w)
            self._final_cache[att.idx] = num / den if den > 0 else 0.0
        return self._final_cache[att.idx]

    # ---------------------------------------------------------------- selection
    def current(self) -> Optional[Attempt]:
        """Last accepted attempt of the current branch (the state the loop continues from)."""
        for a in reversed(self.attempts[self._branch_start:]):
            if a.accepted:
                return a
        return None

    def best(self) -> Optional[Attempt]:
        """``select: last`` = last accepted; ``best`` = argmax of final_value over
        every non-empty attempt, accepted or not (ties -> accepted, then later);
        ``best_accepted`` = the same argmax over accepted attempts (all attempts if
        none was accepted, so a build is still returned)."""
        if self.select in ("best", "best_accepted"):
            pick: Optional[Attempt] = None
            best_v = float("-inf")
            pool = ([a for a in self.attempts[self.select_from:] if a.blocks > 0]
                    or [a for a in self.attempts if a.blocks > 0])
            if self.select == "best_accepted":
                pool = [a for a in pool if a.accepted] or pool
            if not self.kit.final_scorers and self.stale():
                # A failed final rescore left scores measured against different
                # dreams; only those anchored to the CURRENT dreams are comparable.
                did = self.dream_id()
                pool = [a for a in pool if a.score is not None and a.score.dream_id == did] or pool
            # On a score tie an accepted attempt beats a rejected one (the accept
            # policy saw something the score did not); among equals, the later wins.
            best_key = (best_v, False)
            for a in pool:
                key = (self.final_value(a), bool(a.accepted))
                if key >= best_key:
                    pick, best_key = a, key
            return pick
        for a in reversed(self.attempts):
            if a.accepted:
                return a
        return None

    def new_branch(self) -> None:
        """Start an independent candidate in the same episode (best_of_n): empty
        world, no dialogue, no plan. Attempts are kept, so selection spans branches."""
        self.world = self.kit.world_factory(self.size)
        self.history = []
        self.plan = ""
        self._pending = None
        self._target_note = []
        self._first_build_done = False
        self._branch_start = len(self.attempts)

    # ------------------------------------------------------------------- result
    def result(self, elapsed_s: float = 0.0) -> MethodResult:
        best = self.best()
        if self.kit.final_scorers:
            # The values best() ranked by (never recomputed here: a final scorer
            # may be costly), so the trace shows what selection actually used.
            for a in self.attempts:
                if a.score is not None and a.score.gated:
                    a.selection = -2.0 if a.score.gated == "score_error" else -1.0
                else:
                    a.selection = self._final_cache.get(a.idx)
        if best is not None and best.world is not None:
            world = best.world
            structure = world.to_structure()
            canvas = getattr(world, "canvas", None)
            if not isinstance(canvas, Canvas):
                canvas = structure_to_canvas(world.to_structure(crop=False), self.size)
            program_text, report = best.program_text, best.report
        else:
            canvas = Canvas(self.size)
            structure = canvas.to_structure(crop=False)
            program_text, report = "", ExecutionReport()
        blocks = int(structure.occupied_mask.sum())
        error = "" if blocks else "empty build (no blocks placed)"
        if self.truncated:
            note = f"truncated: {self.truncated}" + (f" ({self.error})" if self.error else "")
            error = f"{error}; {note}" if error else note
        return MethodResult(
            description=self.prompt, structure=structure, canvas=canvas,
            program_text=program_text, report=report, rounds=list(self.attempts),
            plan_text=self.plan, transcript=list(self.transcript), config=self.config,
            elapsed_s=elapsed_s, error=error, spec_name=self.spec.name,
            recipe_hash=self.spec.hash(), dreams={k: list(v) for k, v in self.dreams.items()},
            charges=list(self.ledger.charges),
            best_idx=(best.idx if best is not None else None), truncated=self.truncated,
            trace_dir=(str(self.trace.root) if getattr(self.trace, "root", None) else None),
            provenance=self.spec.provenance())


# ===========================================================================
# run_episode
# ===========================================================================
def run_episode(kit: Kit, prompt: str, *, images: Sequence[bytes] = (),
                seed: Optional[int] = None, trace: Optional[TraceLike] = None) -> MethodResult:
    """Run ``kit.method`` on one prompt and return its result. Never raises for a
    budget stop, a read-only cache miss or a method error: those truncate, and the
    best attempt so far is returned. SpendNotAllowed does propagate, because it
    means the run is misconfigured, not that one build went wrong.
    """
    from blockgen.agentic.kit import episode_budget

    trace = trace if trace is not None else NullTrace()
    ledger = kit.ledger.child(episode_budget(kit.spec), name=prompt[:40])
    ep = Episode(prompt, kit, ledger=ledger, seed=seed, images=images, trace=trace)
    t0 = time.time()
    try:
        kit.method(ep)              # params already bound by kit.resolve_method
    except SpendNotAllowed:
        raise
    except BudgetExceeded as exc:
        ep.truncated, ep.error = "budget", str(exc)
        trace.event("budget_stop", error=str(exc))
    except CacheMiss as exc:
        ep.truncated, ep.error = "cache_miss", str(exc)
        trace.event("cache_miss", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad build must not kill a battery
        ep.truncated, ep.error = "error", repr(exc)
        trace.event("error", error=repr(exc), traceback=traceback.format_exc())
    if ep.stale():
        try:
            ep.rescore_all()
        except SpendNotAllowed:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the build, record the failure
            trace.event("rescore_error", error=repr(exc))
            if not ep.truncated:
                kind = ("budget" if isinstance(exc, BudgetExceeded)
                        else "cache_miss" if isinstance(exc, CacheMiss) else "error")
                ep.truncated, ep.error = kind, (str(exc) if kind != "error" else repr(exc))
    res = ep.result(elapsed_s=time.time() - t0)
    trace.close(res)
    return res


__all__ = ["BUILDER_SYSTEM_ROLES", "DEFAULT_SCORE_VIEWS", "Episode", "PROGRAM_ROLES",
           "REVIEWER_SYSTEM", "extract_json", "run_episode"]
