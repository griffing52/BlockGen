"""Reading a scorecard whose shape is older than the code reading it.

Every scorecard on disk says `"schema_version": "bench/1"`, and that string is a
lie of omission: the eleven cards in `outputs/` were written over five weeks by
three different versions of the bench and come in **four incompatible shapes**.
Verified by reading all eleven this session:

    run_20260803_033522_bench_compare   8 arms   6 ctl + 1 ar + 1 agentic  both  houses_32   A
    run_20260816_081955_bench_pnp       9 arms   6 ctl + 2 ar + 1 agentic  both  houses_32   A
    run_20260831_195140_bench           7 arms   7 ctl                     fast  houses_32   B
    run_20260831_195352_bench           7 arms   7 ctl                     fast  houses_32   B
    run_20260831_195452_bench           8 arms   8 ctl                     fast  houses_32   B
    run_20260831_195547_bench          11 arms   8 ctl + 2 ar + 1 agentic  both  houses_32   C
    run_20260831_202936_bench          11 arms   8 ctl + 2 ar + 1 agentic  both  houses_32   C
    run_20260831_204112_bench          11 arms   8 ctl + 2 ar + 1 agentic  both  houses_32   C
    run_20260831_225029_bench          16 arms   8 ctl + 5 base + 2 ar + 1 ag  both  houses_32   C
    run_20260906_185828_bench          12 arms   8 ctl + 4 agentic         fast  houses_32   D
    run_20260906_190029_bench          12 arms   8 ctl + 4 agentic         fast  houses_48   D

    A  no `run.blockscore`, has `fidelity`, no `geometry_scalars`, and `run.dir`
       is an ABSOLUTE path to a scratchpad directory outside `outputs/`
    B  `blockscore` + `geometry_scalars`, every arm a control, no `head_to_head`
    C  `blockscore` + `fidelity` + `geometry_scalars`, the later ones `head_to_head`
    D  `cost`, no `fidelity`

**So nothing here branches on the version string.** A migration keyed on
`schema_version` would be wrong about eight of those eleven cards before it ran.
Every question this module answers -- which sections exist, what the card
predates, whether an arm is a submission -- is answered from the *presence* of a
section or a key. The string is copied into `compat.version` for display and used
for exactly one decision: a major version above `SCHEMA_MAJOR_SUPPORTED` sets
`compat.unsupported`, and the page then renders whatever parses behind an amber
banner rather than pretending it understood the file.

**The memo rule, which is why `migrate` looks paranoid.**
`catalog._read_scorecard` memoizes the parsed blob process-wide, keyed on mtime,
and `catalog.load_scorecard` hands out a *shallow* copy of it (catalog.py:598) --
so `blob["arms"]` is the one object in the cache, shared by every reader for the
life of the process. One `arms[name]["meta"]["kind"] = ...` at read time poisons
that cache permanently, and the poisoning is cumulative and invisible: the second
request sees a card the file does not contain. `migrate` therefore returns a
**new top-level dict** and shares `arms` **by identity** -- `migrate(b, ...)["arms"]
is b["arms"]` -- and writes into nothing nested except a freshly built `run`
block. Derived per-arm data belongs in `arms_index`, built per request, beside
`arms` and never inside it. The invariant is structural rather than intended, so
`tests/lab/test_cards.py` can assert it with `is`.

`migrate` also never raises and is idempotent. Never raises because the lab's
whole read side degrades to an empty result rather than a stack trace, and a card
written by a future bench must still open. Idempotent because the memo, the API
payload and the tests each pass blobs around and nobody should have to track how
many times one has been through here.

One thing this module deliberately does not treat as identity: `run.dir`. Shape A
records an absolute scratchpad path, so the run id is always `path.parent.name`,
passed in by the caller.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

#: The schema major this reader understands. `compat.unsupported` is set above
#: it; it is never consulted to decide how to *read* a card (see the docstring).
SCHEMA_MAJOR_SUPPORTED = 2
READ_AS = f"bench/{SCHEMA_MAJOR_SUPPORTED}"

#: `outputs/run_<YYYYMMDD_HHMMSS>_<name>`, the convention `blockgen.utils.runs`
#: writes and every run directory in this repo follows. The trailing name is
#: optional and unconstrained -- `--name` puts a slug there.
RUN_DIR_RE = re.compile(r"^run_(\d{8})_(\d{6})(?:_(.*))?$")

#: What an arm is, for grouping and for the leaderboard's controls toggle.
KINDS = ("submission", "control", "baseline")
#: Where an arm's builds came from. "in_memory" means nothing was ever written to
#: disk, which is a different sentence from "the dataset join failed".
ORIGINS = ("npz", "in_memory")

#: The features a current card carries, in the order `compat.missing` reports
#: them. Each name is checked by presence, never by version, and each is a thing
#: the page can offer to show; a name here means "this card cannot show it".
MISSING_CHECKS = ("run.blockscore", "run.head_to_head", "ladder", "run.examples",
                  "meta.provenance", "geometry_scalars", "cost", "fidelity")

#: How many submission names a derived label spells out before it says `+N`.
LABEL_ARMS = 3

_RECIPES_MEMO: dict[str, str] | None = None


def _iso(epoch: float) -> str:
    """UTC ISO-8601 to the second with a `Z`, matching `catalog._when`."""
    return (datetime.fromtimestamp(epoch, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def dir_stamp(name: str) -> str | None:
    """`"run_20260831_225029_bench"` -> `"2026-08-31T22:50:29Z"`, else None.

    Used as the run-picker's sort key and as the fallback for `run.started_at` on
    the eleven cards that predate it. All eleven directory names parse, so
    ordering the picker costs no JSON reads and -- unlike `st_mtime`, which is
    what it replaces -- a `git checkout` or a stray `touch` cannot reorder it.

    The honest caveat: `runs.run_stamp()` formats **local** time, so the `Z` here
    is a convenient fiction. Both consumers survive it. The sort compares stamps
    written by the same clock, and the display line is the wall-clock time the
    researcher remembers starting the run. A card that carries a real
    `run.started_at` (`bench/2` writes a true UTC instant) always wins over this.

    Rejects a malformed date rather than emitting one: `strptime` is the check,
    because the regex alone accepts `run_99999999_999999_x`.
    """
    m = RUN_DIR_RE.match(str(name or ""))
    if m is None:
        return None
    try:
        when = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def arm_kind(meta: dict[str, Any]) -> str:
    """`"submission"` | `"control"` | `"baseline"` for one arm's `meta` block.

    The single definition of this rule. It was being re-derived as
    `track not in {"control", "baseline"}` in three places across two languages,
    which is how "16 arms" ended up on the picker for a run with 3 submissions.

    A written `meta.kind` wins (`bench/2` records it at scoring time, where the
    arm's construction is actually known); an unrecognised value falls back to
    the derivation rather than reaching the page as a mystery pill.
    """
    meta = meta if isinstance(meta, dict) else {}
    kind = meta.get("kind")
    if isinstance(kind, str) and kind in KINDS:
        return kind
    track = meta.get("track")
    if isinstance(track, str) and track in ("control", "baseline"):
        return track
    return "submission"


def arm_origin(meta: dict[str, Any]) -> str:
    """`"npz"` | `"in_memory"`: was anything written to disk for this arm?

    Legacy cards only say so obliquely -- `meta.source` is a path or the literal
    string `"in-memory"` (note the hyphen; it is a display string that five
    consumers key off, so `bench/2` adds a field beside it rather than changing
    it).
    """
    meta = meta if isinstance(meta, dict) else {}
    origin = meta.get("origin")
    if isinstance(origin, str) and origin in ORIGINS:
        return origin
    return "in_memory" if meta.get("source") == "in-memory" else "npz"


def sections(arms: dict[str, Any]) -> list[str]:
    """Every section name present on any arm, in first-seen order.

    A UNION, not the first arm's keys: `fast.py` returns a truncated block set
    for an arm that produced no non-empty structures, and `real_test` is only
    first when `--no-controls` was not passed. Reading the first arm alone would
    make a whole family of metrics vanish from the page because one arm failed.
    """
    out: list[str] = []
    if not isinstance(arms, dict):
        return out
    for arm in arms.values():
        if not isinstance(arm, dict):
            continue
        for name in arm:
            if name not in out:
                out.append(name)
    return out


def missing(blob: dict[str, Any], run: dict[str, Any]) -> list[str]:
    """Names of what this card cannot show, so the page can say so once.

    Presence checks against `MISSING_CHECKS`, in that order. The page turns these
    into prose ("Unavailable here: BlockScore ranking, example builds, arm
    provenance") and renders every table that *does* parse underneath. An empty
    list means the card carries the lot.

    Absence is reported, never repaired: nothing here writes a default score, an
    empty ladder or a synthetic provenance block, because a fabricated value is
    indistinguishable on the page from a measured one.
    """
    blob = blob if isinstance(blob, dict) else {}
    run = run if isinstance(run, dict) else {}
    arms = blob.get("arms") or {}
    # Same guard `sections` carries. Without it a card whose `arms` is a non-empty
    # non-dict raises here, `migrate` takes its except branch, and the card loses
    # its whole `compat` block -- the one thing that was meant to explain it.
    arms = arms if isinstance(arms, dict) else {}
    present = sections(arms)
    have_provenance = any(
        isinstance(a, dict) and (a.get("meta") or {}).get("provenance")
        for a in arms.values() if isinstance(a, dict))

    got = {
        "run.blockscore": bool(run.get("blockscore")),
        "run.head_to_head": bool(run.get("head_to_head")),
        "ladder": bool(blob.get("ladder")),
        "run.examples": bool(run.get("examples")),
        "meta.provenance": have_provenance,
        "geometry_scalars": "geometry_scalars" in present,
        "cost": "cost" in present,
        "fidelity": "fidelity" in present,
    }
    return [name for name in MISSING_CHECKS if not got[name]]


def _major(version: Any) -> int | None:
    """`"bench/2"` -> 2. None when the string is absent or not of that form."""
    if not isinstance(version, str) or "/" not in version:
        return None
    tail = version.rsplit("/", 1)[1]
    digits = re.match(r"\d+", tail.strip())
    return int(digits.group(0)) if digits else None


def migrate(blob: dict[str, Any], run_id: str, mtime: float) -> dict[str, Any]:
    """Normalise one parsed scorecard for reading. Never raises; idempotent.

    Returns a NEW top-level dict that shares `arms` -- and `context`, and every
    metric leaf -- with the input BY IDENTITY. `migrate(b, ...)["arms"] is
    b["arms"]`. See the module docstring: the input is the object sitting in
    `catalog._SCORECARDS`, and writing anything into it poisons every later read
    in the process. The only nested block replaced here is `run`, and it is
    replaced with a fresh dict rather than updated in place.

    Fills exactly three run-level defaults, all of them things a reader would
    otherwise have to compute in four places: `name` and `note` (empty strings,
    so the page can test truthiness without caring about the schema), and
    `started_at` (the recorded instant, else the directory stamp, else the file
    mtime -- see `dir_stamp`). It does NOT invent the rest of `bench/2`'s run
    keys: a missing `git_dirty` must read as unknown on the page, not as clean.

    Adds a top-level `compat` block -- version, what it was read as, the union of
    sections, what is missing, and whether it was written by a newer BlockGen.
    That block is derived, never written back to disk; this module and the lab
    never modify a run directory.
    """
    if not isinstance(blob, dict):
        print(f"[lab.cards] {run_id}: scorecard is {type(blob).__name__}, "
              f"not an object; ignoring", flush=True)
        return {}
    try:
        run = dict(blob.get("run") or {}) if isinstance(blob.get("run"), dict) else {}
        run.setdefault("name", "")
        run.setdefault("note", "")
        if not run.get("started_at"):
            run["started_at"] = dir_stamp(run_id) or _iso(mtime)

        version = blob.get("schema_version")
        major = _major(version)
        unsupported = None
        if major is not None and major > SCHEMA_MAJOR_SUPPORTED:
            unsupported = (f"written by a newer BlockGen ({version}); this page "
                           f"reads {READ_AS} and is showing whatever parses")

        out = dict(blob)            # shallow ON PURPOSE: arms shared by identity
        out["run"] = run            # the one block we own a copy of
        out["compat"] = {
            "version": version if isinstance(version, str) else "",
            "read_as": READ_AS,
            "sections": sections(blob.get("arms") or {}),
            "missing": missing(blob, run),
            "unsupported": unsupported,
        }
        return out
    except Exception as exc:                          # never break the page
        print(f"[lab.cards] {run_id}: could not normalise scorecard: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return dict(blob)           # still a new top-level dict, arms still shared


def run_label(blob: dict[str, Any], run_id: str) -> str:
    """A human-readable name for a run. DISPLAY ONLY -- never an id.

    **This string is free text and it is not unique.** Four cards on disk
    (`run_20260831_195547/202936/204112/225029_bench`) derive the identical label
    `native_oriented vs agentic_oneshot vs pick_n_place`, and two more collide as
    well. That is fine, and it must stay fine: nothing may ever key on this
    string -- not a path segment, not a URL parameter, not a dict key, not an
    `<option value>`. The id doing those three jobs is the directory name, and it
    stays the directory name (`?run=` bookmarks keep resolving). Making the label
    unique by construction would quietly turn it into a second id, and then
    renaming a run would break a bookmark.

    Three tiers, most honest first:

    1. `run.name`, the string the researcher typed at `--name`.
    2. the run's submission arms, `"a vs b vs c"`, `+N` past the third -- because
       an unnamed run is nearly always remembered as "the one comparing X and Y".
       Controls and baselines are excluded: a run with 3 submissions and 13
       calibration rungs is not "16 arms" to a human.
    3. the run id itself. The three all-control cards have no submissions at all,
       and inventing a name for them would be worse than showing the directory.
    """
    blob = blob if isinstance(blob, dict) else {}
    run = blob.get("run") or {}
    if isinstance(run, dict):
        name = str(run.get("name") or "").strip()
        if name:
            return name

    arms = blob.get("arms") or {}
    subs: list[str] = []
    if isinstance(arms, dict):
        for name, arm in arms.items():
            meta = arm.get("meta") if isinstance(arm, dict) else None
            if arm_kind(meta or {}) == "submission":
                subs.append(str(name))
    if subs:
        label = " vs ".join(subs[:LABEL_ARMS])
        if len(subs) > LABEL_ARMS:
            label += f" +{len(subs) - LABEL_ARMS}"
        return label
    return str(run_id)


def RECIPES() -> dict[str, str]:  # noqa: N802 -- a memoized constant, not a verb
    """One sentence per in-memory arm, `{arm_name: recipe}`, or `{}`.

    The read-time fallback for the cards that already exist. 13 of 16 rows on the
    user's real card are generated in-process and say "nothing on disk to show";
    `bench/2` writes the recipe into `meta.provenance`, but a writer-only fix
    helps none of the eleven cards on disk, and the prose that explains what
    `real@solidify` actually is has been sitting in a docstring the whole time.
    `catalog.arm_recipe` prefers the card's own recipe and falls back to this.

    Imported from the eval package rather than copied, so a new calibration rung
    cannot ship with prose in one place and none in the other; `catalog` already
    imports `blockgen.eval.bench.splits`, so the direction is established. Lazy
    and fail-soft: the lab must start in a checkout where the eval extras are not
    installed. Memoized because the answer cannot change without restarting the
    process anyway (Python will not re-execute an imported module).
    """
    global _RECIPES_MEMO
    if _RECIPES_MEMO is not None:
        return _RECIPES_MEMO
    merged: dict[str, str] = {}
    try:
        from blockgen.eval.bench import baselines, fast
        for source in (getattr(fast, "CONTROL_RECIPES", None),
                       getattr(baselines, "RECIPES", None)):
            if isinstance(source, dict):
                merged.update({str(k): str(v) for k, v in source.items()})
    except Exception as exc:                          # never break the page
        print(f"[lab.cards] recipes unavailable: {type(exc).__name__}: {exc}",
              flush=True)
        merged = {}
    _RECIPES_MEMO = merged
    return merged
