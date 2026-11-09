"""The cross-run board: one row per model, over every run of one protocol.

`leaderboard.html` used to rank the *arms inside one card*, which is what the
bench measures and not what anyone means by a leaderboard. This module answers
the other question -- "which model is best" -- by reading every scorecard on
disk, keeping the runs that satisfy a pinned `protocol`, and collapsing each
model's appearances down to one row.

Three things this refuses to do, each because the alternative produces a number
that looks authoritative and is not:

**It never ranks across protocols.** BlockScore's unit is a spread of the
`real_test` arm scored in the same run, so a `houses_48` score and a `houses_32`
score are readings from different rulers. Runs that do not match are not
silently dropped either -- `skipped` carries them with the reason, so the board
can say "4 runs excluded" rather than quietly being a board about half the data.

**It separates "ranked" from "provisional" rather than lowering the bar.** Every
submission on disk today is below `houses32-v1`'s `min_n` of 128 (n=64, 16 and
12), so a board that ranked only qualifying arms would be empty, and a board that
dropped `min_n` to fit them would be a board whose confidence intervals mean
nothing. Both tiers are returned, ordered identically, and the caller is required
to keep them visually apart. The provisional row carries its own reason.

**It does not trust `run.protocol`.** The runner records its verdict at scoring
time, but this re-derives it from `context` and `run` on every call. Tightening a
protocol therefore takes effect on the next page load instead of requiring every
card on disk to be rewritten -- and cards written before protocols existed get a
verdict anyway.

Model identity is `provenance.model` where the card has it and the arm name
otherwise, which is the same join `api._model_cards` uses and the same convention
`sample_to_npz.py` writes: an arm is named after the model it ran. Two runs of
one model therefore collapse to one row, and `history` keeps the rest.

**It groups by n and says when a view mixes them.** BlockScore is comparable
only at equal n: a pillar is a bootstrap distance whose spread shrinks with the
sample, so an n=399 arm and an n=128 arm are not two readings on one ruler even
inside one protocol. Every row carries its `n`, `n_groups` counts the arms at
each distinct n, `default_n` names the group a page should open on (the largest
n with at least two arms), and `board(n=...)` filters to one group while
reporting how many rows that hid. Nothing is unranked by this: a provisional arm
stays provisional for the protocol's reasons, and grouping only decides which
rows are shown side by side.

**It surfaces what an arm trained on.** A generated arm's `provenance.report`
carries `model`, `checkpoint`, `train_split` and `trained_on_signals` (the
rewards or preference signals a post-training stage optimised). The first three
are lifted onto the row; `trained_on_signals` is both lifted and flagged, the
way an `oracle_train` arm is flagged today, because a model that trained against
a reward the bench also measures has to be read with that in mind.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from blockgen.eval.bench import protocol as proto

from tools.lab import cards, catalog

#: How a model's several official runs collapse to one row. "best" is the lowest
#: finite BlockScore; a model whose every appearance was disqualified keeps the
#: DQ (a DQ is a refusal to rank, so it can never be beaten by a worse number).
SELECT = "best"


def _rows(card: dict) -> Dict[str, dict]:
    """`arm name -> BlockScore row`, from the card's own leaderboard block."""
    out: Dict[str, dict] = {}
    for row in card.get("leaderboard") or []:
        if isinstance(row, dict) and row.get("arm"):
            out[str(row["arm"])] = row
    return out


def model_id(name: str, meta: dict) -> str:
    """The identity two runs of one model share.

    `provenance.model` first: it is what the sampler recorded, so it survives an
    arm being given a different `--arms` label in a later run. The arm name is
    the fallback, and is what every card written before `bench/2` has.
    """
    prov = meta.get("provenance") if isinstance(meta, dict) else None
    if isinstance(prov, dict):
        model = prov.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        report = prov.get("report")
        model = report.get("model") if isinstance(report, dict) else None
        # `report.model` is a dict (arch, preset, widths) on a post-training
        # arm; that describes a model and does not name one, so only a string
        # counts as an identity.
        if isinstance(model, str) and model.strip():
            return model.strip()
    return str(name)


def model_label(model: Any) -> str:
    """A string for the `model` field however it was written: the name itself,
    or `arch/preset` (then `d_model x layers`) when the writer recorded the
    architecture dict instead of a name."""
    if isinstance(model, str):
        return model.strip()
    if isinstance(model, dict):
        head = "/".join(str(model[k]) for k in ("arch", "preset")
                        if model.get(k) not in (None, ""))
        dims = "x".join(str(model[k]) for k in ("d_model", "layers")
                        if model.get(k) not in (None, ""))
        return " ".join(part for part in (head, dims) if part)
    return "" if model is None else str(model)


def training(meta: dict) -> dict:
    """What a generated arm says about its own training, from `provenance`.

    Reads the top-level provenance block first (`sample_to_npz.py` writes
    `model` and `checkpoint` there) and the nested sample `report` second (the
    trainer and the post-training stages write `model`, `checkpoint`,
    `train_split` and `trained_on_signals` there). Every key is present, None or
    `[]` when unrecorded, so a page never has to probe two shapes.
    """
    prov = meta.get("provenance") if isinstance(meta, dict) else None
    prov = prov if isinstance(prov, dict) else {}
    report = prov.get("report")
    report = report if isinstance(report, dict) else {}

    def first(key: str) -> Any:
        for block in (prov, report):
            value = block.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    model = first("model")
    detail = next((b.get("model") for b in (prov, report) if isinstance(b.get("model"), dict)), None)
    signals = first("trained_on_signals")
    if isinstance(signals, str):
        signals = [signals]
    signals = [str(x) for x in signals] if isinstance(signals, (list, tuple)) else []
    ckpt, step = checkpoint_file(first("checkpoint"))
    return {
        "model": model_label(model) or None,
        "model_detail": detail,
        "checkpoint": ckpt,
        "checkpoint_step": step,
        "train_split": first("train_split"),
        "trained_on_signals": signals,
        "training_run": report.get("run") or None,
    }


def checkpoint_file(ckpt: Any) -> Tuple[Optional[str], Optional[int]]:
    """`(path, step)` from either shape a writer records a checkpoint in.

    `sample_to_npz.py` writes the path; the trainer's sample report
    (`blockgen/train/sample.py`) writes `{"file", "step"}`. A page gets one
    string either way, so a dict never reaches `String()` and renders as
    "[object Object]", and the `run_` segment stays findable.
    """
    if isinstance(ckpt, dict):
        step = ckpt.get("step")
        path = ckpt.get("file") or ckpt.get("path")
        return ((str(path) if path not in (None, "") else None),
                (step if isinstance(step, int) and not isinstance(step, bool) else None))
    return (str(ckpt) if ckpt not in (None, "", [], {}) else None), None


def _appearance(card: dict, run_id: str, name: str, meta: dict,
                row: dict, examples: Dict[str, List[str]]) -> dict:
    """One model's showing in one run -- the unit `history` is made of."""
    score = row.get("score")
    trained = training(meta)
    train_run = card.get("train_run") if isinstance(card.get("train_run"), dict) else None
    return {
        "run": run_id,
        "run_dir": card.get("run_dir") or run_id.split("/", 1)[0],
        "run_label": card.get("label") or run_id,
        "train_run_name": (train_run or {}).get("name"),
        "started": card.get("started") or card.get("when") or "",
        "arm": name,
        "track": meta.get("track") or "",
        "n": meta.get("n"),
        # `model_label`, not `model`: the row's `model` is its identity (the key
        # two runs of one checkpoint share), and `board` writes it last. This is
        # the model as the WRITER described it, which for a post-training arm is
        # an architecture dict rendered as `arch/preset`.
        "model_label": trained["model"],
        "model_detail": trained["model_detail"],
        "checkpoint": trained["checkpoint"],
        "checkpoint_step": trained["checkpoint_step"],
        "train_split": trained["train_split"],
        "trained_on_signals": trained["trained_on_signals"],
        "training_run": trained["training_run"],
        "score": score if isinstance(score, (int, float)) else None,
        "status": row.get("status") or "",
        "worst_pillar": row.get("worst_pillar") or "",
        "pillars": row.get("pillars") or {},
        "disqualified": row.get("disqualified"),
        "provenance": meta.get("provenance") or {},
        "source_run_id": meta.get("source_run_id"),
        "structures_sha": meta.get("structures_sha"),
        "examples": examples.get(name) or [],
    }


#: Two BlockScores this close are the same score. Not cosmetic: re-scoring one
#: model under one protocol in two runs reproduced 87.14427315448478 and
#: 87.14427315440719 -- agreement to ten significant figures, which is the
#: protocol working. Comparing those with `<` let 8e-11 of float noise decide
#: which run represented the model, and it chose the older one, whose card
#: predates example builds. The row lost its renders to arithmetic dust.
SCORE_TIE = 1e-9


def _tied(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=SCORE_TIE, abs_tol=SCORE_TIE)


def _better(a: dict, b: dict) -> dict:
    """The appearance that represents the model.

    Between appearances at different n, the one at ``BOARD_N`` (else the larger n)
    wins, because their scores are not comparable. At equal n the lower score wins;
    a scored appearance always beats a disqualified one,
    because a DQ is not a position. Genuine ties go to the NEWER run -- it is the
    one whose card carries provenance and example builds, and "the most recent
    time this model achieved its best score" is a defensible thing for a row to
    mean, where "whichever float sorted lower" is not.
    """
    a_ok = a["status"] != "DQ" and isinstance(a["score"], (int, float))
    b_ok = b["status"] != "DQ" and isinstance(b["score"], (int, float))
    if a_ok and b_ok and a.get("n") != b.get("n"):
        # Scores at different n are not ordered (a pillar's spread shrinks with n), so the
        # board's n wins, then the larger n, never the lower number.
        return a if _n_pref(a) >= _n_pref(b) else b
    if a_ok and b_ok:
        if _tied(float(a["score"]), float(b["score"])):
            return a if (a["started"] or "") >= (b["started"] or "") else b
        return a if a["score"] < b["score"] else b
    if a_ok != b_ok:
        return a if a_ok else b
    return a if (a["started"] or "") >= (b["started"] or "") else b


#: The sample size every card is drawn at since 2026-11-05; an arm's row prefers its appearance at this n.
BOARD_N = 256


def _n_pref(row: dict) -> Tuple[bool, int]:
    n = row.get("n") if isinstance(row.get("n"), int) else 0
    return (n == BOARD_N, n)


def _sort_key(row: dict) -> Tuple[int, float, str]:
    """Ranked ascending by score; DQ and unscored sink below every number, in
    name order, because they are unordered among themselves."""
    if row["status"] == "DQ" or not isinstance(row["score"], (int, float)):
        return (1, 0.0, row["model"])
    return (0, float(row["score"]), row["model"])


def n_groups(rows: List[dict]) -> List[dict]:
    """`[{"n", "arms", "ranked", "provisional"}, ...]`, largest n first, over the
    rows of both tables. `n` is None for a row whose card never recorded one;
    that group sorts last so the page can name it rather than lose it."""
    counts: Dict[Optional[int], dict] = {}
    for row in rows:
        n = row.get("n") if isinstance(row.get("n"), int) else None
        slot = counts.setdefault(n, {"n": n, "arms": 0, "ranked": 0, "provisional": 0})
        slot["arms"] += 1
        slot["ranked" if not row.get("unranked_reason") else "provisional"] += 1
    return sorted(counts.values(), key=lambda g: (g["n"] is None, -(g["n"] or 0)))


def default_n(groups: List[dict]) -> Optional[int]:
    """The group a page opens on: the largest n holding at least two arms,
    because one arm is not a ranking; else the largest n at all; else None."""
    sized = [g for g in groups if isinstance(g.get("n"), int)]
    for g in sized:                     # already largest first
        if g["arms"] >= 2:
            return g["n"]
    return sized[0]["n"] if sized else None


def board(protocol_id: Optional[str] = None, n: Optional[int] = None) -> Dict[str, Any]:
    """The whole board for one protocol. Never raises; empty on a fresh clone.

    `n` restricts `ranked` and `provisional` to the arms scored at that sample
    size; `n_hidden` then says how many rows that put out of view, and
    `n_groups` still counts every arm, so the filter can never pass for the
    whole. Without it both tables carry every n and `mixed_n` says whether a
    reader is looking at more than one.
    """
    pro = proto.get(protocol_id or proto.DEFAULT_ID)
    if pro is None:
        return {"protocol": None, "ranked": [], "provisional": [], "skipped": [],
                "n_runs": 0, "n_matching": 0, "n_groups": [], "default_n": None,
                "n_filter": n, "n_hidden": 0, "mixed_n": False,
                "error": f"no protocol named {protocol_id!r}"}

    by_model: Dict[str, dict] = {}
    skipped: List[dict] = []
    n_runs = n_matching = 0

    for entry in catalog.list_scorecards():
        n_runs += 1
        run_id = entry.get("run") or ""
        try:
            card = catalog.load_scorecard(run_id)
        except Exception as exc:                       # a card that will not open
            skipped.append({"run": run_id, "label": run_id,
                            "reasons": [f"unreadable: {type(exc).__name__}"]})
            continue

        why = proto.match_run(card.get("context") or {}, card.get("run") or {},
                              pro, card.get("arms") or {})
        if why:
            skipped.append({"run": run_id, "label": card.get("label") or run_id,
                            "started": card.get("started") or "", "reasons": why})
            continue
        n_matching += 1

        rows = _rows(card)
        try:
            examples = catalog.example_build_ids(
                catalog.OUTPUTS_ROOT / run_id, card)
        except Exception:
            examples = {}

        for name, arm in (card.get("arms") or {}).items():
            meta = (arm or {}).get("meta") or {}
            if cards.arm_kind(meta) != "submission":
                continue                                # the board is models only
            row = rows.get(name)
            if not isinstance(row, dict):
                continue                                # scored, but never ranked
            seen = _appearance(card, run_id, name, meta, row, examples)
            key = model_id(name, meta)
            slot = by_model.setdefault(
                key, {"model": key, "best": seen, "history": []})
            slot["history"].append(seen)
            slot["best"] = _better(slot["best"], seen)

    ranked: List[dict] = []
    provisional: List[dict] = []
    for slot in by_model.values():
        best = slot["best"]
        out = dict(best)
        out["model"] = slot["model"]
        out["appearances"] = len(slot["history"])
        out["history"] = sorted(slot["history"],
                                key=lambda h: h.get("started") or "", reverse=True)
        out["recipe"] = catalog.arm_recipe(best["arm"], {"provenance": best["provenance"]})
        # The provenance block carries the arm's `report`, which is where a
        # generated arm records the split it TRAINED on. Passing only `n` here is
        # what let a pair of arms trained on the protocol's own test builds sit on
        # the ranked board with the evidence visible one field away.
        why = proto.match_arm(
            {"n": best.get("n"), "report": (best.get("provenance") or {}).get("report")}, pro)
        out["unranked_reason"] = "; ".join(why)
        (provisional if why else ranked).append(out)

    ranked.sort(key=_sort_key)
    provisional.sort(key=_sort_key)
    groups = n_groups(ranked + provisional)
    hidden = 0
    if n is not None:
        keep = lambda row: row.get("n") == n     # noqa: E731
        hidden = sum(1 for row in ranked + provisional if not keep(row))
        ranked = [row for row in ranked if keep(row)]
        provisional = [row for row in provisional if keep(row)]
    shown_ns = {row.get("n") for row in ranked + provisional}
    return {
        "protocol": pro.to_json(),
        "protocols": [p.to_json() for p in proto.OFFICIAL],
        "ranked": ranked,
        "provisional": provisional,
        "skipped": sorted(skipped, key=lambda s: s.get("started") or "", reverse=True),
        "n_runs": n_runs,
        "n_matching": n_matching,
        "select": SELECT,
        "n_groups": groups,
        "default_n": default_n(groups),
        "n_filter": n,
        "n_hidden": hidden,
        "mixed_n": len(shown_ns) > 1,
    }
