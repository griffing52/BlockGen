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
    return str(name)


def _appearance(card: dict, run_id: str, name: str, meta: dict,
                row: dict, examples: Dict[str, List[str]]) -> dict:
    """One model's showing in one run -- the unit `history` is made of."""
    score = row.get("score")
    return {
        "run": run_id,
        "run_label": card.get("label") or run_id,
        "started": card.get("started") or card.get("when") or "",
        "arm": name,
        "track": meta.get("track") or "",
        "n": meta.get("n"),
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

    Lower score wins; a scored appearance always beats a disqualified one,
    because a DQ is not a position. Genuine ties go to the NEWER run -- it is the
    one whose card carries provenance and example builds, and "the most recent
    time this model achieved its best score" is a defensible thing for a row to
    mean, where "whichever float sorted lower" is not.
    """
    a_ok = a["status"] != "DQ" and isinstance(a["score"], (int, float))
    b_ok = b["status"] != "DQ" and isinstance(b["score"], (int, float))
    if a_ok and b_ok:
        if _tied(float(a["score"]), float(b["score"])):
            return a if (a["started"] or "") >= (b["started"] or "") else b
        return a if a["score"] < b["score"] else b
    if a_ok != b_ok:
        return a if a_ok else b
    return a if (a["started"] or "") >= (b["started"] or "") else b


def _sort_key(row: dict) -> Tuple[int, float, str]:
    """Ranked ascending by score; DQ and unscored sink below every number, in
    name order, because they are unordered among themselves."""
    if row["status"] == "DQ" or not isinstance(row["score"], (int, float)):
        return (1, 0.0, row["model"])
    return (0, float(row["score"]), row["model"])


def board(protocol_id: Optional[str] = None) -> Dict[str, Any]:
    """The whole board for one protocol. Never raises; empty on a fresh clone."""
    pro = proto.get(protocol_id or proto.DEFAULT_ID)
    if pro is None:
        return {"protocol": None, "ranked": [], "provisional": [], "skipped": [],
                "n_runs": 0, "n_matching": 0,
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
        why = proto.match_arm({"n": best.get("n")}, pro)
        out["unranked_reason"] = why[0] if why else ""
        (provisional if why else ranked).append(out)

    ranked.sort(key=_sort_key)
    provisional.sort(key=_sort_key)
    return {
        "protocol": pro.to_json(),
        "protocols": [p.to_json() for p in proto.OFFICIAL],
        "ranked": ranked,
        "provisional": provisional,
        "skipped": sorted(skipped, key=lambda s: s.get("started") or "", reverse=True),
        "n_runs": n_runs,
        "n_matching": n_matching,
        "select": SELECT,
    }
