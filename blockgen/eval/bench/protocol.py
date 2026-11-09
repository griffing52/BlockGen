"""What makes two runs comparable, and therefore what "official" means.

BlockScore is expressed in *real-sample spreads*: every pillar is
`(value - real) / sd(real)`, where `real` is the `real_test` control arm scored
**in that same run**, on that run's corpus, that run's split, at that run's `n`.
That calibration is what makes the number interpretable without hand-set weights
-- and it is also what makes a naive cross-run leaderboard wrong. A 2.9 measured
against 399 held-out `houses_32` builds and a 3.4 measured against 463
`houses_48` builds are not two positions on one ladder; they are readings from
two different rulers that happen to share a unit name.

So the leaderboard does not rank runs. It ranks runs *that ran the same
protocol*, and this module is where a protocol is written down.

**Why a pinned constant rather than a `--official` flag.** A flag records an
intention; a protocol records a fact, and the fact is checkable after the run
from the card alone. Nothing stops someone passing `--official` to a 16-build
smoke run on the wrong corpus, and once one such row is on the board every other
row's meaning is gone. `matches()` re-derives the verdict from `context` and
`run` every time the page is loaded, so a run cannot be grandfathered in by the
version of the code that happened to score it. This mirrors what
`blockgen.eval.bench.splits` already does for the split itself: the split is
pinned, fingerprinted, and committed, because it "defines every number in the
eval suite".

**Ambiguity worth recording.** `min_n` is checked per *arm*, not per run, because
`--n` caps controls and baselines but not file-backed arms (`runner.main` applies
it at two of three sites), so a single run legitimately holds a 128-build control
beside a 64-build submission. The run can therefore be official while one of its
arms is unranked, and the leaderboard says which -- an arm silently vanishing
because it was 8 builds short is the failure mode this exists to prevent.

**The slate.** A protocol also pins what gets rendered, because the leaderboard's
whole visual claim is that column *i* means the same thing on every row. Where an
arm has prompts, the slate is those prompts and the claim is literally true. For
unconditional arms there is no correspondence to align on -- you cannot ask two
unconditional models for "the same" build -- so the slate degrades to *k* slots
drawn under a pinned seed, which buys reproducibility and stability across
re-runs but NOT cross-model correspondence. `Slate.aligned` says which of the two
you are looking at, and the page is required to label it, because a strip that
implies a comparison it cannot support is worse than no strip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The arm BlockScore is calibrated against. Every pillar is expressed in
#: multiples of THIS arm's spread, so its sample size decides the unit: measured
#: on the cards on disk, one model scored 87.14 against a 128-build `real_test`
#: and 19.05 against a 32-build one. Same model, same corpus, same split, same
#: tier -- a 4.5x swing from the ruler alone. That is why `min_n` is checked
#: against the control and not only against the submission.
#:
#: (A seventh hard-coded `"real_test"`, after composite.leaderboard's default,
#: composite.validate's C1/C2, human_study_analyze.ALIASES, bench_report and the
#: two lab pages. Namespacing arm names would move all seven together.)
CONTROL_ARM = "real_test"

#: Tiers that produce the rendered `appearance` pillar. `fast` never does, so a
#: protocol asking for appearance cannot be satisfied by a fast-tier run no
#: matter how many builds it scored.
FULL_TIERS = ("full", "both")


@dataclass(frozen=True)
class Slate:
    """What every row of the leaderboard renders, so columns line up.

    `aligned` is the honest bit. True means every arm rendered the same prompts
    and column *i* is one comparison. False means the columns are only slots:
    same count, same seed, same order, different builds -- useful for eyeballing
    a model's output, useless for comparing two models cell by cell.
    """

    k: int = 8
    prompts: Tuple[str, ...] = ()

    @property
    def aligned(self) -> bool:
        return bool(self.prompts)

    def to_json(self) -> Dict[str, Any]:
        return {"k": self.k, "prompts": list(self.prompts), "aligned": self.aligned}


@dataclass(frozen=True)
class Protocol:
    """One pinned evaluation. Runs matching it are rankable against each other.

    Every field is a thing that changes what BlockScore *means*, which is the
    test for whether something belongs here. `seed` is included for that reason
    even though it moves scores only slightly: two runs that disagree on the
    split seed are scoring against different held-out sets.
    """

    id: str
    corpus: str
    split_key: str
    tier: str = "both"
    min_n: int = 128
    #: The held-out real sample the distribution distances are measured against.
    #: Separate from `min_n`: `--n` caps the arms, `--n-ref` caps the reference,
    #: and a run can legitimately have a large control and a starved reference.
    min_ref: int = 256
    seed: int = 0
    slate: Slate = field(default_factory=Slate)
    note: str = ""
    #: The geometric descriptor `geom_kid` and the scalar panel were measured
    #: with, as `ladder.geom_view_key()` spells it. Part of the protocol because
    #: a run scored with a different descriptor produces a BlockScore in a
    #: different unit of the same name. Empty means "not pinned".
    descriptor: str = ""
    #: `report.train_split` values that count as "trained clean" for this
    #: protocol (see `match_arm`). Defaults to the protocol's own split.
    clean_train_splits: Tuple[str, ...] = ("canonical", "protocol")

    @property
    def needs_full_tier(self) -> bool:
        return self.tier in FULL_TIERS

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "corpus": self.corpus, "split_key": self.split_key,
                "tier": self.tier, "min_n": self.min_n,
                "min_ref": self.min_ref, "seed": self.seed,
                "slate": self.slate.to_json(), "note": self.note,
                "descriptor": self.descriptor,
                "clean_train_splits": list(self.clean_train_splits)}


#: The protocol the leaderboard ranks. One entry, deliberately: a leaderboard
#: with two protocols is two leaderboards, and the moment they are drawn in one
#: table someone reads across them. Add a second only alongside the UI that keeps
#: them apart.
#:
#: `houses32-v1` is the protocol every T23/T25 number in results.md was measured
#: under. Since 2026-10-23 it pins the geometric descriptor as well: the panel
#: gained the player's room (standing share, storeys, headroom) and BlockScore a
#: `traversal` pillar, so a card scored before that carries a BlockScore in a
#: different unit and is unranked here with that reason until re-scored. The
#: re-score of every ranked arm is `run_20261023_011035_bench_traversal_pillar_
#: leaderboard_re-score` (results.md T64).
OFFICIAL: Tuple[Protocol, ...] = (
    Protocol(
        id="houses32-v1",
        corpus="houses_32",
        split_key="houses_32.s0.70-15-15.v1",
        tier="both",
        min_n=128,
        seed=0,
        slate=Slate(k=8),
        descriptor="patch2_orb55_t5_h8_s12",
        note="The canonical house benchmark: 32^3 GrabCraft houses, the committed "
             "70/15/15 split, both tiers so the appearance pillar is measured, and "
             "at least 128 builds per arm so the bootstrap intervals mean something.",
    ),
    # The CVPR 2027 metric paper's leaderboard of record
    # (run_20261103_090315_bench_n_256_leaderboard_of_record). Same corpus and
    # split as houses32-v1, but the pre-2026-10-23 descriptor (77-d panel,
    # five-pillar BlockScore without `traversal`, `--descriptor v1`) so every
    # number matches the 0925 protocol the paper's ladder and gates were
    # validated under, and n >= 256 per arm. Its BlockScore is in a different
    # unit from houses32-v1's; the lab never ranks across the two.
    # `external` admits published generators trained on data disjoint from the
    # corpus (Dream-Cubed: procedural 1.12 worlds, no GrabCraft/3D-Craft/text2mc).
    Protocol(
        id="houses32-paper-v1",
        corpus="houses_32",
        split_key="houses_32.s0.70-15-15.v1",
        tier="both",
        min_n=256,
        seed=0,
        slate=Slate(k=8),
        descriptor="patch2_orb55_t5_h8",
        clean_train_splits=("canonical", "protocol", "external"),
        note="The metric paper's board: houses32-v1's corpus and split, the 77-d "
             "geometric descriptor and five-pillar BlockScore it was validated with, "
             "n >= 256 per arm, and published external generators admitted.",
    ),
)

#: The descriptor every card written before 2026-10-23 was scored with. Those
#: cards carry no `geom_descriptor` in their context and no `walk_*` scalars in
#: any arm, which is how `run_descriptor` tells them apart.
LEGACY_DESCRIPTOR = "patch2_orb55_t5_h8"


def run_descriptor(context: Dict[str, Any], arms: Optional[Dict[str, Any]]) -> str:
    """Which geometric descriptor a card was scored with.

    The context stamp (`geom_descriptor`) is authoritative. A card without it is
    read by presence, never by date or version string: an arm whose
    `geometry_scalars` carry the room scalars was scored by the current
    descriptor; otherwise the legacy one.
    """
    stamp = (context or {}).get("geom_descriptor")
    if stamp:
        return str(stamp)
    if isinstance(arms, dict):
        for arm in arms.values():
            scal = (arm or {}).get("geometry_scalars") if isinstance(arm, dict) else None
            if isinstance(scal, dict) and "walk_standing" in scal:
                return BY_ID[DEFAULT_ID].descriptor if DEFAULT_ID in BY_ID else LEGACY_DESCRIPTOR
    return LEGACY_DESCRIPTOR

#: `id -> Protocol`, for the lab and the runner.
BY_ID: Dict[str, Protocol] = {p.id: p for p in OFFICIAL}

DEFAULT_ID = OFFICIAL[0].id


def get(protocol_id: Optional[str]) -> Optional[Protocol]:
    """A protocol by id, or None. Never raises: an unknown id is a page that
    says "no such protocol", not a 500."""
    if not protocol_id:
        return None
    return BY_ID.get(str(protocol_id))


def _tier_ok(tier: str, protocol: Protocol) -> bool:
    return (not protocol.needs_full_tier) or str(tier) in FULL_TIERS


def match_run(context: Dict[str, Any], run: Dict[str, Any],
              protocol: Protocol,
              arms: Optional[Dict[str, Any]] = None) -> List[str]:
    """Why this run does NOT satisfy `protocol`. Empty list means it does.

    Reasons are phrased for a human reading a leaderboard, not for a log: they
    are shown verbatim on the run's row, because "unranked" with no reason is
    the thing that makes people stop trusting a board.
    """
    context = context if isinstance(context, dict) else {}
    run = run if isinstance(run, dict) else {}
    why: List[str] = []

    corpus = str(context.get("corpus") or "")
    if corpus != protocol.corpus:
        why.append(f"corpus {corpus or '?'} (protocol wants {protocol.corpus})")

    split = str(context.get("split_key") or "")
    if split and split != protocol.split_key:
        why.append(f"split {split} (protocol wants {protocol.split_key})")
    elif not split:
        why.append("no split recorded")

    tier = str(run.get("tier") or "")
    if not _tier_ok(tier, protocol):
        why.append(f"tier {tier or '?'} measures no appearance pillar "
                   f"(protocol wants {protocol.tier})")

    seed = context.get("seed")
    if seed is not None and int(seed) != protocol.seed:
        why.append(f"split seed {seed} (protocol wants {protocol.seed})")

    if protocol.descriptor:
        desc = run_descriptor(context, arms)
        if desc != protocol.descriptor:
            why.append(f"geometric descriptor {desc} (protocol wants "
                       f"{protocol.descriptor}; BlockScore is in a different unit, "
                       f"re-score with the current bench)")

    # The ruler itself. Checked here rather than in `match_arm` because a small
    # calibration arm invalidates every score in the run, not one row.
    if isinstance(arms, dict) and arms:
        control = (arms.get(CONTROL_ARM) or {}).get("meta") or {}
        c_n = control.get("n")
        if not isinstance(c_n, int):
            why.append(f"no {CONTROL_ARM} control (nothing calibrates the scores)")
        elif c_n < protocol.min_n:
            why.append(f"{CONTROL_ARM} calibrated on {c_n} builds "
                       f"(protocol wants at least {protocol.min_n}); "
                       f"BlockScore's unit is this arm's spread")

    ref_used = context.get("n_ref_used")
    if isinstance(ref_used, int) and ref_used < protocol.min_ref:
        why.append(f"real reference {ref_used} builds "
                   f"(protocol wants at least {protocol.min_ref})")

    return why


#: Values of an arm's `report.train_split` that mean "trained on the protocol's own
#: train split". Anything else -- including a missing value -- is a model whose
#: training provenance the board cannot vouch for.
CLEAN_TRAIN_SPLITS = ("canonical", "protocol")


def match_arm(meta: Dict[str, Any], protocol: Protocol) -> List[str]:
    """Why this arm is not rankable under `protocol`, even in a matching run.

    Separate from `match_run` because `--n` caps controls and baselines but not
    file-backed arms, so one run holds arms of different sizes.

    **Training provenance is checked here too, and it is the check that was
    missing.** `match_run` validates the *evaluation* -- corpus, split, tier, n --
    which is necessary and turned out not to be sufficient: a pair of LLM arms
    trained on Track D's own 90/10 shuffle (which predates the pinned split, and
    puts 53 canonical-*test* builds in the training set) scored on the canonical
    split at n=128 and were ranked, with `train_split: "trackd"` sitting in the
    provenance the page already displays. A board whose whole claim is
    comparability cannot rank a model that trained on the test set, so an arm now
    has to *say* it trained on the protocol's split to be ranked. Silence is not
    consent: an arm with no recorded `train_split` is unranked rather than
    assumed clean, because most contaminated arms will not know they are.
    """
    meta = meta if isinstance(meta, dict) else {}
    why: List[str] = []
    n = meta.get("n")
    if not isinstance(n, int) or n < protocol.min_n:
        why.append(f"n={n if isinstance(n, int) else '?'} "
                   f"(protocol wants at least {protocol.min_n})")

    report = meta.get("report") if isinstance(meta.get("report"), dict) else {}
    if report:                      # only generated arms carry one; controls do not
        train_split = str(report.get("train_split") or "")
        if not train_split:
            why.append("training split not recorded (cannot rule out training on the "
                       "protocol's test builds)")
        # First word only, so an annotated value such as
        # "external (Dream-Cubed ...; no GrabCraft/3D-Craft/text2mc)" matches "external".
        elif train_split.split(" ")[0] not in protocol.clean_train_splits:
            why.append(f"trained on the {train_split!r} split, not {protocol.split_key} "
                       f"(its training set overlaps this protocol's test builds)")
    return why


def verdict(context: Dict[str, Any], run: Dict[str, Any],
            protocol: Optional[Protocol] = None,
            arms: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The block the runner writes into `run.protocol` and the lab reads back.

    Recorded on the card as a convenience and a record of what the code believed
    at scoring time -- but the lab re-derives it rather than trusting it, so a
    protocol tightened after a run was scored takes effect on the next page load
    rather than requiring eleven cards to be rewritten.
    """
    protocol = protocol or get(DEFAULT_ID)
    if protocol is None:                      # no protocols defined at all
        return {"id": None, "official": False, "reasons": ["no protocol defined"]}
    why = match_run(context, run, protocol, arms)
    return {"id": protocol.id, "official": not why, "reasons": why}
