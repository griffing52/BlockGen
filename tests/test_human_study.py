"""Invariants of the 2AFC instrument.

These are not tests of analysis code -- `tests/test_human_study_analyze.py`
covers that -- they are tests of the *experiment*. Each one corresponds to a way
the study could run to completion, produce data, and be unanalysable or invalid
under its own pre-registered protocol. That failure mode is expensive: it is
only discoverable after the raters have gone home.

Run before every collection round:

    pytest tests/test_human_study.py -q
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts import human_study_export as hse


# --- the experiment's own arithmetic ---------------------------------------

def _trials(n_arms: int = 8, per_pair: int = 6, seed: int = 0):
    """A trial pool over synthetic arms, including the real/uniform catch pair."""
    names = ["real", "uniform_random", "real@solidify", "patchwork",
             "gabled_house", "canon_oriented", "llm_control", "shell_box"]
    index = {n: list(range(i * 4, i * 4 + 4)) for i, n in enumerate(names[:n_arms])}
    rng = np.random.default_rng(seed)
    return hse.make_trials(index, rng, per_pair), index


def test_priority_pairs_get_more_trials_than_ordinary_ones():
    """Rater time is scarce; the decisive pairs must be oversampled."""
    trials, _ = _trials()
    per = collections.Counter(
        frozenset({t["left_arm"], t["right_arm"]}) for t in trials)
    central = per[frozenset({"real", "real@solidify"})]
    ordinary = per[frozenset({"shell_box", "patchwork"})]
    assert central > ordinary, (central, ordinary)


def test_every_session_has_the_full_catch_quota():
    """The bug this pins: a uniformly sampled session gets a *random* number of
    catch trials, often zero, and the pre-registered exclusion rule ("drop a
    session below 75% on catch trials") is then uncomputable. The quota is
    fixed, not sampled."""
    trials, _ = _trials()
    rng = np.random.default_rng(0)
    sessions = hse.build_sessions(trials, rng, n_sessions=24,
                                  session_trials=40, session_catch=4)
    catch = {t["id"] for t in trials if t["catch"]}
    counts = [sum(1 for i in s if i in catch) for s in sessions]
    assert set(counts) == {4}, sorted(set(counts))


def test_sessions_are_the_requested_length_and_have_no_repeats():
    trials, _ = _trials()
    sessions = hse.build_sessions(trials, np.random.default_rng(0),
                                  n_sessions=8, session_trials=40,
                                  session_catch=4)
    for s in sessions:
        assert len(s) == 40
        assert len(set(s)) == 40, "a rater must not see the same trial twice"


def test_catch_trials_are_not_all_at_the_start():
    """Front-loaded catch trials measure attention at minute one, which is not
    the attention the exclusion rule is trying to detect."""
    trials, _ = _trials()
    sessions = hse.build_sessions(trials, np.random.default_rng(0),
                                  n_sessions=24, session_trials=40,
                                  session_catch=4)
    catch = {t["id"] for t in trials if t["catch"]}
    late = sum(any(i in catch for i in s[20:]) for s in sessions)
    assert late >= len(sessions) * 0.5, (
        f"only {late}/{len(sessions)} sessions have a catch trial in the "
        "second half")


def test_too_few_catch_trials_fails_loudly_rather_than_silently():
    trials, _ = _trials()
    with pytest.raises(SystemExit, match="catch"):
        hse.build_sessions(trials, np.random.default_rng(0), n_sessions=2,
                           session_trials=40, session_catch=999)


def test_sides_are_balanced_so_a_side_bias_is_not_a_preference():
    trials, _ = _trials(per_pair=40)
    left = collections.Counter(t["left_arm"] for t in trials)
    right = collections.Counter(t["right_arm"] for t in trials)
    for arm in left:
        n = left[arm] + right[arm]
        assert abs(left[arm] - right[arm]) < 0.25 * n, (
            f"{arm} appears on the left {left[arm]}/{n} times")


# --- the generated artifact, when one exists -------------------------------

STUDY = Path(__file__).resolve().parents[1] / "outputs" / "run_20260918_human_study"
needs_study = pytest.mark.skipif(
    not (STUDY / "study.html").exists(),
    reason="no generated study on disk; run scripts/human_study_export.py")


def _payload(page: str) -> str:
    """The embedded data, with base64 image blobs removed."""
    m = re.search(r'const STIM\s*=\s*(\{.*?\});', page, re.S)
    blob = m.group(1) if m else page
    return re.sub(r'"[A-Za-z0-9+/=]{200,}"', '""', blob)


@needs_study
def test_no_arm_label_reaches_the_rendered_page():
    """A condition visible in the DOM is a condition the rater can read.

    Checked against the embedded payload rather than the whole page, because
    one arm is named `real` and the page's own prose asks which build looks
    real -- a naive substring scan over the HTML reports that as a leak.
    """
    key = json.loads((STUDY / "answer_key.json").read_text())
    data = _payload((STUDY / "study.html").read_text())
    leaked = sorted(a for a in key["index"] if a in data)
    assert not leaked, leaked
    for forbidden in ("left_arm", "right_arm", '"catch"'):
        assert forbidden not in data, forbidden


@needs_study
def test_ten_raters_cover_every_pair_and_power_the_priority_pairs():
    """The collection target has to actually produce an analysable table."""
    key = json.loads((STUDY / "answer_key.json").read_text())
    stim = json.loads((STUDY / "stimuli.json").read_text())
    pair = {t["id"]: frozenset({t["left_arm"], t["right_arm"]})
            for t in key["trials"]}
    seen = collections.Counter()
    for sess in stim["sessions"][:10]:
        for i in sess:
            seen[pair[i]] += 1
    assert sum(1 for v in seen.values() if v) == 28, "a pair gets no judgements"
    assert seen[frozenset({"real", "real@solidify"})] >= 25
    assert seen[frozenset({"gabled_house", "patchwork"})] >= 25
