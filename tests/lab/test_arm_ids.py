"""`catalog._arm_id`: the path-to-dataset-id rule, pinned byte for byte.

**Build ids are the primary key of `outputs/lab/lab.db`.** Every label, every
note and every 2AFC comparison ever recorded is stored against
`"<dataset_id>:<index>"`, and a dataset id is this function's output for the npz
the builds live in. Change the slugging -- a different separator, a stripped
prefix, a resolved symlink, a `Path.stem` where there was a `with_suffix("")` --
and every one of those rows is orphaned in place: the labels stay in the
database, the page shows an unlabelled grid, and nothing anywhere reports an
error. There is no migration for it either, because the old ids are not
recoverable from the new ones.

So the ids are asserted as literal strings against a literal table of paths,
rather than by round-tripping the rule through itself. A test that computes the
expectation the same way the code does cannot fail when the rule changes, which
is the only failure this file is here to catch.
`arm:bench_arms__native_oriented_32` is also hard-coded in
`tests/lab/test_lab_data.py:101,117,124`; this is where that string is decided.

The second job here is the examples cache added in `bench/2`
(`examples_<max_dim>.npz` beside a scorecard): it reaches the browser as ordinary
build ids through the thumbnail route that already exists, so its id shape is
part of the same contract and is pinned with the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.lab import catalog

#: `relative path under outputs/` -> `dataset id`. Read this table as the
#: specification; the function is the implementation of it.
IDS = {
    # The curated benchmark arms: the shape every label in the store was
    # recorded against.
    "bench_arms/native_oriented_32.npz": "arm:bench_arms__native_oriented_32",
    "bench_arms/agentic_oneshot_32.npz": "arm:bench_arms__agentic_oneshot_32",
    # A bench run's example builds, written beside its scorecard (bench/2).
    "run_20260906_190029_bench/examples_48.npz":
        "arm:run_20260906_190029_bench__examples_48",
    "run_20260102_000000_bench_fixture/examples_4.npz":
        "arm:run_20260102_000000_bench_fixture__examples_4",
    # Anything deeper is reached by `_arm_paths`' rglob; every separator becomes
    # `__` because the id is a path segment in `/api/thumb/<build_id>`.
    "run_20260721_013119_native_oriented_native/native_bpe/samples.npz":
        "arm:run_20260721_013119_native_oriented_native__native_bpe__samples",
    # `.` and `-` survive: they are safe in a path segment and dropping them
    # would silently merge `v1.2` with `v1_2`.
    "bench_arms/ont-mined_v1.2_48.npz": "arm:bench_arms__ont-mined_v1.2_48",
    # Everything else collapses to a single `_`, and leading/trailing runs are
    # stripped, so an id is never `arm:_x_`.
    "bench_arms/ont mined 48.npz": "arm:bench_arms__ont_mined_48",
    "bench_arms/arm (copy).npz": "arm:bench_arms__arm_copy",
}


@pytest.fixture
def outputs_root(tmp_path, monkeypatch):
    """A private `outputs/`. The ids depend only on the path *below* the root,
    which is exactly the property that lets a checkout move without orphaning a
    label -- and is why this test needs no files to exist."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    return Path("outputs")


@pytest.mark.parametrize("rel,expected", sorted(IDS.items()))
def test_arm_id_is_byte_stable(outputs_root, rel, expected):
    assert catalog._arm_id(outputs_root / rel) == expected


def test_examples_path_id_shape(outputs_root):
    """The one id shape `bench/2` adds. The bench records a run-dir-relative
    *path* and never an id, so this rule stays the lab's alone."""
    run = outputs_root / "run_20260906_190029_bench_ontology_arms"
    assert catalog._arm_id(run / "examples_48.npz") == \
        "arm:run_20260906_190029_bench_ontology_arms__examples_48"


def test_id_is_independent_of_where_the_checkout_lives(tmp_path, monkeypatch):
    """Two clones in different directories must mint the same id for the same
    file, or a label recorded on one machine means nothing on another."""
    ids = []
    for name in ("clone_a", "clone_b"):
        root = tmp_path / name / "outputs"
        root.mkdir(parents=True)
        monkeypatch.chdir(tmp_path / name)
        monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
        ids.append(catalog._arm_id(root / "bench_arms" / "native_oriented_32.npz"))
    assert ids == ["arm:bench_arms__native_oriented_32"] * 2


def test_an_absolute_path_gives_the_same_id_as_a_relative_one(outputs_root):
    rel = outputs_root / "bench_arms" / "native_oriented_32.npz"
    assert catalog._arm_id(rel) == catalog._arm_id(rel.resolve())


def test_a_path_outside_outputs_falls_back_to_the_filename(tmp_path, outputs_root):
    """`--out` may point anywhere and two cards on disk record an absolute
    scratchpad path. The id must still be well formed -- `example_build_ids` is
    what refuses to *serve* such a file, and it refuses on the path, not on a
    malformed id."""
    stray = tmp_path / "scratchpad" / "arms" / "native_oriented_32.npz"
    assert catalog._arm_id(stray) == "arm:native_oriented_32"


def test_ids_round_trip_through_the_build_id_grammar(outputs_root):
    """Ids carry no `:`, so `parse_build_id`'s right-split lands on the index."""
    for rel, expected in IDS.items():
        bid = catalog.make_build_id(catalog._arm_id(outputs_root / rel), 7)
        assert bid == f"{expected}:7"
        assert catalog.parse_build_id(bid) == (expected, 7)


def test_ids_are_safe_in_a_url_path_segment(outputs_root):
    """`/api/thumb/<build_id>` takes the id unescaped."""
    import urllib.parse

    for rel in IDS:
        arm_id = catalog._arm_id(outputs_root / rel)
        assert urllib.parse.quote(arm_id, safe="") == \
            arm_id.replace(":", "%3A")


def test_distinct_files_get_distinct_ids(outputs_root):
    assert len({catalog._arm_id(outputs_root / rel) for rel in IDS}) == len(IDS)
