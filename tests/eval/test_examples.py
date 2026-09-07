"""The examples writer: the aliasing scar and the caption scar, pinned.

Two of these tests are regressions for named defects and matter more than the
rest.

`test_write_does_not_mutate_the_inputs` is the aliasing one.
`ArmSpec.load()` hands back `self.structures` *uncropped and unwrapped* for an
in-memory arm, so the writer is holding the very objects the control builder
made; `real@single_mode` (fast.py) is `[pool[0]] * n` -- one `Structure`
repeated -- and that object is simultaneously row 0 of the `real_test` arm and a
member of the shared `test` split the lab serves to Curate and memoizes for the
life of the process. A single in-place `s.metadata["category"] = arm` would
therefore rewrite one object seen from three datasets and would keep doing so
until the process exits. The fixtures below reproduce that exact sharing --
one structure repeated inside an arm *and* reused across arms -- and the test
compares a deep copy of every input's metadata taken before the call.

`test_captions_are_set_for_metadata_less_builds` is the caption one.
`save_house_cache` persists only `corpus/category/title/url` per row
(houses.py:385-390) and the probe and baseline builders construct
`Structure(block_ids=..., block_data=...)` with no metadata at all
(probes.py, baselines.py:97) -- as does `sample_to_npz`, whose
`outputs/bench_arms/native_oriented_32_manifest.json` item 0 is
`{"source_path":"","corpus":"","category":"","title":"","url":""}`, five empty
fields out of seven. Captions are not free: without the explicit relabel most
of a card's rows persist blank and the strip is a wall of unlabelled
thumbnails.

`test_write_produces_an_openable_cache` asserts through
`tools.lab.catalog._openable` itself rather than re-checking for the two files
by hand. Tying the producer to the consumer's own admission test is the point:
the pair on disk is only worth writing if the lab's discovery would accept it.

No corpus, no renderer, no GPU: every build here is a 4x4x4 hut.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from blockgen.curation.houses import load_structures_from_cache
from blockgen.eval.bench import examples as ex
from blockgen.eval.bench.scorecard import ArmSpec
from blockgen.utils.data import Structure
from tools.lab import catalog

CORPUS = "houses_32"


# --- fixtures in miniature -------------------------------------------------
def _struct(code: int, n: int = 4, **meta: str) -> Structure:
    """A tiny hut whose voxels identify it, so a round-trip can be checked."""
    ids = np.zeros((n, n, n), dtype=np.int32)
    ids[:, 0, :] = 4                     # floor
    ids[0, :, 0] = code                  # a corner post that differs per build
    return Structure(block_ids=ids, block_data=np.zeros_like(ids), metadata=dict(meta))


def _arms(n_each: int = 5) -> List[ArmSpec]:
    """Three arms of five builds, shaped like the real ones.

    `real_test` carries GrabCraft titles; `real@single_mode` is the aliasing
    case -- `[pool[0]] * n`, the same object repeated, and that object is also
    `real_test` row 0; `probe` carries no metadata at all, which is what the
    probe and baseline builders actually produce.
    """
    real = [_struct(10 + i, title=f"American Middle Class House {i}",
                    corpus=CORPUS, url=f"http://example/{i}")
            for i in range(n_each)]
    single = [real[0]] * n_each
    probe = [_struct(30 + i) for i in range(n_each)]
    return [ArmSpec(name="real_test", track="control", structures=real),
            ArmSpec(name="real@single_mode", track="control", structures=single),
            ArmSpec(name="probe", track="control", structures=probe)]


def _picked(arms: List[ArmSpec], k: int = 3, seed: int = 0
            ) -> "OrderedDict[str, Tuple[List[Structure], List[int]]]":
    """What the runner assembles: the k chosen builds plus their source rows."""
    out: "OrderedDict[str, Tuple[List[Structure], List[int]]]" = OrderedDict()
    rng = np.random.default_rng(seed)
    for arm in arms:
        structs = arm.structures or []
        rows = ex.pick(len(structs), k, rng)
        out[arm.name] = ([structs[i] for i in rows], list(rows))
    return out


def _manifest(run_dir: Path, blob: dict) -> dict:
    return json.loads((run_dir / blob["run"]["manifest"]).read_text())


# --- pick ------------------------------------------------------------------
def test_pick_is_deterministic_and_sorted():
    a = ex.pick(20, 5, np.random.default_rng(7))
    b = ex.pick(20, 5, np.random.default_rng(7))
    assert a == b
    assert a == sorted(a)
    assert len(set(a)) == len(a) == 5
    assert all(0 <= i < 20 for i in a)
    # A permutation, not the head of the list: `dump_samples.py` keeps source
    # order when it is not subsampling, so `[:k]` would show the same corner of
    # the corpus every time.
    assert ex.pick(50, 5, np.random.default_rng(7)) != list(range(5))


def test_pick_clamps():
    rng = np.random.default_rng(0)
    assert ex.pick(4, 4, rng) == [0, 1, 2, 3]
    assert ex.pick(4, 9, rng) == [0, 1, 2, 3]
    assert ex.pick(4, 0, rng) == []
    assert ex.pick(4, -1, rng) == []
    assert ex.pick(0, 8, rng) == []


# --- the pair on disk ------------------------------------------------------
def test_write_produces_an_openable_cache(tmp_path):
    arms = _arms()
    picked = _picked(arms)
    blob = ex.write_run_examples(tmp_path, picked, corpus=CORPUS, k=3, seed=0,
                                 run_name="smoke run")

    path = tmp_path / blob["run"]["npz"]
    # The lab's OWN admission test, not a reimplementation of it.
    assert catalog._openable(path)
    assert blob["run"]["npz"] == "examples_4.npz"          # true max over the builds
    assert blob["run"]["manifest"] == "examples_4_manifest.json"
    assert Path(blob["run"]["npz"]).name == blob["run"]["npz"]   # never absolute
    assert blob["run"]["k"] == 3
    assert blob["run"]["seed"] == 0
    assert blob["run"]["count"] == 9

    loaded, manifest = load_structures_from_cache(str(path))
    assert len(loaded) == manifest["count"] == 9
    flat = [s for structs, _ in picked.values() for s in structs]
    for got, want in zip(loaded, flat):
        assert np.array_equal(got.block_ids, want.block_ids)
        assert np.array_equal(got.block_data, want.block_data)


def test_rows_are_contiguous_and_index_the_right_builds(tmp_path):
    arms = _arms()
    picked = _picked(arms)
    blob = ex.write_run_examples(tmp_path, picked, corpus=CORPUS, k=3, seed=0)
    loaded, _ = load_structures_from_cache(str(tmp_path / blob["run"]["npz"]))

    assert list(blob["arms"]) == [a.name for a in arms]     # card order
    seen: List[int] = []
    for arm in arms:
        entry = blob["arms"][arm.name]
        rows, src = entry["rows"], entry["source_rows"]
        assert len(rows) == len(src) == 3
        assert rows == list(range(rows[0], rows[0] + len(rows)))    # contiguous
        assert not set(rows) & set(seen)                            # disjoint
        seen.extend(rows)
        for row, source_row in zip(rows, src):
            want = arm.structures[source_row]
            assert np.array_equal(loaded[row].block_ids, want.block_ids)
    assert seen == list(range(9))


# --- the aliasing regression ----------------------------------------------
def test_write_does_not_mutate_the_inputs(tmp_path):
    """D17. The scar is in the module docstring; this is the guard on it."""
    arms = _arms()
    # The cross-arm alias, made explicit: `real@single_mode` row 0 IS
    # `real_test` row 0, exactly as `[pool[0]] * n` makes it.
    assert arms[1].structures[0] is arms[0].structures[0]
    assert arms[1].structures[0] is arms[1].structures[1]

    every = [s for arm in arms for s in (arm.structures or [])]
    before = {id(s): copy.deepcopy(s.metadata) for s in every}
    voxels = {id(s): s.block_ids.copy() for s in every}

    picked = _picked(arms)
    ex.write_run_examples(tmp_path, picked, corpus="a_different_corpus", k=3, seed=0)

    for s in every:
        assert s.metadata == before[id(s)], "input metadata was written through"
        assert np.array_equal(s.block_ids, voxels[id(s)])
    # The metadata-less probe builds are the loudest case: nothing at all.
    assert all(s.metadata == {} for s in arms[2].structures)
    # The manifest still got the labels, so this is not passing by writing
    # nothing: the copies carry them and the originals do not.
    assert all(item["category"] for item in
               json.loads((tmp_path / "examples_4_manifest.json").read_text())["items"])


def test_relabel_returns_a_copy_with_a_fresh_metadata_dict():
    """The one-line unit behind the regression above."""
    s = _struct(1, title="House 3")
    out = ex._relabel(s, "real@single_mode", 7, CORPUS)
    assert out is not s
    assert out.metadata is not s.metadata
    assert s.metadata == {"title": "House 3"}
    assert out.metadata == {"title": "House 3", "category": "real@single_mode",
                            "corpus": CORPUS}
    # Mutating the copy's dict afterwards cannot reach the source.
    out.metadata["category"] = "other"
    assert s.metadata == {"title": "House 3"}
    # A metadata-less build gets the positional fallback caption.
    assert ex._relabel(_struct(2), "probe", 4, CORPUS).metadata["title"] == "probe #4"


# --- the caption regression ------------------------------------------------
def test_captions_are_set_for_metadata_less_builds(tmp_path):
    """Probe and baseline builds carry no metadata; the manifest must not."""
    probe = ArmSpec(name="probe", track="control",
                    structures=[_struct(30 + i) for i in range(5)])
    assert all(s.metadata == {} for s in probe.structures)

    picked = _picked([probe], k=3)
    blob = ex.write_run_examples(tmp_path, picked, corpus=CORPUS, k=3, seed=0)
    items = _manifest(tmp_path, blob)["items"]

    entry = blob["arms"]["probe"]
    for row, source_row in zip(entry["rows"], entry["source_rows"]):
        assert items[row]["category"] == "probe"
        assert items[row]["title"] == f"probe #{source_row}"
        assert items[row]["corpus"] == CORPUS
    # The npz's own corpus column is written from the same copied metadata.
    loaded, _ = load_structures_from_cache(str(tmp_path / blob["run"]["npz"]))
    assert {s.metadata["category"] for s in loaded} == {"probe"}


def test_existing_title_is_preserved(tmp_path):
    """A GrabCraft name or an agentic arm's full prompt is a real caption."""
    title = "American Middle Class House 10"
    arm = ArmSpec(name="real_test", track="control",
                  structures=[_struct(1, title=title, corpus="houses_48")])
    blob = ex.write_run_examples(tmp_path, OrderedDict([("real_test",
                                                         (list(arm.structures), [0]))]),
                                 corpus=CORPUS, k=1, seed=0)
    item = _manifest(tmp_path, blob)["items"][0]
    assert item["title"] == title
    assert item["corpus"] == "houses_48"      # the source's own corpus wins
    assert item["category"] == "real_test"    # but the arm always names the row


# --- the sweep -------------------------------------------------------------
def test_sweep_removes_a_stale_examples_pair(tmp_path):
    """D20: a data-dependent filename would leave an orphan dataset on a rerun."""
    big = [_struct(1, n=8), _struct(2, n=8)]
    ex.write_run_examples(tmp_path, OrderedDict([("a", (big, [0, 1]))]),
                          corpus=CORPUS, k=2, seed=0)
    assert (tmp_path / "examples_8.npz").is_file()

    small = [_struct(3), _struct(4)]
    blob = ex.write_run_examples(tmp_path, OrderedDict([("a", (small, [0, 1]))]),
                                 corpus=CORPUS, k=2, seed=0)

    assert blob["run"]["npz"] == "examples_4.npz"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "examples_4.npz", "examples_4_manifest.json"]


def test_sweep_leaves_unrelated_files_alone(tmp_path):
    """`_STALE` matches exactly what `house_cache_path` writes, nothing else."""
    (tmp_path / "examples_notes.npz").write_bytes(b"")
    (tmp_path / "scorecard.json").write_text("{}")
    ex.write_run_examples(tmp_path, OrderedDict([("a", ([_struct(1)], [0]))]),
                          corpus=CORPUS, k=1, seed=0)
    assert (tmp_path / "examples_notes.npz").is_file()
    assert (tmp_path / "scorecard.json").is_file()


# --- nothing to write ------------------------------------------------------
def test_empty_input_writes_nothing(tmp_path):
    assert ex.write_run_examples(tmp_path, OrderedDict(), corpus=CORPUS,
                                 k=8, seed=0) == {}
    # An arm that produced no non-empty structures gets no `examples` key at
    # all, and a run of only such arms writes no n=0 dataset for the lab to list.
    assert ex.write_run_examples(tmp_path, OrderedDict([("a", ([], []))]),
                                 corpus=CORPUS, k=8, seed=0) == {}
    assert list(tmp_path.iterdir()) == []


def test_empty_input_leaves_an_existing_pair_alone(tmp_path):
    """Deliberate: the sweep guards THIS write, and an empty run made none."""
    ex.write_run_examples(tmp_path, OrderedDict([("a", ([_struct(1)], [0]))]),
                          corpus=CORPUS, k=1, seed=0)
    assert ex.write_run_examples(tmp_path, OrderedDict(), corpus=CORPUS,
                                 k=0, seed=0) == {}
    assert (tmp_path / "examples_4.npz").is_file()


def test_an_arm_with_nothing_is_skipped_without_shifting_the_others(tmp_path):
    picked: "OrderedDict[str, Tuple[List[Structure], List[int]]]" = OrderedDict([
        ("empty", ([], [])),
        ("a", ([_struct(1), _struct(2)], [0, 1])),
    ])
    blob = ex.write_run_examples(tmp_path, picked, corpus=CORPUS, k=2, seed=0)
    assert "empty" not in blob["arms"]
    assert blob["arms"]["a"]["rows"] == [0, 1]
    assert blob["run"]["count"] == 2


# --- the report the lab reads ---------------------------------------------
def test_manifest_report_arms_matches_the_returned_mapping(tmp_path):
    arms = _arms()
    picked = _picked(arms)
    blob = ex.write_run_examples(tmp_path, picked, corpus=CORPUS, k=3, seed=0,
                                 run_name="ontology arms v2")
    report = _manifest(tmp_path, blob)["report"]

    assert report["arms"] == {name: entry["rows"]
                              for name, entry in blob["arms"].items()}
    assert report["kind"] == "bench_examples"
    assert report["run"] == tmp_path.name
    assert report["run_name"] == "ontology arms v2"
    assert report["k"] == 3 and report["seed"] == 0
