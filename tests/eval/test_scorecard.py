"""The `bench/2` schema contract: identity, context, and arm provenance.

There was no test file for `scorecard.py` at all until now, which is how the
provenance chain came to be severed by a single line for eleven runs: `load()`
unpacked `(structs, manifest)` and dropped the manifest into `_`, so the only
record of which checkpoint, seed and temperature produced an arm was read off
disk, held in a local, and discarded -- on every card, every time. Nothing
failed, because nothing looked.

The assertions here are therefore mostly about *what must still be true after
the next edit*, not about arithmetic:

* `test_armspec_load_keeps_the_manifest` closes that severing point.
* `test_provenance_works_before_load` pins that `provenance()` needs no caller
  ordering, because `score_fast`, `score_full` and the examples writer each call
  `load()` independently and nothing in the type states which runs first.
* `test_provenance_promotes_only_the_allowlist` keeps `PROMOTED` closed, so a
  manifest writer cannot inject arbitrary keys into the top of a provenance
  block by naming them in its `report`.
* `test_context_to_json_is_a_superset_of_the_legacy_keys` freezes the key set
  the deleted hand-built dict in `runner.main` used to write, because
  `scripts/bench_report.py` reads three of them bare and hard-crashes.
* `test_text_backbone_is_declared_and_not_emitted_bare` keeps a fast-tier card
  from claiming a backbone it never loaded.

Everything is fast and unmarked: no corpus, no GPU, no renderer. The one file
read from disk is the committed `bench/2` fixture.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from blockgen.curation.houses import save_house_cache
from blockgen.eval.bench import scorecard as sc
from blockgen.utils.data import Structure

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CARD = (REPO_ROOT / "tests" / "fixtures" / "scorecards"
                / "run_20260102_000000_bench_fixture" / "scorecard.json")

#: The `context` keys the runner hand-built beside `BenchContext` until
#: `bench/2` deleted the parallel dict (the block removed from `runner.main`,
#: visible as the `-` lines of `git diff blockgen/eval/bench/runner.py`).
#: `BenchContext.to_json()` must stay a strict superset of these: three of them
#: -- `corpus`, `split_key`, `n_ref_used` -- are read bare by
#: `scripts/bench_report.py`, which crashes rather than degrading.
#:
#: Frozen as a literal tuple, not derived from anything, so that deleting a key
#: from `to_json` fails here loudly instead of quietly shrinking a card.
LEGACY_CONTEXT_KEYS = (
    "corpus", "seed", "grid", "min_n", "split_key", "split_sha",
    "n_train", "n_val", "n_test", "n_ref_used", "vocab_size",
    "palette_keys_exact", "palette_keys_family", "bootstrap",
)

#: What the runner puts in `ctx.derived` -- values it can only measure once the
#: reference is built. Kept here so the superset check exercises the real path
#: (declared fields ∪ derived) rather than a context nobody ever scores with.
DERIVED = {"n_train": 100, "n_val": 30, "n_test": 30, "n_ref_used": 30,
           "vocab_size": 441, "palette_keys_exact": 406,
           "palette_keys_family": 32}


def _hut(fill: int = 5, n: int = 4) -> Structure:
    """A tiny solid-walled cube. Small enough that a cache write is instant."""
    ids = np.zeros((n, n, n), dtype=np.int32)
    ids[0, :, :] = ids[-1, :, :] = fill
    ids[:, 0, :] = fill
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _cache(tmp_path: Path, report: dict, name: str = "fixture_arm") -> str:
    """Write a real two-build cache with `save_house_cache` and return its npz."""
    return save_house_cache([_hut(5), _hut(6)], max_dim=4,
                            cache_dir=str(tmp_path), report=report, name=name)


# --- the version string -----------------------------------------------------
def test_schema_version_is_bench2():
    assert sc.SCHEMA_VERSION == "bench/2"


# --- the context block ------------------------------------------------------
def test_context_to_json_is_a_superset_of_the_legacy_keys():
    """Every key the deleted hand-built dict wrote is still written.

    `bench/2` is `bench/1` plus keys. This is the assertion that says so for
    `context`, and it is the one that fails when someone "tidies" `to_json`.
    """
    ctx = sc.BenchContext(corpus="houses_48", split_key="houses_48.s0.70-15-15.v1",
                          split_sha="10317983170dd1fd",
                          sizes={"train": 100, "val": 30, "test": 30},
                          derived=dict(DERIVED))
    out = ctx.to_json()
    assert set(LEGACY_CONTEXT_KEYS) <= set(out), (
        "context lost legacy keys: "
        f"{sorted(set(LEGACY_CONTEXT_KEYS) - set(out))}")


def test_context_to_json_adds_the_four_dropped_values():
    """`n_ref`, `palette_level`, `dup_threshold` and `sizes` reach a card at last.

    All four were used to score every arm on all eleven existing cards and
    reported on none of them, because the runner's parallel dict did not carry
    them. A context that is not the context is worse than no context.
    """
    ctx = sc.BenchContext(n_ref=256, palette_level="family", dup_threshold=0.9,
                          sizes={"train": 100, "val": 30, "test": 30},
                          n_boot=250, alpha=0.01)
    out = ctx.to_json()
    assert out["n_ref"] == 256
    assert out["palette_level"] == "family"
    assert out["dup_threshold"] == 0.9
    assert out["sizes"] == {"train": 100, "val": 30, "test": 30}
    assert out["bootstrap"] == {"n_boot": 250, "alpha": 0.01, "unit": "structure"}


def test_context_sizes_is_a_copy_not_the_live_dict():
    """A card must not change under the object that produced it."""
    sizes = {"train": 1, "val": 2, "test": 3}
    ctx = sc.BenchContext(sizes=sizes)
    out = ctx.to_json()
    sizes["train"] = 999
    assert out["sizes"]["train"] == 1


def test_text_backbone_is_declared_and_not_emitted_bare():
    """D10: a fast-tier card may not claim a backbone it never loaded.

    `text_backbone` used to be monkey-patched onto the context from the runner
    and read back through a `getattr` default -- a field with none of a field's
    guarantees. It is now declared, which is what makes it serializable; the
    price of declaring it is that a default value now exists, so it must appear
    only *inside* `backbone`, beside the image backbone that was actually used.
    """
    names = {f.name for f in dataclasses.fields(sc.BenchContext)}
    assert "text_backbone" in names

    fast_ctx = sc.BenchContext()
    assert fast_ctx.text_backbone == "clipL"      # a default, not a measurement
    out = fast_ctx.to_json()
    assert "text_backbone" not in out
    assert "backbone" not in out

    full_ctx = sc.BenchContext(image_backbone="dinov2b", text_backbone="clipL")
    full = full_ctx.to_json()
    assert "text_backbone" not in full, "text_backbone must never be emitted bare"
    assert full["backbone"] == {"image": "dinov2b", "text": "clipL", "pool": "mean"}


def test_full_tier_context_keys_appear_only_when_set():
    ctx = sc.BenchContext()
    fast_out = ctx.to_json()
    for key in ("view", "backbone", "mmd_sigma", "noise_floor_sd"):
        assert key not in fast_out, f"{key} is a full-tier fact on a fast card"

    ctx.view = {"key": "o4x224", "px": 224, "views": [[45, 30]]}
    ctx.image_backbone = "dinov2b"
    ctx.mmd_sigma = 12.44
    # Measured whenever the controls ran, fast tier included -- keyed off
    # content, not off the tier, which is why it is checked separately.
    ctx.noise_floor_sd = {"realism.geom_kid": 0.163}
    full_out = ctx.to_json()
    assert full_out["view"]["px"] == 224
    assert full_out["backbone"]["image"] == "dinov2b"
    assert full_out["mmd_sigma"] == pytest.approx(12.44)
    assert full_out["noise_floor_sd"] == {"realism.geom_kid": pytest.approx(0.163)}


def test_context_to_json_is_json_serializable():
    """numpy leaks in from every measurement site; `_clean` is what stops it."""
    ctx = sc.BenchContext(seed=int(np.int64(3)),
                          sizes={"train": int(np.int32(7))},
                          derived={"n_ref_used": np.int64(30),
                                   "vocab_size": np.int32(441),
                                   "oov": np.float32(0.5),
                                   "nan_is_null": float("nan")})
    ctx.mmd_sigma = np.float64(12.44)
    ctx.noise_floor_sd = {"realism.geom_kid": np.float32(0.163)}
    out = ctx.to_json()
    round_tripped = json.loads(json.dumps(out))
    assert round_tripped["seed"] == 3 and round_tripped["sizes"] == {"train": 7}
    assert round_tripped["vocab_size"] == 441
    assert round_tripped["nan_is_null"] is None, "non-finite floats become null"
    assert isinstance(round_tripped["mmd_sigma"], float)


# --- ArmSpec: load, and the manifest it used to drop -------------------------
def test_armspec_load_keeps_the_manifest(tmp_path):
    """The severing point inside `ArmSpec.load`, closed.

    Until `bench/2` that line -- `scorecard.py:152` in the tree this was written
    against -- read `structs, _ = load_structures_from_cache(...)` and threw the
    manifest away. It was the single place where the whole provenance chain
    died: the checkpoint, seed and temperature that produced an arm were already
    in memory, already in scope, on that exact line, and every card written for
    eleven runs lost them there. `load()` now keeps it on the spec, and this is
    the test that fails if anyone drops it again.
    """
    npz = _cache(tmp_path, {"model": "m",
                            "checkpoint": "outputs/run_20260101_000000_x/best.pt",
                            "seed": 1, "temperature": 0.9, "top_k": 50})
    arm = sc.ArmSpec(name="a", npz=npz)
    assert arm.manifest is None

    structs = arm.load()
    assert len(structs) == 2
    assert arm.manifest is not None, "load() dropped the manifest again"
    assert arm.manifest["report"]["model"] == "m"
    assert arm.manifest["report"]["checkpoint"].endswith("best.pt")
    assert arm.manifest["count"] == 2


def test_load_returns_in_memory_structures_untouched():
    """In-memory arms short-circuit: the same objects, not copies.

    Stated because the examples writer depends on knowing it -- `load()` hands
    back the caller's own list, `real@single_mode` is `[pool[0]] * n`, and an
    in-place metadata edit downstream would corrupt three datasets at once.
    """
    structs = [_hut(5), _hut(6)]
    arm = sc.ArmSpec(name="mem", track="control", structures=structs)
    assert arm.load() is structs


def test_load_without_structures_or_npz_raises():
    with pytest.raises(ValueError):
        sc.ArmSpec(name="nothing").load()


# --- ArmSpec.provenance -----------------------------------------------------
def test_provenance_works_before_load(tmp_path):
    """D12: `provenance()` is self-sufficient; no caller ordering is required.

    The first version depended on `score_fast` calling `load()` before it built
    the meta block -- an invariant nothing in the type states, and one that
    `score_full` and the examples writer each break by calling `load()` on their
    own. So it reads the `_manifest.json` sibling itself, and memoizes.
    """
    npz = _cache(tmp_path, {"model": "native_oriented",
                            "checkpoint": "outputs/run_20260721_013119_x/model.pt",
                            "seed": 0, "temperature": 1.0, "top_k": 40})
    arm = sc.ArmSpec(name="a", npz=npz)

    prov = arm.provenance()                       # load() has NOT run
    assert arm.manifest is not None, "provenance() must memoize what it read"
    assert prov["writer"] == "npz"
    assert prov["npz"] == npz
    assert prov["manifest"] == npz.replace(".npz", "_manifest.json")
    assert prov["count"] == 2 and prov["max_dim"] == 4
    assert prov["written_at"] and prov["written_at"].endswith("Z")
    assert prov["model"] == "native_oriented"
    assert prov["seed"] == 0 and prov["temperature"] == 1.0 and prov["top_k"] == 40
    assert prov["checkpoint"].endswith("model.pt")


def test_provenance_after_load_agrees_with_provenance_before_load(tmp_path):
    npz = _cache(tmp_path, {"model": "m", "seed": 3})
    before = sc.ArmSpec(name="a", npz=npz).provenance()
    after_spec = sc.ArmSpec(name="a", npz=npz)
    after_spec.load()
    assert after_spec.provenance() == before


def test_provenance_promotes_only_the_allowlist(tmp_path):
    """D13: an unexpected report key stays inside `report` and never rises.

    The three manifest writers on disk share no key but `report` itself, so the
    promotion is a convenience for whoever renders the page and never the
    record. A closed allowlist is what keeps a new writer from silently adding
    top-level provenance keys nobody documented.
    """
    npz = _cache(tmp_path, {"model": "m", "seed": 1, "secret": "do-not-promote",
                            "n": 2, "n_empty": 0})
    prov = sc.ArmSpec(name="a", npz=npz).provenance()

    assert prov["report"]["secret"] == "do-not-promote"
    assert "secret" not in prov
    assert prov["model"] == "m" and prov["seed"] == 1
    # `n`/`n_empty` are written by `sample_to_npz` and are deliberately NOT
    # promoted -- the arm's own `meta.n` is the scored count and may differ.
    assert "n" not in prov and "n_empty" not in prov
    promoted_here = set(prov) & set(sc.PROMOTED)
    assert promoted_here == {"model", "seed"}


def test_promoted_keys_are_absent_when_the_report_omits_them(tmp_path):
    npz = _cache(tmp_path, {"source": "dump_samples"})
    prov = sc.ArmSpec(name="a", npz=npz).provenance()
    assert prov["source"] == "dump_samples"
    for key in ("model", "checkpoint", "temperature", "top_k", "arm", "epochs"):
        assert key not in prov


def test_provenance_of_a_missing_manifest_does_not_raise(tmp_path):
    """A missing manifest is a fact about the arm, not a reason to lose a run."""
    arm = sc.ArmSpec(name="gone", npz=str(tmp_path / "nope.npz"))
    prov = arm.provenance()
    assert prov["writer"] == "npz"
    assert prov["report"] == {}
    assert prov["count"] is None and prov["max_dim"] is None
    assert prov["written_at"] is None
    # It still names both paths it looked at, which is the whole diagnostic.
    assert prov["npz"].endswith("nope.npz")
    assert prov["manifest"].endswith("nope_manifest.json")


def test_provenance_of_an_unparseable_manifest_does_not_raise(tmp_path):
    npz = tmp_path / "broken.npz"
    npz.write_bytes(b"")
    (tmp_path / "broken_manifest.json").write_text("{not json")
    prov = sc.ArmSpec(name="broken", npz=str(npz)).provenance()
    assert prov["writer"] == "npz" and prov["report"] == {}


def test_provenance_in_memory():
    """An in-memory arm reports its recipe, not "nothing on disk to show"."""
    override = {"writer": "in_memory",
                "builder": "blockgen.eval.bench.fast.control_arms",
                "recipe": "every enclosed air cell filled",
                "drawn_from": "split:houses_32:test", "n_requested": 128, "seed": 1}
    arm = sc.ArmSpec(name="real@solidify", track="control",
                     structures=[_hut()], provenance_override=override)
    prov = arm.provenance()
    assert prov["writer"] == "in_memory"
    assert prov["recipe"] == "every enclosed air cell filled"
    assert prov["drawn_from"] == "split:houses_32:test"


def test_provenance_override_is_copied_not_aliased():
    """A card stashing this block must not be able to reach back into the spec."""
    override = {"writer": "in_memory", "builder": "b"}
    arm = sc.ArmSpec(name="x", track="control", structures=[_hut()],
                     provenance_override=override)
    prov = arm.provenance()
    prov["builder"] = "mutated"
    assert arm.provenance_override["builder"] == "b"
    assert arm.provenance()["builder"] == "b"


def test_provenance_of_an_unrecorded_in_memory_arm_says_so():
    """No npz and no override: say exactly that, do not invent a recipe."""
    prov = sc.ArmSpec(name="x", track="control", structures=[_hut()]).provenance()
    assert prov == {"writer": "in_memory", "builder": "?"}


# --- kind / origin ----------------------------------------------------------
@pytest.mark.parametrize("track,kind", [("ar", "submission"),
                                        ("agentic", "submission"),
                                        ("control", "control"),
                                        ("baseline", "baseline")])
def test_arm_kind_and_origin(track, kind, tmp_path):
    """One definition of the rule, so three readers cannot disagree about it."""
    on_disk = sc.ArmSpec(name="a", track=track, npz=str(tmp_path / "x.npz"))
    in_memory = sc.ArmSpec(name="a", track=track, structures=[_hut()])
    assert on_disk.kind == kind and in_memory.kind == kind
    assert on_disk.origin == "npz"
    assert in_memory.origin == "in_memory"


def test_armspec_appended_fields_stay_at_the_end():
    """`bench_doseresponse.py:44` and `human_study_export.py:40` build ArmSpecs
    POSITIONALLY, so a field inserted above `manifest` shifts their arguments
    silently instead of failing. The order is part of the contract.
    """
    names = [f.name for f in dataclasses.fields(sc.ArmSpec)]
    assert names == ["name", "track", "npz", "structures", "prompts", "cost",
                     "manifest", "provenance_override"]
    positional = sc.ArmSpec("n", "control", None, [_hut()], None, {"usd": 1.0})
    assert positional.name == "n" and positional.track == "control"
    assert positional.cost == {"usd": 1.0}


# --- the run block contract -------------------------------------------------
def test_run_keys_contract():
    """D34: the committed `bench/2` card carries exactly `RUN_KEYS` + optionals.

    The synthesized fixture is not a real run's output, so without this it would
    drift from what the runner writes. The other half of the pin -- a real card
    on disk -- is the `slow` test in `tests/lab/test_lab_data.py`.
    """
    blob = json.loads(FIXTURE_CARD.read_text())
    keys = set(blob["run"])
    assert set(sc.RUN_KEYS) <= keys, (
        f"fixture is missing required run keys: {sorted(set(sc.RUN_KEYS) - keys)}")
    extra = keys - set(sc.RUN_KEYS)
    assert extra <= set(sc.RUN_KEYS_OPTIONAL), (
        f"fixture carries undeclared run keys: {sorted(extra - set(sc.RUN_KEYS_OPTIONAL))}")


def test_run_keys_and_optionals_are_disjoint():
    assert not set(sc.RUN_KEYS) & set(sc.RUN_KEYS_OPTIONAL)
    assert len(set(sc.RUN_KEYS)) == len(sc.RUN_KEYS)


def test_scorecard_to_json_stamps_the_version_and_keeps_run_verbatim():
    card = sc.Scorecard(context={"corpus": "houses_32"},
                        run={"dir": "outputs/run_x", "elapsed_s": np.float64(1.5)})
    card.add_arm("a", {"realism": {"m": sc.metric(0.5, (0.1, 0.9))}})
    out = card.to_json()
    assert out["schema_version"] == "bench/2"
    assert out["run"]["dir"] == "outputs/run_x"
    assert out["arms"]["a"]["realism"]["m"]["value"] == pytest.approx(0.5)
    json.dumps(out)                                # numpy must be gone


# --- timestamps and git -----------------------------------------------------
def test_iso_utc_shape():
    """One timestamp format, so `started_at` and `written_at` sort as strings."""
    stamp = sc.iso_utc(1767312000.0)               # 2026-01-02T00:00:00Z
    assert stamp == "2026-01-02T00:00:00Z"
    assert stamp.endswith("Z") and "+00:00" not in stamp


def test_git_info_never_raises(monkeypatch):
    """A machine with no `git` on PATH yields the same shape, not an exception
    in the middle of writing a card."""
    monkeypatch.setenv("PATH", "")
    info = sc.git_info()
    assert set(info) == {"sha", "branch", "dirty"}
    assert info["sha"] == "unknown"
    assert info["branch"] is None
    # None means "we could not ask git", which is not the same as clean.
    assert info["dirty"] is None
