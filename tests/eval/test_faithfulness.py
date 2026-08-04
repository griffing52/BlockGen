"""Prompt faithfulness: retrieval, CLIPScore, material agreement.

Retrieval is the lead metric because it has a chance floor. The tests below pin
that floor (a random image-text pairing must land near 1/32, not near zero and
not near one) and the caption-index choice, which is not cosmetic: caption 3 in
`houses_32_captions.json` is a template embedding the scraped title, so scoring
against it would reward matching a *name* rather than a building.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import faithfulness as fth
from blockgen.utils.data import Structure


def _unit(n, d=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _wood_house() -> Structure:
    ids = np.zeros((6, 5, 6), dtype=np.int32)
    ids[:, 0, :] = 5          # oak planks
    ids[0, :, :] = 5
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _stone_house() -> Structure:
    ids = np.zeros((6, 5, 6), dtype=np.int32)
    ids[:, 0, :] = 4          # cobblestone
    ids[0, :, :] = 4
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


# --- retrieval -------------------------------------------------------------
def test_perfect_pairing_retrieves_everything():
    f = _unit(40)
    out = fth.retrieval(f, f.copy(), 31, np.random.default_rng(0))
    assert out["correct"].mean() == pytest.approx(1.0)
    assert out["rank"].max() == 1


def test_random_pairing_lands_on_the_chance_floor():
    """The property that makes an absolute retrieval number readable."""
    img, txt = _unit(200, seed=1), _unit(200, seed=2)
    out = fth.retrieval(img, txt, 31, np.random.default_rng(0))
    assert abs(out["correct"].mean() - 1.0 / 32) < 0.05


def test_chance_floor_tracks_the_distractor_count():
    img, txt = _unit(200, seed=1), _unit(200, seed=2)
    few = fth.retrieval(img, txt, 3, np.random.default_rng(0))["correct"].mean()
    many = fth.retrieval(img, txt, 63, np.random.default_rng(0))["correct"].mean()
    assert few > many
    assert abs(few - 0.25) < 0.1


def test_retrieval_degenerate_input_is_empty():
    assert fth.retrieval(_unit(1), _unit(1), 31, np.random.default_rng(0))[
        "correct"].size == 0


def test_mrr_is_between_chance_and_one():
    img, txt = _unit(120, seed=1), _unit(120, seed=2)
    out = fth.retrieval(img, txt, 31, np.random.default_rng(0))
    mrr = float(np.mean(1.0 / out["rank"]))
    assert 0.0 < mrr < 1.0


# --- captions --------------------------------------------------------------
def test_caption_lookup_uses_manifest_row_ids():
    caps = {"h00000": ["a", "b"], "h00007": ["c", "d"]}
    assert fth.captions_for_indices(caps, [0, 7], caption_idx=0) == ["a", "c"]
    assert fth.captions_for_indices(caps, [0, 7], caption_idx=1) == ["b", "d"]


def test_missing_caption_becomes_empty_not_an_error():
    assert fth.captions_for_indices({}, [3]) == [""]
    assert fth.captions_for_indices({"h00003": ["only"]}, [3], caption_idx=2) == [""]


def test_default_caption_index_avoids_the_title_template():
    """Caption 3 leaks the scraped build title; the default must not be 3."""
    assert fth.DEFAULT_CAPTION_IDX == 0


# --- material agreement ----------------------------------------------------
def test_material_agreement_rewards_a_correct_description():
    out = fth.material_agreement([_wood_house()], ["a small oak wood house"])
    assert out["agreement"].size == 1
    assert out["agreement"][0] > 0.5
    assert out["coverage"] == pytest.approx(1.0)


def test_material_agreement_penalizes_a_wrong_description():
    right = fth.material_agreement([_wood_house()], ["an oak wood cabin"])
    wrong = fth.material_agreement([_stone_house()], ["an oak wood cabin"])
    assert wrong["agreement"][0] < right["agreement"][0]


def test_captions_naming_no_material_are_excluded_and_counted():
    """Agreement over 3 captions is not the same claim as over 60."""
    out = fth.material_agreement([_wood_house(), _wood_house()],
                                 ["a nice building", "an oak house"])
    assert out["agreement"].size == 1
    assert out["coverage"] == pytest.approx(0.5)


def test_material_words_match_on_word_boundaries():
    """Regression: `"ice" in "a nice building"` was True under substring matching.

    Substring matching invented materials no caption named -- "ice" from "nice",
    "sand" from "sandwich", "gold" from "golden" -- which both inflated coverage
    and corrupted the agreement denominator.
    """
    out = fth.material_agreement([_wood_house()], ["a nice building"])
    assert out["agreement"].size == 0
    assert out["coverage"] == 0.0


def test_material_agreement_on_empty_input():
    out = fth.material_agreement([], [])
    assert out["agreement"].size == 0
    assert out["coverage"] == 0.0


# --- end to end (GPU) ------------------------------------------------------
@pytest.mark.slow
def test_real_builds_beat_chance_and_shuffled_captions_do_not():
    """The anchor that makes an observed retrieval number readable."""
    from blockgen.eval.bench import splits

    caps = fth.load_captions("houses_32")
    if not caps:
        pytest.skip("captions not available")
    split = splits.load_split("houses_32")
    idx = list(split.val[:48])
    structs = [splits.load_corpus("houses_32")[0][i].crop_to_non_air() for i in idx]
    prompts = fth.captions_for_indices(caps, idx)

    real = fth.evaluate(structs, prompts, n_distractors=31, n_boot=200,
                        rng=np.random.default_rng(0))
    assert real.retrieval_acc > 3 * real.chance

    shuffled = list(prompts[1:]) + [prompts[0]]
    mism = fth.evaluate(structs, shuffled, n_distractors=31, n_boot=200,
                        rng=np.random.default_rng(0))
    assert mism.retrieval_acc < real.retrieval_acc
