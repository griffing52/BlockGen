"""View configuration, pooling, and cache invalidation.

The pooling tests are the important ones. `blockgen/eval/perceptual.py` treats
each rendered view as an independent sample, which inflates n fourfold with
correlated rows; measured on real builds that biases the KID null to +8e-5 at
n=16 while the pooled null sits at 0.000. `test_pooled_makes_structure_the_unit`
pins the property that fixes it.

Tests that need a GPU or the renderer are marked `slow`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.eval.bench import features as ft
from blockgen.utils.data import Structure


def _feats(n=8, v=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, v, d))
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def _struct(nx=4, ny=4, nz=4):
    ids = np.zeros((nx, ny, nz), dtype=np.int32)
    ids[:, 0, :] = 4
    ids[0, :, :] = 5
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


# --- view config -----------------------------------------------------------
def test_view_key_is_stable_and_descriptive():
    key = ft.ViewConfig().key()
    assert key == ft.ViewConfig().key()
    assert "px224" in key and "ortho" in key and key.startswith("v4_")


def test_view_key_changes_with_every_camera_field():
    base = ft.ViewConfig()
    assert ft.ViewConfig(px=384).key() != base.key()
    assert ft.ViewConfig(ortho=False).key() != base.key()
    assert ft.ViewConfig(views=((0.0, 30.0),)).key() != base.key()


# --- pooling ---------------------------------------------------------------
def test_pooled_makes_structure_the_unit():
    f = _feats(n=8, v=4)
    assert ft.pooled(f, "mean").shape == (8, 16)
    assert ft.pooled(f, "per_view").shape == (8, 4, 16)


def test_pooled_output_is_unit_norm():
    norms = np.linalg.norm(ft.pooled(_feats(), "mean"), axis=-1)
    assert np.allclose(norms, 1.0)


def test_pooled_is_view_order_invariant():
    """Reordering cameras must not change a single number."""
    f = _feats()
    assert np.allclose(ft.pooled(f, "mean"), ft.pooled(f[:, ::-1], "mean"))


def test_pooled_rejects_concat_and_bad_shapes():
    with pytest.raises(ValueError):
        ft.pooled(_feats(), "concat")
    with pytest.raises(ValueError):
        ft.pooled(np.zeros((4, 8)), "mean")


def test_identical_views_pool_to_themselves():
    one = _feats(n=5, v=1)
    repeated = np.repeat(one, 4, axis=1)
    assert np.allclose(ft.pooled(repeated, "mean"), one[:, 0])


# --- cache keys ------------------------------------------------------------
def test_feature_key_paths_separate_every_dimension():
    a = ft.FeatureKey("houses_32", "val", "v4_px224", "dinov2b")
    for other in (ft.FeatureKey("all_32", "val", "v4_px224", "dinov2b"),
                  ft.FeatureKey("houses_32", "train", "v4_px224", "dinov2b"),
                  ft.FeatureKey("houses_32", "val", "v4_px384", "dinov2b"),
                  ft.FeatureKey("houses_32", "val", "v4_px224", "clipL")):
        assert a.path() != other.path()


def test_structures_sha_is_content_addressed():
    a, b = _struct(), _struct()
    assert ft.structures_sha([a]) == ft.structures_sha([b])
    assert ft.structures_sha([a]) != ft.structures_sha([_struct(5, 4, 4)])
    assert ft.structures_sha([a, b]) != ft.structures_sha([a])


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        ft._load_backbone("resnet50")


# --- cache behaviour (no GPU: monkeypatched embedder) ----------------------
@pytest.fixture
def fake_embed(monkeypatch):
    calls = {"n": 0}

    def _embed(structures, view=ft.ViewConfig(), backbone="dinov2b", device="cuda",
               batch=64, verbose=True):
        calls["n"] += 1
        return _feats(n=len(structures), v=view.n_views, d=8)

    monkeypatch.setattr(ft, "embed_views", _embed)
    monkeypatch.setattr(ft, "render_canary", lambda view, n=4: "canary0")
    return calls


def test_cache_round_trips(tmp_path, fake_embed):
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    structs = [_struct(), _struct(5, 4, 4)]
    a = ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    b = ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    assert fake_embed["n"] == 1, "second call should have hit the cache"
    assert np.allclose(a, b)


def test_cache_invalidates_when_structures_change(tmp_path, fake_embed):
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    ft.load_or_build(key, [_struct()], root=tmp_path, verbose=False)
    ft.load_or_build(key, [_struct(6, 4, 4)], root=tmp_path, verbose=False)
    assert fake_embed["n"] == 2


def test_cache_invalidates_when_renderer_changes(tmp_path, fake_embed, monkeypatch):
    """A silently changed texture pack must not poison every cached number."""
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    structs = [_struct()]
    ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    monkeypatch.setattr(ft, "render_canary", lambda view, n=4: "DIFFERENT")
    ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    assert fake_embed["n"] == 2


def test_cache_meta_records_what_it_depends_on(tmp_path, fake_embed):
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    ft.load_or_build(key, [_struct()], root=tmp_path, verbose=False)
    meta = json.loads(key.meta_path(tmp_path).read_text())
    for field in ("model_id", "view", "structures_sha", "render_canary_sha", "n",
                  "git_sha", "dim"):
        assert field in meta, field


def test_force_rebuilds(tmp_path, fake_embed):
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    structs = [_struct()]
    ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    ft.load_or_build(key, structs, root=tmp_path, verbose=False, force=True)
    assert fake_embed["n"] == 2


def test_corrupt_cache_is_rebuilt_not_fatal(tmp_path, fake_embed):
    key = ft.FeatureKey("toy", "val", "v", "dinov2b")
    structs = [_struct()]
    ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    key.meta_path(tmp_path).write_text("{not json")
    ft.load_or_build(key, structs, root=tmp_path, verbose=False)
    assert fake_embed["n"] == 2


# --- real renderer / GPU ---------------------------------------------------
@pytest.mark.slow
def test_render_canary_is_deterministic():
    view = ft.ViewConfig(px=64)
    assert ft.render_canary(view) == ft.render_canary(view)


@pytest.mark.slow
def test_render_canary_changes_with_view():
    assert ft.render_canary(ft.ViewConfig(px=64)) != ft.render_canary(
        ft.ViewConfig(px=96))


@pytest.mark.slow
def test_embed_views_shape_and_norm():
    feats = ft.embed_views([_struct(), _struct(6, 5, 4)],
                           ft.ViewConfig(px=224), "dinov2b", verbose=False)
    assert feats.shape == (2, 4, 768)
    assert np.allclose(np.linalg.norm(feats, axis=-1), 1.0, atol=1e-4)


# --- the render path must not fail open ------------------------------------
def test_render_views_raises_when_the_renderer_is_broken(monkeypatch):
    """A blank frame scores as a perfect match, so a broken renderer used to
    look like a flawless generator. Worse, `load_or_build` verifies the render
    canary before trusting its cache, so an all-white canary would have silently
    rebuilt the feature cache from blank frames.
    """
    import blockgen.renderer.textured as textured

    def boom(*args, **kwargs):
        raise RuntimeError("simulated EGL context loss")

    monkeypatch.setattr(textured, "render_structure", boom)
    ids = np.zeros((4, 4, 4), dtype=np.int32)
    ids[:, 0, :] = 4
    structs = [Structure(block_ids=ids, block_data=np.zeros_like(ids))] * 4
    with pytest.raises(RuntimeError, match="renderer failed"):
        ft.render_views(structs, ft.ViewConfig(px=32), verbose=False)


def test_render_views_tolerates_a_single_pathological_sample(monkeypatch):
    """A scattered failure on one odd build is not a broken renderer."""
    import blockgen.renderer.textured as textured

    real = textured.render_structure
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("one bad sample")
        return real(*args, **kwargs)

    monkeypatch.setattr(textured, "render_structure", flaky)
    ids = np.zeros((4, 4, 4), dtype=np.int32)
    ids[:, 0, :] = 4
    structs = [Structure(block_ids=ids, block_data=np.zeros_like(ids))] * 32
    out = ft.render_views(structs, ft.ViewConfig(px=32), verbose=False)
    assert len(out) == 32 * ft.ViewConfig().n_views
