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
    # Not "resnet50": that used to be the example of an unknown name here, and
    # became a real backbone when the encoder ablation landed.
    with pytest.raises(ValueError):
        ft._load_backbone("not_a_real_backbone")


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
    pytest.importorskip("pyrender")
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
    pytest.importorskip("pyrender")
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


# --- cutaway view protocol (T26) -------------------------------------------
def _hollow_box(n=9) -> Structure:
    """A sealed shell with a large air cavity -- the thing `solidify` destroys."""
    ids = np.zeros((n, n, n), dtype=np.int32)
    ids[0, :, :] = ids[-1, :, :] = 4
    ids[:, 0, :] = ids[:, -1, :] = 4
    ids[:, :, 0] = ids[:, :, -1] = 4
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_empty_cut_spec_is_identity():
    s = _hollow_box()
    assert ft.apply_cut(s, "") is s
    assert ft.apply_cut(s, "none") is s


def test_half_cut_keeps_one_half_of_the_axis():
    s = _hollow_box(9)
    cut = ft.apply_cut(s, "half:y:lo")
    ys = np.argwhere(cut.occupied_mask)[:, 1]
    assert ys.max() < 9 // 2
    assert cut.occupied_mask.sum() < s.occupied_mask.sum()


def test_slab_cut_keeps_exactly_the_requested_thickness():
    cut = ft.apply_cut(_hollow_box(9), "slab:z:50:1")
    zs = np.unique(np.argwhere(cut.occupied_mask)[:, 2])
    assert len(zs) == 1 and zs[0] == 4


def test_quad_cut_removes_the_named_corner_only():
    s = _hollow_box(8)
    cut = ft.apply_cut(s, "quad:hi:hi")
    occ = cut.occupied_mask
    assert not occ[4:, :, 4:].any()          # the removed quadrant
    assert occ[:4, :, :4].any()              # the opposite one survives


def test_a_cut_never_adds_blocks():
    s = _hollow_box()
    for spec in ("half:x:hi", "slab:y:20:1", "quad:lo:hi"):
        assert not (ft.apply_cut(s, spec).occupied_mask & ~s.occupied_mask).any()


def test_a_cross_section_separates_hollow_from_solid_where_the_silhouette_cannot():
    """The positive control, stated as a property of the geometry.

    `solidify` is defined to preserve the silhouette, so the outside of a build
    is literally unchanged and no exterior view can carry the difference. A
    one-voxel cross-section is the opposite: for a hollow box it is a ring, for
    the solidified box it is a filled square. If this ever stops holding, the
    cutaway condition has stopped being a positive control and any null measured
    under it means nothing.
    """
    from blockgen.eval.bench import probes

    hollow = _hollow_box(9)
    solid = probes.solidify(hollow)
    assert np.array_equal(hollow.occupied_mask.any(axis=2),
                          solid.occupied_mask.any(axis=2))   # same silhouette
    a = ft.apply_cut(hollow, "slab:z:50:1").occupied_mask.sum()
    b = ft.apply_cut(solid, "slab:z:50:1").occupied_mask.sum()
    assert b > 2 * a


def test_an_empty_cut_falls_back_to_the_uncut_build():
    """A blank frame scores as a perfect match, so a cut may never empty a build."""
    ids = np.zeros((9, 9, 9), dtype=np.int32)
    ids[:, 0, :] = 4                       # a floor slab only: no upper blocks
    s = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    cut = ft.apply_cut(s, "half:y:hi")
    assert cut.occupied_mask.any()
    assert cut.occupied_mask.sum() == s.crop_to_non_air().occupied_mask.sum()


def test_unknown_cut_spec_raises():
    with pytest.raises(ValueError, match="unknown cut"):
        ft.apply_cut(_hollow_box(), "diagonal:42")


def test_cut_empty_rate_reports_the_fallback():
    ids = np.zeros((9, 9, 9), dtype=np.int32)
    ids[:, 0, :] = 4
    flat = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    assert ft.cut_empty_rate([flat], ()) == 0.0
    assert ft.cut_empty_rate([flat], ("half:y:hi",)) == 1.0
    assert ft.cut_empty_rate([_hollow_box()], ("half:y:hi",)) == 0.0


def test_cuts_must_match_the_view_count():
    with pytest.raises(ValueError, match="one per view"):
        ft.ViewConfig(cuts=("half:y:lo",))


def test_cut_free_view_keys_are_unchanged_so_old_caches_stay_valid():
    """The cut suffix is appended only when there are cuts. Dropping this would
    silently invalidate every feature cache written before cutaways existed."""
    assert ft.ViewConfig().key() == "v4_az45-135-225-315_el30-30-30-30_px224_ortho"
    keyed = ft.ViewConfig(cuts=("half:y:lo",) * 4).key()
    assert keyed.startswith(ft.ViewConfig().key() + "_cut")
    assert keyed != ft.ViewConfig(cuts=("slab:y:50:1",) * 4).key()


@pytest.mark.slow
def test_a_cut_actually_changes_the_rendered_pixels():
    s = _hollow_box(12)
    plain = ft.render_views([s], ft.ViewConfig(views=((0.0, 0.0),), px=64),
                            verbose=False)[0]
    sliced = ft.render_views([s], ft.ViewConfig(views=((0.0, 0.0),), px=64,
                                                cuts=("slab:z:50:1",)),
                             verbose=False)[0]
    assert np.abs(plain.astype(float) - sliced.astype(float)).mean() > 1.0


# --- ablation backbones ----------------------------------------------------
def test_ablation_backbones_are_registered_with_their_dims():
    for name, dim in (("clipB16", 512), ("resnet50", 2048), ("inceptionv3", 2048)):
        assert ft.BACKBONES[name][1] == dim


@pytest.mark.slow
@pytest.mark.parametrize("backbone,dim", [("clipB16", 512), ("resnet50", 2048),
                                          ("inceptionv3", 2048)])
def test_ablation_backbones_embed_to_unit_norm_features(backbone, dim):
    """Every encoder must land L2-normalized, or KID's gamma=1/D means something
    different per arm and the encoder ablation compares feature scales."""
    imgs = [np.full((224, 224, 3), 255, np.uint8),
            np.zeros((224, 224, 3), np.uint8)]
    f = ft.embed_images(imgs, backbone, device="cpu")
    assert f.shape == (2, dim)
    assert np.allclose(np.linalg.norm(f, axis=-1), 1.0, atol=1e-4)
