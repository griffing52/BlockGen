import numpy as np
import pytest

from blockgen.data.worldcrop import anvil, blocks, detect


def test_nibbles_low_first():
    out = anvil._nibbles(np.array([0x21, 0xF0], dtype=np.uint8))
    assert out.tolist() == [1, 2, 0, 15]


def test_yzx_to_xyz_roundtrip():
    a = np.arange(4096, dtype=np.uint16)
    x = anvil._yzx_to_xyz(a)
    # element at (x=1, y=2, z=3) came from flat index y*256 + z*16 + x
    assert x[1, 2, 3] == 2 * 256 + 3 * 16 + 1


def test_unpack_states_padded_and_bitstream():
    bits = 5
    vals = np.arange(4096) % 31
    # padded (1.16+): 12 values per long
    per = 64 // bits
    longs = np.zeros((4096 + per - 1) // per, dtype=np.uint64)
    for i, v in enumerate(vals):
        longs[i // per] |= np.uint64(int(v)) << np.uint64((i % per) * bits)
    got = anvil._unpack_states(longs.astype(np.int64), bits, 4096, padded=True)
    assert np.array_equal(got, vals)
    # bitstream (1.13-1.15)
    stream = 0
    for i, v in enumerate(vals):
        stream |= int(v) << (i * bits)
    nlongs = (4096 * bits + 63) // 64
    longs2 = np.array([(stream >> (64 * k)) & ((1 << 64) - 1) for k in range(nlongs)], dtype=np.uint64)
    got2 = anvil._unpack_states(longs2.astype(np.int64), bits, 4096, padded=False)
    assert np.array_equal(got2, vals)


def test_flat_layers_formats():
    assert blocks.flat_layers("") == [(7, 1), (3, 2), (2, 1)]
    assert blocks.flat_layers("2;7,2x3,2;1;village") == [(7, 1), (3, 2), (2, 1)]
    assert blocks.flat_layers("3;minecraft:bedrock,59*minecraft:stone,3*minecraft:dirt,minecraft:grass;1;") == \
        [(7, 1), (1, 59), (3, 3), (2, 1)]
    assert blocks.flat_layers("3;minecraft:air;1") == [(0, 1)]
    col = blocks.flat_column([(7, 1), (3, 2), (2, 1)])
    assert col[:5].tolist() == [7, 3, 3, 2, 0]


def test_class_tables_disjoint_and_promotions():
    s, w = blocks.class_tables()
    assert not (s & w).any()
    assert s[5] and s[35] and not s[1] and not s[4] and w[4]
    ids = np.array([[[1, 1, 24]]], dtype=np.uint16)
    data = np.array([[[0, 2, 1]]], dtype=np.uint8)
    strong, weak = blocks.classify(ids, data, s, w)
    assert strong.tolist() == [[[False, True, True]]]


def test_cluster_chunks_bridges_gap():
    mk = lambda cx, cz, n: detect.ChunkStat(cx, cz, n, 0, 0, 0, 0, 0, 0)
    stats = {(0, 0): mk(0, 0, 5), (2, 0): mk(2, 0, 5), (10, 10): mk(10, 10, 5), (5, 5): mk(5, 5, 0)}
    groups = detect.cluster_chunks(stats, gap=1)
    assert sorted(len(g) for g in groups) == [1, 2]
    groups0 = detect.cluster_chunks(stats, gap=0)
    assert sorted(len(g) for g in groups0) == [1, 1, 1]


def test_label_and_crop_synthetic(tmp_path):
    """A planks box on a dirt plane inside a fake chunk volume yields one component."""
    ids = np.zeros((32, 32, 32), np.uint16)
    data = np.zeros_like(ids, dtype=np.uint8)
    ids[:, :4, :] = 3  # dirt ground
    ids[8:16, 4:10, 8:16] = 5  # planks box
    ids[20, 4, 20] = 50  # lone torch -> too small
    s, w = blocks.class_tables()
    seed, weak = blocks.classify(ids, data, s, w)
    labels, n = detect.label_tile(ids, data, seed, weak, detect.DetectConfig())
    assert n == 2
    sizes = sorted(int((labels == i).sum()) for i in range(1, n + 1))
    assert sizes == [1, 8 * 6 * 8]


def test_parse_thread_page_extracts_title_author_and_members():
    from blockgen.data.worldcrop.provenance import parse_thread_page
    page = """<html><head><title>X</title></head><body><header><h1>Castle And Village</h1>
    <h2>by MarshmellowTaco</h2></header>
    <section class="description"><p>Description from <a href="http://planetminecraft.com/project/castle/">original post</a>
    (created 2015-08-26 15:13):</p><p>Ok so a castle &amp; a village.</p></section>
    <section class="resources"><h2>Mirrored resources</h2><table><tr><th>Text</th><th>Mirror</th></tr>
    <tr><td>-</td><td><a href="../../201512/planetminecraft.com/maps/downloads/www.dropbox.com/55a/Castle%20Wars.zip">Castle Wars.zip</a></td>
    <td><a href="https://www.dropbox.com/s/abc/Castle%20Wars.zip?dl=0">orig</a></td><td>2 MiB</td><td>archive</td></tr></table></section></body></html>"""
    t = parse_thread_page(page)
    assert t["title"] == "Castle And Village" and t["author"] == "MarshmellowTaco"
    assert t["url"] == "http://planetminecraft.com/project/castle/" and t["created"] == "2015-08-26 15:13"
    assert t["description"] == "Ok so a castle & a village."
    assert t["members"] == ["201512/planetminecraft.com/maps/downloads/www.dropbox.com/55a/Castle Wars.zip"]
    assert t["originals"][0].startswith("https://www.dropbox.com/s/abc/")


def test_gate_features_shape_and_terrain_vs_box():
    from blockgen.data.worldcrop import gate
    names = gate.feature_names()
    # a hollow planks box with a roof: enclosed air, not solid columns
    box = np.zeros((9, 6, 9), np.int32)
    box[:, 0, :] = 5; box[:, 5, :] = 5
    box[0, :, :] = 5; box[-1, :, :] = 5; box[:, :, 0] = 5; box[:, :, -1] = 5
    fb = gate.features(box)
    # a solid dirt mound: every column gap-free, nothing enclosed, no strong blocks
    yy = np.arange(6)[None, :, None]
    mound = np.where(yy <= 2 + ((np.arange(9) % 3)[:, None, None] == 0), 3, 0).astype(np.int32)
    fm = gate.features(mound)
    assert fb.shape == fm.shape == (len(names),)
    i = {n: k for k, n in enumerate(names)}
    assert fb[i["enclosed"]] > 0.3 and fm[i["enclosed"]] == 0
    assert fm[i["solid_col"]] == 1.0 and fb[i["solid_col"]] < 1.0
    assert fb[i["f_strong"]] == 1.0 and fm[i["f_terrain"]] == 1.0
    assert not np.isnan(fb).any() and not np.isnan(fm).any()
