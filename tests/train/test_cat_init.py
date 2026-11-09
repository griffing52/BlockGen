"""init_from across the category extension: a voxel checkpoint grows into the voxels-cat vocabulary."""
import numpy as np
import pytest
import torch

from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.train import archs
from blockgen.train.config import TrainConfig
from blockgen.train.trainer import extend_vocab, init_from_checkpoint
from blockgen.utils.serialize import BOS_TOKEN

TINY = dict(d_model=32, nhead=4, layers=1, ff=64, max_seq_len=96)


def _ckpt(tmp_path, tok, cfg):
    torch.manual_seed(0)
    m = archs.get("ar2").build(cfg, tok)
    p = tmp_path / "pre.pt"
    torch.save({"model": m.state_dict(), "step": 7, "config": cfg.to_dict(), "meta": {"tokenizer": tok.hash}}, p)
    return m, str(p)


def test_extension_preserves_the_pretrained_function(tmp_path):
    base, cat = CorpusTokenizer.default(), CorpusTokenizer.default(cats=True)
    cfg = TrainConfig(**TINY)
    old, path = _ckpt(tmp_path, base, cfg)
    cfg.init_from = path
    new = archs.get("ar2").build(cfg, cat)
    info = init_from_checkpoint(new, cfg, cat.hash, "cpu", tokenizer=cat)
    ext = info["extended"]
    assert ext["old_vocab"] == base.vocab_size and ext["new_vocab"] == cat.vocab_size and ext["from"] == base.hash
    assert set(ext["keys"]) == {"token_embedding.weight", "lm_head.weight", "lm_head.bias"}
    assert info["missing"] == [] and info["unexpected"] == []
    old.eval(); new.eval()
    head = base.prefix({"src": "curated", "q": "4", "size": "32", "win": "whole"})
    b0 = base.vocab.block_offset
    x = torch.tensor([head + [3, 4, 5, b0 + 4, 3, 4, 6, b0 + 9]])
    xc = x.clone(); xc[0, 0] = cat.tag_id("cat", "ship")
    with torch.no_grad():
        a, b = old(x), new(xc)
    assert torch.allclose(a, b[..., :base.vocab_size], atol=1e-6)            # a class token starts as BOS
    emb = new.token_embedding.weight
    assert torch.equal(emb[cat.cat_offset:], emb[BOS_TOKEN].expand(cat.n_cats, -1))


def test_unrelated_tokenizers_are_still_refused(tmp_path):
    cfg = TrainConfig(**TINY)
    _, path = _ckpt(tmp_path, CorpusTokenizer.default(oriented=True), cfg)
    cfg.init_from = path
    cat = CorpusTokenizer.default(cats=True)
    with pytest.raises(ValueError, match="tokenizer"):
        init_from_checkpoint(archs.get("ar2").build(cfg, cat), cfg, cat.hash, "cpu", tokenizer=cat)
    # no tokenizer object (the old call): any hash mismatch is refused as before
    _, path = _ckpt(tmp_path, CorpusTokenizer.default(), cfg)
    cfg.init_from = path
    with pytest.raises(ValueError, match="tokenizer"):
        init_from_checkpoint(archs.get("ar2").build(cfg, cat), cfg, cat.hash, "cpu")


def test_same_tokenizer_does_not_extend(tmp_path):
    tok = CorpusTokenizer.default()
    cfg = TrainConfig(**TINY)
    _, path = _ckpt(tmp_path, tok, cfg)
    cfg.init_from = path
    info = init_from_checkpoint(archs.get("ar2").build(cfg, tok), cfg, tok.hash, "cpu", tokenizer=tok)
    assert "extended" not in info


def test_extend_vocab_leaves_other_tensors_alone():
    cfg = TrainConfig(**TINY)
    m = archs.get("ar2").build(cfg, CorpusTokenizer.default(cats=True))
    small = archs.get("ar2").build(cfg, CorpusTokenizer.default()).state_dict()
    sd, grown = extend_vocab(m, small, 507)
    assert sorted(grown) == ["lm_head.bias", "lm_head.weight", "token_embedding.weight"]
    for k in small:
        if k not in grown:
            assert sd[k] is small[k]
        else:
            assert sd[k].shape[0] == 520 and torch.equal(sd[k][:507], small[k])
