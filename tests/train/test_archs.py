"""The architecture seam: registry, model contract, cache-optional sampling, select hooks."""
import io
import contextlib

import numpy as np
from blockgen.utils.data import Structure
import pytest
import torch
from torch import nn

from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.train import archs
from blockgen.train.config import TrainConfig
from blockgen.utils.serialize import PAD_TOKEN

TINY = dict(d_model=32, nhead=4, layers=1, ff=64, max_seq_len=96)


class Toy(nn.Module):
    """A model that honours only the training half of the contract: forward(input_ids, pad_mask)."""
    def __init__(self, vocab_size: int, max_seq_len: int, d: int = 16):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, input_ids, pad_mask=None):
        return self.head(self.emb(input_ids))


def _toy_arch():
    return archs.Arch(name="toy_test", build=lambda cfg, tok: Toy(tok.vocab_size, cfg.max_seq_len, **cfg.model),
                      loss=archs.loss_ar2, sample=archs.sample_ar2, track="ar")


@pytest.fixture
def toy():
    archs.register(_toy_arch())
    yield "toy_test"
    archs.ARCHS.pop("toy_test", None)


def test_unknown_arch_lists_the_known_names():
    with pytest.raises(KeyError) as e:
        archs.get("no_such_arch")
    assert "ar2" in str(e.value) and "no_such_arch" in str(e.value)


def test_ar2_build_matches_the_config_and_rejects_bad_model_options():
    tok = CorpusTokenizer.default(kind="voxels")
    m = archs.get("ar2").build(TrainConfig(**TINY), tok)
    assert m.max_seq_len == 96 and m.vocab_size == tok.vocab_size and len(m.blocks) == 1
    with pytest.raises(ValueError, match="unknown model option"):
        archs.get("ar2").build(TrainConfig(**TINY, model={"geo_biass": True}), tok)
    with pytest.raises(ValueError, match="duplicates a config field"):
        archs.get("ar2").build(TrainConfig(**TINY, model={"dropout": 0.2}), tok)
    with pytest.raises(TypeError):
        TrainConfig(model=["geo_bias"])


def test_toy_arch_trains_one_step_and_samples_without_a_cache(toy):
    from blockgen.train.sample import Decoder, generate
    from blockgen.train.trainer import build_model
    tok = CorpusTokenizer.default(kind="voxels")
    cfg = TrainConfig(arch=toy, model={"d": 8}, **TINY)
    arch = archs.get(toy)
    model = build_model(cfg, tok)
    assert isinstance(model, Toy) and model.emb.embedding_dim == 8
    seq = np.array(tok.prefix(cfg.sample_tags) + [tok.vocab.coord_id(1), tok.vocab.coord_id(2), tok.vocab.coord_id(3), tok.vocab.block_id_token(1), 2])
    arr = np.full((2, len(seq) + 2), PAD_TOKEN, np.int64); arr[0, :len(seq)] = seq; arr[1, :len(seq) - 1] = seq[:-1]
    t = torch.from_numpy(arr)
    batch = (t[:, :-1], t[:, 1:], t[:, :-1] == PAD_TOKEN)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = arch.loss(model, batch)
    before.backward(); opt.step(); opt.zero_grad()
    after = arch.loss(model, batch)
    assert torch.isfinite(before) and float(after) < float(before)
    dec = Decoder(model)
    assert dec.fell_back and not dec.use_cache                  # decided up front from forward's signature
    logits = dec.next_logits(batch[0][:1, :4])
    assert logits.shape == (1, tok.vocab_size)
    seqs = generate(model, tok, cfg.sample_tags, 3, max_blocks=4, temperature=1.0, top_k=5, seed=0)
    assert len(seqs) == 3 and all(len(s) <= cfg.max_seq_len for s in seqs)


def test_decoder_uses_the_cache_when_forward_offers_it_and_propagates_its_errors():
    from blockgen.train.sample import Decoder, supports_cache
    tok = CorpusTokenizer.default(kind="voxels")
    m = archs.get("ar2").build(TrainConfig(**TINY), tok)
    assert supports_cache(m) and Decoder(m).use_cache

    class Cached(Toy):
        def forward(self, input_ids, pad_mask=None, past=None, return_cache=False):
            if return_cache:
                raise TypeError("a bug inside a real cache path")
            return super().forward(input_ids, pad_mask)
    dec = Decoder(Cached(tok.vocab_size, 96))
    assert dec.use_cache                                        # the signature says yes, so no silent fallback
    with pytest.raises(TypeError, match="real cache path"):
        dec.next_logits(torch.zeros(1, 3, dtype=torch.long))


def test_register_refuses_a_different_arch_under_a_taken_name():
    original = archs.get("ar2")
    assert archs.register(original) is original                 # the same object again is fine
    with pytest.raises(ValueError, match="already registered"):
        archs.register(archs.Arch(name="ar2", build=original.build, loss=original.loss, sample=original.sample))
    assert archs.get("ar2") is original


def test_sample_and_render_uses_the_arch_sample(tmp_path, monkeypatch):
    """A registered arch's own ``sample`` is what the loop draws from; the next-token
    loop is ar2's choice, not the trainer's."""
    import blockgen.curation.houses as houses
    import blockgen.renderer.grid as grid
    from blockgen.train.sample import sample_and_render
    from blockgen.utils.data import Structure
    calls = []

    def toy_sample(model, tokenizer, cfg, n, seed=None):
        calls.append((n, seed))
        ids = np.zeros((3, 3, 3), np.int32); ids[1, 1, 1] = 1
        return [Structure(block_ids=ids, block_data=np.zeros_like(ids), metadata={"corpus": cfg.name})] * n, \
               {"n": n, "seed": 0, "n_empty": 0, "lengths": [5] * n}
    arch = archs.Arch(name="toy_sampler", build=lambda cfg, tok: Toy(tok.vocab_size, cfg.max_seq_len),
                      loss=archs.loss_ar2, sample=toy_sample)
    archs.register(arch)
    rendered, cached = [], []
    monkeypatch.setattr(grid, "save_grid", lambda structs, path, cols=8: rendered.append(len(structs)))
    monkeypatch.setattr(houses, "save_house_cache",
                        lambda structs, max_dim, cache_dir, name, report: cached.append((name, report)) or str(tmp_path / f"{name}.npz"))
    try:
        tok = CorpusTokenizer.default(kind="voxels")
        cfg = TrainConfig(arch="toy_sampler", **TINY)
        model = Toy(tok.vocab_size, 96)
        npz = sample_and_render(model, tok, cfg, str(tmp_path), step=30, n=4, ckpt="best.pt")
    finally:
        archs.ARCHS.pop("toy_sampler", None)
    assert calls == [(4, None)] and rendered == [4]
    name, report = cached[0]
    assert npz.endswith("samples_step30.npz") and name == "samples_step30"
    assert report["checkpoint"] == {"file": "best.pt", "step": 30} and report["arch"] == "toy_sampler"


def test_alibi_samples_through_the_uncached_path():
    from blockgen.train.sample import Decoder, generate
    tok = CorpusTokenizer.default(kind="voxels")
    cfg = TrainConfig(pe="alibi", **TINY)
    m = archs.get("ar2").build(cfg, tok)
    assert not Decoder(m).use_cache
    seqs = generate(m, tok, cfg.sample_tags, 2, max_blocks=3, temperature=1.0, top_k=5, seed=0)
    assert len(seqs) == 2 and all(len(s) >= len(tok.prefix(cfg.sample_tags)) for s in seqs)


def test_sample_structures_returns_cropped_non_empty_builds_and_extra():
    from blockgen.train.sample import sample_structures
    tok = CorpusTokenizer.default(kind="voxels")
    cfg = TrainConfig(sample_max_blocks=4, top_k=0, **TINY)
    m = archs.get("ar2").build(cfg, tok)
    structs, extra = sample_structures(m, tok, cfg, 4, seed=3)
    assert extra["n"] == 4 and extra["seed"] == 3 and len(structs) <= 4
    assert len(extra["lengths"]) == len(structs) + extra["n_empty"]   # empty draws are replaced, and counted
    for s in structs:
        assert s.occupied_mask.any() and s.metadata["corpus"] == cfg.name


def test_sample_structures_replaces_empty_draws_until_n(monkeypatch):
    from blockgen.train import sample as S
    full = Structure(block_ids=np.ones((2, 2, 2), np.int32), block_data=np.zeros((2, 2, 2), np.int32))
    empty = Structure(block_ids=np.zeros((2, 2, 2), np.int32), block_data=np.zeros((2, 2, 2), np.int32))
    seeds = []

    def fake(model, tok, cfg, n, seed):
        seeds.append((n, seed))
        got = [empty if i % 2 == 0 and n > 1 else full for i in range(n)]   # half empty until a draw of one
        return got, [1] * n, None
    monkeypatch.setattr(S, "_draw", fake)
    structs, extra = S.sample_structures(None, CorpusTokenizer.default(kind="voxels"), TrainConfig(**TINY), 4, seed=3)
    assert len(structs) == 4 and extra["n_empty"] == 3
    assert seeds == [(4, 3), (2, 3 + S.TOPUP_SEED_STRIDE), (1, 3 + 2 * S.TOPUP_SEED_STRIDE)]


def test_strip_compiled_prefix_and_load_weights():
    from blockgen.train.trainer import load_weights, strip_compiled
    tok = CorpusTokenizer.default(kind="voxels")
    m = archs.get("ar2").build(TrainConfig(**TINY), tok)
    sd = {"_orig_mod." + k: torch.randn_like(v) if v.is_floating_point() else v for k, v in m.state_dict().items()}
    assert set(strip_compiled(sd)) == set(m.state_dict())
    load_weights(m, sd)
    assert all(torch.equal(m.state_dict()[k], sd["_orig_mod." + k]) for k in m.state_dict())


def test_init_from_refuses_a_partial_match_unless_allowed(tmp_path):
    from blockgen.train.trainer import init_from_checkpoint, run_record
    tok = CorpusTokenizer.default(kind="voxels")
    cfg = TrainConfig(**TINY)
    m = archs.get("ar2").build(cfg, tok)
    sd = dict(m.state_dict())
    dropped = next(k for k in sd if k.startswith("blocks.0."))
    del sd[dropped]; sd["bogus.weight"] = torch.zeros(1)
    torch.save({"model": sd, "step": 5, "config": cfg.to_dict(), "meta": {"tokenizer": "abc"}}, tmp_path / "p.pt")
    cfg.init_from = str(tmp_path / "p.pt")
    with pytest.raises(ValueError, match="1 missing"):
        init_from_checkpoint(m, cfg, "abc", "cpu")
    cfg.allow_partial_init = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = init_from_checkpoint(m, cfg, "abc", "cpu")
    assert info["missing"] == [dropped] and info["unexpected"] == ["bogus.weight"] and info["step"] == 5
    assert info["run"] == str(tmp_path)
    assert "WARNING" in buf.getvalue() and "1 missing" in buf.getvalue() and "1 unexpected" in buf.getvalue()
    with pytest.raises(ValueError, match="tokenizer"):
        init_from_checkpoint(m, cfg, "other", "cpu")
    rec = run_record(cfg, "rd", {"tokenizer": "t", "tiers": {}, "git": "g", "init_from": info}, tok, 5, 10, 100, 1.0, [],
                     {}, {}, None, {}, 10, "last.pt", {"nll": 1.0, "step": 10}, {"ok": False})
    assert rec["init_missing"] == 1 and rec["init_unexpected"] == 1 and rec["parent_run"] == str(tmp_path)


def test_init_from_a_matching_checkpoint_is_clean(tmp_path):
    from blockgen.train.trainer import init_from_checkpoint
    tok = CorpusTokenizer.default(kind="voxels")
    cfg = TrainConfig(**TINY)
    m = archs.get("ar2").build(cfg, tok)
    torch.save({"model": m.state_dict(), "step": 5, "config": cfg.to_dict(), "meta": {}}, tmp_path / "p.pt")
    cfg.init_from = str(tmp_path / "p.pt")
    info = init_from_checkpoint(archs.get("ar2").build(cfg, tok), cfg, "abc", "cpu")
    assert info["missing"] == [] and info["unexpected"] == [] and info["tokenizer"] is None


def test_select_hook_dispatch_and_unknown_select():
    from blockgen.train import trainer
    calls = []

    @trainer.register_select("toy_select")
    def pick(model, run_dir, cfg, best, step, final_nll, device, tokenizer=None):
        calls.append((run_dir, step, tokenizer))
        return {"eval-holdout": 0.123}, 7
    try:
        cfg = TrainConfig(select="toy_select")
        assert trainer.select_checkpoint(None, "rd", cfg, {"nll": 1.0, "step": 1}, 60, {"eval-holdout": 0.9}, "cpu", tokenizer="tok") == ({"eval-holdout": 0.123}, 7)
        assert calls == [("rd", 60, "tok")]
    finally:
        trainer.SELECT_FNS.pop("toy_select", None)
    with pytest.raises(ValueError, match="toy_select|best"):
        trainer.resolve_select("toy_select")
    assert trainer.resolve_select("best") is None and trainer.resolve_select("last") is None


def test_select_reward_resolves_to_the_posttrain_hook(monkeypatch):
    """``select=reward`` is the rft recipe's setting; the trainer must find the hook
    before any training happens, from a process that has not imported posttrain yet.
    The module objects are put back afterwards so other suites keep their identities."""
    import sys
    from blockgen.train import trainer
    monkeypatch.setattr(trainer, "SELECT_FNS", {})
    for mod in [m for m in sys.modules if m.startswith("blockgen.posttrain")]:
        monkeypatch.delitem(sys.modules, mod)
    fn = trainer.resolve_select(TrainConfig(select="reward").select)
    assert callable(fn) and fn.__module__ == "blockgen.posttrain.select"


def test_run_record_carries_the_lineage_fields():
    from blockgen.train.trainer import run_record
    cfg = TrainConfig(name="r", preset="5m", trained_on_signals=["x"], config_path="configs/train/x.yaml", init_from="/runs/p/best.pt")
    meta = {"tokenizer": "t", "tiers": {}, "git": "g", "init_from": {"run": "/runs/p"}}
    rec = run_record(cfg, "rd", meta, CorpusTokenizer.default(kind="voxels"), 5, 10, 100, 1.0, [{"loss": 2.0}],
                     {"eval-holdout": 1.0}, {"eval-holdout": 1.1}, {"eval-holdout": 1.05}, {"periodic": 256, "final": {"eval-holdout": 736}},
                     8, "best.pt", {"nll": 1.0, "step": 8}, {"ok": True, "card": "c.json", "dir": "d"})
    for k in ("arch", "kind", "layout", "augment", "preset", "params", "seed", "dropout", "lr", "pe", "config", "init_from",
              "parent_run", "card", "trained_on_signals", "eval_seqs", "train_split", "val_nll_all", "selected_ckpt"):
        assert k in rec, k
    assert rec["parent_run"] == "/runs/p" and rec["card"] == "c.json" and rec["train_loss"] == 2.0
    assert "dir" not in rec["bench"] and rec["train_split"] == "canonical"
