"""Best-of-N selection on a fake sampler, the report it writes, and rft's tier naming and command."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.posttrain import bon, rft
from blockgen.posttrain import rewards as rw
from blockgen.utils.data import Structure


def _cube(edge: int, fill: int = 5) -> Structure:
    ids = np.full((edge, edge, edge), fill, np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids), metadata={"corpus": "fake"})


class EdgeReward:
    """Scores a build by its edge, so the winner of every group is known in advance."""
    name = "edge"
    pillar = False

    def score(self, structs):
        return np.array([s.shape[0] for s in structs], dtype=float)


def _cfg(**kw):
    from blockgen.train.config import TrainConfig
    kw.setdefault("dropout", 0.0)
    return TrainConfig(name="fake_run", train=["sft-houses32"], **kw)


def test_select_groups_by_k_and_keeps_the_argmax():
    kept, rows = bon.select([1, 5, 2, 9, 3, 3, 7], k=3)
    assert kept == [1, 3, 6]                     # the short last group still elects a winner
    df = pd.DataFrame(rows)
    assert df.group.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert df["index"].tolist() == list(range(7))
    assert df.kept.sum() == 3 and df.loc[df.kept, "score"].tolist() == [5.0, 9.0, 7.0]
    assert bon.select([2, 2, 1], k=3)[0] == [0]  # ties to the first


def test_bon_run_on_a_fake_sampler(tmp_path, monkeypatch):
    edges = [3, 6, 4, 8, 2, 5, 7, 7]             # n=4 groups of k=2: winners 6, 8, 5, 7

    def fake_sample_structures(model, tokenizer, cfg, n, seed=None):
        assert n == 8 and cfg.temperature == 0.7 and cfg.top_k == 20
        return [_cube(e) for e in edges], {"lengths": [e ** 3 for e in edges]}

    monkeypatch.setattr(bon, "sample_structures", fake_sample_structures)
    tok = CorpusTokenizer.default()
    run_dir = str(tmp_path / "run_20260101_000000_fake")
    os.makedirs(run_dir)
    rep = bon.run(run_dir, n=4, k=2, reward_name="edge", temperature=0.7, top_k=20, ckpt="best.pt",
                  reward=EdgeReward(), loaded=(None, tok, _cfg()))
    out = os.path.join(run_dir, "bon_edge_k2_n4")
    assert os.path.isdir(out) and rep["npz"] == os.path.join(out, f"bon_{tok.max_dim}.npz")
    df = pd.read_parquet(os.path.join(out, "scores.parquet"))
    assert list(df.columns) == ["group", "index", "score", "kept"]
    assert df.loc[df.kept, "score"].tolist() == [6.0, 8.0, 5.0, 7.0]
    assert df.group.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    man = json.load(open(os.path.join(out, f"bon_{tok.max_dim}_manifest.json")))
    r = man["report"]
    assert r["method"] == "bon" and r["k"] == 2 and r["n"] == 4 and r["reward"] == "edge"
    assert r["trained_on_signals"] == ["edge"] and r["n_kept"] == 4 and r["n_drawn"] == 8
    assert r["train_split"] == "canonical" and r["checkpoint"] == os.path.join(run_dir, "best.pt")
    assert r["model"]["dropout"] == 0.0 and r["model"]["d_model"] == 256
    assert r["temperature"] == 0.7 and r["top_k"] == 20 and r["tokenizer"] == tok.hash
    assert [d[0] for d in (it["dims"] for it in man["items"])] == [6, 8, 5, 7]
    assert rep["arms"] == f"bon/fake_run_bon_edge_k2:{rep['npz']}"
    assert rep["score_kept_mean"] == pytest.approx(6.5) and rep["median_tokens"] > 0


def test_bon_report_uses_train_split_and_model_fields_when_the_trainer_has_them():
    tok = CorpusTokenizer.default()
    cfg = SimpleNamespace(name="x", train=["a", "b"], train_split="houses32-v1", model={"flavour": "ar2"},
                          seed=0, temperature=1.0, top_k=40, sample_tags={}, dropout=0.1, arch="ar2")
    r = bon.parent_report("/r", cfg, "last.pt", tok)
    assert r["train_split"] == "houses32-v1" and r["model"]["flavour"] == "ar2" and r["model"]["dropout"] == 0.1
    cfg.train_split = None
    assert bon.parent_report("/r", cfg, "last.pt", tok)["train_split"] == "a,b"


def test_rft_tier_name_k_and_command():
    assert rft.tier_name("m2b_sft_s0", "face_solidity") == "rft-m2b_sft_s0-face_solidity"
    assert rft.k_for(0.25) == 4 and rft.k_for(0.125) == 8 and rft.k_for(1.0) == 1
    with pytest.raises(ValueError):
        rft.k_for(0.0)
    cmd = rft.train_command("outputs/run_x", _cfg(dropout=0.1), "rft-fake_run-lcc26", "a4ca45ac1fe5", "lcc26", "facades")
    assert cmd.startswith("python -m blockgen.train --config configs/train/rft-facades-joint-d4-20m.yaml ")
    assert "init_from=outputs/run_x/best.pt" in cmd and "train=rft-fake_run-lcc26:1,sft-houses32:1" in cmd
    assert "tokenizer=a4ca45ac1fe5" in cmd and "trained_on_signals=lcc26" in cmd and "dropout=0.1" in cmd
    assert "rft-voxel-5m.yaml" in rft.train_command("r", _cfg(), "t", "h", "lcc26", "voxels")


def test_rft_run_writes_the_tier_through_from_structures(tmp_path, monkeypatch):
    edges = [3, 6, 4, 8]
    monkeypatch.setattr(bon, "sample_structures", lambda m, t, c, n, seed=None: ([_cube(e) for e in edges], {}))
    calls = {}

    def fake_from_structures(structures, tier, tok, *, tags=None, source=None, max_blocks=None):
        calls.update(n=len(structures), tier=tier, hash=tok.hash, tags=tags, source=source)
        return str(tmp_path / "shard")

    monkeypatch.setattr(rft, "from_structures", fake_from_structures)
    monkeypatch.setattr(rw, "REWARDS", {**rw.REWARDS, "edge": EdgeReward})
    tok = CorpusTokenizer.default()
    run_dir = str(tmp_path / "run_20260101_000000_fake"); os.makedirs(run_dir)
    rep = rft.run(run_dir, "edge", n=4, keep_frac=0.5, loaded=(None, tok, _cfg()))
    assert calls["n"] == 2 and calls["tier"] == "rft-fake_run-edge" and calls["hash"] == tok.hash
    assert calls["source"] == rep["npz"]
    assert calls["tags"] == {"src": "curated", "q": "4", "win": "whole"}   # size is inferred per build
    assert rep["method"] == "rft" and rep["tier"] == "rft-fake_run-edge" and rep["shard"] == str(tmp_path / "shard")
    assert "train=rft-fake_run-edge:1,sft-houses32:1" in rep["train_command"] and "dropout=0.0" in rep["train_command"]


def test_bon_report_records_the_seed_the_draw_used(tmp_path, monkeypatch):
    seen = {}

    def fake(model, tokenizer, cfg, n, seed=None):
        seen["seed"] = seed
        return [_cube(4)], {"seed": cfg.seed if seed is None else seed, "lengths": [64]}

    monkeypatch.setattr(bon, "sample_structures", fake)
    tok = CorpusTokenizer.default()
    run_dir = str(tmp_path / "run_20260101_000000_fake"); os.makedirs(run_dir)
    rep = bon.run(run_dir, n=1, k=1, reward_name="edge", seed=123, reward=EdgeReward(), loaded=(None, tok, _cfg(seed=7)))
    assert seen["seed"] == 123 and rep["seed"] == 123
    rep = bon.run(run_dir, n=1, k=1, reward_name="edge", reward=EdgeReward(), loaded=(None, tok, _cfg(seed=7)))
    assert rep["seed"] == 7


def test_cli_offers_only_scorable_rewards_and_builds_composites_from_weights():
    import argparse
    for name in ("composite",) + rw.PILLAR_SIGNALS:
        ap = argparse.ArgumentParser(); bon.add_reward_args(ap)
        with pytest.raises(SystemExit):
            ap.parse_args(["--reward", name])
    r = bon.reward_from_args("lcc26", None)
    assert r.name == "lcc26"
    c = bon.reward_from_args("lcc26", "face_solidity:2,lcc26:0.5")   # no on-disk split needed
    assert c.name == "composite-face_solidity@2-lcc26@0.5"
    assert rw.signals_of(c) == ["face_solidity@2", "lcc26@0.5"]
    assert bon.reward_from_args(None, None).name == "face_solidity"
    with pytest.raises(rw.PillarReward):
        bon.reward_from_args(None, "coherence:1")


def test_composite_names_survive_tier_and_arm_specs(tmp_path, monkeypatch):
    """The reward name lands in a tier name, a train spec and an --arms spec: no ':' ',' '/' in it."""
    from blockgen.corpus.shards import _check_derived_name
    from blockgen.eval.bench.runner import parse_arm
    c = rw.get("composite", weights={"lcc26": 0.5, "face_solidity": 2})
    assert c.name == "composite-face_solidity@2-lcc26@0.5" and not any(ch in c.name for ch in ":,/ ")
    tier = rft.tier_name("fake_run", c.name)
    _check_derived_name(tier)
    cmd = rft.train_command("outputs/run_x", _cfg(), tier, "h", c.name, signals=rw.signals_of(c))
    assert f"train={tier}:1,sft-houses32:1" in cmd and "trained_on_signals=face_solidity@2,lcc26@0.5" in cmd
    from blockgen.train.config import TrainConfig, apply_overrides
    cfg = apply_overrides(TrainConfig(name="x", train=["sft-houses32"]),
                          [f"train={tier}:1,sft-houses32:1", "trained_on_signals=face_solidity@2,lcc26@0.5"])
    assert cfg.train_specs()[0] == (tier, 1.0) and cfg.trained_on_signals == ["face_solidity@2", "lcc26@0.5"]
    arm = parse_arm(bon.arm_spec("fake_run", c.name, 8, "/tmp/x/bon_32.npz"))
    assert (arm.track, arm.name, arm.npz) == ("bon", f"fake_run_bon_{c.name}_k8", "/tmp/x/bon_32.npz")
    assert bon.out_dir_for("/r", c.name, 8, 4) == f"/r/bon_{c.name}_k8_n4"


def test_select_by_reward_picks_the_best_checkpoint():
    best = rft.select_by_reward({"a.pt": [_cube(3)], "b.pt": [_cube(9), _cube(2)], "c.pt": []}, EdgeReward())
    assert best == "b.pt"


def test_rft_configs_load_and_match_their_parents():
    from blockgen.train.config import load
    v = load("configs/train/rft-voxel-5m.yaml", [])
    assert (v.stage, v.steps, v.lr, v.select, v.dropout, v.tokenizer, v.kind) == ("posttrain", 4000, 5e-5, "best", 0.0, "60684456e598", "voxels")
    f = load("configs/train/rft-facades-joint-d4-20m.yaml", [])
    assert (f.stage, f.steps, f.lr, f.select, f.dropout, f.tokenizer, f.kind, f.augment) == ("posttrain", 4000, 5e-5, "best", 0.1, "a4ca45ac1fe5", "facades", "")
    assert f.d_model == 512 and f.layers == 8
    for c in (v, f):
        assert c.train == ["sft-houses32"]       # the derived tier arrives on the command line
    for p in ("configs/train/rft-voxel-5m.yaml", "configs/train/rft-facades-joint-d4-20m.yaml"):
        assert "honesty" in open(p).read().lower() and "\u2014" not in open(p).read()


def test_no_em_dashes_in_the_package():
    import glob
    for p in glob.glob("blockgen/posttrain/*.py") + ["blockgen/eval/facade_stats.py", "scripts/facade_sheets.py"] + glob.glob("tests/posttrain/*.py"):
        assert "\u2014" not in open(p).read(), p
