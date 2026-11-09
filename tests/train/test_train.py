import numpy as np
import pytest

from blockgen.train.config import TrainConfig, load, PRESETS
from blockgen.train.trainer import lr_at


def test_wsd_schedule_shape():
    cfg = TrainConfig(steps=1000, warmup=100, lr=1e-3, min_lr_frac=0.1, decay_frac=0.2)
    assert abs(lr_at(0, cfg) - 1e-5) < 1e-12 and abs(lr_at(99, cfg) - 1e-3) < 1e-12
    assert lr_at(500, cfg) == 1e-3 and lr_at(799, cfg) == 1e-3
    assert lr_at(800, cfg) == 1e-3 and abs(lr_at(999, cfg) - 1e-3 * (1 - 0.9 * 199 / 200)) < 1e-12
    cfg2 = TrainConfig(steps=1000, warmup=0, decay_frac=0.0)
    assert lr_at(999, cfg2) == cfg2.lr


def test_config_overrides_and_presets():
    cfg = load(None, ["preset=20m", "steps=5", "train=anneal:3,sft-houses32", "amp=false", "sample_tags={\"src\":\"vanilla\",\"q\":\"4\",\"size\":\"48\",\"win\":\"whole\"}"])
    assert cfg.d_model == PRESETS["20m"]["d_model"] and cfg.steps == 5 and cfg.amp is False
    assert cfg.train_specs() == [("anneal", 3.0), ("sft-houses32", 1.0)]
    assert cfg.sample_tags["src"] == "vanilla"


def test_token_budget_batching():
    from blockgen.train.data import MixedBatches

    class FakeShard:
        spec = {"tier": "fake"}
        def __init__(self, lens):
            self._l = np.array(lens)
        def lengths(self):
            return self._l
        def __len__(self):
            return len(self._l)
        def __getitem__(self, i):
            return np.full(self._l[i], 7, np.uint16)

    mb = MixedBatches([FakeShard([10, 20, 30, 40, 50] * 20)], np.array([1.0]), batch_tokens=128, max_seq_len=45, seed=0)
    it = iter(mb)
    for _ in range(10):
        inp, tgt, pad = next(it)
        assert inp.shape[0] * (inp.shape[1] + 1) <= 128 and inp.shape[1] + 1 <= 45
        # every sequence is whole: its last real token is still there, nothing was truncated
        assert (tgt[:, -1][~pad[:, -1]] == 7).all()


def test_init_from_refuses_dropout_change():
    import pytest
    from blockgen.train.trainer import check_init_compat
    ck = {"config": {"dropout": 0.0}}
    check_init_compat(ck, TrainConfig(dropout=0.0))
    with pytest.raises(ValueError):
        check_init_compat(ck, TrainConfig(dropout=0.1))
    check_init_compat(ck, TrainConfig(dropout=0.1, allow_dropout_change=True))
    check_init_compat({}, TrainConfig(dropout=0.1))        # legacy checkpoint without a config: nothing to check


def test_select_best_checkpoint_restores_earlier_weights(tmp_path):
    import torch
    from blockgen.train.trainer import build_model, select_checkpoint
    cfg = TrainConfig(d_model=32, nhead=4, layers=1, ff=64, max_seq_len=64, val=["eval-holdout"])
    model = build_model(cfg, 507)
    best_sd = {k: torch.randn_like(v) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model": best_sd, "step": 10, "val": {"eval-holdout": 0.5}}, tmp_path / "best.pt")
    final = {"eval-holdout": 0.9}
    # select=last, or best is the final step: nothing happens
    assert select_checkpoint(model, str(tmp_path), TrainConfig(select="last"), {"nll": 0.5, "step": 10}, 60, final, "cpu") == (final, 60)
    assert select_checkpoint(model, str(tmp_path), cfg, {"nll": 0.9, "step": 60}, 60, final, "cpu") == (final, 60)
    ev, step = select_checkpoint(model, str(tmp_path), cfg, {"nll": 0.5, "step": 10}, 60, final, "cpu")
    assert (ev, step) == ({"eval-holdout": 0.5}, 10)
    assert all(torch.equal(model.state_dict()[k], best_sd[k]) for k in best_sd)


def test_load_rejects_unknown_yaml_keys_and_records_the_path(tmp_path):
    import pytest
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nstepz: 5\n")
    with pytest.raises(ValueError, match="stepz"):
        load(str(p), [])
    good = tmp_path / "good.yaml"
    good.write_text("description: fine\nname: x\nsteps: 5\nmodel: {geo_bias: true}\n")
    cfg = load(str(good), [])
    assert cfg.steps == 5 and cfg.model == {"geo_bias": True} and cfg.config_path == str(good)
    with pytest.raises(SystemExit):
        load(str(good), ["stepz=5"])


def test_sample_overrides_coerce_like_config():
    from blockgen.train.config import apply_overrides
    cfg = TrainConfig(bench=True, top_k=40, temperature=1.0)
    apply_overrides(cfg, ["bench=false", "top_k=20", "temperature=0.9", "seed=3", "trained_on_signals=a,b", "train_split=none"])
    assert cfg.bench is False and cfg.top_k == 20 and cfg.temperature == 0.9 and cfg.seed == 3
    assert cfg.trained_on_signals == ["a", "b"] and cfg.train_split is None
    apply_overrides(cfg, ['model={"geo_bias": true}'])
    assert cfg.model == {"geo_bias": True}
    import pytest
    with pytest.raises(SystemExit, match="JSON object"):
        apply_overrides(cfg, ["model=geo_bias"])
    with pytest.raises(SystemExit, match="JSON object"):
        apply_overrides(cfg, ["sample_tags=[1]"])
    assert cfg.model == {"geo_bias": True}


def test_split_name_inference_and_explicit():
    assert TrainConfig().split_name() == "canonical"
    assert TrainConfig(train=["anneal:3", "sft-houses32"]).split_name() == "anneal:3,sft-houses32"
    assert TrainConfig(train_split="houses32-v1").split_name() == "houses32-v1"


def test_existing_configs_load_and_pin_bench_n():
    import glob
    for p in glob.glob("configs/train/*.yaml"):
        cfg = load(p, [])
        assert cfg.bench_n in (16, 64, 256), p          # smoke, base, the recorded M2 and RFT arms
        assert cfg.arch == "ar2" and cfg.model == {}


def _smoke_possible():
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        from blockgen.corpus.shards import find
        find("sft-houses32", None, "voxels"); find("eval-holdout", None, "voxels")
        return True
    except Exception:
        return False


@pytest.mark.slow
def test_smoke_run_writes_the_run_artifacts(tmp_path, monkeypatch):
    """configs/train/smoke.yaml end to end on the GPU (about two minutes): the run dir has
    run.json, metrics.jsonl, samples.png, summary.json, and the registry line carries the
    arch, provenance and the eval_seqs the final pass used. The run lands under tmp_path
    and the registry line in a scratch file, so a test never shows up in the lab's /runs."""
    import json
    import os
    if not _smoke_possible():
        pytest.skip("needs a GPU and the sft-houses32 / eval-holdout shards")
    from blockgen.train import registry
    from blockgen.train.trainer import load_run_checkpoint, train
    monkeypatch.setattr(registry, "PATH", str(tmp_path / "registry.jsonl"))
    cfg = load("configs/train/smoke.yaml", [f"out={tmp_path / 'outputs'}"])
    run_dir = train(cfg)
    assert os.path.basename(run_dir).endswith("_smoke") and run_dir.startswith(str(tmp_path))
    lines = registry.load()
    assert len(lines) == 1 and lines[0]["run"] == run_dir
    _, _, _, info = load_run_checkpoint(run_dir, "last.pt", device="cpu")
    assert info == {"file": "last.pt", "step": cfg.steps, "val": None}
    for f in ("run.json", "metrics.jsonl", "samples.png", "summary.json", "last.pt"):
        assert os.path.exists(os.path.join(run_dir, f)), f
    rec = json.load(open(os.path.join(run_dir, "summary.json")))
    assert rec["arch"] == "ar2" and rec["kind"] == "voxels" and rec["train_split"] == "canonical"
    assert rec["eval_seqs"]["periodic"] == cfg.eval_seqs and rec["eval_seqs"]["final"]["eval-holdout"] > cfg.eval_seqs
    assert rec["val_nll_all"]["eval-holdout"] > 0 and rec["selected_ckpt"] in ("best.pt", "last.pt")
    assert rec["config"] == "configs/train/smoke.yaml" and rec["trained_on_signals"] == [] and rec["samples_png"] is True
    run = json.load(open(os.path.join(run_dir, "run.json")))
    assert run["meta"]["arch"] == "ar2" and run["config"]["bench_n"] == 16
    manifest = json.load(open([os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith("_manifest.json")][0]))
    assert manifest["report"]["checkpoint"]["file"] == rec["selected_ckpt"] and manifest["report"]["model"] == "smoke"
