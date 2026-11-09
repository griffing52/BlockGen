"""scripts/status_matrix.py builds docs/status.md from the repo state plus configs/status_notes.yaml.

Two trees are exercised: a small fixture tree built here (so the test does not
depend on outputs/), and the live repo when its registry exists. Both pages
must carry the eleven sections, the passed date and no em dash.

    PYTHONPATH=. python -m pytest tests/test_status_matrix.py -q
"""
import importlib.util
import json
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "status_matrix.py")
NOTES = os.path.join(ROOT, "configs", "status_notes.yaml")
EM_DASH = "\u2014"


@pytest.fixture(scope="module")
def sm():
    spec = importlib.util.spec_from_file_location("status_matrix", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


@pytest.fixture
def fixture_root(tmp_path):
    """A repo tree with one split tier, one pretraining tier, one shard, two configs, two runs."""
    root = str(tmp_path)
    _write(os.path.join(root, "configs", "corpus", "sft-houses32.yaml"),
           "description: train split\nholdout_split: data/minecraft/splits/split.json\nholdout_parts: [train]\ndedup: false\n")
    _write(os.path.join(root, "configs", "corpus", "anneal.yaml"),
           "description: decay mix\nsources: [worldcrop, grabcraft]\nfilter: \"n_blocks >= 80\"\ndedup: true\nexclude: [eval-holdout]\n")
    _write(os.path.join(root, "configs", "corpus", "vanilla.yaml"),
           "description: village pieces\nsources: [worldcrop]\nfilter: \"kind == 'village'\"\ndedup: true\n")
    _write(os.path.join(root, "data", "minecraft", "splits", "split.json"),
           json.dumps({"train": ["a", "b", "c"], "val": ["d"], "test": ["e"]}))
    tiers = os.path.join(root, "data", "minecraft", "corpus", "tiers")
    _write(os.path.join(tiers, "sft-houses32.json"),
           json.dumps({"name": "sft-houses32", "n": 3, "sha256": "aa", "by_source": {"grabcraft": 2, "3dcraft": 1},
                       "built": "2026-10-14 10:11"}))
    _write(os.path.join(tiers, "anneal.json"),
           json.dumps({"name": "anneal", "n": 100, "sha256": "bb", "by_source": {"worldcrop": 90, "grabcraft": 10},
                       "built": "2026-10-13 13:37"}))
    _write(os.path.join(tiers, "vanilla.json"),
           json.dumps({"name": "vanilla", "n": 5, "sha256": "cc", "by_source": {"worldcrop": 5}, "built": "2026-10-13"}))
    _write(os.path.join(root, "data", "minecraft", "corpus", "shards", "sft-houses32", "deadbeef0000", "spec.json"),
           json.dumps({"tier": "sft-houses32", "tier_sha256": "aa", "tokenizer": "deadbeef0000", "max_blocks": 8192,
                       "max_windows": 32, "n_builds": 3, "n_seqs": 3, "n_tokens": 1200, "seq_len": {"0.99": 500}}))
    _write(os.path.join(root, "data", "minecraft", "corpus", "shards", "sft-houses32", "deadbeef0000-d4", "spec.json"),
           json.dumps({"tier": "sft-houses32", "tier_sha256": "aa", "tokenizer": "deadbeef0000", "augment": "d4",
                       "max_blocks": 8192, "max_windows": 32, "n_builds": 3, "n_seqs": 24, "n_tokens": 9600}))
    _write(os.path.join(root, "data", "minecraft", "corpus", "shards", "anneal", "deadbeef0000", "spec.json"),
           json.dumps({"tier": "anneal", "tier_sha256": "stale-sha", "tokenizer": "deadbeef0000", "max_blocks": 2048,
                       "max_windows": 32, "n_builds": 100, "n_seqs": 150, "n_tokens": 50000}))
    _write(os.path.join(root, "configs", "train", "base.yaml"),
           "description: defaults\npreset: 5m\nlr: 3.0e-4\nbench_n: 64\neval_seqs: 256\n")
    _write(os.path.join(root, "configs", "train", "houses32-sft-5m.yaml"),
           "description: sft\nextends: [base]\nname: houses32_sft_5m\nstage: sft\ntrain: [sft-houses32]\n"
           "val: [eval-holdout]\nsteps: 16000\ndropout: 0.0\nbench_n: 256\n")
    _write(os.path.join(root, "configs", "train", "rft-voxel-5m.yaml"),
           "description: rft\nextends: [base]\nname: rft_voxel_5m\nstage: posttrain\nsteps: 4000\n")
    run_a = "outputs/run_20261015_235516_fixture_sft"
    run_b = "outputs/run_20261031_010000_smoke"
    run_c = "outputs/run_20261028_120000_fixture_d4"   # written before the registry carried augment/select/model
    _write(os.path.join(root, run_a, "run.json"),
           json.dumps({"config": {"dropout": 0.0, "lr": 1e-4, "bench_n": 256, "pe": "phase4", "select": "best"}}))
    _write(os.path.join(root, run_a, "bench", "scorecard.json"),
           json.dumps({"run": {"tier": "fast"}, "arms": {"fixture_sft": {"meta": {"n": 256}}}}))
    _write(os.path.join(root, run_b, "run.json"), json.dumps({"config": {"dropout": 0.0, "bench_n": 16}}))
    _write(os.path.join(root, run_c, "run.json"), json.dumps({"config": {"dropout": 0.1, "bench_n": 64, "augment": "d4"}}))
    rows = [
        {"time": "2026-10-15 23:55", "run": run_a, "name": "fixture_sft", "stage": "sft", "arch": "ar2",
         "preset": "5m", "params": 5525243, "kind": "voxels", "augment": "", "pe": "phase4", "model": {},
         "steps": 16000, "train": ["sft-houses32"], "val": ["eval-holdout"], "tokenizer": "deadbeef0000",
         "config": "configs/train/houses32-sft-5m.yaml", "init_from": "outputs/run_x/last.pt",
         "trained_on_signals": [], "val_nll": {"eval-holdout": 0.42}, "val_nll_all": {"eval-holdout": 0.44},
         "eval_seqs": {"periodic": 256, "final": {"eval-holdout": 736}}, "selected_step": 5500, "select": "best",
         "bench": {"ok": True, "tier": "fast", "blockscore": 323.2}, "card": run_a + "/bench/scorecard.json"},
        {"time": "2026-10-28 12:00", "run": run_c, "name": "fixture_d4", "stage": "sft", "arch": "ar2", "preset": "5m",
         "kind": "voxels", "steps": 4000, "train": ["anneal"], "val": ["eval-holdout"], "tokenizer": "deadbeef0000",
         "val_nll": {"eval-holdout": 0.9}, "bench": {"ok": False, "skipped": True}, "card": None},
        {"time": "2026-10-31 01:00", "run": run_b, "name": "smoke", "stage": "sft", "arch": "ar2", "preset": "5m",
         "kind": "voxels", "steps": 60, "train": ["anneal:1", "sft-houses32:1"], "val": ["eval-holdout"],
         "tokenizer": "deadbeef0000", "val_nll": {"eval-holdout": 3.3}, "bench": {"ok": False, "skipped": True},
         "card": None},
    ]
    _write(os.path.join(root, "outputs", "registry.jsonl"), "\n".join(json.dumps(r) for r in rows) + "\n")
    return root


@pytest.fixture
def notes():
    with open(NOTES) as fh:
        return yaml.safe_load(fh)


def _check_page(sm, page, date):
    assert EM_DASH not in page
    assert f"on {date} " in page.splitlines()[2]
    for sec in sm.SECTIONS:
        assert f"## {sec}" in page, sec
    assert "| Option | Status | Where | Current number |" in page
    assert "### Rewards" in page


def test_fixture_tree(sm, fixture_root, notes):
    repo = sm.Repo(fixture_root)
    page = sm.build_page(repo, notes, "2026-10-31")
    _check_page(sm, page, "2026-10-31")
    # tiers: status comes from the registry, counts from the tier json and the shard
    assert "| anneal | used (train, 2 runs) |" in page
    assert "| sft-houses32 | used (train, 2 runs) |" in page
    assert "| vanilla | unused (no shard) |" in page
    assert "150 seqs" in page and "100 builds" in page
    # a mixture run is counted
    assert "Tier mixtures tier:weight | used (1 run)" in page
    # tokenizer rows: run_c records no augment field, its run.json says d4, so it lands on the d4 sibling
    assert "| voxels (hash deadbeef0000) | used (2 runs) |" in page
    assert "| voxels, augment d4 (hash deadbeef0000) | used (1 run) |" in page
    # counts name the rows that predate a registry field instead of omitting them
    assert "model {} recorded on 1 row, field absent on 2 older rows" in page
    assert "runs by select best 1; no select recorded on 2 older runs" in page
    # a planned rung shows its path as proposed
    assert "| proposed: blockgen/posttrain/grpo.py; " in page
    # the stale anneal shard is reported by the freshness fact
    assert "1 stale: data/minecraft/corpus/shards/anneal/deadbeef0000" in page
    # the per-source row counts reach section 1
    assert "anneal 90" in page
    # configs: base first, the sft config matched by its config path
    lines = page.splitlines()
    i_base = next(i for i, l in enumerate(lines) if l.startswith("| base (defaults"))
    i_sft = next(i for i, l in enumerate(lines) if l.startswith("| houses32-sft-5m |"))
    assert i_base < i_sft
    assert "| houses32-sft-5m | used (1 run) |" in page
    assert "| rft-voxel-5m | unused |" in page
    assert "| stage: posttrain | unused (configs only) |" in page
    # runs: NLL, selected step, score with tier and n, card, status
    run_line = next(l for l in lines if l.startswith("| run_20261015_235516_fixture_sft |"))
    assert "0.420 (all 0.440) @5500" in run_line
    assert "323.2 (fast, n=256)" in run_line
    assert "bench/scorecard.json" in run_line and "| current |" in run_line
    assert "| run_x |" in run_line
    smoke_line = next(l for l in lines if l.startswith("| run_20261031_010000_smoke |"))
    assert "| smoke |" in smoke_line and "| none | none |" in smoke_line
    # the smoke row sits under the wiring-checks table, after the main one
    assert lines.index(smoke_line) > lines.index(run_line)


def test_registry_date_and_check(sm, fixture_root, tmp_path):
    out = str(tmp_path / "status.md")
    assert sm.main(["--root", fixture_root, "--notes", NOTES, "--out", out]) == 0
    page = open(out).read()
    assert "on 2026-10-31 from" in page          # the newest registry line's date, no clock
    assert sm.main(["--root", fixture_root, "--notes", NOTES, "--out", out, "--check"]) == 0
    with open(out, "a") as fh:
        fh.write("\nedited by hand\n")
    assert sm.main(["--root", fixture_root, "--notes", NOTES, "--out", out, "--check"]) == 1


def test_bad_status_word_is_refused(sm, fixture_root, tmp_path):
    bad = {"data_sources": [{"option": "x", "status": "maybe", "where": "", "number": ""}]}
    with pytest.raises(ValueError, match="status_notes.data_sources"):
        sm.build_page(sm.Repo(fixture_root), bad, "2026-10-31")


def test_em_dash_in_notes_is_scrubbed(sm, fixture_root):
    notes = {"data_sources": [{"option": "a " + EM_DASH + " b", "status": "used", "where": "", "number": ""}]}
    page = sm.build_page(sm.Repo(fixture_root), notes, "2026-10-31")
    assert EM_DASH not in page and "| a, b | used |" in page


def test_notes_file_is_clean(notes):
    text = open(NOTES).read()
    assert EM_DASH not in text
    for key in ("data_sources", "curation", "tokens_legacy", "architectures_legacy", "mechanics",
                "posttrain_extra", "controls", "open_decisions"):
        assert notes.get(key), key
    assert notes["runs"]["smoke_names"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "outputs", "registry.jsonl")),
                    reason="live registry not present")
def test_live_repo(sm, notes):
    page = sm.build_page(sm.Repo(ROOT), notes, "2026-10-31")
    _check_page(sm, page, "2026-10-31")
    assert "| sft-houses32 | used (train" in page
    assert "| ar2 (trainer arch, track ar) | used" in page
    assert "Protocol houses32-v1" in page
    assert "| (none) |" not in page.split("## 11.")[0]   # every table before the last has rows
