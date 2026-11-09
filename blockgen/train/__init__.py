"""The shared trainer (docs/pipeline.md §5): one step-based loop for every arm.

``python -m blockgen.train --config configs/train/<name>.yaml [key=value ...]``

Reads tokenised shards (``blockgen.corpus.shards``), trains a causal transformer
with a warmup-stable-decay schedule, writes full-state checkpoints that record the
tier hashes, tokenizer hash and git sha, evaluates validation NLL per tier,
renders samples, scores them on the benchmark tier the config names (``bench_tier``,
fast by default), and appends one
line to ``outputs/registry.jsonl``. A later stage (anneal, SFT) is a new run that
``init_from`` a checkpoint of the previous one.
"""
