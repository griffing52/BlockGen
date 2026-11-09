"""Corpus layer: one manifest over every build we own, dedup, and tiers.

See docs/pipeline.md §3. Modules:

- manifest   per-source Parquet tables (+ float16 gate features) built once,
             resumable per month
- dedup      exact (canonical hash) + near (MinHash/LSH) duplicate clusters
- tiers      a tier is a filter over the manifest plus a hash of its member ids
"""
