# Golden scorecards

Five committed benchmark cards: four real ones cut down to three arms, and one
whole `bench/2` run directory synthesized from nothing.

**Why they are here.** `outputs/` is gitignored (`.gitignore:27` — `git ls-files
outputs` is empty), so not one scorecard is in the repository, and the only test
that ever checked `catalog.load_scorecard`'s key set — `tests/lab/test_lab_data.py:142`
— is `@pytest.mark.slow` and *skips* on a machine with no bench output. A key
could be added to the reader or dropped from the writer and CI would stay green.
`tests/` is covered by no ignore rule, so these files are the cheap half of a
committed reference card: enough to pin every shape a reader must survive,
without a 15-minute GPU re-run every time a metric changes.

**Nothing here is a benchmark result.** The four legacy cards are excerpts, so
their `blockscore` rankings are of three arms out of eight-to-sixteen and mean
nothing on their own; the synthesized card's numbers were typed, not measured.
Read `results.md` for results.

Everything is produced by [`../make_scorecard_fixtures.py`](../make_scorecard_fixtures.py),
which is committed and is *not* collected by pytest (the filename is not
`test_*.py` and it defines no top-level `test_` function).

## What each file is, and the exact command that made it

Run from the repository root, with the interpreter that has the project
installed (`.venv/bin/python`).

| File | Source run | Command | Pins |
|---|---|---|---|
| `card_legacy_a.json` | `outputs/run_20260803_033522_bench_compare` (8 arms, tier `both`, `houses_32`) | `python tests/fixtures/make_scorecard_fixtures.py --trim outputs/run_20260803_033522_bench_compare --out tests/fixtures/scorecards/card_legacy_a.json` | Shape A: `fidelity`, **no** `geometry_scalars`, **no** `run.blockscore`, `ladder` present, `run` carrying only the five `bench/1` keys — and an **absolute `run.dir`** pointing at a scratchpad outside `outputs/`, which is what proves run identity comes from the directory name and never from `run.dir` (D7) |
| `card_legacy_allcontrol.json` | `outputs/run_20260831_195140_bench` (7 arms, tier `fast`, `houses_32`) | `python tests/fixtures/make_scorecard_fixtures.py --trim outputs/run_20260831_195140_bench --out tests/fixtures/scorecards/card_legacy_allcontrol.json` | Zero submissions → the run-id label fallback (D8). Also the only card whose `blockscore_validation` **fails** two checks, so the Calibration panel has a real red row to render |
| `card_legacy_c.json` | `outputs/run_20260831_225029_bench` (16 arms, tier `both`, `houses_32`) | `python tests/fixtures/make_scorecard_fixtures.py --trim outputs/run_20260831_225029_bench --out tests/fixtures/scorecards/card_legacy_c.json` | Shape C: `run.head_to_head`, `blockscore` + validation, `ladder`, both `fidelity` and `geometry_scalars`, and one arm of each `kind` (control `real_test`, baseline `uniform_random`, submission `native_oriented`) |
| `card_legacy_d.json` | `outputs/run_20260906_190029_bench` (12 arms, tier `fast`, `houses_48`) | `python tests/fixtures/make_scorecard_fixtures.py --trim outputs/run_20260906_190029_bench --out tests/fixtures/scorecards/card_legacy_d.json` | `cost` (on the submission arm **only**, so `sections` must be a union — D28), **no** `fidelity`, and `corpus == "houses_48"`, the one card whose `real_test` drill-down fell through the `CANONICAL_CORPUS` hard-wire |
| `run_20260102_000000_bench_fixture/` | *(none — synthesized)* | `python tests/fixtures/make_scorecard_fixtures.py --synthesize` | `bench/2`: the full `run` block against `scorecard.RUN_KEYS`, `meta.provenance` for **both** origins, `meta.examples`, and a real `examples_4.npz` + `examples_4_manifest.json` pair beside the card |

Check them all — every file parses, `tools.lab.cards.migrate` is total and
idempotent on each, and the `bench/2` card still satisfies `RUN_KEYS`:

```bash
python tests/fixtures/make_scorecard_fixtures.py --verify
```

## Sizes (bytes, as committed)

```
card_legacy_a.json                                           29528
card_legacy_allcontrol.json                                  22972
card_legacy_c.json                                           44254
card_legacy_d.json                                           36506
run_20260102_000000_bench_fixture/scorecard.json             29298
run_20260102_000000_bench_fixture/examples_4.npz              1211
run_20260102_000000_bench_fixture/examples_4_manifest.json    1853
run_20260102_000000_bench_fixture/fixture_ar_4.npz            1144
run_20260102_000000_bench_fixture/fixture_ar_4_manifest.json   593
                            total (excluding this README)   167359  (163 KB)
```

The floor is set by the real arms: `coherence` and `geometry_scalars` are ~3.3 KB
each per arm on a `bench/1` card, so three arms cost ~25–37 KB whatever else is
dropped. Cutting further would mean deleting metric leaves out of a real card,
which would stop it being a faithful example of the shape it exists to pin.

## The synthesized fixture

`run_20260102_000000_bench_fixture/` is a whole run directory built by calling
the real writers — `sc.Scorecard`, `sc.ArmSpec`, `save_house_cache`,
`examples.write_run_examples`, `composite.leaderboard`, `compare.rank_table` —
over hand-built `Metric`s and 4x4x4 huts. It needs no corpus, no GPU and no
renderer, which is what makes it the fixture a contributor can regenerate.

* **Four arms, `k = 2`, eight example rows.** One of each `kind` (control
  `real_test`, baseline `gabled_house`, submission `fixture_ar`) plus
  `real@canon8`, and both `origin`s (`fixture_ar` is backed by the two-build
  `fixture_ar_4.npz` beside the card; the rest are `in_memory` with a recipe).
  Eight rows is exactly `catalog.SMOKE_N`, so the examples cache sits on the
  threshold and proves the smoke-flag exemption (D23) actually fires — beside
  `fixture_ar_4.npz`, which has two builds and *is* flagged.
* **Every BlockScore is the arm's damage `d`, exactly.** Each metric is written
  as `base + d` with a 95% interval of ±1.96, so the control arm's implied
  spread is 1.0 and `composite.leaderboard` returns `0.0 / 2.0 / 3.0 / 6.0`.
  Nothing here was measured; the numbers are legible on purpose.
* **`blockscore_validation` carries `C1` and `C3_real@canon8`.** The nine-check
  dict, including two *failing* checks, lives on `card_legacy_allcontrol`.
* **`compat.missing` is empty**, which is the point of it being a full-tier
  card: `fidelity` only exists on a full-tier run, and `cards.missing()` names
  it. `cost` is on the submission arm only, so `sections()` has to be a union.
* **`meta.source` and `provenance.npz` are repo-relative paths**, so the
  committed card names a file that exists in every checkout.

## The trim rule

`--trim` keeps `schema_version`, `run`, `context`, `ladder` and `warnings`
verbatim, and exactly **three** arms: `real_test`, then one other reference row
(a baseline in preference to a control, so a card that has both keeps the harder
shape), then one submission when the card has one. `run.blockscore` is filtered
to those arms and `run.head_to_head` to a single rank table, itself restricted to
those arms — a table ranking thirteen arms the card no longer contains would be
an internally inconsistent card, and no real run can produce one.

Consequences a test author should know before asserting against these files:

* `card_legacy_c` is **1 submission + 1 baseline + 1 control** post-trim. Its
  three-submission run label (`native_oriented vs agentic_oneshot vs
  pick_n_place`) does **not** survive the trim; `run_label` on the fixture
  returns `native_oriented`. Assert the `+N` join form against a hand-built card.
* `card_legacy_allcontrol` has **two** arms, not three: the source run scored no
  submissions, which is the whole point of that fixture.
* `warnings` are copied verbatim and mention arms that were trimmed away. That
  is deliberate — a warning list is a log of the run, not an index of the card.

## Regenerating

The generator is deterministic. `--trim` only ever copies. `--synthesize` pins
`git_sha`, `host`, both timestamps and the mtimes of the npz manifests it writes
(`FIXTURE_EPOCH`, 2026-01-02T00:00:00Z), so re-running it yields a byte-identical
directory and a review diff shows the schema change instead of this week's commit
hash. Re-running `--synthesize` deletes and rewrites the whole run directory.

`--trim` needs the source run on disk. Those runs are **not** in the repository,
so a contributor without them can regenerate only the synthesized fixture — which
is why it, and not a trimmed card, is the one that carries the `bench/2` contract.

If a shape ever changes and the sources are gone, delete the affected fixture and
say so in the test that used it; do not hand-edit a card. A hand-edited fixture is
indistinguishable from a real one to every test that reads it, and it would pin a
shape no writer has ever emitted.
