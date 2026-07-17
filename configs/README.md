# Experiment configs

Reproducible, composable YAML configs for the experiment batteries. One file
pins one run; shared knobs live in fragments that runs pull in via `extends`.

## Usage

Pass `--config <name-or-path>` to either entry point. A bare name resolves to a
file anywhere under `configs/` (e.g. `ideas-full` → `experiments/ideas-full.yaml`):

```bash
.venv/bin/python -m blockgen.experiments_ideas     --config ideas-full     --stamp 20260704_x
.venv/bin/python -m blockgen.experiments_overnight --config overnight-full --stamp 20260704_y
```

The config supplies **defaults**; any flag you also pass on the command line
still wins, so one-off tweaks don't need a new file:

```bash
# run ideas-full but only 5 AR epochs (a smoke), everything else from the config
.venv/bin/python -m blockgen.experiments_ideas --config ideas-full --epochs-ar 5
```

The launcher scripts forward extra flags, so `--config` works through them too:

```bash
./scripts/run_ideas.sh --config ideas-pe-only
```

## Layout

Keys are argparse **dest** names (`epochs_ar`, `diff_grid`, `pe_arms`, …). The
folders group fragments by concern; they are *not* YAML nesting — a merged config
is one flat mapping, so every key maps unambiguously to one flag.

| Folder         | What it pins                                              |
|----------------|----------------------------------------------------------|
| `datasets/`    | data-prep knobs (min-dim/blocks, val split, augment, canon) |
| `training/`    | epoch/batch/seq budgets for AR and diffusion             |
| `models/`      | model-variant sweeps (PE arm lists)                      |
| `experiments/` | top-level runs that `extends:` the above + set selectors |

The dataset *selector* (`dataset:` for ideas, `datasets:` for overnight) lives in
the experiment file, since the two entry points name that flag differently; the
`datasets/` fragments carry only the reusable prep knobs.

## Composition

`extends:` lists bases (string or list), each a bare name or path relative to the
file then to `configs/`. Bases merge left-to-right; the current file's own keys
override them. Cycles are detected and error out. Example:

```yaml
extends: [datasets/houses, training/ar, training/diffusion, models/pe-full]
description: Ideas battery — houses, all groups
dataset: gc-houses-large
groups: "pe,ordering,samplers,twostage"
```

## Adding a config

- New sweep of an existing kind → add an `experiments/*.yaml` that `extends`
  existing fragments and overrides a knob or two.
- New reusable knob set → add a fragment under the matching folder, then
  `extends` it. A typo'd key fails loudly (it must match a real flag), so bad
  configs are caught before a run starts, not silently ignored.
