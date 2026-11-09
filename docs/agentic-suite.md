# Agentic methods suite

The methods suite is a harness for agentic structure generation methods. A
*method* is a short Python function that drives one build through a few verbs (ask
an LLM, try a program, render it, dream an image, score, pick the best). Every
expensive or optional dependency (LLM vendor, image generator, renderer, action
representation, context module, scorer, acceptance rule) sits behind a registry
name, and a YAML *recipe* picks one choice per axis. Swapping gpt-image for local
SDXL, or the DSL for a sandboxed Python API, is a one-line YAML change.

Every component has an offline twin, so any recipe runs end to end with no network,
no GPU and no spend (`--mock`). Output goes through the same `report.write_run` as
the legacy battery, so every run is benchmarkable with `python -m blockgen.eval.bench`
unchanged.

The headline method is the **dream loop**: imagine the target as an image (one or
several views, an interior cutaway, a floor plan), build, render the build at the
dream's camera, compare, and iterate.
The **staged methods** (section 5.4) work in steps instead: plan the floor plan
first and build room by room or storey by storey, each unit against its own dream,
or draw all four target views in one turnaround image.

Status (2026-10-03): the suite is implemented and verified offline. One small
real-API pilot of the dream loop exists (entry T41 and its 2026-10-03 follow-up in the
repo-root `results.md`). The staged methods (`blueprint`, `floors`, `turnaround`,
section 5.4) are verified offline only. See the review checklist at the end of this
page.

---

## 1. Concepts

```
             configs/agentic/recipes/<name>.yaml   (extends:, --set overrides)
                                 |
                                 v
  Battery YAML ---------> MethodSpec (recipe) --- recipe_hash (12 hex) ---> manifest
  (recipes, prompts,             |
   budget, sweep)                v   kit.build_kit (once per arm)
                     +----------------------------------------------+
                     | Kit                                          |
                     |  llms[role]  image  actions  renderer        |
                     |  contexts    scorers  final_scorers  accept  |
                     |  dreams      method   ledger (arm)           |
                     +----------------------------------------------+
                                 |
                                 v   episode.run_episode (once per prompt)
                     +----------------------------------------------+
                     | Episode   (what a method talks to)           |
                     |  ask(turn, role)   try_(stage, text)         |
                     |  render(target)    dream(key)                |
                     |  score(att)        rescore_all()   best()    |
                     |  world, dreams, attempts, history, ledger    |
                     +----------------------------------------------+
                         |                          |
                         v                          v
                  MethodResult               Trace (builds/<arm>/NNN/)
              (a BuildResult subclass)       dreams, iters, messages,
                         |                   charges, contact.png
                         v
                 report.write_run  ---> outputs/run_<stamp>_<name>/
                                          <arm>_32.npz + manifest (provenance)
                                          metrics.json, summary.md, samples.png
                         |
                         v
            python -m blockgen.eval.bench --arms agentic/<arm>:<run>/<arm>_32.npz
```

### Glossary

| Term | Meaning |
|---|---|
| **Recipe** (`MethodSpec`) | One experimental arm: method name, LLM per role, image generator, action space, contexts, dreams, scorer weights and gates, accept policy, selection rule, canvas, generation params, method params, budget, cache mode. Loaded from YAML with `extends:`. Hashes to `recipe_hash`; two runs are comparable iff their hashes match. |
| **Method** | A registered function `(ep, **params) -> None` that owns the control flow of one build. Built-ins: `legacy`, `classic`, `dream`, `best_of_n`, `dream_vote`. YAML params are checked against its signature when the kit is built. |
| **Episode** | The object a method talks to. It holds the kit, the current accepted world, the dreams, the attempts, the message history, a per-build ledger and a trace, and centralizes history, rollback, rendering, cost and transcripts. |
| **Kit** | The services a recipe resolves to, built once per arm, so heavy models load once per arm. |
| **Role** | A named LLM slot: `builder`, `planner`, `critic`, `judge`, `dream_critic`. Missing roles fall back to `builder`, so a cheap builder can be paired with a strong judge. |
| **ActionSpace** | How the model writes a build: `dsl`, `blocks`, `python`, `mixed`. Every space lowers to canonical DSL text, the single IR, executed by the legacy `dsl.run_program`. |
| **World** | The executable voxel state (`MinecraftWorld`, a wrapper over the legacy Canvas whose `clone()` also copies the clipboard). |
| **Attempt** | One executed candidate (a subclass of the legacy `Round`): raw reply, cumulative canonical DSL, world snapshot, renders, Score, accepted flag and reason. |
| **View** | A frozen camera (azimuth, elevation, px) plus an optional structure transform: `keep_below` gives cutaways and floor plans, `section` gives vertical halves. |
| **DreamSpec / DreamVersion** | A DreamSpec is data: key, prompt template, the View it is compared against (or none, for context only), weight, the dreams it is conditioned on, extra views to search. A DreamVersion is one generated image; regenerations append versions. |
| **ImageGen** | `generate(ImageRequest) -> ImageResult` plus a price estimate. `openai`, `diffusers` (lazy), `mock`. A content-addressed cache wraps all of them. |
| **Renderer** | `render(structure, views) -> [RGB]`: `textured` (pyrender/EGL, real textures), `flat` (numpy/PIL, no GL), `null` (placeholder). |
| **Scorer / Score** | A scorer returns a term in [0, 1]. The Episode combines terms with weights and gates into a Score stamped with the `dream_id` it was measured against. After a re-dream every attempt is re-scored, so best-of selection compares like with like. |
| **Accept policy** | `(prev, cand) -> (bool, reason)`: which state the loop continues from. **Selection** (`select: last or best`) is separate and decides what the build returns. |
| **Context module** | `(ep) -> ContextParts`: system text (palette, ontology, dream palette) and/or image parts for the first build turn (texture sheet, reference images). |
| **Ledger / Budget / Charge** | Every LLM and image call yields a Charge: paid `usd` (0 on a cache hit), full-price `would_cost_usd`, `priced`, `cached`, `billing`. Ledgers form a tree (build, arm, battery) and check caps before each call. |
| **Cache mode** | `off`, `read_write`, `read_only`. `read_only` raises `CacheMiss` instead of calling out, so a re-run provably costs $0. |
| **Trace** | Per-build directory: dream PNGs, per-iteration structures and renders, scores, full requests, charges, `contact.png`. The replay and offline-rescore substrate. |
| **Battery** | A YAML list of recipes plus a prompt set, an optional sweep and a total budget, run into one run directory. |

---

## 2. Quickstart

### 2.1 Offline, in a few seconds

```bash
.venv/bin/python -m blockgen.agentic list                 # the whole catalogue
.venv/bin/python -m blockgen.agentic show dream-refine    # resolved recipe + recipe_hash
.venv/bin/python -m blockgen.agentic run --recipe dream-multiview --prompts short:2 --mock
```

`--mock` swaps every backend for its offline twin: `mock:auto` LLM (answers every
suite turn with the action space's sample house, or valid JSON for verdict turns),
`mock:render` dreams (textured renders of real **train-split** houses), and `pixel`
in place of the GPU scorers. The renderer is **not** swapped: builds are drawn with
the same textured renderer (`blockgen.renderer.textured`) a real run and
`samples.png` use, since it is local and offline. Only on a machine without the
texture pack or EGL does `--mock` fall back to the `flat` numpy renderer, with a
warning. The run directory gets a `_mock` suffix and the provenance says
`mock: true`. Explicit flags still win, for example `--mock --renderer flat` for
the fastest possible CI run or `--mock --imagegen mock:solid` on a machine without
the house corpus.

A whole battery of every shipped recipe:

```bash
.venv/bin/python -m blockgen.agentic battery smoke-mock --mock
```

### 2.2 Real OpenAI backends

```bash
# 1. what would it cost? (zero network calls; upper bound)
.venv/bin/python -m blockgen.agentic estimate --recipe dream-refine --prompts short:4
# 2. a small paid run, capped at $2 for the whole run
.venv/bin/python -m blockgen.agentic run --recipe dream,classic-matched --prompts short:4 \
    --allow-spend --budget-usd 2
# 3. re-run for free (any cache miss raises instead of calling out)
.venv/bin/python -m blockgen.agentic run --recipe dream,classic-matched --prompts short:4 \
    --cache read_only
```

Keys come from `.env` / the environment (`OPENAI_API_KEY`), as for the legacy track.
Without `--allow-spend`, the first cache miss on a paying backend raises
`SpendNotAllowed` before any network call, and the CLI exits with code 2.

### 2.3 Claude Code CLI (subscription quota, no API key)

```bash
claude --version                        # installed and logged in
.venv/bin/python -m blockgen.agentic run --recipe dream-refine --prompts short:4 \
    --llm-role dream_critic=claude-code:sonnet --allow-spend
.venv/bin/python -m blockgen.agentic run --recipe dream --prompts short:4 \
    --llm claude-code:sonnet --allow-spend          # every role on Claude Code
```

The `claude-code` vendor (alias `cc`) runs `claude -p` in an isolated temporary
workspace: `--safe-mode` (no hooks, plugins, CLAUDE.md or MCP, while keeping
subscription auth), `ANTHROPIC_API_KEY` and the parent session's `CLAUDE_CODE_*`
variables stripped from the child, images written as content-addressed files the
child reads with its Read tool, long system prompts passed via
`--system-prompt-file`. It spends quota, so it is gated like a paid API. Its cost is
notional: `cost_known=False`, and the CLI's `total_cost_usd` is kept in
`LLMResponse.meta` and never summed. The CLI version is part of the cache key, so an
auto-update invalidates the claude-code response cache. Replies are not
deterministic; the cache is what makes reruns reproducible. Concurrency is limited by
a per-binary semaphore; heavy use hits subscription rate limits.

### 2.4 Local diffusers (free images, GPU)

```bash
.venv/bin/pip install diffusers accelerate
.venv/bin/python -m blockgen.agentic run --recipe dream-local --prompts short:4 \
    --allow-spend --dreams-first
```

`dream-local` uses `claude-code:sonnet` for every role and
`diffusers:stabilityai/sdxl-turbo?steps=4` for dreams (native 512 px; FLUX models
resolve to bf16 automatically). Two model aliases exist: `sdxl-turbo`
(`stabilityai/sdxl-turbo`) and `flux-schnell` (`black-forest-labs/FLUX.1-schnell`).
FLUX.1 needs about 34 GiB of VRAM to load fully, so on a 16 GB GPU
`diffusers:flux-schnell` refuses to load (before any download) unless you pass
`offload=true`. For FLUX `offload=true` means sequential CPU offload (submodule by
submodule): it fits a 16 GB card but is slow and needs about 35 GB of system RAM.
For other models `offload=true` is model-level CPU offload. Use
`diffusers:flux-schnell?offload=true&steps=4` only when sdxl-turbo's dreams are not
good enough. Diffusers is imported lazily, so nothing else in the suite needs it. `--dreams-first` generates every prompt's dreams first, then unloads
the pipeline, so the image model and the renderer / scorer models never compete for
the GPU. `--allow-spend` is needed only because of the Claude Code quota; the images
cost nothing (`billing: local`, unpriced).

**Local LLM servers.** `local` and `ollama` default to `max_tokens=4096`, not the
hosted 16000, because vLLM rejects (HTTP 400) any request whose prompt plus
`max_tokens` exceeds its `--max-model-len`. A recipe's `gen.max_tokens` still
overrides the default; if the server answers that the budget is over its context, the
call is retried once with no cap and the server fills the remaining context. Ollama's
OpenAI-compatible endpoint cannot set `num_ctx` and silently truncates long prompts to
the server default, so raise it on the server with `OLLAMA_CONTEXT_LENGTH`.

Two more `gen` keys apply to `local` and `ollama` only, and are part of the cache key:

- `gen.thinking: false` sends `chat_template_kwargs: {enable_thinking: false}`. Hybrid
  models (Gemma 4, Qwen 3) think by default; on a program-writing turn gemma-4-12b spent
  4000 tokens reasoning and returned no answer. `true` forces thinking on; unset leaves
  the template default.
- `gen.vision: false` replaces every image part with the text
  `[image omitted: this model cannot see images]`. A server started without image input
  (vLLM `--limit-mm-per-prompt '{"image": 0}'`) rejects any request with an image, and
  dream, critique and room turns all attach renders, so this is what lets any method
  run on a text-only model. The model is then blind to renders and dreams: use it with
  methods that have no dream (`classic`, `blueprint-nodream`) unless you want that blind
  condition on purpose.

```bash
python -m blockgen.agentic run --recipe blueprint-nodream --prompts short:4 \
    --llm local:gemma-4-12b-it-qat-w4a16-ct --set gen.thinking=false --set gen.vision=false
```

### 2.5 From Python

```python
from blockgen.agentic import load_recipe, run_recipes
run = run_recipes([load_recipe("dream", {"params.max_iters": 2})],
                  ["a small oak cottage with a stone chimney"], name="try", mock=True)
```

The legacy driver also routes here: `python -m blockgen.experiments_agentic
--recipes dream,classic-matched --mock` or `--battery dream-v1`.

---

## 3. The catalogue

`python -m blockgen.agentic list [kind]` prints the live catalogue with every
parameter and its default. As of this writing:

| kind | names (params) |
|---|---|
| method | `legacy(arm, config)`, `classic(plan, n_examples, repair_rounds, critique_rounds, critique_mode, critique_views)`, `dream(refine, refine_min_score, max_iters, stop_at, patience, mode, feedback, redream, redream_patience, redream_dependants, repair_rounds, plan, n_examples, skip_dreams)`, `best_of_n(n, inner, inner_params)`, `dream_vote(n_dreams, **dream params)`, `blueprint(...)`, `floors(...)`, `turnaround(...)` (section 5.4), `inspect(plan, n_examples, repair_rounds, inspect_rounds, inspect_mode, clean_at, show_passed, critique_rounds, critique_mode, critique_views)` (section 5.7) |
| actions | `dsl(commands)`, `blocks(allow_states)` (alias `block_lines`), `python(timeout_s, max_ops, mem_mb, seed, annotate, size)` (alias `pyapi`), `mixed(parts, sub, size)` |
| image | `openai:<model>(quality, size, background)`, `diffusers:<model>(steps, guidance, offload, strength, dtype, size)` (lazy, alias `local-image`), `mock:solid`, `mock:render(split, corpus, renderer, px)`, `mock:dir(path)` |
| renderer | `textured(px, pack_dir, ortho, max_fallback_frac)`, `flat(px, margin, textures)`, `null(px)` |
| context | `palette(skip_if_ontology)`, `ontology(variant, path, seed)`, `examples(k, pool)`, `texture_sheet(blocks, tile, cols, shuffled, seed, face)`, `dream_palette(k, key)`, `reference(note)`, `scale(strict, canvas, dream_cue)` (section 5.5), `lessons(k, path, always, shuffled, seed)` (section 5.7) |
| scorer | `embed(backbone, reduce, crop_fg, baseline, device)`, `silhouette(thresh, px, reduce)`, `color_lab(bins, thresh, reduce)`, `pixel(px, reduce)`, `clip_text(backbone, crop_fg, views, device)`, `material(k)`, `validity()`, `scale(ceiling)` (diagnostic, section 5.5), `vlm(rubric)`, `interior(min_room, furnish_frac, slab_frac, wall_run)` (offline only, section 5.4), `inspection()` (section 5.7) |
| accept | `always`, `nonempty_legacy`, `nonempty(min_frac)`, `improve(eps)`, `improve_valid(eps, max_validity_drop)`, `anneal(t0, decay, seed)` |
| llm (vendors) | `openai`, `gemini`, `anthropic`, `claude-code` / `cc`, `local` (OpenAI-compatible server, `$LOCAL_LLM_URL`), `ollama` (`$OLLAMA_URL`), `mock` (`mock:auto` = AutoMock) |

Views (`list views`): `iso`, `iso_back`, `iso_left`, `iso_right`, `front`, `side`,
`top`, `cutaway` (top 40% removed, steep camera), `plan` (bottom 30%, top-down),
`section_x` (vertical half), `legacy_a`, `legacy_b` (the legacy critique cameras),
`room` (steep camera for a room cutaway) and `room_top` (top-down), used by the staged
methods' unit renders.

Dream presets (`list dreams`): `iso` (weight 1, `view_search` over the other three
iso views), `back`, `left`, `right`, `interior` (scored on `cutaway`), `plan` (scored
on `plan`), all conditioned on `iso`; `mood` (context only, never scored); the
expansion `turnaround` = iso, back, left, right.

Component spec grammar, anywhere a component is named: `"name"`,
`"vendor:model?k=v&k2=v2"`, `{type: name, k: v}` or `{name: {k: v}}`. A typo in a
name or a parameter raises with a "did you mean" suggestion.

---

## 4. Recipes

### 4.1 Fields

| Field | Default | Meaning |
|---|---|---|
| `method` | `classic` | registered method name |
| `description` | | one line, shown by `list recipes` |
| `llm` | `openai:gpt-5-mini` | a spec for every role, or `{builder: ..., judge: ...}` |
| `image` | none | image generator spec (needed when `dreams` is set) |
| `actions` | `dsl` | action space spec |
| `contexts` | `[palette]` | context module specs, in order |
| `renderer` | `textured` | `textured`, `flat`, `null`, or `none` |
| `dreams` | `[]` | preset names or inline DreamSpec dicts |
| `dream_style` | none | replaces `dreams.DEFAULT_STYLE` in every template; a style name (`default`, `render_match`) or literal text |
| `salt` | `""` | non-empty: fresh image seeds and a `[sample <salt>]` system line, so the arm draws new samples (section 5.4); empty keeps the recipe hash |
| `score` | `{}` | `{scorer: weight}` or `{scorer: {weight, type, **params}}` |
| `gates` | `{}` | `{scorer: min term value}`; below it the whole score is 0 |
| `final_score` | `{}` | dream-independent selector used by `select: best` (empty = `score`) |
| `accept` | `nonempty_legacy` | accept policy spec |
| `select` | `last` | `last` or `best` |
| `canvas` | `[32, 32, 32]` | an int or three ints |
| `gen` | `{max_tokens: 16000, temperature: null, reasoning_effort: medium}` | per-call LLM params (same keys as AgentConfig, so legacy cache keys match) |
| `params` | `{}` | method keyword arguments |
| `budget` | `{}` | `usd_per_build`, `max_llm_calls`, `max_images` (per build), `usd_per_arm` |
| `cache` | `read_write` | `off`, `read_write`, `read_only` |
| `seed` | `0` | dream seed when the caller gives none |
| `keep_history` | `true` | accepted turns join the dialogue |
| `target_blocks` | none | scale hint, as in AgentConfig |
| `tags` | `{}` | provenance overrides, e.g. `train_split: oracle_train` |

### 4.2 Inheritance and overrides

- `extends: <name>` (or a list) deep-merges the parents first. Mappings merge key by
  key; **lists and scalars replace**. So `contexts: [palette]` in a child means
  exactly that list, and a child's `params` mapping is merged into the parent's (a
  method switch, such as `best-of-4`, should therefore extend `_base`, not a recipe
  with another method's params).
- Dotted overrides apply last: CLI `--set params.max_iters=5`, battery entries
  `{dream: {as: dream_vdm, params.feedback: vdm}}`. Values are YAML-parsed (`5` is an
  int, `[iso, plan]` a list). `--set llm.judge=X` on a string `llm` keeps the other
  roles on the original model.
- Unknown fields, roles, budget keys, `select` or `cache` values raise. Method
  params are validated against the function's signature when the kit is built,
  before the first call of the first prompt.
- Files whose name starts with `_` are fragments (`_base.yaml`), not recipes.
- Every legacy arm of `experiments_agentic.ARMS` is a virtual recipe
  `legacy-<arm>` (for example `legacy-ont_mined`), with no YAML file needed.

### 4.3 Shipped recipes

| Recipe | Method | What it tests | Its control |
|---|---|---|---|
| `dsl-classic` | legacy | BuildAgent's `full` arm, byte-identical requests and cache keys | itself (parity) |
| `classic-full` | classic | native re-implementation of `full`; change any axis legacy cannot | `dsl-classic` |
| `classic-matched` | classic | no-dream loop with 3 render critiques, best-of by `clip_text` + `validity` | the control for every dream arm |
| `best-of-4` | best_of_n | four independent classic builds, best by the same score | "is it just more samples?" |
| `dream` | dream | one iso dream, build, compare against the matched render, up to 3 iterations | `classic-matched`, `best-of-4` |
| `dream-refine` | dream | a dream critic vets each dream (up to K=2 regenerations) plus one mid-loop re-dream on stall | `dream` |
| `dream-multiview` | dream | iso + back + interior cutaway + floor plan (+ a context-only mood image), feedback = images + judge text | `dream-refine` |
| `dream-vote` | dream_vote | sample 4 iso dreams, the critic picks the most buildable | `dream-refine` (pick vs refine) |
| `dream-python-vlm` | dream | sandboxed Python action space and a VLM judge as a score term | `dream` |
| `mixed-dream` | dream | fenced python (massing) + blocks (touch-ups), patch mode | `dream` |
| `dream-texpalette` | dream | the dream's colours pick a 12-block palette, shown as a texture sheet | `dream` |
| `dream-local` | dream | Claude Code + local SDXL-turbo, no API spend | `dream` (backend swap) |
| `dream-oracle` | dream | the dream is a render of a real train-split house: ceiling and scorer sanity, never ranked | none (upper bound) |
| `dream-plan` | dream | `dream` plus a text plan read from the dream, in blocks at player scale, before the first program | `dream` (same images) |
| `dream-noscale` | dream | `dream` without the `scale` context or dream cue (the pre-2026-10-03 `dream`, same recipe hash) | control for `dream` |
| `dream-describe-before` | dream | a planner writes a detailed design spec first; the dream is drawn from it and the builder sees it (section 5.6) | `dream` |
| `dream-caption` | dream | the judge captions what the dream shows (materials, storeys, roof, block sizes, faces); builder and match critic see it | `dream` (same images), `dream-caption-shuffled` |
| `dream-describe-both` | dream | spec first, image drawn from it, then the image captioned; the builder follows image plus caption | `dream-describe-before` (same images) |
| `dream-caption-shuffled` | dream | `dream-caption` with the caption of a different prompt's dream | token-matched control |
| `dream-grid` | dream | `dream` plus the `grid` context: which axis a flat vertical part (sails, signs) holds fixed, and which build side each TARGET camera face shows (measured: `iso` shows z=max on the left, x=max on the right; x=0 and z=0 are hidden) | `dream` |
| `blocks-oneshot` | classic | one `<block[state]> x y z` line per voxel | `classic-full` one-shot |
| `python-oneshot` | classic | sandboxed Python API, DSL examples auto-converted | `classic-full` one-shot |
| `texsheet-oneshot` | classic | labelled texture contact sheet in the first turn | `texsheet-shuffled` |
| `texsheet-shuffled` | classic | the same sheet with every label on the wrong tile (a derangement) | token-matched control |
| `legacy-<arm>` | legacy | the 10 legacy arms (zeroshot ... ont_stats), virtual | each other, as before |
| `blueprint` | blueprint | plan, deterministic shell, massing-anchored exterior dream, rooms built one by one against their own dreams, roof, exterior pass | `blueprint-nodream` |
| `blueprint-nodream` | blueprint | the same stages with no image at all | `classic_long` (staged-v1) |
| `blueprint-image-plan` | blueprint | the floor plan is dreamed on grid paper and read by a VLM | `blueprint` |
| `blueprint-text-plan` | blueprint | a prose design brief (rooms, sizes, circulation) first, then JSON transcribed from it | `blueprint`, `blueprint-image-plan` |
| `blueprint-noscale` | blueprint | `blueprint` without the `scale` context | control for `blueprint` |
| `floors` | floors | storey-by-storey builds, bottom up, each against a storey dream | `blueprint` (unit granularity) |
| `turnaround` | turnaround | one 2x2 sheet split into the four view targets | `turnaround_perview`, `dream-multiview` |
| `inspect` | inspect | 1 execution repair, then 3 inspect-and-repair rounds: geometry checks (grounded, connected, supported, interior, roofed, enterable, headroom, openings) as TEXT feedback, no images; best-of by `inspection` + `validity` (section 5.7) | `classic-matched` (same turns, visual feedback) |
| `inspect-critique` | inspect | 2 inspect-and-repair rounds, then 1 render critique | `inspect`, `classic-matched` |
| `inspect-lessons` | inspect | `inspect` plus the `lessons` context: curated building rules retrieved by tag for the request | `inspect`, `inspect-lessons-shuffled` |
| `inspect-lessons-shuffled` | inspect | the same number of lessons with non-matching texts (the `always` rules stay) | token-matched control |
| `classic-lessons` | classic | `classic-matched` plus the retrieved lessons | isolates retrieval from inspection |

### 4.4 Batteries

```yaml
# configs/agentic/batteries/dream-v1.yaml
name: dream_v1
prompts: {source: detailed, n: 12, seed: 0}     # n>=256 for leaderboard claims
budget: {usd: 8.0}                              # battery-level cap
cache: read_write                               # applied to every arm
recipes:
  - classic-matched
  - dream
  - dream-refine
  - dream-oracle
  - {dream: {as: dream_vdm, params.feedback: vdm}}    # inline override -> arm 'dream_vdm'
sweep:                                           # cartesian product over ONE base
  base: dream
  grid: {params.max_iters: [1, 3], dreams: [[iso], [iso, interior, plan]]}
  max_arms: 8                                    # more combinations raise, never truncate
```

Sweep arms are named `<base>@<k>=<v>,...` (`dream@max_iters=1,dreams=iso+interior+plan`).
Rules: sweep at most two axes from a fixed base, and label any run with n < 256 a
PILOT. Shipped batteries: `smoke-mock` (every recipe, run with `--mock`), `repr-v1`
(action spaces and the texture sheet vs its control), `dream-v1` (the first dream
pilot), `legacy-parity` (legacy arms through the suite plus `classic-full`), `staged-v1`
(the staged methods and their controls, section 5.4), `scale-plan-v1` (player scale and
text planning against their controls, section 5.5).

```bash
python -m blockgen.agentic battery dream-v1 --only classic-matched,dream --prompts short:4
```

---

## 5. The dream loop

### 5.1 Algorithm (`method: dream`)

```
1. DREAM     for each DreamSpec in dependency order:
               image = gen(template(style, desc=prompt, W/D/H), refs = parent dreams)
               refine: up to K times, the dream_critic reviews it (buildable, on-prompt,
                       single structure, view, consistent with refs; score 0-10);
                       if rejected, regenerate with its revised description
2. BUILD     [plan] -> generate (dreams shown as TARGET images) -> repair (executor errors)
3. MATCH     for i in 1..max_iters:
               stop if the current score >= stop_at
               critic turn: TARGET vs YOURS per scored dream, in the `feedback` channel(s)
               reply is a program  -> try_ (render matched views, score, accept or roll back)
               reply is "REDREAM: <desc>" (if allowed) -> regenerate the primary dream
                                                          (and its dependants), rescore all
               stall counter: if stall >= redream_patience and re-dreams remain, ask the
                 dream critic whether the target is reachable from the best build; if not,
                 re-dream anchored on that build's render
               stop if stall >= patience
4. SELECT    best attempt by final_score (or score) on the FINAL dream set, else last
```

Re-anchoring: every Score records the `dream_id` (a digest of the latest versions of
the scored dreams). After any re-dream, `rescore_all` re-scores the attempts measured
against old dreams, so `select: best` never compares scores against different
targets. `final_score` (for example `clip_text` + `validity`) is dream-independent,
which makes it the right tie-break when the dream itself changed.

### 5.2 Knobs

| Knob | Default | Effect |
|---|---|---|
| `refine` | 0 | K, maximum dream regenerations before building (per dream) |
| `refine_min_score` | 7 | critic verdict (0-10) a dream needs to be accepted |
| `max_iters` | 3 | match iterations after the first build |
| `stop_at` | 0.85 | stop once the current score reaches this |
| `patience` | 2 | stop after this many non-improving iterations |
| `mode` | `rewrite` | `rewrite` (whole program each turn) or `patch` (additions to the accepted build; needs a patchable action space) |
| `feedback` | `raw` | `raw` (TARGET and YOURS images per scored dream), `score` (a score table as text), `vdm` (a judge's list of differences, verbal difference model), `both` (raw + vdm). CaP-X found that raw RGB interleaving can hurt, so this knob tests it. |
| `redream` | 0 | maximum mid-loop re-imaginations (builder REDREAM or stall-triggered) |
| `redream_patience` | 2 | stall length that triggers the reachability check (must be <= `patience`) |
| `redream_dependants` | true | a re-dream also regenerates the dreams conditioned on it |
| `repair_rounds` | 1 | executor-error repair turns after the first build |
| `plan` | false | a natural-language design pass first; with dreams the planner sees them and reads them into blocks (section 5.5) |
| `n_examples` | 0 | example pairs from the action space (when no `examples` context) |
| `select`, `final_score` | recipe fields | what the build returns; `best` takes the argmax of `final_score` (else `score`) over every non-empty attempt, gated attempts rank last, and ties prefer the accepted attempt, then the later one |
| `gates` | recipe field | e.g. `{validity: 0.5}`: a fragmented build scores 0 |
| `dreams`, `dream_style`, `view_search` | recipe fields | which images, in what style, matched against which cameras |
| `budget.max_images` | recipe field | caps dream generation; a refine or re-dream that does not fit is skipped and the build goes on |

Views used for dreams can carry structure transforms (`cutaway`, `plan`,
`section_x`), so an interior or a floor plan is compared with a render of exactly
that part of the build.

### 5.3 What the builder sees

The first build turn carries the context images (texture sheet, reference images),
then each builder-visible dream labelled `TARGET <key>`, then the prompt and the
action space's output rule. Every compare turn carries the execution summary, the
mode instruction, the REDREAM escape hatch (only while re-dream budget remains) and
the output rule. Verdict turns (dream review, reachability, VLM judge, dream vote)
end with a fixed JSON sentence and are parsed tolerantly; an unparseable verdict
never costs a regeneration.

### 5.4 Staged and hierarchical methods

Three methods build in steps instead of in one go: `blueprint` plans the house and
then furnishes it room by room, `floors` builds it storey by storey from the bottom
up, and `turnaround` draws all four target views in a single image. All three are
verified offline only (`--mock`, fakes); no real-API run exists yet.

#### Why staging, and what the dream pilot taught us

The first dream pilot (`outputs/run_20261003_185128_pilot_dream_v1`, see
`results.md`) showed four problems, and each one shaped a design choice here:

| Pilot finding | Cause | Response in the staged methods |
|---|---|---|
| the match loop barely improves (embed band 0.56 to 0.61) | the dream is a soft "clay" picture with its own camera, the render a crisp orthographic view, so most of the distance is style and a rewrite moves the score by noise | **render-anchored dreams**: the exterior dream is an image edit of a render of OUR massing (shell + roof), and each room dream an edit of a render of the empty room, so the target shares the render's camera, framing and silhouette; the unit (one room) is small enough that one turn changes a large part of its image |
| the `plan` dream was just another exterior, and the cutaway was weak | both were conditioned on the iso exterior image, and an edit keeps its reference's composition and camera | a plan or a cutaway is never conditioned on the exterior; it starts from an image that already has the right composition (a schematic grid, a render of the empty room), with the exterior as a style-only second reference |
| the boxed best on the contact sheet looked worse than an accepted neighbour | the header showed the dream score while selection used `final_score` | `score.json` carries `selection`, and the header has a third line (`sel 0.62`, `sel gated`, or `unit 0.41` for a room attempt) |
| `dream` and `dream-refine` were identical | the critic accepted dream v0 and everything after hit the shared cache (by design: a paired comparison) | recipe field `salt` (below) to opt out |

#### Shared vocabulary

- **Blueprint** (`blockgen/agentic/blueprint.py`): a typed plan of footprint,
  storeys, rooms, openings, stairs, roof and materials. Inclusive integer
  coordinates; x right, z toward the front (the front door is on the z1 side), y up.
  A room's rect includes its walls, and neighbouring rooms share a wall line.
- **Repair** (`Blueprint.repaired(size, max_rooms, max_storeys)`): always run, and
  idempotent (a repaired plan repairs to itself). It fixes materials, the roof,
  the footprint (clipped to the canvas), storey heights (3 to 8), room names,
  limits and overlaps (double walls are merged), openings (snapped to a wall cell,
  narrowed off corners, a window never on a door, an exterior door above the ground
  floor becomes a window), stairs (one per storey transition, a ladder if no straight
  run fits) and connectivity (front door, then shared-wall doors). Every change is a
  plain-language note in `bp.notes`, and `validate()` lists the notes without
  changing anything. A plan that is too tall loses top storeys first.
- **Unit**: one room (`blueprint`) or one storey (`floors`). It has a **region**
  (the cells it may write: a room's interior with its floor, minus doorways, the
  stair run and the stairwell), a **view box** (the room with its walls, cut at half
  height, rendered at the steep `room` view) and a runtime dream `u_<name>`.
- **Runtime dreams**: per-unit dreams are traced and metered like any dream, but are
  never part of the global score, `dream_id` or the builder's TARGET prefix.
- **Unit score**: weighted image similarity between a unit's dream and the unit's
  cutaway render (default `silhouette 1.0, color_lab 0.5, embed 1.0`). A unit's
  attempts are accepted on it, not on the global score, because an interior change
  barely moves an exterior iso score.

#### Blueprint format

The planner sees this schema (`blueprint.SCHEMA_TEXT`) and answers with one JSON
object. The worked example below is the one in the prompt, and it passes `validate`
on a 32 x 32 x 32 canvas:

```json
{"footprint": [8, 12, 24, 23], "storeys": [{"height": 4}],
 "rooms": [{"name": "hall", "purpose": "entrance hall", "storey": 0, "rect": [8, 12, 16, 23],
            "desc": "bench, coat hooks, lantern"},
           {"name": "kitchen", "purpose": "kitchen", "storey": 0, "rect": [16, 12, 24, 23],
            "desc": "furnace, crafting table, barrel, table and chairs"}],
 "openings": [{"kind": "door", "storey": 0, "at": [12, 23], "between": ["hall", "outside"]},
              {"kind": "door", "storey": 0, "at": [16, 17], "between": ["hall", "kitchen"]},
              {"kind": "window", "storey": 0, "at": [20, 23], "width": 2,
               "between": ["kitchen", "outside"]}],
 "stairs": [], "roof": {"kind": "gable", "axis": "x", "overhang": 1},
 "materials": {"wall": "spruce_planks", "trim": "oak_log", "roof": "dark_oak_stairs"},
 "style": "a cosy timber cottage with a dark roof"}
```

Storey 0's floor is at y=0 and each storey starts on the one below's ceiling. A stair
starts inside a room at `at` and climbs `height + 1` blocks in `dir`. Roofs: `gable`
(ridge along `axis`), `hip`, `pyramid`, `flat`, `none`. From a plan the module draws
`shell_dsl()` (foundation, floors, walls, corners, partitions, openings, stairs;
blocks headed `# shell: <part> storey <i>`), `roof_dsl()`, and a labelled
`render_schematic()` (one panel per storey, grid and axis labels every 4 cells; the
`blueprint` dream in each trace).

#### Region building

A unit turn runs `Episode.try_(..., region=unit.region, offset=..., accept=...)`:

1. The reply runs on the current build (`patch`) or on an empty world (`rewrite`).
2. Every write outside the region is reverted, and the reverted writes are charged to
   their source lines (through the `# L<n>` tags), at most a few issues plus a
   summary. A reply with reverted writes gets one repair turn that quotes them.
3. The unit's accept rule decides: with a target, the candidate must beat the
   current state's unit score by more than `room_eps` and must not lose more than 0.1
   connectivity (floating furniture games an image score). Without a target (the
   image-free control), the region must gain blocks and connectivity must hold.
4. A unit stops after 2 consecutive rejections (the opening attempt counts).

`local_coords: true` gives the model the room's own frame (origin at the interior
floor corner) and translates its program back. Every unit turn says where the floor
is and which cells to keep clear. In `rewrite` mode it also asks the model to rebuild
the floor, because the region includes it.

Selection: `ep.select_from` is set when the roof starts, so `select: best` only
ranks complete buildings (roof and later), by the dream-independent `final_score`.
After the method returns, every unit is rendered on those attempts, so the boxed
column of the contact sheet shows each room as returned.

#### `method: blueprint` (room by room)

```
1. PLAN      a Blueprint from an LLM (plan_source llm), from a plan dreamed on our grid
             paper and read by a VLM (image), or LLM JSON checked against a pictured
             plan (both); repaired; recorded as the free `blueprint` schematic dream
2. SHELL     foundation, floors, walls, partitions, openings, stairs: the plan's own
             DSL (det) or the builder (llm)
3. EXTERIOR  the recipe's primary dream, an image edit of our massing render (shell +
             deterministic roof), then every score is re-anchored
4. ROOMS     per room (public rooms first): a room dream edited from a cutaway of the
             EMPTY room (+ the exterior for materials), then build / repair / compare
             inside the room's region, accepted on the unit score
5. ROOF      deterministic (det) or the builder (llm), never inside a room
6. FINAL     exterior compare (or critique) rounds that cannot touch the finished rooms
7. SELECT    best of attempts[select_from:] by final_score
```

| Knob | Default | Effect |
|---|---|---|
| `plan_source` | `llm` | `llm` JSON, `image` (grid-paper plan dream read by a VLM; one more image), `both` (LLM JSON verified against a pictured plan), `text` (a prose design brief, then JSON transcribed from it; one more LLM call, section 5.5) |
| `max_rooms`, `max_storeys` | 4, 2 | limits stated to the planner and enforced by repair |
| `plan_retries` | 1 | times the planner is shown its JSON and the problems `validate` lists (or a parse failure) and asked again; repair runs after; unparseable to the end = the default plan |
| `shell` | `det` | `det` (the plan's DSL, free) or `llm` (the builder writes it from the plan) |
| `roof` | `det` | `det` or `llm` |
| `exterior_dream` | true | the anchored exterior dream (the only globally scored dream) |
| `exterior_ref` | `massing` | `massing` (render of shell + roof), `schematic` (the plan drawing), `none` (text only) |
| `refine`, `refine_min_score` | 0, 7 | dream-critic rounds on the exterior dream |
| `interior_dreams` | true | one `u_<room>` dream per room; false = rooms built from text |
| `room_refs` | `[cutaway, exterior]` | references of a room dream, from `cutaway`, `exterior`, `schematic` |
| `room_dream_shuffle` | false | control: every room is scored against ANOTHER room's dream (a rotation); needs `interior_dreams` |
| `room_order` | `public_first` | `public_first`, `blueprint` (as planned), `size` (largest first) |
| `max_room_iters` | 2 | compare (or critique) rounds per room after the first build |
| `room_mode` | `patch` | `patch` or `rewrite` (the reply replaces the region) |
| `local_coords` | false | room-local coordinates in unit turns |
| `room_score`, `room_eps` | default terms, 0.0 | unit score terms and the margin an improvement needs |
| `max_iters`, `stop_at`, `patience` | 2, 0.85, 2 | the final exterior pass |
| `final_feedback` | `both` | `raw`, `score`, `vdm`, `both` (as the dream loop's `feedback`) |
| `final_mode` | `patch` | `patch` or `rewrite` in the final pass |
| `protect_interiors` | true | the roof and the final pass may not write inside rooms |
| `stage_budget` | plan 0.10, shell 0.10, rooms 0.55, roof 0.05, final 0.20 | share of `usd_per_build` per stage; the rooms share is split adaptively (what is left / rooms left); a stage that runs out ends quietly and the build goes on |

#### `method: floors` (storey by storey)

```
1. PLAN      as blueprint (LLM JSON); rooms are listed to the builder if plan_rooms
2. EXTERIOR  the primary dream first (nothing is built yet: text or the schematic)
3. STOREYS   bottom up, per storey: optional deterministic slab + walls (shell det),
             a storey dream (refs: the storey below as built, the exterior), then
             build / repair / compare inside the storey's region
4. ROOF, 5. FINAL, 6. SELECT as blueprint
```

It changes one thing relative to `blueprint`, the unit, so `floors` vs `blueprint`
isolates unit granularity. Knobs: `max_storeys` (2), `plan_rooms` (true),
`plan_retries` (1), `shell` (`none` or `det`), `exterior_dream` (true),
`exterior_ref` (`none` or `schematic`), `refine` (0), `storey_dreams` (true),
`storey_refs` (`[below, exterior]`, from `below`, `exterior`, `schematic`, `cutaway`),
`max_storey_iters` (2), `storey_mode` (`patch`), `local_coords` (false),
`unit_score_terms`, `unit_eps` (0.0), `roof` (`llm`; also `det`, `none`), then
`max_iters`, `stop_at`, `patience`, `final_feedback`, `final_mode` as in `blueprint`.
Default stage shares: plan 0.10, storeys 0.60, roof 0.10, final 0.20.

#### `method: turnaround` (one sheet, four views)

```
1. SHEET   one image: a 2x2 turnaround sheet (runtime dream `sheet`, never scored)
2. REFINE  optional: a critic checks the sheet; regenerate while rejected
3. SPLIT   cut into four tiles (sheets.split_grid: gutters found per row band, the
           backdrop matted to white); a bad split falls back to per-view edits
4. FILL    each tile becomes the next version of its recipe dream (no image charge)
5. BUILD   [plan] -> generate -> repair
6. ASSIGN  cyclic: which tile is which camera, by silhouette IoU against the first
           build's renders over the 8 cyclic orders; done once
7. MATCH   the dream method's match loop (no re-dream: the sheet is the target)
```

Why: per-view dreams are edits of the iso dream and inherit its camera, so the back
and side targets drift; one sheet is drawn in one pass, so its panels show one
building. Knobs: `mode` (`sheet`; `per_view` is today's per-view generation, the
within-method control), `order` (`[iso, iso_left, iso_back, iso_right]`, reading
order), `assign` (`cyclic`; `fixed` trusts the order, `free` scores every key at the
best of all four views), `sheet_ref` (`none`; `iso` dreams the iso view first as the
sheet's reference), `refine` (0), `refine_min_score` (7), `fallback_per_view` (true),
`max_iters` (3), `stop_at` (0.85), `patience` (2), `build_mode` (`rewrite`),
`feedback` (`raw`), `repair_rounds` (1), `plan` (false), `n_examples` (0). The recipe
budget allows 5 images, because a failed split needs 1 + 4.

#### Recipes, controls and what each comparison isolates

Recipes: `blueprint`, `blueprint-nodream`, `blueprint-image-plan`, `floors`,
`turnaround`. Battery `staged-v1` runs them with their controls on 8 detailed prompts:

| Arm | What it is | Compared with | Isolates |
|---|---|---|---|
| `classic-matched` | no dream, no stages, 3 critiques | everything | the baseline loop |
| `classic_long` | `classic-matched` with 12 critique rounds | `blueprint-nodream` | staging vs simply more turns (upper bounds 14 vs 19 LLM calls per build; a stalled room stops early, so real counts are closer) |
| `blueprint-nodream` | the same stages, text-only room turns, no images | `classic_long` | value of the plan and the stages without any image |
| `blueprint` | full: plan, anchored exterior dream, room dreams, room match loops | `blueprint-nodream` | value of the images given the stages |
| `blueprint_extonly` | `interior_dreams: false` | `blueprint` | value of the interior dreams specifically |
| `blueprint_shuffled` | `room_dream_shuffle: true` | `blueprint` | whether the RIGHT room dream matters, or just an image and more turns |
| `blueprint_llmshell` | `shell: llm` | `blueprint` | deterministic scaffolding vs an LLM-written shell |
| `blueprint-image-plan` | the plan dreamed on grid paper, read by a VLM | `blueprint` | an image-made plan vs an LLM JSON plan |
| `floors` | storey units | `blueprint` | unit granularity |
| `turnaround` | one 2x2 sheet, split | `dream-multiview`, `turnaround_perview` | single-image consistency vs per-view edits |
| `dream` vs `dream_rm` | `dream_style: render_match` | each other | the render-matched dream style |
| `dream` vs `dream-noscale` | the `scale` context (rules + dream cue) | each other | player scale, section 5.5 |
| `blueprint-text-plan` | `plan_source: text` | `blueprint`, `blueprint-image-plan` | a reasoned prose plan vs direct JSON vs a dreamed plan |

Every staged arm and its controls share `final_score: {clip_text: 1.0, validity:
0.5}`, `select: best`, the same LLM per role and the 32^3 canvas. Interiors are
measured offline for every arm with the `interior` scorer and are never optimized in
the loop:

```bash
python -m blockgen.agentic rescore outputs/run_<stamp>_staged_v1 --scorers interior
```

`interior` (`recipes/staged_scoring.py`): enclosed air is air that is sealed from the
canvas border, or covered by a block along all six axis rays; `rooms` = enclosed-air
pockets of at least 8 voxels; `furnish` = blocks touching enclosed air that are not
hull, slab, partition or doorway separators. Value = 0.5 min(1, rooms / 2) + 0.5
min(1, furnish / (0.06 enclosed air)). A solid block and an open platform score 0, a
bare shell 0.5. Report `would_cost_usd` and `n_llm_calls` next to every number.

#### `salt` and `dream_style: render_match`

- `salt: "<text>"` (recipe field) shifts every image seed and appends `[sample
  <salt>]` to the builder and reviewer system messages, so the arm draws fresh image
  and LLM samples. Empty keeps today's recipe hash. Use it to measure sampling
  variance. Do not salt arms you want paired: shared dreams make a paired
  comparison, which is the better design at small n.
- `dream_style: render_match` names a style that asks for a flat, evenly lit,
  orthographic-looking render on white. The default style is unchanged (switching it
  would change every existing dream recipe's requests); the staged recipes opt in,
  and `staged-v1` A/Bs it on `dream`.

#### Cost

`estimate --recipe blueprint,floors,turnaround,dream,classic-matched --prompts
short:12` (upper bound: every loop runs to `max_iters`, every repair fires;
gpt-5-mini roles, gpt-image-1-mini low quality; zero network calls):

| Arm | LLM calls / build | Images / build | Upper-bound USD / build | Recipe cap |
|---|---|---|---|---|
| `blueprint` | 21 | 5 | 0.27 | 0.80 |
| `floors` | 14 | 3 | 0.18 | 0.60 |
| `turnaround` | 5 | 1 | 0.06 | 0.40 |
| `dream` | 5 | 1 | 0.06 | 0.30 |
| `classic-matched` | 5 | 0 | 0.05 | 0.25 |

`estimate --battery staged-v1` (8 prompts, 14 arms): $19.52 upper bound for 112
builds, inside the battery's $25 cap. Run `estimate` again before any paid run;
prices and loop lengths change.

#### Writing a staged variant

A new ordering of the stages (for example rooms first, exterior dream last) is one
function in a file of your own, composed from the stage helpers in
`blueprint.py` (`plan_stage`, `exterior_stage`, `stage_shares`) and
`steps_staged.py` (`build_shell`, `room_units`, `unit_dream`, `build_unit`,
`build_roof`, `final_match`, `render_units`). It does not need to live in the repo:
`--plugin` imports it for any command.

```python
# my_variants/rooms_first.py
from blockgen.agentic.core import EpisodeLike, register
from blockgen.agentic.recipes import steps_staged as ss
from blockgen.agentic.recipes.blueprint import exterior_stage, plan_stage, stage_shares

@register("method", "rooms_first", prefetch_dreams=False)
def rooms_first(ep: EpisodeLike, *, max_rooms: int = 4, max_room_iters: int = 2,
                max_iters: int = 2) -> None:
    """Plan, shell, rooms against the plan only, roof, THEN the exterior dream."""
    shares = stage_shares(None)
    ep.inject_first_turn = False                 # unit turns carry their own context
    bp = plan_stage(ep, source="llm", max_rooms=max_rooms, max_storeys=2, retries=1,
                    frac=shares["plan"])
    with ss.stage(ep, "shell", shares["shell"]):
        ss.build_shell(ep, bp, how="det")
    units = ss.room_units(bp, order="public_first", max_units=max_rooms)
    with ss.stage(ep, "rooms", shares["rooms"]):
        for u in units:
            target = ss.unit_dream(ep, bp, u, refs=["cutaway", "schematic"])
            ss.build_unit(ep, bp, u, target=target, iters=max_room_iters, mode="patch",
                          local_coords=False, terms={}, eps=0.0, feedback="both")
    ep.select_from = len(ep.attempts)            # rank complete buildings only
    protect = ss.interior_boxes(bp)
    with ss.stage(ep, "roof", shares["roof"]):
        ss.build_roof(ep, bp, how="det", protect=protect)
    exterior_stage(ep, bp, ref="massing", refine=0, refine_min_score=7)  # + rescore
    ss.fresh_dialogue(ep)
    with ss.stage(ep, "final", shares["final"]):
        ss.final_match(ep, max_iters=max_iters, feedback="both", mode="patch",
                       protect=protect)
    ss.render_units(ep, units, start=ep.select_from)   # room rows on the contact sheet
```

Its recipe extends `_base` and copies the `blueprint` recipe's fields except
`method` and `params` (a recipe path works wherever a name does):

```bash
python -m blockgen.agentic run --plugin my_variants/rooms_first.py \
    --recipe my_variants/rooms-first.yaml,blueprint --prompts short:2 --mock
python -m blockgen.agentic estimate --plugin my_variants/rooms_first.py \
    --recipe my_variants/rooms-first.yaml,blueprint --prompts short:12
```

Four details matter. `prefetch_dreams=False` stops `--dreams-first` from buying the
recipe's dreams before the method can anchor them. `ep.select_from` set before the
roof keeps half-built houses out of `select: best`. `exterior_stage` rescores every
earlier attempt, because they were scored before the exterior dream existed (the
contact sheet shows the rescored values). `render_units` draws each room on the
attempts after `select_from`, so the boxed column shows the rooms as returned.
`--plugin` also sets `BLOCKGEN_AGENTIC_PLUGINS` (`os.pathsep`-separated), so the
Python API and child processes load the same file; `argv` in the provenance records
the flag. A quick knob sweep needs no code at all:
`run --recipe blueprint --mock --set params.room_order=size --set params.max_room_iters=3`.

#### Ideas to try next

1. **Furniture library.** Give room turns a small catalogue of multi-block furniture
   (bed, table and chairs, bookshelf wall) as named macros, so a turn spends its
   tokens on layout instead of block-by-block geometry.
2. **Room dreams from the plan, not the exterior.** Condition each room dream on the
   schematic with the room highlighted plus the empty-room render only, and A/B the
   exterior style reference (`room_refs`).
3. **Facade units.** Treat each wall face as a unit with its own elevation dream
   (front, side), scored on an orthographic `front`/`side` render.
4. **Plan critics.** A VLM checks the schematic against the prompt ("a castle with a
   great hall") before any block is placed; cheap, and errors in the plan are the
   most expensive to fix later.
5. **Depth from the turnaround.** The four consistent tiles of a sheet are enough for
   a rough visual hull; seed the build with it in patch mode.
6. **Per-unit best-of-n.** Sample n furnishings per room in parallel and keep the best
   by unit score; rooms are independent, so this parallelizes well.

### 5.5 Player scale and text planning

A model that is never told the unit sizes a house by eye. The common failure is a
build that looks right from outside and is a doll's house inside: 1-block doorways,
2-block ceilings, 2 x 2 rooms. Since 2026-10-03 the suite states the unit.

#### The `scale` context

`scale(strict=false, canvas=true, dream_cue=true)` (`context.py`, text in
`scale.py`) adds player-scale rules to the system prompt: a player is 2 blocks tall
and 1 wide; doors 1 x 2 (double doors 2 wide), their tops 2 above the floor; at
least 3 blocks of clear ceiling, so a storey is 4-5 blocks floor-to-floor; rooms
measured as open floor inside the walls, at least 3 x 3 (bedroom about 4 x 5 to
6 x 6, hall or living room 6 x 8 or more); corridors 1 wide or better 2; windows
from 1 block above the floor, 1-2 tall; stairs 1 up per 1 across with at least 2
blocks of air above every step; furniture 1 block; outside size follows the rooms
(a 2-3 room cottage is about 10-16 blocks per side, walls included, 1-2 storeys).
`canvas` adds what the canvas holds at this scale and the rule to simplify (fewer
rooms, storeys, a smaller footprint) rather than shrink doors or ceilings; for a
method with its own storey cap (`max_storeys`: blueprint and floors, 2 by default)
it offers that many storeys instead of the canvas's count. `strict` states the rules
as hard constraints.

It is system text, so every role with the builder system prompt (builder, planner,
critic) gets it, in every action space and method. Three more places read it from
the recipe's `contexts`; each adds its text only with `scale`, so the switch itself
is byte-neutral. One change is not tied to it: the dream-reading plan turn (below)
replaces the classic plan text for ANY recipe with dreams and `plan: true`, so
`dream-noscale --set params.plan=true` or a user dream recipe with `plan: true`
sends new bytes and misses its old cache. No shipped recipe other than `dream-plan`
combines the two, so the shipped controls are unaffected.

| where | what it adds |
|---|---|
| `turns.plan` | one line: size it for a 2-tall player, footprint from the rooms it needs; in the dream-reading form instead: keep the TARGET's proportions and set the scale by counting its doors and storeys (the build is scored against that image) |
| blueprint planner turns (`blueprint_json`, `blueprint_fix`, `plan_extract`, `design_brief`) | a paragraph with only the blueprint-unit facts (the system prompt already has the rules): rects include walls, so rooms at least 5 x 5, a bedroom about 6 x 7 to 8 x 8; a hall or corridor is a room like any other, at least 5 wide (the repair drops narrower rects); storey `height` 3 or more |
| dream prompts (`dreams.render_prompt`) | with `dream_cue` (default): "Sized for a Minecraft player: doors are two blocks tall and one block wide, ceilings three blocks high, windows one block above the floor", after the style, so iso, room and storey dreams all carry it |

The dream cue describes the openings and never asks for a person: the style says
"no people", and a figure would be a second structure for the silhouette and
embedding scores to match.

Shipped recipes with `scale`: `dream` (and everything extending it), `dream-texpalette`,
`blueprint`, `blueprint-image-plan`, `blueprint-nodream`, `blueprint-text-plan`,
`floors`, `turnaround`, `classic-matched`, `best-of-4`. Without it: `classic-full`
(the parity arm of `legacy-parity`), `dsl-classic`, the `legacy-*` arms, the
representation battery's one-shot recipes, and the controls `dream-noscale` and
`blueprint-noscale`. In `classic`'s legacy-prompt mode `scale` is an additive context
(`LEGACY_ADDITIVE_CONTEXTS`): BuildAgent's exact system bytes, then the scale text,
nothing else changes; so `classic-matched` keeps the legacy prompts and carries the
same rules as the dream arms it controls for.

Recipe hashes that changed on purpose: `dream` fc8963b57243 -> 6974c014229e,
`dream-multiview` ce2a800873a3 -> 69253ce81bcf, `classic-matched` 59523f8f03dd ->
5557761f11dd, and every recipe extending them, plus `blueprint`, `blueprint-image-plan`,
`blueprint-nodream`, `floors`, `turnaround`, `best-of-4` and `dream-texpalette`. The
controls keep the old hashes (`dream-noscale` = old `dream`, `blueprint-noscale` = old
`blueprint`), and `dream-noscale` sends the old `dream` requests, so it reuses images
and replies cached before the change. The blueprint planner turns changed for every
arm (one schema sentence), so old blueprint plan replies miss the cache.

Blueprint repair: `MIN_ROOM_SIDE` is now 5 (rect including walls, a 3 x 3 interior;
was 4). Smaller rooms are dropped, as before. Storey `height` is the clear air
between slabs, and its minimum was already 3, the player-scale ceiling, so it is
unchanged. Repair stays idempotent.

#### Planning in text

Two ways to let the model reason about the building in words before it commits to
blocks:

- **`dream-plan`** (`dream` + `params.plan: true`): the plan turn shows the TARGET
  dream(s) and asks the planner to read them into blocks: what the image shows
  (massing, storeys, roof, openings, materials), footprint and storey floor heights,
  palette, parts with coordinate ranges, the door and windows with coordinates, two
  details, plus the dream form of the scale line. The build turn gets the plan as
  "Your plan". Its
  dream prompts are `dream`'s, so the shared cache gives both arms the same images
  and the difference is the plan turn. Without dreams, `turns.plan` is unchanged.
- **`blueprint-text-plan`** (`plan_source: text`): turn 1 (`design_brief`, tag
  `DESIGN BRIEF`) asks for prose under fixed headings: footprint, storeys and their
  clear heights, rooms with sizes in blocks and shared walls, circulation (how a
  player walks from the front door to every room, where the stairs go), openings,
  roof and materials, with an arithmetic check. Turn 2 is the usual JSON turn with
  the brief shown and the instruction to transcribe it faithfully; the fix turn shows
  the brief too, and repair runs as for `llm`. One extra LLM call, no image. `llm`
  (direct JSON) and `image` (dreamed plan) are unchanged.

Plans are kept in the trace: `plan.txt` (any `plan` turn), `plan_brief.txt` (the
brief), events `plan_text` / `design_brief`, and `result.plan_text` (so the run's
`<i>_plan.txt` files). The offline `mock:auto` LLM answers a plan turn (it contains
`turns.PLAN_HEAD`) and a brief turn with prose (`mock_plan`, `mock_brief`); the brief
describes the same house as `mock_blueprint`, which answers the JSON turn.

#### The `scale` diagnostic

`scale.scale_stats` reads the final voxel grid. It is a heuristic and is reported,
never selected on:

| metric | meaning |
|---|---|
| `scale_cells` | covered standing cells: a passable cell on a solid block with a solid block somewhere above it (inside or under a roof), minus sloped low cells (ceiling not level with the passable neighbours' at the same y AND under 3 of air: roof eaves, attic edges, overhangs, the doorway itself) |
| `scale_headroom2`, `scale_headroom3` | share of those cells with at least 2 / 3 passable cells straight up |
| `scale_doorways` | wall openings at least 2 tall, one per opening: a 1-wide floor-level gap under a lintel (walls on both sides along one axis, open on both sides along the other, a non-furniture floor on at least one open side) through a wall at most 2 thick; longer runs are corridors and do not count, except a door block, which always does |
| `scale_low_openings` | the same openings only 1 tall (a player cannot pass) |
| `scale_door_blocks` | door columns (a door block on a door block) |
| `scale_door_ok` | doorways / (doorways + low openings) |

Passable means air, doors, carpets, torches, plants, signs, ladders and similar;
slabs and fences count as solid, and furniture tops count as floor, so a furnished
room reads a little lower than an empty one. Furniture (beds, chests, tables,
fences and the like) is never a doorway's floor, so a bed under an open window is
not a low opening. Roofs: a gable or hip attic's low eaves drop out as sloped
cells, so a pitched roof does not lower a correctly scaled house; the attic's
middle, where it has 3 of air, counts as good headroom, and a 2-high level ceiling
under any roof still reads low. Every suite build reports these in
`metrics.json` (per build and averaged per arm); `summary.md` shows
`scale_headroom3` and `scale_door_ok`. The scorer `scale(ceiling=3)` is the same
numbers as one value (mean of headroom and `scale_door_ok`, 0 for an empty or
roofless build), for `rescore` or an ablation. A covered build with no opening gets
a door term of 0 there (the metric `scale_door_ok` stays None), so a sealed box
cannot outscore a house with a door.

#### Battery `scale-plan-v1`

Dream family: `dream` vs `dream-noscale` vs `dream_scale_textonly` (rules without
the dream cue; same images as `dream-noscale`) vs `dream-plan` vs `classic-matched`.
Blueprint family: `blueprint` vs `blueprint-noscale` vs `blueprint-text-plan` vs
`blueprint-image-plan`. 8 detailed prompts, PILOT. `estimate --battery scale-plan-v1`
(2026-10-03): 72 builds, 888 LLM calls, 200 images, $11.34 upper bound without cache
sharing, inside the $25 cap. Only the dream-family pairs share images in a real run
(`dream-plan` with `dream`, `dream_scale_textonly` with `dream-noscale`), since their
dreams come before the plan. `blueprint-text-plan` pays for its own dreams: its
exterior dream follows its own blueprint's massing and its room dreams its own rooms,
which match `blueprint`'s only in the mock.

```bash
python -m blockgen.agentic estimate --battery scale-plan-v1
python -m blockgen.agentic battery scale-plan-v1 --mock --prompts short:2
```

Known gap: the planner role gets the builder system prompt, which ends with the
action space's output rule ("Output only command lines"), while plan and brief turns
ask for plain text. The user turn wins in practice; a planner system prompt of its
own would change every plan request and is left for a separate change.

### 5.6 Describing the building in words (`describe`)

A dream is a picture, and a builder reads it imperfectly: it miscounts storeys,
guesses a material from a colour, and cannot see the back. The describe stage adds
words on either side of the image. It is a parameter of `dream`, `turnaround`,
`blueprint` and `floors` (code in `describe.py`), off by default: with
`describe: none` no call is made and every request is byte-identical to before.

| `describe` | What runs | Who sees the text |
|---|---|---|
| `before` (expand) | one `planner` text turn before any dream: the request written out as a design spec (summary, footprint, storeys and clear heights, roof shape and material, materials per slot as palette blocks, openings, distinctive features, what to leave out); it respects the canvas and, with the `scale` context, player scale | the image prompt (`expand_into_image`: the spec's description is appended to `{desc}` of every recipe dream) and the build turn (`show_spec`) |
| `after` (caption) | one `judge` vision turn per scored dream (`caption_keys: primary` or `all`): what the image actually shows, in a schema of its own (the spec's visible fields, without the floor material and the clear heights an exterior view cannot show, plus block dimensions, per-face notes and an `uncertain` list; a field the image does not show is `unknown`). The captioner is not told the request, so it describes the image alone | the build turn and, with `caption_in_match`, every match turn, as `TARGET DESCRIPTION` |
| `both` | expand, dream, caption | both texts; where they disagree (visible material slots, storey count, roof shape) the builder is told to follow the image and its caption, since the image is what is scored, and the disagreement is logged. Disagreements are computed only with `caption_format: json`; under prose a `describe_disagreement_skipped` event marks them as not computed, so a prose `both` arm has no disagreement count |

Other knobs: `caption_format` (`json` or `prose`, the reply format of both turns;
JSON is shown to models as short labelled lines) and `caption_control`. Captions
are cached on the image digest: a refine, re-dream or turnaround reassignment that
changes an image gets a new caption at the next build or match turn, and an
unchanged image never costs a second call. Files per build: `dreams/spec.json`,
`dreams/<key>_v<n>.caption.json` next to the dream, and `describe.md` (spec,
captions, disagreements, and what the shuffled control showed); trace events
`describe_spec`, `describe_caption`, `describe_disagreement`, `describe_shuffled`
(plus `describe_disagreement_skipped` and, in the staged methods,
`describe_spec_partial`). Without `expand_into_image` the build turn says the spec
was written separately from the image, never that the image was drawn from it.
The suite's dream prefetch writes the spec first when it expands into the image,
so the prefetched images are the ones the build asks for.

In the staged methods the spec reaches only the exterior dream's description and
the caption only the final exterior pass (their unit turns carry their own targets).
The blueprint is planned without the spec and already fixes storeys, roof and
materials, so only the spec's openings, distinctive features and surroundings to
leave out are appended to the exterior description (a prose spec cannot be split
and is left out); `show_spec` has no effect there. Their `describe_disagreement`
events compare the caption with the spec, not with the blueprint the build follows,
so they do not measure how well the image follows the plan.

**The control.** `caption_control: shuffled` makes the same calls but shows the
builder and critic the caption of a DIFFERENT prompt's dream from the same arm: the
previous prompt the arm captioned, or, for the arm's first prompt, one of five
fixed library captions (a sunken crypt, a desert temple, a factory, an office
block, a netherrack gatehouse; palette block names only), the one sharing the
fewest words with the prompt, ties broken by a hash. Since the captioner never
sees the request, a shown caption carries no trace of the prompt it came from. The pairing follows run order, so a
re-run of one prompt alone pairs it with the library. How to read battery
`describe-v1`:

- `dream-caption` vs `dream`: the caption's total effect (same images).
- `dream-caption` vs `dream-caption-shuffled`: what a correct description adds
  beyond more text. If they tie, the gain is tokens, not description.
- `dream-caption-shuffled` vs `dream`: the cost of a wrong description. A large
  drop means the builder follows the text over the image.
- `dream-describe-before` vs `dream`: an image drawn from a detailed spec (new
  images, so this also moves the dream lottery).
- `dream-describe-both` vs `dream-describe-before`: the caption on top of the spec
  (same images). Count `describe_disagreement` events: they show how often the image
  model ignored the spec.

```bash
python -m blockgen.agentic estimate --battery describe-v1
python -m blockgen.agentic run --recipe dream-caption,dream-caption-shuffled --prompts short:2 --mock
```

`estimate --recipe dream,dream-describe-before,dream-caption,dream-describe-both,dream-caption-shuffled`
on the 4 diverse prompts (2026-10-05, no network): $0.058 per build for `dream`,
$0.069 with a caption or a spec, $0.079 with both (one extra LLM call each, two for
`both`); the caption arms reuse `dream`'s cached images.

---

### 5.7 Inspect-and-repair and the lessons knowledge base

`turns.repair` tells the builder what the *executor* saw: lines that failed and
commands that placed nothing. It says nothing about what was *built*. A sealed box
with no door, a roof floating above its walls and a shell open to the sky all execute
cleanly, and the traversal probe (`eval/traversal.py`) measures the result: real houses
are 95% enterable, LLM builds 2%, agentic arms 50 to 68%. The render critique
(`critique_rounds`) is the suite's only answer so far, and it costs images.

ldraw-nova, an LLM agent that writes LDraw LEGO models (2026), runs the cycle
*brief, retrieve, build, inspect, repair, deliver*, where inspect is a set of geometry
checks (collision, gaps, contacts, connectivity) whose report the agent reads before it
repairs. `blockgen/agentic/inspection.py` is the Minecraft twin, built from probes the
benchmark already trusts:

| check | severity when failed | what it measures | fix hint the builder gets |
|---|---|---|---|
| `built` | error | zero blocks | start from the ground |
| `grounded` | warning | the lowest block is above y=0 | move the build down |
| `connected` | error > 5% stray, warning > 2%, else note | blocks in pieces not attached to the main one, and where the largest stray piece is | attach or remove |
| `supported` | warning | blocks with nothing below them in their column and no supported neighbour beside them (> 25%) | posts, lower the roof, extend the walls |
| `interior` | error | fewer than 8 enclosed air cells; names a solid mass separately | hollow the footprint; close walls and roof |
| `roofed` | warning | floor area inside the walls with nothing above it (> 25%) | roof the whole footprint |
| `enterable` | error / warning | the player walk cannot get in; storeys it cannot reach | a 1 x 2 door at floor level; stairs |
| `headroom` | warning | under half the interior has 3 clear blocks | raise the ceiling |
| `openings` | note | no glazed block and no window aperture over a sill | windows at floor+1 |
| `size` | note | footprint under 6% of the canvas footprint | build for the player |

Checks that do not apply (an empty build has no interior to walk) are marked `[skip]`
and the turn tells the builder a finding that does not fit the request (a bridge has no
room) may be skipped. Calibrated on 400 real `houses_32` builds: mean score 0.93, 55%
clean at warning level; the per-check failure rates on real houses are supported 14%,
enterable 17%, roofed 10%, headroom 9%, connected 6%, the rest 0%. The raw column
rule for support flagged 44% of real houses (corpus crops often lack their floor), which
is why the rule forgives a block with a supported horizontal neighbour.

```
INSPECTION of the build (automatic geometry checks: 7 of 10 passed)
[FAIL] enterable: a player cannot get inside: no doorway reachable from the ground (0 openings are only 1 block tall)
       fix: cut a door 1 wide and 2 tall at floor level through an outer wall (air at y=floor and y=floor+1), facing open ground
[note] openings: no window openings in the walls
       fix: cut 1 x 1 or 2 x 1 openings at y=floor+1..+2 and glaze them
[ok]   built, grounded, connected, supported, interior, roofed, headroom
Numbers: blocks 344, bbox 10x6x10, pieces 1, unsupported 0, enclosed_air 256, rooms 1, open_sky 0%, doorways 0, window_cells 0
```

**The method.** `inspect` is `classic` with inspect-and-repair rounds between the
execution repair and the render critiques: `plan? -> generate -> repair -> inspect x N
-> critique x M`. A round inspects the current build, stops when it is clean at
`clean_at` severity (`error | warning | info`, default `warning`), otherwise sends
`turns.inspect` (execution summary, inspection block, mode instruction) and tries the
reply; an empty rejected reply ends the stage. Every round's inspection is traced
(`inspection` events) and the last one sits in `Attempt.extra['inspection']`. The
`inspection` scorer (share of applicable checks passed) lets `accept: improve_valid`
and `select: best` keep the build that passes most. `inspect_mode: patch` works on
patchable action spaces.

**Reported for every arm.** `MethodResult.metrics()` carries `inspect_pass`,
`inspect_enterable` and `inspect_failed` for the final build of any method, so a
`classic` arm and an `inspect` arm compare on the same checks (summary.md has the two
numeric columns). Like `scale`, this is a diagnostic: the benchmark pillars stay
untouched, and a build that passes every check is not therefore a good house.

**Lessons.** `configs/agentic/lessons.yaml` is a small curated knowledge base of
building rules (door, room, roof-sits, windows, storeys, chimney, tower, supports,
materials, size, ...), each with tags. The `lessons` context puts the `always` rules
plus the `k` best tag matches for the request in the system prompt, lexically (the
`examples.select_examples` device), and records the chosen ids in `ep.notes['lessons']`
and the trace. `shuffled: true` is the token-matched control: the `always` rules stay
and each retrieved rule's text is swapped for a rule that does not match the request.
ldraw-nova's authors keep research and admitted lessons apart; so does this file,
which is edited by hand and never written by a run.

**Battery.** `inspect-v1` (`configs/agentic/batteries/inspect-v1.yaml`) runs
`classic-matched`, `classic-lessons`, `inspect`, `inspect-lessons`,
`inspect-lessons-shuffled` and `inspect-critique` on 12 detailed prompts; `estimate`
prices it at $3.70 as an upper bound (every round firing). Read `inspect_pass` and
`inspect_enterable` first, since every arm is measured on the same checks, then the
bench. n=12 is a pilot.

```bash
.venv/bin/python -m blockgen.agentic battery inspect-v1 --mock --renderer flat   # offline, seconds
.venv/bin/python -m blockgen.agentic estimate --battery inspect-v1               # no network
.venv/bin/python -m blockgen.agentic battery inspect-v1 --allow-spend --budget-usd 6
```

---

## 6. How to add a component

Each extension is one file (or a few lines in an existing one) plus one decorator.
The registry validates parameters, so a recipe typo fails at load time. Test
templates: `tests/agentic/fakes.py` provides `make_fake_kit`, `ScriptedLLM`,
`FakeRenderer`, `FakeImageGen` and `RecordingTrace`; the `registry_snapshot`
fixture (`tests/agentic/conftest.py`) lets a test register throwaway components.

### 6.1 A method

```python
# blockgen/agentic/recipes/variants.py, or any file passed with --plugin (section 5.4)
from blockgen.agentic.core import register, EpisodeLike
from blockgen.agentic import dreams
from blockgen.agentic.recipes import steps

@register("method", "plan_then_dream")
def plan_then_dream(ep: EpisodeLike, *, max_iters: int = 2) -> None:
    """Write a plan, dream from the plan, then the normal match loop."""
    steps.plan(ep)
    # prompt= is the raw final prompt: fill only {desc} so style, camera and W/H/D stay.
    desc = f"{ep.prompt}. {ep.plan[:400]}"
    ep.dream("iso", prompt=dreams.render_prompt(ep.dream_spec("iso"), ep, desc=desc))
    att = steps.generate(ep)
    for i in range(max_iters):
        reply = steps.match_step(ep, i, ep.current() or att, feedback="raw",
                                 mode="rewrite", allow_redream=False)
        ep.try_(f"match{i}", reply)
```
```yaml
# configs/agentic/recipes/plan-then-dream.yaml
# Extends _base, not dream: `params` mappings MERGE down an extends chain, and
# dream's params (refine, stop_at, ...) are not this method's.
extends: _base
method: plan_then_dream
llm: {builder: openai:gpt-5-mini, critic: openai:gpt-5-mini}
image: openai:gpt-image-1-mini?quality=low&size=1024x1024
dreams: [iso]
score: {embed: 1.0, silhouette: 0.5, validity: 0.3}
accept: improve_valid
select: best
params: {max_iters: 2}
```
`ep.dream(key, prompt=...)` sends `prompt` to the image generator verbatim and
skips the dream template, so a bare description loses the style, the camera wording
and the canvas size while the dream is still scored against its view. Build the text
with `dreams.render_prompt(ep.dream_spec(key), ep, desc=...)`, as the shipped steps
and the REDREAM path do. `dream_spec` lives on the concrete `Episode`; it is not yet
part of the `EpisodeLike` protocol.

The shipped `dream_vote` (`recipes/variants.py`) is a complete worked example: it
samples dreams, asks the critic to pick one, then calls `dream(ep, skip_dreams=...)`.

### 6.2 A dream variant

YAML only, inline:
```yaml
dreams:
  - iso
  - {key: courtyard, template: "{style} The SAME building as the reference image, seen from inside its courtyard. {desc}", view: iso_back, weight: 0.3, condition_on: [iso]}
```
A reusable preset: add a `DreamSpec` to `DREAMS` in `blockgen/agentic/dreams.py`. A
new camera or cutaway: add one line to `VIEWS` in `core.py`, for example
`"attic": View("attic", 45, 50, keep_below=0.8)`; every dream, scorer and contact
sheet can then use it.

### 6.3 An image backend

```python
# blockgen/agentic/imagegen.py
@register("image", "gemini", env="GEMINI_API_KEY")
class GeminiImageGen:
    """Gemini image generation (paid API)."""
    name, spends, billing = "gemini", True, "api"

    def __init__(self, model: str = "gemini-2.5-flash-image", **params): self.model = model
    def generate(self, req: ImageRequest) -> ImageResult:
        from google import genai            # lazy: importing imagegen must stay light
        ...
        return ImageResult(images=[png], model=self.model, cost_usd=0.039)
    def estimate_usd(self, req): return 0.039     # None = unpriced
    def supports_refs(self) -> bool: return True
    def describe(self) -> str: return f"gemini:{self.model}"
```
```yaml
image: gemini:gemini-2.5-flash-image
```
`get_image_gen` wraps it in the content-addressed cache and the spend gate, and the
Episode meters it; nothing else changes.

### 6.4 An LLM vendor

LLMs are not a registry kind: they keep the single legacy source of truth,
`providers._PROVIDERS` and `get_provider("vendor:model")`. Add a subclass of
`LLMProvider` with `complete(messages, **overrides) -> LLMResponse`, set `spends` and
`billing`, add one line to `_PROVIDERS` (and `DEFAULT_MODELS`), and a price row to
`PRICES` if you want dollars instead of `cost_known=False`. Use it as
`llm: myvendor:model` or `--llm-role judge=myvendor:model`.

### 6.5 An action space

```python
# blockgen/agentic/actions.py
@register("actions", "layers")
class LayerActions:
    """One ASCII grid per y layer; '#' = the block named in the layer header."""
    name, patchable = "layers", False

    def instructions(self, ctx: PromptCtx) -> str: return "Write layers as ... " + self.output_rule()
    def output_rule(self) -> str: return "Output only layer grids."      # must be unique
    def examples(self, query: str, k: int): return []
    def lower(self, text: str) -> Lowered:
        lines, issues = [], []
        ...                                   # emit 'set <block> x y z' lines, never raise
        return Lowered(source=text, program_text="\n".join(lines), issues=issues)
    def mock_reply(self) -> str: return "..."  # a valid small house, for mock:auto
```
```yaml
actions: layers
```
Every method then runs with it: lowering to the DSL keeps execution, replay and
reports shared. The output rule must be distinct from every other space's, because
`mock:auto` recognizes a build turn by it.

### 6.6 A scorer

```python
# blockgen/agentic/scoring.py
@register("scorer", "symmetry", needs_dream=False, needs_renders=False)
def symmetry(ep, att, *, axis: str = "x") -> ScoreTerm:
    """Mirror symmetry of the build along one axis."""
    s = att.world.to_structure()
    occ = s.occupied_mask
    flip = occ[::-1] if axis == "x" else occ[:, :, ::-1]
    return ScoreTerm(float((occ & flip).sum() / max(1, occ.sum())))
```
```yaml
score: {embed: 1.0, symmetry: {weight: 0.2, axis: z}}
```
The two meta flags are read by the Episode, so declare them truthfully:

- `needs_renders` (default True when undeclared): the Episode renders the score views
  of each non-empty attempt before scoring only if some `score` or `final_score` term
  sets it. A recipe whose terms are all `needs_renders=False` (`validity`, `material`,
  `symmetry` above) skips those renders entirely, so such a scorer must not read
  `att.renders`; the suite still renders the trace views after the episode.
- `needs_dream` (default False): if any `score` term sets it, each Score is stamped
  with the current `dream_id`, and after a re-dream `rescore_all` re-scores the
  attempts measured against older dreams. A `final_score` term with
  `needs_dream=True` also clears its cached values on every re-dream. A term that
  compares against dream images but leaves the flag False keeps stale scores.

An attribute on the function (`symmetry.needs_renders = False`) overrides the
registry meta, which is convenient for an unregistered scorer in a test.

### 6.7 A context module

```python
# blockgen/agentic/context.py
@register("context", "style_guide")
def style_guide(ep, *, text: str = "Prefer gabled roofs and timber framing.") -> ContextParts:
    """A fixed style note in the system prompt."""
    return ContextParts(system_text=f"STYLE GUIDE: {text}")
```
```yaml
contexts: [palette, {style_guide: {text: "Use dark oak and cobblestone."}}]
```
Images go in `user_parts` (they ride in the first build turn, because Anthropic and
Claude Code system prompts cannot carry images).

### 6.8 An accept policy

```python
# blockgen/agentic/scoring.py
@register("accept", "monotone_blocks")
def monotone_blocks(prev, cand, *, min_growth: float = 1.0):
    """Accept only builds at least as large as the current one."""
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or cand.blocks >= prev.blocks * min_growth:
        return True, "ok"
    return False, f"shrank {prev.blocks} -> {cand.blocks}"
```
```yaml
accept: {monotone_blocks: {min_growth: 0.9}}
```

---

## 7. Artifacts

```
outputs/run_<stamp>_<name>[_mock]/
  config.json        battery, prompts, every resolved recipe + recipe_hash, git commit /
                     dirty / branch, CLI versions (e.g. claude), argv
  metrics.json       per-build metrics, per-arm aggregates (+ provenance, + ledger
                     summary), run-level ledger
  summary.md         the legacy table, a method-metrics and cost table, the ledger
  samples.png        textured render of every build, titled by prompt (repo rule)
  samples_<arm>.png
  <arm>_32.npz + <arm>_32_manifest.json   structure cache; manifest 'report' = provenance
  contact_<arm>.png  the first 8 builds' contact sheets, stacked
  variants.png       rows = prompts, columns = arms, each cell 'dream | final'
  rescore_<stamp>.json   written by `rescore`
  builds/<arm>/
    NNN_program.txt NNN_build.json NNN_transcript.json   (legacy save_build, unchanged)
    NNN/                                                (the Trace)
      dreams/<key>_v<k>.png + .json     prompt, generator, seed, critique, cost, cached
      iters/<idx>_<stage>/source.txt program.dsl delta.dsl report.json score.json
                          structure.npz render_<view>.png
      messages.jsonl    every request: role, stage, messages (images as assets/<sha>.png),
                        response, usage
      events.jsonl      redream, budget_stop, json_parse_fail, refs_dropped, ...
      charges.jsonl     one Charge per backend call
      assets/<sha16>.png
      contact.png       rows = scored views (+ dream versions strip),
                        columns = dream | generate | repair | match0 ...;
                        header 'score acc|rej' and a third line: 'sel x.xx' (the
                        selection value), 'sel gated' or 'unit x.xx' (a room or
                        storey attempt); best column boxed in red; staged methods
                        add one row per unit (u_<room>, u_storey<k>) with its dream.
                        On a unit column, line 2's number is the global score but
                        its acc|rej is the unit rule's verdict on line 3's unit
                        score; a column before select_from that is not a unit (the
                        shell) has no line 3, since selection never ranks it
      result.json       MethodResult.to_dict()
      (score.json per attempt carries 'selection' and 'extra', e.g. unit_score)
```

`program.dsl` is the cumulative canonical DSL that rebuilds the build from empty, for
every action space, so deploy replay works unchanged. Arms without scorers (classic,
legacy) are rendered after the build, for the contact sheets only, so the sheets are
never blank and nothing the model saw changes.

`rescore` re-runs scorers offline over a run's traces (each attempt's world from
`structure.npz`, scored against the build's final dreams, saved renders reused):

```bash
python -m blockgen.agentic rescore outputs/run_..._dream_v1 --scorers clip_text,material
```

---

## 8. Cost and safety

**Spend gate.** Any cache miss on a backend with `spends=True` (OpenAI, Gemini,
Anthropic, Claude Code) requires `BLOCKGEN_ALLOW_SPEND=1` (`--allow-spend`), checked
before any client is built. Tests remove the variable, so a test can never spend.
Mocks, local servers and diffusers are never gated.

**Budgets.** `budget.usd_per_build`, `max_llm_calls`, `max_images` (per build),
`usd_per_arm`, and `--budget-usd` / a battery's `budget.usd` (whole run). The ledger
checks the next call's estimated price at every level before the call and raises
`BudgetExceeded`; the episode then stops with `truncated='budget'` and keeps its best
build. Calls and images count cache misses only. An LLM call is checked against its
estimated price (completion tokens are unknown until it returns), so a `usd` cap can
be overshot by at most one call.

**Paid vs would-cost.** `usd` is what was paid (0 on a cache hit); `would_cost_usd` is
the full price of every call, the honest figure for compute comparisons between arms.
`cost_known=False` means some call was unpriced (Claude Code, local, mock), so dollar
totals are a lower bound, never "free".

**Cache modes.** `read_write` (default), `read_only` (a miss raises `CacheMiss`: a
rerun to add a metric provably costs $0), `off`. LLM responses use the legacy cache
(`outputs/agentic_cache`), so `dsl-classic` and `legacy-*` hit entries written by
`experiments_agentic`. Dreams use one content-addressed cache
(`outputs/cache/imagegen`) shared across arms: hold the dream spec and seed fixed when
ablating the loop and every arm gets the same images, which isolates loop variance
from dream variance.

**Estimate.** `estimate` runs the recipes on free mocks (`mock:auto`, `mock:solid`,
`null` renderer) and prices every call as the real model would: prompt tokens
chars/4 + 800 per image, completion tokens `gen.max_tokens x --completion-frac`
(default 0.3), images from `IMAGE_PRICES`, requests already in the real cache at $0.
It is an upper bound for the native methods: `stop_at` is infinite, patience is
disabled, every repair round runs, and the mock dream critic rejects every dream and
calls every stall unreachable, so refine and re-dream run to their caps. The
`legacy-*` arms (and `dsl-classic`) are the exception: they run BuildAgent, whose
repair rounds are not forced and follow the always clean mock replies, so their figure
undercounts. It makes zero network calls.

**Approximate cost per build** (estimate, gpt-5-mini + gpt-image-1-mini low,
`--completion-frac 0.3`, short prompts; upper bounds, measured 2026-10-03):

| Recipe | LLM calls / build | images / build | est. $ / build |
|---|---|---|---|
| `classic-matched` | 4 | 0 | 0.041 |
| `dream` | 4 | 1 | 0.048 |
| `dream-refine` | 7 | 4 | 0.102 |
| `dream-multiview` | 14 | 10 | 0.243 |

At these rates the 12-prompt `dream-v1` pilot (9 arms: 5 named plus a 4-arm
`max_iters` x `dreams` sweep, 108 builds) estimates at $5.97 upper bound against its
$8 cap; actual spend is lower because the arms share the dream cache; n >= 256
runs should use Claude Code and diffusers (`dream-local`), whose marginal cost is
quota and GPU time. The ~$30 credit plan: calibrate first (free), then the n=12
pilot, then scale on free backends. `IMAGE_PRICES` and the per-reference surcharge in
`imagegen.py` are list prices that were not re-verified online; check them against
the OpenAI pricing page before a paid battery.

**Python sandbox.** The `python` action space runs model code in a child process
with an AST allow-list (no imports, no dunder or private attributes, no
`open`/`eval`/`exec`), restricted builtins, rlimits (CPU, memory, no file writes, no
forks), an empty environment, a temporary cwd and a wall-clock timeout. It is a guard
against what an LLM plausibly writes, **not a security boundary**: there is no
network or filesystem namespace. Never serve it to untrusted users without a
container or `bubblewrap`.

---

## 9. Benchmark integration

```bash
python -m blockgen.eval.bench --tier fast \
    --arms agentic/dream:outputs/run_..._dream_v1/dream_32.npz \
           agentic/classic-matched:outputs/run_..._dream_v1/classic-matched_32.npz
```

- The npz is the standard structure cache at `max_dim=32` (the houses32-v1 protocol);
  the manifest titles are the prompts. Cost is attached from the sibling
  `metrics.json` by the bench as for legacy agentic runs.
- Manifest `report` provenance: `method`, `recipe`, `recipe_hash`, `action_repr`,
  `backend` (per role), `image_generator`, `renderer`, `contexts`, `dreams`, `loop`,
  `train_split`, `n_requested`, `git_commit`, `git_dirty`, `mock`.
- `train_split`: `none`, `train_only` (an examples context drew from the corpus), or
  `oracle_train`. Any arm whose dreams are `mock:render` (the `dream-oracle` recipe,
  and every dream arm of a `--mock` run) is tagged `oracle_train` and must never be
  ranked: its dreams are renders of real training houses.
- The in-loop dream score is a method-internal metric (`dream_score_first`,
  `dream_score_best`, `dream_gain` in metrics.json). It is never a BlockScore pillar,
  because the loop optimizes it. Methods are ranked by BlockScore, `geom_kid` and
  faithfulness, with n >= 256 for leaderboard claims; smaller runs are pilots.
- Goodhart: the in-loop backbones (`siglipB` for `embed`, `clipB16` for `clip_text`)
  are disjoint from the bench's (`dinov2b`, `clipL`). `clip_text` warns if `clipL` is
  chosen.
- Compute-matched controls: compare a dream arm with `classic-matched` and
  `best-of-4` at similar `would_cost_usd`; `dream-oracle` is the ceiling.
- Calibrate a scorer before trusting it in a loop:
  `python -m blockgen.agentic calibrate --scorer embed?backbone=siglipB --n 32`
  (does it rank a build's own render, from another view, above 31 distractors?). On
  oracle dreams (n=32, 31 distractors, seed 0, textured renderer, chance 0.03;
  `outputs/run_20261003_130834_calibrate/`): `embed` siglipB top-1 1.00, clipB16
  0.97, `color_lab` 1.00, `pixel` 0.88, `silhouette` 0.25. Silhouette is a weak
  identity signal; keep its weight small. These are ceilings (no domain gap); re-measure with `--dreams-dir` on real
  generated dreams.

A mock run is benchmarkable but meaningless: the mock builder returns one house for
every prompt, so the bench disqualifies it for mode collapse, which is the expected
verdict and a useful end-to-end check of the plumbing.

---

## 10. Legacy compatibility

- `agent.py`, `canvas.py`, `dsl.py`, `prompts.py`, `examples.py`, `tasks.py`,
  `blockstate.py` and `deploy/**` are byte-untouched. BuildAgent stays the live `/gen`
  path and the published scaffolding and ontology numbers stay valid.
- Method `legacy` wraps BuildAgent itself: byte-identical requests, so existing cache
  entries hit, with traces, ledger, contact sheets and batteries on top.
- Method `classic` is the native re-implementation. `tests/agentic/test_legacy_parity.py`
  checks, for every legacy arm x critique mode x keep_history x 3 prompts x 4 reply
  scripts, through the real `build_kit` path, that provider requests (cache keys),
  final program, blocks and stage names are identical to BuildAgent.
- Known quirks kept in legacy paths only: the `nonempty_legacy` rule (reject only an
  empty build after a non-empty one, i.e. `== 0`, so a shrinking build is accepted),
  and `Canvas.clone` dropping the clipboard (a patch-mode `paste` of an earlier
  `copy` fails in BuildAgent; `MinecraftWorld.clone` keeps it).
- `report.py` changes are additive: legacy callers get three new aggregate keys
  (`cost_known`, `total_would_cost_usd`, `n_truncated`) and nothing else.

---

## 11. Known limits

- **Domain gap.** Image models draw Minecraft-like pictures, not renders: lighting,
  perspective and texture scale differ from the textured renderer. Mitigations: a
  pinned style and camera, foreground crop, a calibrated embed baseline, the
  texture-blind silhouette, the VLM judge. Not yet measured on real dreams.
- **Viewpoint.** gpt-image cameras are approximate; `view_search` takes the max over
  nearby views, and the dream critic checks `view_ok`.
- **Mock realism.** `mock:auto` returns one fixed house, so mock runs test plumbing,
  never quality. Loops with identical replies stop on patience.
- **Contexts are evaluated once per build**; a mid-loop re-dream does not refresh
  `dream_palette` or a `texture_sheet{blocks: dream_palette}`.
- **Example turns.** The `examples` context reaches the model as first-turn text, not
  as prior user/assistant turns (`Episode.inserts_example_turns` is off), so its
  message shape differs from legacy's one-shot pairs (classic's legacy mode builds
  legacy pairs itself, so parity is unaffected).
- **`dream_score_*` for arms without scorers** reads 0.0, not "none"; ignore it for
  classic and legacy arms.
- **The VLM judge** sees the dream's own view, not the best `view_search` match, and
  is re-asked after every re-dream (paid misses).
- **Empty builds** are dropped from the npz (`n_requested` records how many were
  asked) until the bench reads `n_requested`.
- **Estimate** can only find a run's first turns in the real cache; later turns depend
  on model replies.
- **Camera inheritance (pilot P2).** A dream conditioned on an exterior reference
  inherits its camera and composition, so the `plan` and `interior` presets of
  `dream-multiview` mostly redraw the exterior. Not changed (provenance); the staged
  methods anchor plans on a grid and cutaways on an empty-room render instead.
- **Staged methods are mock-verified only.** Under `--mock` the room dreams are
  unrelated pictures, so most room attempts are rejected; that exercises the reject
  path, not quality.
- **`blueprint-image-plan` reads only the ground floor** from the drawn plan; upper
  storeys are the reader's proposal.
- **Region replay is structure-exact, not program-exact.** Reverted writes are
  dropped from the world, so replaying the cumulative program can differ
  (`extra.replay_exact = false` on such attempts); `structure.npz` is the truth.
- **Rooms at 32^3 are small** (interiors of about 4 x 3 x 4), which limits what
  furnishing and the `interior` scorer can show; repair may widen a room by one block
  to merge a double wall.
- **About 6% of random messy plans** keep an unreachable room after repair (an upper
  room with no stair landing, an inner room with no shared wall); repair notes it and
  does not invent geometry.
- **`floors` with `shell: det`**: storey k's slab fills the stairwell of the storey
  below; the storey-k builder must open it.
- **Turnaround assignment** scores the tiles against a build made from the given
  order, which biases it toward that order; check how often `turnaround_assign`
  reports `changed: true`.

## 12. Ideas to try next

1. **Dream to voxels via depth.** Estimate depth (and normals) from the iso dream,
   back-project into a voxel occupancy prior, and give the builder a coarse massing
   (or seed the world with it and work in patch mode).
2. **Multi-view consistency.** The one-image turnaround is now `method: turnaround`
   (section 5.4); still open: check pairwise consistency with the critic before
   building and reject dream sets that disagree.
3. **Image-edit refinement.** Instead of regenerating, edit the dream with the render
   as a reference ("make this match the render, then fix X") so the target moves
   towards what is buildable.
4. **VLM-judge feedback only.** `feedback: vdm` with a strong judge (Claude Code) and a
   cheap builder; compare against raw image feedback (the CaP-X question).
5. **Render-in-the-loop for the dream critic.** Show the critic the build render next
   to the dream when deciding REDREAM, with an explicit "reachable at this canvas"
   rubric.
6. **Per-part dreams.** Rooms and storeys are now units (`blueprint`, `floors`,
   section 5.4); facades and the roof as their own units are still open (see 5.4,
   "Ideas to try next").
7. **Retrieval dreams.** Use the nearest train-split house render as a second
   reference (`dream-oracle` style, but as conditioning rather than the target).
8. **Learned scorer.** Fit a small head on (dream, render, human 2AFC) pairs from
   BlockLab to replace the hand-weighted composite.
9. **Program-level search.** `best_of_n` over the match loop with a diversity bonus,
   or annealing (`accept: anneal`) for longer loops.
10. **LEGO world.** A second `World` behind the same protocol (typed connections), so
    the same methods run on bricks.

---

## 13. Review checklist

- [ ] `python -m blockgen.agentic list` shows the components you expect.
- [ ] `python -m blockgen.agentic battery smoke-mock --mock` completes; open one
      `builds/<arm>/000/contact.png` and `samples.png`.
- [ ] `python -m blockgen.agentic battery staged-v1 --mock --prompts short:2` completes;
      open `builds/blueprint/000/contact.png` (u_<room> rows, header line 3) and the
      `dreams/blueprint_v0.png` schematic.
- [ ] `estimate` for the planned battery is under budget.
- [ ] `calibrate` the in-loop scorers (oracle first, then on a few real dreams with
      `--dreams-dir`).
- [ ] Run the pilot with `--allow-spend --budget-usd`; rerun with `--cache read_only`
      to confirm it costs $0.
- [ ] Bench the pilot npz files; label n < 256 as PILOT; never rank `oracle_train`.
- [ ] Record the run in `results.md` with `recipe_hash`, n, BlockScore, `geom_kid`,
      faithfulness, `dream_gain`, `would_cost_usd` and `cost_known`.
