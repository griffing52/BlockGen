# Block ontology — what each material *is*, measured rather than asserted

The agentic track hands a frontier model a list of ~70 block names and asks for a
build program. That list is the model's entire view of the material world:
`oak_planks`, `stone_bricks`, `glass_pane`, in alphabetical order, with nothing
attached. Everything else it "knows" about those blocks comes from its own
priors.

The ontology attaches knowledge to those names — colour and surface measured from
the shipped textures, placement behaviour and material affinity mined from the
corpus, game rules written down once — and renders it as a table the model reads
in its system prompt, in place of the bare list.

```
                    ┌── textures (data/textures/vanilla) ──► colour, surface
palette (70 names) ─┼── corpus  (houses_32, 2661 builds) ──► layer, form, affinity, style
                    └── rules   (authored + derived)     ──► flammable, not-full, states
                                          │
                                          ▼
                            one table  ──►  system prompt  ──►  build program
```

The question it exists to answer: **does a grounded ontology beat the model's own
priors?** That question has a specific shape, which is why this is not just a
JSON file of adjectives.

---

## The experiment, and why the control is the point

An ontology helps for one of two reasons. Either it carries information the model
lacks, or it is 3,000 extra tokens of plausible-looking context and the model
does better because prompts with more structure produce more structured output.
Those are very different findings and they look identical in a results table.

So every catalog ships with a **shuffled** variant: the same table, with every
attribute permuted onto a different block. Oak planks get cobblestone's colour,
glass's neighbours, hay's layer. It is *token-matched to the character* with the
real table.

| arm | what the model sees | system prompt |
|---|---|---|
| `ont_none` | today's bare palette list | ~1.3k tokens |
| `ont_mined` | the full catalog | ~4.3k tokens |
| `ont_shuffled` | **the control** — same table, wrong blocks | ~4.3k tokens |
| `ont_stats` | mined fields only, no colours or rules | ~3.2k tokens |

`ont_mined` beating `ont_none` is not a result. `ont_mined` beating
`ont_shuffled` is.

```bash
.venv/bin/python -m blockgen.ontology                                  # build it
.venv/bin/python -m blockgen.experiments_agentic --config agentic-ontology
```

---

## What is measured, and how

Provenance is a field, not a comment. Every attribute declares where it came
from, and `Catalog.subset_by_source(...)` turns that into an ablation rather than
a claim.

### `mined` — from the corpus (`blockgen/ontology/mine.py`)

Measured over `Structure` lists, the same objects the benchmark scores, so a
claim made from the ontology and a claim made from the bench are about the same
builds. The whole corpus (2661 builds, 2.78M placements) mines in **~1 second**;
every statistic is a numpy accumulation over the volume, never a Python loop over
voxels.

| attribute | how | what it tells the model |
|---|---|---|
| `layer` | mean height, normalized inside each build's own occupied bbox | ground / low / mid / upper / roof |
| `form` | run-length anisotropy + air-facing exposure | post, plate, mass, trim |
| `pairs_with` | **normalized PMI** over 6-adjacent faces | what it is built against *more than chance* |
| `styles` | presence lift per corpus category | "medieval-houses", "quartz-houses" |
| `frequency` | share of all placements | how central it is to this corpus |

Statistics live on the **display-name symbol** (`palette.block_key(level="exact")`),
so "Oak Wood Stairs" is one entry regardless of which way it faces — that is the
granularity an ontology entry should mean. The mapping is a precomputed
4096-entry lookup table, so a voxel grid becomes a symbol grid in one fancy-index.

Two choices here are worth knowing about:

**Affinity is normalized PMI, over building materials only.** Raw co-occurrence
gives the same top-k list for every block (the most common neighbour of
everything is the most common block). Plain PMI overcorrects and fills the list
with trivia — a jukebox seen twice, both times against oak. NPMI damps the
rare-item bias, and candidates are restricted to partners with real support. What
comes back is species-consistent and obviously right: spruce planks pair with
spruce stairs, spruce slab, spruce fence.

**Sparse counts are withheld, not smoothed.** Below `min_support` (200
placements) the numbers are still reported but the *words* are not, and the entry
says so. A role inferred from a dozen placements is noise wearing a label.

### `asset` — from the textures (`blockgen/ontology/appearance.py`)

Colour is not a matter of opinion and is not worth an API call.
`data/textures/vanilla/` holds the real 16×16 vanilla PNGs and
`renderer.textures.resolve_face_names` already maps a legacy `(id, data)` pair to
them, so `primary_color` is the mean over the *side* face — the face you see in a
wall — of the exact pixels the eval renders. The renderer's foliage and water
tints are applied, so the catalog cannot disagree with the picture.

`transparency` (opaque / cutout / translucent) and `texture` (flat / subtle /
busy) come from the alpha channel and the per-channel standard deviation. The
second is the one models most often get wrong from the name alone: it is what
separates concrete from cobblestone at the same colour.

Missing textures produce `None`, never a guess.

### `authored` / `derived` — the rules (`blockgen/ontology/authored.py`)

Whether oak burns is a rule of the game, not a property of our corpus, and no
amount of looking at houses will reveal it. These are written as *rules over
names* ("everything wooden is flammable") rather than 70 hand-typed rows, so they
are auditable and they cover palette entries added later.

`not-full` earns its place: a wall built out of fences, panes or stairs has holes
in it, which is the single most common way an agentic build fails its coherence
check.

The state grammar (`oak_stairs[facing=,half=]`, `oak_log[axis=]`,
`stone_slab[type=]`) is **derived from `blockgen/agentic/blockstate.py`** rather
than authored — that module is what turns those properties into legacy metadata
bits, so asking it is a fact about what the executor will actually honour. A
property we advertised but dropped would be a silent no-op in every program.

---

## The table the model sees

```
Block reference — every column below is MEASURED from 2661 real builds
(houses_32) or from the block textures, not from taste. …

block            colour            look    layer  form   freq%  pairs with                        styles
---------------  ----------------  ------  -----  -----  -----  --------------------------------  --------------
oak_planks       #a2834f tan       busy    mid    plate  6.76   oak_stairs·oak_log·torch          medieval-houses
oak_stairs       #a2834f tan       busy    roof   trim   2.79   oak_planks·oak_slab·oak_fence     medieval-houses
spruce_log       #3b2611 dark bro…  subtle  mid    post   2.02   spruce_planks·spruce_stairs       wooden-houses
glass_pane       #b0d6db pale cyan  busy…   mid    post   1.36   glass·oak_planks·white_wool       modern-houses
```

A table, not JSON: the same content in braces and repeated key names costs ~5×
the tokens, and the key names are exactly what the legend says once. The table
**replaces** the bare palette list rather than joining it — its first column *is*
the palette — so the ontology arm pays for knowledge, not for a second copy of
the vocabulary.

**The vocabulary is closed.** Every block named anywhere in the table, including
inside a `pairs with` list, is a palette entry the DSL can resolve. A neighbour
list that mentioned "Double Oak Wood Slab" would teach the model a name that
fails to parse — knowledge converted into per-line errors.

The thresholds that turn numbers into words (`height_mean = 0.61` → `roof`) live
in one visible place, `blockgen/ontology/minecraft.py`, next to the reasoning. An
arm that wants raw numbers instead of words asks for the `stats.*` fields, which
carry them unrounded.

---

## Looking at it

The ontology is the first artifact here that is authored by a measurement and
then read by a model, so "is it any good?" is not a scalar. BlockLab has an
[**Ontology** page](lab.md#ontology) for the two questions the table cannot
answer about itself: is each word backed by a believable number (every column
header carries its provenance; the detail panel shows the raw statistic under
every word), and what is it costing (the Prompt tab renders the exact string the
agent sends, with its size).

Flipping the variant switch to `shuffled` and finding that the table still looks
completely plausible is the fastest way to understand why the control exists.

---

## Beyond Minecraft

The container is domain-agnostic on purpose. `blockgen/ontology/schema.py` knows
about `Part`s, `AttributeSpec`s and provenance; it knows nothing about blocks.
`minecraft.py` is the first backend.

That is not speculative tidiness — LEGO parts and electronic schematics are the
stated next domains ([roadmap](roadmap.md)), and in both the catalog is thousands
of entries the model cannot be assumed to know. The parts of this that transfer
are exactly the parts that are generic: mine adjacency from a corpus of real
assemblies, measure appearance from the part's own asset, write the physical
rules once, render a closed-vocabulary table, and ship a shuffled control so the
result means something.

---

## Known limits

* **The mined attributes come from the same corpus the bench scores against**, so
  "matches the corpus palette better" is partly circular. The honest headline is
  held-out geometry and human preference; palette JSD is a mechanism check.
* **Family-level materials.** `stripped_oak_log` and `oak_log` remap to the same
  legacy pair and therefore carry identical statistics. The catalog records the
  collision in `notes` rather than pretending otherwise.
* **One corpus, one style.** `houses_32` is houses. A block's "layer" is its layer
  *in a house*; mining a castle corpus would move it. Rebuild per corpus
  (`--corpus`), do not assume transfer.
* **Category lift is only as good as the labels.** Half of `houses_32` is
  labelled with the generic `house`, which is filtered out because every block is
  "over-represented" in it by arithmetic accident.
* **No inverse compiler, still.** The ontology tells the model about materials; it
  does not give the track a supervised training signal.
