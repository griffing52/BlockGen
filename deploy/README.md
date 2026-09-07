# BlockGen in Minecraft

Watch the models build. An inference server on the GPU box streams a structure block
by block as it is sampled; a Fabric mod on your Minecraft client places each block as
it arrives.

```
  Minecraft 1.21.1 (laptop)                     GPU box
  ┌──────────────────────────┐                  ┌───────────────────────────┐
  │ /gen                     │   WebSocket      │ blockgen_server           │
  │  deploy/mod (Fabric)     │ ───────────────► │  deploy/inference         │
  │  places blocks on tick   │ ◄─────────────── │  AR sampler -> blocks     │
  └──────────────────────────┘  [[x,y,z,state]] └───────────────────────────┘
```

The server does all the model-specific work — sampling, token decoding, and mapping
the models' legacy `(block_id, data)` vocabulary onto modern block states. The mod
receives ready-made strings like `minecraft:spruce_planks` and only has to place
them. Adding a model is a `models.json` entry; the mod never changes.

Two kinds of model are served. The **trained checkpoints** sample tokens on the GPU.
The **agentic models** ([Track E](../docs/agentic.md)) have no checkpoint at all: a
frontier LLM writes a build *program*, and the server executes it one command at a
time, so the build goes up part by part — foundation, walls, openings, roof — with
each command named in chat. Those cost API tokens instead of GPU time, and the cost
is reported when the build finishes.

## Quick start

**1. Start the server** (on the machine with the GPU):

```bash
cd deploy/inference
python -m blockgen_server.server --host 0.0.0.0 --port 8000 --preload
```

Check it: `curl http://localhost:8000/models`

**2. Build the mod** (needs Java 21):

```bash
cd deploy/mod
./gradlew build
# -> build/libs/blockgen-mod-0.1.0.jar
```

**3. Install** — drop that jar (plus [Fabric API](https://modrinth.com/mod/fabric-api))
into your 1.21.1 Fabric instance's `mods/` folder.

**4. In game:**

```
/blockgen server ws://192.168.1.20:8000/ws   # your GPU box's LAN address
/gen                                          # build at your feet
/model                                        # list models, show current
/model canon16_flat                           # switch
/blockgen undo                                # revert the last build

/model agentic list                           # which LLMs the agentic entry serves
/model agentic gemini-3.5-flash               # switch to one of them
/gen a wooden windmill on a stone base        # real text conditioning
```

## Commands

| Command | Effect |
|---|---|
| `/gen` | Generate with the current model at your feet |
| `/gen <prompt...>` | Text-conditioned generation (needs a `[text]` model) |
| `/generate`, `/blockgen gen` | Aliases of `/gen` |
| `/blockgen at <x y z> [prompt]` | Build at explicit coords — works from the console and command blocks |
| `/model` | List models; `>` marks the current one. A model *group* shows as one row |
| `/model <name>` | Switch model (`default` = server's default) |
| `/model <group> list` | List the models a group serves — e.g. `/model agentic list` |
| `/model <group> <model>` | Switch to one of them (stored as `group:model`) |
| `/blockgen steps on\|off` | Show each build step in chat (agentic models; default on) |
| `/blockgen cancel` | Stop the running generation |
| `/blockgen undo` | Revert the last build (last 5 kept) |
| `/blockgen status` | Server URL, model, sampling settings, live progress |
| `/blockgen server <url>` | Point at an inference server |
| `/blockgen speed <n>` | Blocks placed per tick (default 32 ≈ 640/s) |
| `/blockgen seed <n>` | Generate with a fixed seed (reproducible) |

`/gen <prompt>` takes the rest of the line, so `/gen cancel` prompts with the word
"cancel". Control verbs live under `/blockgen` so a prompt can be any text.

Ops only on a dedicated server; anyone in singleplayer (where you normally have
permission level 0, so requiring op would make it unusable).

## Models

`deploy/inference/models.json` is the registry. Current entries:

| Name | Kind | What it is |
|---|---|---|
| `native_oriented` *(default)* | `piece_ar` | 3D-BPE piece AR at native 32³, **oriented**. Places blocks facing the right way — stairs point N/S/E/W, logs sit on x/y/z axes, slabs top/bottom. Improved successor to `native_bpe`. |
| `native_bpe` | `piece_ar` | 3D-BPE piece AR at native 32³. Big builds (~1.7k blocks), streams a piece at a time. Non-oriented (stairs/logs get default facing). |
| `canon16_flat` | `voxel_ar` | Per-voxel AR on canon-16 miniatures. Highest validity; one block per token. |
| `text_cond` | `cond_piece_ar` | CLIP-text-conditioned. `/gen a small stone house`. Bare `/gen` samples its null branch. |
| `agentic` | `agentic` | **Group (12 models).** An LLM writes a build program; executed command by command. Real text conditioning, any canvas size, ~2k-block builds. `/model agentic list`. |
| `agentic_plus` | `agentic` | Same group with the full loop — plan, 2 repair rounds, **visual self-critique**. Slower, ~4× the cost, better builds. |

Add one by appending an entry — no server code changes if its `kind` already exists:

```json
"my_run": {
  "kind": "piece_ar",
  "description": "shown by /model",
  "checkpoint": "outputs/my_run/model.pt",
  "piece_vocab": "outputs/my_run/piece_vocab.json",
  "max_seq_len": 5480, "pe": "phase4", "d_model": 256, "layers": 6
}
```

**Every entry must name the vocab file its checkpoint was trained against.** A
checkpoint does not record its own vocabulary, and a piece token id is meaningless
without the patterns it expands to. The server cross-checks vocab size against the
checkpoint's `lm_head` and refuses to run on a mismatch — but two vocabularies can
share a size and disagree on every pattern, so that check catches swapped files, not
subtler mixups. See "Vocabulary recovery" below.

New architectures need a `Backend` subclass in `blockgen_server/backends.py`
implementing `load()` and `stream()`.

### Model groups (the agentic entry)

One entry can serve many models. The agentic entry is a *provider*, not a checkpoint,
so listing `agentic:gpt-5-mini`, `agentic:gemini-3.5-flash`, … as separate rows would
bury the four trained models. Instead it declares a `models` list, shows as **one row**
in `/model`, and members are addressed as `agentic:<model>`:

```
/model agentic list                 -> the 12 models it serves, with availability
/model agentic gemini-3.5-flash     -> sets model = agentic:gemini-3.5-flash
```

The vendor is inferred from the model name (`gpt-*` → OpenAI, `gemini*` → Gemini,
`claude*` → Anthropic), so you type the model, not the vendor twice. Write
`vendor:model` explicitly for anything that does not match a prefix. `strict_models:
false` lets an unlisted name through, which is how you try a model released after
this table was written.

Keys come from the repo `.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`). A member whose key is missing is listed as unavailable **with
that reason**, rather than failing at `/gen` time.

Each entry pins its own loop (`canvas`, `plan`, `examples`, `repair_rounds`,
`critique_rounds`, `reasoning_effort`, `max_tokens`) — that is the whole difference
between `agentic` and `agentic_plus`. See [docs/agentic.md](../docs/agentic.md).

### What a build looks like in chat

```
Generating "a wooden windmill on a stone base"...
  [1/48] fill cobblestone 15 0 15 25 4 25
  [2/48] replace cobblestone mossy_cobblestone 15 0 15 25 1 25
  [3/48] fill stone_bricks 16 5 16 24 9 24
  ...
Done: 2915 blocks in 65.1s from agentic:gemini-3.5-flash (seed 11). /blockgen undo to remove.
  gemini:gemini-3.5-flash  48 commands  2147+1071 tokens  cost unknown (model not in the price table)
```

Steps print as their blocks land, so the labels track the build going up. A model
missing from the server's price table reports **cost unknown** rather than `$0.00` —
printing zero for a paid call would read as free. Add a row to
`blockgen/agentic/providers.py::PRICES` to get a number.

## Known limitations

**Stairs, logs and doors face the wrong way — on the non-oriented models only.**
The texture-variant corpora derive `block_data` from GrabCraft *texture-variant*
indices rather than true legacy metadata, so orientation was never in their training
data. Every stairs id in those vocabs carries exactly one data value (`53 -> {2}`,
`114 -> {7}`) where real metadata would spread over 0..7. For those models
(`native_bpe`, `canon16_flat`, `text_cond`) the server places default states instead
of inventing facings — a facing fabricated from a texture index would be wrong.

The **`native_oriented`** model is trained on true legacy metadata (stairs 53 spread
over data 0..7, logs 17 over 0..15). Its vocab is flagged `oriented=true`, so
`blockmap.modern_state(..., oriented=True)` translates the legacy facing/axis bits
into modern blockstate properties: stairs get `[facing=…,half=…]`, logs get `[axis=…]`
(or become the all-bark `*_wood` block), slabs get `[type=top|bottom]`. Doors keep a
valid default (their facing is entangled with open/hinge bits the corpus merged
away). No mod change was needed — `GenerationSession.resolve()` already parses full
state strings like `minecraft:oak_stairs[facing=east,half=bottom]`.

**Doors, beds and other multi-block pieces come out broken.** They are single voxels
in the vocabulary, so only the lower half is placed.

**Text conditioning is weak — on `text_cond`.** Palette transfers; geometry mostly
does not. Expect a house, not your house. See `results.md` T15. The **agentic** models
do not share this limitation (the conditioning is the base model's), but they cost
money per build and take 25–70s before the first block appears, since the whole
program is written before any of it is executed.

**An agentic build is not streamed as it is *generated*** — it is streamed as it is
*executed*. The LLM call finishes first (nothing appears during it), then the program
replays command by command. The wait is real; the build-up afterwards is fast.

**Builds are placed at your feet, extending +x/+y/+z**, and overwrite what is there.
`/blockgen undo` reverts.

## Vocabulary recovery

*(Runs from 2026-07-16 on save their vocab automatically — `experiments_native` now
writes `piece_vocab.json` / `block_vocab.json` before training starts. This section
is for older runs. To reproduce T18 itself, see `docs/reproduce-native.md`.)*

`experiments_native.py` used to save neither arm's vocabulary — `arm_bpe` wrote a
`cluster_meta.json` of counts but not the `ClusterVocab`, and `arm_flat` built its
`BlockVocab` inline and dropped it. Both are recoverable because every input is seeded
and deterministic:

```bash
python deploy/inference/scripts/rebuild_native_vocab.py --run outputs/run_20260715_065938_native
python deploy/inference/scripts/rebuild_native_vocab.py --run outputs/run_20260715_062404_native --arm canon16_flat
```

Both rebuilds match their checkpoint's `lm_head` exactly (713 and 435). The BPE
rebuild is *not* interchangeable with the exported `data/minecraftace/houses_32_bpe`
vocab — same size, different patterns (different md5), because it was learned over
the augmented train split rather than all houses. Using the wrong one decodes to
noise while looking perfectly healthy, so if you train a new BPE run, save its vocab
with `save_piece_vocab`.

## Development

```bash
# server-side tests (decoder parity, cached-decode parity)
python -m pytest deploy/inference/tests/ -q

# generate without Minecraft: prints block count, bbox and palette
python deploy/inference/scripts/smoke_generate.py --model native_bpe --seed 0
python deploy/inference/scripts/smoke_generate.py --list
python deploy/inference/scripts/smoke_generate.py --list --group agentic

# the agentic path with no API key and no network (scripted provider)
python deploy/inference/scripts/smoke_generate.py --model agentic:mock \
    --prompt "a small cottage" --steps

# re-validate the block map against a real Minecraft registry
java -DbundlerMainClass=net.minecraft.data.Main -jar server.jar --reports
python deploy/inference/scripts/export_blockmap.py \
    --report generated/reports/blocks.json \
    --vocab data/minecraftace/houses_32_bpe/houses_32_bpe_piece_vocab.json

# run the mod in a dev client
cd deploy/mod && ./gradlew runClient
```

### Verified end-to-end

Against a real 1.21.1 dedicated server driven from its console, with the mod talking
to a live inference server:

```
> blockgen model
Models (current: native_bpe)
> native_bpe [loaded]
  canon16_flat
  text_cond [text]
> blockgen at 0 -59 0
Done: 473 blocks in 1.2s from native_bpe (seed 399017250). /blockgen undo to remove.
> blockgen undo
Reverted 461 blocks.
```

(Fewer reverted than placed is correct: overlapping BPE pieces write some positions
twice, and undo keys by position.)

Reproduce with `./gradlew runServer`, then type those commands into the console —
`blockgen at` needs no player, which is what makes this testable headlessly.
