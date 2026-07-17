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
```

## Commands

| Command | Effect |
|---|---|
| `/gen` | Generate with the current model at your feet |
| `/gen <prompt...>` | Text-conditioned generation (needs a `[text]` model) |
| `/generate`, `/blockgen gen` | Aliases of `/gen` |
| `/blockgen at <x y z> [prompt]` | Build at explicit coords — works from the console and command blocks |
| `/model` | List models; `>` marks the current one |
| `/model <name>` | Switch model (`default` = server's default) |
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
| `native_bpe` *(default)* | `piece_ar` | 3D-BPE piece AR at native 32³. Big builds (~1.7k blocks), streams a piece at a time. |
| `canon16_flat` | `voxel_ar` | Per-voxel AR on canon-16 miniatures. Highest validity; one block per token. |
| `text_cond` | `cond_piece_ar` | CLIP-text-conditioned. `/gen a small stone house`. Bare `/gen` samples its null branch. |

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

## Known limitations

**Stairs, logs and doors face the wrong way.** Not a bug in the mod — the corpora
derive `block_data` from GrabCraft *texture-variant* indices rather than true legacy
metadata, so orientation was never in the training data. Every stairs id in the piece
vocab carries exactly one data value (`53 -> {2}`, `114 -> {7}`) where real metadata
would spread over 0..7. The server places default states instead of inventing
facings. Fixing this properly means orienting stairs from neighbours in a post-pass,
or a corpus that preserves metadata.

**Doors, beds and other multi-block pieces come out broken.** They are single voxels
in the vocabulary, so only the lower half is placed.

**Text conditioning is weak.** Palette transfers; geometry mostly does not. Expect a
house, not your house. See `results.md` T15.

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
