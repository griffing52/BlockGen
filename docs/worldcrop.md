# World-crop corpus: structures mined from whole worlds

`blockgen/data/worldcrop/` turns full Minecraft world saves into a corpus of
player-built structures. It exists because community *schematics* are scarce (tens of
thousands) while community *worlds* are plentiful: the Minecraft Archive Project on
archive.org alone holds ~175k worlds (2011-2018, ~3.6 TB), each with anything from a
single house to a whole city inside. The research write-up (sources, licensing,
alternatives) is `research.md §F`; this page is the operator's manual.

## Where the worlds come from

| Source | How we read it | Notes |
|---|---|---|
| **Minecraft Archive Project** `Worlds_YYYYMM` items on archive.org | `remote.MonthArchive` opens the monthly 10-65 GB zip **over HTTP range requests** and pulls one world archive (1-10 MB) at a time. Nothing is downloaded whole. | 89 months, ~1,800 world archives per month, provenance in the month's `threads.ndjson` / `resources.ndjson` (post permalink, title, description) |
| Local directories / archives | `--local DIR` walks for `level.dat` or `.zip/.rar/.7z` | e.g. CurseForge downloads, Hermitcraft seasons, our own worlds |

Archive formats: zip and 7z via `7z`; **RAR needs `unrar`** (Debian's 7zip has no RAR
codec; about a quarter of archived worlds are `.rar`). Install one without touching the
system, then point the runner at it; `bsdtar` is the fallback:

```bash
conda create -y -p ~/.local/unrar-env -c conda-forge unrar
export BLOCKGEN_UNRAR=~/.local/unrar-env/bin/unrar
```

## Reading worlds (`anvil.py`)

A dependency-light reader (nbtlib + numpy; no Amulet, whose licence is not free) for
all three chunk encodings seen 2009-2018+:

* McRegion (`.mcr`, 2011): `Level.Blocks` 16×128×16 XZY bytes;
* Anvil numeric (`.mca`, 1.2-1.12): 16³ sections, `Blocks` + `Add` + `Data` nibbles;
* Anvil palette (1.13+): per-section palettes with packed `BlockStates` (bitstream
  before 20w17a, padded after) or 1.18 `block_states`; names are folded onto the legacy
  `(id, data)` vocabulary through `utils.block_remap`.

Chunks come back as sparse dicts of non-empty 16³ `(ids uint16, data uint8)` sections
in **XYZ** order, the layout of `utils.data.Structure`. `level.dat` gives generator,
preset, seed and version; `data/<Village|Mineshaft|Stronghold|Temple|Fortress|Monument>.dat`
gives the bounding boxes of every vanilla structure piece.

## Finding structures (`detect.py`)

Two routes, chosen per world:

* **Flat worlds** (`generatorName: flat`, ~half of PlanetMinecraft project maps): parse
  the superflat preset into its layer stack; every block that is not the layer block at
  that height (and not a decoration/fluid the flat generator can spawn) is player-placed.
  A `flat` label is **rejected** when ores occur in >5% of chunks (converted worlds lie),
  falling back to the heuristic route.
* **Terrain worlds**: block ids are tiered (`blocks.py`). *STRONG* ids (planks, stairs,
  slabs, glass, wool, doors, fences, redstone, quartz, …) seed candidates; *WEAK* ids
  (cobblestone, logs, hardened clay, obsidian, chests, …) may grow a component but never
  start one, so dungeons and forests yield nothing; everything else is terrain. Vanilla
  structure boxes (dilated 3) zero out seeds, so villages, strongholds and mineshafts
  cancel out even though they are built of planks and cobble.

Then: seed-bearing chunks are clustered (union-find, one-chunk gaps bridged) and tiled
to ≤24×24 chunks; each tile is materialised only over the seeds' y-range; the
`strong|weak` mask is dilated by one voxel (slabs and stairs across one-block gaps
connect) and labelled with 26-connectivity; a component survives with ≥12 seed voxels,
≥24 member voxels, seeds ≥10% of members, ≥2 block ids and a sane bbox. Finally
**ore-like families** are dropped: ≥8 components in one world sharing the same top-2
seed ids with median <150 seeds are modded ores / replaced ores read as vanilla blocks.

Each crop keeps the **full bbox** (`ids`, `data`), the component `mask`, and metadata
(world id, provenance, bbox in world coordinates, seed/weak counts, mean
`InhabitedTime`, route). Downstream can train on "structure only" (`ids*mask`) or
"structure in context". Components longer than 320 blocks on a side are recorded but
not saved.

## Running

```bash
export BLOCKGEN_UNRAR=/path/to/unrar          # optional, for .rar worlds
python -m blockgen.data.worldcrop.run --month 201407 --limit 400 --workers 12
python -m blockgen.data.worldcrop.run --local /path/to/worlds
python -m blockgen.data.worldcrop.analyze data/minecraft/worldcrop/201407   # stats + sample sheets
```

Outputs under `data/minecraft/worldcrop/<month>/`: `worlds.jsonl` (one record per
archive: provenance, `level.dat` facts, scan summary, status), `crops.jsonl`,
`crops/<world_id>_<k>.npz`, `analysis.json`, `sheet_masked.png`, `sheet_context.png`.
Runs are resumable (processed world ids are skipped).

## Licensing stance

Worlds are their creators' copyright and the archive carries no licence, so this corpus
is **pretraining-only and never redistributed**. What we can release is the code, the
per-crop provenance index (archive item, member path, post permalink, bbox) and the
statistics. See `research.md §F.3` for the two-tier plan and the creator-outreach list.

## Bulk ingest driver

`scripts/worldcrop_ingest.sh [WORKERS] [MONTH ...]` loops over archive months
(all 89 by default, best-tested formats first) and appends to
`data/minecraft/worldcrop/<month>/`. It is resumable: re-running skips worlds
already listed in each month's `worlds.jsonl`. Progress is in
`data/minecraft/worldcrop/logs/ingest.log`, per-month logs sit beside it.
Launch detached with `nohup scripts/worldcrop_ingest.sh 12 &`.

Two failure modes seen on the first full pass are handled automatically:
archive.org answers the month-opening request with a sporadic 500 (now retried
with backoff), and a worker can be OOM-killed on a huge world, which used to abort
the whole month. Workers now run under an address-space cap (`--mem-gb`, default
5) so a giant world fails cleanly with `MemoryError`, and if a worker still dies
the pool is rebuilt and unfinished jobs re-queued; a job that kills the pool
twice is recorded as `worker_killed`. `scripts/worldcrop_retry.sh` waits for the
driver and re-runs every month that exited non-zero.

## Tile edges and cut builds (2026-10-11)

Components are labelled per tile (24 chunks on a side). A build that straddles a tile
edge used to be saved as two halves, or as a half plus a whole, because the dedupe only
merged near-identical boxes. `extract_crops` now treats any component touching a tile's
x/z edge as *cut*: it is not saved, and a new tile is loaded around the component's
bounding box (plus a chunk of halo), up to `retile_depth` times. A component that still
does not fit inside one tile is recorded with `cut: true` and `oversize: true`. On 110
re-run 2014-07 worlds this merged 26 split builds (ships, rail lines, domes) and flagged
23 as too large; everything else was byte-identical. Months ingested before this fix
(2014-07 through 2017-12, early 2014) still contain the split versions; re-running those
months regenerates them (resumable runs skip finished worlds, so delete or rename the
month's `worlds.jsonl` first).

## Provenance

Each `Worlds_YYYYMM` zip ships a static finding aid under `threads/detail/*.html`: one page
per forum thread with title, author, original URL, creation date, description and a
"Mirrored resources" table linking to the archive members. The ndjson sidecars carry no
referer for download resources, so the HTML table is the only reliable archive-to-thread
join. `provenance.py` fetches all pages of a month in a single range request (about 8 MB)
and inflates them by hand; `MonthArchive.worlds()` applies it, so new ingests carry
`entry.title / description / thread / meta.author`. For months ingested before this,
`python -m blockgen.data.worldcrop.provenance --rejoin` rewrites `worlds.jsonl` in place
(it skips the month the driver is working on). Join rate is about 99% of archives.

## Learned gate: built vs terrain

`gate.py` scores each saved crop with `p_built`. Positives are the GrabCraft and 3D-Craft
raw corpora plus the text2mc houses in `houses_48`, masked to their blocks the way a crop
is masked to its component. Negatives are natural terrain patches cut by `gate_data.py`
from chunks that contain no built block at all (`gate/terrain_<month>.npz`). Two more negative sets were needed before real edited terrain was caught: synthetic
surface skins (`gate_data.py synth --kind skin`; the detector masks a mountain to the 1-8
block skin of its seed material, not the solid mound) and, decisively, 10k crops mined
from the corpus by composition (at most three block ids, at least 97% terrain-family
blocks; `gate/mined_terrain_0.npz`). Features are
295 cheap descriptors (block-id histogram, material-family fractions, heightmap and
column statistics, exposed-face ratio, enclosed air, mirror symmetry, planarity, vertical
profile); the model is a gradient-boosted tree. A hand-labelled held-out set of 192 random
crops (`gate/eval_labels.jsonl`, labels built / terrain / fragment / unsure) is never used
for training and is the acceptance test.

    python -m blockgen.data.worldcrop.gate_data --month 201407 --worlds 30   # negatives
    python -m blockgen.data.worldcrop.gate train                            # gate/model.pkl
    python -m blockgen.data.worldcrop.gate score --workers 8                # <month>/gate_scores.jsonl
    python -m blockgen.data.worldcrop.gate sheets --thr 0.5                 # pass / reject / border sheets

The lab (`python -m tools.lab`) lists the crops as the raw corpus `raw:worldcrop`; labels
made there key on corpus position, which shifts while the ingest is still adding months.
