#!/usr/bin/env bash
# Rebuild the corpus from the raw sources up to the shards the training configs read.
#
# Order matters: every stage reads the one before it.
#   manifest build    world-crop months -> manifest/worldcrop_<month>.parquet (resumable)
#   manifest curated  the all_32 / houses_32 / houses_48 caches -> manifest/curated_<cache>.parquet
#                     (houses_32 is the cache sft-houses32 and eval-holdout index; without it both tiers are empty)
#   templates         jar structure templates -> manifest/templates.parquet
#   dedup             exact + MinHash clusters -> dedup.parquet (tiers read is_rep / cluster)
#   kinds             build / mixed / village / mineshaft -> kinds.parquet (reads dedup)
#   tiers build       every configs/corpus/*.yaml -> tiers/<name>.{json,ids.txt}; a split-backed tier
#                     with zero rows raises, which stops the script here
#   shards build      every (tier, tokenizer) pair that has a shard on disk today: the pairs
#                     configs/train/*.yaml use plus the pretrain-all / pretrain-clean / templates
#                     voxel shards no current config names
#
# A tier rebuild changes its sha; a shard built from the old sha warns (StaleShardWarning)
# when opened, so the shard stage always follows the tier stage and covers every tier the
# tier stage rebuilt. The vanilla tier has no shard and gets none.
#
# Usage: scripts/corpus_refresh.sh [--dry-run] [--workers N]
#   --dry-run   print the commands instead of running them
set -euo pipefail

cd "$(dirname "$0")/.."

DRY=0
WORKERS=6
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --workers) WORKERS="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

run() {
    echo "+ $*"
    if [ "$DRY" -eq 0 ]; then
        "$@"
    fi
}

# 1. manifest: world crops, then the curated caches
run python -m blockgen.corpus.manifest build --workers "$WORKERS"
run python -m blockgen.corpus.manifest curated --workers "$WORKERS" --caches all_32 houses_32 houses_48

# 2. templates: the game's own structure pieces, a manifest source of their own
run python -m blockgen.corpus.templates --workers "$WORKERS"

# 3. sidecars the tier filters read
run python -m blockgen.corpus.dedup
run python -m blockgen.corpus.kinds

# 4. every tier config; exclusions are ordered inside tiers.build
run python -m blockgen.corpus.tiers build

# 5. shards. The houses tiers hold whole builds up to 8192 blocks (configs/train/*houses32*, smoke,
#    rft-voxel-5m); the pretraining tiers window at the default 2048 (anneal-pretrain-5m names anneal;
#    pretrain-all, pretrain-clean and templates have shards on disk at the same defaults and are
#    rebuilt so they do not warn stale). Facade shards: per-face (houses32-facades, smoke-facades),
#    joint (houses32-facades-joint*, rft-facades-joint-d4-20m) and joint d4 on the train tier only
#    (houses32-facades-joint-d4*); holdout tiers are never augmented.
for tier in sft-houses32 eval-holdout; do
    run python -m blockgen.corpus.shards build "$tier" --max-blocks 8192 --workers "$WORKERS"
    run python -m blockgen.corpus.shards build "$tier" --kind facades --workers "$WORKERS"
    run python -m blockgen.corpus.shards build "$tier" --kind facades --layout joint --workers "$WORKERS"
done
run python -m blockgen.corpus.shards build sft-houses32 --kind facades --layout joint --augment d4 --workers "$WORKERS"
for tier in anneal pretrain-clean pretrain-all templates; do
    run python -m blockgen.corpus.shards build "$tier" --workers "$WORKERS"
done

echo "corpus refresh complete"
