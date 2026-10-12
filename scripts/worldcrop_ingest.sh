#!/usr/bin/env bash
# Stream Minecraft Archive Project months through the world-crop extractor.
# Resumable: re-running skips worlds already present in each month's worlds.jsonl.
# Usage: scripts/worldcrop_ingest.sh [WORKERS] [MONTH ...]
set -u
cd "$(dirname "$0")/.."
WORKERS=${1:-12}; shift || true
export BLOCKGEN_UNRAR=${BLOCKGEN_UNRAR:-$HOME/.local/unrar-env/bin/unrar}
export BLOCKGEN_SCRATCH=${BLOCKGEN_SCRATCH:-$HOME/.cache/blockgen_worldcrop_scratch}
OUT=data/minecraft/worldcrop
mkdir -p "$OUT/logs" "$BLOCKGEN_SCRATCH"
if [ $# -gt 0 ]; then MONTHS="$*"; else
  # Anvil-numeric years first (best-tested reader), then early McRegion, then 1.13+ last.
  MONTHS="$(seq -f '2014%02g' 7 12) $(for y in 2015 2016 2017; do seq -f "${y}%02g" 1 12; done) \
          $(seq -f '2014%02g' 1 6) $(for y in 2013 2012 2011; do seq -f "${y}%02g" 12 -1 1; done) \
          2009-2010 $(seq -f '2018%02g' 1 12)"
fi
for m in $MONTHS; do
  echo "[$(date '+%F %T')] start $m" | tee -a "$OUT/logs/ingest.log"
  python -m blockgen.data.worldcrop.run --month "$m" --workers "$WORKERS" >> "$OUT/logs/$m.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] done  $m rc=$rc" | tee -a "$OUT/logs/ingest.log"
  rm -rf "$BLOCKGEN_SCRATCH"/* 2>/dev/null
done
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OUT/logs/ingest.log"
