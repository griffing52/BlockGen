#!/usr/bin/env bash
# Wait for a running worldcrop_ingest.sh to finish, then re-run every month whose
# last "done" line in ingest.log has rc!=0 (resumable, so only unfinished worlds run).
cd "$(dirname "$0")/.."
LOG=data/minecraft/worldcrop/logs/ingest.log
while pgrep -f "worldcrop_ingest.sh" >/dev/null; do sleep 120; done
FAILED=$(grep -E '^\[.*\] done ' "$LOG" | awk '{m=$4; sub("rc=","",$5); rc[m]=$5} END{for (m in rc) if (rc[m]!=0) print m}' | sort)
echo "[$(date '+%F %T')] retry pass: $FAILED" | tee -a "$LOG"
[ -n "$FAILED" ] && scripts/worldcrop_ingest.sh "${1:-12}" $FAILED
