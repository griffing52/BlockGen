#!/usr/bin/env bash
# Pause / resume the world-crop ingest (driver + all workers) without losing progress.
# Usage: scripts/worldcrop_pause.sh pause|resume|status
sig=${1:-status}
pids=$(pgrep -f "[w]orldcrop_ingest.sh|[b]lockgen.data.worldcrop.run")
case "$sig" in
  pause)  kill -STOP $pids; echo "paused $(echo $pids | wc -w) processes" ;;
  resume) kill -CONT $pids; echo "resumed $(echo $pids | wc -w) processes" ;;
  *) ps -o pid,stat,pcpu,etimes,args -p $(echo $pids | tr ' ' ',') ;;
esac
