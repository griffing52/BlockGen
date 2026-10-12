#!/usr/bin/env bash
# Resume the full-archive pass after an interruption: run the months not yet
# attempted, then the retry pass over months whose last run failed.
# Launch detached from the login session so a closed tmux does not kill it:
#   systemd-run --user --unit blockgen-ingest -p WorkingDirectory=$PWD bash scripts/worldcrop_resume.sh
cd "$(dirname "$0")/.."
LOG=data/minecraft/worldcrop/logs/ingest.log
WORKERS=${1:-12}
ALL="$(seq -f '2014%02g' 7 12) $(for y in 2015 2016 2017; do seq -f "${y}%02g" 1 12; done) \
     $(seq -f '2014%02g' 1 6) $(for y in 2013 2012 2011; do seq -f "${y}%02g" 12 -1 1; done) \
     2009-2010 $(seq -f '2018%02g' 1 12)"
TODO=""
for m in $ALL; do grep -q "] done  $m rc=" "$LOG" || TODO="$TODO $m"; done
echo "[$(date '+%F %T')] resume pass: $TODO" | tee -a "$LOG"
[ -n "$TODO" ] && scripts/worldcrop_ingest.sh "$WORKERS" $TODO
scripts/worldcrop_retry.sh "$WORKERS"
