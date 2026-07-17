#!/bin/bash
# Start the BlockGen inference server on the LAN, using the repo's venv.
#
#   ./deploy/inference/run_server.sh                      # 0.0.0.0:8000, preloaded
#   BLOCKGEN_PORT=9000 ./deploy/inference/run_server.sh   # different port
#
# Then point the mod at it:  /blockgen server ws://<this-machine-ip>:8000/ws
#
# Every override here is BLOCKGEN_-prefixed on purpose. A bare ${HOST:-0.0.0.0} reads
# whatever HOST the shell already has, and conda sets HOST to its build triplet
# (x86_64-conda-linux-gnu) -- so under a conda env the server tried to bind to that as
# a hostname and died with "Name or service not known".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PORT="${BLOCKGEN_PORT:-8000}"
HOST="${BLOCKGEN_HOST:-0.0.0.0}"
PYTHON="${BLOCKGEN_PYTHON:-$REPO/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "No python at $PYTHON. Set PYTHON=/path/to/python or create the venv." >&2
  exit 1
fi

# blockgen_server is not installed as a package; put both it and the repo on the path.
export PYTHONPATH="$HERE:$REPO${PYTHONPATH:+:$PYTHONPATH}"

IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "<this-machine-ip>")
echo "Starting BlockGen inference server on $HOST:$PORT"
echo "In Minecraft:  /blockgen server ws://$IP:$PORT/ws"
echo

cd "$HERE"
exec "$PYTHON" -m blockgen_server.server --host "$HOST" --port "$PORT" --preload
