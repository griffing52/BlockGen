"""Entry point: `python -m tools.lab`.

Kept separate from `api.py` so the server can be imported and driven from a test
or a notebook without argparse or a browser launch happening as a side effect.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from tools.lab import DEFAULT_PORT, LAB_ROOT
from tools.lab.api import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.lab", description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; there is no auth on this thing")
    ap.add_argument("--open", action="store_true", help="launch a browser at the hub")
    args = ap.parse_args(argv)

    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        server = serve(args.host, args.port)
    except OSError as exc:
        print(f"[lab] cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"[lab] BlockLab on {url}  (db: {LAB_ROOT / 'lab.db'})", file=sys.stderr,
          flush=True)
    if args.open:
        # After a beat, so the first request does not race the accept loop.
        #
        # Wrapped because the launcher is not ours and is routinely broken: in a
        # VS Code terminal `BROWSER` points at a helper that opens URLs over a
        # Unix socket, tmux panes inherit that variable, and the socket dies with
        # the window that made it. The helper then fails with ENOENT on a timer
        # thread, which without this reads like the server crashed -- it did not,
        # and the URL above is the whole point of the line.
        def _open() -> None:
            try:
                if not webbrowser.open(url):
                    raise RuntimeError("no browser handler")
            except Exception as exc:
                print(f"[lab] could not open a browser ({type(exc).__name__}: "
                      f"{exc}); open {url} yourself", file=sys.stderr, flush=True)

        threading.Timer(0.4, _open).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[lab] bye", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
