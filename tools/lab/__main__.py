"""Entry point: `python -m tools.lab`.

Kept separate from `api.py` so the server can be imported and driven from a test
or a notebook without argparse or a browser launch happening as a side effect.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

from tools.lab import DEFAULT_PORT, LAB_ROOT
from tools.lab.api import serve


def already_serving(host: str, port: int, timeout: float = 0.6) -> bool:
    """Is the thing holding this port one of ours?

    `/api/datasets` rather than a newer route: an older BlockLab left running
    from before some feature existed is still a BlockLab, and answering "no" for
    it would send the reader hunting for a port conflict that is really just
    their own server. Any JSON on that path is treated as ours -- the question is
    "should I tell them it is already up", not "is this authentic", and there is
    nothing to defend against on loopback.
    """
    try:
        with urllib.request.urlopen(
                f"http://{host}:{port}/api/datasets", timeout=timeout) as r:
            json.load(r)
        return True
    except Exception:
        return False


def _launch(url: str, delay: float = 0.0) -> None:
    """Open a browser without letting the launcher's failure look like ours.

    `delay` so the first request does not race the accept loop when we are about
    to serve; zero when the server is someone else's and already up.

    Wrapped because the launcher is not ours and is routinely broken: in a VS
    Code terminal `BROWSER` points at a helper that opens URLs over a Unix
    socket, tmux panes inherit that variable, and the socket dies with the window
    that made it. The helper then fails with ENOENT on a timer thread, which
    without this reads like the server crashed -- it did not, and the URL we
    printed is the whole point of the line.
    """
    def go() -> None:
        try:
            if not webbrowser.open(url):
                raise RuntimeError("no browser handler")
        except Exception as exc:
            print(f"[lab] could not open a browser ({type(exc).__name__}: "
                  f"{exc}); open {url} yourself", file=sys.stderr, flush=True)

    threading.Timer(delay, go).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.lab", description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"default {DEFAULT_PORT}; 0 asks the OS for a free one")
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; there is no auth on this thing")
    ap.add_argument("--open", action="store_true", help="launch a browser at the hub")
    args = ap.parse_args(argv)

    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        server = serve(args.host, args.port)
    except OSError as exc:
        # The default port is deliberately fixed, not auto-assigned: the bench
        # prints `view: http://127.0.0.1:8765/...` at the end of every run
        # (runner.LAB_PORT) and the docs quote it, so a port that moved per
        # launch would falsify a link the reader is being told to click.
        #
        # Which makes "in use" almost always mean "you already have one open",
        # not "pick another port" -- so check, and answer the question they
        # actually have.
        if exc.errno == errno.EADDRINUSE and already_serving(args.host, args.port):
            url = f"http://{args.host}:{args.port}/"
            print(f"[lab] BlockLab is already running on {url} — "
                  f"using that one (nothing new started).", file=sys.stderr)
            print(f"[lab] to run a second instance anyway: "
                  f"python -m tools.lab --port 0", file=sys.stderr, flush=True)
            if args.open:
                _launch(url)
            # 0, not 1: the caller wanted a reachable lab and there is one.
            return 0
        print(f"[lab] cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        if exc.errno == errno.EADDRINUSE:
            print(f"[lab] something else is on {args.port}. Try "
                  f"'python -m tools.lab --port 0' for any free port.",
                  file=sys.stderr)
        return 1

    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"[lab] BlockLab on {url}  (db: {LAB_ROOT / 'lab.db'})", file=sys.stderr,
          flush=True)
    if args.open:
        _launch(url, delay=0.4)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[lab] bye", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
