"""Serve the 2AFC study and collect responses straight to disk.

The exported `study.html` is self-contained and works from a `file://` path, but
that flow ends with each rater copying a blob of JSON and sending it back. Across
a dozen people that is where a study loses data: mail clients wrap and mangle
JSON, and some fraction of raters simply close the tab. This serves the same
page and takes a POST instead, writing one file per session.

    python scripts/human_study_serve.py --run outputs/run_20260918_human_study

Then open the printed URL. Responses land in the run directory as
`rater_<session>.json`, which is exactly what `human_study_analyze.py --responses`
expects, so there is no reformatting step.

Standard library only, matching `tools/lab` -- a script that has to be re-run
for months should not acquire a dependency that can rot.

Remote work
-----------
If the repository is on a remote machine, the page still opens locally: VS Code
Remote and `ssh -L` both forward the port, and the browser on your laptop
reaches 127.0.0.1 on the remote host. See `--help` and the runbook.

Exposing it to raters is a separate decision, and the reason `--host` defaults
to loopback: **there is no authentication on this server**. Anyone who can reach
the port can submit responses, and the answer key is deliberately NOT served, but
the stimuli are. Prefer a tunnel you control, or a university web space, over
binding to 0.0.0.0 on an open network.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

#: Injected before the page's own script so the page prefers the HTTP path.
#: The page falls back to its existing behaviour when this is absent, so the
#: exported file keeps working unchanged from `file://`.
INJECT = '<script>window.__SUBMIT_URL__="/submit";</script>\n'

MAX_BODY = 8 * 1024 * 1024          # a session is ~40 rows; this is generous


class Handler(BaseHTTPRequestHandler):
    run_dir: Path
    page: bytes
    lock = threading.Lock()

    def log_message(self, fmt: str, *args) -> None:      # quieter than default
        return

    # --- reads ------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/study.html"):
            self._send(200, "text/html; charset=utf-8", self.page)
        elif path == "/status":
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(self._status(), indent=1).encode())
        else:
            # Never serve the run directory wholesale: answer_key.json lives
            # there, and a rater who can read it can read the conditions.
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    # --- writes -----------------------------------------------------------
    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/submit":
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if not 0 < n <= MAX_BODY:
            self._send(413, "text/plain; charset=utf-8", b"bad length\n")
            return
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw)
            responses = payload["responses"]
            assert isinstance(responses, list) and responses
        except Exception:                                   # noqa: BLE001
            self._send(400, "text/plain; charset=utf-8", b"bad payload\n")
            return

        session = str(payload.get("session", "unknown"))
        # A session id reaches the filesystem, so constrain it rather than trust
        # it: anything outside this class becomes an underscore.
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session)[:48] or "unknown"
        payload["received_at"] = datetime.now(timezone.utc).isoformat()

        with self.lock:
            out = self.run_dir / f"rater_{safe}.json"
            # Never overwrite: two raters can be handed the same session, and a
            # silent overwrite would delete someone's work.
            k = 1
            while out.exists():
                out = self.run_dir / f"rater_{safe}_{k}.json"
                k += 1
            out.write_text(json.dumps(payload, indent=1))
            done = len(list(self.run_dir.glob("rater_*.json")))
        print(f"  [{datetime.now():%H:%M:%S}] received {len(responses):3d} "
              f"responses from session {safe} -> {out.name}  "
              f"({done} rater file(s) so far)", flush=True)
        self._send(200, "application/json; charset=utf-8",
                   json.dumps({"ok": True, "saved": out.name}).encode())

    # --- helpers ----------------------------------------------------------
    def _status(self) -> dict:
        files = sorted(self.run_dir.glob("rater_*.json"))
        rows = []
        for f in files:
            try:
                d = json.loads(f.read_text())
                rows.append({"file": f.name, "session": d.get("session"),
                             "n": len(d.get("responses", [])),
                             "received_at": d.get("received_at")})
            except Exception:                               # noqa: BLE001
                rows.append({"file": f.name, "error": "unreadable"})
        return {"raters": len(rows), "target": 8, "sessions": rows}

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _lan_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:                                       # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="Remote: the port forwards over VS Code Remote or "
               "`ssh -L 8777:127.0.0.1:8777 <host>`; open the printed URL "
               "on your laptop.")
    ap.add_argument("--run", default="outputs/run_20260918_human_study",
                    help="the study run directory (holds study.html)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; see the module docstring before "
                         "binding this to 0.0.0.0")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    page_path = run_dir / "study.html"
    if not page_path.exists():
        raise SystemExit(f"no study.html in {run_dir} — run "
                         f"scripts/human_study_export.py first")

    html = page_path.read_text()
    if "__SUBMIT_URL__" not in html:
        html = INJECT + html          # older export; inject the hook

    Handler.run_dir = run_dir
    Handler.page = html.encode()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  study     {page_path}")
    print(f"  responses {run_dir}/rater_<session>.json")
    print(f"\n  open      {url}")
    if args.host not in ("127.0.0.1", "localhost"):
        ip = _lan_ip()
        print(f"  reachable http://{ip}:{args.port}/  "
              f"— NO AUTHENTICATION, anyone on this network can submit"
              if ip else "  bound to a non-loopback address with no auth")
    print(f"  progress  {url}status")
    print(f"\n  Assign each rater a session: {url}?s=0 … ?s=23")
    print("  Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        done = len(list(run_dir.glob("rater_*.json")))
        print(f"\n  stopped — {done} rater file(s) in {run_dir}")
        print(f"  analyse:  .venv/bin/python scripts/human_study_analyze.py \\\n"
              f"              --key {run_dir}/answer_key.json \\\n"
              f"              --responses {run_dir}/rater_*.json \\\n"
              f"              --scorecard <a scorecard.json>\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
