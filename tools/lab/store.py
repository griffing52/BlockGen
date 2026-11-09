"""Where the lab's judgements live: labels, notes, captions, and pairwise comparisons.

**These rows are proposals, never data.** Nothing here is ever written back into
a corpus `.npz` or a manifest. A label made while browsing is a claim about a
build; turning claims into a dataset is the batch pipeline's job, and
`export()` is the seam between the two. Keeping that boundary means the lab can
be wrong -- a mis-click, a half-finished pass, a threshold someone was trying
out -- without any of it silently entering a published number.

**Every row is timestamped and nothing is destroyed by an edit.** Labels and
notes upsert on `build_id` and keep their original `created_at`, so a re-labelled
build still records when it was first seen. Comparisons are append-only: a
forced-choice trial's value is the sequence of trials, and rewriting one would
make the sample dishonest.

Threading. The lab's HTTP server is threaded, so this must be. The choice here is
**one connection with `check_same_thread=False` behind an `RLock`**, rather than
a connection per call. Reasons, in order: the workload is one human clicking, so
lock contention is never real; a single connection keeps `PRAGMA` state and the
schema migration in exactly one place; and separate write connections against the
same file are what produce `database is locked` under concurrent POSTs, which
would surface as a lost label rather than as an error the page can show. WAL is
on so an export reading while a label writes does not block.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tools.lab import DB_PATH
# Imported rather than reimplemented: the `<dataset>:<index>` grammar has exactly
# one definition, and a store that split ids differently from the catalog would
# file labels under a dataset that does not exist. Costs a numpy import; the
# server has already paid it.
from tools.lab.catalog import parse_build_id

LABELS = ("good", "bad", "unsure")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    build_id   TEXT PRIMARY KEY,
    dataset    TEXT NOT NULL,
    label      TEXT NOT NULL CHECK (label IN ('good', 'bad', 'unsure')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS labels_dataset ON labels (dataset);

CREATE TABLE IF NOT EXISTS notes (
    build_id   TEXT PRIMARY KEY,
    dataset    TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_dataset ON notes (dataset);

-- Append-only. `winner` NULL means the judge could not tell, which is a real
-- answer and must be distinguishable from "not yet judged" (absent row).
CREATE TABLE IF NOT EXISTS compares (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    a          TEXT NOT NULL,
    b          TEXT NOT NULL,
    winner     TEXT,
    ms         INTEGER NOT NULL,
    tag        TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS compares_created ON compares (created_at);

-- A caption is not a note. A note says what is wrong with a build and is read
-- by the person who wrote it; a caption says what the build IS, in the words a
-- user would type at the /gen prompt, and is destined for the SFT conditioning
-- set. Keeping them apart means a curation remark can never leak into training
-- text.
CREATE TABLE IF NOT EXISTS captions (
    build_id   TEXT PRIMARY KEY,
    dataset    TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS captions_dataset ON captions (dataset);
"""


def utcnow() -> str:
    """UTC, second resolution, sortable as a string. Not local time: the exports
    end up beside numbers from batch runs on other machines."""
    return (datetime.now(timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def _dataset_of(build_id: str) -> str:
    """Denormalized so `labels(dataset=...)` is an index hit rather than a scan
    over `build_id LIKE 'x:%'`, which cannot use the primary key."""
    return parse_build_id(build_id)[0]


class Store:
    """The lab's SQLite file. Safe to share across the server's threads."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL so a long export read never blocks a label write. FULL sync is
            # unnecessary here: losing the last click to a power cut is fine,
            # losing throughput on every click is not.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --- labels ------------------------------------------------------------
    def set_label(self, build_id: str, label: Optional[str]) -> None:
        """Set or clear one label. `None` deletes the row rather than storing a
        sentinel, so "unlabelled" has one representation and `counts()` cannot
        drift from `labels()`."""
        dataset = _dataset_of(build_id)
        if label is None:
            with self._lock:
                self._conn.execute("DELETE FROM labels WHERE build_id = ?", (build_id,))
                self._conn.commit()
            return
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS} or None, got {label!r}")
        now = utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO labels (build_id, dataset, label, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(build_id) DO UPDATE SET"
                "   label = excluded.label, updated_at = excluded.updated_at",
                (build_id, dataset, label, now, now))
            self._conn.commit()

    def get_label(self, build_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT label FROM labels WHERE build_id = ?", (build_id,)).fetchone()
        return row["label"] if row else None

    def labels(self, dataset: Optional[str] = None) -> Dict[str, str]:
        sql = "SELECT build_id, label FROM labels"
        args: tuple = ()
        if dataset:
            sql += " WHERE dataset = ?"
            args = (dataset,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {r["build_id"]: r["label"] for r in rows}

    def counts(self, dataset: Optional[str] = None) -> Dict[str, int]:
        """Always all three keys, zero-filled. A missing key renders as a blank
        in a template, which reads as "unknown" rather than as "none"."""
        sql = "SELECT label, COUNT(*) AS n FROM labels"
        args: tuple = ()
        if dataset:
            sql += " WHERE dataset = ?"
            args = (dataset,)
        sql += " GROUP BY label"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = {k: 0 for k in LABELS}
        for r in rows:
            out[r["label"]] = int(r["n"])
        return out

    # --- notes -------------------------------------------------------------
    def set_note(self, build_id: str, text: str) -> None:
        """Upsert a note; blank text deletes it, so clearing a textarea and
        saving does what the user obviously means."""
        dataset = _dataset_of(build_id)
        body = (text or "").strip()
        if not body:
            with self._lock:
                self._conn.execute("DELETE FROM notes WHERE build_id = ?", (build_id,))
                self._conn.commit()
            return
        now = utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO notes (build_id, dataset, text, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(build_id) DO UPDATE SET"
                "   text = excluded.text, updated_at = excluded.updated_at",
                (build_id, dataset, body, now, now))
            self._conn.commit()

    def get_note(self, build_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM notes WHERE build_id = ?", (build_id,)).fetchone()
        return row["text"] if row else None

    def notes(self, dataset: Optional[str] = None) -> Dict[str, str]:
        sql = "SELECT build_id, text FROM notes"
        args: tuple = ()
        if dataset:
            sql += " WHERE dataset = ?"
            args = (dataset,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {r["build_id"]: r["text"] for r in rows}

    # --- comparisons -------------------------------------------------------
    def add_compare(self, a: str, b: str, winner: Optional[str], ms: int,
                    tag: str = "") -> None:
        """Record one forced-choice trial.

        `winner` must be `a`, `b`, or `None` ("can't tell"). Validated rather
        than trusted: a trial whose winner is neither side is unusable, and the
        cheapest place to find that out is here rather than in the analysis six
        weeks later. `ms` is kept because response time separates an obvious call
        from a coin flip.
        """
        if winner is not None and winner not in (a, b):
            raise ValueError(f"winner {winner!r} is neither {a!r} nor {b!r}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO compares (a, b, winner, ms, tag, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (a, b, winner, int(ms), tag or "", utcnow()))
            self._conn.commit()

    def compares(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, a, b, winner, ms, tag, created_at FROM compares"
                " ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # --- captions -----------------------------------------------------------
    def set_caption(self, build_id: str, text: str) -> None:
        """Upsert a caption; blank text deletes it (same contract as notes)."""
        dataset = _dataset_of(build_id)
        body = " ".join((text or "").split())
        if not body:
            with self._lock:
                self._conn.execute("DELETE FROM captions WHERE build_id = ?", (build_id,))
                self._conn.commit()
            return
        now = utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO captions (build_id, dataset, text, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(build_id) DO UPDATE SET"
                "   text = excluded.text, updated_at = excluded.updated_at",
                (build_id, dataset, body, now, now))
            self._conn.commit()

    def get_caption(self, build_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM captions WHERE build_id = ?", (build_id,)).fetchone()
        return row["text"] if row else None

    def captions(self, dataset: Optional[str] = None) -> Dict[str, str]:
        sql = "SELECT build_id, text FROM captions"
        args: tuple = ()
        if dataset:
            sql += " WHERE dataset = ?"
            args = (dataset,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {r["build_id"]: r["text"] for r in rows}

    # --- history the queue needs -------------------------------------------
    def compared_pairs(self, tag: Optional[str] = None) -> Dict[str, int]:
        """How often each build has appeared in a trial (optionally under one tag).

        The phone queue uses this to spread trials over builds instead of
        re-sampling the same few, and to avoid re-asking an identical pair. A
        build's appearance count is what a Bradley-Terry fit later cares about,
        so balancing it at collection time is cheaper than reweighting after.
        """
        sql = "SELECT a, b FROM compares"
        args: tuple = ()
        if tag:
            sql += " WHERE tag = ?"
            args = (tag,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out: Dict[str, int] = {}
        for r in rows:
            out[r["a"]] = out.get(r["a"], 0) + 1
            out[r["b"]] = out.get(r["b"], 0) + 1
        return out

    def compared_pair_keys(self, tag: Optional[str] = None) -> set:
        """Unordered `(a, b)` keys already judged, so a pair is not asked twice.
        A second judgement of the same pair is a reliability measurement and
        worth having on purpose, but never by accident."""
        sql = "SELECT a, b FROM compares"
        args: tuple = ()
        if tag:
            sql += " WHERE tag = ?"
            args = (tag,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {(r["a"], r["b"]) if r["a"] <= r["b"] else (r["b"], r["a"]) for r in rows}

    def activity(self, since: Optional[str] = None) -> Dict[str, int]:
        """Row counts per table, optionally only rows touched at/after `since`
        (an ISO-8601 UTC string like `utcnow()` makes). This is the phone page's
        "today: 37" counter -- motivation, not analysis, so `updated_at` is the
        right clock for the upsert tables."""
        out: Dict[str, int] = {}
        for table, col in (("labels", "updated_at"), ("notes", "updated_at"),
                           ("captions", "updated_at"), ("compares", "created_at")):
            sql = f"SELECT COUNT(*) AS n FROM {table}"
            args: tuple = ()
            if since:
                sql += f" WHERE {col} >= ?"
                args = (since,)
            with self._lock:
                out[table] = int(self._conn.execute(sql, args).fetchone()["n"])
        return out

    # --- export ------------------------------------------------------------
    def export(self, what: str) -> List[dict]:
        """Full rows, timestamps included -- this is the handoff to the batch
        pipeline, and a decision without a date is not auditable."""
        if what == "labels":
            sql = ("SELECT build_id, dataset, label, created_at, updated_at"
                   " FROM labels ORDER BY dataset, build_id")
        elif what == "notes":
            sql = ("SELECT build_id, dataset, text, created_at, updated_at"
                   " FROM notes ORDER BY dataset, build_id")
        elif what == "captions":
            sql = ("SELECT build_id, dataset, text, created_at, updated_at"
                   " FROM captions ORDER BY dataset, build_id")
        elif what == "compares":
            sql = ("SELECT id, a, b, winner, ms, tag, created_at"
                   " FROM compares ORDER BY id")
        else:
            raise ValueError(f"what must be labels|notes|captions|compares, got {what!r}")
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
