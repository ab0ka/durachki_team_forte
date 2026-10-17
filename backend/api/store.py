"""SQLite (WAL) хранилище задач. Один коннект + RLock: нагрузка — десятки задач, не тысячи."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

ACTIVE = ("queued", "running")
FINAL = ("done", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,          -- queued | running | done | failed
    stage        TEXT,
    progress     REAL NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,          -- момент загрузки
    queued_at    REAL NOT NULL,          -- база отсчёта бюджета (сбрасывается при рестарте)
    started_at   REAL,
    finished_at  REAL,
    error        TEXT,
    result_path  TEXT,
    video_path   TEXT,
    filename     TEXT,
    size_bytes   INTEGER,
    sha256       TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    deadline_s   REAL,
    client       TEXT,
    warnings     TEXT,                   -- JSON list
    video_meta   TEXT                    -- JSON: длительность/fps/размер от декодера
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at);
"""

_COLUMNS = ("id", "status", "stage", "progress", "created_at", "queued_at", "started_at",
            "finished_at", "error", "result_path", "video_path", "filename", "size_bytes",
            "sha256", "attempts", "deadline_s", "client", "warnings", "video_meta")


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False,
                                     isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)

    # ---- базовые операции ----
    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(args))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _row(r: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if r is None:
            return None
        d = dict(r)
        for k, empty in (("warnings", "[]"), ("video_meta", "{}")):
            try:
                d[k] = json.loads(d.get(k) or empty)
            except ValueError:
                d[k] = json.loads(empty)
        return d

    def create(self, job_id: str, *, video_path: str, filename: str, size_bytes: int,
               sha256: str, client: str = "", video_meta: Optional[dict[str, Any]] = None,
               now: Optional[float] = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        self._exec(
            "INSERT INTO jobs (id, status, stage, progress, created_at, queued_at, video_path, "
            "filename, size_bytes, sha256, client, warnings, video_meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "queued", "queued", 0.0, now, now, video_path, filename, size_bytes, sha256,
             client, "[]", json.dumps(video_meta or {}, ensure_ascii=False)))
        return self.get(job_id) or {}

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._row(self._exec("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        bad = set(fields) - set(_COLUMNS)
        if bad:
            raise KeyError(f"unknown columns: {bad}")
        if "warnings" in fields and not isinstance(fields["warnings"], str):
            fields["warnings"] = json.dumps(list(fields["warnings"]), ensure_ascii=False)
        if "video_meta" in fields and not isinstance(fields["video_meta"], str):
            fields["video_meta"] = json.dumps(fields["video_meta"], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def mark_running(self, job_id: str, now: Optional[float] = None) -> bool:
        """queued -> running (атомарно). False, если задачу уже финализировали."""
        now = time.time() if now is None else now
        cur = self._exec(
            "UPDATE jobs SET status='running', started_at=?, attempts=attempts+1, stage=? "
            "WHERE id=? AND status='queued'", (now, "starting", job_id))
        return cur.rowcount == 1

    def finalize(self, job_id: str, status: str, *, result_path: Optional[str] = None,
                 error: Optional[str] = None, warnings: Optional[list[str]] = None,
                 now: Optional[float] = None) -> bool:
        """Атомарный переход active -> done/failed. Выигрывает первый (воркер или watchdog)."""
        assert status in FINAL
        now = time.time() if now is None else now
        cur = self._exec(
            "UPDATE jobs SET status=?, finished_at=?, result_path=?, error=?, warnings=?, "
            "progress=CASE WHEN ?='done' THEN 1.0 ELSE progress END, stage=? "
            "WHERE id=? AND status IN ('queued','running')",
            (status, now, result_path, error, json.dumps(warnings or [], ensure_ascii=False),
             status, status, job_id))
        return cur.rowcount == 1

    def requeue(self, job_id: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._exec("UPDATE jobs SET status='queued', stage='requeued', progress=0, queued_at=?, "
                   "started_at=NULL WHERE id=?", (now, job_id))

    def active(self) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM jobs WHERE status IN ('queued','running') "
                          "ORDER BY queued_at, created_at").fetchall()
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                          (int(limit),)).fetchall()
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def finished_before(self, ts: float) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM jobs WHERE status IN ('done','failed') AND "
                          "COALESCE(finished_at, created_at) < ?", (ts,)).fetchall()
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def find_done_by_sha(self, sha256: str) -> Optional[dict[str, Any]]:
        r = self._exec("SELECT * FROM jobs WHERE sha256=? AND status='done' "
                       "ORDER BY finished_at DESC LIMIT 1", (sha256,)).fetchone()
        return self._row(r)

    def delete(self, job_id: str) -> None:
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))

    def count_active(self) -> int:
        r = self._exec("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()
        return int(r[0]) if r else 0

    def ids(self) -> set[str]:
        return {r[0] for r in self._exec("SELECT id FROM jobs").fetchall()}
