"""
Evaluation log storage.

  DATABASE_URL set (Supabase Postgres)     -> PostgreSQL via psycopg 3 (persistent)
  otherwise                                -> SQLite file DB_PATH (local development)

On Render's free web service the filesystem is ephemeral, so SQLite data would be lost on every
restart, redeploy or spin-down: use DATABASE_URL there.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH = os.getenv("DB_PATH", str(Path(__file__).resolve().parent / "karirfit.db"))


class Storage:
    def __init__(self):
        self.pg = DATABASE_URL.startswith(("postgres://", "postgresql://"))
        if self.pg:
            import psycopg  # psycopg[binary]
            self._psycopg = psycopg
            self.url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        with self._conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS log(id SERIAL PRIMARY KEY, ts DOUBLE PRECISION, kind TEXT, session TEXT, payload TEXT)"
                if self.pg else
                "CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, session TEXT, payload TEXT)")

    @property
    def kind(self) -> str:
        return "postgres" if self.pg else "sqlite"

    def _conn(self):
        # prepare_threshold=None: works with Supabase's transaction pooler (port 6543)
        return self._psycopg.connect(self.url, autocommit=True, prepare_threshold=None) if self.pg else sqlite3.connect(DB_PATH)

    def log(self, kind: str, session: str, payload: dict):
        ph = "%s" if self.pg else "?"
        with self._conn() as con:
            con.execute(f"INSERT INTO log(ts, kind, session, payload) VALUES ({ph},{ph},{ph},{ph})",
                        (time.time(), kind, session, json.dumps(payload, ensure_ascii=False)))

    def export(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute("SELECT ts, kind, session, payload FROM log ORDER BY id").fetchall()
        return [{"ts": t, "kind": k, "session": s, **json.loads(p)} for t, k, s, p in rows]
