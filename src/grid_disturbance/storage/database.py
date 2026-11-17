"""SQLite database connection and schema initialization for disturbance analyses."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    disturbance_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    dominant_frequency REAL,
    damping_ratio REAL,
    is_sustained INTEGER,
    candidate_source TEXT,
    summary TEXT,
    report_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_event_id ON analyses(event_id);
CREATE INDEX IF NOT EXISTS idx_analyses_disturbance_type ON analyses(disturbance_type);
"""


def get_db_connection(db_path: str | Path = "data/storage.db") -> sqlite3.Connection:
    """Connect to SQLite database and initialize schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn
