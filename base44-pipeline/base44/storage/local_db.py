"""Local database — primary interface for raw case files, facts, witness
statements, and legal notes (Module 1).

Uses the Python standard-library ``sqlite3`` so it works with zero extra
dependencies. Stores documents and an append-only revision log.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..schemas import RawDocument

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    case_id  TEXT,
    kind     TEXT,
    text     TEXT NOT NULL,
    source   TEXT,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS revisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT,
    version      INTEGER,
    author       TEXT,
    summary      TEXT,
    content_hash TEXT,
    timestamp    TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_case ON documents(case_id);
"""


class LocalDB:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(self.settings.local_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert_document(self, doc: RawDocument, case_id: str = "", kind: str = "") -> None:
        self._conn.execute(
            "INSERT INTO documents(doc_id, case_id, kind, text, source, metadata) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET text=excluded.text, "
            "case_id=excluded.case_id, kind=excluded.kind, source=excluded.source, "
            "metadata=excluded.metadata",
            (doc.doc_id, case_id, kind, doc.text, doc.source, json.dumps(doc.metadata)),
        )
        self._conn.commit()

    def get_documents(self, case_id: Optional[str] = None) -> List[RawDocument]:
        if case_id:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE case_id=?", (case_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return [
            RawDocument(
                doc_id=r["doc_id"],
                text=r["text"],
                source=r["source"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    def log_revision(self, doc_id: str, version: int, author: str, summary: str,
                     content_hash: str, timestamp: str) -> None:
        self._conn.execute(
            "INSERT INTO revisions(doc_id, version, author, summary, content_hash, timestamp) "
            "VALUES(?,?,?,?,?,?)",
            (doc_id, version, author, summary, content_hash, timestamp),
        )
        self._conn.commit()

    def revisions(self, doc_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM revisions WHERE doc_id=? ORDER BY version", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
