"""Version Control Protocol (Module 5).

Maintains an auditable log of document revisions with timestamps, author
metadata, and a content hash for integrity. Can persist to the local DB when one
is supplied; otherwise keeps an in-memory log.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..schemas import RevisionRecord


class VersionControl:
    def __init__(self, db=None) -> None:
        self._db = db  # optional base44.storage.local_db.LocalDB
        self._log: Dict[str, List[RevisionRecord]] = {}

    def commit(self, doc_id: str, content: str, author: str, summary: str) -> RevisionRecord:
        history = self._log.setdefault(doc_id, [])
        version = len(history) + 1
        record = RevisionRecord(
            version=version,
            timestamp=datetime.now(timezone.utc),
            author=author,
            summary=summary,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        history.append(record)
        if self._db is not None:
            self._db.log_revision(
                doc_id=doc_id, version=version, author=author, summary=summary,
                content_hash=record.content_hash, timestamp=record.timestamp.isoformat(),
            )
        return record

    def history(self, doc_id: str) -> List[RevisionRecord]:
        return list(self._log.get(doc_id, []))

    def latest(self, doc_id: str) -> Optional[RevisionRecord]:
        h = self._log.get(doc_id)
        return h[-1] if h else None
