"""
KnowledgeGraph — SQLite temporal entity-relationship triples.

Stores (subject, predicate, object) triples with temporal validity windows.
Supports historical snapshot queries (as_of) and entity timelines.

Design principles:
- SQLite WAL mode: concurrent readers + one writer
- Thread-safe: all write operations are locked
- No external service dependencies, pure Python + sqlite3
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memxcore.kg")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT,
    ended TEXT,
    source TEXT DEFAULT 'manual',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_object ON triples(object COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate COLLATE NOCASE);
"""


class KnowledgeGraph:
    """
    Temporal entity-relationship graph.

    Main interface:
        add_triple(subj, pred, obj, ...)  — add a new triple
        invalidate(subj, pred, obj, ...)  — mark a triple as ended
        query_entity(entity, as_of?)      — query all relationships for an entity
        timeline(entity)                   — chronologically sorted entity event timeline
        search(query)                      — fuzzy search subject/object/predicate
    """

    def __init__(self, storage_dir: str) -> None:
        self._db_path = os.path.join(storage_dir, "knowledge.db")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if result and result[0].upper() != "WAL":
                logger.warning("SQLite WAL mode not available, using %s", result[0])
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write API ─────────────────────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        valid_from: Optional[str] = None,
        source: str = "manual",
    ) -> int:
        """
        Add a new triple. Returns the id of the new record.
        Duplicate (subject, predicate, object) triples are not deduplicated — the same relationship may have multiple time periods.
        """
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO triples (subject, predicate, object, valid_from, source, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (subject.strip(), predicate.strip(), object_.strip(),
                     valid_from, source, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def invalidate(
        self,
        subject: str,
        predicate: str,
        object_: str,
        ended: Optional[str] = None,
    ) -> int:
        """
        Mark a triple as ended. Returns the number of affected rows.
        Only updates records where ended IS NULL (to avoid duplicate invalidation).
        """
        if ended is None:
            ended = datetime.utcnow().strftime("%Y-%m-%d")
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE triples SET ended = ? "
                    "WHERE subject = ? COLLATE NOCASE "
                    "AND predicate = ? COLLATE NOCASE "
                    "AND object = ? COLLATE NOCASE "
                    "AND ended IS NULL",
                    (ended, subject.strip(), predicate.strip(), object_.strip()),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    # ── Read API ──────────────────────────────────────────────────────────────

    def query_entity(
        self,
        entity: str,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query all relationships for an entity (as subject or object).
        as_of: ISO date; only returns triples valid at that point in time.
        """
        conn = self._connect()
        try:
            if as_of:
                rows = conn.execute(
                    "SELECT * FROM triples "
                    "WHERE (subject = ? COLLATE NOCASE OR object = ? COLLATE NOCASE) "
                    "AND (valid_from IS NULL OR valid_from <= ?) "
                    "AND (ended IS NULL OR ended > ?) "
                    "ORDER BY valid_from DESC",
                    (entity, entity, as_of, as_of),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM triples "
                    "WHERE subject = ? COLLATE NOCASE OR object = ? COLLATE NOCASE "
                    "ORDER BY valid_from DESC NULLS LAST",
                    (entity, entity),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def timeline(self, entity: str) -> List[Dict[str, Any]]:
        """Chronologically sorted entity event timeline (all triples, including ended ones)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM triples "
                "WHERE subject = ? COLLATE NOCASE OR object = ? COLLATE NOCASE "
                "ORDER BY COALESCE(valid_from, created_at) ASC",
                (entity, entity),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fuzzy search across subject / predicate / object."""
        pattern = f"%{query}%"
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM triples "
                "WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (pattern, pattern, pattern, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        finally:
            conn.close()

    @staticmethod
    def format_triple(t: Dict[str, Any]) -> str:
        """Format a triple as a human-readable string."""
        validity = ""
        if t.get("valid_from"):
            validity = f" (from {t['valid_from']}"
            if t.get("ended"):
                validity += f" to {t['ended']}"
            validity += ")"
        elif t.get("ended"):
            validity = f" (ended {t['ended']})"
        return f"{t['subject']} → {t['predicate']} → {t['object']}{validity}"
