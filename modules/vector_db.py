import json
import logging
import re
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MindGateVectorDB")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "का", "की", "के", "को", "में", "से", "है", "हैं",
    "और", "क्या", "काय", "आहे", "आणि",
}


class VectorStoreManager:
    """Workspace-local SQLite FTS5 store ranked with its built-in BM25."""

    def __init__(self, storage_dir: str = "storage/.mindgate"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.storage_dir / "mindgate.sqlite3"
        self._lock = RLock()
        self._connection = None
        self._connect()
        logger.info("Local SQLite FTS5 store initialized at %s", self.database_path)

    def _connect(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if self._connection is not None:
            try:
                self._connection.execute("SELECT 1")
                if self.database_path.exists():
                    return
            except sqlite3.Error:
                pass
            try:
                self._connection.close()
            except sqlite3.Error:
                pass
            self._connection = None

        self._connection = sqlite3.connect(
            str(self.database_path), check_same_thread=False, timeout=30
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')
                """
            )

    def _ensure_database_present(self) -> None:
        if self._connection is None or not self.database_path.exists():
            self._connect()

    @staticmethod
    def _prepare_rows(chunks: List[Dict[str, Any]]) -> List[tuple[str, str, str, str]]:
        rows = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
            text = str(chunk.get("text") or "").strip()
            file_name = metadata.get("file_name")
            if not chunk_id or not text or not file_name:
                raise ValueError("Each indexed chunk requires an ID, text, and file_name.")
            metadata["chunk_id"] = str(chunk_id)
            metadata["file_name"] = str(file_name)
            rows.append((
                str(chunk_id),
                str(file_name),
                text,
                json.dumps(metadata, ensure_ascii=False),
            ))
        return rows

    def _write_rows(self, rows: List[tuple[str, str, str, str]]) -> None:
        for chunk_id, file_name, text, metadata in rows:
            self._connection.execute(
                """
                INSERT INTO chunks(chunk_id, file_name, text, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    text = excluded.text,
                    metadata = excluded.metadata
                """,
                (chunk_id, file_name, text, metadata),
            )
            self._connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
            )
            self._connection.execute(
                "INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)",
                (chunk_id, text),
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert or replace document chunks and their searchable text atomically."""
        self._ensure_database_present()
        rows = self._prepare_rows(chunks)
        if not rows:
            return
        with self._lock, self._connection:
            self._write_rows(rows)
        logger.info("Indexed %d chunks in local SQLite FTS5.", len(rows))

    def replace_document(self, file_name: str, chunks: List[Dict[str, Any]]) -> None:
        """Atomically replace every indexed chunk for one source document."""
        self._ensure_database_present()
        rows = self._prepare_rows(chunks)
        if any(row[1] != file_name for row in rows):
            raise ValueError("All replacement chunks must belong to the requested file.")
        with self._lock, self._connection:
            self._connection.execute(
                """
                DELETE FROM chunks_fts
                WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_name = ?)
                """,
                (file_name,),
            )
            self._connection.execute(
                "DELETE FROM chunks WHERE file_name = ?", (file_name,)
            )
            self._write_rows(rows)
        logger.info("Replaced indexed content for %s (%d chunks).", file_name, len(rows))

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.upsert_chunks(chunks)

    def query_similar_chunks(
        self,
        query: str,
        top_k: int = 2,
        file_name_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return lexical BM25 matches, optionally confined to one document."""
        self._ensure_database_present()
        terms = [
            term for term in re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
            if term not in _STOP_WORDS
        ]
        if not terms or top_k <= 0:
            return []
        expression = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in dict.fromkeys(terms)
        )
        sql = """
            SELECT chunks.chunk_id, chunks.text, chunks.metadata
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
        """
        parameters: List[Any] = [expression]
        if file_name_filter is not None:
            sql += " AND chunks.file_name = ?"
            parameters.append(file_name_filter)
        sql += " ORDER BY bm25(chunks_fts), chunks.chunk_id LIMIT ?"
        parameters.append(top_k)

        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def get_all_chunks_for_file(self, file_name: str, limit: int = 8) -> List[Dict[str, Any]]:
        self._ensure_database_present()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT chunk_id, text, metadata FROM chunks
                WHERE file_name = ?
                ORDER BY json_extract(metadata, '$.page_number'),
                         json_extract(metadata, '$.line_start'), chunk_id
                LIMIT ?
                """,
                (file_name, max(0, limit)),
            ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def list_documents(self) -> List[str]:
        self._ensure_database_present()
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT file_name FROM chunks ORDER BY file_name COLLATE NOCASE"
            ).fetchall()
        return [row["file_name"] for row in rows]

    def remove_document(self, file_name: str) -> None:
        self._ensure_database_present()
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE file_name = ?)",
                (file_name,),
            )
            self._connection.execute(
                "DELETE FROM chunks WHERE file_name = ?", (file_name,)
            )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
