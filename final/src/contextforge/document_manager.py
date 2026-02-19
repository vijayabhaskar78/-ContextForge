# /contextforge_cli/document_manager.py
"""
Handles document uploading, storage, and tracking of processing status.
Also persists chunk-neighbor metadata for fast context expansion.
"""
import json
import sqlite3
import threading
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rich.panel import Panel
from rich.table import Table, box

# Local Imports
from .config import (
    BM25_CACHE_FILE,
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    INGEST_MAX_WORKERS,
    PDF_SUPPORT,
    SEMANTIC_CHUNKING_AVAILABLE,
    STORAGE_DIR,
    USE_SEMANTIC_CHUNKING,
    console,
)
from .db_migrations import SqliteMigration, apply_sqlite_migrations
from .observability import get_logger
from .storage_provider import FileStorageProvider, LocalFileStorageProvider
from .tokenization import tokenize_for_matching

LOGICAL_PAGE_HEURISTIC_VERSION = 4
logger = get_logger(__name__)


class ChunkRegistry:
    """Persistent + in-memory chunk registry keyed by unique chunk IDs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else (CACHE_DIR / "chunk_registry.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache_maxsize = 5000
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._source_to_ids: dict[str, set[str]] = {}
        self._conn: sqlite3.Connection | None = self._connect()
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        with self._lock:
            if self._conn is None:
                raise RuntimeError("chunk registry connection is closed")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self):
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.commit()
            except sqlite3.Error:
                pass
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_schema(self):
        def _ensure_legacy_columns(conn: sqlite3.Connection):
            chunk_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "logical_page" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN logical_page INTEGER")
            if "logical_page_offset" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN logical_page_offset INTEGER")
            if "logical_page_extracted" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN logical_page_extracted INTEGER NOT NULL DEFAULT 0")
            if "logical_page_version" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN logical_page_version INTEGER NOT NULL DEFAULT 0")

        migrations = [
            SqliteMigration(
                version=1,
                name="create_chunks_table",
                statements=(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_mtime INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        logical_page INTEGER,
                        logical_page_offset INTEGER,
                        logical_page_extracted INTEGER NOT NULL DEFAULT 0,
                        logical_page_version INTEGER NOT NULL DEFAULT 0,
                        content TEXT NOT NULL,
                        prev_chunk_id TEXT,
                        next_chunk_id TEXT
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_source_index ON chunks(source, chunk_index)",
                ),
            ),
            SqliteMigration(
                version=2,
                name="ensure_logical_page_columns",
                runner=_ensure_legacy_columns,
            ),
        ]
        with self._connection() as conn:
            apply_sqlite_migrations(
                conn,
                component="chunk_registry",
                migrations=migrations,
            )

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _source_mtime(source: str) -> int:
        try:
            return int(Path(source).stat().st_mtime)
        except OSError:
            return 0

    @staticmethod
    def _safe_optional_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sanitize_logical_page(value: Any) -> int | None:
        logical_page = ChunkRegistry._safe_optional_int(value)
        if logical_page is None:
            return None
        if 0 < logical_page < 2000:
            return logical_page
        return None

    def _evict_source_cache(self, source: str):
        with self._lock:
            existing = self._source_to_ids.pop(source, set())
            for chunk_id in existing:
                self._cache.pop(chunk_id, None)

    def _cache_rows(self, rows: list[dict[str, Any]]):
        with self._lock:
            for row in rows:
                chunk_id = str(row["chunk_id"])
                source = str(row["source"])
                prior = self._cache.get(chunk_id)
                if prior is not None:
                    prior_source = str(prior.get("source", ""))
                    if prior_source and prior_source != source:
                        prior_ids = self._source_to_ids.get(prior_source)
                        if prior_ids is not None:
                            prior_ids.discard(chunk_id)
                            if not prior_ids:
                                self._source_to_ids.pop(prior_source, None)
                    self._cache.move_to_end(chunk_id)
                self._cache[chunk_id] = row
                self._source_to_ids.setdefault(source, set()).add(chunk_id)
                while len(self._cache) > self._cache_maxsize:
                    evicted_chunk_id, evicted_row = self._cache.popitem(last=False)
                    evicted_source = str(evicted_row.get("source", ""))
                    evicted_ids = self._source_to_ids.get(evicted_source)
                    if evicted_ids is None:
                        continue
                    evicted_ids.discard(str(evicted_chunk_id))
                    if not evicted_ids:
                        self._source_to_ids.pop(evicted_source, None)

    def register_chunks(self, split_docs: list[Document]):
        """Assigns chunk IDs/neighbors and persists all chunks by source."""
        if not split_docs:
            return

        grouped: dict[str, list[Document]] = defaultdict(list)
        for doc in split_docs:
            source = str(doc.metadata.get("source", "")).strip()
            if not source:
                continue
            grouped[source].append(doc)

        for source, docs in grouped.items():
            source_mtime = self._source_mtime(source)
            chunk_ids = [f"{source}::{source_mtime}::{idx}" for idx in range(len(docs))]

            rows_to_insert = []
            cache_rows = []

            for idx, doc in enumerate(docs):
                metadata = dict(doc.metadata or {})
                chunk_id = chunk_ids[idx]
                prev_chunk_id = chunk_ids[idx - 1] if idx > 0 else None
                next_chunk_id = chunk_ids[idx + 1] if idx + 1 < len(chunk_ids) else None
                sanitized_logical_page = self._sanitize_logical_page(metadata.get("logical_page"))
                sanitized_offset = self._safe_optional_int(metadata.get("logical_page_offset"))

                metadata["chunk_id"] = chunk_id
                metadata["chunk_index"] = idx
                metadata["source_mtime"] = source_mtime
                metadata["prev_chunk_id"] = prev_chunk_id
                metadata["next_chunk_id"] = next_chunk_id
                metadata["logical_page"] = sanitized_logical_page
                metadata["logical_page_offset"] = sanitized_offset
                doc.metadata = metadata

                row = {
                    "chunk_id": chunk_id,
                    "source": source,
                    "source_mtime": source_mtime,
                    "chunk_index": idx,
                    "page": self._safe_int(metadata.get("page"), -1),
                    "logical_page": sanitized_logical_page,
                    "logical_page_offset": sanitized_offset,
                    "logical_page_extracted": 1,
                    "logical_page_version": LOGICAL_PAGE_HEURISTIC_VERSION,
                    "content": str(getattr(doc, "page_content", "") or ""),
                    "prev_chunk_id": prev_chunk_id,
                    "next_chunk_id": next_chunk_id,
                }
                cache_rows.append(row)
                rows_to_insert.append(
                    (
                        row["chunk_id"],
                        row["source"],
                        row["source_mtime"],
                        row["chunk_index"],
                        row["page"],
                        row["logical_page"],
                        row["logical_page_offset"],
                        row["logical_page_extracted"],
                        row["logical_page_version"],
                        row["content"],
                        row["prev_chunk_id"],
                        row["next_chunk_id"],
                    )
                )

            with self._connection() as conn:
                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, source, source_mtime, chunk_index, page, logical_page, logical_page_offset, logical_page_extracted, logical_page_version, content, prev_chunk_id, next_chunk_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )

            self._evict_source_cache(source)
            self._cache_rows(cache_rows)

    def get_chunk(self, chunk_id: str | None) -> dict[str, Any] | None:
        if not chunk_id:
            return None
        chunk_id = str(chunk_id)
        with self._lock:
            cached = self._cache.get(chunk_id)
            if cached is not None:
                self._cache.move_to_end(chunk_id)
        if cached is not None:
            return cached

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT chunk_id, source, source_mtime, chunk_index, page, logical_page, logical_page_offset, content, prev_chunk_id, next_chunk_id
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if not row:
            return None

        data = dict(row)
        self._cache_rows([data])
        return data

    def get_chunks_batch(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Returns chunk rows keyed by chunk_id using a single SQL IN query for cache misses.
        Missing ids are omitted from the result.
        """
        ordered_ids: list[str] = []
        seen_ids = set()
        for raw_id in chunk_ids or []:
            chunk_id = str(raw_id or "").strip()
            if not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            ordered_ids.append(chunk_id)
        if not ordered_ids:
            return {}

        rows_by_id: dict[str, dict[str, Any]] = {}
        missing_ids: list[str] = []
        with self._lock:
            for chunk_id in ordered_ids:
                cached = self._cache.get(chunk_id)
                if cached is not None:
                    self._cache.move_to_end(chunk_id)
                    rows_by_id[chunk_id] = cached
                else:
                    missing_ids.append(chunk_id)

        if missing_ids:
            placeholders = ",".join("?" for _ in missing_ids)
            with self._connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT chunk_id, source, source_mtime, chunk_index, page, logical_page, logical_page_offset, content, prev_chunk_id, next_chunk_id
                    FROM chunks
                    WHERE chunk_id IN ({placeholders})
                    """,
                    missing_ids,
                ).fetchall()
            fetched_rows = [dict(row) for row in rows]
            if fetched_rows:
                self._cache_rows(fetched_rows)
                for row in fetched_rows:
                    rows_by_id[str(row["chunk_id"])] = row

        return rows_by_id

    def source_has_chunks(self, source: str) -> bool:
        source = str(source or "")
        if not source:
            return False
        with self._lock:
            if self._source_to_ids.get(source):
                return True
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE source = ? LIMIT 1",
                (source,),
            ).fetchone()
        return bool(row)

    def source_has_logical_pages(self, source: str) -> bool:
        source = str(source or "")
        if not source:
            return False
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM chunks
                WHERE source = ? AND logical_page_extracted = 1 AND logical_page_version >= ?
                LIMIT 1
                """,
                (source, LOGICAL_PAGE_HEURISTIC_VERSION),
            ).fetchone()
        return bool(row)

    def has_chunk(self, chunk_id: str | None) -> bool:
        return self.get_chunk(chunk_id) is not None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip().lower()

    def resolve_chunk_id(self, source: str, page: int, content: str) -> str | None:
        """
        Resolves a chunk_id for legacy documents that only contain source/page metadata.
        Picks the best textual overlap candidate from registry rows for that source/page.
        """
        source = str(source or "").strip()
        if not source:
            return None
        safe_page = self._safe_int(page, -1)
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, source, source_mtime, chunk_index, page, logical_page, logical_page_offset, content, prev_chunk_id, next_chunk_id
                FROM chunks
                WHERE source = ? AND page = ?
                ORDER BY chunk_index ASC
                """,
                (source, safe_page),
            ).fetchall()
        if not rows:
            return None
        if len(rows) == 1:
            data = dict(rows[0])
            self._cache_rows([data])
            return str(data["chunk_id"])

        target_tokens = set(tokenize_for_matching(self._normalize_text(content), min_len=1))
        best_score = -1.0
        best_row = None
        for row in rows:
            row_dict = dict(row)
            row_tokens = set(tokenize_for_matching(self._normalize_text(row_dict.get("content", "")), min_len=1))
            if not target_tokens and not row_tokens:
                score = 0.0
            else:
                overlap = len(target_tokens.intersection(row_tokens))
                score = overlap / max(1, len(target_tokens))
                normalized_target = self._normalize_text(content)[:260]
                normalized_row = self._normalize_text(row_dict.get("content", ""))[:260]
                if normalized_target and normalized_row:
                    if normalized_target in normalized_row or normalized_row in normalized_target:
                        score += 0.5
            if score > best_score:
                best_score = score
                best_row = row_dict

        if not best_row:
            return None
        self._cache_rows([best_row])
        return str(best_row["chunk_id"])

    def get_neighbor_chunk_ids(self, chunk_id: str | None) -> list[str]:
        row = self.get_chunk(chunk_id)
        if not row:
            return []
        ids = []
        if row.get("prev_chunk_id"):
            ids.append(str(row["prev_chunk_id"]))
        ids.append(str(row["chunk_id"]))
        if row.get("next_chunk_id"):
            ids.append(str(row["next_chunk_id"]))
        return ids

    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> Document:
        metadata = {
            "source": row.get("source"),
            "page": row.get("page", -1),
            "logical_page": row.get("logical_page"),
            "logical_page_offset": row.get("logical_page_offset"),
            "chunk_id": row.get("chunk_id"),
            "chunk_index": row.get("chunk_index", -1),
            "source_mtime": row.get("source_mtime", 0),
            "prev_chunk_id": row.get("prev_chunk_id"),
            "next_chunk_id": row.get("next_chunk_id"),
        }
        return Document(page_content=str(row.get("content", "")), metadata=metadata)

    def get_documents(self, chunk_ids: list[str]) -> list[Document]:
        docs = []
        for chunk_id in chunk_ids:
            row = self.get_chunk(chunk_id)
            if row:
                docs.append(self._row_to_document(row))
        return docs

    def get_documents_batch(self, chunk_ids: list[str]) -> list[Document]:
        """Returns documents for chunk_ids in input order using batched row lookup."""
        rows_by_id = self.get_chunks_batch(chunk_ids)
        docs: list[Document] = []
        for raw_id in chunk_ids or []:
            chunk_id = str(raw_id or "").strip()
            if not chunk_id:
                continue
            row = rows_by_id.get(chunk_id)
            if row:
                docs.append(self._row_to_document(row))
        return docs

    def get_source_documents(self, source: str, limit: int | None = None) -> list[Document]:
        """
        Returns registry documents for a source in deterministic order.
        Primary ordering is page, then chunk_index.
        """
        source = str(source or "").strip()
        if not source:
            return []
        sql = (
            """
            SELECT chunk_id, source, source_mtime, chunk_index, page, logical_page, logical_page_offset, content, prev_chunk_id, next_chunk_id
            FROM chunks
            WHERE source = ?
            ORDER BY page ASC, chunk_index ASC
            """
        )
        params: list[Any] = [source]
        if limit is not None and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        row_dicts = [dict(row) for row in rows]
        if row_dicts:
            self._cache_rows(row_dicts)
        return [self._row_to_document(row) for row in row_dicts]


CHUNK_REGISTRY = ChunkRegistry()


class DocumentUploader:
    """Manages the lifecycle of documents for the RAG pipeline."""

    def __init__(
        self,
        storage_dir: Path = STORAGE_DIR,
        db_path: Path | None = None,
        storage_provider: FileStorageProvider | None = None,
    ):
        self.storage: FileStorageProvider = storage_provider or LocalFileStorageProvider(Path(storage_dir))
        self.storage.ensure_ready()
        self.storage_dir = Path(self.storage.root)
        self.docs_file = self.storage_dir / "uploaded_docs.json"
        self.db_path = Path(db_path) if db_path else (CACHE_DIR / "documents.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._ensure_schema()
        self._migrate_legacy_json_if_needed()
        self.documents = self._load_documents()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        with self._lock:
            if self._conn is None:
                raise RuntimeError("document metadata connection is closed")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self):
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.commit()
            except sqlite3.Error:
                pass
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_schema(self):
        migrations = [
            SqliteMigration(
                version=1,
                name="create_uploaded_documents_table",
                statements=(
                    """
                    CREATE TABLE IF NOT EXISTS uploaded_documents (
                        doc_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        stored_file TEXT NOT NULL UNIQUE,
                        file_type TEXT,
                        size INTEGER NOT NULL DEFAULT 0,
                        upload_date TEXT NOT NULL,
                        is_indexed INTEGER NOT NULL DEFAULT 0
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_uploaded_documents_indexed ON uploaded_documents(is_indexed)",
                    "CREATE INDEX IF NOT EXISTS idx_uploaded_documents_file ON uploaded_documents(stored_file)",
                ),
            ),
        ]
        with self._connection() as conn:
            apply_sqlite_migrations(
                conn,
                component="uploaded_documents",
                migrations=migrations,
            )

    def _row_to_doc_info(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "title": str(row["title"]),
            "stored_file": str(row["stored_file"]),
            "file_type": str(row["file_type"] or ""),
            "size": int(row["size"] or 0),
            "upload_date": str(row["upload_date"]),
            "is_indexed": bool(int(row["is_indexed"] or 0)),
        }

    def _upsert_document_row(self, doc_id: str, doc_info: dict[str, Any]):
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO uploaded_documents (doc_id, title, stored_file, file_type, size, upload_date, is_indexed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title = excluded.title,
                    stored_file = excluded.stored_file,
                    file_type = excluded.file_type,
                    size = excluded.size,
                    upload_date = excluded.upload_date,
                    is_indexed = excluded.is_indexed
                """,
                (
                    str(doc_id),
                    str(doc_info.get("title", "")),
                    str(doc_info.get("stored_file", "")),
                    str(doc_info.get("file_type", "")),
                    int(doc_info.get("size", 0) or 0),
                    str(doc_info.get("upload_date", datetime.now().isoformat())),
                    1 if bool(doc_info.get("is_indexed")) else 0,
                ),
            )

    def _migrate_legacy_json_if_needed(self):
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM uploaded_documents").fetchone()
            existing_count = int(row["cnt"]) if row else 0
        if existing_count > 0:
            return
        if not self.docs_file.exists():
            return
        try:
            with open(self.docs_file, "r", encoding="utf-8") as handle:
                legacy = json.load(handle) or {}
        except Exception:
            legacy = {}
        if not isinstance(legacy, dict):
            return
        for doc_id, doc_info in legacy.items():
            if not isinstance(doc_info, dict):
                continue
            doc_info.setdefault("is_indexed", False)
            self._upsert_document_row(str(doc_id), doc_info)

    def _load_documents(self) -> dict:
        """Loads document metadata from SQLite."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, title, stored_file, file_type, size, upload_date, is_indexed
                FROM uploaded_documents
                ORDER BY upload_date ASC, doc_id ASC
                """
            ).fetchall()
        return {str(row["doc_id"]): self._row_to_doc_info(row) for row in rows}

    def _save_documents(self, doc_id: str | None = None):
        """
        Persists metadata atomically in SQLite.
        doc_id scope keeps updates O(1) for common operations.
        """
        if doc_id is not None:
            info = self.documents.get(str(doc_id))
            if info is None:
                return
            self._upsert_document_row(str(doc_id), info)
            return
        for current_doc_id, doc_info in self.documents.items():
            self._upsert_document_row(str(current_doc_id), doc_info)

    def upload_document(self, file_path: str, title: str = None) -> bool:
        """Copies a document to storage and records its metadata."""
        source_path = Path(file_path)
        if not source_path.exists():
            console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
            return False
        if not source_path.is_file():
            console.print(f"[bold red]Error: Path is not a file: {file_path}[/bold red]")
            return False

        suffix = source_path.suffix.lower()
        if suffix == ".pdf" and not PDF_SUPPORT:
            console.print(
                "[bold red]Error: PDF support is not available. Install 'PyMuPDF' (fitz) and retry upload.[/bold red]"
            )
            logger.error("upload_blocked_missing_dependency", file_path=str(source_path), dependency="PyMuPDF")
            return False

        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_path.stem}"
        stored_path = self.storage.save_file(source_path, f"{doc_id}{source_path.suffix}")
        self.documents[doc_id] = {
            "title": title or source_path.name,
            "stored_file": str(stored_path),
            "file_type": suffix,
            "size": source_path.stat().st_size,
            "upload_date": datetime.now().isoformat(),
            "is_indexed": False,
        }
        self._save_documents(doc_id)
        logger.info(
            "document_uploaded",
            doc_id=doc_id,
            file_type=str(source_path.suffix.lower()),
            bytes=int(source_path.stat().st_size),
            stored_file=str(stored_path),
        )
        console.print(
            Panel(
                f"[green]OK Document uploaded: [bold]{doc_id}[/bold]\n"
                f"       Title: {self.documents[doc_id]['title']}",
                title="Upload Success",
                border_style="green",
            )
        )
        return True

    def list_documents(self):
        """Displays a table of all uploaded documents."""
        if not self.documents:
            console.print("[yellow]No documents uploaded yet.[/yellow]")
            return

        table = Table(title="Uploaded Documents", border_style="blue", header_style="bold", box=box.SQUARE)
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("Size (MB)", style="yellow")
        table.add_column("Status", style="white")

        for doc_id, doc_info in self.documents.items():
            size_mb = f"{doc_info['size'] / (1024 * 1024):.2f}"
            status = "[green]Processed[/green]" if doc_info.get("is_indexed") else "[red]Not Processed[/red]"
            table.add_row(doc_id, doc_info["title"], size_mb, status)
        console.print(table)

    def get_unindexed_paths(self) -> list:
        """Returns file paths of documents not yet processed."""
        return [doc["stored_file"] for doc in self.documents.values() if not doc.get("is_indexed")]

    def get_all_paths(self) -> list:
        """Returns all stored document file paths."""
        return [doc["stored_file"] for doc in self.documents.values()]

    def mark_as_indexed(self, stored_file_path: str):
        """Marks a document as processed in the metadata."""
        changed_doc_id = None
        for doc_id, doc_info in self.documents.items():
            if doc_info["stored_file"] == stored_file_path:
                doc_info["is_indexed"] = True
                changed_doc_id = doc_id
                break
        if changed_doc_id is not None:
            self._save_documents(changed_doc_id)
            logger.info("document_marked_indexed", doc_id=str(changed_doc_id), stored_file=str(stored_file_path))

    def register_chunks(self, split_docs: list[Document]):
        """Persists chunk neighbor links for fast context expansion."""
        CHUNK_REGISTRY.register_chunks(split_docs)

    def has_chunk_registry_for_paths(
        self,
        file_paths: list[str] | None = None,
        *,
        require_logical_pages: bool = False,
    ) -> bool:
        """Returns True when every source path has chunk registry rows (and optional logical pages)."""
        paths = file_paths if file_paths is not None else self.get_all_paths()
        if not paths:
            return False
        if require_logical_pages:
            return all(
                CHUNK_REGISTRY.source_has_chunks(path)
                and CHUNK_REGISTRY.source_has_logical_pages(path)
                for path in paths
            )
        return all(CHUNK_REGISTRY.source_has_chunks(path) for path in paths)
