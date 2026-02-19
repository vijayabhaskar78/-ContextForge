"""
Conversational memory manager for the ContextForge CLI assistant.

Implements hierarchical memory with:
- short-term memory (recent turns)
- episodic memory (retrieved historical turns)
- semantic/profile memory (stable user facts)
- session state (active chapter, last topic, etc.)
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    CACHE_DIR,
    MEMORY_CONTEXT_EPISODIC_TURNS,
    MEMORY_CONTEXT_RECENT_TURNS,
    MEMORY_EPISODIC_SCAN_LIMIT,
    MEMORY_EPISODIC_TOKEN_FILTER_LIMIT,
    MEMORY_PROFILE_FACT_LIMIT,
    MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS,
    MEMORY_SUMMARY_REFRESH_EVERY_N_TURNS,
    OLLAMA_AVAILABLE,
    PROMPT_DOCS_RATIO,
    PROMPT_MEMORY_RATIO,
    PROMPT_QUERY_RATIO,
    PROMPT_TOTAL_TOKEN_BUDGET,
    STORAGE_DIR,
    console,
)
from .db_migrations import SqliteMigration, apply_sqlite_migrations
from .observability import get_logger
from .query_processor import QueryProcessor
from .tokenization import tokenize_for_matching

logger = get_logger(__name__)

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None

_TOKEN_ENCODER = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokenize(text: str) -> list[str]:
    return tokenize_for_matching(text, min_len=3)


def _token_blob(text: str) -> str:
    """Stores a compact normalized token cache for retrieval scoring."""
    return " ".join(sorted(set(_tokenize(text))))


def _get_token_encoder():
    global _TOKEN_ENCODER
    if tiktoken is None:
        return None
    if _TOKEN_ENCODER is None:
        try:
            _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKEN_ENCODER = None
    return _TOKEN_ENCODER


def _estimate_token_count(text: str) -> int:
    payload = str(text or "")
    if not payload:
        return 0
    encoder = _get_token_encoder()
    if encoder is not None:
        try:
            return int(len(encoder.encode(payload)))
        except Exception:
            pass
    # Fallback heuristic when tokenizer backend is unavailable.
    return max(1, len(tokenize_for_matching(payload, min_len=1)))


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    budget = max(0, int(max_tokens))
    payload = str(text or "").strip()
    if budget <= 0 or not payload:
        return ""
    if _estimate_token_count(payload) <= budget:
        return payload

    encoder = _get_token_encoder()
    if encoder is not None:
        try:
            encoded = encoder.encode(payload)
            return encoder.decode(encoded[:budget]).strip()
        except Exception:
            pass

    tokens = payload.split()
    return " ".join(tokens[:budget]).strip()


def _json_loads_or_default(raw: str | None, default: Any):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


FOLLOWUP_PRONOUN_RE = re.compile(r"\b(this|that|it|them|these|those)\b", re.IGNORECASE)
CHAPTER_FOLLOWUP_RE = re.compile(r"\b(this|that|same|previous)\s+chapter\b", re.IGNORECASE)
GENERIC_CONTINUATION_RE = re.compile(
    r"^\s*(?:can you|could you|please)?\s*"
    r"(?:explain|summarize|describe|elaborate|expand|clarify|simplify)"
    r"(?:\s+more|\s+further)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
LOW_SIGNAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:tell me|give me)\s+more(?:\s+about\s+(?:this|that|it))?\s*[?.!]*\s*$"
    r"|^\s*(?:more details|details|more)\s*[?.!]*\s*$"
    r"|^\s*explain\s+more\s*[?.!]*\s*$",
    re.IGNORECASE,
)
ACK_EXACT_PHRASES = {
    "ok", "okay", "thanks", "thank you", "got it", "yes", "yep",
    "cool", "nice", "alright", "all right", "understood", "sounds good",
}
ACK_PREFIX_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|thanks?|thank you|got it|yes|yep|cool|nice|alright|all right|understood)"
    r"(?:\s*(?:[.!]+|so far|for now|i see|makes sense))?\s*$",
    re.IGNORECASE,
)
PRONOUN_ONLY_RE = re.compile(
    r"^\s*(?:this|that|it|them|these|those|this one|that one)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
EXPLICIT_TOPIC_RE = re.compile(
    r"\bchapter\s+([1-9]|[1-2][0-9]|3[0-9])\b"
    r"|^\s*(?:what is|who is|define|meaning of)\s+([a-z0-9][a-z0-9\- ]{1,120})\s*[?.!]*$"
    r"|\"[^\"]{2,120}\"|'[^']{2,120}'",
    re.IGNORECASE,
)
TOPIC_STOPWORDS = {
    "what", "which", "when", "where", "who", "why", "how", "many", "number",
    "need", "count", "tell", "about", "explain", "summary", "summarize",
    "describe", "clarify", "simplify", "elaborate", "expand", "chapter", "chapters",
    "book", "pdf", "please", "give", "show", "there", "simple", "simpler", "terms",
    "term", "detail", "details", "more", "further", "in", "with", "for", "the",
    "a", "an", "to", "of", "on", "and", "or", "me", "you", "this", "that", "it",
}
MAX_REGEX_INPUT_CHARS = 2048


def _bounded_regex_input(text: str, max_chars: int = MAX_REGEX_INPUT_CHARS) -> str:
    """Bounds user-controlled regex input length to reduce ReDoS risk."""
    return str(text or "")[: max(64, int(max_chars))]


def _query_mentions_topic(query: str, topic: str) -> bool:
    """Checks whether query already contains the topic explicitly."""
    if not topic:
        return False
    q_lower = (query or "").lower()
    t_lower = topic.lower()
    if t_lower in q_lower:
        return True

    query_tokens = set(_tokenize(query))
    topic_tokens = set(_tokenize(topic))
    if not query_tokens or not topic_tokens:
        return False

    overlap = len(query_tokens.intersection(topic_tokens))
    if len(topic_tokens) == 1:
        return overlap >= 1
    if len(topic_tokens) == 2:
        return overlap >= 2
    return overlap >= 2


def _is_ambiguous_followup_query(query: str) -> bool:
    """Returns True for pronoun-heavy or low-signal follow-up prompts."""
    q = (query or "").strip()
    if not q:
        return False
    lower = q.lower()
    if CHAPTER_FOLLOWUP_RE.search(lower):
        return True
    if GENERIC_CONTINUATION_RE.match(lower):
        return True
    if LOW_SIGNAL_FOLLOWUP_RE.match(lower):
        return True
    if FOLLOWUP_PRONOUN_RE.search(lower):
        has_followup_verb = bool(
            re.search(r"\b(explain|summarize|describe|elaborate|expand|clarify|simplify|details?|more)\b", lower)
        )
        if has_followup_verb:
            return True
        return len(_tokenize(lower)) <= 8
    return False


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def is_acknowledgment_query(user_query: str) -> bool:
    """Detects short acknowledgment-only user turns."""
    q = _compact_spaces(user_query).lower()
    if not q:
        return False
    if q in ACK_EXACT_PHRASES:
        return True
    if ACK_PREFIX_RE.match(q):
        return True
    tokens = tokenize_for_matching(q, min_len=1)
    if 0 < len(tokens) <= 3:
        joined = " ".join(tokens)
        if joined in ACK_EXACT_PHRASES:
            return True
    return False


def has_semantic_topic_signal(user_query: str) -> bool:
    """
    True when a query is topic-bearing enough to update active/last topic.
    Requires >=2 meaningful words unless explicit reference is present.
    """
    q = _compact_spaces(user_query)
    if not q:
        return False
    lower = q.lower()
    if is_acknowledgment_query(lower):
        return False
    if PRONOUN_ONLY_RE.match(lower):
        return False
    if EXPLICIT_TOPIC_RE.search(lower):
        return True

    meaningful = [t for t in _tokenize(lower) if t not in TOPIC_STOPWORDS]
    return len(meaningful) >= 2


def _rewrite_followup_with_anchor(query: str, anchor: str) -> str:
    """Rewrites ambiguous follow-up text to include an explicit anchor topic."""
    q = _compact_spaces(query)
    if not q or not anchor:
        return q

    # Resolve explicit chapter pronouns first.
    rewritten = CHAPTER_FOLLOWUP_RE.sub(anchor, q)

    # "Explain this ..." -> "Explain <anchor> ..."
    rewritten = re.sub(
        r"^\s*(?P<prefix>(?:can you|could you|please)\s+)?"
        r"(?P<verb>explain|summarize|describe|elaborate|expand|clarify|simplify)\s+"
        r"(?:this|that|it|them|these|those)\b(?P<tail>.*)$",
        lambda m: _compact_spaces(
            f"{(m.group('prefix') or '')}{m.group('verb')} {anchor}{m.group('tail') or ''}"
        ),
        rewritten,
        flags=re.IGNORECASE,
    )
    if rewritten != q:
        return rewritten

    # "tell me more" / "more details" style follow-ups.
    if re.match(r"^\s*(?:tell me|give me)\s+more(?:\s+about\s+(?:this|that|it))?\s*[?.!]*\s*$", q, re.IGNORECASE):
        return f"Tell me more about {anchor}"
    if re.match(r"^\s*(?:more details|details|more)\s*[?.!]*\s*$", q, re.IGNORECASE):
        return f"Explain {anchor} in more detail"
    if re.match(
        r"^\s*(?P<verb>explain|summarize|describe|elaborate|expand|clarify|simplify)\s+"
        r"(?:more|further)\s*[?.!]*\s*$",
        q,
        re.IGNORECASE,
    ):
        verb = re.match(
            r"^\s*(?P<verb>explain|summarize|describe|elaborate|expand|clarify|simplify)\b",
            q,
            re.IGNORECASE,
        ).group("verb")
        return f"{verb.capitalize()} {anchor} in more detail"

    # Fallback: replace first pronoun token.
    pronoun_rewritten = FOLLOWUP_PRONOUN_RE.sub(anchor, q, count=1)
    if pronoun_rewritten != q:
        return _compact_spaces(pronoun_rewritten)

    # Fallback: low-signal "Explain" with no direct object.
    base_match = re.match(
        r"^\s*(?:can you|could you|please)?\s*(?P<verb>explain|summarize|describe|elaborate|expand|clarify|simplify)\b",
        q,
        re.IGNORECASE,
    )
    if base_match:
        return f"{base_match.group('verb').capitalize()} {anchor}"
    return q


def resolve_followup_query(user_query: str, session_state: dict) -> str:
    """
    Resolves ambiguous follow-up references using session state topic anchors.

    Expected behavior:
    - "Explain this in simpler terms" + active_topic="hierarchical memory"
      -> "Explain hierarchical memory in simpler terms"
    """
    query = _compact_spaces(user_query)
    if not query:
        return user_query

    state = session_state or {}
    lower = query.lower()
    if "referring to" in lower:
        return query

    # If query is already explicit, do not rewrite.
    if is_acknowledgment_query(query):
        return query
    if re.search(r"\bchapter\s+([1-9]|[1-2][0-9]|3[0-9])\b", lower):
        return query

    active_topic = str(state.get("active_topic") or "").strip()
    last_topic = str(state.get("last_topic") or "").strip()
    topic_anchor = active_topic or last_topic

    if topic_anchor and _query_mentions_topic(query, topic_anchor):
        return query
    if not _is_ambiguous_followup_query(query):
        return query

    active_chapter = state.get("active_chapter")
    active_chapter_title = str(state.get("active_chapter_title") or "").strip()
    chapter_anchor = ""
    if active_chapter:
        chapter_anchor = f"chapter {active_chapter}"
        if active_chapter_title:
            chapter_anchor = f"{chapter_anchor} {active_chapter_title}"

    anchor = ""
    if CHAPTER_FOLLOWUP_RE.search(lower) and chapter_anchor:
        anchor = chapter_anchor
    elif topic_anchor:
        anchor = topic_anchor
    elif chapter_anchor:
        anchor = chapter_anchor
    else:
        return query

    rewritten = _rewrite_followup_with_anchor(query, anchor)
    return rewritten or query


class HierarchicalMemoryManager:
    """SQLite-backed memory manager used by the interactive Q&A loop."""

    def __init__(self, db_path: str | Path | None = None):
        default_dir = CACHE_DIR
        default_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else default_dir / "conversation_memory.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = self._connect()
        self._fts5_enabled = False
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
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
                raise RuntimeError("memory manager connection is closed")
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
        def _ensure_turn_token_blob(conn: sqlite3.Connection):
            turn_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "token_blob" not in turn_columns:
                conn.execute("ALTER TABLE turns ADD COLUMN token_blob TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session_token ON turns(session_id, token_blob)")

        migrations = [
            SqliteMigration(
                version=1,
                name="create_memory_tables",
                statements=(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        last_active_at TEXT NOT NULL,
                        state_json TEXT NOT NULL DEFAULT '{}',
                        summary TEXT NOT NULL DEFAULT '',
                        doc_signature TEXT
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        intent TEXT,
                        topic TEXT,
                        metadata_json TEXT,
                        token_blob TEXT NOT NULL DEFAULT '',
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS profile_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.5,
                        updated_at TEXT NOT NULL,
                        UNIQUE(session_id, fact_key),
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_turns_session_created ON turns(session_id, created_at)",
                ),
            ),
            SqliteMigration(
                version=2,
                name="ensure_turn_token_blob",
                runner=_ensure_turn_token_blob,
            ),
        ]

        with self._connection() as conn:
            apply_sqlite_migrations(
                conn,
                component="memory_manager",
                migrations=migrations,
            )
            self._ensure_fts_index(conn)

    def _ensure_fts_index(self, conn: sqlite3.Connection):
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    session_id UNINDEXED,
                    role UNINDEXED,
                    text,
                    token_blob,
                    topic
                )
                """
            )
            conn.execute(
                """
                INSERT INTO turns_fts(rowid, session_id, role, text, token_blob, topic)
                SELECT
                    turns.id,
                    turns.session_id,
                    turns.role,
                    turns.text,
                    turns.token_blob,
                    COALESCE(turns.topic, '')
                FROM turns
                LEFT JOIN turns_fts ON turns_fts.rowid = turns.id
                WHERE turns_fts.rowid IS NULL
                """
            )
            self._fts5_enabled = True
        except sqlite3.Error as exc:
            self._fts5_enabled = False
            logger.warning("memory_fts_unavailable", error=str(exc))

    def start_session(self, doc_paths: list[str] | None = None) -> str:
        now = _utcnow_iso()
        session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        doc_signature = "|".join(sorted(Path(p).name for p in (doc_paths or [])))
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, started_at, last_active_at, state_json, summary, doc_signature)
                VALUES (?, ?, ?, '{}', '', ?)
                """,
                (session_id, now, now, doc_signature),
            )
        return session_id

    def _touch_session(self, session_id: str):
        with self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (_utcnow_iso(), session_id),
            )

    def get_state(self, session_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        return _json_loads_or_default(row["state_json"], {})

    def update_state(self, session_id: str, updates: dict[str, Any]):
        if not updates:
            return
        with self._connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return
            state = _json_loads_or_default(row["state_json"], {})
            now = _utcnow_iso()
            history = state.get("_state_history", [])
            if not isinstance(history, list):
                history = []
            for key, value in updates.items():
                if state.get(key) != value:
                    history.append({"at": now, "key": key, "from": state.get(key), "to": value})
                    state[key] = value
            if history:
                state["_state_history"] = history[-20:]
            conn.execute(
                "UPDATE sessions SET state_json = ?, last_active_at = ? WHERE session_id = ?",
                (json.dumps(state, ensure_ascii=True), now, session_id),
            )

    def get_session_summary(self, session_id: str) -> str:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["summary"] if row and row["summary"] else ""

    def _set_summary(self, session_id: str, summary: str):
        with self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ?, last_active_at = ? WHERE session_id = ?",
                (summary, _utcnow_iso(), session_id),
            )

    def record_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        intent: str | None = None,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        clean_text = (text or "").strip()
        if role not in {"user", "assistant"} or not clean_text:
            return

        metadata_json = json.dumps(metadata, ensure_ascii=True) if metadata else None
        token_cache = _token_blob(clean_text)
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO turns (session_id, role, text, created_at, intent, topic, metadata_json, token_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, clean_text, _utcnow_iso(), intent, topic, metadata_json, token_cache),
            )
            turn_id = int(getattr(cursor, "lastrowid", 0) or 0)
            if self._fts5_enabled and turn_id > 0:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO turns_fts(rowid, session_id, role, text, token_blob, topic)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (turn_id, session_id, role, clean_text, token_cache, str(topic or "")),
                    )
                except sqlite3.Error as exc:
                    # Keep memory writes resilient when FTS is unavailable at runtime.
                    self._fts5_enabled = False
                    console.print(f"[yellow]FTS insert failed; episodic retrieval will use fallback scan: {exc}[/yellow]")

        if role == "user":
            self._extract_and_upsert_profile_facts(session_id, clean_text)
        else:
            self._maybe_refresh_summary(session_id)

        self._touch_session(session_id)

    def register_user_query(
        self,
        session_id: str,
        raw_query: str,
        resolved_query: str | None = None,
        intent: str | None = None,
        topic: str | None = None,
    ):
        metadata = {}
        if resolved_query and resolved_query.strip() and resolved_query.strip() != raw_query.strip():
            metadata["resolved_query"] = resolved_query.strip()
        self.record_turn(
            session_id=session_id,
            role="user",
            text=raw_query,
            intent=intent,
            topic=topic,
            metadata=metadata or None,
        )

    def register_assistant_answer(
        self,
        session_id: str,
        answer: str,
        intent: str | None = None,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.record_turn(
            session_id=session_id,
            role="assistant",
            text=answer,
            intent=intent,
            topic=topic,
            metadata=metadata,
        )

    def get_recent_turns(self, session_id: str, limit: int = MEMORY_CONTEXT_RECENT_TURNS) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, text, intent, topic, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max(1, limit)),
            ).fetchall()
        turns = [dict(row) for row in rows]
        turns.reverse()
        return turns

    def get_recent_assistant_turns(
        self,
        session_id: str,
        limit: int = MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS,
    ) -> list[dict[str, Any]]:
        """Returns recent assistant turns (newest first) including parsed metadata."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, text, intent, topic, created_at, metadata_json
                FROM turns
                WHERE session_id = ? AND role = 'assistant'
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max(1, limit)),
            ).fetchall()

        turns: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_loads_or_default(item.pop("metadata_json", None), {})
            turns.append(item)
        return turns

    def get_conversation_history(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Returns chronological conversation turns for explicit recall queries."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, text, intent, topic, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max(1, limit)),
            ).fetchall()
        turns = [dict(row) for row in rows]
        turns.reverse()
        return turns

    def get_profile_facts(self, session_id: str, limit: int = MEMORY_PROFILE_FACT_LIMIT) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT fact_key, fact_value, confidence, updated_at
                FROM profile_facts
                WHERE session_id = ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (session_id, max(1, limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def retrieve_episodic_turns(
        self,
        session_id: str,
        query: str,
        limit: int = MEMORY_CONTEXT_EPISODIC_TURNS,
    ) -> list[dict[str, Any]]:
        perf_start = time.perf_counter()
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            console.print("[dim][perf] retrieve_episodic_turns: 0.0 ms (query_tokens=0, rows=0, selected=0)[/dim]")
            return []

        token_filters = sorted(query_tokens)[:MEMORY_EPISODIC_TOKEN_FILTER_LIMIT]
        rows = []
        query_mode = "fts5" if self._fts5_enabled else "fallback_like"
        fallback_scan_required = not self._fts5_enabled

        if self._fts5_enabled and token_filters:
            fts_query = " OR ".join(f"token_blob:{token}" for token in token_filters)
            with self._connection() as conn:
                try:
                    rows = conn.execute(
                        """
                        SELECT t.id, t.role, t.text, t.intent, t.topic, t.created_at, t.token_blob
                        FROM turns_fts
                        JOIN turns AS t ON t.id = turns_fts.rowid
                        WHERE turns_fts.session_id = ?
                          AND turns_fts MATCH ?
                        ORDER BY bm25(turns_fts), t.id DESC
                        LIMIT ?
                        """,
                        (session_id, fts_query, MEMORY_EPISODIC_SCAN_LIMIT),
                    ).fetchall()
                except sqlite3.Error as exc:
                    # Fallback keeps compatibility if SQLite build lacks FTS5.
                    self._fts5_enabled = False
                    query_mode = "fallback_like"
                    fallback_scan_required = True
                    console.print(f"[yellow]FTS query failed; using fallback scan: {exc}[/yellow]")

        if fallback_scan_required:
            token_where = " OR ".join("token_blob LIKE ?" for _ in token_filters)
            sql = (
                "SELECT id, role, text, intent, topic, created_at, token_blob "
                "FROM turns "
                "WHERE session_id = ?"
            )
            params: list[Any] = [session_id]
            if token_where:
                sql += f" AND ({token_where})"
                params.extend([f"%{token}%" for token in token_filters])
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(int(MEMORY_EPISODIC_SCAN_LIMIT))
            with self._connection() as conn:
                rows = conn.execute(sql, params).fetchall()

        scanned_rows = len(rows)
        scored = []
        for row in rows:
            token_cache = str(row["token_blob"] or "").strip()
            text_tokens = set(token_cache.split()) if token_cache else set(_tokenize(row["text"]))
            if not text_tokens:
                continue
            overlap = len(query_tokens.intersection(text_tokens))
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_tokens))
            if row["role"] == "assistant":
                score += 0.05
            if row["topic"]:
                topic_tokens = set(_tokenize(row["topic"]))
                if query_tokens.intersection(topic_tokens):
                    score += 0.2
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_ids = set()
        for _, row in scored:
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            selected.append(dict(row))
            if len(selected) >= max(1, limit):
                break
        selected.sort(key=lambda r: r["id"])
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0
        console.print(
            f"[dim][perf] retrieve_episodic_turns: {elapsed_ms:.1f} ms "
            f"(mode={query_mode}, query_tokens={len(query_tokens)}, rows={scanned_rows}, selected={len(selected)})[/dim]"
        )
        logger.info(
            "retrieve_episodic_turns",
            mode=query_mode,
            query_tokens=len(query_tokens),
            rows=scanned_rows,
            selected=len(selected),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return selected

    def build_memory_context(
        self,
        session_id: str,
        query: str,
        max_recent_turns: int = MEMORY_CONTEXT_RECENT_TURNS,
        max_episodic_turns: int = MEMORY_CONTEXT_EPISODIC_TURNS,
    ) -> str:
        state = self.get_state(session_id)
        summary = self.get_session_summary(session_id).strip()
        recent = self.get_recent_turns(session_id, limit=max_recent_turns)
        episodic = self.retrieve_episodic_turns(session_id, query, limit=max_episodic_turns)
        facts = self.get_profile_facts(session_id, limit=MEMORY_PROFILE_FACT_LIMIT)
        return self._format_memory_context(state, summary, recent, episodic, facts)

    def _format_memory_context(
        self,
        state: dict[str, Any],
        summary: str,
        recent: list[dict[str, Any]],
        episodic: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []

        visible_state = {
            key: value
            for key, value in state.items()
            if not key.startswith("_") and value not in (None, "", [], {})
        }
        if visible_state:
            lines.append("Session state:")
            for key in ("active_topic", "active_chapter", "active_chapter_title", "last_topic", "last_intent", "response_mode"):
                if key in visible_state:
                    lines.append(f"- {key}: {visible_state[key]}")

        if facts:
            lines.append("Known user facts:")
            for fact in facts:
                lines.append(f"- {fact['fact_key']}: {fact['fact_value']}")

        if summary:
            lines.append("Rolling session summary:")
            lines.append(f"- {summary}")

        if recent:
            lines.append("Recent dialogue:")
            for turn in recent:
                role = "User" if turn["role"] == "user" else "Assistant"
                clipped = " ".join(turn["text"].split())
                if len(clipped) > 220:
                    clipped = clipped[:220].rsplit(" ", 1)[0] + "..."
                lines.append(f"- {role}: {clipped}")

        if episodic:
            lines.append("Relevant earlier turns:")
            for turn in episodic:
                role = "User" if turn["role"] == "user" else "Assistant"
                clipped = " ".join(turn["text"].split())
                if len(clipped) > 220:
                    clipped = clipped[:220].rsplit(" ", 1)[0] + "..."
                lines.append(f"- {role}: {clipped}")

        return "\n".join(lines).strip()

    def compose_model_input(self, session_id: str, query: str) -> str:
        user_query = str(query or "").strip()
        memory_context = self.build_memory_context(session_id, user_query)
        if not memory_context:
            return user_query

        total_budget = max(512, int(PROMPT_TOTAL_TOKEN_BUDGET))
        docs_budget = max(1, int(total_budget * float(PROMPT_DOCS_RATIO)))
        memory_ratio_cap = max(1, int(total_budget * float(PROMPT_MEMORY_RATIO)))
        prefix = (
            "Conversation memory for continuity (use only for resolving references; "
            "ground factual answers in the provided document context):\n"
        )
        query_prefix = "\n\nCurrent question:\n"

        reserved_tokens = (
            _estimate_token_count(prefix)
            + _estimate_token_count(query_prefix)
            + _estimate_token_count(user_query)
            + docs_budget
        )
        available_for_memory = max(0, total_budget - reserved_tokens)
        memory_budget = min(memory_ratio_cap, available_for_memory)
        truncated_memory = _truncate_to_token_budget(memory_context, memory_budget)

        if not truncated_memory:
            logger.warning(
                "memory_context_dropped_due_budget",
                session_id=session_id,
                total_budget=total_budget,
                docs_budget=docs_budget,
                query_tokens=_estimate_token_count(user_query),
            )
            return user_query

        original_tokens = _estimate_token_count(memory_context)
        final_tokens = _estimate_token_count(truncated_memory)
        if final_tokens < original_tokens:
            logger.info(
                "memory_context_truncated",
                session_id=session_id,
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                memory_budget=memory_budget,
                total_budget=total_budget,
                docs_budget=docs_budget,
            )

        return f"{prefix}{truncated_memory}{query_prefix}{user_query}"

    def resolve_followup_query(self, session_id: str, query: str) -> str:
        state = self.get_state(session_id)
        return resolve_followup_query(query, state)

    def enhance_retrieval_query(
        self,
        session_id: str,
        query: str,
        *,
        query_is_resolved: bool = False,
        strict: bool = False,
    ) -> str:
        resolved = _compact_spaces(query) if query_is_resolved else self.resolve_followup_query(session_id, query)
        # Enforce strict resolved-query retrieval: no additional expansion/hints.
        if query_is_resolved:
            return resolved
        state = self.get_state(session_id)
        hints = []
        active_chapter = state.get("active_chapter")
        active_title = state.get("active_chapter_title")
        active_topic = str(state.get("active_topic") or "").strip()
        last_topic = str(state.get("last_topic") or "").strip()
        topic_anchor = active_topic or last_topic

        if strict:
            return resolved

        # If user asks an explicit new topic (e.g. "What is RAG?"),
        # do not append previous-topic hints that can contaminate retrieval scope.
        lowered_resolved = resolved.lower()
        if EXPLICIT_TOPIC_RE.search(lowered_resolved) and not _is_ambiguous_followup_query(resolved):
            return resolved

        if active_chapter and not re.search(r"\bchapter\s+\d{1,2}\b", resolved.lower()):
            chapter_hint = f"chapter {active_chapter}"
            if active_title:
                chapter_hint = f"{chapter_hint} {active_title}"
            hints.append(chapter_hint)

        if topic_anchor and not _query_mentions_topic(resolved, topic_anchor):
            hints.append(topic_anchor)

        if not hints:
            return resolved
        return _compact_spaces(f"{resolved} {' '.join(hints)}")

    def derive_topic(self, query: str) -> str:
        q = _bounded_regex_input(query).strip()
        if not q:
            return ""
        if is_acknowledgment_query(q):
            return ""
        if PRONOUN_ONLY_RE.match(q.lower()):
            return ""
        chapter_number = self._extract_chapter_number(q)
        if chapter_number is not None:
            title_match = re.search(
                r"\bchapter\s+(?:[1-9]|[1-2][0-9]|3[0-9])\s+([a-z0-9][a-z0-9 ,:&'()/\-]{2,80})",
                q.lower(),
            )
            if title_match:
                return f"chapter {chapter_number} {title_match.group(1).strip()}"
            return f"chapter {chapter_number}"

        tokens = [t for t in _tokenize(q) if t not in TOPIC_STOPWORDS]
        if not tokens:
            return ""
        if len(tokens) < 2 and not EXPLICIT_TOPIC_RE.search(q.lower()):
            return ""
        return " ".join(tokens[:6])

    def is_acknowledgment_query(self, query: str) -> bool:
        return is_acknowledgment_query(query)

    def has_semantic_topic_signal(self, query: str) -> bool:
        return has_semantic_topic_signal(query)

    def _extract_chapter_number(self, query: str) -> int | None:
        q = _bounded_regex_input(query).lower()
        patterns = (
            r"\bchapter\s+([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\b",
            r"\b([1-9]|[1-2][0-9]|3[0-9])(?:st|nd|rd|th)?\s+ch[a-z]*ter[s]?\b",
        )
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return int(match.group(1))
        return None

    def _extract_and_upsert_profile_facts(self, session_id: str, user_text: str):
        lower = _bounded_regex_input(user_text).lower()
        extractions: list[tuple[str, str, float]] = []

        name_match = re.search(r"\b(?:my name is|call me)\s+([a-z][a-z '\-]{1,30})\b", lower)
        if name_match:
            extractions.append(("name", name_match.group(1).strip().title(), 0.95))

        location_match = re.search(r"\bi am from\s+([a-z][a-z '\-]{1,40})\b", lower)
        if location_match:
            extractions.append(("location", location_match.group(1).strip().title(), 0.8))

        role_match = re.search(r"\bi (?:work as|am a|am an)\s+([a-z][a-z0-9 '\-/]{1,40})\b", lower)
        if role_match:
            extractions.append(("role", role_match.group(1).strip(), 0.75))

        preference_match = re.search(r"\bi prefer\s+([a-z0-9 ,'\-/]{3,80})\b", lower)
        if preference_match:
            extractions.append(("preference", preference_match.group(1).strip(), 0.65))

        for key, value, confidence in extractions:
            self._upsert_profile_fact(session_id, key, value, confidence)

    def _upsert_profile_fact(self, session_id: str, fact_key: str, fact_value: str, confidence: float):
        now = _utcnow_iso()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, confidence
                FROM profile_facts
                WHERE session_id = ? AND fact_key = ?
                """,
                (session_id, fact_key),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO profile_facts (session_id, fact_key, fact_value, confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, fact_key, fact_value, confidence, now),
                )
                return

            updated_confidence = max(float(row["confidence"]), confidence)
            conn.execute(
                """
                UPDATE profile_facts
                SET fact_value = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (fact_value, updated_confidence, now, row["id"]),
            )

    def _maybe_refresh_summary(self, session_id: str):
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        total_turns = int(row["n"]) if row else 0
        interval = max(2, int(MEMORY_SUMMARY_REFRESH_EVERY_N_TURNS))
        if total_turns == 0 or total_turns % interval != 0:
            return
        summary = self._build_summary(session_id)
        if summary:
            self._set_summary(session_id, summary)

    def _build_summary(self, session_id: str) -> str:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT role, text
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 16
                """,
                (session_id,),
            ).fetchall()

        snippets = []
        for row in reversed(rows):
            role = "U" if row["role"] == "user" else "A"
            text = " ".join(row["text"].split())
            text = re.sub(r"[`*_#>-]+", "", text).strip()
            if len(text) > 120:
                text = text[:120].rsplit(" ", 1)[0] + "..."
            snippets.append(f"{role}: {text}")
        if not snippets:
            return ""
        return " | ".join(snippets[-8:])
