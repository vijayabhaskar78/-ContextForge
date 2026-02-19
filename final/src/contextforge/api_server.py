"""
FastAPI service layer for the ContextForge conversational RAG system.

Wraps existing RAG pipeline modules without modifying core logic.
Exposes POST /query and GET /metrics endpoints.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import re
import time
import uuid
import asyncio
import hashlib
from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Existing module imports (treated as black box) ---
from .document_manager import DocumentUploader
from .memory_manager import HierarchicalMemoryManager
from .rag_pipeline import build_rag_pipeline
from .metrics import metrics_collector

# ---------------------------------------------------------------------------
# Compile-time constants & patterns
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_MAX_CONTEXT_DOCS = 3          # Top-N docs to send to LLM (fewer = faster)
_MAX_CONTEXT_CHARS = 500       # Max chars per context chunk
_CACHE_MAX_SIZE = 128          # LRU cache capacity
_THREAD_POOL_WORKERS = 8       # Concurrent query workers


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier for conversational memory",
    )


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    tokens_used: int
    session_id: str


# ---------------------------------------------------------------------------
# Application state populated at startup
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}

# Thread pool for running synchronous pipeline calls off the event loop.
_executor = ThreadPoolExecutor(max_workers=_THREAD_POOL_WORKERS)


# ---------------------------------------------------------------------------
# LRU query cache (thread-safe)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple thread-safe LRU cache using OrderedDict."""

    def __init__(self, max_size: int = _CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max = max_size
        import threading
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max:
                    self._cache.popitem(last=False)
                self._cache[key] = value


_query_cache = _LRUCache()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline once at startup; clean up on shutdown."""
    import logging

    retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid")
    uploader = DocumentUploader()
    retriever, doc_chain = build_rag_pipeline(uploader, retrieval_mode=retrieval_mode)
    if retriever is None or doc_chain is None:
        logging.warning(
            "RAG pipeline not ready — no documents or LLM unavailable. "
            "POST /query will return 503 until pipeline is initialized."
        )
    memory = HierarchicalMemoryManager()

    _state["retriever"] = retriever
    _state["doc_chain"] = doc_chain
    _state["memory"] = memory
    _state["uploader"] = uploader

    yield  # Application is running.

    # Shutdown: release resources.
    memory.close()
    _executor.shutdown(wait=False)
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ContextForge API",
    description="Conversational RAG service powered by ContextForge",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

# Track which sessions have been ensured this process lifetime.
_known_sessions: set[str] = set()


def _ensure_session(memory: HierarchicalMemoryManager, session_id: str) -> None:
    """Create the session row if it doesn't already exist (idempotent)."""
    if session_id in _known_sessions:
        return
    import sqlite3
    try:
        with memory._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (session_id, started_at, last_active_at, state_json, summary, doc_signature)
                VALUES (?, datetime('now'), datetime('now'), '{}', '', '')
                """,
                (session_id,),
            )
    except sqlite3.Error:
        pass  # Best-effort; don't block the query.
    _known_sessions.add(session_id)


def _cache_key(query: str) -> str:
    """Deterministic cache key for a query (session-independent for factual Q&A)."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _trim_context(docs: list, max_docs: int = _MAX_CONTEXT_DOCS, max_chars: int = _MAX_CONTEXT_CHARS) -> list:
    """Trim retrieved docs to reduce LLM input tokens and latency."""
    trimmed = docs[:max_docs]
    for doc in trimmed:
        content = getattr(doc, "page_content", "")
        if len(content) > max_chars:
            doc.page_content = content[:max_chars] + "..."
    return trimmed


def _run_query(query: str, session_id: str) -> dict:
    """
    Synchronous query execution using the initialised pipeline.

    Bypasses the CLI-coupled handle_user_query and calls retriever/chain
    directly to avoid Rich console side-effects.
    """
    retriever = _state["retriever"]
    doc_chain = _state["doc_chain"]
    memory: HierarchicalMemoryManager = _state["memory"]

    # Ensure session exists in the memory database.
    _ensure_session(memory, session_id)

    # Resolve follow-up references using memory.
    effective_query = memory.resolve_followup_query(session_id, query)

    # Check cache first.
    ckey = _cache_key(effective_query)
    cached = _query_cache.get(ckey)
    if cached is not None:
        # Still record in memory for follow-up context.
        memory.record_turn(session_id, role="user", text=query)
        memory.record_turn(session_id, role="assistant", text=cached["answer"])
        return cached

    # Retrieval.
    docs = retriever.invoke(effective_query)
    if not docs:
        return {
            "answer": "No relevant context found for your query.",
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # Trim context to reduce latency.
    docs = _trim_context(docs)

    # Generation.
    answer_raw = doc_chain.invoke({"context": docs, "input": effective_query})
    answer = _THINK_RE.sub("", str(answer_raw)).strip()

    # Token estimates (1 token ≈ 4 chars).
    context_text = " ".join(getattr(d, "page_content", "") for d in docs)
    input_tokens = (len(context_text) + len(effective_query)) // 4
    output_tokens = len(answer) // 4

    result = {
        "answer": answer,
        "tokens_used": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    # Cache the result.
    _query_cache.put(ckey, result)

    # Record turn in memory for follow-up resolution.
    memory.record_turn(session_id, role="user", text=query)
    memory.record_turn(session_id, role="assistant", text=answer)

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a user query through the RAG pipeline."""
    if _state.get("retriever") is None or _state.get("doc_chain") is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Upload documents and ensure the LLM is available.",
        )
    start = time.perf_counter()

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_query,
            request.query,
            request.session_id,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        model_name = os.getenv("API_MODEL_NAME", "")
        metrics_collector.record_request(
            latency_ms,
            success=True,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            model=model_name,
        )

        return QueryResponse(
            answer=result["answer"],
            latency_ms=round(latency_ms, 2),
            tokens_used=result["tokens_used"],
            session_id=request.session_id,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics_collector.record_request(latency_ms, success=False)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
async def metrics_endpoint():
    """Return aggregated service metrics."""
    return metrics_collector.get_summary()
