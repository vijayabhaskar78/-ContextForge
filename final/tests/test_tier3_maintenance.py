import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contextforge import app
from contextforge import document_manager
from contextforge import memory_manager
from contextforge import rag_pipeline
from contextforge.config import ASYNC_RETRIEVER_MAX_WORKERS, LOG_PATH, MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS
from contextforge.document_manager import ChunkRegistry, DocumentUploader
from contextforge.memory_manager import HierarchicalMemoryManager
from contextforge.observability import get_logger
from contextforge.storage_provider import LocalFileStorageProvider
from contextforge.tokenization import tokenize_for_matching


class _FakeUploader:
    def get_all_paths(self):
        return []


class _MemoryStub:
    def __init__(self):
        self.limit_seen = None

    def get_state(self, _session_id):
        return {}

    def get_recent_assistant_turns(self, _session_id, limit=0):
        self.limit_seen = int(limit)
        return []


class _TinyRetriever:
    def __init__(self, docs):
        self.docs = list(docs)

    def invoke(self, _query):
        return list(self.docs)


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _CapturedPool:
    last_workers = None

    def __init__(self, max_workers=1):
        _CapturedPool.last_workers = int(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _CompletedFuture(fn(*args, **kwargs))


class TestTier3Maintenance(unittest.TestCase):
    def test_memory_schema_migrations_are_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "memory.sqlite"
            mm = HierarchicalMemoryManager(db_path=db_path)
            try:
                with mm._connection() as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM schema_migrations WHERE component = ?",
                        ("memory_manager",),
                    ).fetchone()
                self.assertIsNotNone(row)
                self.assertGreaterEqual(int(row[0]), 1)
            finally:
                mm.close()

    def test_chunk_registry_schema_migrations_are_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ChunkRegistry(db_path=Path(td) / "chunk_registry.sqlite")
            try:
                with registry._connection() as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM schema_migrations WHERE component = ?",
                        ("chunk_registry",),
                    ).fetchone()
                self.assertIsNotNone(row)
                self.assertGreaterEqual(int(row[0]), 1)
            finally:
                registry.close()

    def test_document_uploader_accepts_storage_provider(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "doc.txt"
            source.write_text("hello", encoding="utf-8")
            provider = LocalFileStorageProvider(root / "docs")
            uploader = DocumentUploader(
                storage_dir=root / "unused",
                db_path=root / "documents.sqlite",
                storage_provider=provider,
            )
            try:
                ok = uploader.upload_document(str(source), title="Doc")
                self.assertTrue(ok)
                self.assertTrue(any(str(provider.root) in path for path in uploader.get_all_paths()))
            finally:
                uploader.close()

    def test_multilingual_tokenization_keeps_unicode_words(self):
        tokens = tokenize_for_matching("Café 数据 こんにちは memory", min_len=1)
        self.assertIn("café", tokens)
        self.assertIn("数据", tokens)
        self.assertIn("こんにちは", tokens)

    def test_followup_recall_uses_centralized_limit(self):
        memory = _MemoryStub()
        _ = app._recover_followup_docs_from_memory(memory, "sess1", _FakeUploader())
        self.assertEqual(memory.limit_seen, MEMORY_RECALL_LOOKBACK_ASSISTANT_TURNS)

    def test_async_hybrid_uses_configured_worker_count(self):
        docs_a = [type("Doc", (), {"metadata": {"source": "a", "page": 0}, "page_content": "x"})()]
        docs_b = [type("Doc", (), {"metadata": {"source": "b", "page": 1}, "page_content": "y"})()]
        retriever = rag_pipeline.AsyncHybridRetriever(
            bm25_retriever=_TinyRetriever(docs_a),
            vector_retriever=_TinyRetriever(docs_b),
            execution_mode="thread",
        )
        original_pool = rag_pipeline.ThreadPoolExecutor
        try:
            rag_pipeline.ThreadPoolExecutor = _CapturedPool
            retriever.invoke("query")
        finally:
            rag_pipeline.ThreadPoolExecutor = original_pool
        self.assertEqual(_CapturedPool.last_workers, ASYNC_RETRIEVER_MAX_WORKERS)

    def test_structured_logging_writes_event(self):
        logger = get_logger("tests.tier3")
        marker = "tier3_logging_probe"
        logger.info(marker, detail="ok")
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        self.assertTrue(Path(LOG_PATH).exists())
        content = Path(LOG_PATH).read_text(encoding="utf-8")
        self.assertIn(marker, content)

    def test_splade_batch_scales_with_gpu_vram(self):
        with patch.object(rag_pipeline, "get_model_kwargs", return_value={"device": "cuda"}), patch.object(
            rag_pipeline, "_get_available_vram_gb", return_value=24.0
        ):
            batch = rag_pipeline.get_optimal_splade_batch_size()
        self.assertGreaterEqual(batch, 64)

    def test_splade_batch_caps_for_low_vram(self):
        with patch.object(rag_pipeline, "get_model_kwargs", return_value={"device": "cuda"}), patch.object(
            rag_pipeline, "_get_available_vram_gb", return_value=4.0
        ):
            batch = rag_pipeline.get_optimal_splade_batch_size()
        self.assertLessEqual(batch, rag_pipeline.SPLADE_LOW_VRAM_BATCH_CAP)

    def test_memory_compose_input_truncates_to_budget(self):
        with tempfile.TemporaryDirectory() as td:
            mm = HierarchicalMemoryManager(Path(td) / "memory.sqlite")
            try:
                session_id = mm.start_session(["doc.txt"])
                long_memory = " ".join(["memory"] * 4000)
                with patch.object(mm, "build_memory_context", return_value=long_memory), patch.object(
                    memory_manager, "PROMPT_TOTAL_TOKEN_BUDGET", 512
                ), patch.object(
                    memory_manager, "PROMPT_DOCS_RATIO", 0.60
                ), patch.object(
                    memory_manager, "PROMPT_MEMORY_RATIO", 0.30
                ):
                    payload = mm.compose_model_input(session_id, "What is RAG?")
                token_count = memory_manager._estimate_token_count(payload)
                self.assertLess(token_count, 700)
                self.assertIn("Current question:", payload)
                self.assertIn("What is RAG?", payload)
            finally:
                mm.close()

    def test_upload_refuses_pdf_when_dependency_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")
            uploader = DocumentUploader(storage_dir=root / "docs", db_path=root / "documents.sqlite")
            try:
                with patch.object(document_manager, "PDF_SUPPORT", False):
                    ok = uploader.upload_document(str(pdf_path), title="Sample PDF")
                self.assertFalse(ok)
            finally:
                uploader.close()

    def test_upload_path_resolution_sanitizes_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f = root / "doc.txt"
            f.write_text("ok", encoding="utf-8")
            resolved, err = app._resolve_upload_path(f"  \"{f}\"  ")
            self.assertIsNone(err)
            self.assertEqual(resolved, f.resolve())


if __name__ == "__main__":
    unittest.main()
