import io
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


from contextforge import app
from contextforge import rag_pipeline
from contextforge.document_manager import DocumentUploader
from contextforge.memory_manager import _bounded_regex_input


class _DummyRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = 0

    def invoke(self, _query):
        self.calls += 1
        return self.docs


class _BrokenStreamingDocChain:
    def __init__(self, fallback_answer: str):
        self.stream_calls = 0
        self.invoke_calls = 0
        self.fallback_answer = fallback_answer

    def stream(self, _payload):
        self.stream_calls += 1
        raise RuntimeError("simulated stream disconnect")
        yield  # pragma: no cover

    def invoke(self, _payload):
        self.invoke_calls += 1
        return self.fallback_answer


class _StalledStreamingDocChain:
    def __init__(self, fallback_answer: str, stall_s: float = 0.4):
        self.stream_calls = 0
        self.invoke_calls = 0
        self.fallback_answer = fallback_answer
        self.stall_s = float(stall_s)

    def stream(self, _payload):
        self.stream_calls += 1
        time.sleep(self.stall_s)
        if False:
            yield "never"

    def invoke(self, _payload):
        self.invoke_calls += 1
        return self.fallback_answer


class _FakeUploader:
    def __init__(self, paths=None):
        self.paths = list(paths or [])

    def get_all_paths(self):
        return list(self.paths)


class TestBottleneckScalabilitySecurity(unittest.TestCase):
    def test_rrf_smoothing_favors_consistent_cross_retriever_hits(self):
        doc_x = Document(page_content="X", metadata={"source": "s", "page": 0})
        doc_y = Document(page_content="Y", metadata={"source": "s", "page": 1})
        filler = [
            Document(page_content=f"F{i}", metadata={"source": "s", "page": i + 2})
            for i in range(6)
        ]
        extra = Document(page_content="Z", metadata={"source": "s", "page": 99})
        bm25_docs = [doc_x, filler[0], filler[1], filler[2], doc_y]
        vector_docs = [filler[3], filler[4], filler[5], extra, doc_y]
        retriever = rag_pipeline.AsyncHybridRetriever(
            bm25_retriever=None,
            vector_retriever=None,
            rrf_k=60,
        )
        ranked = retriever._rank_fuse(bm25_docs, vector_docs)
        # With standard RRF smoothing, a doc appearing in both lists should outrank a single-list top hit.
        self.assertEqual(ranked[0].page_content, "Y")

    def test_stream_failure_falls_back_without_hang(self):
        docs = [
            Document(
                page_content="Hierarchical memory combines short- and long-term context.",
                metadata={"source": "book.pdf", "page": 0},
            )
        ]
        retriever = _DummyRetriever(docs)
        chain = _BrokenStreamingDocChain(
            (
                "Hierarchical memory combines recent session state with relevant long-term signals so answers stay "
                "consistent across turns. For example, it can retain a user's project constraints while answering "
                "a new implementation question."
            )
        )
        uploader = _FakeUploader(paths=["book.pdf"])

        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            app.handle_user_query(
                "What is hierarchical memory?",
                retriever,
                chain,
                uploader,
                memory_manager=None,
                session_id=None,
                grounded=True,
            )
        elapsed = time.perf_counter() - start
        self.assertEqual(chain.stream_calls, 1)
        self.assertGreaterEqual(chain.invoke_calls, 1)
        self.assertLess(elapsed, 5.0)

    def test_stream_stall_timeout_falls_back_without_hang(self):
        docs = [
            Document(
                page_content="Hierarchical memory combines short- and long-term context.",
                metadata={"source": "book.pdf", "page": 0},
            )
        ]
        retriever = _DummyRetriever(docs)
        chain = _StalledStreamingDocChain(
            "Fallback answer with enough depth and one practical example for stable output.",
            stall_s=0.35,
        )
        uploader = _FakeUploader(paths=["book.pdf"])

        with patch.object(app, "STREAM_IDLE_TIMEOUT_S", 0.1), patch.object(app, "STREAM_POLL_INTERVAL_S", 0.05):
            start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                app.handle_user_query(
                    "What is hierarchical memory?",
                    retriever,
                    chain,
                    uploader,
                    memory_manager=None,
                    session_id=None,
                    grounded=True,
                )
            elapsed = time.perf_counter() - start
        self.assertEqual(chain.stream_calls, 1)
        self.assertGreaterEqual(chain.invoke_calls, 1)
        self.assertLess(elapsed, 5.0)

    def test_document_uploader_uses_sqlite_atomic_updates(self):
        with tempfile.TemporaryDirectory() as td:
            storage = Path(td) / "workspace_docs"
            storage.mkdir(parents=True, exist_ok=True)
            upload_source = Path(td) / "sample.txt"
            upload_source.write_text("hello world", encoding="utf-8")

            uploader = DocumentUploader(storage_dir=storage, db_path=Path(td) / "documents.sqlite")
            try:
                ok = uploader.upload_document(str(upload_source), title="Sample")
                self.assertTrue(ok)
                all_paths = uploader.get_all_paths()
                self.assertEqual(len(all_paths), 1)
                uploader.mark_as_indexed(all_paths[0])
                self.assertEqual(uploader.get_unindexed_paths(), [])
                self.assertTrue(uploader.db_path.exists())
            finally:
                uploader.close()

    def test_splade_cache_uses_json_npz_not_pickle(self):
        if rag_pipeline.sparse is None:
            self.skipTest("scipy sparse unavailable")

        matrix = rag_pipeline.sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
        docs = [
            Document(page_content="doc1", metadata={"source": "a", "page": 0}),
            Document(page_content="doc2", metadata={"source": "a", "page": 1}),
        ]
        retriever = rag_pipeline.SpladeRetriever(
            documents=docs,
            tokenizer=object(),
            model=object(),
            doc_matrix=matrix,
            device="cpu",
            k=2,
        )
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splade_cache_unit"
            saved = retriever.save(base)
            self.assertTrue(saved)
            self.assertTrue(base.with_suffix(".json").exists())
            self.assertTrue(base.with_suffix(".npz").exists())
            self.assertFalse(base.with_suffix(".pkl").exists())
            with patch.object(rag_pipeline, "get_splade_components", return_value=(object(), object(), "cpu")):
                loaded = rag_pipeline.SpladeRetriever.load(base, k=2)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.documents), 2)

    def test_regex_input_bounding_limits_untrusted_payload_size(self):
        long_text = "a" * 10000
        bounded = _bounded_regex_input(long_text)
        self.assertLessEqual(len(bounded), 2048)

    def test_unload_hybrid_components_clears_global_references(self):
        class _Model:
            def __init__(self):
                self.moved_to_cpu = False

            def to(self, device):
                if str(device) == "cpu":
                    self.moved_to_cpu = True
                return self

        emb = _Model()
        rer = _Model()
        with patch.object(rag_pipeline, "_EMBEDDING_MODEL", emb), patch.object(rag_pipeline, "_RERANKER_MODEL", rer):
            rag_pipeline.unload_hybrid_components()
            self.assertTrue(emb.moved_to_cpu)
            self.assertTrue(rer.moved_to_cpu)
            self.assertIsNone(rag_pipeline._EMBEDDING_MODEL)
            self.assertIsNone(rag_pipeline._RERANKER_MODEL)


if __name__ == "__main__":
    unittest.main()
