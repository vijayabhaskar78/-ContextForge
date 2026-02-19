import sys
import tempfile
import time
import unittest
from pathlib import Path

import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from contextforge import app
from contextforge.document_manager import CHUNK_REGISTRY
from contextforge.memory_manager import HierarchicalMemoryManager


def _create_pdf(path: Path, page_count: int):
    doc = fitz.open()
    for idx in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {idx + 1}: Hierarchical memory and retrieval performance.")
        page.insert_text((72, 108), f"This is synthetic test content for page {idx + 1}.")
    doc.save(str(path))
    doc.close()


class TestPerformanceOptimizations(unittest.TestCase):
    def test_query_latency_under_2_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "fifty_pages.pdf"
            _create_pdf(pdf_path, page_count=50)

            docs = PyMuPDFLoader(str(pdf_path)).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
            CHUNK_REGISTRY.register_chunks(chunks)

            # Representative retrieval set from different pages.
            retrieved = [chunks[5], chunks[15], chunks[25], chunks[35], chunks[45]]
            start = time.perf_counter()
            expanded = app.expand_context(retrieved)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            self.assertGreaterEqual(len(expanded), len(retrieved))
            self.assertLessEqual(len(expanded), len(retrieved) * 2)
            self.assertLess(elapsed_ms, 50.0, f"expand_context took {elapsed_ms:.1f} ms; expected < 50 ms")
            self.assertLess(elapsed_ms, 2000.0, f"expand_context took {elapsed_ms:.1f} ms; expected < 2 seconds")

    def test_expanded_content_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "integrity.pdf"
            _create_pdf(pdf_path, page_count=8)

            docs = PyMuPDFLoader(str(pdf_path)).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
            CHUNK_REGISTRY.register_chunks(chunks)

            retrieved = [chunks[3]]
            optimized = app.expand_context(retrieved)
            self.assertLessEqual(len(optimized), len(retrieved) * 2)
            seed_page = app._get_display_page_number(retrieved[0])
            self.assertIsNotNone(seed_page)
            self.assertTrue(
                all(
                    app._get_display_page_number(doc) is not None
                    and abs(app._get_display_page_number(doc) - seed_page) <= 2
                    for doc in optimized
                ),
                "Expanded docs must remain within +/-2 pages of seed topic.",
            )
            min_seed_page = min(app._get_display_page_number(doc) for doc in retrieved if app._get_display_page_number(doc) is not None)
            self.assertTrue(
                all(
                    app._get_display_page_number(doc) is not None
                    and app._get_display_page_number(doc) >= min_seed_page
                    for doc in optimized
                ),
                "Expanded docs must not include pages before the seed topic window.",
            )

    def test_memory_retrieval_speed(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "conversation_memory.sqlite"
            mm = HierarchicalMemoryManager(db_path=db_path)
            try:
                session_id = mm.start_session(["perf.pdf"])

                for idx in range(500):
                    mm.record_turn(
                        session_id=session_id,
                        role="user" if idx % 2 == 0 else "assistant",
                        text=(
                            f"Turn {idx} about hierarchical memory retrieval processors "
                            f"and semantic recall behavior in chapter {(idx % 30) + 1}."
                        ),
                        intent="general_qa",
                        topic="hierarchical memory",
                    )

                start = time.perf_counter()
                results = mm.retrieve_episodic_turns(
                    session_id=session_id,
                    query="explain hierarchical memory retrieval processors",
                    limit=5,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                self.assertGreaterEqual(len(results), 1)
                self.assertLess(elapsed_ms, 100.0, f"retrieve_episodic_turns took {elapsed_ms:.1f} ms; expected < 100 ms")
            finally:
                mm.close()


if __name__ == "__main__":
    unittest.main()
