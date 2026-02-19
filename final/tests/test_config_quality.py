import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


from contextforge import app
from contextforge import rag_pipeline
from contextforge.config import CHUNK_OVERLAP, CHUNK_SIZE
from contextforge.document_manager import ChunkRegistry


class _Status:
    def update(self, *_args, **_kwargs):
        return None


class TestBottleneckQualityAndConfig(unittest.TestCase):
    def test_check_dependencies_splade_reports_missing_runtime(self):
        with patch.object(rag_pipeline, "torch", None):
            ok, errors = rag_pipeline.check_dependencies("splade", ["doc.txt"])
        self.assertFalse(ok)
        self.assertTrue(any("torch" in str(err).lower() for err in errors))

    def test_structured_pdf_page_labels_parser_prefers_numeric_labels(self):
        fake_reader = type("FakeReader", (), {"page_labels": ["i", "ii", "1", "2", "Page 3"]})
        with patch.object(rag_pipeline, "PdfReader", return_value=fake_reader()):
            labels = rag_pipeline._read_structured_pdf_page_labels(Path("dummy.pdf"))
        self.assertEqual(labels.get(3), 1)
        self.assertEqual(labels.get(4), 2)
        self.assertEqual(labels.get(5), 3)

    def test_chapter_context_uses_registry_before_disk_reload(self):
        chapter_docs = [
            Document(
                page_content="1 INTRODUCTION\nThis chapter starts here.",
                metadata={"source": "book.pdf", "page": 0},
            ),
            Document(
                page_content="More introduction details.",
                metadata={"source": "book.pdf", "page": 1},
            ),
            Document(
                page_content="2 NEXT CHAPTER\nAnother section begins.",
                metadata={"source": "book.pdf", "page": 2},
            ),
        ]
        entries = [
            {"chapter_number": 1, "title": "INTRODUCTION", "source": "book.pdf", "toc_doc_page": 0},
            {"chapter_number": 2, "title": "NEXT CHAPTER", "source": "book.pdf", "toc_doc_page": 0},
        ]
        target = entries[0]
        with patch.object(app.CHUNK_REGISTRY, "get_source_documents", return_value=chapter_docs), patch.object(
            app, "_load_document_pages", side_effect=AssertionError("disk reload should not run")
        ):
            context = app._get_chapter_context_docs(target, entries)
        self.assertGreaterEqual(len(context), 1)
        joined = " ".join(doc.page_content for doc in context)
        self.assertIn("introduction", joined.lower())

    def test_recursive_chunker_uses_runtime_config_values(self):
        docs = [Document(page_content="A short test doc", metadata={"source": "doc.txt", "page": 0})]
        status = _Status()

        captured = {}

        class _Splitter:
            def __init__(self, chunk_size, chunk_overlap):
                captured["chunk_size"] = chunk_size
                captured["chunk_overlap"] = chunk_overlap

            def split_documents(self, items):
                return items

        with patch.object(rag_pipeline, "USE_SEMANTIC_CHUNKING", False), patch.object(
            rag_pipeline, "_load_docs_for_path", return_value=(docs, None)
        ), patch.object(rag_pipeline, "RecursiveCharacterTextSplitter", _Splitter):
            split_docs = rag_pipeline._load_and_split_docs(["doc.txt"], status, uploader=None)

        self.assertEqual(captured.get("chunk_size"), CHUNK_SIZE)
        self.assertEqual(captured.get("chunk_overlap"), CHUNK_OVERLAP)
        self.assertEqual(len(split_docs), 1)

    def test_chunk_registry_source_documents_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ChunkRegistry(Path(td) / "chunk_registry.sqlite")
            try:
                source = str((Path(td) / "doc.txt").resolve())
                docs = [
                    Document(page_content="Page 1", metadata={"source": source, "page": 0}),
                    Document(page_content="Page 2", metadata={"source": source, "page": 1}),
                ]
                registry.register_chunks(docs)
                loaded = registry.get_source_documents(source)
                self.assertEqual(len(loaded), 2)
                self.assertIn("Page 1", loaded[0].page_content)
            finally:
                registry.close()


if __name__ == "__main__":
    unittest.main()
