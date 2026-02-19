import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


from contextforge.rag_pipeline import TopicBoundedRetriever


class _FakeBaseRetriever:
    def __init__(self, docs):
        self.docs = list(docs)
        self.calls = 0

    def invoke(self, query):
        self.calls += 1
        return list(self.docs)


class TestTopicBoundedRetriever(unittest.TestCase):
    def test_filters_to_forward_logical_page_window(self):
        docs = [
            Document(page_content="memory mention", metadata={"source": "book.pdf", "page": 24, "logical_page": 25}),
            Document(page_content="hierarchical memory core concept", metadata={"source": "book.pdf", "page": 25, "logical_page": 26}),
            Document(page_content="hierarchical memory and long-term memory", metadata={"source": "book.pdf", "page": 26, "logical_page": 27}),
            Document(page_content="hierarchical memory implementation details", metadata={"source": "book.pdf", "page": 27, "logical_page": 28}),
            Document(page_content="hierarchical memory advanced note", metadata={"source": "book.pdf", "page": 28, "logical_page": 29}),
        ]
        wrapped = TopicBoundedRetriever(base_retriever=_FakeBaseRetriever(docs), forward_window_pages=3)
        filtered = wrapped.invoke("What is hierarchical memory?")
        pages = [int(doc.metadata.get("logical_page")) for doc in filtered]
        self.assertNotIn(25, pages)
        self.assertTrue(all(26 <= p <= 29 for p in pages))

    def test_falls_back_when_pages_missing(self):
        docs = [
            Document(page_content="some text", metadata={"source": "book.pdf"}),
            Document(page_content="some more text", metadata={"source": "book.pdf"}),
        ]
        wrapped = TopicBoundedRetriever(base_retriever=_FakeBaseRetriever(docs), forward_window_pages=3)
        filtered = wrapped.invoke("What is MCP?")
        self.assertEqual(len(filtered), 2)

    def test_ignores_toc_anchor_for_general_queries(self):
        docs = [
            Document(
                page_content=(
                    "17. RAG 73 18. CHOOSING A VECTOR DATABASE 75 "
                    "19. SETTING UP YOUR RAG PIPELINE 77 Chunking 77 Embedding 78 Upsert 78"
                ),
                metadata={"source": "book.pdf", "page": 6, "logical_page": 7},
            ),
            Document(
                page_content=(
                    "Alternatives to RAG and detailed setup tradeoffs for retrieval augmented generation "
                    "pipelines in production systems."
                ),
                metadata={"source": "book.pdf", "page": 97, "logical_page": 98},
            ),
            Document(
                page_content=(
                    "How to setup a retrieval augmented generation pipeline with indexing, querying, "
                    "reranking, and monitoring."
                ),
                metadata={"source": "book.pdf", "page": 98, "logical_page": 99},
            ),
        ]
        wrapped = TopicBoundedRetriever(base_retriever=_FakeBaseRetriever(docs), forward_window_pages=3)
        filtered = wrapped.invoke("How to setup a retrieval augmented generation pipeline")
        pages = [int(doc.metadata.get("logical_page")) for doc in filtered if doc.metadata.get("logical_page") is not None]
        self.assertTrue(any(p >= 98 for p in pages))
        self.assertNotIn(7, pages)


if __name__ == "__main__":
    unittest.main()
