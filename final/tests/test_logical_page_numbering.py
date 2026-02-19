import sys
import unittest
from pathlib import Path
from unittest.mock import patch


from contextforge.rag_pipeline import _extract_logical_page_number
from contextforge import rag_pipeline


class _FakeRect:
    def __init__(self, height: float):
        self.height = height


class _FakePage:
    def __init__(self, number: int, label: str = "", blocks=None, height: float = 1000.0):
        self.number = number
        self._label = label
        self._blocks = list(blocks or [])
        self.rect = _FakeRect(height)

    def get_label(self):
        return self._label

    def get_text(self, mode: str):
        if mode == "blocks":
            return self._blocks
        if mode == "text":
            return ""
        raise ValueError("Unsupported mode for fake page.")


class _FakePdfDoc:
    def __init__(self, pages):
        self._pages = list(pages)

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        return None


class TestLogicalPageNumbering(unittest.TestCase):
    def test_uses_pdf_label_when_distinct_from_physical(self):
        page = _FakePage(number=40, label="26", blocks=[])
        self.assertEqual(_extract_logical_page_number(page), 26)

    def test_uses_footer_heuristic_when_label_missing(self):
        # block tuple format: (x0, y0, x1, y1, text, ...)
        blocks = [
            (10, 920, 200, 940, "Page 26", 0, 0),
        ]
        page = _FakePage(number=40, label="", blocks=blocks, height=1000.0)
        self.assertEqual(_extract_logical_page_number(page), 26)

    def test_uses_header_heuristic_for_running_header(self):
        blocks = [
            (10, 20, 300, 40, "26\nSAM BHAGWAT\n", 0, 0),
        ]
        page = _FakePage(number=40, label="", blocks=blocks, height=1000.0)
        self.assertEqual(_extract_logical_page_number(page), 26)

    def test_falls_back_to_physical_page_when_no_signals(self):
        page = _FakePage(number=7, label="", blocks=[], height=1000.0)
        self.assertEqual(_extract_logical_page_number(page), 8)

    def test_load_pdf_docs_locks_offset_and_rejects_outliers(self):
        # Physical pages are 21..25 here, so a logical extraction of 7 on page 23
        # produces a sane positive offset of 16 that should be locked.
        fake_pages = [_FakePage(number=i, label="", blocks=[], height=1000.0) for i in range(20, 25)]
        extracted_sequence = [21, 22, 7, 8, 99]

        with patch.object(rag_pipeline, "fitz") as mock_fitz, patch.object(
            rag_pipeline, "_extract_logical_page_number", side_effect=extracted_sequence
        ):
            mock_fitz.open.return_value = _FakePdfDoc(fake_pages)
            docs = rag_pipeline._load_pdf_docs_with_logical_pages(Path("dummy.pdf"))

        logical_pages = [doc.metadata.get("logical_page") for doc in docs]
        offsets = [doc.metadata.get("logical_page_offset") for doc in docs]
        self.assertEqual(logical_pages, [21, 22, 7, 8, 9])
        self.assertTrue(all(offset == 16 for offset in offsets))

    def test_load_pdf_docs_rejects_implausible_initial_offset(self):
        fake_pages = [_FakePage(number=i, label="", blocks=[], height=1000.0) for i in range(5)]
        extracted_sequence = [1, 2, 26, 27, 28]

        with patch.object(rag_pipeline, "fitz") as mock_fitz, patch.object(
            rag_pipeline, "_extract_logical_page_number", side_effect=extracted_sequence
        ):
            mock_fitz.open.return_value = _FakePdfDoc(fake_pages)
            docs = rag_pipeline._load_pdf_docs_with_logical_pages(Path("dummy.pdf"))

        logical_pages = [doc.metadata.get("logical_page") for doc in docs]
        offsets = [doc.metadata.get("logical_page_offset") for doc in docs]
        self.assertEqual(logical_pages, [1, 2, 3, 4, 5])
        self.assertTrue(all(offset is None for offset in offsets))


if __name__ == "__main__":
    unittest.main()
