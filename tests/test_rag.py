import os
import unittest

from src.manual_processor import (
    load_pdf,
    process_manual
)

from src.rag_retriever import (
    retrieve_context,
    initialize_rag
)


class TestMember2RAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_path = os.path.join(
            "data",
            "manuals",
            "isuzu_f_series_service_manual.pdf"
        )

        # Initialize once for the entire test suite.
        initialize_rag(cls.pdf_path)

    def test_pdf_exists_and_readable(self):
        self.assertTrue(
            os.path.exists(self.pdf_path),
            "PDF Manual path does not exist!"
        )

        reader = load_pdf(self.pdf_path)

        self.assertGreater(
            len(reader.pages),
            0,
            "PDF manual is empty!"
        )

    def test_manual_processing(self):
        chunks = process_manual(self.pdf_path)

        self.assertGreater(
            len(chunks),
            0,
            "Chunking pipeline returned 0 chunks!"
        )

        self.assertIn(
            "page",
            chunks[0]
        )

        self.assertIn(
            "text",
            chunks[0]
        )

        # Poor tiny fragments should no longer exist.
        shortest_chunk = min(
            len(chunk["text"])
            for chunk in chunks
        )

        self.assertGreaterEqual(
            shortest_chunk,
            120,
            "Tiny/useless manual chunks were found."
        )

    def test_charging_system_retrieval(self):
        result = retrieve_context(
            "charging system undercharge"
        )

        self.assertGreater(
            len(result["sources"]),
            0,
            "No relevant charging-system sources retrieved."
        )

        combined_text = result[
            "context"
        ].lower()

        self.assertTrue(
            (
                "undercharge" in combined_text
                or
                "charging" in combined_text
                or
                "generator" in combined_text
            ),
            "Charging query did not retrieve useful charging information."
        )

    def test_low_oil_pressure_retrieval(self):
        result = retrieve_context(
            "low oil pressure"
        )

        self.assertGreater(
            len(result["sources"]),
            0,
            "No relevant oil-pressure source retrieved."
        )

        combined_text = result[
            "context"
        ].lower()

        self.assertTrue(
            (
                "oil" in combined_text
                or
                "pressure" in combined_text
                or
                "lubrication" in combined_text
            ),
            "Oil-pressure query returned irrelevant context."
        )

    def test_sources_have_page_metadata(self):
        result = retrieve_context(
            "charging system undercharge"
        )

        self.assertGreater(
            len(result["sources"]),
            0
        )

        for source in result["sources"]:
            self.assertIn(
                "page",
                source
            )

            self.assertIn(
                "text",
                source
            )

            self.assertIn(
                "similarity",
                source
            )

    def test_empty_query(self):
        result = retrieve_context("")

        self.assertEqual(
            result["context"],
            "Not found in retrieved manual context."
        )

        self.assertEqual(
            result["sources"],
            []
        )

    def test_unrelated_query_is_rejected(self):
        result = retrieve_context(
            "What is the capital of France?"
        )

        self.assertEqual(
            result["context"],
            "Not found in retrieved manual context."
        )

        self.assertEqual(
            result["sources"],
            []
        )


if __name__ == "__main__":
    unittest.main()