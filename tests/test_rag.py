import unittest
import os
from src.rag_retriever import retrieve_context
from src.manual_processor import process_manual, load_pdf

class TestMember2RAG(unittest.TestCase):
    
    def setUp(self):
        self.pdf_path = os.path.join("data", "manuals", "isuzu_f_series_service_manual.pdf")
        
    def test_pdf_exists_and_readable(self):
        self.assertTrue(os.path.exists(self.pdf_path), "PDF Manual path does not exist!")
        reader = load_pdf(self.pdf_path)
        self.assertGreater(len(reader.pages), 0, "PDF manual is empty!")

    def test_manual_processing(self):
        chunks = process_manual(self.pdf_path)
        self.assertGreater(len(chunks), 0, "Chunking pipeline returned 0 chunks!")
        self.assertIn("page", chunks[0], "Chunk is missing 'page' metadata!")
        self.assertIn("text", chunks[0], "Chunk is missing 'text' content!")

    def test_retrieve_context_relevant_query(self):
        result = retrieve_context("low oil pressure")
        self.assertIn("query", result)
        self.assertIn("context", result)
        self.assertIn("sources", result)
        self.assertGreater(len(result["sources"]), 0, "No sources retrieved for oil pressure query!")
        self.assertIn("page", result["sources"][0])

    def test_empty_query(self):
        result = retrieve_context("")
        self.assertEqual(result["context"], "Not found in retrieved manual context.")
        self.assertEqual(len(result["sources"]), 0)

if __name__ == "__main__":
    unittest.main()