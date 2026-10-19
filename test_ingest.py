import tempfile
import unittest
from pathlib import Path

import pymupdf

from modules.app import MindGateEngine


class DocumentIngestionTests(unittest.TestCase):
    def test_pdf_ingestion_cache_and_local_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "uploads" / "policy.pdf"
            source.parent.mkdir()
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text(
                    (72, 72),
                    "Refund requests are accepted within fourteen calendar days.",
                )
                document.save(source)

            engine = MindGateEngine(workspace_dir=str(workspace))
            first_ingestion = engine.ingest_document(str(source))
            cached_ingestion = engine.ingest_document(str(source))
            retrieved = engine.retrieve_relevant_chunks(
                "fourteen calendar days", active_doc="policy.pdf"
            )
            engine.vector_store.close()

        self.assertFalse(first_ingestion["cache_hit"])
        self.assertTrue(cached_ingestion["cache_hit"])
        self.assertEqual(first_ingestion["total_chunks_indexed"], 1)
        self.assertEqual([chunk["metadata"]["file_name"] for chunk in retrieved], ["policy.pdf"])
        self.assertIn("fourteen calendar days", retrieved[0]["text"])


if __name__ == "__main__":
    unittest.main()
