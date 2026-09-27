import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.document_processor import extract_document_text, clean_whitespace, DocumentProcessingError
from services.ocr_service import preprocess_image
from services.ai_service import (
    get_demo_analysis,
    is_groq_configured,
    clean_json_response
)
from PIL import Image

class TestFormFixAI(unittest.TestCase):

    def test_digital_pdf_text_extraction(self):
        """Test 1: Normal PDF text extraction with PyMuPDF."""
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        test_content = (
            "STATE DEPARTMENT OF HIGHER EDUCATION\n"
            "NOTIFICATION: SCHOLARSHIP RENEWAL APPLICATION (2026-2027)\n"
            "Deadline: 15 October 2026\n"
            "Required Documents: Bonafide Certificate, Income Certificate, Marksheet\n"
        )
        page.insert_text((50, 72), test_content)
        pdf_bytes = doc.tobytes()

        extracted_text = extract_document_text("test_notice.pdf", pdf_bytes)
        self.assertIsInstance(extracted_text, str)
        self.assertIn("SCHOLARSHIP RENEWAL APPLICATION", extracted_text)
        self.assertIn("15 October 2026", extracted_text)
        self.assertIn("Bonafide Certificate", extracted_text)
        self.assertIn("Income Certificate", extracted_text)
        print("  [PASS] Test 1: Digital PDF extracted successfully.")

    def test_whitespace_cleaner(self):
        """Test whitespace normalization."""
        raw = "Line 1   with    extra    spaces\n\n\n\n\nLine 2\n\n"
        cleaned = clean_whitespace(raw)
        self.assertEqual(cleaned, "Line 1 with extra spaces\n\nLine 2")
        print("  [PASS] Whitespace normalization passed.")

    def test_unsupported_file_type(self):
        """Test unsupported file type rejection."""
        with self.assertRaises(DocumentProcessingError) as ctx:
            extract_document_text("test.docx", b"some binary data")
        self.assertIn("Unsupported file type", str(ctx.exception))
        print("  [PASS] Unsupported file handling passed.")

    def test_empty_file_handling(self):
        """Test empty content handling."""
        with self.assertRaises(DocumentProcessingError):
            extract_document_text("empty.pdf", b"")
        print("  [PASS] Empty document handling passed.")

    def test_ocr_image_preprocessing(self):
        """Test OCR image preprocessor converts to grayscale and enhances."""
        img = Image.new("RGB", (200, 200), color="blue")
        processed = preprocess_image(img)
        self.assertEqual(processed.mode, "L")
        print("  [PASS] OCR image preprocessing passed.")

    def test_demo_analysis_schema(self):
        """Test 4, 5, 6: Verify analysis schema meets all hackathon specifications."""
        demo_data = get_demo_analysis()
        expected_keys = [
            "document_title",
            "document_type",
            "summary",
            "deadline",
            "required_documents",
            "important_fields",
            "action_items",
            "warnings"
        ]
        for key in expected_keys:
            self.assertIn(key, demo_data, f"Key '{key}' must be present in AI analysis.")

        # Test Deadline
        self.assertEqual(demo_data["deadline"], "15 October 2026")
        
        # Test Required Documents
        self.assertIsInstance(demo_data["required_documents"], list)
        self.assertGreaterEqual(len(demo_data["required_documents"]), 3)

        # Test Important Fields structure
        for field in demo_data["important_fields"]:
            self.assertIn("field", field)
            self.assertIn("explanation", field)
            self.assertIn("required_document", field)

        # Test Action items
        self.assertIsInstance(demo_data["action_items"], list)
        self.assertGreaterEqual(len(demo_data["action_items"]), 4)

        print("  [PASS] Structured schema & Action Plan passed.")

    def test_json_cleaner(self):
        """Test markdown code fence stripping from LLM responses."""
        fenced = "```json\n{\"document_title\": \"Test\"}\n```"
        cleaned = clean_json_response(fenced)
        self.assertEqual(cleaned, '{"document_title": "Test"}')
        print("  [PASS] JSON fence cleaner passed.")

if __name__ == "__main__":
    unittest.main()
