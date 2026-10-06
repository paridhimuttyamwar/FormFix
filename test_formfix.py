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

    def test_ml_authenticity_classifier(self):
        """Test ML Model 1: Authenticity and Scam Classifier (NLP Logistic Regression)."""
        from services.ml_service import NoticeAuthenticityModel
        model = NoticeAuthenticityModel()

        # Official text test
        official_text = (
            "GOVERNMENT NOTIFICATION Ref: DHE/SCH/2026/894. "
            "In exercise of powers, eligible candidates are notified to submit application "
            "for scholarship renewal through the official government portal."
        )
        res_official = model.predict(official_text)
        self.assertTrue(res_official["is_authentic"])
        self.assertGreaterEqual(res_official["authenticity_score"], 50.0)

        # Scam text test
        scam_text = (
            "URGENT!! Win free laptop and ₹50,000 cash! Transfer ₹499 registration fee "
            "via Google Pay UPI immediately. Join our Telegram group now!"
        )
        res_scam = model.predict(scam_text)
        self.assertFalse(res_scam["is_authentic"])
        self.assertGreaterEqual(res_scam["fraud_risk_score"], 60.0)
        print("  [PASS] ML Model 1 (Authenticity & Scam Logistic Regression) passed.")

    def test_ml_urgency_risk_model(self):
        """Test ML Model 2: Urgency & Disqualification Risk (Tabular Logistic Regression)."""
        from services.ml_service import UrgencyRiskModel
        urgency_model = UrgencyRiskModel()

        # Critical urgency scenario: 2 days left, 5 documents, strict rejection penalty, offline submission
        crit_res = urgency_model.predict(
            deadline_text="2 days remaining",
            required_docs=["Income Certificate", "Caste Certificate", "Bonafide", "Marksheet", "Affidavit"],
            action_items=["Get notary", "Submit offline", "Pay fee", "Upload scan", "Send speed post"],
            full_text="Hard copy must be sent by speed post. Late submissions will be summarily rejected."
        )
        self.assertGreaterEqual(crit_res["urgency_score"], 60.0)
        self.assertIn("Critical", crit_res["urgency_level"])

        # Low urgency scenario: 45 days left, 1 document, online only
        low_res = urgency_model.predict(
            deadline_text="30 November 2026",
            required_docs=["College ID card"],
            action_items=["Upload ID online"],
            full_text="Online submission open on the university portal. Ample time provided."
        )
        self.assertLessEqual(low_res["urgency_score"], 50.0)
        print("  [PASS] ML Model 2 (Urgency & Risk Logistic Regression) passed.")

    def test_ml_full_assessment(self):
        """Test full dual-model assessment function."""
        from services.ml_service import assess_document_ml
        sample_text = "STATE UNIVERSITY Memo No: 124. Deadline: 15 October 2026. Required Documents: Marksheet."
        sample_analysis = {
            "deadline": "15 October 2026",
            "required_documents": ["Marksheet"],
            "action_items": ["Collect marksheet", "Submit online"]
        }
        assessment = assess_document_ml(sample_text, sample_analysis)
        self.assertIn("authenticity", assessment)
        self.assertIn("urgency", assessment)
        self.assertIn("authenticity_score", assessment["authenticity"])
        self.assertIn("urgency_score", assessment["urgency"])
        print("  [PASS] ML Full Assessment pipeline passed.")

if __name__ == "__main__":
    unittest.main()

