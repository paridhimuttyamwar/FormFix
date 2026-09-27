import io
import os
import re
from typing import Tuple
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    from services.ocr_service import extract_text_from_image
except ImportError:
    from ocr_service import extract_text_from_image

class DocumentProcessingError(Exception):
    """Custom exception for user-facing document processing errors."""
    pass

def clean_whitespace(text: str) -> str:
    """Normalize whitespace and collapse multiple linebreaks to max 2."""
    if not text:
        return ""
    # Normalize tabs and multiple spaces to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip leading/trailing space from each line
    lines = [line.strip() for line in text.splitlines()]
    joined = "\n".join(lines)
    # Collapse 3 or more consecutive newlines into 2 (one blank line between paragraphs)
    cleaned = re.sub(r'\n{3,}', '\n\n', joined)
    return cleaned.strip()

def process_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF using PyMuPDF.
    If extracted text is insufficient, fallback to page-by-page OCR.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise DocumentProcessingError(f"Failed to read PDF file: {str(e)}")

    if len(doc) == 0:
        raise DocumentProcessingError("The uploaded PDF has no pages.")

    extracted_pages = []
    total_text_length = 0

    # First pass: try direct text extraction
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text") or ""
        extracted_pages.append(page_text)
        total_text_length += len(page_text.strip())

    # Heuristic: if text per page averages less than 40 chars, it's likely a scanned PDF
    is_scanned = (total_text_length / len(doc)) < 40

    if is_scanned:
        # Fallback to OCR for scanned pages
        ocr_pages = []
        ocr_error_encountered = False
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_ocr_text = extract_text_from_image(img)
                if page_ocr_text:
                    ocr_pages.append(page_ocr_text)
            except RuntimeError:
                ocr_error_encountered = True
                break
            except Exception:
                continue

        if ocr_pages:
            combined_text = "\n\n".join(ocr_pages)
            cleaned = clean_whitespace(combined_text)
            if cleaned:
                return cleaned

        if ocr_error_encountered and total_text_length == 0:
            raise DocumentProcessingError(
                "This appears to be a scanned document, but Tesseract OCR is not available. "
                "Please install Tesseract OCR or upload a digital PDF with selectable text."
            )

    combined_text = "\n\n".join(extracted_pages)
    cleaned = clean_whitespace(combined_text)

    if not cleaned:
        raise DocumentProcessingError("We couldn’t find readable text in this document.")

    return cleaned

def process_image(file_bytes: bytes) -> str:
    """Extract text from an uploaded image using OCR."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise DocumentProcessingError("Unsupported or corrupted image file.")

    try:
        text = extract_text_from_image(image)
        cleaned = clean_whitespace(text)
        if not cleaned:
            raise DocumentProcessingError("We couldn’t read this image clearly. Please upload a clearer document.")
        return cleaned
    except RuntimeError as e:
        raise DocumentProcessingError(str(e))
    except Exception:
        raise DocumentProcessingError("We couldn’t read this image clearly. Please upload a clearer document.")

def extract_document_text(file_name: str, file_bytes: bytes) -> str:
    """
    Main entry point for document text extraction.
    Detects file extension and routes to appropriate processor.
    """
    if not file_name or not file_bytes:
        raise DocumentProcessingError("No file or content was provided.")

    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == ".pdf":
        return process_pdf(file_bytes)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return process_image(file_bytes)
    else:
        raise DocumentProcessingError("Unsupported file type. Please upload PDF, JPG or PNG.")
