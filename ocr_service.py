import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Common Windows installation locations for Tesseract-OCR
COMMON_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

def _configure_tesseract():
    """Attempt to configure pytesseract with the binary location."""
    # Check if tesseract is already in PATH
    if shutil.which("tesseract"):
        return True
    
    # Check environment variable
    custom_cmd = os.getenv("TESSERACT_CMD")
    if custom_cmd and os.path.isfile(custom_cmd):
        pytesseract.pytesseract.tesseract_cmd = custom_cmd
        return True

    # Search standard paths
    for path in COMMON_TESSERACT_PATHS:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
            
    return False

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Lightweight image preprocessing for OCR:
    Convert to grayscale and enhance contrast for clearer text edges.
    """
    try:
        # Convert to grayscale
        gray = image.convert("L")
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        return enhanced
    except Exception:
        return image

def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from a PIL Image using pytesseract.
    Returns cleaned text string.
    Raises RuntimeError if Tesseract is not available or readable text cannot be extracted.
    """
    is_available = _configure_tesseract()
    if not is_available:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found in system PATH. "
            "For scanned documents or images, please install Tesseract OCR from "
            "https://github.com/UB-Mannheim/tesseract/wiki, or upload a digital PDF."
        )

    try:
        processed = preprocess_image(image)
        try:
            text = pytesseract.image_to_string(processed, lang="eng+hin+mar")
        except Exception:
            text = pytesseract.image_to_string(processed)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Please install Tesseract-OCR to enable image OCR."
        )
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {str(e)}")
