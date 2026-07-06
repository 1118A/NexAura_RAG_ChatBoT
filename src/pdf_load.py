import os
import fitz  # PyMuPDF


def load_pdf(pdf_path: str) -> str:
    """
    Load and extract text from a PDF file.

    Security guardrails:
    - Validates the file exists before opening.
    - Uses a finally block to ensure the document handle is always closed.
    - Raises a clear, user-friendly error for image-only (no-text) PDFs.
    - Does NOT expose internal file paths in error messages.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError("The uploaded PDF could not be found. Please re-upload the file.")

    document = None
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page in document:
            text += page.get_text()
    except Exception:
        raise RuntimeError(
            "Failed to read the PDF. The file may be corrupted or password-protected."
        )
    finally:
        if document:
            document.close()

    text = text.strip()
    if not text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image-only document. "
            "Please use a PDF with selectable text."
        )

    return text
