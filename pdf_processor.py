import re
from io import BytesIO
import fitz  # PyMuPDF

CHUNK_WORDS = 180
OVERLAP_WORDS = 35


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract text page by page. Page numbers shown to users are 1-based."""
    pages = []
    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
        if doc.is_encrypted:
            raise ValueError("This PDF is password-protected. Please upload an unlocked PDF.")
        for page_index, page in enumerate(doc):
            text = page.get_text("text")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append({"page": page_index + 1, "text": text})
    return pages


def make_chunks(pages: list[dict], chunk_words: int = CHUNK_WORDS,
                overlap_words: int = OVERLAP_WORDS) -> list[dict]:
    """Split each page independently so citations always point to a real page."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append({"page": page["page"], "text": text})
            if end == len(words):
                break
            start = max(end - overlap_words, start + 1)
    return chunks
