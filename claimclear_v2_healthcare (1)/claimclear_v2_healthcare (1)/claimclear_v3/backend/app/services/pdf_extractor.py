"""
PDF text extraction with OCR fallback for scanned insurance documents.
Uses PyMuPDF for native text (fast), falls back to pytesseract OCR.
Results are in-memory cached to avoid re-processing.
"""
from __future__ import annotations
import io
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_TEXT_CACHE: dict[str, str] = {}


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 60) -> str:
    """
    Extract full text from a PDF.
    - Tries native PyMuPDF text first (fast, accurate for digital PDFs)
    - Falls back to pytesseract OCR for scanned/image PDFs
    - Caches result in memory
    """
    key = str(pdf_path.resolve())
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]

    logger.info(f"Extracting text from: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    pages_to_process = min(len(doc), max_pages)
    parts: list[str] = []

    for page_num in range(pages_to_process):
        page = doc[page_num]
        native = page.get_text("text").strip()

        if len(native) > 80:
            parts.append(f"[Page {page_num + 1}]\n{native}")
        else:
            # Scanned page — use OCR
            try:
                import pytesseract
                from PIL import Image

                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                if ocr_text.strip():
                    parts.append(f"[Page {page_num + 1}]\n{ocr_text.strip()}")
            except Exception as e:
                logger.warning(f"OCR failed on page {page_num + 1}: {e}")

    doc.close()
    result = "\n\n".join(parts)
    _TEXT_CACHE[key] = result
    logger.info(f"Extracted {len(result):,} chars from {pdf_path.name} ({pages_to_process} pages)")
    return result


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks suitable for embedding.
    Tries to split on paragraph/sentence boundaries.
    """
    if not text.strip():
        return []

    # Split by page markers first, then paragraph
    pages = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in pages:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from previous
            if current and overlap > 0:
                overlap_text = current[-overlap:]
                current = overlap_text + "\n\n" + para
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    # Further split any oversized chunks
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:
            final_chunks.append(chunk)
        else:
            sentences = chunk.replace(". ", ".\n").split("\n")
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) < chunk_size:
                    sub = (sub + " " + sent).strip()
                else:
                    if sub:
                        final_chunks.append(sub)
                    sub = sent
            if sub:
                final_chunks.append(sub)

    return [c for c in final_chunks if len(c) > 50]


def get_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def clear_cache() -> None:
    _TEXT_CACHE.clear()
