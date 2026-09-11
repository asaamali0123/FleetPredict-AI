import os
import re
from pypdf import PdfReader


def load_pdf(pdf_path: str):
    """Open and return the PDF reader."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF Manual not found at: {pdf_path}")

    return PdfReader(pdf_path)


def clean_text(text: str) -> str:
    """
    Lightly clean extracted PDF text while keeping the original
    manual information intact.
    """
    if not text:
        return ""

    # Replace repeated whitespace/newlines with single spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pages(pdf_path: str):
    """
    Extract readable text from each PDF page.

    Page numbers are preserved so RAG results can cite their source.
    """
    reader = load_pdf(pdf_path)
    extracted = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1

        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        text = clean_text(raw_text)

        if text:
            extracted.append({
                "page": page_num,
                "text": text
            })

    return extracted


def chunk_pages(
    pages_data: list,
    chunk_size: int = 600,
    overlap: int = 100,
    min_chunk_length: int = 120
):
    """
    Split manual pages into useful chunks.

    Improvements over the original implementation:
    - avoids cutting words wherever possible
    - removes tiny/useless fragments
    - preserves page metadata
    - keeps small overlap between chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    for page_item in pages_data:
        page_num = page_item["page"]
        text = clean_text(page_item["text"])

        if len(text) < min_chunk_length:
            # Extremely small extracted page fragments are usually
            # headers, OCR noise, or incomplete text.
            continue

        start = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(start + chunk_size, text_length)
            end = target_end

            # If we are not at the end of the page, try to stop at
            # a natural sentence or word boundary.
            if target_end < text_length:
                search_start = max(
                    start + min_chunk_length,
                    target_end - 120
                )

                sentence_positions = [
                    text.rfind(". ", search_start, target_end),
                    text.rfind("? ", search_start, target_end),
                    text.rfind("! ", search_start, target_end),
                ]

                best_sentence_end = max(sentence_positions)

                if best_sentence_end != -1:
                    end = best_sentence_end + 1
                else:
                    whitespace_end = text.rfind(
                        " ",
                        search_start,
                        target_end
                    )

                    if whitespace_end != -1:
                        end = whitespace_end

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= min_chunk_length:
                chunks.append({
                    "page": page_num,
                    "text": chunk_text
                })

            if end >= text_length:
                break

            next_start = max(0, end - overlap)

            # Move overlap start to a word boundary.
            if next_start > 0:
                space_position = text.find(" ", next_start)

                if space_position != -1 and space_position < end:
                    next_start = space_position + 1

            # Absolute safety against an infinite loop.
            if next_start <= start:
                next_start = end

            start = next_start

    return chunks


def process_manual(
    pdf_path: str,
    chunk_size: int = 600,
    overlap: int = 100
):
    """
    Complete PDF -> page extraction -> chunking pipeline.
    """
    pages = extract_pages(pdf_path)

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap
    )

    return chunks