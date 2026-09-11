import os
from pypdf import PdfReader


def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF Manual not found at: {pdf_path}")
    return PdfReader(pdf_path)


def extract_pages(pdf_path: str):
    reader = load_pdf(pdf_path)
    extracted = []
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            extracted.append({
                "page": page_num,
                "text": text
            })
            
    return extracted


def chunk_pages(pages_data: list, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    
    for page_item in pages_data:
        page_num = page_item["page"]
        text = page_item["text"]
        
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "page": page_num,
                    "text": chunk_text.strip()
                })
                
            start += (chunk_size - overlap)
            
    return chunks


def process_manual(pdf_path: str, chunk_size: int = 500, overlap: int = 50):
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return chunks