import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.manual_processor import process_manual

DEFAULT_PDF_PATH = os.path.join("data", "manuals", "isuzu_f_series_service_manual.pdf")

# Global Cache to prevent reloading models repeatedly
_MODEL = None
_INDEX = None
_CHUNKS = None


def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        # Lightweight, super fast sentence transformer model
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def initialize_rag(pdf_path: str = DEFAULT_PDF_PATH):
    """PDF process karta hai, embeddings banata hai, aur FAISS index ready karta hai."""
    global _INDEX, _CHUNKS
    
    if _INDEX is not None and _CHUNKS is not None:
        return _INDEX, _CHUNKS

    # 1. Extract Chunks
    chunks = process_manual(pdf_path)
    if not chunks:
        raise ValueError("No readable text found in the provided manual PDF.")
        
    # 2. Generate Embeddings
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # 3. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    _INDEX = index
    _CHUNKS = chunks
    
    return _INDEX, _CHUNKS


def retrieve_context(query: str, top_k: int = 3, pdf_path: str = DEFAULT_PDF_PATH) -> dict:
    """Member 3 aur Member 4 ke use ke liye main Public Function."""
    if not query or not query.strip():
        return {
            "query": query,
            "context": "Not found in retrieved manual context.",
            "sources": []
        }
        
    index, chunks = initialize_rag(pdf_path)
    model = get_embedding_model()
    
    # Encode Query
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    
    # Search Vector Index
    distances, indices = index.search(query_vector, top_k)
    
    retrieved_sources = []
    context_blocks = []
    
    for idx in indices[0]:
        if idx < len(chunks) and idx >= 0:
            item = chunks[idx]
            retrieved_sources.append({
                "page": item["page"],
                "text": item["text"]
            })
            context_blocks.append(f"[Page {item['page']}]: {item['text']}")
            
    if not context_blocks:
        return {
            "query": query,
            "context": "Not found in retrieved manual context.",
            "sources": []
        }
        
    combined_context = "\n\n".join(context_blocks)
    
    return {
        "query": query,
        "context": combined_context,
        "sources": retrieved_sources
    }