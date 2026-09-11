import os
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.manual_processor import process_manual


DEFAULT_PDF_PATH = os.path.join(
    "data",
    "manuals",
    "isuzu_f_series_service_manual.pdf"
)

VECTOR_STORE_DIR = "vector_store"

INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "manual_faiss.index"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "manual_chunks.json"
)

METADATA_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "manual_index_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine similarity threshold.
# Results below this are considered too weak to trust.
DEFAULT_RELEVANCE_THRESHOLD = 0.30


_MODEL = None
_INDEX = None
_CHUNKS = None


def get_embedding_model():
    """Load the embedding model once per Python process."""
    global _MODEL

    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)

    return _MODEL


def _get_pdf_signature(pdf_path: str):
    """
    Used to determine whether the saved vector index still belongs
    to the current PDF.
    """
    stat = os.stat(pdf_path)

    return {
        "pdf_size": stat.st_size,
        "pdf_modified": stat.st_mtime,
        "model_name": MODEL_NAME
    }


def _cache_is_valid(pdf_path: str):
    """
    Check whether saved FAISS index + chunks can safely be reused.
    """

    required_files = [
        INDEX_PATH,
        CHUNKS_PATH,
        METADATA_PATH
    ]

    if not all(os.path.exists(path) for path in required_files):
        return False

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            saved_metadata = json.load(file)

        current_metadata = _get_pdf_signature(pdf_path)

        return (
            saved_metadata.get("pdf_size")
            == current_metadata["pdf_size"]
            and
            saved_metadata.get("pdf_modified")
            == current_metadata["pdf_modified"]
            and
            saved_metadata.get("model_name")
            == MODEL_NAME
        )

    except Exception:
        return False


def _save_cache(index, chunks, pdf_path: str):
    """Save FAISS index and chunk metadata to disk."""

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False
        )

    metadata = _get_pdf_signature(pdf_path)

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2
        )


def _load_cache():
    """Load previously created FAISS index and chunks."""

    index = faiss.read_index(INDEX_PATH)

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    return index, chunks


def initialize_rag(
    pdf_path: str = DEFAULT_PDF_PATH,
    force_rebuild: bool = False
):
    """
    Initialize the manual RAG index.

    First run:
        PDF -> chunks -> embeddings -> FAISS -> save cache

    Later runs:
        load saved FAISS index + chunks
    """

    global _INDEX, _CHUNKS

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF Manual not found at: {pdf_path}"
        )

    # In-memory cache
    if (
        not force_rebuild
        and _INDEX is not None
        and _CHUNKS is not None
    ):
        return _INDEX, _CHUNKS

    # Persistent disk cache
    if (
        not force_rebuild
        and _cache_is_valid(pdf_path)
    ):
        _INDEX, _CHUNKS = _load_cache()

        return _INDEX, _CHUNKS

    print(
        "Building manual RAG index. "
        "This may take several minutes on the first run..."
    )

    chunks = process_manual(pdf_path)

    if not chunks:
        raise ValueError(
            "No readable text found in the provided manual PDF."
        )

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    # Normalize vectors so inner product behaves as cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _INDEX = index
    _CHUNKS = chunks

    _save_cache(
        index,
        chunks,
        pdf_path
    )

    return _INDEX, _CHUNKS


def retrieve_context(
    query: str,
    top_k: int = 3,
    pdf_path: str = DEFAULT_PDF_PATH,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
) -> dict:
    """
    Retrieve relevant Isuzu manual context.

    Returns:
    {
        "query": "...",
        "context": "...",
        "sources": [...]
    }

    Unrelated or low-confidence queries return no manual context.
    """

    if not query or not query.strip():
        return {
            "query": query,
            "context": "Not found in retrieved manual context.",
            "sources": []
        }

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    index, chunks = initialize_rag(pdf_path)

    model = get_embedding_model()

    query_vector = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Required for cosine similarity using IndexFlatIP.
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    retrieved_sources = []
    context_blocks = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx < 0 or idx >= len(chunks):
            continue

        similarity = float(score)

        # Reject weak / unrelated results.
        if similarity < relevance_threshold:
            continue

        item = chunks[idx]

        source = {
            "page": item["page"],
            "text": item["text"],
            "similarity": round(similarity, 4)
        }

        retrieved_sources.append(source)

        context_blocks.append(
            f"[Page {item['page']} | "
            f"Similarity {similarity:.3f}]: "
            f"{item['text']}"
        )

    if not retrieved_sources:
        return {
            "query": query,
            "context": "Not found in retrieved manual context.",
            "sources": []
        }

    return {
        "query": query,
        "context": "\n\n".join(context_blocks),
        "sources": retrieved_sources
    }