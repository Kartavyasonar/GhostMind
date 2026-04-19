"""Embedding service — fastembed (ONNX Runtime, no PyTorch, ~50 MB RSS).

WHY THIS CHANGE:
  sentence-transformers pulls in full PyTorch which alone uses ~350 MB RSS.
  On Render's free tier (512 MB limit) that leaves only ~160 MB for the rest
  of the app, causing the OOM crashes seen in deployment logs.

  fastembed uses ONNX Runtime instead. Same model (all-MiniLM-L6-v2 / L3-v2),
  same 384-dim vectors, same cosine similarity — but ~50 MB RSS total.
  The API surface (embed / embed_one / batch_cosine_similarity) is unchanged
  so no other files need modification.

  Model used: paraphrase-MiniLM-L3-v2
  - 61 MB model weight (smaller than L6-v2's 90 MB)
  - Same 384-dim output, slightly lower accuracy but fine for RAG retrieval
  - Falls within free-tier memory budget with headroom to spare

  To switch back to L6-v2 (better accuracy, more RAM):
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 in .env
"""
from typing import List
import numpy as np
import structlog

from app.core.config import settings

log = structlog.get_logger()

_model = None


def get_embedder():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        model_name = settings.EMBED_MODEL
        # Resolve short names to fully-qualified HuggingFace IDs that fastembed supports
        _SHORT_NAME_MAP = {
            "paraphrase-MiniLM-L3-v2": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-MiniLM-L3-v2": "sentence-transformers/all-MiniLM-L3-v2",
        }
        resolved = _SHORT_NAME_MAP.get(model_name, model_name)
        _model = TextEmbedding(
            model_name=resolved,
            max_length=256,      # cap input length — saves RAM during encode
        )
        log.info("Embedding model loaded (fastembed/ONNX)", model=resolved)
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    if not texts:
        return []
    embedder = get_embedder()
    # fastembed.embed() returns a generator of numpy arrays
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_one(text: str) -> List[float]:
    """Embed a single string."""
    return embed([text])[0]


def batch_cosine_similarity(query: List[float], corpus: List[List[float]]) -> List[float]:
    """Cosine similarity of query against every vector in corpus."""
    if not corpus:
        return []
    q = np.array(query, dtype=np.float32)
    C = np.array(corpus, dtype=np.float32)
    norms = np.linalg.norm(C, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-9, norms)
    return (C @ q / norms).tolist()
