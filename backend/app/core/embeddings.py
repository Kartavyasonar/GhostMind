"""Sentence-transformer embedding service (runs locally, no API key needed)."""
from typing import List
import numpy as np
import structlog

from app.core.config import settings

log = structlog.get_logger()

_model = None


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.EMBED_MODEL)
        log.info("Embedding model loaded", model=settings.EMBED_MODEL)
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def embed_one(text: str) -> List[float]:
    return embed([text])[0]


def batch_cosine_similarity(query: List[float], corpus: List[List[float]]) -> List[float]:
    q = np.array(query, dtype=np.float32)
    C = np.array(corpus, dtype=np.float32)
    norms = np.linalg.norm(C, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-9, norms)
    return (C @ q / norms).tolist()
