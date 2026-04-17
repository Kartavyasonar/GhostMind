"""Agentic RAG retriever with self-correction loop (rate-limit optimised)."""
from typing import List, Tuple
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.embeddings import embed_one, batch_cosine_similarity
from app.core.llm import get_llm
from app.core.models import Paper

log = structlog.get_logger()

RELEVANCE_THRESHOLD = 0.45
MAX_REWRITES = 2
TOP_K = 8

# Embedding similarity threshold — if avg top-k cosine score is above this,
# skip the LLM relevance grader entirely (saves 1 LLM call on most queries)
EMBED_SCORE_TRUST_THRESHOLD = 0.35


async def _semantic_search(
    query_embedding: List[float],
    db: AsyncSession,
    top_k: int = TOP_K,
) -> List[Tuple[Paper, float]]:
    result = await db.execute(select(Paper).where(Paper.embedding.isnot(None)))
    papers = result.scalars().all()

    if not papers:
        return []

    corpus = [p.embedding for p in papers]
    scores = batch_cosine_similarity(query_embedding, corpus)
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _embed_based_relevance(ranked: List[Tuple[Paper, float]]) -> float:
    """Cheap relevance estimate from embedding scores — no LLM needed."""
    if not ranked:
        return 0.0
    top_scores = [s for _, s in ranked[:4]]
    return sum(top_scores) / len(top_scores)


async def _grade_relevance_llm(query: str, docs: List[Paper]) -> float:
    """LLM relevance grader — only called when embedding score is borderline."""
    if not docs:
        return 0.0

    snippets = "\n".join(
        f"[{i+1}] {d.title}: {d.abstract[:200]}" for i, d in enumerate(docs[:4])
    )
    prompt = (
        f"Query: {query}\n\nRetrieved documents:\n{snippets}\n\n"
        "Rate relevance 0-1. Respond with ONLY a single decimal number like 0.82"
    )
    try:
        llm = get_llm()
        resp = await llm.complete(
            system="You are a relevance grader. Respond ONLY with a single decimal number between 0 and 1.",
            user=prompt,
            max_tokens=8,
            temperature=0.0,
        )
        score = float(resp.strip().split()[0])
        return max(0.0, min(1.0, score))
    except Exception as e:
        log.warning("Relevance grading failed", error=str(e))
        return 0.5


async def _combined_grade_and_rewrite(query: str, docs: List[Paper], attempt: int) -> Tuple[float, str]:
    """Single LLM call that both grades relevance AND rewrites the query.
    Saves 1 LLM call vs doing them separately.
    """
    snippets = "\n".join(
        f"[{i+1}] {d.title}: {d.abstract[:150]}" for i, d in enumerate(docs[:4])
    )
    prompt = (
        f"Query: {query}\n\nDocuments retrieved:\n{snippets}\n\n"
        "1. Rate relevance 0-1 (single decimal)\n"
        "2. Rewrite the query with better technical keywords for academic paper search\n\n"
        "Respond ONLY in this exact format:\n"
        "RELEVANCE: <number>\n"
        "REWRITE: <improved query>"
    )
    try:
        llm = get_llm()
        resp = await llm.complete(
            system="You are a retrieval optimizer. Respond ONLY in the specified format.",
            user=prompt,
            max_tokens=80,
            temperature=0.3,
        )
        import re
        relevance = 0.5
        rewrite = query

        for line in resp.splitlines():
            line = line.strip()
            if line.upper().startswith("RELEVANCE:"):
                m = re.search(r"[\d.]+", line)
                if m:
                    relevance = max(0.0, min(1.0, float(m.group())))
            elif line.upper().startswith("REWRITE:"):
                rewrite = line.split(":", 1)[1].strip() or query

        return relevance, rewrite
    except Exception as e:
        log.warning("Combined grade+rewrite failed", error=str(e))
        return 0.5, query


async def retrieve(
    query: str,
    db: AsyncSession,
    strategy: str = "semantic",
    top_k: int = TOP_K,
) -> Tuple[List[Paper], int, float]:
    current_query = query
    rewrite_count = 0
    best_docs: List[Paper] = []
    best_score = 0.0

    for attempt in range(MAX_REWRITES + 1):
        q_emb = embed_one(current_query)
        ranked = await _semantic_search(q_emb, db, top_k=top_k)
        docs = [p for p, _ in ranked]

        # First: try cheap embedding-based relevance
        embed_score = _embed_based_relevance(ranked)

        if embed_score >= EMBED_SCORE_TRUST_THRESHOLD:
            # Embedding score is good enough — trust it, skip LLM grader
            relevance = min(1.0, embed_score * 1.4)  # scale to 0-1 range
            log.info("Retrieval attempt (embed-only)", attempt=attempt, relevance=round(relevance, 3), strategy=strategy)
        else:
            # Borderline — use LLM grader only when needed
            relevance = await _grade_relevance_llm(current_query, docs)
            log.info("Retrieval attempt (LLM graded)", attempt=attempt, relevance=round(relevance, 3), strategy=strategy)

        if relevance > best_score:
            best_score = relevance
            best_docs = docs

        if relevance >= RELEVANCE_THRESHOLD or attempt == MAX_REWRITES:
            break

        # Combined grade+rewrite in one LLM call
        _, current_query = await _combined_grade_and_rewrite(current_query, docs, attempt)
        rewrite_count += 1
        log.info("Query rewritten", attempt=attempt, new_query=current_query[:60])

    return best_docs, rewrite_count, best_score
