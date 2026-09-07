"""Agentic RAG retriever v2.1 — RRF + MMR + CRAG + HyDE with honest scores.

v2.1 FIX (crash): _mmr_select previously called
    remaining.remove((best, best_val))
but best_val is the MMR-adjusted value, NOT the (paper, original_score)
tuple stored in the list → ValueError: list.remove(x): x not in list.
Now removal is done by object identity only, and MMR is None-embedding safe.
"""
import math
import re
from typing import List, Tuple, Dict
import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.embeddings import embed_one, batch_cosine_similarity
from app.core.llm import get_llm
from app.core.models import Paper

log = structlog.get_logger()

TOP_K = 8
MAX_REWRITES = 2


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Core search ────────────────────────────────────────────────────────────

async def _semantic_search(q_emb: List[float], db: AsyncSession,
                           top_k: int) -> List[Tuple[Paper, float]]:
    result = await db.execute(select(Paper).where(Paper.embedding.isnot(None)))
    papers = result.scalars().all()
    if not papers:
        return []
    scores = batch_cosine_similarity(q_emb, [p.embedding for p in papers])
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _raw_relevance(ranked: List[Tuple[Paper, float]]) -> float:
    """HONEST relevance: mean raw cosine of top-3. No inflation multipliers."""
    if not ranked:
        return 0.0
    top = [s for _, s in ranked[:3]]
    return round(sum(top) / len(top), 4)


# ── BM25-lite (for RRF) ────────────────────────────────────────────────────

def _bm25_rank(query: str, papers: List[Paper],
               k1: float = 1.5, b: float = 0.75) -> List[float]:
    docs = [f"{p.title} {p.abstract}".lower().split() for p in papers]
    N = len(docs) or 1
    avgdl = sum(len(d) for d in docs) / N
    df: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q_terms = re.findall(r"\w{3,}", query.lower())
    scores = []
    for d in docs:
        dl = len(d) or 1
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _rrf_fuse(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion — scale-free ranking combination."""
    scores: Dict[str, float] = {}
    for ranking in rank_lists:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# ── MMR (FIXED) ────────────────────────────────────────────────────────────

def _mmr_select(candidates: List[Tuple[Paper, float]], top_k: int,
                lambda_: float = None) -> List[Paper]:
    """
    Maximal Marginal Relevance: relevance minus redundancy.

    FIXED: removal from the candidate pool is done by object identity
    (`x[0] is not best`). The old value-based .remove() raised
    "list.remove(x): x not in list" because it searched for the
    MMR-adjusted tuple which is never stored in the pool.
    """
    lambda_ = lambda_ if lambda_ is not None else settings.MMR_LAMBDA
    selected: List[Paper] = []
    remaining = [c for c in candidates if c is not None and c[0] is not None]

    while remaining and len(selected) < top_k:
        best, best_val = None, -math.inf
        for p, s in remaining:
            p_emb = getattr(p, "embedding", None)
            if p_emb is None or not selected:
                val = lambda_ * s                      # first pick = pure relevance
            else:
                redundancy = max(
                    (_cos(p_emb, sel.embedding)
                     for sel in selected
                     if getattr(sel, "embedding", None) is not None),
                    default=0.0,
                )
                val = lambda_ * s - (1 - lambda_) * redundancy
            if val > best_val:
                best, best_val = p, val
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x[0] is not best]   # identity-safe
    return selected


# ── CRAG / HyDE helpers ────────────────────────────────────────────────────

async def _rewrite_query(query: str, docs: List[Paper]) -> str:
    snippets = "\n".join(f"[{i+1}] {d.title}: {d.abstract[:150]}"
                         for i, d in enumerate(docs[:4]))
    prompt = (f"Query: {query}\n\nDocuments retrieved:\n{snippets}\n\n"
              "Rewrite the query with better technical keywords for academic search.\n"
              "Respond ONLY with the rewritten query.")
    try:
        llm = get_llm()
        resp = await llm.complete(system="You are a retrieval optimizer.",
                                  user=prompt, max_tokens=60, temperature=0.3)
        return resp.strip() or query
    except Exception:
        return query


async def _hyde_expand(query: str) -> str:
    """HyDE: embed query + a hypothetical abstract written by the LLM."""
    try:
        llm = get_llm()
        resp = await llm.complete(
            system="Write 2 sentences of a hypothetical academic abstract that would "
                   "perfectly answer this query. Only the sentences.",
            user=query, max_tokens=80, temperature=0.4)
        return f"{query} {resp.strip()}"
    except Exception:
       
