"""Agentic RAG retriever v2 — honest scores, modern fusion, real strategy diversity.

FIXES vs v1:
  1. REMOVED score inflation (*1.4 / *1.3 / *1.2). Raw cosine is reported as-is,
     so thresholds mean something and the CRAG rewrite loop actually fires.
  2. HYBRID now uses Reciprocal Rank Fusion (RRF, k=60) — the industry-standard
     way to fuse cosine + BM25 rankings. v1 mixed incompatible score scales
     (0.6*cos + 0.4*kw) which is mathematically meaningless.
  3. MMR diversification (λ=0.7) removes near-duplicate papers from top-K.
  4. CRAG (Corrective RAG): if raw top-3 cosine < CRAG_REWRITE_THRESHOLD the
     query is rewritten and re-retrieved — rewrites now happen when needed,
     so the "Rewrites" column becomes a real signal again.
  5. aggressive_rewrite uses HyDE (hypothetical document embedding) instead of
     a naive keyword suffix.
  6. Returns per-document scores so the agent can verify citations.
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
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

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

def _bm25_rank(query: str, papers: List[Paper], k1: float = 1.5, b: float = 0.75) -> List[float]:
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

def _mmr_select(candidates: List[Tuple[Paper, float]], top_k: int,
                lambda_: float = None) -> List[Paper]:
    """Maximal Marginal Relevance — relevance minus redundancy."""
    lambda_ = lambda_ if lambda_ is not None else settings.MMR_LAMBDA
    selected: List[Paper] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best, best_val = None, -math.inf
        for p, s in remaining:
            redundancy = max((_cos(p.embedding, sel.embedding) for sel in selected),
                             default=0.0)
            val = lambda_ * s - (1 - lambda_) * redundancy
            if val > best_val:
                best, best_val = p, val
        selected.append(best)
        remaining.remove((best, best_val and best_val or best_val))
        remaining = [x for x in remaining if x[0] is not best]
    return selected

# ── CRAG helpers ───────────────────────────────────────────────────────────

async def _rewrite_query(query: str, docs: List[Paper]) -> str:
    snippets = "\n".join(f"[{i+1}] {d.title}: {d.abstract[:150]}" for i, d in enumerate(docs[:4]))
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
        return query

async def _crag_loop(query: str, db: AsyncSession, top_k: int,
                     strategy: str) -> Tuple[List[Tuple[Paper, float]], int, float]:
    """Corrective RAG: retrieve → grade (raw cosine) → rewrite if weak."""
    current, rewrites = query, 0
    best_ranked, best_score = [], 0.0
    for attempt in range(MAX_REWRITES + 1):
        q_emb = embed_one(current)
        ranked = await _semantic_search(q_emb, db, top_k=top_k * 2)
        score = _raw_relevance(ranked)
        if score > best_score:
            best_score, best_ranked = score, ranked
        if score >= settings.CRAG_REWRITE_THRESHOLD or attempt == MAX_REWRITES or not ranked:
            break
        current = await _rewrite_query(current, [p for p, _ in ranked[:4]])
        rewrites += 1
        log.info("CRAG rewrite", strategy=strategy, attempt=attempt, new_query=current[:60])
    return best_ranked, rewrites, best_score

# ── Strategies ─────────────────────────────────────────────────────────────

async def _retrieve_semantic(query, db, top_k):
    ranked, rewrites, score = await _crag_loop(query, db, top_k, "semantic")
    docs = _mmr_select(ranked, top_k)
    return docs, rewrites, score, {p.id: s for p, s in ranked[:top_k]}

async def _retrieve_hybrid(query, db, top_k):
    q_emb = embed_one(query)
    ranked = await _semantic_search(q_emb, db, top_k=top_k * 3)
    if not ranked:
        return [], 0, 0.0, {}
    papers = [p for p, _ in ranked]
    emb_order = [p.id for p, _ in ranked]
    bm25_scores = _bm25_rank(query, papers)
    bm25_order = [p.id for p, _ in sorted(zip(papers, bm25_scores),
                                          key=lambda x: x[1], reverse=True)]
    fused_ids = _rrf_fuse([emb_order, bm25_order], k=settings.RRF_K)
    by_id = {p.id: (p, s) for p, s in ranked}
    fused = [by_id[pid] for pid in fused_ids if pid in by_id]
    docs = _mmr_select(fused, top_k)
    return docs, 0, _raw_relevance(fused), {p.id: s for p, s in fused[:top_k]}

async def _retrieve_graph(query, db, top_k):
    from app.graph.knowledge_graph import build_graph, graph_expand
    q_emb = embed_one(query)
    ranked = await _semantic_search(q_emb, db, top_k=top_k)
    score = _raw_relevance(ranked)
    seeds = [p for p, _ in ranked[:5]]
    seed_ids = [d.arxiv_id.split(":")[0] for d in seeds]
    await build_graph(db)
    expanded_ids = graph_expand(seed_ids, hops=2, max_nodes=15)
    docs = list(seeds)
    seen = {d.id for d in docs}
    for arxiv_id in expanded_ids:
        r = await db.execute(select(Paper).where(Paper.arxiv_id.like(f"{arxiv_id}%")).limit(2))
        for p in r.scalars().all():
            if p.id not in seen:
                docs.append(p); seen.add(p.id)
    return docs[:top_k], 0, score, {p.id: s for p, s in ranked[:top_k]}

async def _retrieve_aggressive(query, db, top_k):
    hyde_q = await _hyde_expand(query)
    q_emb = embed_one(hyde_q)
    ranked = await _semantic_search(q_emb, db, top_k=top_k * 2)
    score = _raw_relevance(ranked)
    rewrites = 1
    if score < settings.CRAG_REWRITE_THRESHOLD and ranked:
        rewritten = await _rewrite_query(query, [p for p, _ in ranked[:4]])
        q2 = embed_one(rewritten)
        ranked2 = await _semantic_search(q2, db, top_k=top_k * 2)
        s2 = _raw_relevance(ranked2)
        if s2 > score:
            ranked, score = ranked2, s2
        rewrites = 2
    docs = _mmr_select(ranked, top_k)
    return docs, rewrites, score, {p.id: s for p, s in ranked[:top_k]}

async def retrieve(query: str, db: AsyncSession, strategy: str = "semantic",
                   top_k: int = TOP_K):
    """Returns (docs, rewrite_count, honest_relevance_score, per_doc_scores)."""
    if strategy == "hybrid":
        return await _retrieve_hybrid(query, db, top_k)
    if strategy == "graph":
        return await _retrieve_graph(query, db, top_k)
    if strategy == "aggressive_rewrite":
        return await _retrieve_aggressive(query, db, top_k)
    return await _retrieve_semantic(query, db, top_k)
