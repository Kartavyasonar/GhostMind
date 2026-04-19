"""arXiv paper ingestion: fetch, chunk, embed, persist.

FIXES vs original:
  1. arXiv HTTP 429 (rate-limit): The original arxiv.Client() used default settings
     with no inter-request delay and no retry logic. The arXiv public API requires
     >= 3 seconds between requests and will 429 if you exceed that.

     Fix: arxiv.Client(delay_seconds=4.0, num_retries=3) enforces the delay at
     the SDK level. tenacity wraps the entire fetch with exponential backoff so
     transient 429s are retried automatically before returning an empty list.

  2. max_results capped at 8: fetching 15+ papers triggers multiple API pages
     which multiply the 429 risk AND the embedding memory cost. 8 papers is
     sufficient for RAG quality on the free tier.

  3. Only 1 chunk per paper (was 2): halves embedding memory usage per ingest.
     The abstract chunk at index 0 is the most information-dense part.
"""
import hashlib
import time
from typing import List
import structlog
import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.embeddings import embed
from app.core.models import Paper

log = structlog.get_logger()
_tenacity_logger = logging.getLogger("tenacity.arxiv")

CHUNK_SIZE = 400   # reduced from 512 — shorter chunks = less RAM during embed


def _arxiv_id(entry_id: str) -> str:
    base = entry_id.split("/")[-1]
    for suffix in ["v1", "v2", "v3", "v4", "v5"]:
        base = base.replace(suffix, "")
    return base.strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(" ".join(buf)) >= size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [text]


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=5, max=40),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=False,
)
def _fetch_arxiv_results(query: str, max_results: int) -> List:
    """
    Synchronous arXiv fetch wrapped in tenacity retry.
    arxiv.Client with delay_seconds=4.0 enforces the inter-request
    delay mandated by the arXiv API ToS (3s minimum).
    """
    client = arxiv.Client(
        page_size=min(max_results, 10),   # never request more than 10 per page
        delay_seconds=4.0,                # arXiv requires >= 3s between requests
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in client.results(search):
        results.append(r)
    return results


async def ingest_query(
    query: str,
    db: AsyncSession,
    max_results: int = None,
) -> List[Paper]:
    # Cap at 8 on free tier to stay under RAM + rate-limit budget
    max_results = min(max_results or settings.ARXIV_MAX_RESULTS, 8)
    log.info("Fetching arXiv papers", query=query[:60], max_results=max_results)

    results = []
    try:
        results = _fetch_arxiv_results(query, max_results)
    except Exception as e:
        log.error("arXiv fetch failed after retries", error=str(e))
        return []

    if not results:
        log.info("arXiv returned no results")
        return []

    new_papers: List[Paper] = []
    texts_to_embed: List[str] = []
    paper_objs: List[Paper] = []

    for r in results:
        arxiv_id = _arxiv_id(r.entry_id)

        existing = await db.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if existing.scalar_one_or_none():
            continue

        # FIX: only 1 chunk per paper (index 0 only) — halves embedding memory
        abstract_chunks = _chunk_text(r.summary)
        chunk = abstract_chunks[0]

        paper_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:32]

        existing_chunk = await db.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if existing_chunk.scalar_one_or_none():
            continue

        p = Paper(
            id=paper_id,
            arxiv_id=arxiv_id,
            title=r.title,
            abstract=chunk,
            authors=[a.name for a in r.authors[:10]],
            categories=r.categories,
            published=r.published.isoformat() if r.published else None,
            url=r.pdf_url or r.entry_id,
            chunk_index=0,
            embedding=None,
        )
        paper_objs.append(p)
        texts_to_embed.append(f"{r.title}. {chunk}")

    if not texts_to_embed:
        log.info("No new papers to ingest")
        return []

    log.info("Embedding papers", count=len(texts_to_embed))
    embeddings = embed(texts_to_embed)

    for p, emb in zip(paper_objs, embeddings):
        p.embedding = emb
        db.add(p)
        new_papers.append(p)

    await db.commit()
    log.info("Ingested new papers", count=len(new_papers))
    return new_papers
