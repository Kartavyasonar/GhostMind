"""arXiv paper ingestion: fetch, chunk, embed, persist."""
import hashlib
from typing import List
import structlog
import arxiv

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.embeddings import embed
from app.core.models import Paper

log = structlog.get_logger()

CHUNK_SIZE = 512


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


async def ingest_query(
    query: str,
    db: AsyncSession,
    max_results: int = None,
) -> List[Paper]:
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    log.info("Fetching arXiv papers", query=query[:60], max_results=max_results)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    try:
        for r in client.results(search):
            results.append(r)
    except Exception as e:
        log.error("arXiv fetch failed", error=str(e))
        return []

    new_papers: List[Paper] = []
    texts_to_embed: List[str] = []
    paper_objs: List[Paper] = []

    for r in results:
        arxiv_id = _arxiv_id(r.entry_id)

        existing = await db.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if existing.scalar_one_or_none():
            continue

        abstract_chunks = _chunk_text(r.summary)
        for i, chunk in enumerate(abstract_chunks[:2]):
            chunk_id = f"{arxiv_id}:{i}" if i > 0 else arxiv_id
            paper_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]

            existing_chunk = await db.execute(select(Paper).where(Paper.arxiv_id == chunk_id))
            if existing_chunk.scalar_one_or_none():
                continue

            p = Paper(
                id=paper_id,
                arxiv_id=chunk_id,
                title=r.title,
                abstract=chunk,
                authors=[a.name for a in r.authors[:10]],
                categories=r.categories,
                published=r.published.isoformat() if r.published else None,
                url=r.pdf_url or r.entry_id,
                chunk_index=i,
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
