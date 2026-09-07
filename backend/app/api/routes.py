"""FastAPI API routes. (v2: sessions expose measured metrics + resolved sources.)"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.core.database import get_db
from app.core.models import ResearchSession, BenchmarkRun, EpisodicTriplet, FailureLog, Paper
from app.agent import run_agent
from app.memory.memrl import get_memory_summary, get_recent_failures
from app.graph.knowledge_graph import compute_graph_stats

log = structlog.get_logger()
router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_number: Optional[int] = None

class FeedbackRequest(BaseModel):
    session_id: str
    score: float

@router.post("/query")
async def query_endpoint(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    try:
        return await run_agent(req.query, db, req.session_number)
    except Exception as e:
        log.error("Agent error", error=str(e))
        raise HTTPException(500, f"Agent error: {str(e)}")

@router.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ResearchSession).where(ResearchSession.id == req.session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    session.feedback_score = max(0.0, min(1.0, req.score))
    db.add(session)
    await db.commit()
    return {"ok": True}

@router.get("/sessions")
async def list_sessions(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchSession).order_by(desc(ResearchSession.session_number)).limit(limit))
    return [
        {
            "id": s.id,
            "query": s.query[:80],
            "intent": s.intent,
            "confidence": round(s.confidence, 3),
            "hallucination_score": round(s.hallucination_score, 3),
            "retrieval_strategy": s.retrieval_strategy,
            "session_number": s.session_number,
            "rewrite_count": s.rewrite_count,
            "sources_count": s.sources_count or len(s.papers_retrieved or []),
            "outcome_quality": round(s.outcome_quality, 3),
            "faithfulness": round(s.faithfulness, 3),
            "citation_precision": round(s.citation_precision, 3),
            "created_at": str(s.created_at),
            "duration_ms": s.duration_ms,
        }
        for s in result.scalars().all()
    ]

@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, "Session not found")
    sources = []
    for aid in (s.papers_retrieved or [])[:8]:
        r = await db.execute(select(Paper).where(Paper.arxiv_id.like(f"{aid}%")).limit(1))
        p = r.scalar_one_or_none()
        if p:
            sources.append({"title": p.title, "url": p.url, "arxiv_id": p.arxiv_id})
    return {
        "id": s.id, "query": s.query, "intent": s.intent, "answer": s.answer,
        "confidence": s.confidence, "hallucination_score": s.hallucination_score,
        "retrieval_strategy": s.retrieval_strategy, "papers_retrieved": s.papers_retrieved,
        "rewrite_count": s.rewrite_count, "session_number": s.session_number,
        "sources_count": s.sources_count or len(sources),
        "outcome_quality": s.outcome_quality, "faithfulness": s.faithfulness,
        "citation_precision": s.citation_precision,
        "eval_breakdown": s.eval_breakdown or {}, "sources": sources,
        "created_at": str(s.created_at), "duration_ms": s.duration_ms,
    }

@router.get("/benchmarks")
async def get_benchmarks(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BenchmarkRun).order_by(BenchmarkRun.session_number.asc()).limit(100))
    return [
        {"session_number": r.session_number,
         "avg_confidence": round(r.avg_confidence, 3),
         "avg_hallucination": round(r.avg_hallucination, 3),
         "answer_quality": round(r.answer_quality, 3),
         "retrieval_precision": round(r.retrieval_precision, 3),
         "faithfulness": round(r.faithfulness, 3),
         "citation_precision": round(r.citation_precision, 3)}
        for r in result.scalars().all()
    ]

@router.get("/memory")
async def get_memory(db: AsyncSession = Depends(get_db)):
    return {"strategy_summary": await get_memory_summary(db),
            "recent_failures": await get_recent_failures(db, limit=10),
            "graph_stats": compute_graph_stats()}

@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    session_count = (await db.execute(select(func.count(ResearchSession.id)))).scalar() or 0
    triplet_count = (await db.execute(select(func.count(EpisodicTriplet.id)))).scalar() or 0
    failure_count = (await db.execute(select(func.count(FailureLog.id)))).scalar() or 0
    avg_conf = (await db.execute(select(func.avg(ResearchSession.confidence)))).scalar() or 0
    avg_hall = (await db.execute(select(func.avg(ResearchSession.hallucination_score)))).scalar() or 0
    avg_quality = (await db.execute(select(func.avg(ResearchSession.outcome_quality)))).scalar() or 0
    return {"total_sessions": session_count, "total_triplets": triplet_count,
            "total_failures": failure_count, "avg_confidence": round(float(avg_conf), 3),
            "avg_hallucination": round(float(avg_hall), 3),
            "avg_quality": round(float(avg_quality), 3)}
