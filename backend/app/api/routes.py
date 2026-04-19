"""FastAPI API routes."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.core.database import get_db
from app.core.models import ResearchSession, BenchmarkRun, EpisodicTriplet, FailureLog
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
    """Run the full GhostMind pipeline."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    try:
        result = await run_agent(req.query, db, req.session_number)
        return result
    except Exception as e:
        log.error("Agent error", error=str(e))
        raise HTTPException(500, f"Agent error: {str(e)}")


@router.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.id == req.session_id)
    )
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
        select(ResearchSession).order_by(desc(ResearchSession.session_number)).limit(limit)
    )
    sessions = result.scalars().all()
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
            "created_at": str(s.created_at),
            "duration_ms": s.duration_ms,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.id == session_id)
    )
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "id": s.id,
        "query": s.query,
        "intent": s.intent,
        "answer": s.answer,
        "confidence": s.confidence,
        "hallucination_score": s.hallucination_score,
        "retrieval_strategy": s.retrieval_strategy,
        "papers_retrieved": s.papers_retrieved,
        "rewrite_count": s.rewrite_count,
        "session_number": s.session_number,
        "created_at": str(s.created_at),
        "duration_ms": s.duration_ms,
    }


@router.get("/benchmarks")
async def get_benchmarks(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BenchmarkRun).order_by(BenchmarkRun.session_number.asc()).limit(100)
    )
    runs = result.scalars().all()
    return [
        {
            "session_number": r.session_number,
            "avg_confidence": round(r.avg_confidence, 3),
            "avg_hallucination": round(r.avg_hallucination, 3),
            "answer_quality": round(r.answer_quality, 3),
            "retrieval_precision": round(r.retrieval_precision, 3),
        }
        for r in runs
    ]


@router.get("/memory")
async def get_memory(db: AsyncSession = Depends(get_db)):
    summary = await get_memory_summary(db)
    failures = await get_recent_failures(db, limit=10)
    stats = compute_graph_stats()
    return {
        "strategy_summary": summary,
        "recent_failures": failures,
        "graph_stats": stats,
    }


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    session_count = (await db.execute(select(func.count(ResearchSession.id)))).scalar() or 0
    triplet_count = (await db.execute(select(func.count(EpisodicTriplet.id)))).scalar() or 0
    failure_count = (await db.execute(select(func.count(FailureLog.id)))).scalar() or 0
    avg_conf = (await db.execute(select(func.avg(ResearchSession.confidence)))).scalar() or 0
    avg_hall = (await db.execute(select(func.avg(ResearchSession.hallucination_score)))).scalar() or 0

    return {
        "total_sessions": session_count,
        "total_triplets": triplet_count,
        "total_failures": failure_count,
        "avg_confidence": round(float(avg_conf), 3),
        "avg_hallucination": round(float(avg_hall), 3),
    }
