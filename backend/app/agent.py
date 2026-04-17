"""GhostMind agent orchestrator — full pipeline (rate-limit optimised)."""
import time
from typing import Dict, Any, Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.llm import get_llm
from app.core.models import ResearchSession, BenchmarkRun, Paper
from app.core.config import settings
from app.retrieval.ingestion import ingest_query
from app.retrieval.retriever import retrieve
from app.graph.knowledge_graph import build_graph, graph_expand
from app.memory.memrl import select_strategy, record_experience, log_failure
from app.evaluation.self_eval import evaluate_answer, compute_outcome_quality

log = structlog.get_logger()

SYSTEM_PROMPT = """You are GhostMind, an expert AI research analyst.
Answer questions about AI research with precision and nuance.
Always base your answer on the provided source documents.
If sources don't fully answer the question, acknowledge the gap.
Be specific — cite paper titles when relevant.
Structure your answer clearly with key findings first.
"""

# Combined prompt: classify intent + evaluate answer in ONE call
COMBINED_EVAL_PROMPT = """You are a research assistant doing two things at once.

Given the query, answer, and sources below:

1. Classify the research intent in 3-6 words (e.g. "Survey transformer architectures")
2. Rate CONFIDENCE (0-1): how well the answer matches the sources
3. Rate HALLUCINATION (0-1): fraction of claims NOT grounded in sources

Respond ONLY in this exact format (no extra text):
INTENT: <phrase>
CONFIDENCE: <number>
HALLUCINATION: <number>
"""


async def _classify_and_evaluate(
    query: str,
    answer: str,
    docs: list,
) -> tuple:
    """Single LLM call that does intent classification + self-evaluation together."""
    context = "\n\n".join(
        f"[{i+1}] {d.title}: {d.abstract[:250]}" for i, d in enumerate(docs[:4])
    )
    user_msg = (
        f"Query: {query}\n\n"
        f"Answer: {answer[:600]}\n\n"
        f"Sources:\n{context}"
    )
    try:
        llm = get_llm()
        resp = await llm.complete(
            system=COMBINED_EVAL_PROMPT,
            user=user_msg,
            max_tokens=48,
            temperature=0.0,
        )
        intent = "Research query"
        confidence = 0.5
        hallucination = 0.3

        import re
        for line in resp.splitlines():
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip()
            elif "CONFIDENCE" in line.upper():
                m = re.search(r"[\d.]+", line)
                if m:
                    confidence = max(0.0, min(1.0, float(m.group())))
            elif "HALLUCINATION" in line.upper():
                m = re.search(r"[\d.]+", line)
                if m:
                    hallucination = max(0.0, min(1.0, float(m.group())))

        return intent, confidence, hallucination

    except Exception as e:
        log.warning("Combined eval failed, using defaults", error=str(e))
        return "Research query", 0.5, 0.3


async def _count_sessions(db: AsyncSession) -> int:
    result = await db.execute(select(func.count(ResearchSession.id)))
    return result.scalar() or 0


async def run_agent(
    query: str,
    db: AsyncSession,
    session_number: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.time()

    # 1. MemRL strategy selection (no LLM call — pure DB lookup)
    # We use a placeholder intent hash based on the raw query for first-pass selection;
    # the real intent is resolved after the answer is generated (saves 1 LLM call)
    strategy = await select_strategy(query[:80], db)

    # 2. Ingest fresh papers from arXiv (no LLM)
    await ingest_query(query, db)

    # 3. Agentic retrieval with self-correction  [uses up to 3 LLM calls internally]
    docs, rewrite_count, relevance_score = await retrieve(query, db, strategy=strategy)

    # 4. GraphRAG expansion for graph/hybrid strategies (no LLM)
    if docs and strategy in ("graph", "hybrid"):
        seed_ids = [d.arxiv_id.split(":")[0] for d in docs[:5]]
        await build_graph(db)
        expanded_ids = graph_expand(seed_ids, hops=2, max_nodes=12)
        for arxiv_id in expanded_ids:
            result = await db.execute(
                select(Paper).where(Paper.arxiv_id.like(f"{arxiv_id}%")).limit(2)
            )
            extra = result.scalars().all()
            existing_ids = {d.id for d in docs}
            docs.extend([p for p in extra if p.id not in existing_ids])
        docs = docs[:12]

    # 5. Generate answer  [1 LLM call]
    context = "\n\n".join(
        f"[{i+1}] Title: {d.title}\nAbstract: {d.abstract[:350]}"
        for i, d in enumerate(docs[:6])
    )
    user_prompt = (
        f"Research question: {query}\n\n"
        f"Source documents ({len(docs)} retrieved):\n{context}\n\n"
        "Provide a comprehensive, accurate answer based on these sources."
    )
    llm = get_llm()
    answer = await llm.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )

    # 6. Combined intent + self-evaluation  [1 LLM call instead of 2]
    intent, confidence, hallucination = await _classify_and_evaluate(query, answer, docs)
    outcome_quality = compute_outcome_quality(confidence, hallucination)

    log.info("Agent started", query=query[:60], intent=intent)

    # 7. Session number
    sn = session_number or ((await _count_sessions(db)) + 1)

    # 8. Persist session
    session = ResearchSession(
        query=query,
        intent=intent,
        answer=answer,
        confidence=confidence,
        hallucination_score=hallucination,
        retrieval_strategy=strategy,
        papers_retrieved=[d.arxiv_id for d in docs],
        rewrite_count=rewrite_count,
        session_number=sn,
        duration_ms=int((time.time() - t0) * 1000),
    )
    db.add(session)
    await db.flush()

    # 9. MemRL update (no LLM)
    new_q = await record_experience(
        session_id=session.id,
        intent=intent,
        strategy=strategy,
        outcome_quality=outcome_quality,
        db=db,
    )

    # 10. Log failures (no LLM)
    if outcome_quality < 0.4:
        better = next(
            (s for s in ["hybrid", "graph", "semantic"] if s != strategy),
            "hybrid",
        )
        await log_failure(
            session_id=session.id,
            failure_type="low_confidence" if confidence < 0.5 else "hallucination",
            description=f"Quality {outcome_quality:.2f} with strategy '{strategy}'",
            strategy_before=strategy,
            strategy_after=better,
            delta_q=new_q - 0.5,
            db=db,
        )

    # 11. Benchmark run
    benchmark = BenchmarkRun(
        session_number=sn,
        avg_confidence=confidence,
        avg_hallucination=hallucination,
        answer_quality=outcome_quality,
        retrieval_precision=relevance_score,
        total_queries=1,
    )
    db.add(benchmark)
    await db.commit()

    duration = round((time.time() - t0) * 1000)
    log.info("Agent complete", duration_ms=duration, confidence=round(confidence, 2), strategy=strategy)

    return {
        "session_id": session.id,
        "session_number": sn,
        "query": query,
        "intent": intent,
        "answer": answer,
        "confidence": round(confidence, 3),
        "hallucination_score": round(hallucination, 3),
        "outcome_quality": round(outcome_quality, 3),
        "retrieval_strategy": strategy,
        "papers_retrieved": len(docs),
        "rewrite_count": rewrite_count,
        "relevance_score": round(relevance_score, 3),
        "q_value_after": round(new_q, 3),
        "duration_ms": duration,
        "sources": [
            {
                "title": d.title,
                "arxiv_id": d.arxiv_id,
                "url": d.url,
                "authors": d.authors[:3] if d.authors else [],
                "abstract_snippet": d.abstract[:220],
            }
            for d in docs[:6]
        ],
    }
