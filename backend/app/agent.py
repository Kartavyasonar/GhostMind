"""GhostMind agent v2 — citation-grounded generation with verified metrics.

FIXES vs v1:
  1. Generator MUST cite sources inline as [1], [2]. Each citation is then
     VERIFIED in code (embedding cosine of citing sentence vs cited doc).
     → citation_precision is a real, varying, honest metric (ALCE-style).
  2. Confidence/hallucination come from MEASURED faithfulness (claim-level
     embedding NLI) + verified citations + raw retrieval support + rubric
     judge — never the old flat 0.8/0.2 LLM self-grade.
  3. Q-learning reward = 0.35·faithfulness + 0.25·citation_precision
     + 0.25·raw_retrieval + 0.15·completeness → real gradient per strategy.
  4. Cold-start hack removed — UCB1 in memrl handles unvisited arms.
  5. Full eval breakdown persisted & returned for the UI.
"""
import re
import time
import math
from typing import Dict, Any, List, Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.llm import get_llm
from app.core.models import ResearchSession, BenchmarkRun, Paper
from app.core.config import settings
from app.core.embeddings import embed
from app.core.embeddings import batch_cosine_similarity
from app.retrieval.ingestion import ingest_query
from app.retrieval.retriever import retrieve
from app.graph.knowledge_graph import build_graph, graph_expand
from app.memory.memrl import (select_strategy, record_experience, log_failure,
                              get_best_strategy_for_intent, _fuzzy_bucket)
from app.evaluation.self_eval import (measure_faithfulness, llm_judge_score,
                                      calibrate, compute_outcome_quality, split_claims)

log = structlog.get_logger()

SYSTEM_PROMPT = """You are GhostMind, an expert AI research analyst.
Answer questions about AI research with precision and nuance.
Base every factual statement on the provided source documents.
CITE sources inline: after each sentence that uses a source, add its index like [1] or [2, 3].
If sources don't fully answer the question, acknowledge the gap explicitly.
Be specific — mention paper titles and concrete findings.
Structure: key findings first, then supporting detail."""

INTENT_CLASSIFY_PROMPT = """Classify this research query into a 3-6 word intent phrase.
Examples: "Survey RL algorithms", "Explain attention mechanism", "Compare LLM benchmarks"
Respond with ONLY the intent phrase, nothing else."""

_CITE_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
_MARK_RE = re.compile(r"\[(\d{1,2}(?:,\s*\d{1,2})*)\]")
_INTENT_CACHE: dict = {}

async def _classify_intent_fast(query: str) -> str:
    cache_key = query.strip().lower()
    if cache_key in _INTENT_CACHE:
        return _INTENT_CACHE[cache_key]
    try:
        llm = get_llm()
        resp = await llm.complete(system=INTENT_CLASSIFY_PROMPT, user=query,  max_tokens=20, temperature=0.0)
        intent = resp.strip().strip('"').strip("'")
        if not (2 <= len(intent.split()) <= 8 and "\n" not in intent):
           return intent
    except Exception:
        pass
    return query[:60]

def _verify_citations(answer: str, docs: list) -> Dict[str, Any]:
    """ALCE-style attribution check: is each cited sentence supported by its cited doc?"""
    sentences = [s.strip() for s in _CITE_SENT_RE.split(answer) if s.strip()]
    cited = []   # (sentence, [doc indices 0-based])
    for sent in sentences:
        marks = _MARK_RE.findall(sent)
        if not marks:
            continue
        idxs = []
        for group in marks:
            for num in group.split(","):
                i = int(num.strip()) - 1
                if 0 <= i < len(docs):
                    idxs.append(i)
        if idxs:
            cited.append((re.sub(_MARK_RE, "", sent).strip(), idxs))

    if not cited or not docs:
        return {"citation_precision": 0.0, "citations": 0, "verified": 0}

    try:
        sent_embs = embed([s for s, _ in cited])
    except Exception:
        return {"citation_precision": 0.5, "citations": len(cited), "verified": 0}

    verified = 0
    for (sent, idxs), se in zip(cited, sent_embs):
        target_embs = [docs[i].embedding for i in idxs if docs[i].embedding is not None]
        if not target_embs:
            continue
        sims = batch_cosine_similarity(se, target_embs)
        if sims and max(sims) >= settings.CITATION_VERIFY_THRESHOLD:
            verified += 1
    return {"citation_precision": round(verified / len(cited), 4),
            "citations": len(cited), "verified": verified}

async def _count_sessions(db: AsyncSession) -> int:
    result = await db.execute(select(func.count(ResearchSession.id)))
    return result.scalar() or 0

async def run_agent(query: str, db: AsyncSession,
                    session_number: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.time()
    sn = session_number or ((await _count_sessions(db)) + 1)

    # 1. Intent
    intent = await _classify_intent_fast(query)
    log.info("Agent started", query=query[:60], intent=intent, session=sn)

    # 2. UCB1 strategy selection (auto cold-start for unvisited arms)
    strategy, was_exploring = await select_strategy(intent, db)
    memory_before = await get_best_strategy_for_intent(intent, db)

    # 3. Ingest + strategy-specific retrieval (honest scores)
    await ingest_query(query, db)
    docs, rewrite_count, relevance_score, per_doc_scores = await retrieve(query, db, strategy=strategy)

    # 4. GraphRAG expansion for graph/hybrid
    if docs and strategy in ("graph", "hybrid"):
        seed_ids = [d.arxiv_id.split(":")[0] for d in docs[:5]]
        await build_graph(db)
        for arxiv_id in graph_expand(seed_ids, hops=2, max_nodes=12):
            r = await db.execute(select(Paper).where(Paper.arxiv_id.like(f"{arxiv_id}%")).limit(2))
            existing = {d.id for d in docs}
            docs.extend([p for p in r.scalars().all() if p.id not in existing])
        docs = docs[:10]

    # 5. Citation-grounded generation
    memory_hint = ""
    best_q = memory_before.get("best_q", 0)
    if best_q >= 0.55 and not was_exploring:
        memory_hint = (f"\n\n[Memory note: '{memory_before.get('best_strategy')}' scored "
                       f"{best_q:.0%} on this topic previously. Prioritise precision.]")
    context = "\n\n".join(f"[{i+1}] Title: {d.title}\nAbstract: {d.abstract[:400]}"
                          for i, d in enumerate(docs[:8]))
    answer = await get_llm().complete(
        system=SYSTEM_PROMPT,
        user=(f"Research question: {query}\n\nSource documents ({len(docs)}):\n{context}\n\n"
              f"Provide a comprehensive, cited answer." + memory_hint),
        max_tokens=settings.LLM_MAX_TOKENS, temperature=settings.LLM_TEMPERATURE)

    # 6. MEASURED evaluation (no more flat 0.8/0.2)
    faith = measure_faithfulness(answer, docs)
    cite = _verify_citations(answer, docs)
    judge = await llm_judge_score(query, answer, docs)
    confidence, hallucination = calibrate(faith["faithfulness"], cite["citation_precision"],
                                          relevance_score, judge)
    code_quality = compute_outcome_quality(faith["faithfulness"], cite["citation_precision"],
                                           relevance_score, answer)
    log.info("Measured quality", faithfulness=faith["faithfulness"],
             citation_precision=cite["citation_precision"], judge=judge,
             relevance=relevance_score, quality=code_quality)

    # 7. Persist
    eval_breakdown = {
        "claims": faith["claims"], "grounded_claims": faith["grounded"],
        "ungrounded_claims": faith["ungrounded_claims"],
        "mean_claim_similarity": faith["mean_claim_similarity"],
        "citations": cite["citations"], "verified_citations": cite["verified"],
        "judge_score": judge, "retrieval_score": relevance_score,
    }
    session = ResearchSession(
        query=query, intent=intent, answer=answer, confidence=confidence,
        hallucination_score=hallucination, retrieval_strategy=strategy,
        papers_retrieved=[d.arxiv_id for d in docs], rewrite_count=rewrite_count,
        session_number=sn, duration_ms=int((time.time() - t0) * 1000),
        sources_count=len(docs), outcome_quality=code_quality,
        faithfulness=faith["faithfulness"], citation_precision=cite["citation_precision"],
        eval_breakdown=eval_breakdown)
    db.add(session)
    await db.flush()

    # 8. MemRL UCB update with the measured reward
    q_debug = await record_experience(session.id, intent, strategy, code_quality, db, was_exploring)
    memory_after = await get_best_strategy_for_intent(intent, db)

    if code_quality < 0.4:
        better = next((s for s in ["hybrid", "graph", "semantic"] if s != strategy), "hybrid")
        await log_failure(session.id,
                          "low_faithfulness" if faith["faithfulness"] < 0.5 else "weak_retrieval",
                          f"Quality {code_quality:.2f} (faith={faith['faithfulness']:.2f}, "
                          f"rel={relevance_score:.2f}) with '{strategy}'",
                          strategy, better, q_debug["delta_q"], db)

    db.add(BenchmarkRun(session_number=sn, avg_confidence=confidence,
                        avg_hallucination=hallucination, answer_quality=code_quality,
                        retrieval_precision=relevance_score,
                        faithfulness=faith["faithfulness"],
                        citation_precision=cite["citation_precision"], total_queries=1))
    await db.commit()

    duration = round((time.time() - t0) * 1000)
    return {
        "session_id": session.id, "session_number": sn, "query": query, "intent": intent,
        "answer": answer, "confidence": confidence, "hallucination_score": hallucination,
        "outcome_quality": code_quality, "retrieval_strategy": strategy,
        "papers_retrieved": len(docs), "sources_count": len(docs),
        "rewrite_count": rewrite_count, "relevance_score": relevance_score,
        "faithfulness": faith["faithfulness"], "citation_precision": cite["citation_precision"],
        "q_value_after": q_debug["new_q"], "duration_ms": duration,
        "eval_breakdown": eval_breakdown,
        "memrl_debug": {"was_exploring": was_exploring, "intent_bucket": _fuzzy_bucket(intent),
                        "memory_before": memory_before, "memory_after": memory_after,
                        "q_old": q_debug.get("old_q"), "q_new": q_debug["new_q"],
                        "q_delta": q_debug["delta_q"], "visit_count": q_debug["visit_count"]},
        "sources": [{"title": d.title, "arxiv_id": d.arxiv_id, "url": d.url,
                     "authors": d.authors[:3] if d.authors else [],
                     "abstract_snippet": d.abstract[:220]} for d in docs[:8]],
    }
