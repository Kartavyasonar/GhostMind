"""GhostMind agent — fixed long-form grounded RAG + stable MemRL metrics.

Fixes included:
1. Fixes NameError: code_quality was used before assignment.
2. Stops hallucination from being stuck at 0% by using measured claim grounding,
   citation checks, retrieval support, and an uncertainty floor.
3. Produces longer, more detailed answers by improving the generation prompt and
   giving the model more source context.
4. Keeps Q-learning reward separate from display confidence/hallucination.
5. Supports both older and newer DB models by only writing fields that actually
   exist on the SQLAlchemy model.
6. Supports both retrieve() return formats:
   - docs, rewrite_count, relevance_score
   - docs, rewrite_count, relevance_score, per_doc_scores
"""

import inspect
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.llm import get_llm
from app.core.models import BenchmarkRun, Paper, ResearchSession
from app.graph.knowledge_graph import build_graph, graph_expand
from app.memory.memrl import (
    _fuzzy_bucket,
    get_best_strategy_for_intent,
    log_failure,
    record_experience,
    select_strategy,
)
from app.retrieval.ingestion import ingest_query
from app.retrieval.retriever import retrieve

try:
    from app.core.embeddings import batch_cosine_similarity, embed
except Exception:  # keeps app booting even if embedding import fails
    embed = None
    batch_cosine_similarity = None


log = structlog.get_logger()


SYSTEM_PROMPT = """You are GhostMind, an expert AI research analyst.

Your task is to answer the user's research question using ONLY the provided source documents.

Rules:
- Produce a detailed, substantive answer.
- Prefer 700-1200 words when enough source evidence exists.
- Start with a concise direct answer.
- Then explain the key findings in depth.
- Cite source numbers inline like [1], [2], or [1, 3].
- Every factual claim based on a paper should have a citation.
- Mention paper titles when useful.
- Compare and contrast sources when possible.
- If the provided sources are incomplete, clearly say what is missing.
- Do NOT invent paper details, results, datasets, metrics, or author claims.
- If no source supports a claim, mark it as uncertain instead of presenting it as fact.

Suggested structure:
1. Short answer
2. Key findings
3. Evidence from the retrieved papers
4. Limitations / gaps in the sources
5. Practical takeaway
"""


INTENT_CLASSIFY_PROMPT = """Classify this research query into a 3-6 word intent phrase.

Examples:
- "Survey reinforcement learning algorithms"
- "Explain attention mechanism"
- "Compare LLM evaluation benchmarks"
- "Analyze retrieval augmented generation"

Respond with ONLY the intent phrase, nothing else.
"""


LLM_EVAL_PROMPT = """You are a strict grounded-answer evaluator.

Given a user query, an answer, and source abstracts, evaluate the answer.

Return ONLY this exact format:
INTENT: <3-6 word phrase>
CONFIDENCE: <number from 0 to 1>
HALLUCINATION: <number from 0 to 1>

Definitions:
- CONFIDENCE: how well the answer is supported by the supplied sources.
- HALLUCINATION: fraction of answer claims that are not supported by the supplied sources.

Be strict. If the answer makes claims not directly supported by the sources, hallucination should be above 0.
"""


_CITE_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
_MARK_RE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")

_INTENT_CACHE: Dict[str, str] = {}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "their", "there", "these", "those", "which", "while", "where", "when",
    "what", "how", "why", "can", "could", "would", "should", "may", "might",
    "are", "was", "were", "been", "being", "has", "have", "had", "not",
    "but", "they", "them", "its", "our", "your", "also", "such", "than",
    "then", "using", "used", "use", "based", "paper", "papers", "study",
    "research", "model", "models", "method", "methods", "approach",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _round4(x: float) -> float:
    return round(_clamp(x), 4)


def _model_kwargs(model_cls: Any, **values: Any) -> Dict[str, Any]:
    """Only keep kwargs that are real SQLAlchemy columns on the model."""
    try:
        cols = set(model_cls.__table__.columns.keys())
        return {k: v for k, v in values.items() if k in cols}
    except Exception:
        return values


def _token_set(text: str) -> set:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def _lexical_overlap(a: str, b: str) -> float:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _split_claims(answer: str) -> List[str]:
    """Split answer into likely factual claims/sentences."""
    if not answer or not answer.strip():
        return []

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    claims: List[str] = []

    for part in raw_parts:
        s = part.strip()
        if not s:
            continue

        # Remove markdown bullets/headings noise.
        s = re.sub(r"^[#*\-\d.\s]+", "", s).strip()
        if not s:
            continue

        # Remove pure headings.
        if len(s.split()) <= 4 and not s.endswith((".", "!", "?")):
            continue

        # Ignore very short fragments.
        if len(s) < 35:
            continue

        claims.append(s)

    return claims[:40]


def _citation_indices_from_sentence(sentence: str, docs_len: int) -> List[int]:
    idxs: List[int] = []
    for group in _MARK_RE.findall(sentence or ""):
        for n in group.split(","):
            try:
                i = int(n.strip()) - 1
                if 0 <= i < docs_len:
                    idxs.append(i)
            except Exception:
                continue
    return sorted(set(idxs))


def _verify_citations(answer: str, docs: List[Paper]) -> Dict[str, Any]:
    """Check whether cited sentences are actually close to their cited docs."""
    if not answer or not docs:
        return {
            "citation_precision": 0.0,
            "citations": 0,
            "verified": 0,
            "citation_coverage": 0.0,
        }

    sentences = [s.strip() for s in _CITE_SENT_RE.split(answer) if s.strip()]
    cited_sentences: List[Tuple[str, List[int]]] = []
    total_claims = max(1, len(_split_claims(answer)))

    for sent in sentences:
        idxs = _citation_indices_from_sentence(sent, len(docs))
        if idxs:
            clean = re.sub(_MARK_RE, "", sent).strip()
            if clean:
                cited_sentences.append((clean, idxs))

    if not cited_sentences:
        return {
            "citation_precision": 0.0,
            "citations": 0,
            "verified": 0,
            "citation_coverage": 0.0,
        }

    verified = 0

    # If embeddings are available, use semantic similarity.
    if embed is not None and batch_cosine_similarity is not None:
        try:
            sent_embs = embed([s for s, _ in cited_sentences])
            threshold = float(getattr(settings, "CITATION_VERIFY_THRESHOLD", 0.38))

            for (sent, idxs), se in zip(cited_sentences, sent_embs):
                target_embs = [
                    docs[i].embedding
                    for i in idxs
                    if i < len(docs) and getattr(docs[i], "embedding", None) is not None
                ]

                if target_embs:
                    sims = batch_cosine_similarity(se, target_embs)
                    if sims and max(sims) >= threshold:
                        verified += 1
                    continue

                # Fallback lexical check if cited docs have no embedding.
                best_lex = max(
                    _lexical_overlap(sent, docs[i].title + " " + docs[i].abstract)
                    for i in idxs
                    if i < len(docs)
                )
                if best_lex >= 0.18:
                    verified += 1

        except Exception as e:
            log.warning("Citation embedding verification failed; using lexical fallback", error=str(e))
            verified = 0
            for sent, idxs in cited_sentences:
                best_lex = max(
                    _lexical_overlap(sent, docs[i].title + " " + docs[i].abstract)
                    for i in idxs
                    if i < len(docs)
                )
                if best_lex >= 0.18:
                    verified += 1
    else:
        # Pure lexical fallback.
        for sent, idxs in cited_sentences:
            best_lex = max(
                _lexical_overlap(sent, docs[i].title + " " + docs[i].abstract)
                for i in idxs
                if i < len(docs)
            )
            if best_lex >= 0.18:
                verified += 1

    citations = len(cited_sentences)
    citation_precision = verified / max(1, citations)
    citation_coverage = min(1.0, citations / total_claims)

    return {
        "citation_precision": round(citation_precision, 4),
        "citations": citations,
        "verified": verified,
        "citation_coverage": round(citation_coverage, 4),
    }


def _measure_claim_grounding(answer: str, docs: List[Paper]) -> Dict[str, Any]:
    """Measure faithfulness and hallucination by checking claims against docs."""
    claims = _split_claims(answer)

    if not answer or not answer.strip():
        return {
            "claims": 0,
            "grounded": 0,
            "ungrounded": 0,
            "faithfulness": 0.0,
            "measured_hallucination": 1.0,
            "mean_claim_support": 0.0,
            "ungrounded_claims": [],
        }

    if not docs:
        return {
            "claims": len(claims),
            "grounded": 0,
            "ungrounded": len(claims),
            "faithfulness": 0.0,
            "measured_hallucination": 1.0,
            "mean_claim_support": 0.0,
            "ungrounded_claims": claims[:8],
        }

    if not claims:
        return {
            "claims": 0,
            "grounded": 0,
            "ungrounded": 0,
            "faithfulness": 0.5,
            "measured_hallucination": 0.5,
            "mean_claim_support": 0.0,
            "ungrounded_claims": [],
        }

    doc_texts = [(d.title or "") + " " + (d.abstract or "") for d in docs[:10]]
    support_scores: List[float] = []
    grounded_flags: List[bool] = []

    # Semantic support if embeddings are available.
    semantic_scores: Optional[List[float]] = None
    if embed is not None and batch_cosine_similarity is not None:
        try:
            claim_embs = embed(claims)
            doc_embs = [
                d.embedding
                for d in docs[:10]
                if getattr(d, "embedding", None) is not None
            ]

            if doc_embs:
                semantic_scores = []
                for ce in claim_embs:
                    sims = batch_cosine_similarity(ce, doc_embs)
                    semantic_scores.append(max(sims) if sims else 0.0)
        except Exception as e:
            log.warning("Claim grounding embedding check failed", error=str(e))
            semantic_scores = None

    for idx, claim in enumerate(claims):
        lex = max((_lexical_overlap(claim, dt) for dt in doc_texts), default=0.0)

        if semantic_scores is not None and idx < len(semantic_scores):
            sem = semantic_scores[idx]
            # Blend semantic similarity with lexical overlap.
            support = max(sem, min(1.0, lex * 1.25))
            grounded = support >= 0.36 or lex >= 0.22
        else:
            support = min(1.0, lex * 1.35)
            grounded = lex >= 0.18

        support_scores.append(float(support))
        grounded_flags.append(bool(grounded))

    grounded_count = sum(1 for x in grounded_flags if x)
    ungrounded_count = len(claims) - grounded_count
    faithfulness = grounded_count / max(1, len(claims))
    measured_hallucination = ungrounded_count / max(1, len(claims))
    mean_support = sum(support_scores) / max(1, len(support_scores))

    ungrounded_claims = [
        c for c, g in zip(claims, grounded_flags) if not g
    ][:8]

    return {
        "claims": len(claims),
        "grounded": grounded_count,
        "ungrounded": ungrounded_count,
        "faithfulness": round(faithfulness, 4),
        "measured_hallucination": round(measured_hallucination, 4),
        "mean_claim_support": round(mean_support, 4),
        "ungrounded_claims": ungrounded_claims,
    }


def _compute_code_quality(
    answer: str,
    docs: List[Paper],
    relevance_score: float,
    faithfulness: float,
    citation_precision: float,
    hallucination: float,
    rewrite_count: int,
) -> float:
    """Q-learning reward. This must vary by strategy/session."""
    if not answer or not answer.strip():
        return 0.05

    length = len(answer)

    # Rewards detailed answers but saturates; does not require massive output.
    length_score = 1.0 / (1.0 + math.exp(-0.0045 * (length - 700)))

    # Check how many source titles/concepts are actually reflected.
    answer_lower = answer.lower()
    mentioned = 0
    for doc in docs[:8]:
        title_words = [
            w.lower()
            for w in _WORD_RE.findall(doc.title or "")
            if len(w) >= 5 and w.lower() not in STOPWORDS
        ]
        if title_words and any(w in answer_lower for w in title_words[:8]):
            mentioned += 1
    source_coverage = mentioned / max(1, min(len(docs), 8))

    words = [w.lower() for w in _WORD_RE.findall(answer)]
    lexical_diversity = len(set(words)) / max(1, len(words))
    diversity_score = _clamp((lexical_diversity - 0.25) / 0.35)

    rewrite_penalty = min(0.12, 0.04 * max(0, rewrite_count))

    quality = (
        0.30 * _clamp(faithfulness)
        + 0.18 * _clamp(citation_precision)
        + 0.20 * _clamp(relevance_score)
        + 0.14 * length_score
        + 0.10 * source_coverage
        + 0.08 * diversity_score
        - 0.18 * _clamp(hallucination)
        - rewrite_penalty
    )

    return round(max(0.05, min(0.99, quality)), 4)


async def _classify_intent_fast(query: str) -> str:
    cache_key = query.strip().lower()
    if cache_key in _INTENT_CACHE:
        return _INTENT_CACHE[cache_key]

    try:
        llm = get_llm()
        resp = await llm.complete(
            system=INTENT_CLASSIFY_PROMPT,
            user=query,
            max_tokens=24,
            temperature=0.0,
        )
        intent = resp.strip().strip('"').strip("'")
        if 2 <= len(intent.split()) <= 8 and "\n" not in intent:
            _INTENT_CACHE[cache_key] = intent
            return intent
    except Exception as e:
        log.warning("Intent classification failed", error=str(e))

    fallback = query[:60]
    _INTENT_CACHE[cache_key] = fallback
    return fallback


async def _llm_classify_and_evaluate(
    query: str,
    answer: str,
    docs: List[Paper],
) -> Tuple[str, float, float]:
    """LLM eval is only one signal; not trusted alone."""
    context = "\n\n".join(
        f"[{i + 1}] {d.title}: {d.abstract[:450]}"
        for i, d in enumerate(docs[:6])
    )

    user_msg = (
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer[:1400]}\n\n"
        f"Sources:\n{context}\n\n"
        "Evaluate strictly."
    )

    try:
        llm = get_llm()
        resp = await llm.complete(
            system=LLM_EVAL_PROMPT,
            user=user_msg,
            max_tokens=80,
            temperature=0.0,
        )

        intent = "Research query"
        confidence = 0.5
        hallucination = 0.25

        for line in resp.splitlines():
            line = line.strip()
            upper = line.upper()

            if upper.startswith("INTENT:"):
                parsed = line.split(":", 1)[1].strip()
                if parsed:
                    intent = parsed[:80]

            elif "CONFIDENCE" in upper:
                m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", line)
                if m:
                    confidence = _clamp(float(m.group()))

            elif "HALLUCINATION" in upper:
                m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", line)
                if m:
                    hallucination = _clamp(float(m.group()))

        return intent, confidence, hallucination

    except Exception as e:
        log.warning("LLM eval failed, using neutral defaults", error=str(e))
        return "Research query", 0.5, 0.25


def _calibrate_display_metrics(
    relevance_score: float,
    grounding: Dict[str, Any],
    citation: Dict[str, Any],
    llm_confidence: float,
    llm_hallucination: float,
) -> Tuple[float, float]:
    """Combine measured signals into display confidence/hallucination."""
    faithfulness = _clamp(grounding.get("faithfulness", 0.0))
    measured_hall = _clamp(grounding.get("measured_hallucination", 1.0))
    citation_precision = _clamp(citation.get("citation_precision", 0.0))
    citation_coverage = _clamp(citation.get("citation_coverage", 0.0))

    # Missing citations increases risk for a cited research assistant.
    citation_risk = 1.0 - (0.65 * citation_precision + 0.35 * citation_coverage)

    confidence = (
        0.42 * faithfulness
        + 0.20 * citation_precision
        + 0.18 * citation_coverage
        + 0.12 * _clamp(relevance_score)
        + 0.08 * _clamp(llm_confidence)
    )

    hallucination = (
        0.62 * measured_hall
        + 0.20 * _clamp(llm_hallucination)
        + 0.18 * citation_risk
    )

    # Important:
    # This prevents the UI from showing fake-looking 0% for nearly every answer.
    # It represents residual uncertainty because we only have abstracts/chunks,
    # not full papers and not a formal proof checker.
    claims = int(grounding.get("claims", 0) or 0)
    if claims > 0:
        if citation.get("citations", 0) > 0:
            hallucination = max(hallucination, 0.015)
        else:
            hallucination = max(hallucination, 0.04)

    return round(_clamp(confidence), 4), round(_clamp(hallucination), 4)


async def _count_sessions(db: AsyncSession) -> int:
    result = await db.execute(select(func.count(ResearchSession.id)))
    return result.scalar() or 0


async def _call_record_experience(
    session_id: str,
    intent: str,
    strategy: str,
    code_quality: float,
    db: AsyncSession,
    was_exploring: bool,
) -> Dict[str, Any]:
    """Handle both positional and keyword signatures of record_experience."""
    try:
        return await record_experience(
            session_id=session_id,
            intent=intent,
            strategy=strategy,
            outcome_quality=code_quality,
            db=db,
            was_exploring=was_exploring,
        )
    except TypeError:
        return await record_experience(
            session_id,
            intent,
            strategy,
            code_quality,
            db,
            was_exploring,
        )


async def run_agent(
    query: str,
    db: AsyncSession,
    session_number: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    sn = session_number or ((await _count_sessions(db)) + 1)

    # 1. Intent
    pre_intent = await _classify_intent_fast(query)
    log.info("Agent started", query=query[:80], intent=pre_intent, session=sn)

    # 2. Strategy selection
    strategy, was_exploring = await select_strategy(pre_intent, db)
    memory_before = await get_best_strategy_for_intent(pre_intent, db)

    # 3. Ingest and retrieve
    await ingest_query(query, db)

    retrieval_result = await retrieve(query, db, strategy=strategy)

    # Supports old and new retriever return formats.
    if isinstance(retrieval_result, tuple) and len(retrieval_result) == 4:
        docs, rewrite_count, relevance_score, per_doc_scores = retrieval_result
    else:
        docs, rewrite_count, relevance_score = retrieval_result
        per_doc_scores = None

    docs = list(docs or [])
    rewrite_count = int(rewrite_count or 0)
    relevance_score = _clamp(relevance_score)

    # 4. GraphRAG expansion for graph/hybrid
    if docs and strategy in ("graph", "hybrid"):
        try:
            seed_ids = [d.arxiv_id.split(":")[0] for d in docs[:5]]
            await build_graph(db)
            expanded_ids = graph_expand(seed_ids, hops=2, max_nodes=12)

            existing_ids = {d.id for d in docs}
            for arxiv_id in expanded_ids:
                result = await db.execute(
                    select(Paper).where(Paper.arxiv_id.like(f"{arxiv_id}%")).limit(2)
                )
                extra = result.scalars().all()
                for p in extra:
                    if p.id not in existing_ids:
                        docs.append(p)
                        existing_ids.add(p.id)

            docs = docs[:12]
        except Exception as e:
            log.warning("Graph expansion failed; continuing without expansion", error=str(e))

    # 5. Build stronger generation prompt
    memory_hint = ""
    best_q = float(memory_before.get("best_q", 0) or 0)
    best_strategy = memory_before.get("best_strategy")

    if best_q >= 0.55 and not was_exploring and best_strategy:
        memory_hint = (
            f"\n\nMemory note: Previous sessions for this topic found that "
            f"'{best_strategy}' retrieval worked well with quality about {best_q:.0%}. "
            f"Use the sources carefully and prioritize grounding."
        )
    elif was_exploring:
        memory_hint = (
            f"\n\nExploration note: This run is testing the '{strategy}' retrieval strategy. "
            f"Still provide the best possible grounded answer."
        )

    context = "\n\n".join(
        f"[{i + 1}] Title: {d.title}\n"
        f"Authors: {', '.join((d.authors or [])[:5]) if getattr(d, 'authors', None) else 'Unknown'}\n"
        f"Abstract: {(d.abstract or '')[:700]}"
        for i, d in enumerate(docs[:10])
    )

    user_prompt = (
        f"Research question:\n{query}\n\n"
        f"Retrieved source documents ({len(docs)} total, {min(len(docs), 10)} shown):\n"
        f"{context if context else 'No source documents were retrieved.'}\n\n"
        "Write a comprehensive, carefully grounded research answer. "
        "Use inline citations like [1] after source-supported claims. "
        "If sources are weak or missing, say so explicitly."
        f"{memory_hint}"
    )

    llm = get_llm()

    # Force enough space for a real answer even if config is still 1024.
    max_tokens = max(int(getattr(settings, "LLM_MAX_TOKENS", 1024) or 1024), 1800)

    answer = await llm.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=max_tokens,
        temperature=float(getattr(settings, "LLM_TEMPERATURE", 0.2) or 0.2),
    )

    # 6. Evaluation
    llm_intent, llm_confidence, llm_hallucination = await _llm_classify_and_evaluate(
        query=query,
        answer=answer,
        docs=docs,
    )

    # Prefer pre_intent if LLM eval returns generic text.
    intent = llm_intent if llm_intent and llm_intent != "Research query" else pre_intent

    grounding = _measure_claim_grounding(answer, docs)
    citation = _verify_citations(answer, docs)

    confidence, hallucination = _calibrate_display_metrics(
        relevance_score=relevance_score,
        grounding=grounding,
        citation=citation,
        llm_confidence=llm_confidence,
        llm_hallucination=llm_hallucination,
    )

    # 7. The missing critical line: compute code_quality before using it.
    code_quality = _compute_code_quality(
        answer=answer,
        docs=docs,
        relevance_score=relevance_score,
        faithfulness=grounding["faithfulness"],
        citation_precision=citation["citation_precision"],
        hallucination=hallucination,
        rewrite_count=rewrite_count,
    )

    eval_breakdown = {
        "claims": grounding["claims"],
        "grounded_claims": grounding["grounded"],
        "ungrounded_claims_count": grounding["ungrounded"],
        "ungrounded_claims": grounding["ungrounded_claims"],
        "faithfulness": grounding["faithfulness"],
        "measured_hallucination": grounding["measured_hallucination"],
        "display_hallucination": hallucination,
        "mean_claim_support": grounding["mean_claim_support"],
        "citations": citation["citations"],
        "verified_citations": citation["verified"],
        "citation_precision": citation["citation_precision"],
        "citation_coverage": citation["citation_coverage"],
        "llm_confidence": llm_confidence,
        "llm_hallucination": llm_hallucination,
        "retrieval_score": relevance_score,
        "rewrite_count": rewrite_count,
        "answer_length": len(answer or ""),
        "per_doc_scores": per_doc_scores,
    }

    log.info(
        "Evaluation complete",
        confidence=confidence,
        hallucination=hallucination,
        code_quality=code_quality,
        faithfulness=grounding["faithfulness"],
        citation_precision=citation["citation_precision"],
        retrieval_score=relevance_score,
    )

    # 8. Persist session.
    session_values = _model_kwargs(
        ResearchSession,
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

        # Newer schema fields, safely ignored if your model does not have them.
        sources_count=len(docs),
        outcome_quality=code_quality,
        faithfulness=grounding["faithfulness"],
        citation_precision=citation["citation_precision"],
        eval_breakdown=eval_breakdown,
    )

    session = ResearchSession(**session_values)
    db.add(session)
    await db.flush()

    # 9. MemRL update.
    q_debug = await _call_record_experience(
        session_id=session.id,
        intent=intent,
        strategy=strategy,
        code_quality=code_quality,
        db=db,
        was_exploring=was_exploring,
    )

    memory_after = await get_best_strategy_for_intent(intent, db)

    # 10. Failure logging.
    if code_quality < 0.4 or hallucination > 0.45:
        better = next(
            (s for s in ["hybrid", "graph", "semantic", "aggressive_rewrite"] if s != strategy),
            "hybrid",
        )

        if hallucination > 0.45:
            failure_type = "high_hallucination"
        elif grounding["faithfulness"] < 0.5:
            failure_type = "low_faithfulness"
        elif relevance_score < 0.35:
            failure_type = "weak_retrieval"
        else:
            failure_type = "low_quality"

        await log_failure(
            session.id,
            failure_type,
            (
                f"Quality={code_quality:.2f}, hallucination={hallucination:.2f}, "
                f"faithfulness={grounding['faithfulness']:.2f}, "
                f"citation_precision={citation['citation_precision']:.2f}, "
                f"retrieval={relevance_score:.2f}, strategy='{strategy}'"
            ),
            strategy,
            better,
            q_debug.get("delta_q", 0.0),
            db,
        )

    # 11. Benchmark.
    benchmark_values = _model_kwargs(
        BenchmarkRun,
        session_number=sn,
        avg_confidence=confidence,
        avg_hallucination=hallucination,
        answer_quality=code_quality,
        retrieval_precision=relevance_score,
        faithfulness=grounding["faithfulness"],
        citation_precision=citation["citation_precision"],
        total_queries=1,
    )

    db.add(BenchmarkRun(**benchmark_values))
    await db.commit()

    duration_ms = int((time.time() - t0) * 1000)

    return {
        "session_id": session.id,
        "session_number": sn,
        "query": query,
        "intent": intent,
        "answer": answer,
        "confidence": round(confidence, 3),
        "hallucination_score": round(hallucination, 3),
        "hallucination_percent": round(hallucination * 100, 1),
        "outcome_quality": round(code_quality, 3),
        "retrieval_strategy": strategy,
        "papers_retrieved": len(docs),
        "sources_count": len(docs),
        "rewrite_count": rewrite_count,
        "relevance_score": round(relevance_score, 3),
        "faithfulness": round(grounding["faithfulness"], 3),
        "citation_precision": round(citation["citation_precision"], 3),
        "q_value_after": round(float(q_debug.get("new_q", 0.0)), 3),
        "duration_ms": duration_ms,
        "eval_breakdown": eval_breakdown,
        "memrl_debug": {
            "pre_intent": pre_intent,
            "post_intent": intent,
            "intent_bucket": _fuzzy_bucket(intent),
            "was_exploring": was_exploring,
            "strategy_selected": strategy,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "q_old": q_debug.get("old_q"),
            "q_new": q_debug.get("new_q"),
            "q_delta": q_debug.get("delta_q"),
            "visit_count": q_debug.get("visit_count"),
            "code_quality": round(code_quality, 3),
            "confidence": round(confidence, 3),
            "hallucination": round(hallucination, 3),
            "quality_breakdown": {
                "faithfulness": round(grounding["faithfulness"], 3),
                "measured_hallucination": round(grounding["measured_hallucination"], 3),
                "citation_precision": round(citation["citation_precision"], 3),
                "citation_coverage": round(citation["citation_coverage"], 3),
                "retrieval_score": round(relevance_score, 3),
                "answer_length": len(answer or ""),
                "docs_retrieved": len(docs),
                "rewrite_count": rewrite_count,
            },
        },
        "sources": [
            {
                "title": d.title,
                "arxiv_id": d.arxiv_id,
                "url": d.url,
                "authors": d.authors[:3] if getattr(d, "authors", None) else [],
                "abstract_snippet": (d.abstract or "")[:260],
            }
            for d in docs[:10]
        ],
    }
