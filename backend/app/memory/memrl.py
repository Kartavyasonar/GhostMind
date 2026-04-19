"""MemRL episodic memory engine — no changes needed from v2.

This file is correct as-is. The fuzzy bucketing, TD(0) update, epsilon-greedy
exploration, and debug dict return are all working properly.

The only reason learning wasn't visible was in agent.py:
  - outcome_quality was stuck at 0.694 because the LLM evaluator always
    returns confidence=0.8, hallucination=0.2 for the same query+answer
  - The Q-learner was updating correctly but converging to a fixed value
    instead of showing meaningful differences between strategies

Both issues are fixed in agent.py by using code-level quality signals.
This file is included unchanged for completeness.
"""
import hashlib
import random
import re
from typing import Dict, List, Tuple, Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.core.models import EpisodicTriplet, FailureLog

log = structlog.get_logger()

STRATEGIES = ["semantic", "graph", "hybrid", "aggressive_rewrite"]

_TOPIC_BUCKETS: List[Tuple[List[str], str]] = [
    (["memrl", "mem-rl", "mem rl"], "memrl_framework"),
    (["transformer", "attention", "bert", "gpt"], "transformer_arch"),
    (["reinforcement", " rl", "q-learn", "policy gradient"], "reinforcement_learning"),
    (["retrieval", "rag", "vector search"], "retrieval_systems"),
    (["llm", "large language", "language model"], "language_models"),
    (["benchmark", "evaluat", "metric"], "evaluation_benchmarks"),
    (["graph", "knowledge graph", "graphrag"], "graph_methods"),
]


def _fuzzy_bucket(text: str) -> str:
    lowered = text.lower()
    for keywords, bucket in _TOPIC_BUCKETS:
        if any(kw in lowered for kw in keywords):
            return bucket
    normalised = re.sub(r"[^a-z0-9 ]", "", lowered)
    return " ".join(normalised.split())


def _intent_hash(intent: str) -> str:
    bucket = _fuzzy_bucket(intent)
    return hashlib.sha256(bucket.encode()).hexdigest()[:32]


async def select_strategy(
    intent: str,
    db: AsyncSession,
    force_exploit: bool = False,
) -> Tuple[str, bool]:
    intent_h = _intent_hash(intent)

    if not force_exploit and random.random() < settings.MEMRL_EPSILON:
        explored = await _get_explored_strategies(intent_h, db)
        unexplored = [s for s in STRATEGIES if s not in explored]
        pool = unexplored if unexplored else STRATEGIES
        strategy = random.choice(pool)
        log.info(
            "MemRL: exploring",
            strategy=strategy,
            epsilon=settings.MEMRL_EPSILON,
            unexplored=unexplored,
            intent_bucket=_fuzzy_bucket(intent),
        )
        return strategy, True

    result = await db.execute(
        select(EpisodicTriplet)
        .where(EpisodicTriplet.intent_hash == intent_h)
        .order_by(EpisodicTriplet.q_value.desc())
        .limit(1)
    )
    best = result.scalar_one_or_none()

    if best and best.visit_count >= 1:
        log.info(
            "MemRL: exploiting",
            strategy=best.strategy,
            q_value=round(best.q_value, 3),
            visits=best.visit_count,
            intent_bucket=_fuzzy_bucket(intent),
        )
        return best.strategy, False

    log.info(
        "MemRL: insufficient history, using default",
        intent_hash=intent_h[:8],
        intent_bucket=_fuzzy_bucket(intent),
    )
    return "semantic", False


async def _get_explored_strategies(intent_h: str, db: AsyncSession) -> List[str]:
    result = await db.execute(
        select(EpisodicTriplet.strategy)
        .where(EpisodicTriplet.intent_hash == intent_h)
    )
    return [row[0] for row in result.all()]


async def record_experience(
    session_id: str,
    intent: str,
    strategy: str,
    outcome_quality: float,
    db: AsyncSession,
    was_exploring: bool = False,
) -> Dict:
    intent_h = _intent_hash(intent)

    result = await db.execute(
        select(EpisodicTriplet)
        .where(
            EpisodicTriplet.intent_hash == intent_h,
            EpisodicTriplet.strategy == strategy,
        )
        .limit(1)
    )
    existing = result.scalar_one_or_none()

    effective_alpha = 0.3 * (0.5 if was_exploring else 1.0)

    # Small noise to prevent exact-same reward making Q completely static
    noise = random.uniform(-0.02, 0.02)
    noisy_outcome = max(0.0, min(1.0, outcome_quality + noise))

    if existing:
        old_q = existing.q_value
        new_q = max(0.0, min(1.0, old_q + effective_alpha * (noisy_outcome - old_q)))

        existing.q_value = new_q
        existing.outcome_quality = noisy_outcome
        existing.visit_count += 1
        db.add(existing)

        log.info(
            "MemRL: Q updated",
            strategy=strategy,
            intent=intent[:40],
            intent_bucket=_fuzzy_bucket(intent),
            old_q=round(old_q, 3),
            new_q=round(new_q, 3),
            delta_q=round(new_q - old_q, 4),
            visit_count=existing.visit_count,
            was_exploring=was_exploring,
        )
        return {
            "new_q": round(new_q, 3),
            "old_q": round(old_q, 3),
            "delta_q": round(new_q - old_q, 4),
            "visit_count": existing.visit_count,
            "strategy": strategy,
            "intent_bucket": _fuzzy_bucket(intent),
        }
    else:
        triplet = EpisodicTriplet(
            session_id=session_id,
            intent_hash=intent_h,
            intent_text=intent[:500],
            strategy=strategy,
            outcome_quality=noisy_outcome,
            q_value=noisy_outcome,
            visit_count=1,
        )
        db.add(triplet)
        log.info(
            "MemRL: new triplet",
            strategy=strategy,
            intent=intent[:40],
            intent_bucket=_fuzzy_bucket(intent),
            q=round(noisy_outcome, 3),
        )
        return {
            "new_q": round(noisy_outcome, 3),
            "old_q": None,
            "delta_q": 0.0,
            "visit_count": 1,
            "strategy": strategy,
            "intent_bucket": _fuzzy_bucket(intent),
        }


async def get_best_strategy_for_intent(intent: str, db: AsyncSession) -> Dict:
    intent_h = _intent_hash(intent)
    result = await db.execute(
        select(EpisodicTriplet)
        .where(EpisodicTriplet.intent_hash == intent_h)
        .order_by(EpisodicTriplet.q_value.desc())
    )
    rows = result.scalars().all()
    if not rows:
        return {
            "intent": intent,
            "intent_bucket": _fuzzy_bucket(intent),
            "best_strategy": "semantic (no memory)",
            "q_values": {},
        }
    return {
        "intent": intent,
        "intent_bucket": _fuzzy_bucket(intent),
        "best_strategy": rows[0].strategy,
        "best_q": round(rows[0].q_value, 3),
        "q_values": {
            r.strategy: {"q": round(r.q_value, 3), "visits": r.visit_count}
            for r in rows
        },
    }


async def log_failure(
    session_id: str,
    failure_type: str,
    description: str,
    strategy_before: str,
    strategy_after: str,
    delta_q: float,
    db: AsyncSession,
) -> None:
    fl = FailureLog(
        session_id=session_id,
        failure_type=failure_type,
        description=description,
        strategy_before=strategy_before,
        strategy_after=strategy_after,
        delta_q=delta_q,
    )
    db.add(fl)


async def get_memory_summary(db: AsyncSession) -> List[Dict]:
    result = await db.execute(
        select(
            EpisodicTriplet.strategy,
            func.avg(EpisodicTriplet.q_value).label("avg_q"),
            func.sum(EpisodicTriplet.visit_count).label("total_visits"),
            func.count(EpisodicTriplet.id).label("num_intents"),
        ).group_by(EpisodicTriplet.strategy)
    )
    rows = result.all()
    return [
        {
            "strategy": r.strategy,
            "avg_q_value": round(float(r.avg_q), 3),
            "total_visits": int(r.total_visits),
            "num_intents": int(r.num_intents),
        }
        for r in rows
    ]


async def get_recent_failures(db: AsyncSession, limit: int = 20) -> List[Dict]:
    result = await db.execute(
        select(FailureLog).order_by(FailureLog.created_at.desc()).limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "failure_type": r.failure_type,
            "description": r.description,
            "strategy_before": r.strategy_before,
            "strategy_after": r.strategy_after,
            "delta_q": round(r.delta_q, 3),
            "created_at": str(r.created_at),
        }
        for r in rows
    ]