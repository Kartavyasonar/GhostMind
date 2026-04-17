"""MemRL episodic memory engine with Q-value TD learning."""
import hashlib
import random
from typing import Dict, List
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.core.models import EpisodicTriplet, FailureLog

log = structlog.get_logger()

STRATEGIES = ["semantic", "graph", "hybrid", "aggressive_rewrite"]


def _intent_hash(intent: str) -> str:
    normalised = " ".join(intent.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


async def select_strategy(intent: str, db: AsyncSession) -> str:
    if random.random() < settings.MEMRL_EPSILON:
        strategy = random.choice(STRATEGIES)
        log.info("MemRL: exploring", strategy=strategy)
        return strategy

    intent_h = _intent_hash(intent)
    result = await db.execute(
        select(EpisodicTriplet)
        .where(EpisodicTriplet.intent_hash == intent_h)
        .order_by(EpisodicTriplet.q_value.desc())
        .limit(1)
    )
    best = result.scalar_one_or_none()

    if best:
        log.info("MemRL: exploiting", strategy=best.strategy, q_value=round(best.q_value, 3))
        return best.strategy

    return "semantic"


async def record_experience(
    session_id: str,
    intent: str,
    strategy: str,
    outcome_quality: float,
    db: AsyncSession,
) -> float:
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

    if existing:
        alpha = settings.MEMRL_ALPHA
        gamma = settings.MEMRL_GAMMA
        old_q = existing.q_value
        new_q = old_q + alpha * (outcome_quality + gamma * outcome_quality - old_q)
        new_q = max(0.0, min(1.0, new_q))

        existing.q_value = new_q
        existing.outcome_quality = outcome_quality
        existing.visit_count += 1
        db.add(existing)
        log.info("MemRL: Q updated", strategy=strategy, old=round(old_q, 3), new=round(new_q, 3))
        return new_q
    else:
        triplet = EpisodicTriplet(
            session_id=session_id,
            intent_hash=intent_h,
            intent_text=intent[:500],
            strategy=strategy,
            outcome_quality=outcome_quality,
            q_value=outcome_quality,
            visit_count=1,
        )
        db.add(triplet)
        log.info("MemRL: new triplet", strategy=strategy, q=round(outcome_quality, 3))
        return outcome_quality


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
