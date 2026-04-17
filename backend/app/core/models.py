"""ORM models for GhostMind — SQLite compatible."""
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    String, Text, Float, Integer, DateTime, JSON,
    ForeignKey, Index, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[List] = mapped_column(JSON, default=list)
    categories: Mapped[List] = mapped_column(JSON, default=list)
    published: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List]] = mapped_column(JSON, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    citations: Mapped[List["Citation"]] = relationship(
        "Citation", back_populates="paper", foreign_keys="Citation.paper_id"
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    paper_id: Mapped[str] = mapped_column(String(64), ForeignKey("papers.id"), index=True)
    cites_arxiv_id: Mapped[str] = mapped_column(String(64), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    paper: Mapped["Paper"] = relationship("Paper", foreign_keys=[paper_id], back_populates="citations")


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    hallucination_score: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    papers_retrieved: Mapped[List] = mapped_column(JSON, default=list)
    rewrite_count: Mapped[int] = mapped_column(Integer, default=0)
    session_number: Mapped[int] = mapped_column(Integer, default=1)
    feedback_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    triplets: Mapped[List["EpisodicTriplet"]] = relationship(
        "EpisodicTriplet", back_populates="session"
    )


class EpisodicTriplet(Base):
    __tablename__ = "episodic_triplets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("research_sessions.id"), index=True)
    intent_hash: Mapped[str] = mapped_column(String(64), index=True)
    intent_text: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome_quality: Mapped[float] = mapped_column(Float, default=0.0)
    q_value: Mapped[float] = mapped_column(Float, default=0.5)
    visit_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    session: Mapped["ResearchSession"] = relationship("ResearchSession", back_populates="triplets")

    __table_args__ = (
        Index("ix_triplet_intent_strategy", "intent_hash", "strategy"),
    )


class FailureLog(Base):
    __tablename__ = "failure_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("research_sessions.id"), index=True)
    failure_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    strategy_before: Mapped[str] = mapped_column(String(64))
    strategy_after: Mapped[str] = mapped_column(String(64))
    delta_q: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=gen_uuid)
    session_number: Mapped[int] = mapped_column(Integer, index=True)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    avg_hallucination: Mapped[float] = mapped_column(Float, default=0.0)
    answer_quality: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_precision: Mapped[float] = mapped_column(Float, default=0.0)
    total_queries: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
