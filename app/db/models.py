"""SQLAlchemy ORM models for evaluation_runs, answer_results, claim_results."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EvaluationRun(Base):
    """One experiment run over a set of answers (development or check split)."""

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    split: Mapped[str] = mapped_column(String(32), nullable=False)          # development | check
    agent_version: Mapped[str] = mapped_column(String(16), nullable=False)  # v1 | v2
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(16), nullable=False)
    fact_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(16), nullable=False)
    scheduled_answers: Mapped[int] = mapped_column(Integer, nullable=False)
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Gold-dependent metrics – NULL for check runs (never zero)
    G: Mapped[int | None] = mapped_column(Integer, nullable=True)
    P: Mapped[int | None] = mapped_column(Integer, nullable=True)
    M: Mapped[int | None] = mapped_column(Integer, nullable=True)
    C: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    joint_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    joint_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    joint_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    gold_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Convenience counts (always available)
    answers_ok: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    answers_empty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    answers_error: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claims_extracted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    answer_results: Mapped[list["AnswerResult"]] = relationship(
        "AnswerResult", back_populates="run", cascade="all, delete-orphan",
        order_by="AnswerResult.answer_id",
    )


class AnswerResult(Base):
    """One terminal outcome per scheduled answer per run."""

    __tablename__ = "answer_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", name="uq_answer_result_run_answer"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    answer_id: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)   # ok | valid_empty | error
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_call_details: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    claim_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    langfuse_observation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    run: Mapped["EvaluationRun"] = relationship("EvaluationRun", back_populates="answer_results")
    claim_results: Mapped[list["ClaimResult"]] = relationship(
        "ClaimResult", back_populates="answer_result", cascade="all, delete-orphan",
        order_by="(ClaimResult.start_char, ClaimResult.end_char)",
    )


class ClaimResult(Base):
    """One extracted claim per answer result. Gold-match fields are NULL for check runs."""

    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint("answer_result_id", "claim_id", name="uq_claim_result_answer_claim"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    answer_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("answer_results.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    claim_id: Mapped[str] = mapped_column(String(32), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(16), nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_fact_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Gold-dependent – SQL NULL for check runs
    expected_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    gold_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    answer_result: Mapped["AnswerResult"] = relationship(
        "AnswerResult", back_populates="claim_results"
    )
