"""Data access layer. All DB writes and reads are parameterized through SQLAlchemy ORM."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import AnswerResult, ClaimResult, EvaluationRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# EvaluationRun
# ---------------------------------------------------------------------------

def create_run(
    db: Session,
    *,
    run_id: uuid.UUID | None = None,
    split: str,
    agent_version: str,
    provider: str,
    model: Optional[str],
    prompt_version: str,
    fact_version: str,
    evaluator_version: str,
    scheduled_answers: int,
    settings: dict,
) -> EvaluationRun:
    run = EvaluationRun(
        id=run_id or uuid.uuid4(),
        split=split,
        agent_version=agent_version,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        fact_version=fact_version,
        evaluator_version=evaluator_version,
        scheduled_answers=scheduled_answers,
        settings=settings,
        status="running",
    )
    db.add(run)
    db.flush()
    return run


def finish_run(
    db: Session,
    run: EvaluationRun,
    *,
    answers_ok: int,
    answers_empty: int,
    answers_error: int,
    claims_extracted: int,
    # Gold metrics – pass None for check runs
    G: Optional[int] = None,
    P: Optional[int] = None,
    M: Optional[int] = None,
    C: Optional[int] = None,
    extraction_precision: Optional[float] = None,
    extraction_recall: Optional[float] = None,
    extraction_f1: Optional[float] = None,
    joint_precision: Optional[float] = None,
    joint_recall: Optional[float] = None,
    joint_f1: Optional[float] = None,
    macro_f1: Optional[float] = None,
    gold_coverage: Optional[float] = None,
) -> None:
    run.status = "complete"
    run.finished_at = _utcnow()
    run.answers_ok = answers_ok
    run.answers_empty = answers_empty
    run.answers_error = answers_error
    run.claims_extracted = claims_extracted
    run.G = G
    run.P = P
    run.M = M
    run.C = C
    run.extraction_precision = extraction_precision
    run.extraction_recall = extraction_recall
    run.extraction_f1 = extraction_f1
    run.joint_precision = joint_precision
    run.joint_recall = joint_recall
    run.joint_f1 = joint_f1
    run.macro_f1 = macro_f1
    run.gold_coverage = gold_coverage
    db.flush()


def set_run_trace_id(db: Session, run: EvaluationRun, trace_id: str | None) -> None:
    run.langfuse_trace_id = trace_id
    db.flush()


def get_run(db: Session, run_id: uuid.UUID) -> Optional[EvaluationRun]:
    return db.get(EvaluationRun, run_id)


# ---------------------------------------------------------------------------
# AnswerResult
# ---------------------------------------------------------------------------

def create_answer_result(
    db: Session,
    *,
    run_id: uuid.UUID,
    answer_id: str,
    status: str,
    error_code: Optional[str],
    error_message: Optional[str],
    turns: int,
    tool_calls: list,
    claim_count: int,
    tool_call_details: list | None = None,
    langfuse_trace_id: str | None = None,
    langfuse_observation_id: str | None = None,
) -> AnswerResult:
    ar = AnswerResult(
        run_id=run_id,
        answer_id=answer_id,
        status=status,
        error_code=error_code,
        error_message=error_message,
        turns=turns,
        tool_calls=tool_calls,
        tool_call_details=tool_call_details or [],
        claim_count=claim_count,
        langfuse_trace_id=langfuse_trace_id,
        langfuse_observation_id=langfuse_observation_id,
    )
    db.add(ar)
    db.flush()
    return ar


# ---------------------------------------------------------------------------
# ClaimResult
# ---------------------------------------------------------------------------

def create_claim_result(
    db: Session,
    *,
    answer_result_id: uuid.UUID,
    run_id: uuid.UUID,
    claim_id: str,
    answer_id: str,
    quote: str,
    start_char: int,
    end_char: int,
    label: str,
    evidence_fact_ids: list,
    expected_label: Optional[str] = None,
    correct: Optional[bool] = None,
    gold_match: Optional[bool] = None,
) -> ClaimResult:
    cr = ClaimResult(
        answer_result_id=answer_result_id,
        run_id=run_id,
        claim_id=claim_id,
        answer_id=answer_id,
        quote=quote,
        start_char=start_char,
        end_char=end_char,
        label=label,
        evidence_fact_ids=evidence_fact_ids,
        expected_label=expected_label,
        correct=correct,
        gold_match=gold_match,
    )
    db.add(cr)
    db.flush()
    return cr
