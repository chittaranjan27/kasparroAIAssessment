"""POST /evaluation-runs  and  GET /evaluation-runs/{id}

POST  /evaluation-runs
  Accepts { "split": "development" | "check", "agent_version": "v1" | "v2" }
  Runs the frozen agent on all answers for that split, persists every outcome and
  claim, then returns HTTP 201 with run_id, status, counts and available metrics.

GET   /evaluation-runs/{id}
  Returns the full run with ordered answer_results and claim_results.
  404 if the run_id is not found; 422 (via FastAPI) for a non-UUID id.
"""
from __future__ import annotations

import functools
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.db.repository import (
    create_answer_result,
    create_claim_result,
    create_run,
    finish_run,
    get_run,
)
from app.db.session import get_session
from app.evaluation.runs import EVALUATOR_VERSION, make_provider
from app.evaluation.runs import development_score_values
from app.schemas import Split
from app.tracing import attach_development_scores, flush_traces

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EvaluationRunRequest(BaseModel):
    split: str = Field(..., pattern=r"^(development|check)$")
    agent_version: str = Field(..., pattern=r"^v[12]$")


class AnswerResultOut(BaseModel):
    id: str
    answer_id: str
    status: str
    error_code: Optional[str]
    error_message: Optional[str]
    turns: int
    tool_calls: list
    tool_call_details: list[dict]
    langfuse_trace_id: Optional[str]
    langfuse_observation_id: Optional[str]
    claim_count: int
    claims: list[dict]


class EvaluationRunOut(BaseModel):
    id: str
    status: str
    split: str
    agent_version: str
    provider: str
    model: Optional[str]
    prompt_version: str
    fact_version: str
    evaluator_version: str
    scheduled_answers: int
    answers_ok: int
    answers_empty: int
    answers_error: int
    claims_extracted: int
    # Gold-dependent – null for check runs
    G: Optional[int]
    P: Optional[int]
    M: Optional[int]
    C: Optional[int]
    extraction_precision: Optional[float]
    extraction_recall: Optional[float]
    extraction_f1: Optional[float]
    joint_precision: Optional[float]
    joint_recall: Optional[float]
    joint_f1: Optional[float]
    macro_f1: Optional[float]
    gold_coverage: Optional[float]
    started_at: str
    finished_at: Optional[str]
    langfuse_trace_id: Optional[str]
    answer_results: list[AnswerResultOut] = []


# ---------------------------------------------------------------------------
# Helper: load shared catalogue + provider
# ---------------------------------------------------------------------------

def _get_shared_catalogue():
    from app.agent.fact_tool import FactCatalogue
    from app.config import get_settings
    from app.evaluation import data_loader
    s = get_settings()
    facts = data_loader.load_brand_facts(s.brand_facts_path, s.expected_facts)
    return FactCatalogue(facts, s.fact_version)


def _prompt_for(version: str) -> str:
    if version == "v1":
        from app.agent.prompts_v1 import SYSTEM_PROMPT
    else:
        from app.agent.prompts_v2 import SYSTEM_PROMPT
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# POST /evaluation-runs
# ---------------------------------------------------------------------------

@router.post(
    "/evaluation-runs",
    status_code=status.HTTP_201_CREATED,
    response_model=EvaluationRunOut,
    summary="Run the frozen agent on all answers for a split and persist results",
)
async def post_evaluation_run(body: EvaluationRunRequest):
    from app.agent.provider import AsyncOpenAIProvider
    from app.agent.runner import run_answer
    from app.config import get_settings
    from app.evaluation import data_loader
    from app.evaluation.runs import execute_run, run_metadata

    s = get_settings()
    split_enum = Split.DEVELOPMENT if body.split == "development" else Split.CHECK
    system_prompt = _prompt_for(body.agent_version)
    catalogue = _get_shared_catalogue()

    answers_all = data_loader.load_answers(s.answers_path, s.expected_answers)
    data_loader.validate_answer_splits(answers_all)
    answers = sorted(
        [a for a in answers_all if a.split is split_enum],
        key=lambda a: a.answer_id,
    )

    inner = AsyncOpenAIProvider.from_env()
    provider = make_provider(inner)
    meta = run_metadata(
        provider_name="openai",
        model=inner.model,
        fact_version=s.fact_version,
        data_files={"answers.csv": s.answers_path, "brand_facts.csv": s.brand_facts_path},
        root=s.data_dir.parent,
        prompt_version=body.agent_version,
        system_prompt=system_prompt,
    )

    db = get_session()
    try:
        db_run = create_run(
            db,
            split=body.split,
            agent_version=body.agent_version,
            provider="openai",
            model=inner.model,
            prompt_version=body.agent_version,
            fact_version=s.fact_version,
            evaluator_version=EVALUATOR_VERSION,
            scheduled_answers=len(answers),
            settings=meta.get("settings", {}),
        )
        db.commit()
        run_id = db_run.id

        agent = functools.partial(run_answer, system_prompt=system_prompt)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            execution = await execute_run(
                answers,
                provider,
                catalogue,
                Path(tmp),
                meta,
                run_id=str(run_id),
                split=body.split,
                agent_version=body.agent_version,
                agent=agent,
            )
        predictions = execution.predictions

        # Join gold for development split only
        gold_by_span: dict[tuple, Any] = {}
        if body.split == "development":
            gold_claims = data_loader.load_gold(s.gold_path, answers_all, s.expected_gold)
            gold_by_span = {
                (g.answer_id, g.start_char, g.end_char): g for g in gold_claims
            }

        ok = empty = error = total_claims = 0
        for pred in predictions:
            outcome = execution.outcomes[pred.answer_id]
            if pred.error:
                st = "error"
                error += 1
            elif not pred.claims:
                st = "valid_empty"
                empty += 1
            else:
                st = "ok"
                ok += 1

            total_claims += len(pred.claims)

            # Find matching outcome for turn/tool_calls
            # (predictions list is ordered; use answer_id to look up outcome)
            db2 = get_session()
            try:
                ar = create_answer_result(
                    db2,
                    run_id=run_id,
                    answer_id=pred.answer_id,
                    status=st,
                    error_code=pred.error,
                    error_message=None,  # sanitized – error code only
                    turns=outcome.turns,
                    tool_calls=outcome.tool_calls,
                    tool_call_details=outcome.tool_call_details,
                    claim_count=len(pred.claims),
                    langfuse_trace_id=execution.trace.trace_id,
                    langfuse_observation_id=outcome.langfuse_observation_id,
                )
                for c in pred.claims:
                    gold = gold_by_span.get((pred.answer_id, c.start_char, c.end_char))
                    gold_match = gold is not None
                    expected_label = gold.expected_label.value if gold else None
                    correct = (c.label == gold.expected_label) if gold else None
                    create_claim_result(
                        db2,
                        answer_result_id=ar.id,
                        run_id=run_id,
                        claim_id=c.claim_id,
                        answer_id=pred.answer_id,
                        quote=c.quote,
                        start_char=c.start_char,
                        end_char=c.end_char,
                        label=c.label.value,
                        evidence_fact_ids=c.evidence_fact_ids,
                        expected_label=expected_label,
                        correct=correct,
                        gold_match=gold_match if body.split == "development" else None,
                    )
                db2.commit()
            except Exception:
                db2.rollback()
                raise
            finally:
                db2.close()

        # Compute development metrics
        G = P = M = C = None
        ext_p = ext_r = ext_f1 = jnt_p = jnt_r = jnt_f1 = mf1 = gcov = None

        if body.split == "development":
            from app.evaluation.scorer import score
            gold_claims_list = list(gold_by_span.values())
            evaluated_ids = {answer.answer_id for answer in answers}
            score_result = score(
                predictions,
                gold_claims_list,
                evaluated_answer_ids=evaluated_ids,
            )
            G = score_result.G
            P = score_result.P
            M = score_result.M
            C = score_result.C
            ext_p = score_result.extraction.precision
            ext_r = score_result.extraction.recall
            ext_f1 = score_result.extraction.f1
            jnt_p = score_result.joint.precision
            jnt_r = score_result.joint.recall
            jnt_f1 = score_result.joint.f1
            mf1 = score_result.macro_f1
            gcov = score_result.gold_coverage
            run_scores, answer_scores = development_score_values(
                predictions, gold_claims_list, evaluated_ids
            )
            attach_development_scores(
                trace_id=execution.trace.trace_id,
                run_scores=run_scores,
                answer_scores=answer_scores,
                answer_observation_ids=execution.answer_observation_ids,
            )

        db3 = get_session()
        try:
            refreshed = get_run(db3, run_id)
            refreshed.langfuse_trace_id = execution.trace.trace_id
            finish_run(
                db3, refreshed,
                answers_ok=ok, answers_empty=empty, answers_error=error,
                claims_extracted=total_claims,
                G=G, P=P, M=M, C=C,
                extraction_precision=ext_p, extraction_recall=ext_r, extraction_f1=ext_f1,
                joint_precision=jnt_p, joint_recall=jnt_r, joint_f1=jnt_f1,
                macro_f1=mf1, gold_coverage=gcov,
            )
            db3.commit()
            final = get_run(db3, run_id)
            flush_traces()
            return _run_to_out(final)
        except Exception:
            db3.rollback()
            raise
        finally:
            db3.close()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# GET /evaluation-runs/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/evaluation-runs/{run_id}",
    response_model=EvaluationRunOut,
    summary="Retrieve a completed evaluation run with all answer and claim records",
)
def get_evaluation_run(run_id: uuid.UUID):
    db = get_session()
    try:
        run = get_run(db, run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found")
        return _run_to_out(run)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _run_to_out(run) -> EvaluationRunOut:
    answer_outs = []
    for ar in sorted(run.answer_results, key=lambda r: r.answer_id):
        claims_out = [
            {
                "claim_id": cr.claim_id,
                "answer_id": cr.answer_id,
                "quote": cr.quote,
                "start_char": cr.start_char,
                "end_char": cr.end_char,
                "label": cr.label,
                "evidence_fact_ids": cr.evidence_fact_ids,
                "expected_label": cr.expected_label,
                "correct": cr.correct,
                "gold_match": cr.gold_match,
            }
            for cr in sorted(ar.claim_results, key=lambda c: (c.start_char, c.end_char))
        ]
        answer_outs.append(
            AnswerResultOut(
                id=str(ar.id),
                answer_id=ar.answer_id,
                status=ar.status,
                error_code=ar.error_code,
                error_message=ar.error_message,
                turns=ar.turns,
                tool_calls=ar.tool_calls,
                tool_call_details=ar.tool_call_details,
                claim_count=ar.claim_count,
                claims=claims_out,
                langfuse_trace_id=ar.langfuse_trace_id,
                langfuse_observation_id=ar.langfuse_observation_id,
            )
        )
    return EvaluationRunOut(
        id=str(run.id),
        status=run.status,
        split=run.split,
        agent_version=run.agent_version,
        provider=run.provider,
        model=run.model,
        prompt_version=run.prompt_version,
        fact_version=run.fact_version,
        evaluator_version=run.evaluator_version,
        scheduled_answers=run.scheduled_answers,
        answers_ok=run.answers_ok,
        answers_empty=run.answers_empty,
        answers_error=run.answers_error,
        claims_extracted=run.claims_extracted,
        G=run.G, P=run.P, M=run.M, C=run.C,
        extraction_precision=run.extraction_precision,
        extraction_recall=run.extraction_recall,
        extraction_f1=run.extraction_f1,
        joint_precision=run.joint_precision,
        joint_recall=run.joint_recall,
        joint_f1=run.joint_f1,
        macro_f1=run.macro_f1,
        gold_coverage=run.gold_coverage,
        started_at=run.started_at.isoformat(),
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        langfuse_trace_id=run.langfuse_trace_id,
        answer_results=answer_outs,
    )
