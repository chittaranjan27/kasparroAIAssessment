"""POST /classify

Accepts a single answer input + agent_version, runs the live agent, persists one
AnswerResult + ClaimResults, and returns the extracted claims.

Error mapping
  400 / 422  invalid request body or forbidden fields
  404        unknown brand_id (not in catalogue)
  502        provider failure or invalid model output
  504        request timeout
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.extractor import AgentError
from app.agent.fact_tool import FactCatalogue
from app.agent.provider import AsyncOpenAIProvider, ProviderError
from app.agent.runner import run_answer
from app.db.repository import create_answer_result, create_claim_result
from app.db.session import get_session
from app.evaluation.runs import make_provider
from app.schemas import Answer, Split
from app.tracing import flush_traces, run_trace

router = APIRouter()

# ---------------------------------------------------------------------------
# Forbidden input fields (gold / evaluation data must never enter via API)
# ---------------------------------------------------------------------------
_FORBIDDEN_FIELDS = frozenset(
    [
        "expected_label",
        "gold_span",
        "gold_claim",
        "gold_claims",
        "gold_claim_list",
        "gold_claims_list",
        "gold_match",
        "gold",
        "explanation",
        "difficulty_tags",
        "evidence_fact_ids",
        "claims",
        "claim_list",
        "claim_lists",
        "claim_results",
        "expected_claims",
        "correct",
    ]
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(..., min_length=1, max_length=16)
    brand_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1)
    ai_answer: str = Field(..., min_length=1)
    agent_version: str = Field(..., pattern=r"^v[12]$")

    @field_validator("answer_id", "brand_id", "question", "ai_answer")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden(cls, data: Any) -> Any:
        if isinstance(data, dict):
            bad = _FORBIDDEN_FIELDS & {str(key).casefold() for key in data}
            if bad:
                raise ValueError(f"forbidden fields: {sorted(bad)}")
        return data


class ClaimOut(BaseModel):
    claim_id: str
    quote: str
    start_char: int
    end_char: int
    label: str
    evidence_fact_ids: list[str]


class ClassifyResponse(BaseModel):
    answer_id: str
    agent_version: str
    claims: list[ClaimOut]
    turns: int
    tool_calls: list[str]
    tool_call_details: list[dict]


# ---------------------------------------------------------------------------
# Shared provider / catalogue (initialised lazily at first request)
# ---------------------------------------------------------------------------

_provider_cache: dict[str, Any] = {}


def _get_provider():
    if "provider" not in _provider_cache:
        inner = AsyncOpenAIProvider.from_env()
        _provider_cache["provider"] = make_provider(inner)
    return _provider_cache["provider"]


def _get_shared_catalogue():
    if "catalogue" not in _provider_cache:
        from app.config import get_settings
        from app.evaluation import data_loader

        s = get_settings()
        facts = data_loader.load_brand_facts(s.brand_facts_path, s.expected_facts)
        _provider_cache["catalogue"] = FactCatalogue(facts, s.fact_version)
    return _provider_cache["catalogue"]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract and classify factual claims from one AI answer",
)
async def classify(body: ClassifyRequest):
    catalogue: FactCatalogue = _get_shared_catalogue()

    # 404 for unknown brand
    if body.brand_id not in catalogue.brands:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown brand_id: {body.brand_id!r}",
        )

    # Build Answer (split is irrelevant for API path but required by schema)
    answer = Answer(
        answer_id=body.answer_id,
        split=Split.CHECK,  # API callers are not the evaluation harness
        brand_id=body.brand_id,
        question=body.question,
        ai_answer=body.ai_answer,
    )

    # Choose prompt
    if body.agent_version == "v1":
        from app.agent.prompts_v1 import SYSTEM_PROMPT
    else:
        from app.agent.prompts_v2 import SYSTEM_PROMPT

    run_id = uuid.uuid4()
    with run_trace(
        run_id=str(run_id),
        split="api",
        agent_version=body.agent_version,
        metadata={"prompt_version": body.agent_version},
    ) as trace:
        try:
            provider = _get_provider()
            result = await run_answer(
                answer,
                provider,
                catalogue,
                system_prompt=SYSTEM_PROMPT,
                trace_metadata={"run_id": str(run_id), "split": "api"},
            )
        except ProviderError as exc:
            code = "provider_timeout" if exc.timed_out else "provider_error"
            result = _failed_result(body.answer_id, code, str(exc))
        _persist_classify(
            result,
            body.agent_version,
            run_id=run_id,
            trace_id=trace.trace_id,
        )
    flush_traces()
    if result.error:
        _raise_agent_error(result.error)

    return ClassifyResponse(
        answer_id=result.answer_id,
        agent_version=body.agent_version,
        claims=[ClaimOut(**c.model_dump()) for c in result.claims],
        turns=result.turns,
        tool_calls=result.tool_calls,
        tool_call_details=result.tool_call_details,
    )


def _failed_result(answer_id: str, code: str, message: str):
    from app.agent.extractor import AgentError
    from app.agent.runner import AnswerResult

    return AnswerResult(answer_id=answer_id, error=AgentError(code, message))


def _raise_agent_error(error: AgentError) -> None:
    code = error.code
    response_status = (
        status.HTTP_504_GATEWAY_TIMEOUT
        if code == "provider_timeout"
        else status.HTTP_502_BAD_GATEWAY
    )
    raise HTTPException(status_code=response_status, detail=error.message)


def _persist_classify(
    result, agent_version: str, *, run_id: uuid.UUID, trace_id: str | None
) -> None:
    from app.config import get_settings
    from app.evaluation.runs import EVALUATOR_VERSION

    s = get_settings()
    db = get_session()
    try:
        from app.db.repository import create_run, finish_run

        run = create_run(
            db,
            run_id=run_id,
            split="api",
            agent_version=agent_version,
            provider="openai",
            model=None,
            prompt_version=agent_version,
            fact_version=s.fact_version,
            evaluator_version=EVALUATOR_VERSION,
            scheduled_answers=1,
            settings={},
        )
        status_str = "error" if result.error else ("valid_empty" if not result.claims else "ok")
        ar = create_answer_result(
            db,
            run_id=run.id,
            answer_id=result.answer_id,
            status=status_str,
            error_code=result.error.code if result.error else None,
            error_message=result.error.message if result.error else None,
            turns=result.turns,
            tool_calls=result.tool_calls,
            tool_call_details=result.tool_call_details,
            claim_count=len(result.claims),
            langfuse_trace_id=trace_id,
            langfuse_observation_id=result.langfuse_observation_id,
        )
        for c in result.claims:
            create_claim_result(
                db,
                answer_result_id=ar.id,
                run_id=run.id,
                claim_id=c.claim_id,
                answer_id=result.answer_id,
                quote=c.quote,
                start_char=c.start_char,
                end_char=c.end_char,
                label=c.label.value,
                evidence_fact_ids=c.evidence_fact_ids,
                expected_label=None,
                correct=None,
                gold_match=None,
            )
        finish_run(
            db,
            run,
            answers_ok=1 if not result.error and result.claims else 0,
            answers_empty=1 if not result.error and not result.claims else 0,
            answers_error=1 if result.error else 0,
            claims_extracted=len(result.claims),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
