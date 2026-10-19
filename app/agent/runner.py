"""Agent loop: at most 3 model turns; any failure is an answer-level error."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.agent.extractor import AgentError, ExtractedClaim, locate_spans, parse_output
from app.agent.fact_tool import TOOL_DEFINITIONS, FactCatalogue, ToolError, execute_tool
from app.agent.prompts_v1 import SYSTEM_PROMPT, build_user_message
from app.agent.provider import LLMProvider, ProviderError
from app.agent.validator import validate_claims
from app.schemas import Answer
from app.tracing import answer_trace, observation

MAX_TURNS = 3
MAX_TOOL_CALLS_PER_TURN = 4


@dataclass
class AnswerResult:
    answer_id: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    error: AgentError | None = None
    turns: int = 0
    tool_calls: list[str] = field(default_factory=list)  # names requested by the model
    tool_call_details: list[dict] = field(default_factory=list)
    langfuse_observation_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_answer(
    answer: Answer,
    provider: LLMProvider,
    catalogue: FactCatalogue,
    system_prompt: str = SYSTEM_PROMPT,
    trace_metadata: dict | None = None,
) -> AnswerResult:
    result = AnswerResult(answer_id=answer.answer_id)
    with answer_trace(
        answer_id=answer.answer_id,
        metadata=trace_metadata or {},
    ) as span:
        result.langfuse_observation_id = getattr(span, "id", None)
        try:
            result.claims = await _run(answer, provider, catalogue, result, system_prompt)
        except AgentError as err:
            result.claims = []
            result.error = err
    return result


async def _run(
    answer: Answer, provider: LLMProvider, catalogue: FactCatalogue, result: AnswerResult, system_prompt: str
) -> list[ExtractedClaim]:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(answer)},
    ]
    retrieved_ids: set[str] = set()

    for turn in range(1, MAX_TURNS + 1):
        result.turns = turn
        with observation(
            name="model-generation",
            as_type="generation",
            model=getattr(provider, "model", None),
            metadata={"answer_id": answer.answer_id, "turn": turn},
        ) as generation:
            try:
                resp = await provider.acomplete(
                    messages, TOOL_DEFINITIONS, "required" if turn == 1 else "auto"
                )
            except ProviderError as exc:
                if generation is not None:
                    generation.update(
                        level="ERROR",
                        status_message="provider_timeout" if exc.timed_out else "provider_error",
                    )
                code = "provider_timeout" if exc.timed_out else "provider_error"
                raise AgentError(code, str(exc)) from exc
            if generation is not None and resp.usage:
                generation.update(usage_details=resp.usage)

        if not resp.tool_calls:
            if not result.tool_calls:
                raise AgentError("tool_not_called", "model answered without requesting get_brand_facts")
            raw = parse_output(resp.content)
            claims = locate_spans(answer.ai_answer, raw.claims, answer.answer_id)
            validate_claims(answer.ai_answer, claims, retrieved_ids)
            return claims

        if turn == MAX_TURNS:
            raise AgentError("turns_exhausted", f"no final answer within {MAX_TURNS} model turns")
        if len(resp.tool_calls) > MAX_TOOL_CALLS_PER_TURN:
            raise AgentError("illegal_tool", "too many tool calls in one turn")

        messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in resp.tool_calls
                ],
            }
        )
        for tc in resp.tool_calls:
            result.tool_calls.append(tc.name)
            call_detail = {"id": tc.id, "name": tc.name}
            try:
                args = json.loads(tc.arguments)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except ValueError as exc:
                result.tool_call_details.append(call_detail)
                raise AgentError("illegal_tool", f"unparseable arguments for {tc.name!r}") from exc
            brand_id = args.get("brand_id")
            if isinstance(brand_id, str):
                call_detail["brand_id"] = brand_id
            result.tool_call_details.append(call_detail)
            try:
                with observation(
                    name="agent-tool-call",
                    as_type="tool",
                    metadata={
                        "answer_id": answer.answer_id,
                        "tool_name": tc.name,
                        "brand_id": args.get("brand_id"),
                    },
                ):
                    output = execute_tool(tc.name, args, catalogue)  # fixed allowlist
            except ToolError as exc:
                raise AgentError("illegal_tool", str(exc)) from exc
            retrieved_ids.update(f["fact_id"] for f in output["facts"])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, ensure_ascii=False)})

    raise AgentError("turns_exhausted", f"no final answer within {MAX_TURNS} model turns")  # pragma: no cover
