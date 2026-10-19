import asyncio
import json

import pytest

from app.agent.extractor import AgentError
from app.agent.fact_tool import FactCatalogue
from app.agent.provider import (
    AsyncRetryingProvider,
    ProviderError,
    ProviderResponse,
)
from app.agent.runner import AnswerResult
from app.evaluation.runs import execute_run
from app.schemas import Answer


class Flaky:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    async def acomplete(self, messages, tools, tool_choice):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ProviderResponse("ok")


def test_one_retry_for_transient_then_success():
    async def check():
        inner = Flaky([ProviderError("temporary", transient=True)])
        provider = AsyncRetryingProvider(inner, sleep=lambda _: asyncio.sleep(0))
        response = await provider.acomplete([], [], "auto")
        assert response.content == "ok"
        assert inner.calls == 2

    asyncio.run(check())


def test_at_most_one_retry():
    async def check():
        inner = Flaky([ProviderError("temporary", transient=True)] * 5)
        provider = AsyncRetryingProvider(inner, sleep=lambda _: asyncio.sleep(0))
        with pytest.raises(ProviderError):
            await provider.acomplete([], [], "auto")
        assert inner.calls == 2

    asyncio.run(check())


def test_non_transient_not_retried():
    async def check():
        inner = Flaky([ProviderError("HTTP 401", transient=False)])
        provider = AsyncRetryingProvider(inner, sleep=lambda _: asyncio.sleep(0))
        with pytest.raises(ProviderError):
            await provider.acomplete([], [], "auto")
        assert inner.calls == 1

    asyncio.run(check())


def test_timeout_is_15_seconds_by_default_and_reported():
    async def check():
        provider = AsyncRetryingProvider(Flaky([]))
        assert provider._timeout == 15.0

        class Slow:
            async def acomplete(self, messages, tools, tool_choice):
                await asyncio.sleep(0.05)

        wrapped = AsyncRetryingProvider(
            Slow(), timeout=0.001, sleep=lambda _: asyncio.sleep(0)
        )
        with pytest.raises(ProviderError) as exc:
            await wrapped.acomplete([], [], "auto")
        assert exc.value.timed_out

    asyncio.run(check())


def test_never_more_than_three_active_provider_requests():
    async def check():
        state = {"now": 0, "max": 0}

        class Slow:
            async def acomplete(self, messages, tools, tool_choice):
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
                await asyncio.sleep(0.01)
                state["now"] -= 1
                return ProviderResponse("ok")

        provider = AsyncRetryingProvider(Slow())
        await asyncio.gather(
            *(provider.acomplete([], [], "auto") for _ in range(12))
        )
        assert state["max"] == 3

    asyncio.run(check())


def mk(i):
    return Answer(
        answer_id=f"A{i:02d}",
        split="development",
        brand_id="b",
        question="q",
        ai_answer="x",
    )


def test_one_terminal_outcome_per_answer_with_failure_isolation(tmp_path):
    async def agent(answer, provider, catalogue, **kwargs):
        if answer.answer_id == "A02":
            return AnswerResult(
                answer.answer_id, error=AgentError("provider_error", "safe detail")
            )
        if answer.answer_id == "A03":
            raise RuntimeError("unexpected")
        return AnswerResult(
            answer.answer_id,
            turns=2,
            tool_calls=["get_brand_facts"],
            tool_call_details=[
                {"id": "tool-1", "name": "get_brand_facts", "brand_id": "brand"}
            ],
        )

    async def check():
        execution = await execute_run(
            [mk(i) for i in (3, 1, 2)],
            Flaky([]),
            FactCatalogue([]),
            tmp_path / "r",
            {"m": 1},
            run_id="test-run",
            split="development",
            agent_version="v1",
            agent=agent,
        )
        predictions = execution.predictions
        assert [p.answer_id for p in predictions] == ["A01", "A02", "A03"]
        assert [p.error for p in predictions] == [
            None,
            "provider_error",
            "internal_error",
        ]
        assert execution.outcomes["A01"].turns == 2
        assert execution.outcomes["A01"].tool_calls == ["get_brand_facts"]
        assert execution.outcomes["A01"].tool_call_details == [
            {"id": "tool-1", "name": "get_brand_facts", "brand_id": "brand"}
        ]

    asyncio.run(check())
    lines = (tmp_path / "r" / "outcomes.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert sorted(record["answer_id"] for record in records) == ["A01", "A02", "A03"]
    assert {r["answer_id"]: r["status"] for r in records} == {
        "A01": "valid_empty",
        "A02": "error",
        "A03": "error",
    }
    assert json.loads((tmp_path / "r" / "run.json").read_text())["status"] == "agent_complete"
    assert (tmp_path / "r" / "predictions.json").exists()


def test_unexpected_answer_failure_preserves_other_terminal_outcomes(tmp_path):
    async def agent(answer, provider, catalogue, **kwargs):
        if answer.answer_id == "A02":
            raise RuntimeError("worker failed")
        return AnswerResult(answer.answer_id)

    async def check():
        execution = await execute_run(
            [mk(1), mk(2)],
            Flaky([]),
            FactCatalogue([]),
            tmp_path / "r",
            {},
            run_id="isolated-run",
            split="check",
            agent_version="v2",
            agent=agent,
        )
        assert [prediction.answer_id for prediction in execution.predictions] == [
            "A01",
            "A02",
        ]
        assert execution.predictions[1].error == "internal_error"

    asyncio.run(check())
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert run["status"] == "agent_complete"
    assert len((tmp_path / "r" / "outcomes.jsonl").read_text().splitlines()) == 2


def test_cancelled_run_keeps_partial_evidence_and_is_not_complete(tmp_path):
    async def agent(answer, provider, catalogue, **kwargs):
        if answer.answer_id == "A02":
            raise asyncio.CancelledError
        return AnswerResult(answer.answer_id)

    async def check():
        with pytest.raises(asyncio.CancelledError):
            await execute_run(
                [mk(1), mk(2)],
                Flaky([]),
                FactCatalogue([]),
                tmp_path / "r",
                {},
                run_id="cancelled-run",
                split="check",
                agent_version="v2",
                agent=agent,
            )

    asyncio.run(check())
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert run["status"] == "interrupted"
    assert not (tmp_path / "r" / "predictions.json").exists()
