"""Live single-answer smoke test. Usage: python scripts/smoke_one.py [A01]
Reads only answers.csv + brand_facts.csv. Never loads development_gold.csv."""
import asyncio
import sys
import uuid

from app.agent.fact_tool import FactCatalogue
from app.agent.provider import AsyncOpenAIProvider
from app.agent.runner import run_answer
from app.config import get_settings
from app.evaluation.data_loader import load_answers, load_brand_facts
from app.evaluation.runs import make_provider
from app.tracing import flush_traces, run_trace


async def main():
    s = get_settings()
    answers = {
        a.answer_id: a for a in load_answers(s.answers_path, s.expected_answers)
    }
    catalogue = FactCatalogue(
        load_brand_facts(s.brand_facts_path, s.expected_facts), s.fact_version
    )
    answer = answers[sys.argv[1] if len(sys.argv) > 1 else "A01"]
    provider = make_provider(AsyncOpenAIProvider.from_env())
    run_id = str(uuid.uuid4())
    with run_trace(
        run_id=run_id,
        split=answer.split.value,
        agent_version="v1",
        metadata={"fact_version": s.fact_version, "smoke_test": True},
    ):
        result = await run_answer(
            answer,
            provider,
            catalogue,
            trace_metadata={"run_id": run_id, "split": answer.split.value},
        )
    flush_traces()
    print(
        f"answer={result.answer_id} ok={result.ok} turns={result.turns} "
        f"tool_calls={result.tool_calls}"
    )
    if result.error:
        print("ERROR:", result.error.code, "-", result.error.message)
    for claim in result.claims:
        print(
            f"{claim.claim_id} [{claim.start_char}:{claim.end_char}] "
            f"{claim.label.value} ev={claim.evidence_fact_ids}\n    {claim.quote!r}"
        )
        assert answer.ai_answer[claim.start_char : claim.end_char] == claim.quote


if __name__ == "__main__":
    asyncio.run(main())
