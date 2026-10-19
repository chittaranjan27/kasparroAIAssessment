"""Run baseline agent v1 on development answers A01-A40, save predictions, THEN join gold and score.

Usage: python scripts/run_evaluation.py [--version v1|v2]   (live provider; needs OPENAI_API_KEY in env/.env)
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.fact_tool import FactCatalogue  # noqa: E402
from app.agent.provider import AsyncOpenAIProvider  # noqa: E402
from app.agent.runner import run_answer  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.evaluation import data_loader  # noqa: E402
from app.evaluation.report import build_records, render_text  # noqa: E402
from app.evaluation.runs import (  # noqa: E402
    EVALUATOR_VERSION,
    development_claim_scores,
    development_score_metadata,
    execute_run,
    development_score_values,
    make_provider,
    run_metadata,
    utcnow,
)
from app.evaluation.scorer import evaluate_saved_run, load_predictions  # noqa: E402
from app.tracing import attach_development_scores, flush_traces  # noqa: E402
from app.schemas import Split  # noqa: E402


def prompt_for(version: str) -> str:
    if version == "v1":
        from app.agent.prompts_v1 import SYSTEM_PROMPT
    else:
        from app.agent.prompts_v2 import SYSTEM_PROMPT
    return SYSTEM_PROMPT


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    version = ap.parse_args().version
    system_prompt = prompt_for(version)
    s = get_settings()
    answers = data_loader.load_answers(s.answers_path, s.expected_answers)
    data_loader.validate_answer_splits(answers)
    dev = sorted((a for a in answers if a.split is Split.DEVELOPMENT), key=lambda a: a.answer_id)
    assert [a.answer_id for a in dev] == [f"A{i:02d}" for i in range(1, 41)]
    catalogue = FactCatalogue(data_loader.load_brand_facts(s.brand_facts_path, s.expected_facts), s.fact_version)

    inner = AsyncOpenAIProvider.from_env()
    provider = make_provider(inner)
    started = utcnow()
    report_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    REPORT_PATH = ROOT / "artifacts" / "reports" / f"{version}_development_{report_stamp}.json"
    run_dir = ROOT / "artifacts" / "runs" / f"{version}_development_{report_stamp}"
    meta = run_metadata(
        provider_name="openai", model=inner.model, fact_version=s.fact_version,
        data_files={"answers.csv": s.answers_path, "brand_facts.csv": s.brand_facts_path,
                    "development_gold.csv": s.gold_path},
        root=ROOT, prompt_version=version, system_prompt=system_prompt,
    )
    agent = functools.partial(run_answer, system_prompt=system_prompt)
    run_id = str(uuid.uuid4())

    # ---- phase 1: agent only. No gold is loaded in this phase. ----
    try:
        execution = await execute_run(
            dev,
            provider,
            catalogue,
            run_dir,
            meta,
            run_id=run_id,
            split="development",
            agent_version=version,
            agent=agent,
        )
    except KeyboardInterrupt:
        print(f"INTERRUPTED. Partial evidence kept in {run_dir}; no report written; NOT a complete run.", file=sys.stderr)
        return 130

    # ---- phase 2: predictions are on disk; only now join gold in the evaluation harness ----
    predictions_path = run_dir / "predictions.json"
    gold = data_loader.load_gold(s.gold_path, answers, s.expected_gold)
    result = evaluate_saved_run(predictions_path, gold)
    run_scores, answer_scores = development_score_values(
        load_predictions(predictions_path),
        gold,
        {answer.answer_id for answer in dev},
    )
    attach_development_scores(
        trace_id=execution.trace.trace_id,
        run_scores=run_scores,
        answer_scores=answer_scores,
        answer_observation_ids=execution.answer_observation_ids,
        claim_scores=development_claim_scores(load_predictions(predictions_path), gold),
        run_metadata=development_score_metadata(result),
    )
    flush_traces()
    predictions = load_predictions(predictions_path)
    outcomes = [json.loads(line) for line in (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    outcomes.sort(key=lambda o: o["answer_id"])

    report = {
        "run_status": "complete",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "langfuse_trace_id": execution.trace.trace_id,
        "langfuse_base_url": execution.trace.base_url,
        "langfuse_trace_url": execution.trace.trace_url,
        "langfuse_enabled": execution.trace.trace_id is not None,
        "evaluator_version": EVALUATOR_VERSION,
        **meta,
        "started_at": started,
        "finished_at": utcnow(),
        "scheduled_answers": len(dev),
        "metrics": result.model_dump(mode="json"),
        "records": build_records(result, gold, predictions),
        "answer_outcomes": [
            {
                k: o[k]
                for k in (
                    "answer_id",
                    "status",
                    "error_code",
                    "error_message",
                    "turns",
                    "tool_calls",
                    "tool_call_details",
                    "langfuse_observation_id",
                )
            }
            for o in outcomes
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_text(result))
    print(f"\nReport: {REPORT_PATH}\nRun evidence: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
