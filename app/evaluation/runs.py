"""Async run execution. Gold is joined only by the evaluation layer after predictions exist."""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from app.agent.extractor import AgentError
from app.agent.fact_tool import FactCatalogue
from app.agent.prompts_v1 import SYSTEM_PROMPT
from app.agent.provider import (
    AsyncRetryingProvider,
    LLMProvider,
    MAX_CONCURRENT_PROVIDER_REQUESTS,
    MAX_RETRIES_PER_TURN,
    REQUEST_TIMEOUT_SECONDS,
)
from app.agent.runner import MAX_TURNS, AnswerResult, run_answer
from app.evaluation.scorer import AnswerPrediction, save_predictions
from app.schemas import Answer
from app.tracing import TraceContext, run_trace

PROMPT_VERSION = "v1"
EVALUATOR_VERSION = "1.0"
MAX_ACTIVE_REQUESTS = MAX_CONCURRENT_PROVIDER_REQUESTS


@dataclass
class RunExecution:
    predictions: list[AnswerPrediction]
    trace: TraceContext
    answer_observation_ids: dict[str, str | None]
    outcomes: dict[str, AnswerResult]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_revision(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_metadata(
    *,
    provider_name: str,
    model: str | None,
    fact_version: str,
    data_files: dict[str, Path],
    root: Path,
    prompt_version: str = PROMPT_VERSION,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    return {
        "provider": provider_name,
        "model": model,
        "settings": {
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_active_requests": MAX_ACTIVE_REQUESTS,
            "max_retries_per_turn": MAX_RETRIES_PER_TURN,
            "max_turns_per_answer": MAX_TURNS,
            "temperature": 0,
        },
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "dataset_version": {name: sha256_file(path) for name, path in data_files.items()},
        "fact_version": fact_version,
        "evaluator_version": EVALUATOR_VERSION,
        "source_revision": source_revision(root),
    }


def result_to_prediction(result: AnswerResult) -> AnswerPrediction:
    return AnswerPrediction(
        answer_id=result.answer_id,
        error=result.error.code if result.error else None,
        claims=[] if result.error else result.claims,
    )


def development_score_values(predictions, gold, answer_ids: set[str]) -> tuple[dict, dict]:
    from app.evaluation.scorer import score

    run_score = score(predictions, gold, evaluated_answer_ids=answer_ids)
    run_scores = {
        "extraction_precision": run_score.extraction.precision,
        "extraction_recall": run_score.extraction.recall,
        "extraction_f1": run_score.extraction.f1,
        "joint_precision": run_score.joint.precision,
        "joint_recall": run_score.joint.recall,
        "joint_f1": run_score.joint.f1,
        "macro_f1": run_score.macro_f1,
        "gold_coverage": run_score.gold_coverage,
    }
    gold_by_answer = {
        answer_id: [claim for claim in gold if claim.answer_id == answer_id]
        for answer_id in answer_ids
    }
    pred_by_answer = {prediction.answer_id: prediction for prediction in predictions}
    answer_scores = {}
    for answer_id in sorted(answer_ids):
        per_answer = score(
            [pred_by_answer[answer_id]],
            gold_by_answer[answer_id],
            evaluated_answer_ids={answer_id},
        )
        answer_scores[answer_id] = {
            "extraction_f1": per_answer.extraction.f1,
            "joint_f1": per_answer.joint.f1,
            "macro_f1": per_answer.macro_f1,
            "gold_coverage": per_answer.gold_coverage,
        }
    return run_scores, answer_scores


def outcome_record(result: AnswerResult) -> dict:
    return {
        "answer_id": result.answer_id,
        "status": "error" if result.error else ("ok" if result.claims else "valid_empty"),
        "error_code": result.error.code if result.error else None,
        "error_message": result.error.message if result.error else None,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "tool_call_details": result.tool_call_details,
        "langfuse_observation_id": result.langfuse_observation_id,
        "claims": (
            [claim.model_dump(mode="json") for claim in result.claims]
            if not result.error
            else []
        ),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def execute_run(
    answers: list[Answer],
    provider: LLMProvider,
    catalogue: FactCatalogue,
    run_dir: Path,
    metadata: dict,
    *,
    run_id: str,
    split: str,
    agent_version: str,
    agent: Callable[..., Awaitable[AnswerResult]] = run_answer,
) -> RunExecution:
    """Run every scheduled answer concurrently with per-answer failure isolation."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scheduled = sorted(answer.answer_id for answer in answers)
    run_json = run_dir / "run.json"
    base = {**metadata, "scheduled_answer_ids": scheduled, "started_at": utcnow()}
    _write_json(run_json, {**base, "status": "running"})
    results: dict[str, AnswerResult] = {}
    outcomes_path = run_dir / "outcomes.jsonl"

    with run_trace(
        run_id=run_id,
        split=split,
        agent_version=agent_version,
        metadata={
            "prompt_version": metadata.get("prompt_version", agent_version),
            "fact_version": metadata.get("fact_version"),
            "evaluator_version": metadata.get("evaluator_version", EVALUATOR_VERSION),
        },
    ) as trace:
        tasks = {
            asyncio.create_task(
                agent(
                    answer,
                    provider,
                    catalogue,
                    trace_metadata={
                        "run_id": run_id,
                        "split": split,
                        "agent_version": agent_version,
                    },
                )
            ): answer.answer_id
            for answer in answers
        }
        pending = set(tasks)
        try:
            with outcomes_path.open("w", encoding="utf-8") as output:
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        answer_id = tasks[task]
                        try:
                            result = task.result()
                        except Exception as exc:
                            result = AnswerResult(
                                answer_id,
                                error=AgentError("internal_error", type(exc).__name__),
                            )
                        if result.answer_id != answer_id:
                            raise RuntimeError("agent returned an outcome for a different answer")
                        results[answer_id] = result
                        output.write(
                            json.dumps(outcome_record(result), ensure_ascii=False) + "\n"
                        )
                        output.flush()
        except BaseException as exc:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            _write_json(
                run_json,
                {
                    **base,
                    "status": "interrupted",
                    "interrupted_by": type(exc).__name__,
                    "completed_answer_ids": sorted(results),
                    "langfuse_trace_id": trace.trace_id,
                    "finished_at": utcnow(),
                },
            )
            raise

    if sorted(results) != scheduled:
        _write_json(
            run_json,
            {
                **base,
                "status": "incomplete",
                "completed_answer_ids": sorted(results),
                "finished_at": utcnow(),
            },
        )
        raise RuntimeError("run finished without a terminal outcome for every scheduled answer")

    predictions = [result_to_prediction(results[answer_id]) for answer_id in scheduled]
    for prediction in predictions:
        prediction.claims.sort(key=lambda claim: (claim.start_char, claim.end_char))
    save_predictions(run_dir / "predictions.json", predictions)
    answer_observation_ids = {
        answer_id: results[answer_id].langfuse_observation_id for answer_id in scheduled
    }
    _write_json(
        run_json,
        {
            **base,
            "status": "agent_complete",
            "langfuse_trace_id": trace.trace_id,
            "langfuse_base_url": trace.base_url,
            "langfuse_enabled": trace.trace_id is not None,
            "answer_observation_ids": answer_observation_ids,
            "finished_at": utcnow(),
        },
    )
    _write_json(
        run_dir / "langfuse-traces.json",
        {
            "run_id": run_id,
            "trace_id": trace.trace_id,
            "base_url": trace.base_url,
            "answer_observation_ids": answer_observation_ids,
        },
    )
    return RunExecution(predictions, trace, answer_observation_ids, results)


def make_provider(inner: LLMProvider) -> AsyncRetryingProvider:
    return AsyncRetryingProvider(inner)
