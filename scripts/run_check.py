"""Run frozen v1 or v2 agent on final-check answers A41-A50 WITHOUT joining development_gold.csv.

All gold-dependent quality metrics, expected labels, correctness fields and gold-match records
MUST remain null (not zero, not 'unknown').  Only agent predictions, timings, counts and settings
are recorded.

Usage:
    python scripts/run_check.py --version v1
    python scripts/run_check.py --version v2
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
from app.evaluation.runs import (  # noqa: E402
    EVALUATOR_VERSION,
    execute_run,
    make_provider,
    run_metadata,
    utcnow,
)
from app.evaluation.scorer import load_predictions  # noqa: E402
from app.schemas import Split  # noqa: E402
from app.tracing import flush_traces  # noqa: E402


def prompt_for(version: str) -> str:
    if version == "v1":
        from app.agent.prompts_v1 import SYSTEM_PROMPT
    else:
        from app.agent.prompts_v2 import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def outcome_summary(outcomes: list[dict]) -> dict:
    """Counts from the outcomes.jsonl, purely based on agent results (no gold)."""
    total = len(outcomes)
    errors = [o for o in outcomes if o["status"] == "error"]
    valid_empty = [o for o in outcomes if o["status"] == "valid_empty"]
    ok = [o for o in outcomes if o["status"] == "ok"]
    error_by_code: dict[str, int] = {}
    for o in errors:
        code = o.get("error_code") or "unknown"
        error_by_code[code] = error_by_code.get(code, 0) + 1
    return {
        "total_scheduled": total,
        "completed_ok": len(ok),
        "valid_empty": len(valid_empty),
        "errors": len(errors),
        "error_by_code": error_by_code,
        "all_processed": len(ok) + len(valid_empty) + len(errors) == total,
        "any_errors": len(errors) > 0,
    }


def prediction_summary(predictions) -> dict:
    """Purely agent-side counts; all gold-dependent values are null."""
    total_claims = sum(len(p.claims) for p in predictions if p.error is None)
    per_answer = []
    for p in predictions:
        entry = {
            "answer_id": p.answer_id,
            "error": p.error,
            "claim_count": len(p.claims) if p.error is None else None,
            # ---- gold-dependent fields: all null ----
            "matched_gold": None,
            "correct_label": None,
            "expected_labels": None,
        }
        per_answer.append(entry)
    per_answer.sort(key=lambda e: e["answer_id"])
    return {
        "total_claims_extracted": total_claims,
        "per_answer": per_answer,
        # All quality metrics are null because gold is not joined for check answers
        "extraction_precision": None,
        "extraction_recall": None,
        "extraction_f1": None,
        "joint_precision": None,
        "joint_recall": None,
        "joint_f1": None,
        "macro_f1": None,
        "gold_coverage": None,
        "G": None,
        "M": None,
        "C": None,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    version = ap.parse_args().version

    system_prompt = prompt_for(version)
    s = get_settings()
    answers = data_loader.load_answers(s.answers_path, s.expected_answers)
    data_loader.validate_answer_splits(answers)

    check_answers = sorted(
        (a for a in answers if a.split is Split.CHECK),
        key=lambda a: a.answer_id,
    )
    expected_ids = [f"A{i:02d}" for i in range(41, 51)]
    actual_ids = [a.answer_id for a in check_answers]
    if actual_ids != expected_ids:
        print(
            f"ERROR: expected check answers {expected_ids}, got {actual_ids}",
            file=sys.stderr,
        )
        return 1

    catalogue = FactCatalogue(
        data_loader.load_brand_facts(s.brand_facts_path, s.expected_facts),
        s.fact_version,
    )

    inner = AsyncOpenAIProvider.from_env()
    provider = make_provider(inner)
    started = utcnow()
    report_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    REPORT_PATH = ROOT / "artifacts" / "reports" / f"{version}_check_{report_stamp}.json"

    run_dir = (
        ROOT
        / "artifacts"
        / "runs"
        / f"{version}_check_{report_stamp}"
    )

    # Gold is intentionally omitted from metadata data_files list to make it
    # explicit that this run never touches development_gold.csv.
    meta = run_metadata(
        provider_name="openai",
        model=inner.model,
        fact_version=s.fact_version,
        data_files={
            "answers.csv": s.answers_path,
            "brand_facts.csv": s.brand_facts_path,
            # development_gold.csv intentionally excluded
        },
        root=ROOT,
        prompt_version=version,
        system_prompt=system_prompt,
    )
    agent = functools.partial(run_answer, system_prompt=system_prompt)
    run_id = str(uuid.uuid4())

    # ---- Phase 1: agent only – gold is NEVER loaded or referenced ----
    try:
        execution = await execute_run(
            check_answers,
            provider,
            catalogue,
            run_dir,
            meta,
            run_id=run_id,
            split="check",
            agent_version=version,
            agent=agent,
        )
    except KeyboardInterrupt:
        print(
            f"INTERRUPTED. Partial evidence kept in {run_dir}; no report written; NOT a complete run.",
            file=sys.stderr,
        )
        return 130

    # ---- Phase 2: build check report without joining gold ----
    predictions_path = run_dir / "predictions.json"
    predictions = load_predictions(predictions_path)

    outcomes_raw = [
        json.loads(line)
        for line in (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    outcomes_raw.sort(key=lambda o: o["answer_id"])

    summary = outcome_summary(outcomes_raw)
    pred_summary = prediction_summary(predictions)

    report = {
        "run_status": "complete",
        "langfuse_trace_id": execution.trace.trace_id,
        "langfuse_base_url": execution.trace.base_url,
        "langfuse_enabled": execution.trace.trace_id is not None,
        "split": "check",
        "answer_ids": expected_ids,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "evaluator_version": EVALUATOR_VERSION,
        **meta,
        "started_at": started,
        "finished_at": utcnow(),
        "scheduled_answers": len(check_answers),
        # ---- gold-dependent quality metrics: all null per specification ----
        "metrics": None,
        # ---- agent-only counts ----
        "run_summary": summary,
        "predictions_summary": pred_summary,
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
            for o in outcomes_raw
        ],
        # ---- predictions (no gold labels, no correctness) ----
        "predictions": [
            {
                "answer_id": p.answer_id,
                "error": p.error,
                "claim_count": len(p.claims) if p.error is None else None,
                "claims": [
                    {
                        "claim_id": c.claim_id,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                        "quote": c.quote,
                        "label": c.label.value,
                        # ---- gold-dependent: null ----
                        "expected_label": None,
                        "correct": None,
                        "gold_match": None,
                    }
                    for c in p.claims
                ]
                if p.error is None
                else [],
            }
            for p in sorted(predictions, key=lambda x: x.answer_id)
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flush_traces()

    # ---- Console summary ----
    print(f"\n{'='*60}")
    print(f"Check run complete  ({version.upper()})  split=check  answers=A41-A50")
    print(f"{'='*60}")
    print(f"Scheduled : {summary['total_scheduled']}")
    print(f"OK        : {summary['completed_ok']}")
    print(f"Empty     : {summary['valid_empty']}")
    print(f"Errors    : {summary['errors']}  {summary['error_by_code']}")
    print(f"All processed: {summary['all_processed']}")
    print(f"Total claims extracted: {pred_summary['total_claims_extracted']}")
    print(f"\nNOTE: All gold-dependent metrics are null (gold not joined for check split).")
    print(f"Report : {REPORT_PATH}")
    print(f"Run dir: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
