"""Database persistence tests.

These tests run against a REAL PostgreSQL database (the Docker container).
They require DATABASE_URL to be set (either in .env or the environment).

Run with:
    pytest tests/test_database.py -v

The fixture creates all tables, runs a test, then rolls back or drops rows so
tests stay independent.  All operations use the ORM through the repository layer
(no raw SQL in test bodies).
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import delete

_TEST_RUN_IDS: set[uuid.UUID] = set()

# ---------------------------------------------------------------------------
# Skip if no DATABASE_URL
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    __import__("dotenv").dotenv_values(".env").get("DATABASE_URL", "")
)
if not DATABASE_URL:
    DATABASE_URL = ""

os.environ["DATABASE_URL"] = DATABASE_URL

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL not configured; skipping database tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from app.db.session import get_engine, create_tables
    # Override cached engine with test DATABASE_URL
    from app.db import session as sess_mod
    sess_mod._engine.cache_clear()
    eng = get_engine()
    create_tables()
    yield eng
    if _TEST_RUN_IDS:
        from app.db.models import EvaluationRun

        with eng.begin() as conn:
            conn.execute(
                delete(EvaluationRun).where(EvaluationRun.id.in_(_TEST_RUN_IDS))
            )


@pytest.fixture
def db(engine):
    from sqlalchemy.orm import sessionmaker
    SM = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SM()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_run(db, split="check", agent_version="v1"):
    from app.db.repository import create_run
    run = create_run(
        db,
        split=split,
        agent_version=agent_version,
        provider="openai",
        model="gpt-4o-mini",
        prompt_version=agent_version,
        fact_version="speaker-catalogue-v4",
        evaluator_version="1.0",
        scheduled_answers=2,
        settings={"max_turns": 3},
    )
    _TEST_RUN_IDS.add(run.id)
    db.commit()
    return run


# ---------------------------------------------------------------------------
# EvaluationRun persistence
# ---------------------------------------------------------------------------

class TestEvaluationRunPersistence:
    def test_create_run_has_uuid(self, db):
        run = _make_run(db)
        assert isinstance(run.id, uuid.UUID)
        assert run.status == "running"
        assert run.split == "check"

    def test_finish_run_sets_metrics_null_for_check(self, db):
        from app.db.repository import finish_run, get_run
        run = _make_run(db, split="check")
        finish_run(
            db, run,
            answers_ok=2, answers_empty=0, answers_error=0,
            claims_extracted=8,
            # No gold metrics passed for check
        )
        db.commit()
        refreshed = get_run(db, run.id)
        assert refreshed.status == "complete"
        assert refreshed.G is None
        assert refreshed.extraction_f1 is None
        assert refreshed.macro_f1 is None
        assert refreshed.claims_extracted == 8

    def test_finish_run_sets_metrics_for_development(self, db):
        from app.db.repository import finish_run, get_run
        run = _make_run(db, split="development")
        finish_run(
            db, run,
            answers_ok=2, answers_empty=0, answers_error=0,
            claims_extracted=10,
            G=20, P=18, M=15, C=12,
            extraction_precision=0.833, extraction_recall=0.75, extraction_f1=0.789,
            joint_precision=0.667, joint_recall=0.6, joint_f1=0.632,
            macro_f1=0.72, gold_coverage=0.75,
        )
        db.commit()
        refreshed = get_run(db, run.id)
        assert refreshed.G == 20
        assert refreshed.extraction_f1 == pytest.approx(0.789, abs=1e-3)
        assert refreshed.macro_f1 == pytest.approx(0.72, abs=1e-3)


# ---------------------------------------------------------------------------
# AnswerResult persistence
# ---------------------------------------------------------------------------

class TestAnswerResultPersistence:
    def test_ok_answer_persists(self, db):
        from app.db.repository import create_answer_result, get_run
        run = _make_run(db)
        ar = create_answer_result(
            db,
            run_id=run.id,
            answer_id="A41",
            status="ok",
            error_code=None,
            error_message=None,
            turns=2,
            tool_calls=["get_brand_facts"],
            claim_count=3,
        )
        db.commit()
        assert ar.status == "ok"
        assert ar.error_code is None
        assert ar.claim_count == 3

    def test_error_answer_persists(self, db):
        from app.db.repository import create_answer_result
        run = _make_run(db)
        ar = create_answer_result(
            db,
            run_id=run.id,
            answer_id="A42",
            status="error",
            error_code="invalid_evidence",
            error_message="sanitized error message",
            turns=1,
            tool_calls=[],
            claim_count=0,
        )
        db.commit()
        assert ar.status == "error"
        assert ar.error_code == "invalid_evidence"

    def test_valid_empty_answer_persists(self, db):
        from app.db.repository import create_answer_result
        run = _make_run(db)
        ar = create_answer_result(
            db,
            run_id=run.id,
            answer_id="A43",
            status="valid_empty",
            error_code=None,
            error_message=None,
            turns=2,
            tool_calls=["get_brand_facts"],
            claim_count=0,
        )
        db.commit()
        assert ar.status == "valid_empty"
        assert ar.claim_count == 0

    def test_uniqueness_constraint_enforced(self, db):
        """Two AnswerResults for the same run+answer_id must raise an integrity error."""
        from app.db.repository import create_answer_result
        from sqlalchemy.exc import IntegrityError
        run = _make_run(db)
        create_answer_result(
            db, run_id=run.id, answer_id="A44", status="ok",
            error_code=None, error_message=None, turns=2,
            tool_calls=[], claim_count=1,
        )
        db.commit()
        with pytest.raises(IntegrityError):
            create_answer_result(
                db, run_id=run.id, answer_id="A44", status="ok",
                error_code=None, error_message=None, turns=2,
                tool_calls=[], claim_count=1,
            )
            db.commit()
        db.rollback()


# ---------------------------------------------------------------------------
# ClaimResult persistence
# ---------------------------------------------------------------------------

class TestClaimResultPersistence:
    def test_check_claim_gold_fields_are_null(self, db):
        from app.db.repository import create_answer_result, create_claim_result
        run = _make_run(db, split="check")
        ar = create_answer_result(
            db, run_id=run.id, answer_id="A41", status="ok",
            error_code=None, error_message=None, turns=2,
            tool_calls=[], claim_count=1,
        )
        cr = create_claim_result(
            db,
            answer_result_id=ar.id,
            run_id=run.id,
            claim_id="A41-C01",
            answer_id="A41",
            quote="The speaker outputs 50W.",
            start_char=0,
            end_char=24,
            label="supported",
            evidence_fact_ids=["F001"],
            # Check run: no gold
            expected_label=None,
            correct=None,
            gold_match=None,
        )
        db.commit()
        assert cr.expected_label is None
        assert cr.correct is None
        assert cr.gold_match is None

    def test_development_claim_gold_fields_set(self, db):
        from app.db.repository import create_answer_result, create_claim_result
        run = _make_run(db, split="development")
        ar = create_answer_result(
            db, run_id=run.id, answer_id="A01", status="ok",
            error_code=None, error_message=None, turns=2,
            tool_calls=[], claim_count=1,
        )
        cr = create_claim_result(
            db,
            answer_result_id=ar.id,
            run_id=run.id,
            claim_id="A01-C01",
            answer_id="A01",
            quote="The speaker outputs 50W.",
            start_char=0,
            end_char=24,
            label="supported",
            evidence_fact_ids=["F001"],
            expected_label="supported",
            correct=True,
            gold_match=True,
        )
        db.commit()
        assert cr.expected_label == "supported"
        assert cr.correct is True
        assert cr.gold_match is True

    def test_claim_uniqueness_enforced(self, db):
        from app.db.repository import create_answer_result, create_claim_result
        from sqlalchemy.exc import IntegrityError
        run = _make_run(db)
        ar = create_answer_result(
            db, run_id=run.id, answer_id="A45", status="ok",
            error_code=None, error_message=None, turns=2,
            tool_calls=[], claim_count=2,
        )
        create_claim_result(
            db, answer_result_id=ar.id, run_id=run.id,
            claim_id="A45-C01", answer_id="A45", quote="q1.",
            start_char=0, end_char=3, label="supported", evidence_fact_ids=[],
        )
        db.commit()
        with pytest.raises(IntegrityError):
            create_claim_result(
                db, answer_result_id=ar.id, run_id=run.id,
                claim_id="A45-C01", answer_id="A45", quote="q1.",  # duplicate claim_id
                start_char=0, end_char=3, label="supported", evidence_fact_ids=[],
            )
            db.commit()
        db.rollback()


# ---------------------------------------------------------------------------
# Joined retrieval
# ---------------------------------------------------------------------------

class TestJoinedRetrieval:
    def test_get_run_returns_answer_and_claim_results(self, db):
        from app.db.repository import create_answer_result, create_claim_result, get_run
        run = _make_run(db, split="check")
        ar = create_answer_result(
            db, run_id=run.id, answer_id="A47", status="ok",
            error_code=None, error_message=None, turns=2,
            tool_calls=["get_brand_facts"], claim_count=2,
        )
        create_claim_result(
            db, answer_result_id=ar.id, run_id=run.id,
            claim_id="A47-C01", answer_id="A47", quote="Claim one.",
            start_char=0, end_char=10, label="unknown", evidence_fact_ids=[],
        )
        create_claim_result(
            db, answer_result_id=ar.id, run_id=run.id,
            claim_id="A47-C02", answer_id="A47", quote="Claim two.",
            start_char=11, end_char=21, label="supported", evidence_fact_ids=["F002"],
        )
        db.commit()

        retrieved = get_run(db, run.id)
        assert len(retrieved.answer_results) == 1
        ar_out = retrieved.answer_results[0]
        assert ar_out.answer_id == "A47"
        claims = sorted(ar_out.claim_results, key=lambda c: c.start_char)
        assert len(claims) == 2
        assert claims[0].claim_id == "A47-C01"
        assert claims[1].label == "supported"
        # Check gold fields are NULL for check run
        assert claims[0].expected_label is None
        assert claims[0].correct is None

    def test_answer_results_ordered_by_answer_id(self, db):
        from app.db.repository import create_answer_result, get_run
        run = _make_run(db, split="check")
        for aid in ("A50", "A48", "A49"):
            create_answer_result(
                db, run_id=run.id, answer_id=aid, status="ok",
                error_code=None, error_message=None, turns=1, tool_calls=[], claim_count=0,
            )
        db.commit()
        retrieved = get_run(db, run.id)
        ids = [ar.answer_id for ar in retrieved.answer_results]
        assert ids == sorted(ids)


def test_check_evaluation_endpoint_persists_ordered_terminal_outcomes(engine, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from app.agent.extractor import ExtractedClaim
    from app.agent.provider import AsyncOpenAIProvider
    from app.evaluation.scorer import AnswerPrediction
    from app.agent.runner import AnswerResult
    from app.evaluation.runs import RunExecution
    from app.tracing import TraceContext
    from app.main import app
    from app.schemas import Answer, Label, Split

    answers = [
        Answer(answer_id=f"A{i:02d}", split=Split.CHECK, brand_id="brand", question="q", ai_answer="claim.")
        for i in (3, 1, 2)
    ]
    predictions = [
        AnswerPrediction(answer_id="A01"),
        AnswerPrediction(answer_id="A02", error="provider_error"),
        AnswerPrediction(
            answer_id="A03",
            claims=[
                ExtractedClaim(
                    claim_id="A03-C01",
                    quote="claim.",
                    start_char=0,
                    end_char=6,
                    label=Label.SUPPORTED,
                    evidence_fact_ids=["F001"],
                )
            ],
        ),
    ]
    settings = SimpleNamespace(
        answers_path=Path("answers.csv"),
        expected_answers=3,
        brand_facts_path=Path("brand_facts.csv"),
        fact_version="facts-v1",
        data_dir=Path("."),
    )
    provider = SimpleNamespace(model="test-model")

    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.evaluation_runs._get_shared_catalogue", lambda: object())
    monkeypatch.setattr("app.api.evaluation_runs._prompt_for", lambda version: "prompt")
    monkeypatch.setattr("app.evaluation.data_loader.load_answers", lambda *args: answers)
    monkeypatch.setattr("app.evaluation.data_loader.validate_answer_splits", lambda _: None)
    monkeypatch.setattr(
        AsyncOpenAIProvider, "from_env", classmethod(lambda cls: provider)
    )
    monkeypatch.setattr("app.api.evaluation_runs.make_provider", lambda inner: inner)
    monkeypatch.setattr(
        "app.evaluation.runs.run_metadata",
        lambda **kwargs: {"settings": {"max_turns": 3}},
    )
    monkeypatch.setattr(
        "app.api.evaluation_runs.attach_development_scores",
        lambda **kwargs: pytest.fail("check runs must not receive development scores"),
    )
    async def fake_execute_run(
        answers, provider, catalogue, run_dir, metadata, *,
        run_id, split, agent_version, agent,
    ):
        outcomes = {
            "A01": AnswerResult(
                "A01", turns=1, langfuse_observation_id="answer-observation-01"
            ),
            "A02": AnswerResult(
                "A02", error=AgentError("provider_error", "sanitized"),
                turns=2, tool_calls=["get_brand_facts"],
                tool_call_details=[
                    {"id": "tc-02", "name": "get_brand_facts", "brand_id": "brand"}
                ],
                langfuse_observation_id="answer-observation-02",
            ),
            "A03": AnswerResult(
                "A03", claims=predictions[2].claims, turns=2,
                tool_calls=["get_brand_facts"],
                tool_call_details=[
                    {"id": "tc-03", "name": "get_brand_facts", "brand_id": "brand"}
                ],
                langfuse_observation_id="answer-observation-03",
            ),
        }
        return RunExecution(
            predictions=predictions,
            trace=TraceContext(
                run_id=run_id,
                trace_id="run-trace-id",
                run_observation_id="run-observation-id",
                base_url="https://langfuse.example",
            ),
            answer_observation_ids={
                key: value.langfuse_observation_id for key, value in outcomes.items()
            },
            outcomes=outcomes,
        )

    from app.agent.extractor import AgentError

    monkeypatch.setattr("app.evaluation.runs.execute_run", fake_execute_run)

    with TestClient(app) as client:
        response = client.post(
            "/evaluation-runs",
            json={"split": "check", "agent_version": "v2"},
        )
        assert response.status_code == 201
        run = response.json()
        _TEST_RUN_IDS.add(uuid.UUID(run["id"]))
        assert run["status"] == "complete"
        assert run["scheduled_answers"] == 3
        assert (run["answers_ok"], run["answers_empty"], run["answers_error"]) == (1, 1, 1)
        assert run["G"] is None and run["extraction_f1"] is None
        assert [answer["answer_id"] for answer in run["answer_results"]] == [
            "A01",
            "A02",
            "A03",
        ]
        assert [answer["status"] for answer in run["answer_results"]] == [
            "valid_empty",
            "error",
            "ok",
        ]
        assert [answer["turns"] for answer in run["answer_results"]] == [1, 2, 2]
        assert [answer["tool_calls"] for answer in run["answer_results"]] == [
            [],
            ["get_brand_facts"],
            ["get_brand_facts"],
        ]
        assert run["answer_results"][2]["tool_call_details"] == [
            {"id": "tc-03", "name": "get_brand_facts", "brand_id": "brand"}
        ]
        assert run["langfuse_trace_id"] == "run-trace-id"
        assert [
            answer["langfuse_observation_id"] for answer in run["answer_results"]
        ] == [
            "answer-observation-01",
            "answer-observation-02",
            "answer-observation-03",
        ]
        claim = run["answer_results"][2]["claims"][0]
        assert claim["expected_label"] is None
        assert claim["correct"] is None
        assert claim["gold_match"] is None

        retrieved = client.get(f"/evaluation-runs/{run['id']}")
        assert retrieved.status_code == 200
        assert retrieved.json()["answer_results"] == run["answer_results"]
