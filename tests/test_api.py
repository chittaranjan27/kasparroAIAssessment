"""API tests using FastAPI TestClient (synchronous, no live provider calls).

All tests that exercise /classify mock the agent runner and the catalogue so that:
  - No OPENAI_API_KEY is required
  - No network requests are made
  - No CSV files are required

Tests that exercise /evaluation-runs similarly mock execute_run and data_loader.

The database tests are in tests/test_database.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Do not override DATABASE_URL here: PostgreSQL tests run in the same process.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Patch DB table creation so TestClient does not need a live DB
# ---------------------------------------------------------------------------
with patch("app.db.session.create_tables", return_value=None):
    from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_catalogue(known_brands=("BrandA",)):
    cat = MagicMock()
    cat.brands = frozenset(known_brands)
    return cat


def _mock_result(answer_id="A01", claims=None, error=None, empty=False):
    from app.agent.extractor import AgentError, ExtractedClaim
    from app.schemas import Label
    from app.agent.runner import AnswerResult
    r = AnswerResult(answer_id=answer_id)
    r.turns = 2
    r.tool_calls = ["get_brand_facts"]
    if error:
        r.error = AgentError(error, f"{error} detail")
    elif empty:
        r.claims = []
    else:
        r.claims = claims if claims is not None else [
            ExtractedClaim(
                claim_id=f"{answer_id}-C01",
                quote="The speaker has 50W power output.",
                start_char=0,
                end_char=31,
                label=Label.SUPPORTED,
                evidence_fact_ids=["F001"],
            )
        ]
    return r


def _async_result(result=None, error=None):
    async def call(*args, **kwargs):
        if error is not None:
            raise error
        return result

    return call


# ---------------------------------------------------------------------------
# POST /classify – valid request
# ---------------------------------------------------------------------------

class TestClassifySuccess:
    def test_200_returns_claims(self):
        mock_result = _mock_result()
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify.run_answer", new=_async_result(mock_result)),
            patch("app.api.classify._persist_classify"),
        ):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "BrandA",
                "question": "Is the power 50W?",
                "ai_answer": "The speaker has 50W power output.",
                "agent_version": "v1",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer_id"] == "A01"
        assert body["agent_version"] == "v1"
        assert len(body["claims"]) == 1
        assert body["claims"][0]["label"] == "supported"

    def test_200_empty_claims_valid(self):
        mock_result = _mock_result(empty=True)
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify.run_answer", new=_async_result(mock_result)),
            patch("app.api.classify._persist_classify"),
        ):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "BrandA",
                "question": "Q",
                "ai_answer": "No factual claims here.",
                "agent_version": "v2",
            })
        assert resp.status_code == 200
        assert resp.json()["claims"] == []

    def test_agent_version_v2_accepted(self):
        mock_result = _mock_result()
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify.run_answer", new=_async_result(mock_result)),
            patch("app.api.classify._persist_classify"),
        ):
            resp = client.post("/classify", json={
                "answer_id": "A41",
                "brand_id": "BrandA",
                "question": "Q",
                "ai_answer": "The speaker has 50W power output.",
                "agent_version": "v2",
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /classify – 404 unknown brand
# ---------------------------------------------------------------------------

class TestClassify404:
    def test_unknown_brand_returns_404(self):
        with patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "UNKNOWN_BRAND",
                "question": "Q",
                "ai_answer": "Some answer.",
                "agent_version": "v1",
            })
        assert resp.status_code == 404
        assert "UNKNOWN_BRAND" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /classify – 422 validation errors
# ---------------------------------------------------------------------------

class TestClassify422:
    def test_missing_required_field(self):
        resp = client.post("/classify", json={
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
            # missing answer_id
        })
        assert resp.status_code == 422

    def test_invalid_agent_version(self):
        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v3",  # invalid
        })
        assert resp.status_code == 422

    def test_forbidden_gold_field_expected_label(self):
        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
            "expected_label": "supported",  # FORBIDDEN
        })
        assert resp.status_code == 422

    def test_forbidden_field_claims(self):
        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
            "claims": [],  # FORBIDDEN
        })
        assert resp.status_code == 422

    def test_forbidden_field_gold(self):
        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
            "gold": {"something": "bad"},
        })
        assert resp.status_code == 422

    def test_empty_ai_answer(self):
        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "",
            "agent_version": "v1",
        })
        assert resp.status_code == 422

    @pytest.mark.parametrize("field", ("answer_id", "brand_id", "question", "ai_answer"))
    def test_blank_required_value(self, field):
        payload = {
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
        }
        payload[field] = "   "
        assert client.post("/classify", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /classify – 502 provider / output errors
# ---------------------------------------------------------------------------

class TestClassify502:
    def test_provider_error_returns_502(self):
        from app.agent.provider import ProviderError
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify._get_provider"),
            patch(
                "app.api.classify.run_answer",
                new=_async_result(error=ProviderError("timeout", transient=False)),
            ),
            patch("app.api.classify._persist_classify") as persist,
        ):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "BrandA",
                "question": "Q",
                "ai_answer": "Some answer.",
                "agent_version": "v1",
            })
        assert resp.status_code == 502
        assert persist.call_args.args[0].error.code == "provider_error"

    def test_provider_timeout_returns_504_and_persists_error(self):
        from app.agent.provider import ProviderError
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify._get_provider"),
            patch(
                "app.api.classify.run_answer",
                new=_async_result(
                    error=ProviderError("APITimeoutError", timed_out=True)
                ),
            ),
            patch("app.api.classify._persist_classify") as persist,
        ):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "BrandA",
                "question": "Q",
                "ai_answer": "Some answer.",
                "agent_version": "v1",
            })
        assert resp.status_code == 504
        assert persist.call_args.args[0].error.code == "provider_timeout"

    def test_agent_error_invalid_output_returns_502(self):
        mock_result = _mock_result(error="invalid_output")
        with (
            patch("app.api.classify._get_shared_catalogue", return_value=_mock_catalogue()),
            patch("app.api.classify._get_provider"),
            patch("app.api.classify.run_answer", new=_async_result(mock_result)),
            patch("app.api.classify._persist_classify") as persist,
        ):
            resp = client.post("/classify", json={
                "answer_id": "A01",
                "brand_id": "BrandA",
                "question": "Q",
                "ai_answer": "Some answer.",
                "agent_version": "v1",
            })
        assert resp.status_code == 502
        assert persist.call_args.args[0].error.code == "invalid_output"

    @pytest.mark.parametrize(
        "forbidden",
        ("gold_claims", "gold_claim_list", "claim_list", "expected_claims"),
    )
    def test_forbidden_claim_lists_are_rejected(self, forbidden):
        payload = {
            "answer_id": "A01",
            "brand_id": "BrandA",
            "question": "Q",
            "ai_answer": "Some answer.",
            "agent_version": "v1",
            forbidden: [],
        }
        assert client.post("/classify", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# GET /evaluation-runs/{id} – 404 and 422
# ---------------------------------------------------------------------------

class TestGetEvaluationRun:
    def test_404_for_unknown_uuid(self):
        with patch("app.api.evaluation_runs.get_run", return_value=None), \
             patch("app.api.evaluation_runs.get_session", return_value=MagicMock()):
            resp = client.get(f"/evaluation-runs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_422_for_invalid_uuid(self):
        resp = client.get("/evaluation-runs/not-a-uuid")
        assert resp.status_code == 422

    def test_200_returns_stable_run(self):
        """Return a mocked run and verify shape."""
        from datetime import datetime, timezone
        from app.db.models import EvaluationRun, AnswerResult as DBAns

        mock_run = MagicMock(spec=EvaluationRun)
        mock_run.id = uuid.uuid4()
        mock_run.status = "complete"
        mock_run.split = "check"
        mock_run.agent_version = "v1"
        mock_run.provider = "openai"
        mock_run.model = "gpt-4o-mini"
        mock_run.prompt_version = "v1"
        mock_run.fact_version = "speaker-catalogue-v4"
        mock_run.evaluator_version = "1.0"
        mock_run.scheduled_answers = 10
        mock_run.answers_ok = 10
        mock_run.answers_empty = 0
        mock_run.answers_error = 0
        mock_run.claims_extracted = 50
        mock_run.G = mock_run.P = mock_run.M = mock_run.C = None
        mock_run.extraction_precision = mock_run.extraction_recall = mock_run.extraction_f1 = None
        mock_run.joint_precision = mock_run.joint_recall = mock_run.joint_f1 = None
        mock_run.macro_f1 = mock_run.gold_coverage = None
        mock_run.started_at = datetime(2026, 10, 19, 8, 0, 0, tzinfo=timezone.utc)
        mock_run.finished_at = datetime(2026, 10, 19, 8, 5, 0, tzinfo=timezone.utc)
        mock_run.langfuse_trace_id = None
        mock_run.answer_results = []

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("app.api.evaluation_runs.get_run", return_value=mock_run), \
             patch("app.api.evaluation_runs.get_session", return_value=mock_db):
            resp = client.get(f"/evaluation-runs/{mock_run.id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["split"] == "check"
        assert body["G"] is None
        assert body["extraction_f1"] is None
        assert body["answer_results"] == []


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------

class TestHealthz:
    def test_healthz_returns_json(self):
        with patch("app.db.session.ping", return_value=True):
            resp = client.get("/healthz")
        # Status may be 200 or 503 depending on DB; just check JSON
        assert resp.headers["content-type"].startswith("application/json")
