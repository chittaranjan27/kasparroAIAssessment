import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar

from app.agent.fact_tool import FactCatalogue
from app.agent.extractor import ExtractedClaim
from app.agent.provider import ProviderResponse, ToolCall
from app.agent.runner import run_answer
from app.config import get_settings
from app.evaluation.data_loader import load_brand_facts
from app.tracing import attach_development_scores, run_trace
from app.schemas import Answer, GoldClaim, Label


class FakeObservation:
    def __init__(self, observation_id, name, as_type, metadata, model, parent):
        self.id = observation_id
        self.trace_id = "trace-123"
        self.name = name
        self.as_type = as_type
        self.metadata = metadata
        self.model = model
        self.parent = parent
        self.updates = []

    @property
    def client(self):
        return self._client

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.current = ContextVar("fake_langfuse_current", default=None)
        self.observations = []
        self.scores = []

    @contextmanager
    def start_as_current_observation(
        self, *, as_type, name, metadata, model=None
    ):
        parent = self.current.get()
        item = FakeObservation(
            f"observation-{len(self.observations) + 1}",
            name,
            as_type,
            metadata,
            model,
            parent,
        )
        item._client = self
        self.observations.append(item)
        token = self.current.set(item)
        try:
            yield item
        finally:
            self.current.reset(token)

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def get_trace_url(self, *, trace_id):
        return f"https://langfuse.example/trace/{trace_id}"


class FakeProvider:
    model = "test-model"

    def __init__(self):
        self.responses = [
            ProviderResponse(
                None,
                [
                    ToolCall(
                        "tool-1",
                        "get_brand_facts",
                        json.dumps({"brand_id": "cleartrail"}),
                    )
                ],
            ),
            ProviderResponse(json.dumps({"claims": []}), usage={"input": 10, "output": 2}),
        ]

    async def acomplete(self, _messages, _tools, _tool_choice):
        return self.responses.pop(0)


def test_model_tool_trace_has_safe_metadata_and_run_link(monkeypatch):
    from app import tracing

    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: client)
    settings = get_settings()
    catalogue = FactCatalogue(
        load_brand_facts(settings.brand_facts_path, settings.expected_facts),
        settings.fact_version,
    )
    answer = Answer(
        answer_id="A01",
        split="development",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="The product costs 1299 rupees.",
    )

    with run_trace(
        run_id="run-1",
        split="development",
        agent_version="v1",
        metadata={"fact_version": settings.fact_version},
    ) as trace:
        result = asyncio.run(
            run_answer(
                answer,
                FakeProvider(),
                catalogue,
                trace_metadata={"run_id": "run-1", "split": "development"},
            )
        )

    assert result.ok
    assert trace.trace_id == "trace-123"
    assert trace.trace_url == "https://langfuse.example/trace/trace-123"
    by_name = {observation.name: observation for observation in client.observations}
    assert by_name["kasparro-fse-i-run"].as_type == "span"
    assert by_name["kasparro-fse-i-answer"].parent is by_name["kasparro-fse-i-run"]
    assert by_name["model-generation"].as_type == "generation"
    assert by_name["get_brand_facts"].as_type == "tool"
    assert by_name["model-generation"].parent is by_name["kasparro-fse-i-answer"]
    assert by_name["get_brand_facts"].parent is by_name["kasparro-fse-i-answer"]
    assert any(
        "latency_ms" in update.get("metadata", {})
        for update in by_name["model-generation"].updates
    )
    answer_metadata_updates = [
        update["metadata"]
        for update in by_name["kasparro-fse-i-answer"].updates
        if "metadata" in update
    ]
    assert answer_metadata_updates[-1]["outcome_status"] == "valid_empty"
    assert answer_metadata_updates[-1]["accepted_spans"] == []
    serialized = json.dumps(
        [
            {
                "name": item.name,
                "metadata": item.metadata,
                "updates": item.updates,
            }
            for item in client.observations
        ]
    )
    assert answer.ai_answer not in serialized
    assert answer.question not in serialized
    assert "returned_fact_ids" in serialized
    tool_metadata = by_name["get_brand_facts"].updates[-1]["metadata"]
    assert tool_metadata["returned_fact_count"] == len(tool_metadata["returned_fact_ids"])
    assert set(tool_metadata["returned_fact_ids"]) <= {
        fact["fact_id"] for fact in catalogue.get_brand_facts("cleartrail")["facts"]
    }


def test_answer_trace_records_accepted_spans_without_source_text(monkeypatch):
    from app import tracing

    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: client)
    settings = get_settings()
    catalogue = FactCatalogue(
        load_brand_facts(settings.brand_facts_path, settings.expected_facts),
        settings.fact_version,
    )
    answer = Answer(
        answer_id="A01",
        split="development",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="A claim that is not in the catalogue.",
    )
    provider = FakeProvider()
    provider.responses[1] = ProviderResponse(
        json.dumps(
            {
                "claims": [
                    {
                        "quote": answer.ai_answer,
                        "label": "unknown",
                        "evidence_fact_ids": [],
                    }
                ]
            }
        )
    )

    with run_trace(
        run_id="run-1",
        split="development",
        agent_version="v1",
        metadata={"fact_version": settings.fact_version},
    ):
        result = asyncio.run(
            run_answer(
                answer,
                provider,
                catalogue,
                trace_metadata={"run_id": "run-1", "split": "development"},
            )
        )

    assert result.ok
    answer_observation = next(
        item for item in client.observations if item.name == "kasparro-fse-i-answer"
    )
    metadata = answer_observation.updates[-1]["metadata"]
    assert metadata["accepted_spans"] == [
        {
            "claim_id": result.claims[0].claim_id,
            "span_id": f"A01:0:{len(answer.ai_answer)}",
            "start_char": 0,
            "end_char": len(answer.ai_answer),
            "label": "unknown",
            "fact_ids": [],
        }
    ]
    assert answer.ai_answer not in json.dumps(answer_observation.metadata)


def test_development_scores_attach_only_to_supplied_real_ids(monkeypatch):
    from app import tracing

    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: client)
    attach_development_scores(
        trace_id="trace-123",
        run_scores={"extraction_f1": 0.75, "joint_f1": 0.7},
        answer_scores={"A01": {"joint_f1": 0.5}, "A02": {"joint_f1": 0.0}},
        answer_observation_ids={"A01": "answer-observation-1", "A02": None},
        claim_scores=[
            {
                "answer_id": "A01",
                "claim_id": "A01-C01",
                "span_id": "A01:10:20",
                "start_char": 10,
                "end_char": 20,
                "label_correct": False,
            }
        ],
        run_metadata={"missed_gold_ids": ["A02:1:4"], "unmatched_prediction_ids": ["A01-C02"]},
    )
    assert [score["name"] for score in client.scores    ] == [
        "extraction_f1",
        "joint_f1",
        "joint_f1",
        "label_correct",
    ]
    assert "observation_id" not in client.scores[0]
    assert client.scores[2]["observation_id"] == "answer-observation-1"
    assert client.scores[0].get("metadata") is None
    assert client.scores[1]["metadata"]["missed_gold_ids"] == ["A02:1:4"]
    assert client.scores[3]["value"] == 0
    assert client.scores[3]["metadata"]["span_id"] == "A01:10:20"


def test_development_claim_scores_require_exact_answer_span_and_quote():
    from app.evaluation.runs import development_claim_scores

    quote = "A supported claim."
    prediction = ExtractedClaim(
        claim_id="A01-C01",
        quote=quote,
        start_char=3,
        end_char=3 + len(quote),
        label=Label.SUPPORTED,
        evidence_fact_ids=["F001"],
    )
    predictions = [
        type("Prediction", (), {"answer_id": "A01", "error": None, "claims": [prediction]})()
    ]
    gold = [
        GoldClaim(
            answer_id="A01",
            claim_id=f"A01:{3}:{3 + len(quote)}",
            start_char=3,
            end_char=3 + len(quote),
            quote=quote,
            expected_label=Label.CONTRADICTED,
            evidence_fact_ids=[],
            explanation="",
            difficulty_tags=[],
        ),
        GoldClaim(
            answer_id="A01",
            claim_id=f"A01:{30}:{40}",
            start_char=30,
            end_char=40,
            quote="A different quote.",
            expected_label=Label.SUPPORTED,
            evidence_fact_ids=[],
            explanation="",
            difficulty_tags=[],
        ),
    ]
    result = development_claim_scores(predictions, gold)
    assert result == [
        {
            "answer_id": "A01",
            "claim_id": "A01-C01",
            "span_id": f"A01:3:{3 + len(quote)}",
            "start_char": 3,
            "end_char": 3 + len(quote),
            "label_correct": False,
        }
    ]
