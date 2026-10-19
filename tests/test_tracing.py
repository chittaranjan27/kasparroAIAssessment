import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar

from app.agent.fact_tool import FactCatalogue
from app.agent.provider import ProviderResponse, ToolCall
from app.agent.runner import run_answer
from app.config import get_settings
from app.evaluation.data_loader import load_brand_facts
from app.tracing import attach_development_scores, run_trace
from app.schemas import Answer


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
    by_name = {observation.name: observation for observation in client.observations}
    assert by_name["kasparro-fse-i-run"].as_type == "span"
    assert by_name["kasparro-fse-i-answer"].parent is by_name["kasparro-fse-i-run"]
    assert by_name["model-generation"].as_type == "generation"
    assert by_name["agent-tool-call"].as_type == "tool"
    assert by_name["model-generation"].parent is by_name["kasparro-fse-i-answer"]
    assert by_name["agent-tool-call"].parent is by_name["kasparro-fse-i-answer"]
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
    assert "F001" not in serialized


def test_development_scores_attach_only_to_supplied_real_ids(monkeypatch):
    from app import tracing

    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: client)
    attach_development_scores(
        trace_id="trace-123",
        run_scores={"extraction_f1": 0.75},
        answer_scores={"A01": {"joint_f1": 0.5}, "A02": {"joint_f1": 0.0}},
        answer_observation_ids={"A01": "answer-observation-1", "A02": None},
    )
    assert [score["name"] for score in client.scores] == [
        "extraction_f1",
        "joint_f1",
    ]
    assert "observation_id" not in client.scores[0]
    assert client.scores[1]["observation_id"] == "answer-observation-1"
