"""Regression tests for the diagnosed v1 failure (supported with no cited fact_id) and v1 freeze."""
import hashlib
import asyncio
import json

from app.agent import prompts_v1, prompts_v2
from app.agent.fact_tool import FactCatalogue
from app.agent.provider import ProviderResponse, ToolCall
from app.agent.runner import run_answer as _run_answer
from app.config import get_settings
from app.evaluation.data_loader import load_brand_facts
from app.schemas import Answer

S = get_settings()
CAT = FactCatalogue(load_brand_facts(S.brand_facts_path, S.expected_facts))
# Sentence shape taken from the baseline failure: a plausible feature claim that no catalogue fact covers.
SENT = "ClearTrail Plus accepts two active Bluetooth source connections at the same time."
TEXT = "The Plus costs a lot. " + SENT
V1_SHA = "ddf5a4cc6b39a38be5baa3bc858be422cfd983098d374eb8325497655d6453ba"  # frozen v1 prompt


class Fake:
    def __init__(self, claims):
        self.claims, self.seen = claims, []

    async def acomplete(self, messages, tools, tool_choice):
        self.seen.append(messages[0]["content"])
        if len(self.seen) == 1:
            return ProviderResponse(None, [ToolCall("t", "get_brand_facts", '{"brand_id": "cleartrail"}')])
        return ProviderResponse(json.dumps({"claims": self.claims}))


def run_answer(*args, **kwargs):
    return asyncio.run(_run_answer(*args, **kwargs))


def answer():
    return Answer(answer_id="A01", split="development", brand_id="cleartrail", question="q", ai_answer=TEXT)


def test_v1_prompt_is_frozen():
    assert hashlib.sha256(prompts_v1.SYSTEM_PROMPT.encode()).hexdigest() == V1_SHA


def test_v2_differs_from_v1_only_in_evidence_step():
    v1, v2 = prompts_v1.SYSTEM_PROMPT, prompts_v2.SYSTEM_PROMPT
    assert v1 != v2
    cut = lambda s: s.split("5. evidence_fact_ids", 1)[0] + s.split("6. If the same sentence", 1)[1]
    assert cut(v1) == cut(v2)  # everything outside step 5 is identical
    assert "MUST be \"unknown\"" in v2 and "MUST be \"unknown\"" not in v1


def test_default_run_uses_v1_prompt_and_v2_is_opt_in():
    claim = {"quote": SENT, "label": "unknown", "evidence_fact_ids": [], "preceding_context": ""}
    f1, f2 = Fake([claim]), Fake([claim])
    run_answer(answer(), f1, CAT)
    run_answer(answer(), f2, CAT, system_prompt=prompts_v2.SYSTEM_PROMPT)
    assert f1.seen[0] == prompts_v1.SYSTEM_PROMPT and f2.seen[0] == prompts_v2.SYSTEM_PROMPT


def test_baseline_failure_mode_still_rejected_by_unchanged_validator():
    bad = {"quote": SENT, "label": "supported", "evidence_fact_ids": [], "preceding_context": ""}
    r = run_answer(answer(), Fake([bad]), CAT, system_prompt=prompts_v2.SYSTEM_PROMPT)
    assert r.error and r.error.code == "invalid_evidence" and r.claims == []


def test_uncovered_claim_labelled_unknown_without_evidence_is_accepted():
    ok = {"quote": SENT, "label": "unknown", "evidence_fact_ids": [], "preceding_context": ""}
    r = run_answer(answer(), Fake([ok]), CAT, system_prompt=prompts_v2.SYSTEM_PROMPT)
    assert r.ok and r.claims[0].label.value == "unknown"
