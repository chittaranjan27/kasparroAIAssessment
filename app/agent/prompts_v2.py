"""Prompt v2 = prompt v1 with exactly ONE behaviour changed (step 5, the evidence rule).

Baseline evidence: 6 of 10 v1 answer-level errors were `supported` claims with no cited fact_id, and 13 of 48
v1 wrong labels were gold `unknown` predicted `supported`. v2 makes the rule explicit: no cited retrieved
fact => the label must be "unknown". v1 text is imported, not copied, so v1 stays byte-identical.
"""
from app.agent.prompts_v1 import SYSTEM_PROMPT as _V1_PROMPT
from app.agent.prompts_v1 import build_user_message  # noqa: F401  (unchanged)

_V1_STEP_5 = (
    "5. evidence_fact_ids: the fact_id values (copied exactly from the tool result) that you relied on. "
    "Required for supported and contradicted.\n"
)
_V2_STEP_5 = (
    "5. evidence_fact_ids: the fact_id values (copied exactly from the tool result) that you relied on.\n"
    "   Evidence rule: you may use \"supported\" or \"contradicted\" ONLY if you can cite at least one fact_id from the\n"
    "   tool result that directly bears on the claim. If you cannot name such a fact_id (the catalogue has no fact\n"
    "   about this attribute, model or condition), the label MUST be \"unknown\" with evidence_fact_ids = [].\n"
    "   Never label a claim \"supported\" because it sounds plausible or generic.\n"
)

assert _V1_STEP_5 in _V1_PROMPT, "v1 prompt changed; v2 derivation no longer valid"
SYSTEM_PROMPT = _V1_PROMPT.replace(_V1_STEP_5, _V2_STEP_5)
