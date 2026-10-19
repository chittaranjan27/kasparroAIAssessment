"""Independent validation of located claims against the original answer and retrieved facts."""
from __future__ import annotations

from app.agent.extractor import BOUNDARY_PUNCT, AgentError, ExtractedClaim
from app.schemas import Label


def validate_claims(ai_answer: str, claims: list[ExtractedClaim], allowed_fact_ids: set[str]) -> None:
    seen_ids: set[str] = set()
    prev_end = 0
    for c in claims:
        if c.claim_id in seen_ids:
            raise AgentError("invalid_output", f"duplicate claim_id {c.claim_id}")
        seen_ids.add(c.claim_id)
        if not (0 <= c.start_char < c.end_char <= len(ai_answer)):
            raise AgentError("invalid_span", f"{c.claim_id}: offsets out of bounds")
        if ai_answer[c.start_char:c.end_char] != c.quote:
            raise AgentError("invalid_span", f"{c.claim_id}: quote != ai_answer[start:end]")
        if c.start_char < prev_end:
            raise AgentError("invalid_span", f"{c.claim_id}: out of order or overlapping")
        prev_end = c.end_char
        if c.quote != c.quote.strip() or c.quote[-1] not in BOUNDARY_PUNCT:
            raise AgentError("invalid_span", f"{c.claim_id}: quote is not a whole sentence with final punctuation")
        if c.label not in tuple(Label):
            raise AgentError("invalid_output", f"{c.claim_id}: bad label")
        unknown = [f for f in c.evidence_fact_ids if f not in allowed_fact_ids]
        if unknown:
            raise AgentError("invalid_evidence", f"{c.claim_id}: fact ids not retrieved from catalogue: {unknown}")
        if c.label in (Label.SUPPORTED, Label.CONTRADICTED) and not c.evidence_fact_ids:
            raise AgentError("invalid_evidence", f"{c.claim_id}: {c.label.value} requires evidence_fact_ids")
