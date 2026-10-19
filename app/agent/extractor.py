"""Parse model output and compute offsets in Python against the exact ai_answer."""
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas import Label

BOUNDARY_PUNCT = ".!?"


class AgentError(Exception):
    """Answer-level failure. `code` is machine readable."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RawClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str
    label: Label
    evidence_fact_ids: list[str] = []
    preceding_context: str = ""


class RawOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[RawClaim]


class ExtractedClaim(BaseModel):
    claim_id: str
    quote: str
    start_char: int
    end_char: int
    label: Label
    evidence_fact_ids: list[str]


def parse_output(text: str | None) -> RawOutput:
    if not text:
        raise AgentError("invalid_output", "empty model reply")
    try:
        return RawOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise AgentError("invalid_output", f"unparseable model reply ({type(exc).__name__})") from exc


def _occurrences(text: str, needle: str, start: int) -> list[int]:
    out, i = [], text.find(needle, start)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


def _on_boundaries(text: str, s: int, e: int) -> bool:
    return (s == 0 or text[s - 1].isspace()) and (e == len(text) or text[e].isspace())


def locate_spans(ai_answer: str, raw: list[RawClaim], answer_id: str) -> list[ExtractedClaim]:
    """Offsets are code-point indices into the untouched ai_answer.

    Claims are matched left to right: each search starts at the end of the previous claim, so repeated
    sentences resolve by position. If several candidates remain, `preceding_context` (source context
    supplied by the model) picks one; otherwise the answer fails as ambiguous. Gold spans are never used.
    """
    claims: list[ExtractedClaim] = []
    cursor = 0
    for n, rc in enumerate(raw, start=1):
        if not rc.quote:
            raise AgentError("invalid_output", f"claim {n}: empty quote")
        cands = _occurrences(ai_answer, rc.quote, cursor)
        if not cands:
            raise AgentError("quote_not_found", f"claim {n}: quote is not an exact substring at/after offset {cursor}")
        bounded = [c for c in cands if _on_boundaries(ai_answer, c, c + len(rc.quote))]
        cands = bounded or cands
        if len(cands) > 1 and rc.preceding_context.strip():
            ctx = rc.preceding_context.strip()
            cands = [c for c in cands if ai_answer[:c].rstrip().endswith(ctx)] or []
            if not cands:
                raise AgentError("ambiguous_quote", f"claim {n}: preceding_context matches no occurrence")
        if len(cands) != 1:
            raise AgentError("ambiguous_quote", f"claim {n}: quote occurs {len(cands)} times, no disambiguating context")
        start = cands[0]
        end = start + len(rc.quote)
        claims.append(
            ExtractedClaim(
                claim_id=f"{answer_id}-C{n:02d}",
                quote=rc.quote,
                start_char=start,
                end_char=end,
                label=rc.label,
                evidence_fact_ids=list(rc.evidence_fact_ids),
            )
        )
        cursor = end
    return claims
