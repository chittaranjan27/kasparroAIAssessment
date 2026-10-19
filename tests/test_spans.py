import pytest

from app.agent.extractor import AgentError, RawClaim, locate_spans
from app.agent.validator import validate_claims
from app.schemas import Label

IDS = {"f1", "f2"}


def rc(quote, label="unknown", ctx="", ev=()):
    return RawClaim(quote=quote, label=Label(label), preceding_context=ctx, evidence_fact_ids=list(ev))


def test_offsets_exact_with_unicode_and_whitespace():
    text = "  Caf\u00e9 \u2014 great. \U0001F600 It has ANC.\r\n\tIt costs \u20b9999.  "
    quotes = ["It has ANC.", "It costs \u20b9999."]
    claims = locate_spans(text, [rc(q) for q in quotes], "A01")
    for c, q in zip(claims, quotes):
        assert text[c.start_char:c.end_char] == q
    validate_claims(text, claims, IDS)


def test_repeated_sentence_needs_source_context():
    text = "It has ANC. Other. It has ANC."
    with pytest.raises(AgentError) as e:  # two occurrences, no context -> ambiguous
        locate_spans(text, [rc("It has ANC.")], "A01")
    assert e.value.code == "ambiguous_quote"
    claims = locate_spans(text, [rc("It has ANC.", ctx="Other.")], "A01")
    assert claims[0].start_char == text.rindex("It has ANC.")
    with pytest.raises(AgentError):  # context matching no occurrence
        locate_spans(text, [rc("It has ANC.", ctx="Nowhere.")], "A01")


def test_repeated_sentence_both_claimed_in_order():
    text = "Start. It has ANC. Mid. It has ANC."
    claims = locate_spans(
        text,
        [rc("It has ANC.", ctx="Start."), rc("It has ANC.")],  # second: only one occurrence left after cursor
        "A01",
    )
    assert [c.start_char for c in claims] == [text.index("It has ANC."), text.rindex("It has ANC.")]
    validate_claims(text, claims, IDS)


def test_paraphrase_rejected():
    with pytest.raises(AgentError) as e:
        locate_spans("It has ANC.", [rc("It features ANC.")], "A01")
    assert e.value.code == "quote_not_found"


def test_validator_rules():
    text = "One is true. Two is true."
    good = locate_spans(text, [rc("One is true."), rc("Two is true.")], "A01")
    validate_claims(text, good, IDS)

    def bad(mutate, code):
        cs = [c.model_copy(deep=True) for c in good]
        mutate(cs)
        with pytest.raises(AgentError) as e:
            validate_claims(text, cs, IDS)
        assert e.value.code == code

    bad(lambda cs: setattr(cs[0], "end_char", 999), "invalid_span")
    bad(lambda cs: setattr(cs[0], "start_char", 1), "invalid_span")  # quote != slice
    bad(lambda cs: cs.reverse(), "invalid_span")  # ordering
    bad(lambda cs: setattr(cs[1], "claim_id", cs[0].claim_id), "invalid_output")
    bad(lambda cs: setattr(cs[0], "evidence_fact_ids", ["nope"]), "invalid_evidence")
    bad(lambda cs: setattr(cs[0], "label", Label.SUPPORTED), "invalid_evidence")  # no evidence


def test_overlap_rejected():
    text = "Abc def. Ghi."
    a = locate_spans(text, [rc("Abc def.")], "A01")[0]
    b = a.model_copy(update={"claim_id": "A01-C02", "start_char": 4, "end_char": 8, "quote": "def."})
    with pytest.raises(AgentError):
        validate_claims(text, [a, b], IDS)


def test_bad_label_rejected_at_parse():
    from app.agent.extractor import parse_output

    with pytest.raises(AgentError):
        parse_output('{"claims":[{"quote":"A.","label":"maybe"}]}')
