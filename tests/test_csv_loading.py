from datetime import date

import pytest

from app.agent.fact_tool import FactCatalogue, ToolError, execute_tool
from app.config import get_settings
from app.evaluation.data_loader import (
    DataIntegrityError,
    load_answers,
    load_brand_facts,
    load_gold,
    parse_value,
    validate_answer_splits,
)
from app.schemas import BrandFact, Split, ValueType

S = get_settings()
ANS_HDR = "answer_id,split,brand_id,question,ai_answer\n"
FACT_HDR = "fact_id,brand_id,model_id,field,value,value_type,unit,condition,valid_from,valid_to\n"


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


# ---------- parser edge cases (synthetic files) ----------
def test_quoted_commas_and_embedded_newlines_preserved(tmp_path):
    answer = 'Hello, "world"\r\n  line two  \n\ttrailing  '
    quoted = '"' + answer.replace('"', '""') + '"'
    p = write(tmp_path, "a.csv", ANS_HDR + f'A01,development,b,"Q, with comma",{quoted}\n')
    (a,) = load_answers(p, expected_count=1)
    assert a.ai_answer == answer
    assert a.question == "Q, with comma"


def test_unicode_and_whitespace_not_normalised(tmp_path):
    answer = "  caf\u00e9 \u2014 \u201cquoted\u201d  \U0001F600 x"
    p = write(tmp_path, "a.csv", ANS_HDR + f'A01,development,b,q,"{answer}"\n')
    (a,) = load_answers(p, expected_count=1)
    assert a.ai_answer == answer
    assert a.ai_answer[-1] == "x" and len(a.ai_answer) == len(answer)


def test_empty_cells_become_none(tmp_path):
    p = write(tmp_path, "f.csv", FACT_HDR + "x.1,b,m,f,5,number,,,2026-01-01,2026-12-31\n")
    (f,) = load_brand_facts(p, expected_count=1)
    assert f.unit is None and f.condition is None and f.value == 5


def test_missing_column_rejected(tmp_path):
    p = write(tmp_path, "a.csv", "answer_id,split\nA01,development\n")
    with pytest.raises(DataIntegrityError):
        load_answers(p, expected_count=1)


def test_wrong_row_count_and_duplicate_ids_rejected(tmp_path):
    row = "A01,development,b,q,a\n"
    p = write(tmp_path, "a.csv", ANS_HDR + row + row)
    with pytest.raises(DataIntegrityError):
        load_answers(p, expected_count=3)
    with pytest.raises(DataIntegrityError):
        load_answers(p, expected_count=2)  # duplicate id


def test_value_type_parsing_never_uses_bool_of_string():
    assert parse_value("false", ValueType.BOOLEAN) is False
    assert parse_value("true", ValueType.BOOLEAN) is True
    with pytest.raises(DataIntegrityError):
        parse_value("False", ValueType.BOOLEAN)
    assert parse_value("1299", ValueType.NUMBER) == 1299
    assert isinstance(parse_value("1299", ValueType.NUMBER), int)
    assert parse_value("4.5", ValueType.NUMBER) == 4.5
    assert parse_value("false", ValueType.STRING) == "false"
    with pytest.raises(DataIntegrityError):
        parse_value("abc", ValueType.NUMBER)


def test_inclusive_date_endpoints():
    f = BrandFact(
        fact_id="x", brand_id="b", model_id="m", field="f", value=1,
        value_type=ValueType.NUMBER, valid_from=date(2026, 1, 1), valid_to=date(2026, 12, 31),
    )
    assert f.is_valid_on(date(2026, 1, 1))
    assert f.is_valid_on(date(2026, 12, 31))
    assert not f.is_valid_on(date(2025, 12, 31))
    assert not f.is_valid_on(date(2027, 1, 1))


# ---------- supplied dataset integrity (read-only) ----------
@pytest.fixture(scope="module")
def answers():
    return load_answers(S.answers_path, S.expected_answers)


@pytest.fixture(scope="module")
def facts():
    return load_brand_facts(S.brand_facts_path, S.expected_facts)


def test_answers_integrity(answers):
    assert len(answers) == 50
    validate_answer_splits(answers)
    assert sum(a.split is Split.DEVELOPMENT for a in answers) == 40
    assert sum(a.split is Split.CHECK for a in answers) == 10


def test_facts_integrity(facts):
    assert len(facts) == 132
    assert all(isinstance(f.value, bool) for f in facts if f.value_type is ValueType.BOOLEAN)
    assert all(
        isinstance(f.value, (int, float)) and not isinstance(f.value, bool)
        for f in facts if f.value_type is ValueType.NUMBER
    )


def test_gold_integrity_and_offsets(answers):
    gold = load_gold(S.gold_path, answers, S.expected_gold)
    assert len(gold) == 376


# ---------- fact tool ----------
def test_fact_tool_brand_lookup(facts):
    cat = FactCatalogue(facts)
    out = execute_tool("get_brand_facts", {"brand_id": "cleartrail"}, cat)
    assert out["fact_version"] == "speaker-catalogue-v4"
    assert len(out["facts"]) == 44
    assert {f["brand_id"] for f in out["facts"]} == {"cleartrail"}


def test_fact_tool_rejects_unknown_brand_tool_and_args(facts):
    cat = FactCatalogue(facts)
    with pytest.raises(ToolError):
        execute_tool("get_brand_facts", {"brand_id": "nope"}, cat)
    for bad in ("os.system", "__import__", "get_gold", ""):
        with pytest.raises(ToolError):
            execute_tool(bad, {}, cat)
    with pytest.raises(ToolError):
        execute_tool("get_brand_facts", {"brand_id": "cloudnote", "extra": 1}, cat)
    with pytest.raises(ToolError):
        execute_tool("get_brand_facts", {}, cat)


def test_fact_tool_module_does_not_reference_gold():
    import app.agent.fact_tool as ft
    src = open(ft.__file__, encoding="utf-8").read()
    assert "data_loader" not in src.replace("development_gold.csv", "")
    assert "load_gold" not in src
