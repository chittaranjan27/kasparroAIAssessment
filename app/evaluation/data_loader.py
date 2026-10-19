"""CSV loading and integrity checks. Files are only ever opened read-only."""
from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.schemas import Answer, BrandFact, GoldClaim, Split, ValueType

ANSWER_COLUMNS = ["answer_id", "split", "brand_id", "question", "ai_answer"]
FACT_COLUMNS = [
    "fact_id", "brand_id", "model_id", "field", "value", "value_type",
    "unit", "condition", "valid_from", "valid_to",
]
GOLD_COLUMNS = [
    "answer_id", "claim_id", "start_char", "end_char", "quote",
    "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags",
]


class DataIntegrityError(ValueError):
    pass


def read_csv_rows(path: Path, required: list[str]) -> list[dict[str, str]]:
    """Strict UTF-8 read. newline="" keeps embedded CR/LF inside quoted cells intact."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise DataIntegrityError(f"{path.name}: missing columns {missing}")
        return list(reader)


def _check_count(name: str, rows: list, expected: int | None) -> None:
    if expected is not None and len(rows) != expected:
        raise DataIntegrityError(f"{name}: expected {expected} rows, found {len(rows)}")


def _none_if_empty(v: str) -> str | None:
    return v if v != "" else None


def parse_value(raw: str, value_type: ValueType) -> bool | int | float | str:
    """Explicit typing. Never bool(str)."""
    if value_type is ValueType.BOOLEAN:
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise DataIntegrityError(f"invalid boolean value {raw!r}")
    if value_type is ValueType.NUMBER:
        try:
            d = Decimal(raw)
        except InvalidOperation as exc:
            raise DataIntegrityError(f"invalid number value {raw!r}") from exc
        return int(d) if d == d.to_integral_value() else float(d)
    return raw


def load_answers(path: Path, expected_count: int | None = 50) -> list[Answer]:
    rows = read_csv_rows(path, ANSWER_COLUMNS)
    _check_count("answers.csv", rows, expected_count)
    answers = [Answer(**{c: r[c] for c in ANSWER_COLUMNS}) for r in rows]
    ids = [a.answer_id for a in answers]
    if len(set(ids)) != len(ids):
        raise DataIntegrityError("answers.csv: duplicate answer_id")
    return answers


def validate_answer_splits(answers: list[Answer]) -> None:
    """A01-A40 must be development, A41-A50 must be check."""
    dev = {f"A{i:02d}" for i in range(1, 41)}
    chk = {f"A{i:02d}" for i in range(41, 51)}
    got_dev = {a.answer_id for a in answers if a.split is Split.DEVELOPMENT}
    got_chk = {a.answer_id for a in answers if a.split is Split.CHECK}
    if got_dev != dev or got_chk != chk:
        raise DataIntegrityError("split assignment does not match A01-A40 / A41-A50")


def load_brand_facts(path: Path, expected_count: int | None = 132) -> list[BrandFact]:
    rows = read_csv_rows(path, FACT_COLUMNS)
    _check_count("brand_facts.csv", rows, expected_count)
    facts: list[BrandFact] = []
    for r in rows:
        vt = ValueType(r["value_type"])
        facts.append(
            BrandFact(
                fact_id=r["fact_id"],
                brand_id=r["brand_id"],
                model_id=r["model_id"],
                field=r["field"],
                value=parse_value(r["value"], vt),
                value_type=vt,
                unit=_none_if_empty(r["unit"]),
                condition=_none_if_empty(r["condition"]),
                valid_from=date.fromisoformat(r["valid_from"]),
                valid_to=date.fromisoformat(r["valid_to"]),
            )
        )
    ids = [f.fact_id for f in facts]
    if len(set(ids)) != len(ids):
        raise DataIntegrityError("brand_facts.csv: duplicate fact_id")
    return facts


def load_gold(
    path: Path,
    answers: list[Answer] | None = None,
    expected_count: int | None = 376,
) -> list[GoldClaim]:
    """Evaluation-only loader. Do not call from the agent, tool or API request path."""
    rows = read_csv_rows(path, GOLD_COLUMNS)
    _check_count("development_gold.csv", rows, expected_count)
    claims = [
        GoldClaim(
            answer_id=r["answer_id"],
            claim_id=r["claim_id"],
            start_char=int(r["start_char"]),
            end_char=int(r["end_char"]),
            quote=r["quote"],
            expected_label=r["expected_label"],
            evidence_fact_ids=[x for x in r["evidence_fact_ids"].split("|") if x],
            explanation=r["explanation"],
            difficulty_tags=[x for x in r["difficulty_tags"].split("|") if x],
        )
        for r in rows
    ]
    if len({c.claim_id for c in claims}) != len(claims):
        raise DataIntegrityError("development_gold.csv: duplicate claim_id")
    if answers is not None:
        by_id = {a.answer_id: a for a in answers}
        for c in claims:
            a = by_id.get(c.answer_id)
            if a is None or a.split is not Split.DEVELOPMENT:
                raise DataIntegrityError(f"{c.claim_id}: gold references non-development answer")
            if a.ai_answer[c.start_char:c.end_char] != c.quote:
                raise DataIntegrityError(f"{c.claim_id}: offsets do not match quote")
    return claims
