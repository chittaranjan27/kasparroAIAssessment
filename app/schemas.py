"""Pydantic schemas for the supplied datasets."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Union

from pydantic import BaseModel, ConfigDict, model_validator


class Split(str, Enum):
    DEVELOPMENT = "development"
    CHECK = "check"


class ValueType(str, Enum):
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"


class Label(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


FactValue = Union[bool, int, float, str]


class Answer(BaseModel):
    """ai_answer is stored exactly as decoded from the CSV (no strip/normalise)."""

    model_config = ConfigDict(frozen=True)

    answer_id: str
    split: Split
    brand_id: str
    question: str
    ai_answer: str


class BrandFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: FactValue  # typed according to value_type
    value_type: ValueType
    unit: str | None = None
    condition: str | None = None
    valid_from: date
    valid_to: date

    @model_validator(mode="after")
    def _check_dates(self) -> "BrandFact":
        if self.valid_from > self.valid_to:
            raise ValueError(f"{self.fact_id}: valid_from after valid_to")
        return self

    def is_valid_on(self, on: date) -> bool:
        """Both endpoints are inclusive."""
        return self.valid_from <= on <= self.valid_to


class GoldClaim(BaseModel):
    """Evaluation-only. Must never be passed to the model or the fact tool."""

    model_config = ConfigDict(frozen=True)

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: Label
    evidence_fact_ids: list[str]
    explanation: str
    difficulty_tags: list[str]
