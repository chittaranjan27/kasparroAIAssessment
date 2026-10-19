"""Read-only catalogue lookup exposed to the model as a tool.

This module must never import or touch development_gold.csv.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas import BrandFact

FACT_VERSION = "speaker-catalogue-v4"


class ToolError(ValueError):
    """Raised for unknown brands, unknown tools or invalid arguments."""


class GetBrandFactsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brand_id: str


class FactCatalogue:
    def __init__(self, facts: list[BrandFact], fact_version: str = FACT_VERSION):
        self._facts = tuple(facts)
        self.fact_version = fact_version
        self.brands = frozenset(f.brand_id for f in facts)

    def get_brand_facts(self, brand_id: str) -> dict[str, Any]:
        if brand_id not in self.brands:
            raise ToolError(f"unknown brand_id {brand_id!r}")
        return {
            "fact_version": self.fact_version,
            "brand_id": brand_id,
            "facts": [
                f.model_dump(mode="json") for f in self._facts if f.brand_id == brand_id
            ],
        }


TOOL_DEFINITIONS = [
    {
        "name": "get_brand_facts",
        "description": "Return all catalogue facts for one brand (read-only).",
        "parameters": {
            "type": "object",
            "properties": {"brand_id": {"type": "string"}},
            "required": ["brand_id"],
            "additionalProperties": False,
        },
    }
]


def execute_tool(name: str, arguments: dict[str, Any], catalogue: FactCatalogue) -> dict[str, Any]:
    """Dispatch via a fixed allowlist; model-provided names are never resolved dynamically."""
    handlers: dict[str, Callable[[FactCatalogue, dict[str, Any]], dict[str, Any]]] = {
        "get_brand_facts": _run_get_brand_facts,
    }
    handler = handlers.get(name)
    if handler is None:
        raise ToolError(f"unknown tool {name!r}")
    return handler(catalogue, arguments)


def _run_get_brand_facts(catalogue: FactCatalogue, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        args = GetBrandFactsArgs.model_validate(arguments)
    except ValidationError as exc:
        raise ToolError(f"invalid arguments: {exc.error_count()} error(s)") from exc
    return catalogue.get_brand_facts(args.brand_id)
