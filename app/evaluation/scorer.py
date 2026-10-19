"""Exact-span matching of saved predictions against gold. The only place gold meets predictions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, model_validator

from app.agent.extractor import ExtractedClaim
from app.evaluation.metrics import (
    PRF,
    empty_confusion,
    extraction_metrics,
    joint_metrics,
    label_metrics,
    macro_f1,
)
from app.schemas import GoldClaim


class AnswerPrediction(BaseModel):
    """Saved agent output for one answer. `error` is the answer-level error code (claims must then be empty).
    error=None with claims=[] is a VALID empty prediction, not an error."""

    answer_id: str
    error: str | None = None
    claims: list[ExtractedClaim] = []

    @model_validator(mode="after")
    def _no_claims_on_error(self) -> "AnswerPrediction":
        if self.error is not None and self.claims:
            raise ValueError("errored answer must not carry claims")
        return self


class ScoreResult(BaseModel):
    G: int
    P: int
    M: int
    C: int
    extraction: PRF
    joint: PRF
    per_label: dict[str, PRF]
    macro_f1: float
    gold_coverage: float  # M / G
    missed_gold_ids: list[str]
    unmatched_prediction_ids: list[str]
    wrong_label_ids: list[str]  # span matched, label differs (gold claim ids)
    answers_evaluated: int
    answer_errors_total: int
    answer_errors_by_code: dict[str, int]
    valid_empty_answers: int
    confusion: dict[str, dict[str, int]]


def save_predictions(path: Path, predictions: list[AnswerPrediction]) -> None:
    payload = {"predictions": [p.model_dump(mode="json") for p in predictions]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_predictions(path: Path) -> list[AnswerPrediction]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [AnswerPrediction.model_validate(p) for p in data["predictions"]]


def score(
    predictions: list[AnswerPrediction],
    gold: list[GoldClaim],
    evaluated_answer_ids: set[str] | None = None,
) -> ScoreResult:
    """Deterministic. A prediction matches a gold claim only if answer_id, start_char, end_char are
    identical AND the quotes are identical. No fuzzy matching, no span edits, no LLM judge.

    Evaluated answers default to those that have gold claims. An evaluated answer with no saved prediction
    is treated as an answer error ('missing_result'). Predictions for non-evaluated answers raise."""
    evaluated = set(evaluated_answer_ids) if evaluated_answer_ids is not None else {g.answer_id for g in gold}
    if any(g.answer_id not in evaluated for g in gold):
        raise ValueError("gold contains answers outside the evaluated set")

    keys = [(g.answer_id, g.start_char, g.end_char) for g in gold]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate gold spans")
    gold_by_key = {k: g for k, g in zip(keys, gold)}

    by_answer: dict[str, AnswerPrediction] = {}
    for p in predictions:
        if p.answer_id in by_answer:
            raise ValueError(f"duplicate prediction for {p.answer_id}")
        if p.answer_id not in evaluated:
            raise ValueError(f"prediction for non-evaluated answer {p.answer_id}")
        by_answer[p.answer_id] = p

    confusion = empty_confusion()
    matched_gold: set[tuple[str, int, int]] = set()
    unmatched_pred_ids: list[str] = []
    wrong_label_ids: list[str] = []
    P = M = C = 0
    error_codes: Counter[str] = Counter()
    valid_empty = 0

    for aid in sorted(evaluated):
        pred = by_answer.get(aid)
        if pred is None:
            error_codes["missing_result"] += 1
            continue
        if pred.error is not None:
            error_codes[pred.error] += 1
            continue
        if not pred.claims:
            valid_empty += 1
        for c in pred.claims:
            P += 1
            key = (aid, c.start_char, c.end_char)
            g = gold_by_key.get(key)
            if g is None or key in matched_gold or g.quote != c.quote:
                unmatched_pred_ids.append(c.claim_id)
                confusion["no_gold"][c.label.value] += 1
                continue
            matched_gold.add(key)
            M += 1
            confusion[g.expected_label.value][c.label.value] += 1
            if c.label == g.expected_label:
                C += 1
            else:
                wrong_label_ids.append(g.claim_id)

    missed: list[str] = []
    for k, g in zip(keys, gold):
        if k not in matched_gold:
            missed.append(g.claim_id)
            confusion[g.expected_label.value]["missing_or_error"] += 1

    G = len(gold)
    per_label = label_metrics(confusion)
    return ScoreResult(
        G=G, P=P, M=M, C=C,
        extraction=extraction_metrics(G, P, M),
        joint=joint_metrics(G, P, C),
        per_label=per_label,
        macro_f1=macro_f1(per_label),
        gold_coverage=M / G if G else 0.0,
        missed_gold_ids=missed,
        unmatched_prediction_ids=unmatched_pred_ids,
        wrong_label_ids=wrong_label_ids,
        answers_evaluated=len(evaluated),
        answer_errors_total=sum(error_codes.values()),
        answer_errors_by_code=dict(error_codes),
        valid_empty_answers=valid_empty,
        confusion=confusion,
    )


def evaluate_saved_run(predictions_path: Path, gold: list[GoldClaim]) -> ScoreResult:
    """Entry point for evaluation: predictions must already be on disk; gold is joined only here."""
    return score(load_predictions(predictions_path), gold)
