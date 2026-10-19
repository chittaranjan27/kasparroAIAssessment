"""Plain-text / JSON rendering of a ScoreResult. Pure formatting; no recomputation."""
from __future__ import annotations

from pathlib import Path

from app.evaluation.metrics import COLS, LABELS, ROWS
from app.evaluation.scorer import ScoreResult


def render_text(r: ScoreResult) -> str:
    L = []
    L.append(f"Answers evaluated: {r.answers_evaluated}  errors: {r.answer_errors_total} {r.answer_errors_by_code}  valid-empty: {r.valid_empty_answers}")
    L.append(f"G={r.G} P={r.P} M={r.M} C={r.C}  gold_coverage={r.gold_coverage:.4f}")
    L.append(f"Extraction  P={r.extraction.precision:.4f} R={r.extraction.recall:.4f} F1={r.extraction.f1:.4f}")
    L.append(f"Joint       P={r.joint.precision:.4f} R={r.joint.recall:.4f} F1={r.joint.f1:.4f}")
    for lab in LABELS:
        m = r.per_label[lab]
        L.append(f"{lab:<13} P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")
    L.append(f"Macro F1: {r.macro_f1:.4f}")
    L.append("Confusion (rows=gold, cols=predicted):")
    L.append(f"{'':<13}" + "".join(f"{c:>18}" for c in COLS))
    for row in ROWS:
        L.append(f"{row:<13}" + "".join(f"{r.confusion[row][c]:>18}" for c in COLS))
    L.append(f"Missed gold ({len(r.missed_gold_ids)}): {', '.join(r.missed_gold_ids)}")
    L.append(f"Unmatched predictions ({len(r.unmatched_prediction_ids)}): {', '.join(r.unmatched_prediction_ids)}")
    L.append(f"Wrong-label ({len(r.wrong_label_ids)}): {', '.join(r.wrong_label_ids)}")
    return "\n".join(L)


def write_report(r: ScoreResult, directory: Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    txt, js = directory / "report.txt", directory / "report.json"
    txt.write_text(render_text(r), encoding="utf-8")
    js.write_text(r.model_dump_json(indent=2), encoding="utf-8")
    return txt, js


def build_records(result: ScoreResult, gold: list, predictions: list) -> dict:
    """Detail records for the IDs already classified by the scorer (no re-matching). Stable order:
    answer_id, then source offsets."""
    gold_by_id = {g.claim_id: g for g in gold}
    pred_by_id = {c.claim_id: (p.answer_id, c) for p in predictions for c in p.claims}
    pred_by_span = {(p.answer_id, c.start_char, c.end_char): c for p in predictions for c in p.claims}

    def gkey(g):
        return (g.answer_id, g.start_char, g.end_char)

    missed = sorted((gold_by_id[i] for i in result.missed_gold_ids), key=gkey)
    wrong = sorted((gold_by_id[i] for i in result.wrong_label_ids), key=gkey)
    extra = sorted((pred_by_id[i] for i in result.unmatched_prediction_ids), key=lambda t: (t[0], t[1].start_char, t[1].end_char))
    return {
        "missing_claims": [
            {"claim_id": g.claim_id, "answer_id": g.answer_id, "start_char": g.start_char, "end_char": g.end_char,
             "quote": g.quote, "expected_label": g.expected_label.value}
            for g in missed
        ],
        "extra_predictions": [
            {"claim_id": c.claim_id, "answer_id": a, "start_char": c.start_char, "end_char": c.end_char,
             "quote": c.quote, "predicted_label": c.label.value}
            for a, c in extra
        ],
        "wrong_labels": [
            {"gold_claim_id": g.claim_id, "prediction_claim_id": pred_by_span[gkey(g)].claim_id,
             "answer_id": g.answer_id, "start_char": g.start_char, "end_char": g.end_char, "quote": g.quote,
             "expected_label": g.expected_label.value, "predicted_label": pred_by_span[gkey(g)].label.value}
            for g in wrong
        ],
        "answer_errors": {
            "total": result.answer_errors_total,
            "by_code": dict(sorted(result.answer_errors_by_code.items())),
        },
    }
