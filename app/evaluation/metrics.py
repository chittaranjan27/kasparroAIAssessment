"""Pure metric formulas. No I/O, no gold access, fully deterministic."""
from __future__ import annotations

from pydantic import BaseModel

LABELS = ("supported", "contradicted", "unknown")
ROWS = LABELS + ("no_gold",)
COLS = LABELS + ("missing_or_error",)


class PRF(BaseModel):
    precision: float
    recall: float
    f1: float


def prf(tp: int, predicted: int, gold: int) -> PRF:
    """Zero denominators give 0.0. F1 = 2*tp/(predicted+gold), identical to 2PR/(P+R) but exact."""
    return PRF(
        precision=tp / predicted if predicted else 0.0,
        recall=tp / gold if gold else 0.0,
        f1=2 * tp / (predicted + gold) if (predicted + gold) else 0.0,
    )


def extraction_metrics(g: int, p: int, m: int) -> PRF:
    """G gold claims, P predicted claims, M span-matched claims."""
    return prf(m, p, g)


def joint_metrics(g: int, p: int, c: int) -> PRF:
    """C = span-matched AND label-correct claims."""
    return prf(c, p, g)


def empty_confusion() -> dict[str, dict[str, int]]:
    return {r: {c: 0 for c in COLS} for r in ROWS}


def label_metrics(confusion: dict[str, dict[str, int]]) -> dict[str, PRF]:
    """Per label: TP = diagonal; predicted = column sum (incl. no_gold row); gold = row sum (incl. missing)."""
    out: dict[str, PRF] = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        predicted = sum(confusion[r][lab] for r in ROWS)
        gold = sum(confusion[lab][c] for c in COLS)
        out[lab] = prf(tp, predicted, gold)
    return out


def macro_f1(per_label: dict[str, PRF]) -> float:
    """Unweighted mean of the three label F1 scores."""
    return sum(per_label[lab].f1 for lab in LABELS) / len(LABELS)
