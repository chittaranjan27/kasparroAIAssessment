from pathlib import Path

import pytest

from app.agent.extractor import ExtractedClaim
from app.evaluation.metrics import extraction_metrics, joint_metrics, label_metrics, empty_confusion, macro_f1, prf
from app.evaluation.report import render_text, write_report
from app.evaluation.scorer import AnswerPrediction, load_predictions, save_predictions, score
from app.schemas import GoldClaim, Label


def gold(aid, n, start, label="supported", quote=None):
    return GoldClaim(
        answer_id=aid, claim_id=f"G-{aid}-{n}", start_char=start, end_char=start + 5,
        quote=quote or f"q{n:04d}"[:5], expected_label=label, evidence_fact_ids=[], explanation="", difficulty_tags=[],
    )


def pred(aid, n, start, label="supported", quote=None):
    return ExtractedClaim(
        claim_id=f"P-{aid}-{n}", quote=quote or f"q{n:04d}"[:5], start_char=start, end_char=start + 5,
        label=Label(label), evidence_fact_ids=[],
    )


def ap(aid, claims=(), error=None):
    return AnswerPrediction(answer_id=aid, claims=list(claims), error=error)


# ---- required arithmetic: G=10, P=10, M=8, C=7 ----
def test_arithmetic_G10_P10_M8_C7():
    g = [gold("A01", i, i * 10) for i in range(10)]
    claims = [pred("A01", i, i * 10) for i in range(7)]  # 7 correct
    claims.append(pred("A01", 7, 70, label="unknown"))  # matched, wrong label -> M=8, C=7
    claims += [pred("A01", 8, 1000), pred("A01", 9, 1010)]  # 2 extras -> P=10
    r = score([ap("A01", claims)], g)
    assert (r.G, r.P, r.M, r.C) == (10, 10, 8, 7)
    assert r.extraction.precision == pytest.approx(0.8) and r.extraction.recall == pytest.approx(0.8)
    assert r.extraction.f1 == pytest.approx(0.80)
    assert r.joint.f1 == pytest.approx(0.70)
    assert r.gold_coverage == pytest.approx(0.8)
    assert r.wrong_label_ids == ["G-A01-7"]
    assert r.missed_gold_ids == ["G-A01-8", "G-A01-9"]
    assert r.unmatched_prediction_ids == ["P-A01-8", "P-A01-9"]


def test_formulas_direct():
    assert extraction_metrics(10, 10, 8).f1 == pytest.approx(0.8)
    assert joint_metrics(10, 10, 7).f1 == pytest.approx(0.7)
    z = prf(0, 0, 0)
    assert (z.precision, z.recall, z.f1) == (0.0, 0.0, 0.0)


# ---- matching rules ----
def test_exact_span_only_no_fuzzy():
    g = [gold("A01", 1, 10)]
    for start in (9, 11):  # off by one either side
        r = score([ap("A01", [pred("A01", 1, start)])], g)
        assert (r.M, r.C) == (0, 0) and r.missed_gold_ids == ["G-A01-1"] and r.unmatched_prediction_ids == ["P-A01-1"]


def test_answer_id_must_match():
    g = [gold("A01", 1, 10), gold("A02", 1, 99)]
    r = score([ap("A01"), ap("A02", [pred("A02", 1, 10)])], g)  # A02 span belongs to A01's coordinates
    assert r.M == 0


def test_quote_mismatch_same_span_not_matched():
    g = [gold("A01", 1, 10, quote="hello")]
    r = score([ap("A01", [pred("A01", 1, 10, quote="HELLO")])], g)
    assert r.M == 0 and r.unmatched_prediction_ids == ["P-A01-1"]


def test_duplicate_prediction_of_same_span_counts_once():
    g = [gold("A01", 1, 10)]
    r = score([ap("A01", [pred("A01", 1, 10), pred("A01", 2, 10)])], g)
    assert (r.P, r.M, r.C) == (2, 1, 1) and r.unmatched_prediction_ids == ["P-A01-2"]


# ---- errors / empty / extra / wrong ----
def test_answer_error_counts_gold_as_missed_not_predictions():
    g = [gold("A01", 1, 0, "contradicted"), gold("A01", 2, 10, "unknown"), gold("A02", 1, 0)]
    r = score([ap("A01", error="provider_error"), ap("A02", [pred("A02", 1, 0)])], g)
    assert r.answer_errors_total == 1 and r.answer_errors_by_code == {"provider_error": 1}
    assert (r.G, r.P, r.M, r.C) == (3, 1, 1, 1)
    assert r.missed_gold_ids == ["G-A01-1", "G-A01-2"]
    assert r.confusion["contradicted"]["missing_or_error"] == 1
    assert r.confusion["unknown"]["missing_or_error"] == 1


def test_valid_empty_prediction_is_not_an_error():
    g = [gold("A01", 1, 0)]
    r = score([ap("A01", [])], g)
    assert r.answer_errors_total == 0 and r.valid_empty_answers == 1
    assert r.P == 0 and r.M == 0 and r.missed_gold_ids == ["G-A01-1"]
    assert r.extraction.precision == 0.0 and r.extraction.f1 == 0.0


def test_missing_prediction_for_evaluated_answer_is_error():
    r = score([], [gold("A01", 1, 0)])
    assert r.answer_errors_by_code == {"missing_result": 1} and r.M == 0


def test_errored_answer_with_claims_is_rejected():
    with pytest.raises(ValueError):
        ap("A01", [pred("A01", 1, 0)], error="invalid_output")


def test_prediction_for_non_evaluated_answer_raises():
    with pytest.raises(ValueError):
        score([ap("A41")], [gold("A01", 1, 0)])


def test_duplicate_predictions_for_answer_raise():
    with pytest.raises(ValueError):
        score([ap("A01"), ap("A01")], [gold("A01", 1, 0)])


# ---- confusion matrix and per-label metrics ----
def build_case():
    g = [
        gold("A01", 1, 0, "supported"), gold("A01", 2, 10, "supported"),
        gold("A01", 3, 20, "contradicted"), gold("A01", 4, 30, "unknown"),
        gold("A01", 5, 40, "unknown"), gold("A01", 6, 50, "contradicted"),
    ]
    p = [
        pred("A01", 1, 0, "supported"),      # S->S
        pred("A01", 2, 10, "contradicted"),  # S->C
        pred("A01", 3, 20, "contradicted"),  # C->C
        pred("A01", 4, 30, "supported"),     # U->S
        pred("A01", 7, 200, "unknown"),      # no_gold -> unknown
        pred("A01", 8, 300, "supported"),    # no_gold -> supported
    ]  # gold 5 (unknown) and 6 (contradicted) missed
    return score([ap("A01", p)], g)


def test_confusion_matrix_exact():
    r = build_case()
    cm = r.confusion
    assert cm["supported"] == {"supported": 1, "contradicted": 1, "unknown": 0, "missing_or_error": 0}
    assert cm["contradicted"] == {"supported": 0, "contradicted": 1, "unknown": 0, "missing_or_error": 1}
    assert cm["unknown"] == {"supported": 1, "contradicted": 0, "unknown": 0, "missing_or_error": 1}
    assert cm["no_gold"] == {"supported": 1, "contradicted": 0, "unknown": 1, "missing_or_error": 0}
    assert list(cm) == ["supported", "contradicted", "unknown", "no_gold"]
    assert (r.G, r.P, r.M, r.C) == (6, 6, 4, 2)


def test_per_label_and_macro_f1():
    r = build_case()
    s, c, u = (r.per_label[k] for k in ("supported", "contradicted", "unknown"))
    # supported: TP1, predicted col = 1+0+1+1 = 3, gold row = 2
    assert (s.precision, s.recall) == (pytest.approx(1 / 3), pytest.approx(1 / 2))
    assert s.f1 == pytest.approx(2 * 1 / (3 + 2))
    # contradicted: TP1, predicted col = 1+1(S->C) = 2, gold row = 2
    assert (c.precision, c.recall, c.f1) == (pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.5))
    # unknown: TP0, predicted col = 1 (no_gold), gold = 2
    assert (u.precision, u.recall, u.f1) == (0.0, 0.0, 0.0)
    assert r.macro_f1 == pytest.approx((s.f1 + c.f1 + u.f1) / 3)


def test_label_metrics_zero_division_and_macro_unweighted():
    cm = empty_confusion()
    pl = label_metrics(cm)
    assert all(v.f1 == 0.0 for v in pl.values()) and macro_f1(pl) == 0.0


def test_diagonal_total_equals_C_and_matrix_totals_consistent():
    r = build_case()
    diag = sum(r.confusion[k][k] for k in ("supported", "contradicted", "unknown"))
    assert diag == r.C
    assert sum(sum(row.values()) for row in r.confusion.values()) == r.M + len(r.missed_gold_ids) + len(r.unmatched_prediction_ids)


# ---- determinism, persistence, isolation ----
def test_deterministic_and_roundtrip(tmp_path):
    g = [gold("A01", 1, 0), gold("A02", 1, 0, "unknown")]
    preds = [ap("A02", [pred("A02", 1, 0, "unknown")]), ap("A01", [pred("A01", 1, 0)])]
    a = score(preds, g)
    b = score(list(reversed(preds)), g)
    assert a == b
    f = tmp_path / "p.json"
    save_predictions(f, preds)
    assert score(load_predictions(f), g) == a
    assert "gold" not in f.read_text(encoding="utf-8").lower()


def test_report_renders_and_writes(tmp_path):
    r = build_case()
    text = render_text(r)
    assert "G=6 P=6 M=4 C=2" in text and "no_gold" in text and "missing_or_error" in text
    txt, js = write_report(r, tmp_path)
    assert txt.exists() and js.exists()


def test_agent_package_never_touches_gold_or_scoring():
    import ast

    root = Path(__file__).resolve().parent.parent / "app" / "agent"
    banned_modules = ("app.evaluation",)
    banned_names = {"load_gold", "score", "evaluate_saved_run"}
    for py in root.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(banned_modules), f"{py.name} imports {node.module}"
                assert not banned_names & {a.name for a in node.names}, f"{py.name} imports scoring/gold names"
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith(banned_modules) for a in node.names), f"{py.name} imports evaluation"
