import pytest

from evalbench.runners.base import Prediction
from evalbench.schemas import EvalCase, TicketLabel
from evalbench.scoring import (
    compute_metrics,
    confusion,
    diff_runs,
    expected_calibration_error,
    percentile,
)
from evalbench.storage import Record

BILLING = TicketLabel(category="billing", urgency="high", refund_requested=True)
BUG = TicketLabel(category="bug", urgency="low", refund_requested=False)


def record(case_id, expected, output, cost=0.001, confidence=None, error_kind=None):
    return Record(
        case=EvalCase(id=case_id, text="...", expected=expected),
        prediction=Prediction(
            case_id=case_id,
            output=output,
            error_kind=error_kind,
            latency_s=1.0,
            cost_usd=cost if output else None,
            confidence=confidence,
        ),
    )


def test_metrics_count_failures_as_wrong():
    records = [
        record("a", BILLING, BILLING),
        record("b", BILLING, BILLING.model_copy(update={"urgency": "low"})),
        record("c", BUG, None, error_kind="invalid_output"),
        record("d", BUG, None, error_kind="api_error"),
    ]
    m = compute_metrics(records)
    assert m.accuracy == {"category": 0.5, "urgency": 0.25, "refund_requested": 0.5}
    assert m.exact_match == 0.25
    assert m.invalid_rate == 0.25
    assert m.api_error_rate == 0.25
    assert m.total_cost == pytest.approx(0.002)
    assert m.cost_per_1k == pytest.approx(0.5)
    assert m.ece is None


def test_unknown_price_gives_unknown_cost():
    m = compute_metrics([record("a", BUG, BUG, cost=None)])
    assert m.total_cost is None and m.cost_per_1k is None


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5


def test_calibration_error():
    assert expected_calibration_error([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0)
    assert expected_calibration_error([0.9] * 10, [False] * 10) == pytest.approx(0.9)


def test_ece_is_computed_when_confidence_exists():
    records = [record("a", BUG, BUG, confidence={"category": 0.8, "urgency": 0.5, "refund_requested": 0.9})]
    assert compute_metrics(records).ece == pytest.approx(0.2)


def test_confusion_marks_missing_answers():
    counts = confusion([record("a", BUG, BILLING), record("b", BUG, None)], "category")
    assert counts == {("bug", "billing"): 1, ("bug", "none"): 1}


def test_diff_runs():
    baseline = [record("a", BUG, BUG), record("b", BUG, BILLING)]
    candidate = [record("a", BUG, BILLING), record("b", BUG, BUG)]
    assert diff_runs(baseline, candidate) == {"fixed": ["b"], "regressed": ["a"]}
