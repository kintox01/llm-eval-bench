import math
from collections import Counter
from dataclasses import dataclass

from evalbench.schemas import FIELDS
from evalbench.storage import Record

NO_ANSWER = "none"


@dataclass
class RunMetrics:
    n: int
    accuracy: dict[str, float]
    exact_match: float
    invalid_rate: float
    api_error_rate: float
    latency_p50: float
    latency_p95: float
    total_cost: float | None
    cost_per_1k: float | None
    ece: float | None


def is_correct(record: Record, field: str) -> bool:
    output = record.prediction.output
    return output is not None and getattr(output, field) == getattr(record.case.expected, field)


def is_exact_match(record: Record) -> bool:
    return all(is_correct(record, field) for field in FIELDS)


def compute_metrics(records: list[Record]) -> RunMetrics:
    n = len(records)
    if n == 0:
        raise ValueError("no records")
    kinds = Counter(r.prediction.error_kind for r in records)
    latencies = [r.prediction.latency_s for r in records]
    # None if a model is missing from pricing.py
    answered_costs = [r.prediction.cost_usd for r in records if r.prediction.output]
    total_cost = None if not answered_costs or None in answered_costs else sum(answered_costs)

    confident = [r for r in records if r.prediction.confidence and r.prediction.output]
    ece = None
    if confident:
        ece = expected_calibration_error(
            [r.prediction.confidence["category"] for r in confident],
            [is_correct(r, "category") for r in confident],
        )

    return RunMetrics(
        n=n,
        accuracy={field: sum(is_correct(r, field) for r in records) / n for field in FIELDS},
        exact_match=sum(is_exact_match(r) for r in records) / n,
        invalid_rate=kinds["invalid_output"] / n,
        api_error_rate=kinds["api_error"] / n,
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        total_cost=total_cost,
        cost_per_1k=None if total_cost is None else total_cost / n * 1000,
        ece=ece,
    )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def expected_calibration_error(
    confidences: list[float], corrects: list[bool], n_bins: int = 5
) -> float:
    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        low, high = i / n_bins, (i + 1) / n_bins
        in_bin = [
            (conf, ok)
            for conf, ok in zip(confidences, corrects, strict=True)
            if low < conf <= high or (i == 0 and conf == 0)
        ]
        if not in_bin:
            continue
        avg_conf = sum(conf for conf, _ in in_bin) / len(in_bin)
        accuracy = sum(ok for _, ok in in_bin) / len(in_bin)
        ece += len(in_bin) / total * abs(avg_conf - accuracy)
    return ece


def confusion(records: list[Record], field: str) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for r in records:
        predicted = NO_ANSWER if r.prediction.output is None else str(getattr(r.prediction.output, field))
        counts[(str(getattr(r.case.expected, field)), predicted)] += 1
    return counts


def diff_runs(baseline: list[Record], candidate: list[Record]) -> dict[str, list[str]]:
    before = {r.case.id: is_exact_match(r) for r in baseline}
    after = {r.case.id: is_exact_match(r) for r in candidate}
    shared = sorted(before.keys() & after.keys())
    return {
        "fixed": [cid for cid in shared if not before[cid] and after[cid]],
        "regressed": [cid for cid in shared if before[cid] and not after[cid]],
    }
