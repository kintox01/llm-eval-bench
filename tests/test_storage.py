from evalbench import storage
from evalbench.config import RunConfig
from evalbench.runners.base import Prediction
from evalbench.schemas import EvalCase, TicketLabel


def test_save_and_load_roundtrip(tmp_path):
    conn = storage.connect(tmp_path / "bench.sqlite")
    label = TicketLabel(category="bug", urgency="low", refund_requested=False)
    cases = [EvalCase(id="t01", text="bug", expected=label), EvalCase(id="t02", text="?", expected=label)]
    predictions = [
        Prediction(case_id="t01", output=label, latency_s=0.3, input_tokens=10, output_tokens=5,
                   cost_usd=0.0001, confidence={"category": 0.9}, extra={"model_version": "jev-x"}),
        Prediction(case_id="t02", output=None, error_kind="api_error", error="boom", latency_s=1.0),
    ]
    run_id = storage.save_run(conn, RunConfig(name="jev", model="typesafe:jev-latest"), "data.jsonl", cases, predictions)

    assert storage.get_run(conn, run_id).n_cases == 2
    assert [r.name for r in storage.latest_run_per_name(conn)] == ["jev"]
    loaded = storage.load_records(conn, run_id)
    assert [r.prediction for r in loaded] == predictions
    assert loaded[0].case.expected == label
