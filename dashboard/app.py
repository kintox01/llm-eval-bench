from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from evalbench import scoring, storage
from evalbench.schemas import FIELDS

SERIES_COLOR = "#2a78d6"

st.set_page_config(page_title="LLM Eval Bench", layout="wide")
st.title("LLM Eval Bench")

db_path = Path(st.sidebar.text_input("Results file", str(storage.DEFAULT_DB)))
if not db_path.exists():
    st.info("No results yet. Run `uv run evalbench run configs/tickets.yaml` first.")
    st.stop()

conn = storage.connect(db_path)
all_runs = storage.list_runs(conn)
latest_ids = [r.id for r in storage.latest_run_per_name(conn)]
selected_ids = st.sidebar.multiselect(
    "Runs", [r.id for r in all_runs], default=latest_ids, help="Default: latest run of each name."
)
if not selected_ids:
    st.stop()

runs = {r.id: r for r in all_runs if r.id in selected_ids}
records = {rid: storage.load_records(conn, rid) for rid in selected_ids}
metrics = {rid: scoring.compute_metrics(records[rid]) for rid in selected_ids}

overview = pd.DataFrame(
    [
        {
            "run": runs[rid].name,
            "model": runs[rid].model,
            **{f: m.accuracy[f] for f in FIELDS},
            "exact_match": m.exact_match,
            "invalid": m.invalid_rate,
            "api_errors": m.api_error_rate,
            "p50 (s)": m.latency_p50,
            "p95 (s)": m.latency_p95,
            "$ / 1k tickets": m.cost_per_1k,
            "ECE": m.ece,
            "run_id": rid,
        }
        for rid, m in metrics.items()
    ]
).sort_values("exact_match", ascending=False)

st.subheader("Overview")
percent_cols = [*FIELDS, "exact_match", "invalid", "api_errors"]
percent = st.column_config.NumberColumn(format="percent")
st.dataframe(
    overview,
    hide_index=True,
    width="stretch",
    column_config={
        **{c: percent for c in percent_cols},
        "p50 (s)": st.column_config.NumberColumn(format="%.2f"),
        "p95 (s)": st.column_config.NumberColumn(format="%.2f"),
        "$ / 1k tickets": st.column_config.NumberColumn(format="%.3f"),
        "ECE": st.column_config.NumberColumn(format="%.3f"),
    },
)

left, right = st.columns(2)
with left:
    st.markdown("**Exact match** (all three fields right)")
    fig = px.bar(overview, x="exact_match", y="run", orientation="h", text_auto=".0%")
    fig.update_traces(marker_color=SERIES_COLOR)
    fig.update_layout(
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        yaxis={"title": None, "categoryorder": "total ascending"},
        xaxis_title=None,
        height=120 + 50 * len(overview),
    )
    st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("**Cost vs quality**: top left is best")
    priced = overview.dropna(subset=["$ / 1k tickets"])
    if priced.empty:
        st.caption("No cost data: add the models to src/evalbench/pricing.py.")
    else:
        fig = px.scatter(priced, x="$ / 1k tickets", y="exact_match", text="run", log_x=True)
        fig.update_traces(
            marker={"color": SERIES_COLOR, "size": 12}, textposition="top center", cliponaxis=False
        )
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")

st.subheader("Run detail")
detail_id = st.selectbox("Run", selected_ids, format_func=lambda rid: runs[rid].name)
detail = records[detail_id]

left, right = st.columns(2)
with left:
    field = st.radio("Confusion matrix for", ["category", "urgency"], horizontal=True)
    counts = scoring.confusion(detail, field)
    labels = sorted({e for e, _ in counts} | {p for _, p in counts})
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for (expected, predicted), n in counts.items():
        matrix.loc[expected, predicted] = n
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(xaxis_title="predicted", yaxis_title="expected", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

with right:
    confident = [r for r in detail if r.prediction.confidence and r.prediction.output]
    if confident:
        st.markdown("**Calibration** of the category confidence")
        calib = pd.DataFrame(
            {
                "confidence": [r.prediction.confidence["category"] for r in confident],
                "correct": [scoring.is_correct(r, "category") for r in confident],
            }
        )
        calib["bin"] = pd.cut(calib["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        by_bin = calib.groupby("bin", observed=True).agg(
            confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size")
        )
        fig = px.scatter(by_bin, x="confidence", y="accuracy", size="n", hover_data=["n"])
        fig.update_traces(marker_color=SERIES_COLOR)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": "gray"})
        fig.update_layout(xaxis_range=[0, 1.02], yaxis_range=[0, 1.02], xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("On the dotted line, announced confidence matches observed accuracy.")
    else:
        st.caption("This model reports no confidence: calibration is only available for Jev.")

st.markdown("**Mistakes**")
mistakes = pd.DataFrame(
    [
        {
            "case": r.case.id,
            "ticket": r.case.text,
            **{f"expected {f}": getattr(r.case.expected, f) for f in FIELDS},
            **{
                f"predicted {f}": getattr(r.prediction.output, f) if r.prediction.output else scoring.NO_ANSWER
                for f in FIELDS
            },
            "error": r.prediction.error,
        }
        for r in detail
        if not scoring.is_exact_match(r)
    ]
)
if mistakes.empty:
    st.success("No mistakes on this run.")
else:
    st.dataframe(mistakes, hide_index=True, width="stretch")
