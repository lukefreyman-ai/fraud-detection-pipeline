"""Analyst review dashboard: flagged test transactions, why the model flagged each one (SHAP), and a decision log.

Run after `python run.py`:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudpipe.explain import explain_one  # noqa: E402

RESULTS, MODELS = ROOT / "results", ROOT / "models"
DECISIONS = RESULTS / "review_decisions.csv"

st.set_page_config(page_title="Fraud review queue", layout="wide")


@st.cache_resource
def load():
    scored = pd.read_parquet(RESULTS / "scored_test.parquet")
    meta = json.loads((RESULTS / "metrics.json").read_text())
    model = joblib.load(MODELS / "best_model.pkl")
    return scored, meta, model


def log_decision(row_id: int, decision: str, note: str) -> None:
    rec = pd.DataFrame([{"row_id": row_id, "decision": decision, "note": note, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}])
    rec.to_csv(DECISIONS, mode="a", header=not DECISIONS.exists(), index=False)


try:
    scored, meta, model = load()
except FileNotFoundError:
    st.error("No results found. Run `python run.py` first."); st.stop()

thr = float(next(r["threshold"] for r in meta["results"] if f"{r['model']}/{r['strategy']}" == meta["best"]) if "lightgbm" in meta["best"]
            else meta["best_threshold"])
decisions = pd.read_csv(DECISIONS) if DECISIONS.exists() else pd.DataFrame(columns=["row_id", "decision", "note", "at"])

st.title("Fraud review queue")
c1, c2, c3, c4 = st.columns(4)
flagged = scored[scored["flagged"] == 1].sort_values("score", ascending=False)
c1.metric("Test transactions", f"{len(scored):,}")
c2.metric("Flagged for review", f"{len(flagged):,}", f"{1e4*len(flagged)/len(scored):.1f} per 10k")
c3.metric("Confirmed fraud among flagged", f"{int(flagged.Class.sum()):,}", f"precision {flagged.Class.mean():.0%}")
c4.metric("Reviewed so far", f"{decisions.row_id.nunique():,}")
st.caption(f"Model: {meta['best']} · threshold {thr:.4f} · false-decline cost ${meta['fp_cost']:.0f} · missed fraud costs the transaction amount")

with st.sidebar:
    st.header("Filters")
    min_score = st.slider("Minimum score", 0.0, 1.0, float(round(thr, 3)), 0.005)
    min_amount = st.number_input("Minimum amount ($)", 0.0, float(scored.Amount.max()), 0.0, 10.0)
    hide_reviewed = st.checkbox("Hide already-reviewed", True)
    show_truth = st.checkbox("Show ground truth (Class)", False, help="Off by default: an analyst would not see this.")

queue = scored[(scored.score >= min_score) & (scored.Amount >= min_amount)].sort_values("score", ascending=False)
if hide_reviewed and len(decisions):
    queue = queue[~queue.index.isin(decisions.row_id)]
st.subheader(f"Queue: {len(queue):,} transactions")
show_cols = ["score", "Amount", "hour_of_day", "vel_1h_count", "vel_24h_count", "amount_z_24h", "secs_since_prev"] + (["Class"] if show_truth else [])
st.dataframe(queue[show_cols].head(300).style.format({"score": "{:.4f}", "Amount": "${:,.2f}", "amount_z_24h": "{:.2f}", "secs_since_prev": "{:.0f}"}),
             use_container_width=True, height=320)

st.subheader("Review a transaction")
if len(queue) == 0:
    st.info("Nothing in the queue with these filters."); st.stop()
row_id = st.selectbox("Transaction (row id in the test set)", queue.index[:300].tolist(), format_func=lambda i: f"#{i}  score {queue.loc[i,'score']:.4f}  ${queue.loc[i,'Amount']:,.2f}")
row = scored.loc[[row_id]]
left, right = st.columns([1, 1])
with left:
    st.markdown("**Transaction**")
    st.json({"row_id": int(row_id), "score": round(float(row.score.iloc[0]), 4), "amount": float(row.Amount.iloc[0]),
             "hour_of_day": int(row.hour_of_day.iloc[0]), "txns_last_1h": int(row.vel_1h_count.iloc[0]), "txns_last_24h": int(row.vel_24h_count.iloc[0]),
             "amount_z_vs_24h": round(float(row.amount_z_24h.iloc[0]), 2), "secs_since_prev": float(row.secs_since_prev.iloc[0])}
            | ({"ground_truth": int(row.Class.iloc[0])} if show_truth else {}))
with right:
    st.markdown("**Why the model flagged it (SHAP, top 10)**")
    contrib = explain_one(model.estimator, row[model.features]).head(10)
    st.bar_chart(contrib.set_index("feature")["shap"])
    st.dataframe(contrib.style.format({"value": "{:.3f}", "shap": "{:+.3f}"}), use_container_width=True, height=250)

note = st.text_input("Note (optional)")
b1, b2, b3 = st.columns(3)
if b1.button("Confirm fraud", type="primary"):
    log_decision(int(row_id), "fraud", note); st.success("Logged: fraud"); st.rerun()
if b2.button("Legitimate"):
    log_decision(int(row_id), "legit", note); st.success("Logged: legit"); st.rerun()
if b3.button("Needs more info"):
    log_decision(int(row_id), "hold", note); st.success("Logged: hold"); st.rerun()
if len(decisions):
    st.subheader("Decision log")
    st.dataframe(decisions.tail(50), use_container_width=True)
