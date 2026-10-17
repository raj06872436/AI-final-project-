"""Streamlit dashboard for the fraud detection project.

    streamlit run app.py

Needs the outputs of `python run_experiments.py` (results/ and models/).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fraud import config as C, evaluate  # noqa: E402

st.set_page_config(page_title="Temporal Fraud Detection", layout="wide")
R, F = C.RESULTS_DIR, C.FIG_DIR


@st.cache_data
def table(name):
    return pd.read_csv(R / name)


@st.cache_resource
def load_model():
    bundle = joblib.load(C.MODEL_DIR / "xgb_temporal.joblib")
    return bundle, shap.TreeExplainer(bundle["model"])


if not (R / "summary.json").exists():
    st.error("No results yet. Run `python run_experiments.py` first.")
    st.stop()

summary = json.loads((R / "summary.json").read_text())
st.title("Temporal and Explainable ML for Early Financial Fraud Detection")
st.caption("Leakage-free comparison of transaction-level, temporal and sequence models on the ULB credit card dataset")

tab_data, tab_results, tab_live, tab_method = st.tabs(["Dataset", "Results", "Live scoring", "Method"])

with tab_data:
    a = summary["audit_raw"]; c = summary["audit_clean"]
    cols = st.columns(5)
    cols[0].metric("Transactions", f"{a['rows']:,}")
    cols[1].metric("Fraud", f"{a['fraud']}")
    cols[2].metric("Fraud rate", f"{a['fraud_rate']:.3%}")
    cols[3].metric("Duplicates removed", f"{a['duplicate_rows']:,}")
    cols[4].metric("'All legit' accuracy", f"{a['trivial_all_legit_accuracy']:.2%}", "0 fraud caught", delta_color="inverse")
    st.subheader("Chronological split")
    st.dataframe(table("splits.csv"), hide_index=True)
    c1, c2 = st.columns(2)
    c1.image(str(F / "class_balance.png")); c2.image(str(F / "fraud_over_time.png"))

with tab_results:
    st.subheader("All models (ranked by validation PR-AUC; test metrics at the validation-tuned threshold)")
    st.dataframe(table("all_models.csv").style.format(precision=4), hide_index=True)
    st.success(f"Final model selected on validation data: **{summary['final_model']}**")
    st.image(str(F / "pr_roc_curves.png"))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("H1: Static vs temporal features")
        st.dataframe(table("ablation_temporal.csv")[["model", "setting", "pr_auc_mean", "pr_auc_std", "recall_mean", "f1_mean"]].style.format(precision=4), hide_index=True)
        st.image(str(F / "ablation_temporal.png"))
    with c2:
        st.subheader("H3: Random vs chronological split")
        st.dataframe(table("split_comparison.csv")[["model", "setting", "pr_auc_mean", "roc_auc_mean", "f1_mean"]].style.format(precision=4), hide_index=True)
        st.image(str(F / "split_comparison.png"))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("LSTM / GRU history window")
        st.dataframe(table("sequence_models.csv")[["model", "window", "val_pr_auc", "pr_auc", "roc_auc", "f1"]].style.format(precision=4), hide_index=True)
        st.image(str(F / "history_window.png"))
    with c2:
        st.subheader("Class-imbalance strategies (XGBoost)")
        st.dataframe(table("imbalance_strategies.csv")[["strategy", "pr_auc_mean", "precision_mean", "recall_mean", "f1_mean"]].style.format(precision=4), hide_index=True)
        st.subheader("Concept drift")
        st.image(str(F / "drift.png"))
    st.subheader("H5: Operating thresholds for the final model")
    st.dataframe(table("operating_points.csv")[["rule", "threshold", "precision", "recall", "f1", "tp", "fn", "fp", "cost"]].style.format(precision=4), hide_index=True)
    st.subheader("Explainability (SHAP)")
    c1, c2, c3 = st.columns(3)
    c1.image(str(F / "shap_bar.png")); c2.image(str(F / "shap_summary.png")); c3.image(str(F / "shap_local_waterfall.png"))

with tab_live:
    bundle, explainer = load_model()
    demo = table("demo_transactions.csv")
    st.write("Score real test-period transactions with the saved **XGBoost (static + temporal)** model and explain each decision.")
    th = st.slider("Decision threshold", 0.01, 0.99, float(round(bundle["threshold"], 2)), 0.01,
                   help="Default is the threshold tuned on validation data (max F1).")
    probs = bundle["model"].predict_proba(demo[bundle["features"]])[:, 1]
    m = evaluate.metrics(demo.Class, probs, th)
    k = st.columns(5)
    k[0].metric("Fraud caught", f"{m['tp']} / {m['tp'] + m['fn']}"); k[1].metric("False alarms", m["fp"])
    k[2].metric("Precision", f"{m['precision']:.3f}"); k[3].metric("Recall", f"{m['recall']:.3f}"); k[4].metric("F1", f"{m['f1']:.3f}")
    st.caption(f"Sample: all {int(demo.Class.sum())} test-period frauds + 400 random legitimate test transactions.")
    view = demo[["Time", "Amount", "Class"]].assign(fraud_score=probs, flagged=probs >= th).sort_values("fraud_score", ascending=False)
    idx = st.selectbox("Pick a transaction to explain", view.index,
                       format_func=lambda i: f"#{i} | amount {demo.Amount[i]:.2f} | score {probs[i]:.3f} | actual {'FRAUD' if demo.Class[i] else 'legit'}")
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(view.head(50).style.format({"fraud_score": "{:.4f}", "Amount": "{:.2f}"}), height=420)
    with c2:
        sv = explainer(demo.loc[[idx], bundle["features"]])
        fig = plt.figure(); shap.plots.waterfall(sv[0], max_display=12, show=False); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.info(f"Fraud score {probs[idx]:.4f} -> {'ALERT for analyst review' if probs[idx] >= th else 'no alert'}. "
                "Red bars push toward fraud, blue toward legitimate. SHAP shows model attribution, not causality.")

with tab_method:
    st.markdown((C.ROOT / "docs" / "METHODOLOGY.md").read_text())
