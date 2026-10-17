"""SHAP explanations (global and local) for tree models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def shap_analysis(model, X_test: pd.DataFrame, y_test, fig_dir, seed=42, n=3000):
    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(seed)
    frauds = np.where(np.asarray(y_test) == 1)[0]
    legit = rng.choice(np.where(np.asarray(y_test) == 0)[0], n - len(frauds), replace=False)
    sample = X_test.iloc[np.concatenate([frauds, legit])]
    sv = explainer(sample)

    importance = (pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(sv.values).mean(0)})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

    plt.figure(); shap.summary_plot(sv.values, sample, max_display=15, show=False)
    plt.title("Global SHAP: feature impact on fraud score"); plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=140, bbox_inches="tight"); plt.close("all")

    plt.figure(); shap.summary_plot(sv.values, sample, plot_type="bar", max_display=15, show=False)
    plt.title("Global SHAP: mean |SHAP value|"); plt.tight_layout()
    plt.savefig(fig_dir / "shap_bar.png", dpi=140, bbox_inches="tight"); plt.close("all")

    # Local explanation for the highest-scoring true fraud in the test period
    fraud_X = X_test.iloc[frauds]
    top = fraud_X.iloc[[int(np.argmax(model.predict_proba(fraud_X)[:, 1]))]]
    plt.figure(); shap.plots.waterfall(explainer(top)[0], max_display=12, show=False)
    plt.title("Local SHAP: why this transaction was flagged"); plt.tight_layout()
    plt.savefig(fig_dir / "shap_local_waterfall.png", dpi=140, bbox_inches="tight"); plt.close("all")
    return importance
