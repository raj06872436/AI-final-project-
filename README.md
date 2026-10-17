# Temporal and Explainable Machine Learning for Early Financial Fraud Detection

T.Y. B.Tech Information Technology · AI Laboratory research project

A complete, leakage-free implementation that compares transaction-level, temporal and sequence (LSTM/GRU)
models for credit-card fraud detection on the ULB European Credit Card dataset (284,807 transactions,
492 frauds, 0.172 %), with SHAP explanations, threshold analysis, drift analysis and a Streamlit dashboard.

All numbers below were produced by `run_experiments.py` on the real dataset (3 random seeds, 16 min on a 4-core CPU).
Full tables: [`reports/RESULTS.md`](reports/RESULTS.md). Method: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py        # downloads data/creditcard.csv if missing, runs all 8 experiments
streamlit run app.py             # dashboard: dataset, results, live scoring with SHAP
pytest -q                        # leakage and metric tests
```

`python run_experiments.py --quick` runs one seed and fewer sequence windows. In Google Colab, upload the repo
(or `git clone` it), then run the same commands with a leading `!`.

## Project structure

```
src/fraud/
  config.py      paths, seeds, split ratios, feature groups, cost model
  data.py        download, audit, duplicate removal, chronological and random splits
  features.py    static, past-only temporal, and per-entity behavioral features
  models.py      LR, SMOTE-LR, RF, SMOTE-RF, XGBoost, LightGBM, MLP, imbalance variants
  sequence.py    LSTM / GRU over the last k transactions (PyTorch)
  evaluate.py    PR-AUC, ROC-AUC, F1, FPR, cost; validation-based thresholds; drift windows
  explain.py     SHAP global (summary, bar) and local (waterfall) explanations
  plots.py       figures
run_experiments.py   runs every experiment and writes results/ and reports/RESULTS.md
app.py               Streamlit dashboard
tests/               leakage tests (past-only features, no split overlap) and metric tests
results/             CSV tables, summary.json, figures/
models/              saved XGBoost (+temporal) model used by the dashboard
docs/METHODOLOGY.md  research design, hypotheses, leakage controls
demo/index.html      browser-only interactive explainer (no install needed)
```

## Pipeline

Raw data → audit → remove 1,081 duplicates → sort by time → **chronological 60/20/20 split** → static + past-only
temporal features → models (scaling and SMOTE inside pipelines, fit on train only) → **threshold chosen on validation**
→ single evaluation of the test period → drift, ablation, SHAP.

| Split | Rows | Fraud | Hours |
|---|---|---|---|
| Train | 170,235 | 342 | 0.0 – 33.4 |
| Validation | 56,745 | 57 | 33.4 – 40.3 |
| Test | 56,746 | 74 | 40.3 – 48.0 |

## Results (test period, mean over 3 seeds)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7447 | 0.9811 | 0.960 | 0.649 | 0.774 |
| SMOTE + Logistic Regression | 0.7574 | 0.9792 | 0.980 | 0.649 | 0.781 |
| **Random Forest** | **0.8165** | 0.9614 | 0.885 | 0.757 | 0.816 |
| SMOTE + Random Forest | 0.8143 | 0.9734 | 0.876 | 0.730 | 0.796 |
| XGBoost | 0.7929 | 0.9733 | 0.896 | 0.739 | 0.810 |
| LightGBM | 0.8003 | 0.9838 | 0.907 | 0.743 | 0.817 |
| MLP | 0.7935 | 0.9671 | 0.835 | 0.725 | 0.775 |
| LSTM (k=10) * | 0.8002 | 0.9594 | 0.979 | 0.635 | 0.771 |
| GRU (k=20) * | 0.8054 | 0.9830 | 0.981 | 0.703 | 0.819 |

\* single seed. The final model chosen on **validation** PR-AUC is XGBoost (test: 55 of 74 frauds caught, 7 false alarms).

## Findings for each hypothesis

| | Result |
|---|---|
| **H1** temporal features help | **Small / mixed.** PR-AUC XGBoost 0.793 → 0.797, LightGBM 0.800 → 0.803, LR 0.745 → 0.750; RF 0.817 → 0.814. On ULB the temporal features describe the global stream because there is no card ID, so most of the signal is already in V1–V28. |
| **H2** LSTM/GRU add value | **Comparable, not better.** GRU reaches 0.795–0.805 PR-AUC, similar to tree models; LSTM 0.787–0.800. Gains need real per-card sequences (IEEE-CIS). |
| **H3** random split is optimistic | **Supported.** XGBoost PR-AUC 0.866 on a random split vs 0.793 chronological; RF 0.855 vs 0.817. |
| **H4** PR-AUC > accuracy / ROC-AUC | **Supported.** "All legit" scores 99.83 % accuracy with 0 recall. RF has the lowest ROC-AUC (0.961) of the tree models but the highest PR-AUC (0.817). |
| **H5** best AUC ≠ best operating point | **Supported.** For the final model, the max-F1 threshold catches 55 frauds with 7 false alarms; the min-cost threshold catches 58 with 41; a ≥85 % recall target catches 64 but raises 478 alarms. |

**Other results**
- **History window:** for GRU, PR-AUC rises from k=5 (0.795) to k=20 (0.805) and then levels off at k=50 (0.803).
- **Class imbalance (XGBoost):** class weighting, SMOTE and no reweighting are within 0.001 PR-AUC of each other. Undersampling is the worst (0.784).
- **Drift:** every model loses PR-AUC in the last third of the test period (XGBoost 0.84 → 0.53). That third contains only 12 frauds, so treat this as an indication rather than a firm result.
- **SHAP:** the top features are V4, V14, V12, V11 and V10, which match the published benchmark.

## Figures

| | |
|---|---|
| ![PR and ROC](results/figures/pr_roc_curves.png) | ![Ablation](results/figures/ablation_temporal.png) |
| ![Split comparison](results/figures/split_comparison.png) | ![History window](results/figures/history_window.png) |
| ![SHAP summary](results/figures/shap_summary.png) | ![SHAP local](results/figures/shap_local_waterfall.png) |

## Notes and limitations

- LightGBM with a large `scale_pos_weight` diverged on this data (validation PR-AUC ≈ 0.01). Its settings (no reweighting, stronger leaf regularization) were chosen on the validation period.
- `add_behavioral_features` (per-card history) is implemented and unit-tested, but ULB has no card ID, so it is meant for IEEE-CIS.
- The ULB features are anonymized PCA components and the data covers only two days.
- SHAP shows what the model relied on, not what caused the fraud.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, PyTorch, SHAP, Matplotlib, Seaborn, Streamlit, pytest.

## References

Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015 ·
Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017 ·
Motie & Raahemi, *Financial fraud detection using graph neural networks: a systematic review*, ESWA 2024 ·
ULB / Worldline Credit Card Fraud Detection dataset · IEEE-CIS Fraud Detection dataset.
