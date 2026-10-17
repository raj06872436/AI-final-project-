# Temporal and Explainable Machine Learning for Early Financial Fraud Detection

T.Y. B.Tech Information Technology · AI Laboratory research project.

A leakage-free comparison of transaction-level, temporal and behavioral models for credit-card fraud detection on the ULB European Credit Card dataset (284,807 transactions, 492 fraud, 0.172%).

## Contents

| Path | What it is |
|---|---|
| `fraud_pipeline.py` | Full pipeline for the real dataset: duplicate removal, chronological 60/20/20 split, temporal features, LR / SMOTE-LR / RF / SMOTE-RF / XGBoost, validation-tuned thresholds, static vs temporal ablation, random vs chronological split comparison, drift windows, PR curves, SHAP global + local explanations |
| `demo/index.html` | Interactive viva demo (open in a browser). Trains a logistic regression in the browser on synthetic transaction data to illustrate the method, and shows the verified benchmark results, hypotheses and Q&A |
| `demo/engine.js` | Demo engine: synthetic data, leakage-safe features, splits, logistic regression, PR-AUC / ROC-AUC |
| `demo/page.src.html` | Demo page source (`index.html` = this file with `engine.js` inlined) |

## Run the pipeline (Google Colab)

```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) into this folder
python fraud_pipeline.py
```

Outputs: `results.csv`, `pr_curves.png`, `shap_summary.png`, `shap_local.png`.

## Method

Raw transactions → dataset audit → chronological split → train-only preprocessing → static + temporal features → baseline models → threshold tuning on validation → SHAP → metrics, drift and ablation.

Leakage rules: scaler and SMOTE are fit on training data only, the threshold is chosen on validation, temporal features use only past transactions, and the test period is evaluated once.

## Verified public benchmark results (ULB dataset)

These numbers come from public benchmark reports, not from a fresh run of this code.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 83.95% | 71.58% | 0.7727 | 0.9558 | 0.6939 |
| SMOTE Logistic Regression | 88.46% | 72.63% | 0.7977 | 0.9582 | 0.7179 |
| Random Forest | 89.87% | 74.74% | 0.8161 | 0.9189 | 0.7983 |
| **SMOTE Random Forest** | 92.00% | 72.63% | 0.8118 | 0.9598 | **0.8016** |
| MLP | 82.76% | 75.79% | 0.7912 | 0.9365 | 0.6838 |
| XGBoost (time-ordered) | 93.00% | 71.60% | 0.8090 | 0.9840 | 0.7940 |

LightGBM and LSTM/GRU are planned (LSTM/GRU need a dataset with card IDs such as IEEE-CIS).

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, Matplotlib.
