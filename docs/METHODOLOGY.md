# Methodology

## Research question
How much measurable value does temporal and behavioral information add to fraud detection when strong
baseline models are compared under one leakage-free evaluation framework?

## Hypotheses
| # | Hypothesis | Tested in |
|---|---|---|
| H1 | Leakage-safe temporal features improve PR-AUC and/or recall over static features alone | Experiment 2 (`ablation_temporal.csv`) |
| H2 | Sequence models (LSTM/GRU) add value when valid histories are available | Experiment 5 (`sequence_models.csv`) |
| H3 | Chronological evaluation shows more degradation than random evaluation | Experiment 3 (`split_comparison.csv`) |
| H4 | PR-AUC is a more meaningful selection signal than accuracy or ROC-AUC | Audit + all tables |
| H5 | The best model by AUC is not necessarily best at a practical false-positive limit | Experiment 6 (`operating_points.csv`) |

## Pipeline
1. **Audit** – row counts, imbalance, missing values, duplicates (`fraud/data.py`).
2. **Cleaning** – drop exact duplicates (they can sit on both sides of a split) and sort by `Time`.
3. **Chronological split 60/20/20** – earliest rows train, next validate, latest test. The test period is strictly later than training.
4. **Features** (`fraud/features.py`)
   - *Static*: V1–V28, log(amount), hour of day as sin/cos.
   - *Temporal* (past-only): gap to previous transaction, previous gap, transaction counts in the last 10 s / 60 s / 600 s, rolling mean/std/z-score and rank of amount over the last 100 transactions, amount z-score vs the last hour.
   - *Behavioral* (`add_behavioral_features`): per-entity count, gap, historical mean amount and z-score. The ULB data has no card ID, so this is provided and unit-tested for datasets such as IEEE-CIS rather than faked from row order.
5. **Models** (`fraud/models.py`): Logistic Regression, SMOTE + LR, Random Forest, SMOTE + RF, XGBoost, LightGBM, MLP, and LSTM / GRU (`fraud/sequence.py`, PyTorch).
6. **Imbalance handling**: class weighting, SMOTE, undersampling, threshold tuning. SMOTE and scaling live inside pipelines, so they only ever see training rows.
7. **Thresholds** are chosen on the validation period (max F1, minimum expected cost, ≥90 % precision, ≥85 % recall, analyst review capacity). The test period is evaluated once.
8. **Final model** is selected by validation PR-AUC, never by test results.
9. **Drift**: the test period is split into early / middle / late windows and scored separately.
10. **Explainability**: SHAP TreeExplainer – global (summary, bar) and local (waterfall for one flagged transaction).

## Leakage controls (and how they are checked)
| Control | Where | Test |
|---|---|---|
| Test period strictly after training | `chronological_split` + assert in runner | `test_chronological_split_has_no_time_overlap` |
| Temporal features use only past rows | `add_temporal_features` | `test_temporal_features_are_past_only` (removing future rows leaves earlier features unchanged) |
| Entity features use only past rows | `add_behavioral_features` | `test_behavioral_features_are_past_only` |
| Duplicates removed before split | `clean` | `test_clean_removes_duplicates_and_sorts` |
| Scaler / SMOTE fit on train only | imblearn `Pipeline` | by construction |
| Threshold from validation only | `run_experiments.fit_eval` | by construction |

## Sequence models
Each sample is the window of the last *k* transactions (k ∈ {5, 10, 20, 50}) ending at the current one; the
label is the current transaction's label. A GRU/LSTM reads the window and its last hidden state is combined
with the current transaction's features. Training uses all frauds plus a fresh 25 % sample of legitimate
windows each epoch, weighted BCE loss, and early stopping on validation PR-AUC. On ULB the window is the
global stream (no card ID), which is a stated limitation.

## Metrics
Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1, PR-AUC (average precision, primary), ROC-AUC (secondary),
FPR = FP/(FP+TN), and expected cost = 100·FN + 5·FP for the cost-based threshold.

## Limitations
- ULB features are anonymized PCA components; only two days of data, so long-term drift is limited.
- No card/customer identifier on ULB, so true per-card sequences and behavioral features cannot be built there.
- SHAP explains the model's attribution, not causal relationships.
- Results on a public benchmark may not transfer directly to production banking data.
