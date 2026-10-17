"""
Temporal and Explainable ML for Early Financial Fraud Detection
Leakage-free pipeline on the ULB creditcard.csv dataset.

Run in Google Colab:
    !pip install xgboost lightgbm imbalanced-learn shap
    # upload creditcard.csv (Kaggle: mlg-ulb/creditcardfraud), then
    !python fraud_pipeline.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

SEED = 42

# 1. Dataset audit
df = pd.read_csv("creditcard.csv")
print(f"Rows: {len(df):,}  Fraud: {df.Class.sum()}  Rate: {df.Class.mean():.3%}")
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {dups} duplicate rows -> {len(df):,} rows, {df.Class.sum()} fraud")

# 2. Chronological split 60/20/20 (test is strictly later than train)
df = df.sort_values("Time").reset_index(drop=True)
a, b = int(len(df) * 0.6), int(len(df) * 0.8)
train, val, test = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()

# 3. Temporal features (past-only: rolling windows look backwards in time)
def add_temporal(d):
    d = d.copy()
    d["log_amount"] = np.log1p(d.Amount)
    d["hour"] = (d.Time // 3600) % 24
    d["gap_prev"] = d.Time.diff().fillna(0)
    t = pd.to_datetime(d.Time, unit="s")
    s = pd.Series(d.Amount.values, index=t)
    d["cnt_60s"] = s.rolling("60s").count().values
    d["amt_mean_1h"] = s.rolling("3600s").mean().values
    d["amt_std_1h"] = s.rolling("3600s").std().fillna(0).values
    d["amt_dev_1h"] = (d.Amount - d.amt_mean_1h) / (d.amt_std_1h + 1)
    return d

full = add_temporal(df)  # rolling windows only use earlier rows, so no future leakage
train, val, test = full.iloc[:a], full.iloc[a:b], full.iloc[b:]

STATIC = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]
TEMPORAL = ["gap_prev", "cnt_60s", "amt_mean_1h", "amt_std_1h", "amt_dev_1h"]

# 4. Models (scaler and SMOTE live inside pipelines -> fit on train only)
neg, pos = (train.Class == 0).sum(), (train.Class == 1).sum()
MODELS = {
    "LogReg": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "SMOTE-LogReg": Pipeline([("sc", StandardScaler()), ("sm", SMOTE(random_state=SEED)), ("m", LogisticRegression(max_iter=2000))]),
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED),
    "SMOTE-RF": Pipeline([("sm", SMOTE(random_state=SEED)), ("m", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))]),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=neg / pos, eval_metric="aucpr", random_state=SEED),
}

def tune_threshold(y, p):
    """Pick the threshold that maximizes F1 on VALIDATION data."""
    prec, rec, th = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return th[np.argmax(f1[:-1])]

def evaluate(name, model, feats):
    model.fit(train[feats], train.Class)
    th = tune_threshold(val.Class, model.predict_proba(val[feats])[:, 1])
    p = model.predict_proba(test[feats])[:, 1]          # test evaluated once
    yhat = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(test.Class, yhat).ravel()
    return dict(model=name, features="static+temporal" if feats != STATIC else "static",
                threshold=round(th, 3), precision=precision_score(test.Class, yhat),
                recall=recall_score(test.Class, yhat), f1=f1_score(test.Class, yhat),
                roc_auc=roc_auc_score(test.Class, p), pr_auc=average_precision_score(test.Class, p),
                TP=tp, FN=fn, FP=fp, TN=tn), p

results, probs = [], {}
for name, m in MODELS.items():
    r, p = evaluate(name, m, STATIC)
    results.append(r); probs[name] = p
    print(f"{name:14s} PR-AUC={r['pr_auc']:.4f} ROC-AUC={r['roc_auc']:.4f} F1={r['f1']:.4f}")

# 5. Ablation (H1): static vs static+temporal with XGBoost
r, p = evaluate("XGBoost", MODELS["XGBoost"], STATIC + TEMPORAL)
results.append(r); probs["XGBoost+temporal"] = p
res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n", res.round(4).to_string(index=False))
res.to_csv("results.csv", index=False)

# 6. Leakage demo (H3): random split vs chronological split
from sklearn.model_selection import train_test_split
Xr, Xt, yr, yt = train_test_split(full[STATIC], full.Class, test_size=0.2, stratify=full.Class, random_state=SEED)
xgb_r = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=neg / pos, random_state=SEED).fit(Xr, yr)
print(f"\nRandom split PR-AUC: {average_precision_score(yt, xgb_r.predict_proba(Xt)[:, 1]):.4f}  "
      f"vs chronological: {res[(res.model=='XGBoost') & (res.features=='static')].pr_auc.iloc[0]:.4f}")

# 7. Drift (early / middle / late windows of the test period)
p = probs["XGBoost"]
for i, idx in enumerate(np.array_split(np.arange(len(test)), 3)):
    y = test.Class.values[idx]
    if y.sum():
        print(f"Test window {['early', 'middle', 'late'][i]}: PR-AUC={average_precision_score(y, p[idx]):.4f} ({y.sum()} fraud)")

# 8. Plots: PR curves
plt.figure(figsize=(6, 5))
for name, pp in probs.items():
    pr, rc, _ = precision_recall_curve(test.Class, pp)
    plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(test.Class, pp):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (test period)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("pr_curves.png", dpi=150)

# 9. Explainability with SHAP (global + local)
import shap
xgb = MODELS["XGBoost"].fit(train[STATIC], train.Class)
explainer = shap.TreeExplainer(xgb)
sample = test[STATIC].sample(2000, random_state=SEED)
sv = explainer(sample)
shap.summary_plot(sv.values, sample, show=False); plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150); plt.close()
fraud_row = test[test.Class == 1][STATIC].iloc[[0]]
shap.plots.waterfall(explainer(fraud_row)[0], show=False); plt.tight_layout(); plt.savefig("shap_local.png", dpi=150)
print("\nSaved: results.csv, pr_curves.png, shap_summary.png, shap_local.png")
