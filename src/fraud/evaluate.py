"""Metrics, validation-based threshold selection and operating points."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from .config import COST_FN, COST_FP, REVIEW_CAPACITY


def threshold_max_f1(y, p) -> float:
    prec, rec, th = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(th[int(np.argmax(f1))])


def threshold_min_cost(y, p, c_fn=COST_FN, c_fp=COST_FP) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.90, 0.99999, 400)))
    costs = [c_fn * ((p < t) & (y == 1)).sum() + c_fp * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def threshold_min_precision(y, p, min_precision=0.90) -> float:
    prec, rec, th = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= min_precision)[0]
    return float(th[ok[np.argmax(rec[ok])]]) if len(ok) else float(th[-1])


def threshold_min_recall(y, p, min_recall=0.85) -> float:
    prec, rec, th = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(th[ok[np.argmax(prec[ok])]]) if len(ok) else float(th[0])


def metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "accuracy": float((tp + tn) / len(y)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "cost": float(COST_FN * fn + COST_FP * fp),
    }


def operating_points(y_val, p_val, y_test, p_test) -> list:
    """Thresholds chosen on validation, then applied once to test (Section 18 of the paper)."""
    rules = {
        "Max F1": threshold_max_f1(y_val, p_val),
        "Min expected cost": threshold_min_cost(y_val, p_val),
        "High precision (>=90%)": threshold_min_precision(y_val, p_val),
        "High recall (>=85%)": threshold_min_recall(y_val, p_val),
        "Default 0.5": 0.5,
    }
    rows = [{"rule": r, **metrics(y_test, p_test, t)} for r, t in rules.items()]
    k = max(1, int(round(REVIEW_CAPACITY * len(p_test))))
    t_cap = float(np.sort(p_test)[-k])           # review only the top-k riskiest
    rows.append({"rule": f"Review capacity top {REVIEW_CAPACITY:.1%}", **metrics(y_test, p_test, t_cap)})
    return rows


def window_metrics(y, p, threshold, n_windows=3, names=("early", "middle", "late")) -> list:
    """Performance in successive time windows of the test period (concept drift check)."""
    out = []
    for name, idx in zip(names, np.array_split(np.arange(len(y)), n_windows)):
        yy, pp = np.asarray(y)[idx], np.asarray(p)[idx]
        if yy.sum() == 0:
            continue
        m = metrics(yy, pp, threshold)
        out.append({"window": name, "fraud": int(yy.sum()), **m})
    return out
