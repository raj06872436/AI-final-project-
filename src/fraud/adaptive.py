"""Novelty: explanation-drift-triggered retraining under delayed labels.

A deployed fraud model sees new transactions immediately but their labels (chargebacks, analyst
verdicts) only arrive later. Label-based monitoring therefore reacts late. This module monitors the
model's *SHAP explanations* of incoming, still-unlabelled transactions and retrains only when the way the
model reasons about the stream has shifted ("explanation drift").

Stream protocol (prequential, past-only):
  * The post-training period is cut into consecutive chunks c_0, c_1, ...
  * Chunk c_t is scored by the current model before any of its labels are known.
  * Labels arrive with a delay of `delay` chunks: when c_t is scored, labels of c_0 .. c_{t-1-delay} are known.
  * After scoring c_t a strategy decides whether to retrain on all labels known by then.

Drift scores (both label-free, both calibrated the same way so the comparison is fair):
  * explanation drift  D_x = sum_j W1(SHAP_ref_j, SHAP_cur_j) / sum_j E|SHAP_ref_j|
    i.e. how much of the model's attribution mass has moved (W1 = 1-D Wasserstein distance).
  * input drift        mean Population Stability Index over the input features (classic baseline).
A trigger fires when the score of a chunk exceeds the q-quantile of scores of the most recent training
chunks against the same reference, so no labels and no future data are used to set it.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binom, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------- drift scores
def shap_matrix(model, X) -> np.ndarray:
    """Exact TreeSHAP values from XGBoost's native implementation (bias column dropped)."""
    contribs = model.get_booster().predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), pred_contribs=True)
    return contribs[:, :-1]


def explanation_drift(ref_shap: np.ndarray, cur_shap: np.ndarray) -> float:
    """Share of the reference attribution mass that has moved, summed over features."""
    moved = sum(wasserstein_distance(ref_shap[:, j], cur_shap[:, j]) for j in range(ref_shap.shape[1]))
    return float(moved / (np.abs(ref_shap).mean(0).sum() + 1e-12))


def drift_attribution(ref_shap: np.ndarray, cur_shap: np.ndarray, names) -> pd.Series:
    """Explain the alarm: each feature's share of the moved attribution mass (sums to 1)."""
    moved = np.array([wasserstein_distance(ref_shap[:, j], cur_shap[:, j]) for j in range(ref_shap.shape[1])])
    return pd.Series(moved / (moved.sum() + 1e-12), index=list(names)).sort_values(ascending=False)


def psi(ref: np.ndarray, cur: np.ndarray, bins=10) -> float:
    """Mean Population Stability Index over columns, quantile bins taken from the reference."""
    out = []
    for j in range(ref.shape[1]):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(ref[:, j], edges)[0] / len(ref) + 1e-4
        b = np.histogram(cur[:, j], edges)[0] / len(cur) + 1e-4
        out.append(np.sum((b - a) * np.log(b / a)))
    return float(np.mean(out))


class DriftMonitor:
    """Label-free monitor: reference window + threshold calibrated on recent training chunks."""

    def __init__(self, kind: str, chunk: int, ref_rows: int = 10_000, n_calib: int = 12, q: float = 0.9):
        assert kind in ("explanation", "input")
        self.kind, self.chunk, self.ref_rows, self.n_calib, self.q = kind, chunk, ref_rows, n_calib, q

    def _view(self, model, X):
        return shap_matrix(model, X) if self.kind == "explanation" else np.asarray(X, dtype=np.float32)

    def _score(self, cur):
        return explanation_drift(self.ref, cur) if self.kind == "explanation" else psi(self.ref, cur)

    def fit(self, model, X_train: np.ndarray):
        """Reference = the most recent labelled rows; threshold = q-quantile of earlier training chunks."""
        self.ref = self._view(model, X_train[-self.ref_rows:])
        hist = X_train[:-self.ref_rows]
        starts = [len(hist) - (i + 1) * self.chunk for i in range(self.n_calib)]
        calib = [self._score(self._view(model, hist[s:s + self.chunk])) for s in starts if s >= 0]
        self.threshold = float(np.quantile(calib, self.q))
        return self

    def score(self, model, X_chunk) -> float:
        return self._score(self._view(model, X_chunk))


# ---------------------------------------------------------------- stream simulation
def budget_hits(y, p, budget):
    """Frauds caught when analysts review only the top `budget` share of a chunk."""
    k = max(1, int(round(budget * len(p))))
    return int(np.asarray(y)[np.argsort(-np.asarray(p))[:k]].sum()), k


def memory_score(model, X_mem, y_mem, X_query, k=10) -> np.ndarray:
    """Explanation memory: share of confirmed frauds among the k nearest labelled cases in SHAP space.

    Neighbours are found among the model's explanations, not its raw inputs, so "similar" means
    "the model reasons about them the same way". New labels change the score immediately, without retraining.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(y_mem))).fit(shap_matrix(model, X_mem))
    _, idx = nn.kneighbors(shap_matrix(model, X_query))
    return np.asarray(y_mem)[idx].mean(1)


def run_stream(make_model, X_hist, y_hist, X_stream, y_stream, n_chunks=24, delay=1, strategy="frozen",
               period=4, retrain_at=None, budget=0.002, monitor_kw=None, adapt=4, guard_recall=0.5, guard_alpha=0.05,
               memory=False, memory_k=10, retrain=True):
    """Prequential simulation of one retraining strategy.

    strategy: 'frozen' | 'periodic' (every `period` chunks) | 'scheduled' (at chunk ids in `retrain_at`)
              | 'input_drift' | 'explanation_drift' | 'hybrid'.
    'hybrid' = explanation drift plus a delayed-label guard: when a chunk's labels arrive and the deployed model
    caught significantly fewer than `guard_recall` of its frauds, it also raises an
    alarm. "Fewer" is tested with a one-sided binomial test at level `guard_alpha`, so a chunk with only a
    handful of frauds cannot raise it by chance. This covers concept drift that leaves the unlabelled data looking normal.
    Drift triggers use alarm-then-adapt: an alarm opens an adaptation window of `adapt` chunks during which the
    model is retrained whenever new labels arrive. A single retrain right after the alarm would not help,
    because with delayed labels the drifted chunk's own labels are not known yet.
    memory=True adds the explanation memory: from the first alarm on, each transaction's score becomes
    1 - (1 - p)(1 - q), where q = memory_score() over the stream chunks labelled since that alarm. The memory
    grows the moment labels arrive, so the system adapts between (or, with retrain=False, without) retrains.
    Returns (per-chunk DataFrame, stream scores, number of retrains, seconds spent retraining).
    """
    chunks = np.array_split(np.arange(len(y_stream)), n_chunks)
    chunk_len = len(chunks[0])

    def train(n_known):                       # history + the first n_known labelled stream chunks
        idx = np.concatenate(chunks[:n_known]) if n_known else np.array([], dtype=int)
        X = np.vstack([X_hist, X_stream[idx]]); y = np.concatenate([y_hist, y_stream[idx]])
        t = time.time(); m = make_model(); m.fit(X, y)
        return m, X, time.time() - t

    model, X_tr, fit_s = train(0)
    # Only the monitors a strategy uses are refit after retraining; the frozen run records both signals.
    kinds = {"frozen": ("explanation", "input"), "input_drift": ("input",),
             "explanation_drift": ("explanation",), "hybrid": ("explanation",)}.get(strategy, ())
    monitors = {}
    def refit_monitors():
        for kind in kinds:
            monitors[kind] = DriftMonitor(kind, chunk_len, **(monitor_kw or {})).fit(model, X_tr)
    refit_monitors()
    known, retrains, retrain_s, adapt_left, mem_from = 0, 0, 0.0, 0, None
    rows, scores = [], np.zeros(len(y_stream))
    for t, idx in enumerate(chunks):
        Xc, yc = X_stream[idx], y_stream[idx]
        p = model.predict_proba(Xc)[:, 1]                 # scored before its labels exist
        mem_idx = (np.concatenate(chunks[mem_from:t - delay]) if mem_from is not None and t - delay > mem_from
                   else np.array([], dtype=int))           # chunks labelled since the alarm
        if memory and y_stream[mem_idx].sum() > 0:
            p = 1 - (1 - p) * (1 - memory_score(model, X_stream[mem_idx], y_stream[mem_idx], Xc, memory_k))
        scores[idx] = p
        dx = monitors["explanation"].score(model, Xc) if "explanation" in monitors else np.nan
        di = monitors["input"].score(model, Xc) if "input" in monitors else np.nan
        hits, k = budget_hits(yc, p, budget)
        rows.append({"chunk": t, "rows": len(idx), "fraud": int(yc.sum()),
                     "pr_auc": average_precision_score(yc, p) if yc.sum() else np.nan,
                     "hits_at_budget": hits, "alerts": k,
                     "explanation_drift": dx, "explanation_threshold": getattr(monitors.get("explanation"), "threshold", np.nan),
                     "input_drift": di, "input_threshold": getattr(monitors.get("input"), "threshold", np.nan),
                     "known_chunks": known, "retrained_after": False})
        rows[-1]["explanation_alarm"] = bool(dx > rows[-1]["explanation_threshold"])
        rows[-1]["input_alarm"] = bool(di > rows[-1]["input_threshold"])

        available = t + 1 - delay                         # chunks whose labels are now known
        labelled = rows[t - delay] if t >= delay else None   # chunk whose labels just arrived
        guard = (labelled is not None and labelled["fraud"] > 0
                 and binom.cdf(labelled["hits_at_budget"], labelled["fraud"], guard_recall) < guard_alpha)
        rows[-1]["label_guard_alarm"] = bool(guard)
        alarm = {"input_drift": lambda: di > monitors["input"].threshold,
                 "explanation_drift": lambda: dx > monitors["explanation"].threshold,
                 "hybrid": lambda: dx > monitors["explanation"].threshold or guard}.get(strategy, lambda: False)()
        if alarm:
            adapt_left = adapt if retrain else 0
            if mem_from is None:
                mem_from = max(t - delay, 0)
        fire = {"frozen": False,
                "periodic": (t + 1) % period == 0,
                "scheduled": t in set(retrain_at or ())}.get(strategy, adapt_left > 0)
        adapt_left = max(adapt_left - 1, 0)
        if fire and available > known and t < n_chunks - 1:
            known = available
            model, X_tr, fit_s = train(known)
            refit_monitors()
            retrains += 1; retrain_s += fit_s
            rows[-1]["retrained_after"] = True
    return pd.DataFrame(rows), scores, retrains, retrain_s


def inject_shift(X_stream, n_chunks, onset, cols, shift, scale_sd):
    """Controlled covariate shift: from chunk `onset` on, add `shift` training-SDs to the given columns."""
    X = X_stream.copy()
    start = np.array_split(np.arange(len(X)), n_chunks)[onset][0]
    for j in cols:
        X[start:, j] += shift * scale_sd[j]
    return X


def stream_summary(y_stream, scores, per_chunk: pd.DataFrame) -> dict:
    return {"pr_auc": float(average_precision_score(y_stream, scores)),
            "roc_auc": float(roc_auc_score(y_stream, scores)),
            "frauds_caught_at_budget": int(per_chunk.hits_at_budget.sum()),
            "alerts": int(per_chunk.alerts.sum()),
            "recall_at_budget": float(per_chunk.hits_at_budget.sum() / per_chunk.fraud.sum()),
            "mean_chunk_pr_auc": float(per_chunk.pr_auc.mean())}
