"""Leakage and correctness tests. Run with: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config as C, data, evaluate, features  # noqa: E402


def toy(n=3000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.normal(size=(n, 28)), columns=C.PCA_FEATURES)
    df["Time"] = np.sort(r.uniform(0, 172800, n)).round()
    df["Amount"] = r.lognormal(3, 1, n).round(2)
    df["Class"] = (r.uniform(size=n) < 0.02).astype(int)
    return df


def test_temporal_features_are_past_only():
    """Removing future rows must not change the features of earlier rows."""
    df = toy()
    full = features.build_features(df)
    cut = features.build_features(df.iloc[:2000])
    cols = C.STATIC_FEATURES + C.TEMPORAL_FEATURES
    pd.testing.assert_frame_equal(full[cols].iloc[:2000], cut[cols], check_exact=False, rtol=1e-9)


def test_behavioral_features_are_past_only():
    df = toy().assign(card=lambda d: np.arange(len(d)) % 50)
    full = features.add_behavioral_features(df, "card")
    cut = features.add_behavioral_features(df.iloc[:2000], "card")
    cols = ["ent_txn_count", "ent_gap_prev", "ent_amt_mean", "ent_amt_z"]
    pd.testing.assert_frame_equal(full[cols].iloc[:2000], cut[cols])
    assert (full.groupby("card").ent_txn_count.min() == 0).all()


def test_chronological_split_has_no_time_overlap():
    tr, va, te = data.chronological_split(data.clean(toy()))
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()
    assert len(tr) + len(va) + len(te) == len(data.clean(toy()))


def test_clean_removes_duplicates_and_sorts():
    df = toy()
    dup = pd.concat([df, df.iloc[:10]]).sample(frac=1, random_state=0)
    out = data.clean(dup)
    assert len(out) == len(df) and out.Time.is_monotonic_increasing


def test_metrics_confusion_and_formulas():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([.9, .4, .8, .1, .2, .7])
    m = evaluate.metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_threshold_rules_use_given_data():
    r = np.random.default_rng(1)
    y = (r.uniform(size=5000) < 0.01).astype(int)
    p = np.clip(y * 0.6 + r.uniform(size=5000) * 0.5, 0, 1)
    t = evaluate.threshold_max_f1(y, p)
    assert 0 < t < 1
    assert evaluate.metrics(y, p, evaluate.threshold_min_precision(y, p, 0.9))["precision"] >= 0.9
    assert evaluate.metrics(y, p, evaluate.threshold_min_recall(y, p, 0.85))["recall"] >= 0.85
