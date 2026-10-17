"""Loading, auditing, de-duplication and leakage-free splitting."""
import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_PATH, DATA_URL, SEED, SPLIT, TARGET


def load_raw(path=DATA_PATH, download=True) -> pd.DataFrame:
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict:
    """Basic data-quality report (row counts, imbalance, duplicates, missing values)."""
    dups = df.duplicated()
    return {
        "rows": int(len(df)),
        "fraud": int(df[TARGET].sum()),
        "legit": int((df[TARGET] == 0).sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "duplicate_rows": int(dups.sum()),
        "duplicate_fraud_rows": int(df.loc[dups, TARGET].sum()),
        "missing_values": int(df.isna().sum().sum()),
        "time_span_hours": float(df.Time.max() / 3600),
        "trivial_all_legit_accuracy": float(1 - df[TARGET].mean()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (they can straddle a split boundary) and sort by time."""
    return df.drop_duplicates().sort_values("Time", kind="mergesort").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, ratios=SPLIT):
    """Earliest rows -> train, next -> validation, latest -> test. Test is strictly in the future."""
    assert df.Time.is_monotonic_increasing, "sort by Time before splitting"
    n = len(df)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def random_split(df: pd.DataFrame, ratios=SPLIT, seed=SEED):
    """Stratified random split, used only to show how much it overstates performance (H3)."""
    from sklearn.model_selection import train_test_split

    train, rest = train_test_split(df, train_size=ratios[0], stratify=df[TARGET], random_state=seed)
    val_share = ratios[1] / (ratios[1] + ratios[2])
    val, test = train_test_split(rest, train_size=val_share, stratify=rest[TARGET], random_state=seed)
    return train, val, test


def split_summary(parts, names=("train", "validation", "test")) -> pd.DataFrame:
    return pd.DataFrame([
        {"split": n, "rows": len(p), "fraud": int(p[TARGET].sum()), "fraud_rate": p[TARGET].mean(),
         "time_start_h": p.Time.min() / 3600, "time_end_h": p.Time.max() / 3600}
        for n, p in zip(names, parts)
    ])
