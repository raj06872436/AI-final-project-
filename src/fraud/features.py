"""Feature engineering.

Every temporal feature for a transaction is computed only from that transaction and the
ones that happened before it in the time-ordered stream, so no future information is used.
The ULB data has no card or customer identifier, so temporal features describe the global
transaction stream. `add_behavioral_features` provides per-entity features for datasets that
do have an identifier (e.g. IEEE-CIS card1).
"""
import numpy as np
import pandas as pd


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df.Amount)
    hour = (df.Time // 3600) % 24
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def _past_count(times: np.ndarray, window: float) -> np.ndarray:
    """Number of earlier transactions within `window` seconds before each transaction."""
    left = np.searchsorted(times, times - window, side="left")
    return np.arange(len(times)) - left


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only temporal features. Expects df sorted by Time."""
    assert df.Time.is_monotonic_increasing
    df = df.copy()
    t = df.Time.to_numpy(dtype=float)
    amt = df.Amount.astype(float)
    log_amt = np.log1p(amt)

    gap = np.diff(t, prepend=t[0])
    df["log_gap_prev"] = np.log1p(gap)
    df["log_gap_next_prev"] = np.log1p(pd.Series(gap).shift(1).fillna(0).to_numpy())
    for w in (10, 60, 600):
        df[f"cnt_{w}s"] = _past_count(t, w)

    prev = log_amt.shift(1)                       # history excludes the current row
    past_mean = prev.expanding().mean().fillna(0)  # fallback built from past rows only
    roll = prev.rolling(100, min_periods=10)
    df["amt_mean_100"] = roll.mean().fillna(past_mean)
    df["amt_std_100"] = roll.std().fillna(1.0)
    df["amt_z_100"] = (log_amt - df.amt_mean_100) / (df.amt_std_100 + 1e-6)
    df["amt_rank_100"] = (
        pd.concat([prev.rolling(100, min_periods=10).quantile(q).rename(q) for q in (0.5, 0.9)], axis=1)
        .pipe(lambda q: (log_amt > q[0.5]).astype(int) + (log_amt > q[0.9]).astype(int))
    )

    ts = pd.Series(log_amt.to_numpy(), index=pd.to_datetime(t, unit="s"))
    hour_mean = ts.shift(1).rolling("3600s", min_periods=5).mean().to_numpy()
    hour_std = ts.shift(1).rolling("3600s", min_periods=5).std().to_numpy()
    df["amt_mean_1h"] = np.where(np.isnan(hour_mean), past_mean.to_numpy(), hour_mean)
    df["amt_z_1h"] = np.nan_to_num((log_amt.to_numpy() - hour_mean) / (np.nan_to_num(hour_std, nan=1.0) + 1e-6))
    return df


def add_behavioral_features(df: pd.DataFrame, entity: str, time: str = "Time", amount: str = "Amount") -> pd.DataFrame:
    """Per-entity history features (entity = card/customer id). Past-only via shift/expanding."""
    df = df.sort_values(time, kind="mergesort").copy()
    g = df.groupby(entity, sort=False)
    log_amt = np.log1p(df[amount])
    df["ent_txn_count"] = g.cumcount()
    df["ent_gap_prev"] = g[time].diff().fillna(-1)
    hist = log_amt.groupby(df[entity]).transform(lambda s: s.shift(1).expanding().mean())
    hist_sd = log_amt.groupby(df[entity]).transform(lambda s: s.shift(1).expanding().std())
    df["ent_amt_mean"] = hist.fillna(-1)  # -1 = no history yet
    df["ent_amt_z"] = ((log_amt - hist) / (hist_sd.fillna(1.0) + 1e-6)).fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_temporal_features(add_static_features(df))
