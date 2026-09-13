"""Feature engineering.

Velocity / aggregation features are computed with a strictly backward-looking window, so a transaction only
"sees" transactions that happened before it (no target or future leakage). The functions accept an optional
entity key (card id, device id, account) - on the ULB dataset no such key exists, so the pipeline uses a single
global key, which turns the features into "how busy is the whole system right now" signals. On a dataset with
card/device ids the same code produces per-card / per-device velocity."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"1h": 3600, "6h": 21_600, "24h": 86_400}


def add_time_features(df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"] = ((out[time_col] % 86_400) // 3600).astype(int)
    out["is_night"] = out["hour_of_day"].between(0, 5).astype(int)
    out["log_amount"] = np.log1p(out[amount_col])
    out["amount_is_round"] = ((out[amount_col] % 1 == 0) & (out[amount_col] > 0)).astype(int)
    out["amount_cents_99"] = (np.round(out[amount_col] % 1, 2) == 0.99).astype(int)
    return out


def _trailing_window_stats(t: np.ndarray, a: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """For each i: count and amount-sum of rows j with t[i]-window < t[j] < t[i] (strictly earlier). t must be sorted."""
    # searchsorted gives, for each row, the first index whose time is > t[i]-window
    lo = np.searchsorted(t, t - window, side="right")
    hi = np.searchsorted(t, t, side="left")       # rows strictly before t[i]
    csum = np.concatenate([[0.0], np.cumsum(a)])
    count = (hi - lo).astype(float)
    total = csum[hi] - csum[lo]
    return count, total


def add_velocity_features(df: pd.DataFrame, key_col: str | None = None, time_col: str = "Time", amount_col: str = "Amount",
                          windows: dict[str, float] | None = None) -> pd.DataFrame:
    """Backward-looking counts / amount sums / amount z-scores per key (or globally when key_col is None)."""
    windows = windows or WINDOWS
    out = df.copy()
    order = np.argsort(out[time_col].to_numpy(), kind="stable")
    out = out.iloc[order]
    groups = [out.index] if key_col is None else [idx for _, idx in out.groupby(key_col, sort=False).groups.items()]
    new_cols = {f"vel_{w}_count": np.zeros(len(out)) for w in windows} | {f"vel_{w}_amount": np.zeros(len(out)) for w in windows}
    new_cols["secs_since_prev"] = np.zeros(len(out))
    new_cols["amount_z_24h"] = np.zeros(len(out))
    pos = pd.Series(np.arange(len(out)), index=out.index)
    for idx in groups:
        p = pos.loc[idx].to_numpy()
        t = out[time_col].to_numpy()[p].astype(float)
        a = out[amount_col].to_numpy()[p].astype(float)
        for name, w in windows.items():
            c, s = _trailing_window_stats(t, a, w)
            new_cols[f"vel_{name}_count"][p] = c
            new_cols[f"vel_{name}_amount"][p] = s
        prev = np.concatenate([[np.nan], t[:-1]])
        new_cols["secs_since_prev"][p] = np.where(np.isnan(prev), 1e6, t - prev)
        # z-score of this amount vs. the trailing 24h amounts (mean/std from sums of a and a^2)
        c24, s24 = _trailing_window_stats(t, a, windows.get("24h", 86_400))
        _, sq24 = _trailing_window_stats(t, a * a, windows.get("24h", 86_400))
        mean = np.where(c24 > 0, s24 / np.maximum(c24, 1), 0.0)
        var = np.where(c24 > 1, sq24 / np.maximum(c24, 1) - mean**2, 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            new_cols["amount_z_24h"][p] = np.where(std > 0, (a - mean) / std, 0.0)
    for k, v in new_cols.items():
        out[k] = v
    return out.sort_index()


def build_features(df: pd.DataFrame, key_col: str | None = None) -> pd.DataFrame:
    return add_velocity_features(add_time_features(df), key_col=key_col)


def feature_columns(df: pd.DataFrame, target: str = "Class") -> list[str]:
    drop = {target, "Time"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
