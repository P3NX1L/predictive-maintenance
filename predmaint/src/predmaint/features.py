"""Time-series feature engineering for predictive maintenance.

All features are computed PER MACHINE and use only past/current information
(causal rolling windows). This avoids leaking future sensor readings into the
features for a given cycle, which would inflate offline scores and fail in
production where the future is unknown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSOR_COLUMNS

ROLLING_WINDOWS = (5, 10, 20)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a least-squares line fit over a trailing window.

    Captures the *rate of degradation*, which is often more predictive than the
    raw level. Implemented with a closed-form OLS slope over an integer x-axis.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(values: np.ndarray) -> float:
        if len(values) < window:
            return np.nan
        y = values
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return series.rolling(window).apply(slope, raw=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added.

    Feature families per sensor:
      - rolling mean / std over multiple windows (level + volatility)
      - rolling slope (degradation rate)
      - lag-1 delta (instantaneous change)
      - expanding max (worst-seen so far, monotone degradation proxy)
    """
    df = df.sort_values(["machine_id", "cycle"]).copy()
    grouped = df.groupby("machine_id", group_keys=False)

    new_cols: dict[str, pd.Series] = {}
    for sensor in SENSOR_COLUMNS:
        new_cols[f"{sensor}_delta1"] = grouped[sensor].diff(1)
        new_cols[f"{sensor}_expmax"] = grouped[sensor].cummax()
        for w in ROLLING_WINDOWS:
            new_cols[f"{sensor}_rmean_{w}"] = grouped[sensor].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{sensor}_rstd_{w}"] = grouped[sensor].transform(
                lambda s, w=w: s.rolling(w, min_periods=2).std()
            )
            new_cols[f"{sensor}_slope_{w}"] = grouped[sensor].transform(
                lambda s, w=w: _rolling_slope(s, w)
            )

    feat = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    feat = feat.fillna(0.0)
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns to feed the model: everything except identifiers and targets."""
    exclude = {"machine_id", "cycle", "rul", "label"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from .data import generate_run_to_failure

    raw = generate_run_to_failure()
    feats = add_features(raw)
    cols = feature_columns(feats)
    print(f"raw columns: {raw.shape[1]}  ->  engineered feature columns: {len(cols)}")
    print("sample features:", cols[:6])
