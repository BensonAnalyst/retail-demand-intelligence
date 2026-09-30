"""Forecast accuracy metrics - technical AND business aligned.

Conventions: error e = forecast - actual (positive bias = over-forecast = excess stock).
All "%" metrics are volume-weighted unless stated, because a 50% miss on a SKU that
sells 2 units/week matters less than a 10% miss on one that sells 2,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = ~(np.isnan(y) | np.isnan(f))
    return y[m], f[m]


def wape(y, f) -> float:
    """Weighted APE = sum|e| / sum(y). Robust to zeros; the retail default."""
    y, f = _clean(y, f)
    return float(np.abs(f - y).sum() / max(y.sum(), 1e-9))


def mape(y, f) -> float:
    """Classic MAPE on non-zero actuals only (undefined at 0; explodes on slow movers)."""
    y, f = _clean(y, f)
    m = y > 0
    return float(np.mean(np.abs(f[m] - y[m]) / y[m])) if m.any() else np.nan


def smape(y, f) -> float:
    y, f = _clean(y, f)
    d = np.abs(y) + np.abs(f)
    m = d > 0
    return float(np.mean(2 * np.abs(f[m] - y[m]) / d[m])) if m.any() else np.nan


def bias(y, f) -> float:
    """Volume-weighted bias = sum(e) / sum(y). +ve = over-forecast (overstock risk)."""
    y, f = _clean(y, f)
    return float((f - y).sum() / max(y.sum(), 1e-9))


def vn1_score(y, f) -> float:
    """VN1 Forecasting Competition score = WAPE + |Bias| (lower is better).
    Penalises accuracy AND systematic bias together - which is what drives inventory cost."""
    return wape(y, f) + abs(bias(y, f))


def mase(y, f, y_train, m: int = 7) -> float:
    """Mean Absolute Scaled Error vs in-sample seasonal-naive (m=7 weekly)."""
    y, f = _clean(y, f)
    yt = np.asarray(y_train, float)
    yt = yt[~np.isnan(yt)]
    scale = np.mean(np.abs(yt[m:] - yt[:-m])) if len(yt) > m else np.nan
    return float(np.mean(np.abs(f - y)) / scale) if scale and scale > 0 else np.nan


def summary(y, f) -> dict:
    return {"wape": wape(y, f), "mape": mape(y, f), "smape": smape(y, f),
            "bias": bias(y, f), "vn1": vn1_score(y, f), "n": int(len(_clean(y, f)[0]))}


def evaluate(df: pd.DataFrame, by: list[str], actual="actual", forecast="forecast") -> pd.DataFrame:
    """Metric table grouped by `by` (e.g. ['model'], ['model','segment'])."""
    return (df.groupby(by)
              .apply(lambda g: pd.Series(summary(g[actual], g[forecast])), include_groups=False)
              .reset_index())


def evaluate_at_level(df: pd.DataFrame, level: list[str], freq: str | None = "W",
                      models_col="model", actual="actual", forecast="forecast") -> pd.DataFrame:
    """Aggregate to a decision level first (e.g. SKU-store-WEEK for replenishment,
    category-week for buying), THEN score. Accuracy at the level decisions are made
    is what matters to the business."""
    d = df.copy()
    keys = [models_col] + level
    if freq:
        d["period"] = d["date"].dt.to_period(freq).dt.start_time
        keys.append("period")
    agg = d.groupby(keys)[[actual, forecast]].sum(min_count=1).reset_index()
    return evaluate(agg, [models_col], actual, forecast)
