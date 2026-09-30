"""Forecasting models: honest baselines -> classical stats -> global ML.

Every model shares one contract so the backtester treats them identically:
    fit/predict from history <= cutoff, return rows for cutoff+1 .. cutoff+horizon.

Per-series models (`forecast_one_series`) take a pandas frame for ONE store-SKU, so
on Databricks they drop straight into `groupBy(...).applyInPandas(...)` and scale
out across the cluster. The global LightGBM model trains once across all series.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import CAT_FEATURES, HORIZON

STAT_MODELS = ["naive_dow_avg", "seasonal_naive", "croston_sba", "ets"]


# ---------------------------------------------------------------- per-series models
def seasonal_naive(y: np.ndarray, h: int, m: int = 7) -> np.ndarray:
    last = y[-m:]
    return np.resize(last, h)


def naive_dow_avg(y: np.ndarray, h: int, weeks: int = 4) -> np.ndarray:
    """'Gut-feel' baseline: same weekday, average of last 4 weeks.
    This is what most SME planners do in Excel - the CURRENT STATE to beat."""
    y = y[-7 * weeks:]
    prof = y.reshape(weeks, 7).mean(axis=0)
    return np.resize(prof, h)


def croston_sba(y: np.ndarray, h: int, alpha: float = 0.1) -> np.ndarray:
    """Croston with Syntetos-Boylan bias correction for intermittent demand."""
    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return np.zeros(h)
    z, p, q = y[nz[0]], max(nz[0] + 1, 1), 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    return np.full(h, (1 - alpha / 2) * z / p)


def ets(y: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y[-365:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = ExponentialSmoothing(y + 1.0, trend="add", damped_trend=True, seasonal="mul",
                                     seasonal_periods=7, initialization_method="estimated").fit()
            f = m.forecast(h) - 1.0
        except Exception:
            f = seasonal_naive(y, h)
    return np.clip(f, 0, None)


def forecast_one_series(pdf: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = HORIZON,
                        segment: str | None = None) -> pd.DataFrame:
    """Run all per-series stat models for one store-SKU. Spark applyInPandas-friendly."""
    pdf = pdf.sort_values("date")
    hist = pdf.loc[pdf["date"] <= cutoff, "demand_filled"].to_numpy(float)
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon)
    seg = segment or (pdf["segment"].iloc[0] if "segment" in pdf else "smooth")
    out = {
        "naive_dow_avg": naive_dow_avg(hist, horizon),
        "seasonal_naive": seasonal_naive(hist, horizon),
        "croston_sba": croston_sba(hist, horizon),
        # ETS is poorly specified for intermittent/lumpy series -> route to Croston
        "ets": ets(hist, horizon) if seg in ("smooth", "erratic") else croston_sba(hist, horizon),
    }
    frames = [pd.DataFrame({"date": dates, "model": k, "forecast": v}) for k, v in out.items()]
    res = pd.concat(frames, ignore_index=True)
    res["store_id"], res["sku_id"] = pdf["store_id"].iloc[0], pdf["sku_id"].iloc[0]
    return res[["store_id", "sku_id", "date", "model", "forecast"]]


# ---------------------------------------------------------------- global ML model
LGBM_PARAMS = dict(
    objective="tweedie", tweedie_variance_power=1.2,  # count-like, zero-inflated retail demand
    learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    n_estimators=600, verbose=-1,
)


class GlobalLGBM:
    """One LightGBM across all store-SKUs. Learns cross-series patterns (promo
    response, holiday lift) that no single short series can learn alone."""

    def __init__(self, features: list[str], params: dict | None = None):
        self.features = features
        self.params = {**LGBM_PARAMS, **(params or {})}
        self.model = None

    def fit(self, df: pd.DataFrame, target: str = "demand_target"):
        import lightgbm as lgb
        tr = df[df[target].notna() & df[f"lag_{HORIZON}"].notna()]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(tr[self.features], tr[target],
                       categorical_feature=[c for c in CAT_FEATURES if c in self.features])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[self.features]), 0, None)

    def importance(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.features,
                              "gain": self.model.booster_.feature_importance("gain")})
                  .sort_values("gain", ascending=False).reset_index(drop=True))
