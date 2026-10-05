"""7-day-ahead daily forecasting, trained on (a) raw sales vs (b) recovered demand.

Same direct design as the main project: all target lags are shifted >= 7 days, so one
global LightGBM forecasts every day of the next week from one cutoff. Known-in-advance
inputs (discount plan, holiday/activity flags) are not shifted. Weather uses the actual
values as a stand-in for a weather forecast (stated as an assumption).

Scoring follows the paper: only on days with NO stock-out in operating hours, because
only there is the observed sale the true demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY, OP_HOURS
from .metrics import summary

H = 7
LAGS = [7, 14, 21, 28]
CATS = ["city_id", "store_id", "management_group_id", "first_category_id",
        "second_category_id", "third_category_id"]
KNOWN = ["discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature",
         "avg_humidity", "avg_wind_level"]


def build_frame(df: pd.DataFrame, oos: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    """df/oos/target aligned row-wise, covering train+test days. Adds leakage-safe features."""
    d = df.copy()
    d["y"] = target
    d["oos_hours"] = oos[:, OP_HOURS].sum(1)
    d = d.sort_values(KEY + ["day"]).reset_index(drop=True)
    g = d.groupby(KEY, sort=False)["y"]
    for L in LAGS:
        d[f"lag_{L}"] = g.shift(L)
    base = g.shift(H)
    gb = base.groupby([d[k] for k in KEY])
    for w in (7, 14, 28):
        d[f"rmean_{w}"] = gb.transform(lambda s: s.rolling(w, min_periods=1).mean())
    d["rstd_14"] = gb.transform(lambda s: s.rolling(14, min_periods=2).std())
    d["dow_mean"] = d[[f"lag_{L}" for L in LAGS]].mean(axis=1)
    oos_hist = d.groupby(KEY, sort=False)["oos_hours"].shift(H)
    d["oos_rate_14"] = oos_hist.groupby([d[k] for k in KEY]).transform(
        lambda s: s.rolling(14, min_periods=1).mean()) / 17.0
    d["dow"] = d["dt"].dt.dayofweek
    d["dom"] = d["dt"].dt.day
    d["disc_depth"] = 1 - d["discount"]
    for c in CATS + ["product_id"]:
        d[c] = d[c].astype("category")
    return d


FEATURES = (CATS + ["product_id"] + KNOWN + [f"lag_{L}" for L in LAGS] +
            ["rmean_7", "rmean_14", "rmean_28", "rstd_14", "dow_mean", "oos_rate_14", "dow", "dom", "disc_depth"])

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05, num_leaves=127,
              min_child_samples=100, n_estimators=600, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)


def fit_predict(frame: pd.DataFrame, cutoff_day: int, params: dict | None = None):
    import lightgbm as lgb
    tr = frame[(frame["day"] <= cutoff_day) & frame["lag_28"].notna()]
    te = frame[(frame["day"] > cutoff_day) & (frame["day"] <= cutoff_day + H)]
    m = lgb.LGBMRegressor(**{**PARAMS, **(params or {})})
    m.fit(tr[FEATURES], tr["y"], categorical_feature=[c for c in CATS + ["product_id"]])
    pred = np.clip(m.predict(te[FEATURES]), 0, None)
    return te.index.to_numpy(), pred, m


EVAL_VIEWS = {
    "A": "A. in-stock days only (paper protocol)",
    "B": "B. all days vs raw sales (censored)",
    "C": "C. all days vs recovered demand, LightGBM (censoring-aware)",
    "C2": "C2. all days vs recovered demand, profile (independent check)",
}


def score(actual: np.ndarray, pred: np.ndarray, clean: np.ndarray, sale_group: np.ndarray | None = None,
          actual_recovered: np.ndarray | None = None, actual_recovered_alt: np.ndarray | None = None) -> list[dict]:
    """Three views of the truth, because none is perfect:
    A  in-stock days: truth is exact, but these days are SELECTED - high-demand days are the
       ones that sell out, so A is biased toward low-demand days and flatters under-forecasting.
    B  all days vs raw sales: truth is censored (too low) on stock-out days.
    C  all days vs recovered demand: unbiased if recovery is unbiased (validated by masking)."""
    rows = [{"eval_view": EVAL_VIEWS["A"], **summary(actual[clean], pred[clean])},
            {"eval_view": EVAL_VIEWS["B"], **summary(actual, pred)}]
    if actual_recovered is not None:
        rows.append({"eval_view": EVAL_VIEWS["C"], **summary(actual_recovered, pred)})
    if actual_recovered_alt is not None:
        rows.append({"eval_view": EVAL_VIEWS["C2"], **summary(actual_recovered_alt, pred)})
    if sale_group is not None:
        for gname in ("high-sale", "low-sale"):
            m = clean & (sale_group == gname)
            rows.append({"eval_view": f"A, {gname} series", **summary(actual[m], pred[m])})
            if actual_recovered is not None:
                m2 = sale_group == gname
                rows.append({"eval_view": f"C, {gname} series", **summary(actual_recovered[m2], pred[m2])})
    return rows
