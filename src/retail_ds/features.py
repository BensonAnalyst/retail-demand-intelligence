"""Silver -> Gold feature engineering (leakage-safe for a HORIZON-day forecast).

Design choice (trade-off documented for the interview):
  All demand lags / rolling windows are shifted by >= HORIZON days, so ONE global
  model can forecast every day of the next HORIZON days from a single cutoff with no
  recursive feedback. Cost: slightly weaker short-horizon accuracy vs. a recursive or
  per-horizon model. Benefit: one model, one training job, trivially parallel scoring,
  no error compounding - the right call for weekly batch replenishment.

Known-in-future features (price, promo plan, calendar, holidays) are NOT shifted:
the business plans them in advance, which is exactly what makes promo forecasting work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_gen import HOLIDAYS

HORIZON = 28
LAGS = [HORIZON, HORIZON + 7, HORIZON + 14, HORIZON + 21, 364]
WINDOWS = [7, 28, 91]
CAT_FEATURES = ["store_id", "sku_id", "category", "region", "segment"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_payday_window"] = d.dt.day.isin([25, 26, 27, 28, 29, 30, 31, 1]).astype(int)  # SG/MY month-end pay
    hol = pd.to_datetime(sorted(HOLIDAYS))
    idx = np.searchsorted(hol.values, d.values)
    nxt = hol.values[np.clip(idx, 0, len(hol) - 1)]
    df["days_to_holiday"] = np.clip((nxt - d.values) / np.timedelta64(1, "D"), 0, 60)
    df.loc[idx >= len(hol), "days_to_holiday"] = 60
    return df


def build_features(silver: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame,
                   segments: pd.DataFrame) -> pd.DataFrame:
    df = (silver.merge(products[["sku_id", "category", "unit_cost"]], on="sku_id")
                .merge(stores[["store_id", "region", "store_size"]], on="store_id")
                .merge(segments[["store_id", "sku_id", "segment"]], on=["store_id", "sku_id"])
                .sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True))
    df = add_calendar(df)

    # price / promo (known in advance)
    df["price_ratio"] = df["selling_price"] / df["regular_price"]
    # expanding (past-only) mean: a full-history mean would leak future price levels
    past_mean_price = df.groupby(["store_id", "sku_id"])["regular_price"].transform(lambda s: s.expanding().mean())
    df["rel_price"] = df["selling_price"] / past_mean_price
    df["log_price"] = np.log(df["selling_price"])

    # demand history (shifted >= HORIZON)
    g = df.groupby(["store_id", "sku_id"])["demand_filled"]
    for L in LAGS:
        df[f"lag_{L}"] = g.shift(L)
    base = g.shift(HORIZON)
    gb = base.groupby([df["store_id"], df["sku_id"]])
    for w in WINDOWS:
        df[f"rmean_{w}"] = gb.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["rstd_28"] = gb.transform(lambda s: s.rolling(28, min_periods=2).std())
    df["rzero_28"] = (base == 0).groupby([df["store_id"], df["sku_id"]]).transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["dow_mean_4w"] = df[[f"lag_{L}" for L in LAGS[:4]]].mean(axis=1)  # same weekday, last 4 weeks
    df["trend_ratio"] = df["rmean_28"] / (df["rmean_91"] + 0.1)

    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df


def make_future_frame(silver: pd.DataFrame, horizon: int = HORIZON,
                      promo_plan: pd.DataFrame | None = None,
                      products: pd.DataFrame | None = None) -> pd.DataFrame:
    """Append `horizon` future days per series with the KNOWN inputs filled in
    (last regular price, planned promos) and unknown demand left as NaN.
    `promo_plan` (store_id, sku_id, date, discount_pct) comes from the commercial team."""
    last = silver.sort_values("date").groupby(["store_id", "sku_id"]).tail(1)
    dates = pd.date_range(silver["date"].max() + pd.Timedelta(days=1), periods=horizon)
    fut = last[["store_id", "sku_id", "regular_price"]].merge(pd.DataFrame({"date": dates}), how="cross")
    fut["discount_pct"] = 0.0
    if promo_plan is not None and len(promo_plan):
        fut = fut.drop(columns="discount_pct").merge(promo_plan, on=["store_id", "sku_id", "date"], how="left")
        fut["discount_pct"] = fut["discount_pct"].fillna(0.0)
    fut["on_promo"] = (fut["discount_pct"] > 0).astype(int)
    fut["selling_price"] = fut["regular_price"] * (1 - fut["discount_pct"])
    fut["n_sibling_promo"] = 0
    if products is not None and fut["on_promo"].any():
        f2 = fut.merge(products[["sku_id", "category"]], on="sku_id", how="left")
        tot = f2.groupby(["store_id", "category", "date"])["on_promo"].transform("sum")
        fut["n_sibling_promo"] = (tot - f2["on_promo"]).values
    for c in ["units_sold", "true_demand", "demand_target"]:
        fut[c] = np.nan
    fut["stockout_flag"] = 0
    out = pd.concat([silver, fut], ignore_index=True).sort_values(["store_id", "sku_id", "date"])
    out["demand_filled"] = out["demand_filled"].fillna(0)  # only feeds lags >= HORIZON -> never used
    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "units_sold", "true_demand", "demand_target", "demand_filled",
               "stockout_flag", "regular_price", "unit_cost"}
    return [c for c in df.columns if c not in exclude]
