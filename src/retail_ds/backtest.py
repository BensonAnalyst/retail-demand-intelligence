"""Rolling-origin (expanding-window) backtesting.

   train ........|cutoff1| 28d test
   train ...............|cutoff2| 28d test
   train ......................|cutoff3| 28d test     (last fold covers 11.11 / 12.12)

Rules that keep it honest:
  * a model only ever sees data <= cutoff (features are shifted >= HORIZON)
  * stock-out days are excluded from scoring (sales there are censored, not demand)
  * the same folds + same metric code for every model -> apples to apples
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZON
from .models import GlobalLGBM, forecast_one_series


def default_cutoffs(last_date: pd.Timestamp, n_folds: int = 4, horizon: int = HORIZON) -> list[pd.Timestamp]:
    last = pd.Timestamp(last_date)
    return [last - pd.Timedelta(days=horizon * k) for k in range(n_folds, 0, -1)]


def run_backtest(gold: pd.DataFrame, features: list[str], cutoffs: list[pd.Timestamp],
                 horizon: int = HORIZON, stat_models: bool = True, lgbm_params: dict | None = None,
                 log=print) -> tuple[pd.DataFrame, dict]:
    all_fc, fitted = [], {}
    for k, cut in enumerate(cutoffs, 1):
        end = cut + pd.Timedelta(days=horizon)
        test_mask = (gold["date"] > cut) & (gold["date"] <= end)
        # --- global ML
        m = GlobalLGBM(features, lgbm_params).fit(gold[gold["date"] <= cut])
        te = gold.loc[test_mask, ["store_id", "sku_id", "date"]].copy()
        te["model"], te["forecast"] = "lgbm_global", m.predict(gold.loc[test_mask])
        te["fold"] = k
        all_fc.append(te)
        fitted[k] = m
        # --- per-series stats
        if stat_models:
            hist = gold[gold["date"] <= cut][["store_id", "sku_id", "date", "demand_filled", "segment"]]
            parts = [forecast_one_series(g, cut, horizon) for _, g in hist.groupby(["store_id", "sku_id"], observed=True)]
            st = pd.concat(parts, ignore_index=True)
            st["fold"] = k
            all_fc.append(st)
        log(f"fold {k}: cutoff {cut.date()} -> test {cut.date() + pd.Timedelta(days=1)}..{end.date()}")

    fc = pd.concat(all_fc, ignore_index=True)
    for c in ["store_id", "sku_id"]:
        fc[c] = fc[c].astype(str)
    # simple ensemble: ML + stats (diversified errors usually cancel)
    if stat_models:
        piv = fc[fc.model.isin(["lgbm_global", "ets"])].pivot_table(
            index=["store_id", "sku_id", "date", "fold"], columns="model", values="forecast").reset_index()
        piv["forecast"] = 0.7 * piv["lgbm_global"] + 0.3 * piv["ets"]
        piv["model"] = "ensemble_70_30"
        fc = pd.concat([fc, piv[["store_id", "sku_id", "date", "fold", "model", "forecast"]]], ignore_index=True)

    act = gold[["store_id", "sku_id", "date", "demand_target", "true_demand", "segment", "category",
                "on_promo", "selling_price", "unit_cost"]].copy()
    for c in ["store_id", "sku_id", "segment", "category"]:
        act[c] = act[c].astype(str)
    fc = fc.merge(act, on=["store_id", "sku_id", "date"], how="left").rename(columns={"demand_target": "actual"})
    return fc, fitted


def scored(fc: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for scoring (non-censored actuals)."""
    return fc[fc["actual"].notna()]
