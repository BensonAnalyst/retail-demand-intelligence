"""Bronze -> Silver: data-quality rules + demand segmentation.

Every rule returns counts so the notebook can publish a DQ scorecard (the
"validation rules" the JD asks a DS to define for engineering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["store_id", "sku_id", "date"]


def clean_sales(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dq: dict[str, int] = {"rows_in": len(sales)}
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])

    # R1 exact duplicates from double-loads
    before = len(df)
    df = df.drop_duplicates(subset=KEYS, keep="first")
    dq["r1_duplicates_removed"] = before - len(df)

    # R2 negative units = returns netted into POS -> not demand. Set to NaN, impute later.
    neg = df["units_sold"] < 0
    dq["r2_negative_units"] = int(neg.sum())
    df.loc[neg, "units_sold"] = np.nan

    # R3 price outliers: > 3x or < 0.3x the series median regular price -> replace with regular price
    med = df.groupby(["store_id", "sku_id"])["regular_price"].transform("median")
    bad_price = (df["selling_price"] > 3 * med) | (df["selling_price"] < 0.3 * med)
    dq["r3_price_outliers"] = int(bad_price.sum())
    df.loc[bad_price, "selling_price"] = df.loc[bad_price, "regular_price"] * (1 - df.loc[bad_price, "discount_pct"])

    # R4 complete the calendar (missing POS days) -> explicit rows with NaN target
    full = (df[["store_id", "sku_id"]].drop_duplicates()
            .merge(pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())}), how="cross"))
    df = full.merge(df, on=KEYS, how="left")
    dq["r4_missing_days_filled"] = int(df["units_sold"].isna().sum() - dq["r2_negative_units"])
    df = df.sort_values(KEYS).reset_index(drop=True)
    g = df.groupby(["store_id", "sku_id"])
    for c in ["regular_price", "selling_price"]:
        df[c] = g[c].transform(lambda s: s.ffill().bfill())
    for c in ["on_promo", "discount_pct", "stockout_flag", "n_sibling_promo"]:
        df[c] = df[c].fillna(0)

    # R5 stock-out censoring: sales on stock-out days understate demand.
    #     Target = NaN (excluded from training + evaluation); model input = imputed value.
    df["demand_target"] = df["units_sold"].where(df["stockout_flag"] == 0)
    dq["r5_censored_stockout_days"] = int((df["stockout_flag"] == 1).sum())

    # Imputed series for lag features: same-weekday median of previous 4 weeks, else 0
    df["dow"] = df["date"].dt.dayofweek
    df["demand_filled"] = df["demand_target"]
    fill = (df.groupby(["store_id", "sku_id", "dow"])["demand_target"]
              .transform(lambda s: s.shift(1).rolling(4, min_periods=1).median()))
    df["demand_filled"] = df["demand_filled"].fillna(fill).fillna(0)
    df = df.drop(columns=["dow"])
    dq["rows_out"] = len(df)
    return df, dq


def classify_demand(df: pd.DataFrame, value_col: str = "demand_target") -> pd.DataFrame:
    """Syntetos-Boylan-Croston classification using ADI and CV^2 of non-zero demand."""
    def _cls(s: pd.Series) -> pd.Series:
        s = s.dropna()
        nz = s[s > 0]
        adi = len(s) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        if adi < 1.32 and cv2 < 0.49:
            c = "smooth"
        elif adi < 1.32:
            c = "erratic"
        elif cv2 < 0.49:
            c = "intermittent"
        else:
            c = "lumpy"
        return pd.Series({"adi": adi, "cv2": cv2, "mean_demand": s.mean(), "zero_share": (s == 0).mean(), "segment": c})
    return df.groupby(["store_id", "sku_id"])[value_col].apply(_cls).unstack().reset_index()
