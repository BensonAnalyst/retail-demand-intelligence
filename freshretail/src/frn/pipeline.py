"""End-to-end experiment shared by the Databricks notebooks, the Kaggle notebook and CI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecast, recovery
from .data import KEY, OP_HOURS, Panel, add_day_index, check_oos_orientation, concat

TARGETS = {"observed": "raw sales (baseline)", "rec_profile": "recovered: profile scaling",
           "rec_model": "recovered: self-supervised LightGBM"}


def prepare(train: Panel, eval_: Panel) -> tuple[Panel, int, dict]:
    chk = check_oos_orientation(train)
    if chk["flipped"]:
        eval_.oos = (1 - eval_.oos).astype(np.int8)
    last_train_day = add_day_index(train, eval_)
    full = concat(train, eval_)
    order = np.lexsort((full.df["day"].to_numpy(), full.df["product_id"].to_numpy(), full.df["store_id"].to_numpy()))
    return full.take(order), last_train_day, chk


def sale_groups(full: Panel, cutoff: int) -> np.ndarray:
    daily = pd.Series(full.sales.sum(1))
    hist = daily.where(full.df["day"] <= cutoff)
    mean = hist.groupby([full.df[k] for k in KEY]).transform("mean")
    return np.where(mean >= mean.median(), "high-sale", "low-sale")


def run_fold(full: Panel, cutoff: int, with_model: bool = True, log=print):
    rec = recovery.recover(full, cutoff, with_model=with_model)
    observed = rec["observed"].to_numpy()
    after = full.df["day"].to_numpy() > cutoff
    clean = recovery.clean_days(full)
    groups = sale_groups(full, cutoff)
    # censoring-aware truth for the test days: observed on in-stock days, recovered otherwise
    truth_col = "rec_model" if "rec_model" in rec else "rec_profile"
    recovered_truth = rec[truth_col].to_numpy()
    rows, preds = [], {}
    for col, label in TARGETS.items():
        if col not in rec:
            continue
        target = np.where(after, observed, rec[col].to_numpy())     # test days: never used as labels
        frame = forecast.build_frame(full.df, full.oos, target)
        # frame is re-sorted by key/day; full is already in that order
        idx, pred, _ = forecast.fit_predict(frame, cutoff)
        actual = observed[idx]
        for r in forecast.score(actual, pred, clean[idx], groups[idx], recovered_truth[idx],
                                rec["rec_profile"].to_numpy()[idx]):
            rows.append({"cutoff_day": cutoff, "training_target": label, **r})
        preds[col] = pd.DataFrame({"row": idx, "pred": pred})
        log(f"cutoff {cutoff} | {label}: done")
    return pd.DataFrame(rows), rec, preds


def lost_demand_tables(full: Panel, rec: pd.DataFrame, cutoff: int, col: str = "rec_model"):
    col = col if col in rec else "rec_profile"
    d = full.df[["store_id", "first_category_id", "third_category_id", "day"]].copy()
    d["observed"] = rec["observed"].to_numpy()
    d["recovered"] = rec[col].to_numpy()
    d["oos_hours"] = full.oos[:, OP_HOURS].sum(1)
    d = d[d["day"] <= cutoff]
    d["lost"] = d["recovered"] - d["observed"]
    by_cat = (d.groupby("first_category_id")
                .agg(recovered=("recovered", "sum"), lost=("lost", "sum"), oos_hours=("oos_hours", "mean"))
                .assign(lost_share=lambda x: x.lost / x.recovered, availability=lambda x: 1 - x.oos_hours / 17)
                .reset_index().sort_values("lost", ascending=False))
    by_store = (d.groupby("store_id")
                  .agg(recovered=("recovered", "sum"), lost=("lost", "sum"), oos_hours=("oos_hours", "mean"))
                  .assign(lost_share=lambda x: x.lost / x.recovered, availability=lambda x: 1 - x.oos_hours / 17)
                  .reset_index().sort_values("lost", ascending=False))
    hours = full.oos[(full.df["day"] <= cutoff).to_numpy()][:, OP_HOURS].mean(0)
    by_hour = pd.DataFrame({"hour": range(6, 23), "oos_rate": hours})
    total = {"lost_share_of_demand": float(d["lost"].sum() / d["recovered"].sum()),
             "stockout_day_share": float((d["oos_hours"] > 0).mean()),
             "availability_operating_hours": float(1 - d["oos_hours"].mean() / 17)}
    return total, by_cat, by_store, by_hour
