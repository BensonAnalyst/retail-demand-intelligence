"""End-to-end experiment shared by the Databricks notebooks, the Kaggle notebook and CI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecast, recovery
from .data import KEY, N_OP_HOURS, OP_HOURS, Panel, add_day_index, check_oos_orientation, concat

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


def run_fold(full: Panel, cutoff: int, with_model: bool = True, log=print, scales: dict | None = None):
    """Train the 3 forecasters (raw / profile / LightGBM-recovered targets) on days <= cutoff and score
    days cutoff+1..cutoff+7. If `scales` is given ({target col: multiplier}, learned on the PREVIOUS
    week only), also scores the bias-calibrated forecast, labelled '<target> + calibration'.
    Returns (scores, rec, preds, next_scales): next_scales are learned on THIS test week, for the next fold."""
    rec = recovery.recover(full, cutoff, with_model=with_model)
    observed = rec["observed"].to_numpy()
    after = full.df["day"].to_numpy() > cutoff
    clean = recovery.clean_days(full)
    groups = sale_groups(full, cutoff)
    # censoring-aware truth for the test days: observed on in-stock days, recovered otherwise
    truth_col = "rec_model" if "rec_model" in rec else "rec_profile"
    recovered_truth = rec[truth_col].to_numpy()
    rows, preds, next_scales = [], {}, {}
    for col, label in TARGETS.items():
        if col not in rec:
            continue
        target = np.where(after, observed, rec[col].to_numpy())     # test days: never used as labels
        frame = forecast.build_frame(full.df, full.oos, target)
        # frame is re-sorted by key/day; full is already in that order
        idx, pred, _ = forecast.fit_predict(frame, cutoff)
        actual = observed[idx]
        variants = {label: pred}
        if scales and col in scales:
            variants[f"{label} + calibration"] = pred * scales[col]
        for name, p_ in variants.items():
            for r in forecast.score(actual, p_, clean[idx], groups[idx], recovered_truth[idx],
                                    rec["rec_profile"].to_numpy()[idx]):
                rows.append({"cutoff_day": cutoff, "training_target": name, **r})
            preds[name] = pd.DataFrame({"row": idx, "pred": p_, "horizon": full.df["day"].to_numpy()[idx] - cutoff})
        # what this target "should" sum to this week, judged by its own definition of demand
        own_truth = rec[col].to_numpy()[idx]
        next_scales[col] = float(own_truth.sum() / max(pred.sum(), 1e-9))
        log(f"cutoff {cutoff} | {label}: done (own-target ratio {next_scales[col]:.3f})")
    return pd.DataFrame(rows), rec, preds, next_scales


def horizon_table(full: Panel, rec: pd.DataFrame, preds: dict, cutoff: int) -> pd.DataFrame:
    """Bias and WAPE by days ahead (1..7), scored against recovered demand (view C)."""
    from .metrics import summary
    truth = rec["rec_model" if "rec_model" in rec else "rec_profile"].to_numpy()
    out = []
    for name, d in preds.items():
        for h, g in d.groupby("horizon"):
            out.append({"cutoff_day": cutoff, "training_target": name, "horizon_days": int(h),
                        **summary(truth[g["row"].to_numpy()], g["pred"].to_numpy())})
    return pd.DataFrame(out)


def run_backtest(full: Panel, last_day: int, with_model: bool = True, log=print):
    """Walk-forward backtest: a warm-up week (cutoff last_day-14) is used ONLY to learn the bias
    calibration for the first scored fold; each scored fold then passes its own ratio to the next.
    No fold ever uses information from its own test week. Returns (scores, horizon, scales, rec)."""
    _, _, _, scales = run_fold(full, last_day - 14, with_model, log)
    log(f"calibration learned on warm-up week: { {k: round(v, 3) for k, v in scales.items()} }")
    results, horizons, used = [], [], []
    for cutoff in (last_day - 7, last_day):
        res, rec, preds, nxt = run_fold(full, cutoff, with_model, log, scales=scales)
        results.append(res)
        horizons.append(horizon_table(full, rec, preds, cutoff))
        used += [{"cutoff_day": cutoff, "target": k, "multiplier": v} for k, v in scales.items()]
        scales = nxt
    return (pd.concat(results, ignore_index=True), pd.concat(horizons, ignore_index=True),
            pd.DataFrame(used), rec)


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
                .assign(lost_share=lambda x: x.lost / x.recovered, availability=lambda x: 1 - x.oos_hours / N_OP_HOURS)
                .reset_index().sort_values("lost", ascending=False))
    by_store = (d.groupby("store_id")
                  .agg(recovered=("recovered", "sum"), lost=("lost", "sum"), oos_hours=("oos_hours", "mean"))
                  .assign(lost_share=lambda x: x.lost / x.recovered, availability=lambda x: 1 - x.oos_hours / N_OP_HOURS)
                  .reset_index().sort_values("lost", ascending=False))
    hours = full.oos[(full.df["day"] <= cutoff).to_numpy()][:, OP_HOURS].mean(0)
    by_hour = pd.DataFrame({"hour": range(OP_HOURS.start, OP_HOURS.stop), "oos_rate": hours})
    total = {"lost_share_of_demand": float(d["lost"].sum() / d["recovered"].sum()),
             "stockout_day_share": float((d["oos_hours"] > 0).mean()),
             "availability_operating_hours": float(1 - d["oos_hours"].mean() / N_OP_HOURS)}
    return total, by_cat, by_store, by_hour
