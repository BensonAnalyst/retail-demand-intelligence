"""Latent demand recovery: estimate what customers WANTED on days the shelf ran empty.

Observed daily sales on a stock-out day = demand in the in-stock hours only, so
training a forecaster on raw sales teaches it to under-forecast (the paper reports
-7.37% bias). Three estimators, from simplest to strongest:

  observed      sales as recorded (the biased baseline)
  profile       scale the in-stock hours up by the share of a normal day's demand
                they usually carry:  demand ~= observed / sum(profile[in-stock hours])
  model         LightGBM trained SELF-SUPERVISED: take fully in-stock days, hide hours
                using real stock-out patterns copied from real stock-out days, and
                learn to predict the full day from the partial day

How we know recovery works without ever seeing true demand: the same masking trick
on held-out series gives days where the "lost" hours are hidden but known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY, OP_HOURS, Panel
from .metrics import summary

PROFILE_LEVELS = [KEY, ["store_id", "third_category_id"], ["third_category_id"]]
MIN_SHARE = 0.15       # below this in-stock share, the profile ratio is too unstable


def clean_days(p: Panel) -> np.ndarray:
    return p.oos[:, OP_HOURS].sum(1) == 0


# ------------------------------------------------------------------ hourly profiles
def hourly_profiles(p: Panel, fit_mask: np.ndarray, min_days: int = 5) -> np.ndarray:
    """(n, 24) expected share of the day's demand in each hour, per row, learned from
    clean days only. Falls back store-product -> store-category -> category -> global."""
    n = len(p.df)
    prof = np.full((n, 24), np.nan)
    for level in PROFILE_LEVELS:
        g = p.df.groupby(level, sort=False).ngroup().to_numpy()
        G = g.max() + 1
        hsum = np.zeros((G, 24))
        np.add.at(hsum, g[fit_mask], p.sales[fit_mask])
        cnt = np.bincount(g[fit_mask], minlength=G)
        tot = hsum.sum(1, keepdims=True)
        ok = (cnt >= min_days) & (tot[:, 0] > 0)
        gp = np.divide(hsum, tot, out=np.zeros_like(hsum), where=tot > 0)
        need = np.isnan(prof[:, 0]) & ok[g]
        prof[need] = gp[g[need]]
    glob = p.sales[fit_mask].sum(0)
    glob = glob / glob.sum() if glob.sum() > 0 else np.full(24, 1 / 24)
    prof[np.isnan(prof[:, 0])] = glob
    return prof


def series_level(p: Panel, fit_mask: np.ndarray) -> np.ndarray:
    """Mean daily sales of each series over its clean days (leave-one-out for clean rows)."""
    g = p.df.groupby(KEY, sort=False).ngroup().to_numpy()
    daily = p.sales.sum(1)
    s = np.bincount(g[fit_mask], weights=daily[fit_mask], minlength=g.max() + 1)
    c = np.bincount(g[fit_mask], minlength=g.max() + 1).astype(float)
    num = s[g] - np.where(fit_mask, daily, 0)
    den = c[g] - fit_mask.astype(float)
    lvl = np.divide(num, den, out=np.full(len(g), np.nan), where=den > 0)
    return np.where(np.isnan(lvl), np.nanmean(daily[fit_mask]) if fit_mask.any() else 0.0, lvl)


def profile_recover(sales: np.ndarray, oos: np.ndarray, prof: np.ndarray, level: np.ndarray) -> np.ndarray:
    observed = sales.sum(1)
    share_in = (prof * (1 - oos)).sum(1)
    est = np.where(share_in >= MIN_SHARE, observed / np.maximum(share_in, 1e-9), level)
    est = np.where(oos[:, OP_HOURS].sum(1) == 0, observed, est)      # clean days unchanged
    return np.maximum(observed, est)


# ------------------------------------------------------------------ self-supervised model
def _mask_features(p: Panel, sales: np.ndarray, oos: np.ndarray, prof: np.ndarray, level: np.ndarray) -> pd.DataFrame:
    op = oos[:, OP_HOURS]
    any_oos = op.any(1)
    first = np.where(any_oos, op.argmax(1) + 6, -1)
    last = np.where(any_oos, 22 - op[:, ::-1].argmax(1), -1)
    share_in = (prof * (1 - oos)).sum(1)
    observed = sales.sum(1)
    X = pd.DataFrame({
        "observed": observed,
        "share_in": share_in,
        "profile_est": np.where(share_in >= 0.05, observed / np.maximum(share_in, 1e-9), level),
        "level": level,
        "oos_hours": op.sum(1),
        "first_oos_hour": first,
        "last_oos_hour": last,
        "dow": p.df["dt"].dt.dayofweek.to_numpy(),
        "discount": p.df["discount"].to_numpy(),
        "holiday_flag": p.df["holiday_flag"].to_numpy(),
        "activity_flag": p.df["activity_flag"].to_numpy(),
        "avg_temperature": p.df["avg_temperature"].to_numpy(),
        "precpt": p.df["precpt"].to_numpy(),
        "third_category_id": p.df["third_category_id"].astype("category").to_numpy(),
    })
    return X


def transplant_masks(p: Panel, target_rows: np.ndarray, donor_rows: np.ndarray, rng) -> np.ndarray:
    """Copy real stock-out hour patterns (from donor stock-out days) onto clean target days."""
    picks = rng.choice(donor_rows, size=len(target_rows), replace=True)
    return p.oos[picks].copy()


class RecoveryModel:
    def __init__(self, seed: int = 42, params: dict | None = None):
        self.seed = seed
        self.params = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.05,
                           num_leaves=63, min_child_samples=100, n_estimators=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
        self.params.update(params or {})

    def _synth(self, p, rows, donors, prof, level, rng):
        m = transplant_masks(p, rows, donors, rng)
        s = p.sales[rows] * (1 - m)              # hide the "stock-out" hours
        sub = p.take(rows)
        X = _mask_features(sub, s, m, prof[rows], level[rows])
        y = p.sales[rows].sum(1)                 # the truth we hid
        keep = m[:, OP_HOURS].any(1)
        return X[keep], y[keep], s[keep], m[keep], rows[keep]

    def fit(self, p: Panel, fit_mask: np.ndarray, prof, level):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        clean = np.where(clean_days(p) & fit_mask)[0]
        donors = np.where(~clean_days(p) & fit_mask)[0]
        X, y, *_ = self._synth(p, clean, donors, prof, level, rng)
        self.model = lgb.LGBMRegressor(**self.params).fit(X, y, categorical_feature=["third_category_id"])
        return self

    def predict(self, p: Panel, prof, level) -> np.ndarray:
        X = _mask_features(p, p.sales, p.oos, prof, level)
        est = np.clip(self.model.predict(X), 0, None)
        observed = p.sales.sum(1)
        est = np.where(clean_days(p), observed, est)
        return np.maximum(observed, est)


# ------------------------------------------------------------------ orchestration
def recover(p: Panel, cutoff_day: int, seed: int = 42, with_model: bool = True) -> pd.DataFrame:
    """Recovered daily demand for every row with day <= cutoff. Columns:
    observed, rec_profile, rec_model (if with_model)."""
    fit = (p.df["day"].to_numpy() <= cutoff_day)
    prof = hourly_profiles(p, clean_days(p) & fit)
    level = series_level(p, clean_days(p) & fit)
    out = pd.DataFrame({"observed": p.sales.sum(1)})
    out["rec_profile"] = profile_recover(p.sales, p.oos, prof, level)
    if with_model:
        m = RecoveryModel(seed).fit(p, fit, prof, level)
        out["rec_model"] = m.predict(p, prof, level)
    return out


def validate_recovery(p: Panel, cutoff_day: int, holdout_frac: float = 0.2, seed: int = 7) -> pd.DataFrame:
    """Masking test on held-out SERIES: hide hours on their clean days using real stock-out
    patterns, recover, and score against the hidden truth (WAPE, bias=WPE, VN1)."""
    rng = np.random.default_rng(seed)
    g = p.df.groupby(KEY, sort=False).ngroup().to_numpy()
    hold_series = rng.choice(np.unique(g), size=max(1, int(holdout_frac * (g.max() + 1))), replace=False)
    hold = np.isin(g, hold_series)
    fit = (p.df["day"].to_numpy() <= cutoff_day)
    clean = clean_days(p)
    prof = hourly_profiles(p, clean & fit & ~hold)
    level = series_level(p, clean & fit)              # a series' own normal level is legitimately known
    model = RecoveryModel(seed).fit(p, fit & ~hold, prof, level)

    rows = np.where(clean & fit & hold)[0]
    donors = np.where(~clean & fit)[0]
    X, y, s_masked, m, kept = model._synth(p, rows, donors, prof, level, np.random.default_rng(seed + 1))
    estimates = {
        "observed (no recovery)": s_masked.sum(1),
        "profile scaling": profile_recover(s_masked, m, prof[kept], level[kept]),
        "self-supervised LightGBM": np.maximum(s_masked.sum(1), np.clip(model.model.predict(X), 0, None)),
    }
    res = pd.DataFrame([{"method": k, **summary(y, v)} for k, v in estimates.items()])
    res["n_masked_days"] = len(y)
    res["avg_hidden_share"] = 1 - s_masked.sum() / max(y.sum(), 1e-9)
    return res
