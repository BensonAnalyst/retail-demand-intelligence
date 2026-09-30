"""Pricing & promotion analytics.

1. Elasticity + promo uplift + cannibalisation per category via a Poisson GLM
   with store-SKU fixed effects:
      log E[q] = a_series + b_sku*t + e*log(price) + g*on_promo + d*n_sibling_promo + dow + month + holiday
   -> e  = own-price elasticity
   -> exp(g) = promo display/feature uplift beyond the price cut
   -> exp(d) = demand retained per promoted sibling (1 - exp(d) = cannibalisation)
   Poisson (not log-OLS) because retail demand has zeros and log(0) is undefined.
   The SKU-level trend b_sku*t matters: regular prices creep up with inflation while
   some SKUs grow and others decline. Without it, the trend leaks into the price
   coefficient (on this dataset the naive Snacks elasticity is about 40% too small).
2. Promo depth economics: incremental margin of a promo INCLUDING cannibalisation.
3. Margin-optimal regular price within guardrails (+/- 10%) for constant elasticity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_category_response(df: pd.DataFrame, category: str, control_trend: bool = True) -> dict:
    import statsmodels.api as sm
    d = df[(df["category"].astype(str) == category) & df["demand_target"].notna()].copy()
    d["series"] = d["store_id"].astype(str) + "_" + d["sku_id"].astype(str)
    d["log_price"] = np.log(d["selling_price"])
    # within-series price variation only (fixed effects) -> removes "expensive SKUs sell less" confounding
    d["log_price_dm"] = d["log_price"] - d.groupby("series")["log_price"].transform("mean")
    d["holiday"] = (d["days_to_holiday"] <= 3).astype(int)
    X = pd.get_dummies(d[["series", "dow", "month"]].astype(str), drop_first=True, dtype=float)
    X[["log_price_dm", "on_promo", "n_sibling_promo", "holiday"]] = d[["log_price_dm", "on_promo", "n_sibling_promo", "holiday"]].astype(float)
    if control_trend:
        t = ((d["date"] - d["date"].min()).dt.days / 365.0).to_numpy()
        T = pd.get_dummies(d["sku_id"].astype(str), dtype=float).mul(t, axis=0)
        T.columns = [f"trend_{c}" for c in T.columns]
        X = pd.concat([X, T], axis=1)
    X = sm.add_constant(X)
    res = sm.GLM(d["demand_target"].astype(float), X, family=sm.families.Poisson()).fit(cov_type="HC1")
    ci = res.conf_int()
    return {
        "category": category, "control_trend": control_trend,
        "elasticity": res.params["log_price_dm"],
        "elasticity_lo": ci.loc["log_price_dm", 0], "elasticity_hi": ci.loc["log_price_dm", 1],
        "promo_uplift": float(np.exp(res.params["on_promo"])),
        "promo_uplift_lo": float(np.exp(ci.loc["on_promo", 0])), "promo_uplift_hi": float(np.exp(ci.loc["on_promo", 1])),
        "cannibalisation_per_sibling": float(1 - np.exp(res.params["n_sibling_promo"])),
        "n_obs": int(len(d)),
    }


def promo_depth_economics(resp: pd.Series, price: float, cost: float, base_units: float,
                          sibling_margin_units: float, avg_siblings_hit: float,
                          depths=(0.1, 0.15, 0.2, 0.25, 0.3)) -> pd.DataFrame:
    """Incremental gross margin per promo-day for one representative SKU-store."""
    rows = []
    base_margin = (price - cost) * base_units
    for d in depths:
        p_promo = price * (1 - d)
        units = base_units * resp["promo_uplift"] * (1 - d) ** resp["elasticity"]
        promo_margin = (p_promo - cost) * units
        cannibal_loss = resp["cannibalisation_per_sibling"] * sibling_margin_units * avg_siblings_hit
        rows.append({"depth": d, "units_lift_x": units / base_units,
                     "incremental_margin": promo_margin - base_margin - cannibal_loss,
                     "cannibal_loss": cannibal_loss,
                     "promo_margin_pct": (p_promo - cost) / p_promo})
    return pd.DataFrame(rows)


def optimal_price(p0: float, cost: float, elasticity: float, band: float = 0.10) -> float:
    """Margin-max price for q = q0 (p/p0)^e, clipped to +/- band (brand/competitor guardrail)."""
    if elasticity < -1:
        p_star = cost * elasticity / (1 + elasticity)
    else:  # inelastic -> theory says raise price without bound; guardrail caps it
        p_star = p0 * (1 + band)
    return float(np.clip(p_star, p0 * (1 - band), p0 * (1 + band)))


def price_recommendations(skus: pd.DataFrame, elasticities: pd.DataFrame, band: float = 0.10) -> pd.DataFrame:
    """skus: sku_id, category, regular_price, unit_cost, daily_units (network)."""
    d = skus.merge(elasticities[["category", "elasticity"]], on="category")
    d["rec_price"] = [optimal_price(p, c, e, band) for p, c, e in zip(d.regular_price, d.unit_cost, d.elasticity)]
    d["price_change_pct"] = d.rec_price / d.regular_price - 1
    d["new_daily_units"] = d.daily_units * (d.rec_price / d.regular_price) ** d.elasticity
    d["margin_now"] = (d.regular_price - d.unit_cost) * d.daily_units
    d["margin_new"] = (d.rec_price - d.unit_cost) * d.new_daily_units
    d["annual_margin_gain"] = (d.margin_new - d.margin_now) * 365
    return d.sort_values("annual_margin_gain", ascending=False)
