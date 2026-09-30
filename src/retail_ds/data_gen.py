"""Synthetic but realistic retail demand generator.

Why synthetic? It lets the project run anywhere (Databricks Free Edition, no
Kaggle credentials) AND it gives us a known ground truth: true price elasticity,
true promo uplift and true cannibalisation are baked in, so we can check that the
estimators in `pricing.py` recover them. Swap in M5 / VN1 data via
`notebooks/01_bronze_ingest.py` when you want real-world noise.

Signals deliberately embedded (all things a retail DS has to untangle):
  * weekly + yearly seasonality, SKU-level trend
  * SG/MY holiday spikes (CNY, Hari Raya, Deepavali, 11.11, 12.12, Christmas)
  * price elasticity that differs by category
  * promotions with uplift + within-category cannibalisation
  * smooth / erratic / intermittent / lumpy demand patterns
  * stock-outs that CENSOR sales (sales < true demand)
  * data-quality defects: missing days, duplicates, negative returns, price typos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    # category: (price range, elasticity, promo uplift multiplier)
    "Beverages": ((1.5, 4.0), -2.2, 1.9),
    "Snacks": ((2.0, 6.0), -1.8, 1.7),
    "Household": ((4.0, 15.0), -1.3, 1.5),
    "Personal Care": ((5.0, 20.0), -1.1, 1.4),
}
REGIONS = ["Central", "East", "West"]

HOLIDAYS = {
    # date: demand multiplier (applied to the day and a 3-day pre-holiday ramp)
    "2024-02-10": 1.6, "2025-01-29": 1.6,  # Chinese New Year
    "2024-04-10": 1.3, "2025-03-31": 1.3,  # Hari Raya Puasa
    "2024-10-31": 1.2, "2025-10-20": 1.2,  # Deepavali
    "2024-11-11": 1.35, "2025-11-11": 1.35,  # 11.11
    "2024-12-12": 1.25, "2025-12-12": 1.25,  # 12.12
    "2024-12-25": 1.4, "2025-12-25": 1.4,  # Christmas
    # 2026 entries are used only for forward-scoring features (days_to_holiday)
    "2026-02-17": 1.6, "2026-03-21": 1.3, "2026-11-08": 1.2,
    "2026-11-11": 1.35, "2026-12-12": 1.25, "2026-12-25": 1.4,
}
CANNIBALISATION = 0.025  # each promoted sibling SKU steals ~2.5% of demand


def holiday_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    mult = pd.Series(1.0, index=dates)
    flag = pd.Series(0, index=dates)
    for d, m in HOLIDAYS.items():
        d = pd.Timestamp(d)
        for lag, w in [(0, 1.0), (1, 0.8), (2, 0.5), (3, 0.3)]:
            day = d - pd.Timedelta(days=lag)
            if day in mult.index:
                mult[day] = max(mult[day], 1 + (m - 1) * w)
                flag[day] = 1
    return pd.DataFrame({"date": dates, "holiday_mult": mult.values, "is_holiday_window": flag.values})


def generate(
    n_stores: int = 10,
    n_skus: int = 40,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    inject_dq_issues: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return bronze-style tables: sales, products, stores, promo_calendar, holidays."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)

    # ---------------- master data ----------------
    cats = list(CATEGORIES)
    products = []
    for i in range(n_skus):
        cat = cats[i % len(cats)]
        (lo, hi), elas, uplift = CATEGORIES[cat]
        price = round(float(rng.uniform(lo, hi)), 2)
        # demand pattern mix: 45% smooth, 20% erratic, 20% intermittent, 15% lumpy
        pattern = rng.choice(["smooth", "erratic", "intermittent", "lumpy"], p=[0.45, 0.2, 0.2, 0.15])
        products.append({
            "sku_id": f"SKU{i:03d}", "category": cat, "base_price": price,
            "unit_cost": round(price * rng.uniform(0.55, 0.7), 2),
            "true_elasticity": elas + rng.normal(0, 0.15),
            "true_promo_uplift": uplift * rng.uniform(0.9, 1.1),
            "pattern": pattern,
        })
    products = pd.DataFrame(products)

    stores = pd.DataFrame({
        "store_id": [f"S{j:02d}" for j in range(n_stores)],
        "region": [REGIONS[j % len(REGIONS)] for j in range(n_stores)],
        "store_size": rng.choice([0.7, 1.0, 1.4], size=n_stores, p=[0.3, 0.5, 0.2]),
    })

    hol = holiday_calendar(dates)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    t = np.arange(T)

    # ---------------- promo calendar (SKU x store x date) ----------------
    # promos run in 7-14 day blocks, ~12% of days, deeper around 11.11/12.12
    promo = np.zeros((n_skus, n_stores, T), dtype=np.int8)
    discount = np.zeros((n_skus, T))
    for i in range(n_skus):
        n_events = rng.integers(6, 11)
        for _ in range(n_events):
            s = rng.integers(0, T - 14)
            L = rng.integers(7, 15)
            depth = rng.choice([0.1, 0.15, 0.2, 0.25, 0.3])
            participating = rng.random(n_stores) < 0.8
            promo[i, participating, s:s + L] = 1
            discount[i, s:s + L] = depth

    # regular price drifts (inflation + occasional repricing) -> gives elasticity signal
    rows = []
    for i, p in products.iterrows():
        cat_mask = (products["category"] == p.category).values
        cat_idx = np.where(cat_mask)[0]
        reprice = np.exp(np.cumsum(rng.normal(0, 0.004, T)))  # slow random walk
        reg_price = p.base_price * reprice * (1 + 0.03 * (t / T))  # +3% inflation over period

        # demand pattern params
        base = {"smooth": rng.uniform(15, 40), "erratic": rng.uniform(8, 25),
                "intermittent": rng.uniform(0.3, 1.2), "lumpy": rng.uniform(0.3, 1.0)}[p.pattern]
        trend = rng.normal(0, 0.25) / T
        weekly = 1 + np.array([-0.08, -0.1, -0.05, 0.0, 0.12, 0.22, 0.15])[dow] * (1 if p.category in ("Beverages", "Snacks") else 0.5)
        yearly = 1 + 0.12 * np.sin(2 * np.pi * (doy - 60) / 365.25)

        for j, st in stores.iterrows():
            is_promo = promo[i, j]
            price = reg_price * (1 - discount[i] * is_promo)
            # sibling promos in the same store/category -> cannibalisation
            siblings = [k for k in cat_idx if k != i]
            n_sib_promo = promo[siblings, j, :].sum(axis=0) if siblings else np.zeros(T)
            lam = (base * st.store_size * (1 + trend * t) * weekly * yearly * hol["holiday_mult"].values
                   * (price / p.base_price) ** p.true_elasticity
                   # display / feature effect ON TOP of the price-cut effect already in `price`
                   * np.where(is_promo == 1, p.true_promo_uplift, 1.0)
                   * (1 - CANNIBALISATION) ** n_sib_promo)
            if p.pattern == "smooth":
                demand = rng.poisson(lam)
            elif p.pattern == "erratic":
                demand = rng.negative_binomial(2, 2 / (2 + lam))
            elif p.pattern == "intermittent":
                demand = rng.poisson(lam * 3) * (rng.random(T) < 0.3)
            else:  # lumpy
                demand = rng.negative_binomial(1, 1 / (1 + lam * 6)) * (rng.random(T) < 0.18)

            # stock-outs: ~3% of days in multi-day episodes, sales censored
            stockout = np.zeros(T, dtype=np.int8)
            for _ in range(rng.integers(1, 5)):
                s = rng.integers(0, T - 5)
                stockout[s:s + rng.integers(1, 6)] = 1
            sales = np.where(stockout == 1, np.floor(demand * rng.uniform(0, 0.3, T)), demand).astype(int)

            rows.append(pd.DataFrame({
                "date": dates, "store_id": st.store_id, "sku_id": p.sku_id,
                "units_sold": sales, "true_demand": demand.astype(int),
                "selling_price": np.round(price, 2), "regular_price": np.round(reg_price, 2),
                "on_promo": is_promo, "discount_pct": np.round(discount[i] * is_promo, 2),
                "stockout_flag": stockout, "n_sibling_promo": n_sib_promo.astype(int),
            }))
    sales = pd.concat(rows, ignore_index=True)

    if inject_dq_issues:
        n = len(sales)
        # 1) missing days (POS feed gaps) ~0.5%
        sales = sales.drop(index=rng.choice(n, int(n * 0.005), replace=False))
        # 2) duplicate rows from double-loads ~0.3%
        dups = sales.sample(frac=0.003, random_state=seed)
        sales = pd.concat([sales, dups], ignore_index=True)
        # 3) negative units (returns netted into sales) ~0.2%
        idx = sales.sample(frac=0.002, random_state=seed + 1).index.unique()
        sales.loc[idx, "units_sold"] = -rng.integers(1, 4, len(idx))
        # 4) price typos (x100) ~0.05%
        idx = sales.sample(frac=0.0005, random_state=seed + 2).index.unique()
        sales.loc[idx, "selling_price"] = sales.loc[idx, "selling_price"] * 100
        sales = sales.sample(frac=1, random_state=seed).reset_index(drop=True)

    promo_cal = sales[sales.on_promo == 1][["date", "store_id", "sku_id", "discount_pct"]].drop_duplicates()
    return {"sales": sales, "products": products, "stores": stores,
            "promo_calendar": promo_cal, "holidays": hol}
