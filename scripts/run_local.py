"""Run the whole pipeline locally with pandas (no Spark) - for CI / quick iteration.
Mirrors notebooks 01-07. Usage:  python scripts/run_local.py [--stores 10 --skus 40]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ds import cleaning, data_gen, features, inventory, metrics, monitoring, pricing  # noqa: E402
from retail_ds.backtest import default_cutoffs, run_backtest, scored  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stores", type=int, default=10)
ap.add_argument("--skus", type=int, default=40)
ap.add_argument("--out", default=str(ROOT / "outputs"))
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(exist_ok=True)
t0 = time.time()
lap = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}")  # noqa: E731

# 01 bronze
raw = data_gen.generate(a.stores, a.skus)
lap(f"bronze: {len(raw['sales']):,} sales rows")

# 02 silver
silver, dq = cleaning.clean_sales(raw["sales"])
# segment on history BEFORE the first backtest cutoff only (no test-period leakage)
seg_cutoff = default_cutoffs(silver.date.max(), n_folds=4)[0]
seg = cleaning.classify_demand(silver[silver.date <= seg_cutoff])
lap(f"silver DQ: {dq}")
print(seg.segment.value_counts().to_dict())

# 03 gold
gold = features.build_features(silver, raw["products"], raw["stores"], seg)
FEATS = features.feature_columns(gold)
lap(f"gold: {gold.shape}, {len(FEATS)} features")

# 04 backtest
cuts = default_cutoffs(gold.date.max(), n_folds=4)
fc, fitted = run_backtest(gold, FEATS, cuts, log=lap)
sc = scored(fc)
res = metrics.evaluate(sc, ["model"]).sort_values("vn1")
res_week = metrics.evaluate_at_level(sc, ["store_id", "sku_id"], "W").sort_values("vn1")
res_seg = metrics.evaluate(sc, ["model", "segment"])
res_promo = metrics.evaluate(sc, ["model", "on_promo"])
print("\nDAILY store-SKU\n", res.round(3).to_string(index=False))
print("\nWEEKLY store-SKU (replenishment level)\n", res_week.round(3).to_string(index=False))
print("\nBY SEGMENT (VN1)\n", res_seg.pivot(index="model", columns="segment", values="vn1").round(3))
print("\nPROMO vs NON-PROMO (WAPE)\n", res_promo.pivot(index="model", columns="on_promo", values="wape").round(3))
fi = fitted[len(cuts)].importance()
print("\nTop features\n", fi.head(12).to_string(index=False))
res.to_csv(OUT / "backtest_daily.csv", index=False)
res_week.to_csv(OUT / "backtest_weekly.csv", index=False)
res_seg.to_csv(OUT / "backtest_segment.csv", index=False)
res_promo.to_csv(OUT / "backtest_promo.csv", index=False)
fi.to_csv(OUT / "feature_importance.csv", index=False)

# 05 inventory: calibrate sigma on fold 1, simulate folds 2-4
BEST = "lgbm_global"
f = fc[fc.model == BEST].sort_values(["store_id", "sku_id", "date"])
cal = f[f.fold == 1]
sig = (cal.assign(e=cal.forecast - cal.actual).groupby(["store_id", "sku_id"])["e"].std().rename("sigma"))
sim = f[f.fold > 1]
wide = lambda col: sim.pivot_table(index=["store_id", "sku_id"], columns="date", values=col, dropna=False).fillna(0)  # noqa: E731
D, F = wide("true_demand"), wide("forecast")
keys = D.index
first_sim = sim.date.min()
hist = gold[(gold.date < first_sim) & (gold.date >= first_sim - pd.Timedelta(days=28))]
hist_avg = hist.groupby(["store_id", "sku_id"], observed=True)["demand_filled"].mean()
hist_avg.index = hist_avg.index.set_levels([lvl.astype(str) for lvl in hist_avg.index.levels])
meta = sim.groupby(["store_id", "sku_id"])[["unit_cost", "selling_price"]].median()
inv = inventory.run_policies(D.values, F.values, hist_avg.reindex(keys).fillna(0).values,
                             sig.reindex(keys).fillna(0).values, meta.reindex(keys).unit_cost.values,
                             meta.reindex(keys).selling_price.values)
inv["sim_days"] = D.shape[1]
print("\nINVENTORY POLICIES\n", inv.round(3).to_string(index=False))
inv.to_csv(OUT / "inventory_policies.csv", index=False)

# 06 pricing
naive = pd.DataFrame([pricing.estimate_category_response(gold, c, control_trend=False) for c in data_gen.CATEGORIES])
resp = pd.DataFrame([pricing.estimate_category_response(gold, c) for c in data_gen.CATEGORIES])
resp["elasticity_naive_no_trend"] = naive["elasticity"].values
truth = raw["products"].groupby("category")[["true_elasticity", "true_promo_uplift"]].mean().reset_index()
resp = resp.merge(truth, on="category")
resp["true_cannibalisation"] = data_gen.CANNIBALISATION
print("\nPRICE / PROMO RESPONSE (estimated vs ground truth)\n", resp.round(3).to_string(index=False))
resp.to_csv(OUT / "price_response.csv", index=False)

last90 = gold[gold.date > gold.date.max() - pd.Timedelta(days=90)]
sku_stats = (last90[last90.on_promo == 0].groupby("sku_id", observed=True)
             .agg(category=("category", "first"), regular_price=("regular_price", "median"),
                  unit_cost=("unit_cost", "first"), daily_units=("demand_filled", "sum")).reset_index())
sku_stats["daily_units"] /= last90.date.nunique()
sku_stats["category"] = sku_stats["category"].astype(str)
recs = pricing.price_recommendations(sku_stats, resp)
recs.to_csv(OUT / "price_recommendations.csv", index=False)
print(f"\nPrice recs: {(recs.price_change_pct > 0.001).sum()} up, {(recs.price_change_pct < -0.001).sum()} down; "
      f"annual margin gain ${recs.annual_margin_gain.sum():,.0f}")

promo_rows = []
for _, r in resp.iterrows():
    cs = sku_stats[sku_stats.category == r.category]
    ser = gold[(gold.category == r.category)]
    n_series = ser.groupby(["store_id", "sku_id"], observed=True).ngroups / ser.store_id.nunique()
    e = pricing.promo_depth_economics(r, cs.regular_price.mean(), cs.unit_cost.mean(),
                                      cs.daily_units.mean() / gold.store_id.nunique(),
                                      ((cs.regular_price - cs.unit_cost) * cs.daily_units / gold.store_id.nunique()).mean(),
                                      n_series - 1)
    e.insert(0, "category", r.category)
    promo_rows.append(e)
promo_econ = pd.concat(promo_rows)
print("\nPROMO DEPTH ECONOMICS (per store per promo-day)\n", promo_econ.round(3).to_string(index=False))
promo_econ.to_csv(OUT / "promo_economics.csv", index=False)

# 07 monitoring
base_wape = res.set_index("model").loc[BEST, "wape"]
health = monitoring.weekly_health(fc, BEST, base_wape)
print("\nMONITORING status counts\n", health.status.value_counts().to_dict())
train_w = gold[gold.date <= cuts[0]]
test_w = gold[gold.date > cuts[-1]]
drift = {c: monitoring.psi(train_w[c], test_w[c]) for c in ["rmean_28", "selling_price", "discount_pct", "rel_price"]}
print("PSI:", {k: round(v, 3) for k, v in drift.items()})
health.to_csv(OUT / "monitoring_weekly.csv", index=False)
json.dump({"dq": dq, "psi": drift, "segments": seg.segment.value_counts().to_dict(),
           "rows_gold": len(gold), "n_features": len(FEATS)}, open(OUT / "run_summary.json", "w"), indent=2, default=float)
lap("done")
