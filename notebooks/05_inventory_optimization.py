# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 · Inventory optimisation: from forecast to replenishment and dollars
# MAGIC
# MAGIC A forecast only creates value once it changes an order. This notebook simulates both policies on the **same true demand** so the saving can be attributed to the forecast alone:
# MAGIC
# MAGIC | | Current state (Excel rule) | Forecast-driven |
# MAGIC |---|---|---|
# MAGIC | Order-up-to level | 14 days × last-4-week average | Forecast over (lead time + review period) + safety stock |
# MAGIC | Safety stock | none explicit (buffer comes from the 14 days of cover) | z(service level) × σ(forecast error) × √(L+R) |
# MAGIC | Reacts to promos and holidays | ❌ | ✅ (price and promo plan are forecast features) |
# MAGIC
# MAGIC **Set-up:** weekly review (R = 7), lead time L = 3 days, lost sales (no backorders), holding cost 25% of unit cost per year, lost-sale cost = unit margin.
# MAGIC σ is calibrated on backtest **fold 1** and the policies are simulated on **folds 2–4** (84 days), so the calibration never sees the test days.
# MAGIC
# MAGIC *Simplification:* forecasts for a review day use the backtest forecast made at that fold's cutoff. This approximates the forecast a planner would have had at the time.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

from retail_ds import inventory

fc = read_table("backtest_forecasts")
fc["date"] = pd.to_datetime(fc["date"])
gold = read_table("gold_features")
gold["date"] = pd.to_datetime(gold["date"])

MODEL = "lgbm_global"
f = fc[fc.model == MODEL].sort_values(["store_id", "sku_id", "date"])
cal = f[f.fold == 1]
sigma = cal.assign(e=cal.forecast - cal.actual).groupby(["store_id", "sku_id"])["e"].std().rename("sigma")
sim = f[f.fold > 1]
wide = lambda col: sim.pivot_table(index=["store_id", "sku_id"], columns="date", values=col, dropna=False).fillna(0)  # noqa: E731
D, Fc = wide("true_demand"), wide("forecast")
keys = D.index
first = sim.date.min()
hist_avg = (gold[(gold.date < first) & (gold.date >= first - pd.Timedelta(days=28))]
            .groupby(["store_id", "sku_id"])["demand_filled"].mean())
meta = sim.groupby(["store_id", "sku_id"])[["unit_cost", "selling_price"]].median()

res = inventory.run_policies(D.values, Fc.values, hist_avg.reindex(keys).fillna(0).values,
                             sigma.reindex(keys).fillna(0).values, meta.reindex(keys).unit_cost.values,
                             meta.reindex(keys).selling_price.values)
res["sim_days"] = D.shape[1]
write_table(res, "inventory_policy_results")
display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The trade-off curve: the chart to show leadership
# MAGIC Every point on the curve is a choice between service and working capital. The current policy sits **above and to the left** of the curve: it holds more stock *and* serves customers worse. Any point on the curve beats it on both.
# MAGIC
# MAGIC *Note:* the target is a **cycle service level** (probability of no stock-out in a cycle). The realised **fill rate** (share of units served) is always higher, which is why a 0.85 target still gives a fill rate of about 98%.

# COMMAND ----------

fd, cur = res[res.policy == "forecast_driven"], res[res.policy == "current_state"].iloc[0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fd.avg_inventory_value, fd.fill_rate * 100, "o-", label="Forecast-driven (target service level)")
for _, r in fd.iterrows():
    ax.annotate(f"{r.target_service:.0%}", (r.avg_inventory_value, r.fill_rate * 100), textcoords="offset points", xytext=(6, -10))
ax.scatter(cur.avg_inventory_value, cur.fill_rate * 100, s=120, marker="X", color="crimson", label="Current state (14-day cover rule)")
ax.set_xlabel("Average inventory value ($)"); ax.set_ylabel("Fill rate (%)")
ax.set_title("Service vs working capital: the forecast moves the whole frontier"); ax.legend(); ax.grid(alpha=.3)
display(fig)
plt.close(fig)

# COMMAND ----------

pick = fd[fd.target_service == 0.95].iloc[0]
annualise = 365 / pick.sim_days
story = pd.DataFrame([
    {"metric": "Fill rate", "before": cur.fill_rate, "after": pick.fill_rate},
    {"metric": "Avg inventory value ($)", "before": cur.avg_inventory_value, "after": pick.avg_inventory_value},
    {"metric": "Lost margin, annualised ($)", "before": cur.lost_margin * annualise, "after": pick.lost_margin * annualise},
    {"metric": "Holding cost, annualised ($)", "before": cur.holding_cost * annualise, "after": pick.holding_cost * annualise},
    {"metric": "Stock-out store-SKU-days (%)", "before": cur.stockout_days_pct, "after": pick.stockout_days_pct},
])
story["change"] = story.after / story.before - 1
write_table(story, "inventory_before_after")
display(story)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next week's replenishment recommendation (the table a planner acts on)

# COMMAND ----------

from scipy.stats import norm

fc28 = read_table("forecast_28d")
fc28["date"] = pd.to_datetime(fc28["date"])
p = inventory.InvParams()
w = p.lead_time + p.review_period
nxt = fc28[fc28.date < fc28.date.min() + pd.Timedelta(days=w)]
rec = nxt.groupby(["store_id", "sku_id"]).agg(demand_L_plus_R=("p50", "sum"), sigma=("sigma", "first")).reset_index()
rec["safety_stock"] = norm.ppf(0.95) * rec.sigma.fillna(0) * np.sqrt(w)
rec["order_up_to"] = np.ceil(rec.demand_L_plus_R + rec.safety_stock)
write_table(rec, "replenishment_recommendation")
display(rec.sort_values("order_up_to", ascending=False).head(20))