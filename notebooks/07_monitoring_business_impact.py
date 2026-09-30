# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Monitoring & business-impact summary
# MAGIC
# MAGIC **After deployment the question changes** from "is the model accurate?" to "**is it still accurate, and is it still saving money?**"
# MAGIC
# MAGIC | Signal | Threshold | Owner action |
# MAGIC |---|---|---|
# MAGIC | Weekly WAPE vs backtest baseline | > +20% relative → AMBER | DS checks for data or feed issues and new SKUs |
# MAGIC | Weekly bias | outside ±10% → AMBER | Planner review; retrain if the bias persists for 2 weeks |
# MAGIC | Both | → RED | Fall back to the previous champion alias and open an incident |
# MAGIC | Feature PSI | > 0.2 → investigate | e.g. price inflation shifts `rel_price` → retrain |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from retail_ds import monitoring
from retail_ds.backtest import default_cutoffs

fc = read_table("backtest_forecasts")
fc["date"] = pd.to_datetime(fc["date"])
lb = read_table("backtest_leaderboard")
base_wape = lb[(lb.grain == "daily_store_sku") & (lb.model == "lgbm_global")].wape.iloc[0]

health = monitoring.weekly_health(fc, "lgbm_global", base_wape, by="category")
write_table(health, "monitoring_weekly_health")
display(health.pivot_table(index="week", columns="category", values="status", aggfunc="first").reset_index())

# COMMAND ----------

gold = read_table("gold_features")
gold["date"] = pd.to_datetime(gold["date"])
cuts = default_cutoffs(gold.date.max(), n_folds=N_FOLDS)
ref, cur = gold[gold.date <= cuts[0]], gold[gold.date > cuts[-1]]
drift = pd.DataFrame([{"feature": c, "psi": monitoring.psi(ref[c], cur[c])}
                      for c in ["rmean_28", "selling_price", "rel_price", "discount_pct", "price_ratio", "n_sibling_promo"]])
drift["flag"] = drift.psi.map(lambda v: "investigate" if v > 0.2 else ("watch" if v > 0.1 else "stable"))
write_table(drift, "monitoring_feature_drift")
display(drift)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executive summary: Before vs After
# MAGIC One table that a business leader who relies on gut feel can read in 30 seconds. It feeds the AI/BI dashboard (`dashboards/queries.sql`).

# COMMAND ----------

wk = lb[lb.grain == "weekly_store_sku"].set_index("model")
inv = read_table("inventory_before_after").set_index("metric")
recs = read_table("price_recommendations")
promo = read_table("promo_economics")

exec_rows = [
    ("Forecast error (weekly WAPE)", wk.loc["naive_dow_avg", "wape"], wk.loc["lgbm_global", "wape"], "Excel 4-week average → global LightGBM"),
    ("Forecast bias (weekly)", wk.loc["naive_dow_avg", "bias"], wk.loc["lgbm_global", "bias"], "+ = over-forecast"),
    ("Fill rate", inv.loc["Fill rate", "before"], inv.loc["Fill rate", "after"], "95% service-level policy"),
    ("Avg inventory value ($)", inv.loc["Avg inventory value ($)", "before"], inv.loc["Avg inventory value ($)", "after"], "working capital released"),
    ("Lost margin / yr ($)", inv.loc["Lost margin, annualised ($)", "before"], inv.loc["Lost margin, annualised ($)", "after"], "from stock-outs"),
    ("Regular-price margin opportunity / yr ($)", 0.0, recs.annual_margin_gain.sum(),
     f"= {recs.annual_margin_gain.sum() / (recs.margin_now.sum() * 365):.0%} of current regular-price margin; "
     "upper bound (±10% guardrail binds), validate by A/B test"),
    ("Loss-making promo depth-category combos", None, int((promo.incremental_margin < 0).sum()), f"of {len(promo)} evaluated"),
]
exec_df = pd.DataFrame(exec_rows, columns=["kpi", "before", "after", "note"])
write_table(exec_df, "exec_summary")
display(exec_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scheduling
# MAGIC `databricks.yml` defines a job that runs notebooks 01 → 07 in order. In production:
# MAGIC * **Weekly** (Sunday night): 02 → 03 → score with `@champion` → 05 replenishment table → 07 health
# MAGIC * **Monthly**: full backtest (04). Promote the challenger to `@champion` only if its weekly VN1 improves by more than 2% and bias stays within ±5%
