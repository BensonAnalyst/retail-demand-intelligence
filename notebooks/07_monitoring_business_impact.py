# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
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

# COMMAND ----------

# ===== RESULTS EXPORT: =====
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
def show(title, df, n=50):
    print(f"\n### {title}\n" + df.head(n).round(4).to_string(index=False))

t = lambda name: spark.table(f"{FQ}.{name}").toPandas()

show("DQ SCORECARD", t("dq_scorecard"))
seg = t("demand_segments")
show("SEGMENTS", seg.groupby("segment").agg(series=("sku_id", "count"),
     mean_daily=("mean_demand", "mean"), zero_share=("zero_share", "mean")).reset_index())
show("LEADERBOARD", t("backtest_leaderboard")[["grain", "model", "wape", "mape", "bias", "vn1", "n"]])
show("VN1 BY SEGMENT", t("backtest_by_segment").pivot(index="model", columns="segment", values="vn1").reset_index())
show("WAPE PROMO(1) vs NON-PROMO(0)", t("backtest_by_promo").pivot(index="model", columns="on_promo", values="wape").reset_index())
show("TOP FEATURES", t("feature_importance"), 10)
show("INVENTORY POLICIES", t("inventory_policy_results"))
show("INVENTORY BEFORE/AFTER", t("inventory_before_after"))
show("PRICE RESPONSE", t("price_response")[["category", "elasticity_naive_no_trend", "elasticity", "elasticity_lo",
     "elasticity_hi", "true_elasticity", "promo_uplift", "true_promo_uplift", "cannibalisation_per_sibling"]])
show("PROMO ECONOMICS", t("promo_economics").pivot(index="depth", columns="category", values="incremental_margin").reset_index())
pr = t("price_recommendations")
print(f"\n### PRICE RECS: {len(pr)} SKUs | up {(pr.price_change_pct>0.001).sum()} | down {(pr.price_change_pct<-0.001).sum()} "
      f"| capped at +10%: {(pr.price_change_pct.round(3)>=0.1).sum()} | annual gain ${pr.annual_margin_gain.sum():,.0f}")
h = t("monitoring_weekly_health")
print("\n### HEALTH STATUS COUNTS:", h.status.value_counts().to_dict())
show("FEATURE DRIFT", t("monitoring_feature_drift"))
show("EXEC SUMMARY", t("exec_summary"))
f28 = t("forecast_28d")
print(f"\n### FORECAST_28D: {len(f28):,} rows | {f28.date.min()} to {f28.date.max()} | "
      f"model v{f28.model_version.iloc[0]} | total P50 {f28.p50.sum():,.0f}")