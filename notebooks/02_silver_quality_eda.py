# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Silver: data-quality rules, censored demand, EDA, demand segmentation
# MAGIC
# MAGIC Real POS data is imperfect. These are the **validation rules** a data scientist hands to engineering. Each one is
# MAGIC counted and published as a DQ scorecard table, so defects are visible rather than silently fixed.
# MAGIC
# MAGIC | Rule | Defect | Treatment | Why it matters |
# MAGIC |---|---|---|---|
# MAGIC | R1 | Duplicate rows (double loads) | Deduplicate on store-SKU-date | Inflates demand, which inflates orders |
# MAGIC | R2 | Negative units (returns netted in POS) | Set to missing, impute | Returns are not negative demand |
# MAGIC | R3 | Price typos (×100) | Replace with regular price × (1 − discount) | Would wreck elasticity estimates |
# MAGIC | R4 | Missing POS days | Complete calendar, impute inputs | Lags and rolling windows need a full grid |
# MAGIC | **R5** | **Stock-out days** | **Target = missing (censored)** | Sales on a stock-out day understate **demand**. Training on them teaches the model to under-forecast, which causes more stock-outs: a self-fulfilling loop. |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from retail_ds import cleaning
from retail_ds.backtest import default_cutoffs

bronze = read_table("bronze_sales")
silver, dq = cleaning.clean_sales(bronze)
write_table(silver, "silver_sales")

dq_df = pd.DataFrame([{"rule": k, "count": v} for k, v in dq.items()])
dq_df["pct_of_input"] = dq_df["count"] / dq["rows_in"]
write_table(dq_df, "dq_scorecard")
display(dq_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demand segmentation (Syntetos–Boylan ADI / CV²)
# MAGIC One model does not fit all series. Segmenting first tells us **which forecasting method to use for each series** and **what accuracy is realistic**.
# MAGIC * **smooth**: frequent, stable demand. ML and ETS work well.
# MAGIC * **erratic**: frequent but volatile demand.
# MAGIC * **intermittent / lumpy**: many zero days. Croston/SBA works here, and daily accuracy will always look poor. Judge these at the weekly level.
# MAGIC
# MAGIC Segments are computed **only on history before the first backtest cutoff**, to avoid leaking test data into the segment labels.

# COMMAND ----------

seg_cut = default_cutoffs(silver["date"].max(), n_folds=N_FOLDS)[0]
seg = cleaning.classify_demand(silver[silver["date"] <= seg_cut])
write_table(seg, "demand_segments")
display(seg.groupby("segment").agg(series=("sku_id", "count"), mean_daily=("mean_demand", "mean"),
                                   zero_share=("zero_share", "mean")).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA: the patterns the model must learn

# COMMAND ----------

# Weekly network demand by category: seasonality + holiday spikes (CNY, 11.11, 12.12, Christmas)
display(spark.sql(f"""
SELECT date_trunc('week', s.date) AS week, p.category, SUM(s.demand_target) AS units
FROM {FQ}.silver_sales s JOIN {FQ}.bronze_products p USING (sku_id)
GROUP BY ALL ORDER BY week
"""))

# COMMAND ----------

# Promo lift by category & discount depth (raw, before controlling for anything)
display(spark.sql(f"""
SELECT p.category, s.discount_pct, ROUND(AVG(s.demand_target), 2) AS avg_daily_units, COUNT(*) AS days
FROM {FQ}.silver_sales s JOIN {FQ}.bronze_products p USING (sku_id)
WHERE s.demand_target IS NOT NULL
GROUP BY ALL ORDER BY category, discount_pct
"""))

# COMMAND ----------

# Day-of-week profile (weekend peak) and the size of the stock-out problem
display(spark.sql(f"""
SELECT dayofweek(date) AS dow, ROUND(AVG(demand_target),2) AS avg_units,
       ROUND(100*AVG(stockout_flag),2) AS stockout_pct
FROM {FQ}.silver_sales GROUP BY ALL ORDER BY dow
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Production note:** at larger scale, rules R1–R5 would run as **Lakeflow Declarative Pipelines (DLT) expectations**, e.g.
# MAGIC ```sql
# MAGIC CONSTRAINT non_negative_units EXPECT (units_sold >= 0) ON VIOLATION DROP ROW
# MAGIC ```
# MAGIC The pandas version is kept here so the rule logic is unit-tested (`tests/`) and reusable.