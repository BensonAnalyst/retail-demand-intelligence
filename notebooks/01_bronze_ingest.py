# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — ingest raw POS, master data, promo calendar
# MAGIC
# MAGIC **Business problem:** a regional retailer (10 stores, 40 SKUs, 4 categories, SG/MY calendar) plans replenishment
# MAGIC in Excel using "last 4 weeks' average". Stores still run out of stock during promos and 11.11/12.12, and carry too much slow-moving stock the rest of the time.
# MAGIC
# MAGIC **This notebook** lands the raw data *as-is* (defects included) into Delta. Nothing is cleaned in bronze, so every
# MAGIC downstream fix can be traced back to it.
# MAGIC
# MAGIC **Data source options**
# MAGIC * `synthetic` (default): realistic generator with a *known ground truth* (true elasticity, promo uplift,
# MAGIC   cannibalisation, censored stock-outs), so we can check that our estimators recover the truth.
# MAGIC * `vn1`: VN1 Forecasting Competition weekly sales. Upload the CSVs to a UC Volume and set the path below.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.dropdown("source", "synthetic", ["synthetic", "vn1"])
SOURCE = dbutils.widgets.get("source")

from retail_ds import data_gen

if SOURCE == "synthetic":
    raw = data_gen.generate(n_stores=N_STORES, n_skus=N_SKUS, seed=42, inject_dq_issues=True)
    for name, pdf in raw.items():
        write_table(pdf, f"bronze_{name}")
else:
    # VN1: Phase-0 sales are wide (one column per week). Melt to long format and land as-is.
    VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw/vn1"
    wide = pd.read_csv(f"{VOL}/Phase 0 - Sales.csv")
    long = wide.melt(id_vars=["Client", "Warehouse", "Product"], var_name="date", value_name="units_sold")
    long["date"] = pd.to_datetime(long["date"])
    write_table(long, "bronze_vn1_sales")
    print("NOTE: VN1 is weekly with no price/promo fields. Run notebooks 02-04 with a weekly horizon (13) "
          "and skip 06 (pricing).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick look at what landed

# COMMAND ----------

display(spark.sql(f"""
  SELECT COUNT(*) rows, COUNT(DISTINCT store_id) stores, COUNT(DISTINCT sku_id) skus,
         MIN(date) first_day, MAX(date) last_day,
         SUM(CASE WHEN units_sold < 0 THEN 1 ELSE 0 END) negative_rows,
         SUM(stockout_flag) stockout_days
  FROM {FQ}.bronze_sales"""))
