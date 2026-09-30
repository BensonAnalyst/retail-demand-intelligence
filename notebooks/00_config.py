# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Config
# MAGIC Shared settings, imported by every notebook with `%run ./00_config`.
# MAGIC
# MAGIC | Setting | Default | Notes |
# MAGIC |---|---|---|
# MAGIC | `catalog` | `workspace` | Free Edition default catalog. Use UC catalog on a paid workspace. |
# MAGIC | `schema` | `retail_ds` | All Delta tables + the registered model live here. |
# MAGIC | `n_stores` / `n_skus` | 10 / 40 | 400 store-SKU series × 731 days ≈ 292k rows. Scale up to test Spark parallelism. |

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_ds")
dbutils.widgets.text("n_stores", "10")
dbutils.widgets.text("n_skus", "40")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
N_STORES = int(dbutils.widgets.get("n_stores"))
N_SKUS = int(dbutils.widgets.get("n_skus"))
FQ = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{FQ}.demand_forecast_lgbm"
N_FOLDS = 4

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")

# make the reusable package importable (repo layout: notebooks/ and src/ are siblings)
SRC = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd  # noqa: E402


def write_table(pdf: pd.DataFrame, name: str, mode: str = "overwrite"):
    """pandas -> Delta in Unity Catalog. Categoricals cast to string for Delta."""
    pdf = pdf.copy()
    for c in pdf.columns:
        if isinstance(pdf[c].dtype, pd.CategoricalDtype):
            pdf[c] = pdf[c].astype(str)
    (spark.createDataFrame(pdf).write.mode(mode).option("overwriteSchema", "true")
          .saveAsTable(f"{FQ}.{name}"))
    print(f"wrote {FQ}.{name}: {len(pdf):,} rows")


def read_table(name: str) -> pd.DataFrame:
    return spark.table(f"{FQ}.{name}").toPandas()


print(f"catalog.schema = {FQ} | src on path: {SRC}")