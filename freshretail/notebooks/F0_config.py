# Databricks notebook source
# MAGIC %md
# MAGIC # F0 · Config: FreshRetailNet-50K censored-demand project
# MAGIC Shared settings, run by every F-notebook with `%run ./F0_config`.
# MAGIC
# MAGIC | Widget | Default | Notes |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | `workspace` / `retail_ds` | Same schema as the main project; tables are prefixed `frn_` |
# MAGIC | `data_dir` | `/Volumes/<catalog>/<schema>/raw/freshretail` | Where `train.parquet` and `eval.parquet` live |
# MAGIC | `n_series` | `10000` | Random sample of the 50,000 store-product series (fixed seed, so every notebook uses the same sample). `0` = all, which needs a large driver |

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_ds")
dbutils.widgets.text("n_series", "10000")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
dbutils.widgets.text("data_dir", f"/Volumes/{CATALOG}/{SCHEMA}/raw/freshretail")
DATA_DIR = dbutils.widgets.get("data_dir")
N_SERIES = int(dbutils.widgets.get("n_series"))
FQ = f"{CATALOG}.{SCHEMA}"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.raw")

SRC = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def write_table(pdf: pd.DataFrame, name: str):
    pdf = pdf.copy()
    for c in pdf.columns:
        if isinstance(pdf[c].dtype, pd.CategoricalDtype):
            pdf[c] = pdf[c].astype(str)
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.{name}")
    print(f"wrote {FQ}.{name}: {len(pdf):,} rows")


def load_data():
    """Same series sample in every notebook (fixed seed). Returns full panel, last train day, check."""
    from frn import data, pipeline
    tr, ev = f"{DATA_DIR}/train.parquet", f"{DATA_DIR}/eval.parquet"
    keys = data.sample_series(tr, N_SERIES or None, seed=42)
    full, last_day, chk = pipeline.prepare(data.load_panel(tr, keys), data.load_panel(ev, keys))
    print(f"{len(full.df):,} rows | {full.df.groupby(['store_id','product_id']).ngroups:,} series | "
          f"train days 0..{last_day}, eval days {last_day+1}..{full.df['day'].max()}")
    return full, last_day, chk


print(f"{FQ} | data: {DATA_DIR} | series sample: {N_SERIES or 'all'}")
