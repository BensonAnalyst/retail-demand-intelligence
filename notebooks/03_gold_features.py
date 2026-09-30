# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Gold: feature engineering (leakage-safe)
# MAGIC
# MAGIC **Decision:** use one *direct* global model with all demand lags shifted by at least 28 days (the forecast horizon).
# MAGIC
# MAGIC | Option | Accuracy at day 1–7 | Complexity | Error compounding | Chosen |
# MAGIC |---|---|---|---|---|
# MAGIC | Recursive (lag 1…) | best | medium | yes | |
# MAGIC | One model per horizon | best | 28 models | no | |
# MAGIC | **Direct, lags ≥ 28** | good | **1 model** | **no** | ✅ fits a weekly batch replenishment cadence |
# MAGIC
# MAGIC | Feature group | Examples | Known in the future? |
# MAGIC |---|---|---|
# MAGIC | Demand history | `lag_28…lag_364`, `rmean_7/28/91`, `rstd_28`, `rzero_28`, `dow_mean_4w`, `trend_ratio` | shifted ≥ 28 days |
# MAGIC | Price / promo | `price_ratio`, `rel_price`, `discount_pct`, `on_promo`, `n_sibling_promo` (cannibalisation) | ✅ planned by the commercial team |
# MAGIC | Calendar | `dow`, `month`, `weekofyear`, `is_payday_window`, `days_to_holiday` | ✅ |
# MAGIC | Static | `store_id`, `sku_id`, `category`, `region`, `segment`, `store_size` | ✅ |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from retail_ds import features

silver = read_table("silver_sales")
silver["date"] = pd.to_datetime(silver["date"])
gold = features.build_features(silver, read_table("bronze_products"), read_table("bronze_stores"),
                               read_table("demand_segments"))
FEATS = features.feature_columns(gold)
write_table(gold, "gold_features")
write_table(pd.DataFrame({"feature": FEATS}), "gold_feature_list")
print(len(FEATS), "features:", FEATS)

# COMMAND ----------

# Leakage guard (the same check runs in tests/): lag_28 at t must equal demand at t-28
chk = gold[(gold.store_id == gold.store_id.iloc[0]) & (gold.sku_id == gold.sku_id.iloc[0])].reset_index(drop=True)
assert chk.loc[200, "lag_28"] == chk.loc[172, "demand_filled"], "lag leakage!"
print("leakage guard passed")