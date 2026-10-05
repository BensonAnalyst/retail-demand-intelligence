# Databricks notebook source
# MAGIC %md
# MAGIC # F2 · Latent demand recovery: what would customers have bought?
# MAGIC
# MAGIC On a stock-out day, recorded sales = demand in the **in-stock hours only**. Train a forecaster on that and it learns to under-forecast. The FreshRetailNet paper measures this at about −7% bias, which leads to under-ordering and more stock-outs.
# MAGIC
# MAGIC | Method | Idea | Strength |
# MAGIC |---|---|---|
# MAGIC | `observed` | Use sales as recorded | Baseline: biased low |
# MAGIC | `profile` | If the in-stock hours normally carry 70% of a day's demand, then demand ≈ observed ÷ 0.70. The hourly profile is learned from in-stock days, falling back store-product → store-category → category | Simple and explainable |
# MAGIC | `self-supervised LightGBM` | Take in-stock days, **hide hours using real stock-out patterns** copied from real stock-out days, and train a model to predict the full day from the partial day | Learns where the profile is wrong (promos, weekends, weather) |
# MAGIC
# MAGIC **How do we know it works if true demand is never observed?** The same masking trick, applied to **held-out series**, creates days where the hidden demand is known. That gives a real accuracy and bias score for each method.

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.*
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./F0_config

# COMMAND ----------

from frn import recovery
from frn.data import OP_HOURS

full, last_day, _ = load_data()
val = recovery.validate_recovery(full, last_day, holdout_frac=0.2)
write_table(val, "frn_recovery_validation")
display(val)

# COMMAND ----------

# MAGIC %md
# MAGIC **How to read it:** `observed` shows how much demand the stock-out patterns hide (its bias ≈ the hidden share).
# MAGIC A good recovery method has bias close to 0 and the lowest WAPE. Compare with the paper's best deep-learning imputer (TimesNet): 27.6% WAPE and +1.4% bias on its own recovery test. That figure is a reference point, not like-for-like, because the test design differs.

# COMMAND ----------

rec, rec_model = recovery.recover(full, last_day, return_model=True)
out = full.df[["store_id", "product_id", "dt", "day", "first_category_id", "third_category_id"]].copy()
op = full.oos[:, OP_HOURS]
out["oos_hours"] = op.sum(1)
out["first_oos_hour"] = np.where(op.any(1), op.argmax(1) + OP_HOURS.start, -1)   # -1 = no stock-out
out = pd.concat([out, rec], axis=1)
out = out[out["day"] <= last_day]
write_table(out, "frn_recovered_demand")
stock_out = out["oos_hours"] > 0
display(pd.DataFrame({
    "metric": ["stock-out days", "observed sales on stock-out days", "recovered (profile)", "recovered (LightGBM)",
               "estimated lost demand, % of total demand"],
    "value": [int(stock_out.sum()), out.loc[stock_out, "observed"].sum(), out.loc[stock_out, "rec_profile"].sum(),
              out.loc[stock_out, "rec_model"].sum(),
              (out["rec_model"].sum() - out["observed"].sum()) / out["rec_model"].sum() * 100]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### What does the recovery model actually rely on?
# MAGIC `gain_pct` = share of the model's improvement that comes from each input. Expect the **partial day** (how much sold, when the shelf emptied) to dominate; promo, calendar and weather show how much context adds on top of the simple profile idea.

# COMMAND ----------

imp = recovery.feature_importance(rec_model)
write_table(imp, "frn_recovery_importance")
display(imp)
display(imp.groupby("group", as_index=False)["gain_pct"].sum().sort_values("gain_pct", ascending=False))
