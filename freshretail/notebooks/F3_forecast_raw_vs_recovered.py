# Databricks notebook source
# MAGIC %md
# MAGIC # F3 · Does training on recovered demand forecast better?
# MAGIC
# MAGIC The same global LightGBM (7-day horizon, lags ≥ 7, known discount, holiday and activity plan) is trained three times, with different **targets**:
# MAGIC raw sales, profile-recovered demand, and LightGBM-recovered demand. There are 2 folds: the last train week (internal) and the official 7-day eval split.
# MAGIC
# MAGIC ### Evaluation is the hard part: three views of the truth
# MAGIC | View | Truth used | Problem |
# MAGIC |---|---|---|
# MAGIC | **A. In-stock days only** (the paper's protocol) | Observed sales, exact | **Selection bias:** high-demand days are the ones that sell out, so in-stock days skew low. That flatters a model that under-forecasts |
# MAGIC | **B. All days vs raw sales** | Observed sales | Truth is too low on stock-out days |
# MAGIC | **C. All days vs recovered demand** | Observed on in-stock days, recovered on stock-out days | Only as good as the recovery, which F2 validated. **C2** repeats it with the simpler profile method, so the LightGBM recovery isn't marking its own homework |
# MAGIC
# MAGIC On a mock dataset where true demand was known, view C tracked the true bias within about 1 point, while view A reversed the ranking. **Report all views, and make decisions on C.**

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.*
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./F0_config

# COMMAND ----------

import re

import mlflow

from frn import pipeline

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/freshretail_censored_demand")
full, last_day, _ = load_data()

with mlflow.start_run(run_name="raw_vs_recovered_calibrated"):
    mlflow.log_params({"n_series": N_SERIES, "horizon": 7, "folds": "warmup(-14),last_train_week,official_eval",
                       "calibration": "walk-forward, previous week own-target ratio"})
    results, horizon, scales, _ = pipeline.run_backtest(full, last_day)
    for _, r in results[results["eval_view"].str[:2].isin(["A.", "C."])].iterrows():
        name = re.sub(r"[^A-Za-z0-9]+", "_", r.training_target.split(":")[-1]).strip("_")
        tag = f"c{r.cutoff_day}.{name}.view{r.eval_view[:1]}"
        mlflow.log_metrics({f"{tag}.wape": r.wape, f"{tag}.bias": r.wpe_bias, f"{tag}.vn1": r.vn1})
    mlflow.log_table(results, "forecast_results.json")
    mlflow.log_table(horizon, "forecast_by_horizon.json")

write_table(results, "frn_forecast_results")
write_table(horizon, "frn_forecast_by_horizon")
write_table(scales, "frn_calibration_scales")
display(scales)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Headline: bias and WAPE by evaluation view
# MAGIC Rows without "+ calibration" are the original models (unchanged from the previous run). Rows **with** it multiply the forecast by a factor learned on the **previous week only** (walk-forward, no peeking), which removes a shared level gap. Calibration fixes the level; it cannot fix censoring, which is why raw sales stay biased in view C.

# COMMAND ----------

headline = results[results["eval_view"].str.match(r"^(A|B|C|C2)\.")]
bias = headline.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wpe_bias").reset_index()
wape = headline.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wape").reset_index()
display(bias)
display(wape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average of both folds (the numbers to quote)

# COMMAND ----------

avg = (headline.groupby(["training_target", "eval_view"])[["wape", "wpe_bias"]].mean()
               .unstack("eval_view").round(4))
display(avg.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does the error grow further ahead? (view C, by days ahead)

# COMMAND ----------

display(horizon.pivot_table(index="training_target", columns="horizon_days", values="wpe_bias").round(4).reset_index())
display(horizon.pivot_table(index="training_target", columns="horizon_days", values="wape").round(4).reset_index())

# COMMAND ----------

display(results[~results["eval_view"].str.match(r"^(A|B|C|C2)\.")])
