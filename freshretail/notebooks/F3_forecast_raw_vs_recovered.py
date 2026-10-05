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

results = []
with mlflow.start_run(run_name="raw_vs_recovered"):
    mlflow.log_params({"n_series": N_SERIES, "horizon": 7, "folds": "last_train_week,official_eval"})
    for cutoff in (last_day - 7, last_day):
        res, _, _ = pipeline.run_fold(full, cutoff)
        results.append(res)
        for _, r in res[res["eval_view"].str[:2].isin(["A.", "C."])].iterrows():
            name = re.sub(r"[^A-Za-z0-9]+", "_", r.training_target.split(":")[-1]).strip("_")
            tag = f"c{cutoff}.{name}.view{r.eval_view[:1]}"
            mlflow.log_metrics({f"{tag}.wape": r.wape, f"{tag}.bias": r.wpe_bias, f"{tag}.vn1": r.vn1})
    results = pd.concat(results, ignore_index=True)
    mlflow.log_table(results, "forecast_results.json")

write_table(results, "frn_forecast_results")
headline = results[results["eval_view"].str.match(r"^(A|B|C|C2)\.")]
display(headline.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wpe_bias").reset_index())

# COMMAND ----------

display(headline.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wape").reset_index())
display(results[~results["eval_view"].str.match(r"^(A|B|C|C2)\.")])

# COMMAND ----------

# MAGIC %md
# MAGIC ### What to look for
# MAGIC 1. **Bias under view C:** raw-sales training should come out clearly negative (under-forecasting). Recovered targets should sit near 0. This is the business case: under-forecasting means under-ordering, which means more empty shelves.
# MAGIC 2. **Does the ranking flip under view A?** If raw sales look best only on in-stock days, that is the selection bias described above, not a better model.
# MAGIC 3. **High-sale vs low-sale series:** the paper found recovery helps fast movers most and can over-correct slow movers. Check whether we see the same.
# MAGIC 4. **Paper reference** (TimesNet recovery + TFT forecaster, view A): WAPE 31.75% → 29.02%, bias −7.37% → +2.58%. Our models are simpler; the aim is the direction and the reasoning, not to beat a deep-learning benchmark.
