# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Forecasting: rolling-origin backtest, model selection, MLflow, registry
# MAGIC
# MAGIC **Question:** which method should drive replenishment, and how much better is it than what planners do today?
# MAGIC
# MAGIC | Model | Role |
# MAGIC |---|---|
# MAGIC | `naive_dow_avg` | **Current state.** Same weekday, average of the last 4 weeks (the Excel method) |
# MAGIC | `seasonal_naive` | Minimum bar: repeat last week |
# MAGIC | `croston_sba` | Intermittent-demand specialist |
# MAGIC | `ets` | Classical Holt-Winters (damped trend, weekly seasonality). Routed to SBA for intermittent series |
# MAGIC | `lgbm_global` | One LightGBM across all series (Tweedie loss for zero-inflated counts) with price, promo and calendar |
# MAGIC | `ensemble_70_30` | 0.7 × LGBM + 0.3 × ETS |
# MAGIC
# MAGIC **Backtest:** 4 folds × 28 days, expanding window. The last fold covers 12.12 and Christmas. Stock-out days are excluded from scoring.
# MAGIC
# MAGIC **Metrics** (error = forecast − actual):
# MAGIC * **WAPE** = Σ|e| / Σy: volume-weighted accuracy, the headline number
# MAGIC * **Bias** = Σe / Σy: positive means over-stock, negative means stock-outs
# MAGIC * **VN1** = WAPE + |Bias|: the VN1 competition score. It penalises inaccuracy *and* systematic bias together, which is what drives inventory cost. **Used for model selection.**
# MAGIC * **MAPE**: reported but *not* used for selection. It is undefined at zero and explodes on slow movers.
# MAGIC
# MAGIC All metrics are scored at the **daily** grain and at the **weekly store-SKU** grain (the level replenishment decisions are made at).

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.* statsmodels
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
import numpy as np

from retail_ds import metrics
from retail_ds.backtest import default_cutoffs, scored
from retail_ds.features import CAT_FEATURES, HORIZON
from retail_ds.models import GlobalLGBM, forecast_one_series

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/retail_demand_forecasting")

gold = read_table("gold_features")
gold["date"] = pd.to_datetime(gold["date"])
for c in CAT_FEATURES:
    gold[c] = gold[c].astype("category")
FEATS = read_table("gold_feature_list")["feature"].tolist()
CUTOFFS = default_cutoffs(gold["date"].max(), n_folds=N_FOLDS)
print("cutoffs:", [c.date() for c in CUTOFFS])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical models scaled out with Spark `applyInPandas`
# MAGIC Each store-SKU series is independent, so Spark fits them in parallel across the cluster. The same code handles 400 series or 400,000.

# COMMAND ----------

from pyspark.sql import functions as F

hist_sdf = spark.table(f"{FQ}.gold_features").select("store_id", "sku_id", "date", "demand_filled", "segment")
SCHEMA_OUT = "store_id string, sku_id string, date timestamp, model string, forecast double"


def make_udf(cut):
    def _fn(pdf: pd.DataFrame) -> pd.DataFrame:
        import sys
        if SRC not in sys.path:  # /Workspace is mounted on executors
            sys.path.insert(0, SRC)
        from retail_ds.models import forecast_one_series as f1
        pdf["date"] = pd.to_datetime(pdf["date"])
        return f1(pdf, cut, HORIZON)
    return _fn


stat_parts = []
for k, cut in enumerate(CUTOFFS, 1):
    try:
        sdf = (hist_sdf.where(F.col("date") <= F.lit(cut))
               .groupBy("store_id", "sku_id").applyInPandas(make_udf(cut), SCHEMA_OUT))
        part = sdf.toPandas()
    except Exception as e:  # fallback: driver-side loop (e.g. executor cannot see /Workspace)
        print(f"applyInPandas unavailable ({type(e).__name__}); running on driver")
        h = gold[gold["date"] <= cut][["store_id", "sku_id", "date", "demand_filled", "segment"]]
        part = pd.concat([forecast_one_series(g, cut, HORIZON) for _, g in h.groupby(["store_id", "sku_id"], observed=True)])
    part["fold"] = k
    stat_parts.append(part)
    print(f"fold {k}: {len(part):,} stat forecasts")
stat_fc = pd.concat(stat_parts, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Global LightGBM: one MLflow run per fold

# COMMAND ----------

ml_parts, fold_models = [], {}
with mlflow.start_run(run_name="backtest") as parent:
    mlflow.log_params({"horizon": HORIZON, "n_folds": N_FOLDS, "n_features": len(FEATS),
                       "n_series": gold.groupby(["store_id", "sku_id"], observed=True).ngroups})
    for k, cut in enumerate(CUTOFFS, 1):
        with mlflow.start_run(run_name=f"lgbm_fold{k}", nested=True):
            m = GlobalLGBM(FEATS).fit(gold[gold["date"] <= cut])
            te_mask = (gold["date"] > cut) & (gold["date"] <= cut + pd.Timedelta(days=HORIZON))
            te = gold.loc[te_mask, ["store_id", "sku_id", "date"]].copy()
            te["forecast"], te["model"], te["fold"] = m.predict(gold.loc[te_mask]), "lgbm_global", k
            y = gold.loc[te_mask, "demand_target"]
            mlflow.log_params({k2: v for k2, v in m.params.items() if k2 != "verbose"})
            mlflow.log_metrics({f"daily_{k2}": v for k2, v in metrics.summary(y, te["forecast"]).items()})
            ml_parts.append(te)
            fold_models[k] = m

    # ---------------- combine + ensemble + attach actuals
    fc = pd.concat([stat_fc] + ml_parts, ignore_index=True)
    fc["store_id"], fc["sku_id"] = fc["store_id"].astype(str), fc["sku_id"].astype(str)
    fc["date"] = pd.to_datetime(fc["date"])
    piv = fc[fc.model.isin(["lgbm_global", "ets"])].pivot_table(
        index=["store_id", "sku_id", "date", "fold"], columns="model", values="forecast").reset_index()
    piv["forecast"], piv["model"] = 0.7 * piv["lgbm_global"] + 0.3 * piv["ets"], "ensemble_70_30"
    fc = pd.concat([fc, piv[fc.columns.intersection(piv.columns)]], ignore_index=True)
    act = gold[["store_id", "sku_id", "date", "demand_target", "true_demand", "segment", "category",
                "on_promo", "selling_price", "unit_cost"]].copy()
    for c in ["store_id", "sku_id", "segment", "category"]:
        act[c] = act[c].astype(str)
    fc = fc.merge(act, on=["store_id", "sku_id", "date"], how="left").rename(columns={"demand_target": "actual"})
    sc = scored(fc)

    # ---------------- evaluation tables
    daily = metrics.evaluate(sc, ["model"]).assign(grain="daily_store_sku")
    weekly = metrics.evaluate_at_level(sc, ["store_id", "sku_id"], "W").assign(grain="weekly_store_sku")
    cat_week = metrics.evaluate_at_level(sc, ["category"], "W").assign(grain="weekly_category")
    by_seg = metrics.evaluate(sc, ["model", "segment"])
    by_promo = metrics.evaluate(sc, ["model", "on_promo"])
    by_fold = metrics.evaluate(sc, ["model", "fold"])
    leaderboard = pd.concat([daily, weekly, cat_week]).sort_values(["grain", "vn1"])
    for grain, g in leaderboard.groupby("grain"):
        for _, r in g.iterrows():
            mlflow.log_metrics({f"{grain}.{r.model}.wape": r.wape, f"{grain}.{r.model}.bias": r.bias,
                                f"{grain}.{r.model}.vn1": r.vn1})
    for name, t in {"leaderboard": leaderboard, "by_segment": by_seg, "by_promo": by_promo, "by_fold": by_fold}.items():
        mlflow.log_table(t, f"eval/{name}.json")
    imp = fold_models[N_FOLDS].importance()
    mlflow.log_table(imp, "eval/feature_importance.json")
    BACKTEST_RUN = parent.info.run_id

write_table(fc, "backtest_forecasts")
write_table(leaderboard, "backtest_leaderboard")
write_table(by_seg, "backtest_by_segment")
write_table(by_promo, "backtest_by_promo")
write_table(imp, "feature_importance")
display(leaderboard)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the results: what to say to the business
# MAGIC Pull these from the tables above:
# MAGIC 1. **Headline:** weekly store-SKU WAPE for `lgbm_global` compared with `naive_dow_avg` (the current method). That gap is the accuracy gain.
# MAGIC 2. **Promo weeks** (`backtest_by_promo`): this is where the Excel method fails most, because it cannot anticipate a planned promo. Promo stock-outs are the most expensive lost sales.
# MAGIC 3. **Segments** (`backtest_by_segment`): intermittent and lumpy SKUs have poor *daily* accuracy with every method. Manage them with safety stock and weekly aggregation, not better models. Set expectations honestly.
# MAGIC 4. **Bias ≈ 0** matters as much as WAPE. A persistently positive bias is silent overstock.

# COMMAND ----------

display(by_seg.pivot(index="model", columns="segment", values="vn1").reset_index())
display(by_promo.pivot(index="model", columns="on_promo", values="wape").reset_index())
display(imp.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Champion: retrain on all history, wrap as pyfunc, register in Unity Catalog
# MAGIC The pyfunc wrapper takes care of categorical encoding, so the scoring job and engineers only pass plain strings. This keeps feature logic out of downstream code.

# COMMAND ----------

from mlflow.models import infer_signature

wk = weekly.set_index("model")
champion_name = wk["vn1"].idxmin()
assert champion_name in ("lgbm_global", "ensemble_70_30"), f"unexpected champion {champion_name}"
uplift = 1 - wk.loc["lgbm_global", "wape"] / wk.loc["naive_dow_avg", "wape"]
print(f"weekly WAPE: LGBM {wk.loc['lgbm_global','wape']:.1%} vs current {wk.loc['naive_dow_avg','wape']:.1%} "
      f"-> {uplift:.0%} error reduction")

CAT_LEVELS = {c: list(gold[c].cat.categories.astype(str)) for c in CAT_FEATURES}


class DemandForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, booster, features, cat_levels):
        self.booster, self.features, self.cat_levels = booster, features, cat_levels

    def predict(self, context, model_input, params=None):
        X = model_input.copy()
        for c, lv in self.cat_levels.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=lv)
        return np.clip(self.booster.predict(X[self.features]), 0, None)


final = GlobalLGBM(FEATS).fit(gold)
example = gold[FEATS].tail(50).copy()
for c in CAT_FEATURES:
    example[c] = example[c].astype(str)
wrapper = DemandForecastModel(final.model, FEATS, CAT_LEVELS)

with mlflow.start_run(run_name="champion_lgbm_full_history"):
    mlflow.log_params({"trained_through": str(gold["date"].max().date()), "backtest_run": BACKTEST_RUN})
    mlflow.log_metrics({"bt_weekly_wape": wk.loc["lgbm_global", "wape"], "bt_weekly_bias": wk.loc["lgbm_global", "bias"],
                        "bt_weekly_vn1": wk.loc["lgbm_global", "vn1"], "wape_reduction_vs_current": uplift})
    kw = dict(python_model=wrapper, input_example=example,
              signature=infer_signature(example, wrapper.predict(None, example)),
              pip_requirements=["lightgbm==4.*", "pandas", "numpy"], registered_model_name=MODEL_NAME)
    try:
        info = mlflow.pyfunc.log_model(name="model", **kw)          # MLflow 3.x
    except TypeError:
        info = mlflow.pyfunc.log_model(artifact_path="model", **kw)  # MLflow 2.x

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", info.registered_model_version)
print(f"registered {MODEL_NAME} v{info.registered_model_version} @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production scoring: next 28 days, with P10 / P50 / P90
# MAGIC Quantiles come from the per-series backtest error distribution. They feed safety stock in notebook 05.

# COMMAND ----------

from retail_ds import features as fe

silver = read_table("silver_sales")
silver["date"] = pd.to_datetime(silver["date"])
future_input = fe.make_future_frame(silver, HORIZON, products=read_table("bronze_products"))
fut_gold = fe.build_features(future_input, read_table("bronze_products"), read_table("bronze_stores"),
                             read_table("demand_segments"))
fut = fut_gold[fut_gold["date"] > silver["date"].max()].copy()

champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
X = fut[FEATS].copy()
for c in CAT_FEATURES:
    X[c] = X[c].astype(str)
fut["p50"] = champion.predict(X)

err = sc[sc.model == "lgbm_global"].assign(e=lambda d: d.actual - d.forecast)
sig = err.groupby(["store_id", "sku_id"])["e"].std().rename("sigma").reset_index()
out = fut[["store_id", "sku_id", "date", "p50"]].astype({"store_id": str, "sku_id": str}).merge(sig, on=["store_id", "sku_id"], how="left")
out["p10"] = np.clip(out.p50 - 1.2816 * out.sigma, 0, None)
out["p90"] = out.p50 + 1.2816 * out.sigma
out["model_version"] = info.registered_model_version
write_table(out, "forecast_28d")
display(out.groupby("date")[["p10", "p50", "p90"]].sum().reset_index())