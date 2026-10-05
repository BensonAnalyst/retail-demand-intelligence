# Databricks notebook source
# MAGIC %md
# MAGIC # F4 · Business impact: how much demand do empty shelves cost, and where?
# MAGIC
# MAGIC This turns the recovered demand from F2 into the numbers a category or store-operations manager acts on:
# MAGIC **share of demand lost**, **availability** during operating hours, **which categories and stores lose most**, and **what time of day** shelves run empty.

# COMMAND ----------

# MAGIC %run ./F0_config

# COMMAND ----------

rec = spark.table(f"{FQ}.frn_recovered_demand").toPandas()
rec["lost"] = rec["rec_model"] - rec["observed"]
tot = rec["rec_model"].sum()
kpis = pd.DataFrame([
    {"kpi": "Estimated demand lost to stock-outs (% of total)", "value": rec["lost"].sum() / tot * 100},
    {"kpi": "Store-product-days with a stock-out (%)", "value": (rec["oos_hours"] > 0).mean() * 100},
    {"kpi": "Availability in operating hours 06:00-21:59 (%)", "value": (1 - rec["oos_hours"].mean() / 16) * 100},
    {"kpi": "Lost demand concentrated in top 20% of stores (%)",
     "value": rec.groupby("store_id")["lost"].sum().sort_values(ascending=False)
                 .pipe(lambda s: s.head(max(1, len(s) // 5)).sum() / s.sum() * 100)},
    {"kpi": "Lost demand as % of current recorded sales", "value": rec["lost"].sum() / rec["observed"].sum() * 100},
    {"kpi": "Sales uplift if 25% of lost demand is captured (%) [scenario]",
     "value": 0.25 * rec["lost"].sum() / rec["observed"].sum() * 100},
    {"kpi": "Lost demand from shelves empty before 16:00 (%)",
     "value": rec.loc[(rec["first_oos_hour"] >= 0) & (rec["first_oos_hour"] < 16), "lost"].sum() / rec["lost"].sum() * 100},
])
write_table(kpis, "frn_business_kpis")
display(kpis)

# COMMAND ----------

by_cat = (rec.groupby("first_category_id")
            .agg(demand=("rec_model", "sum"), lost=("lost", "sum"), avg_oos_hours=("oos_hours", "mean"))
            .assign(lost_pct=lambda d: d.lost / d.demand * 100, availability_pct=lambda d: (1 - d.avg_oos_hours / 16) * 100)
            .sort_values("lost", ascending=False).reset_index())
write_table(by_cat, "frn_lost_by_category")
display(by_cat)

# COMMAND ----------

by_store = (rec.groupby("store_id")
              .agg(demand=("rec_model", "sum"), lost=("lost", "sum"), avg_oos_hours=("oos_hours", "mean"))
              .assign(lost_pct=lambda d: d.lost / d.demand * 100)
              .sort_values("lost", ascending=False).reset_index())
write_table(by_store, "frn_lost_by_store")
display(by_store.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### When does the shelf first empty, and what does it cost?
# MAGIC Availability % treats every empty hour the same. Lost demand does not: a shelf empty from 14:00 misses the whole evening peak, one empty from 21:00 misses almost nothing.

# COMMAND ----------

so = rec[rec["first_oos_hour"] >= 0]
by_hour = (so.groupby("first_oos_hour")
             .agg(stockout_days=("lost", "size"), lost=("lost", "sum"), avg_lost_per_day=("lost", "mean"))
             .reset_index())
by_hour["lost_share_pct"] = by_hour["lost"] / by_hour["lost"].sum() * 100
by_hour["cum_lost_share_pct"] = by_hour["lost_share_pct"].cumsum()
write_table(by_hour, "frn_lost_by_first_oos_hour")
display(by_hour)

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(by_hour["first_oos_hour"], by_hour["lost_share_pct"], color="#c0392b", alpha=.75, label="share of lost demand")
ax1.set_xlabel("hour the shelf first ran empty"); ax1.set_ylabel("share of lost demand (%)")
ax2 = ax1.twinx()
ax2.plot(by_hour["first_oos_hour"], by_hour["avg_lost_per_day"], "o-", color="#2c3e50", label="avg lost demand per stock-out day")
ax2.set_ylabel("avg lost per stock-out day (scaled units)")
ax1.set_title("Early sell-outs cost the most per day"); fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); ax1.grid(alpha=.3)
display(fig); plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The story for a retail operations audience
# MAGIC 1. **"We lose X% of demand to empty shelves."** That is demand customers came for, not a forecast error.
# MAGIC 2. **"It is concentrated."** Show the share of lost demand in the top 20% of stores and the top categories. Start fixing there.
# MAGIC 3. **"It happens at a predictable time."** If the out-of-stock rate climbs through the evening (F1 chart), the fix is replenishment timing or a second delivery, not more stock all day.
# MAGIC 4. **"Our forecasts were learning the shortage."** Training on raw sales bakes the shortage into next week's order (F3, view C bias). Recovering demand first breaks that loop.
# MAGIC 5. **Why shelf monitoring matters:** without hourly stock-out flags (from shelf cameras or scans), none of this is measurable. The stock-out flag is what makes demand recovery possible.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export: one text block with every result (copy everything this cell prints)

# COMMAND ----------

def show(t, d, n=40):
    print(f"\n### {t}\n" + d.head(n).round(4).to_string(index=False))

tb = lambda n: spark.table(f"{FQ}.{n}").toPandas()  # noqa: E731
print("n_series:", N_SERIES)
for title, name, n in [("EDA SUMMARY", "frn_eda_summary", 40), ("WEEKLY TREND", "frn_weekly_trend", 40),
                       ("RECOVERY VALIDATION", "frn_recovery_validation", 40),
                       ("RECOVERY FEATURE IMPORTANCE", "frn_recovery_importance", 20),
                       ("CALIBRATION MULTIPLIERS", "frn_calibration_scales", 20)]:
    try:
        show(title, tb(name), n)
    except Exception as e:  # a table from a notebook not yet rerun
        print(f"\n### {title}: not available ({type(e).__name__})")
fr = tb("frn_forecast_results")
main = fr[fr.eval_view.str.match(r"^(A|B|C|C2)\.")]
show("BIAS (WPE) by fold", main.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wpe_bias").reset_index())
show("WAPE by fold", main.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wape").reset_index())
show("AVERAGE OF FOLDS", main.groupby(["training_target", "eval_view"])[["wape", "wpe_bias"]].mean().reset_index(), 60)
show("BY SALE GROUP", fr[~fr.eval_view.str.match(r"^(A|B|C|C2)\.")][["cutoff_day", "training_target", "eval_view", "wape", "wpe_bias"]], 60)
try:
    hz = tb("frn_forecast_by_horizon")
    show("BIAS BY DAYS AHEAD (view C)", hz.pivot_table(index="training_target", columns="horizon_days", values="wpe_bias").reset_index())
except Exception as e:
    print(f"\n### BIAS BY DAYS AHEAD: not available ({type(e).__name__})")
show("BUSINESS KPIS", tb("frn_business_kpis"))
show("LOST BY CATEGORY (top 10)", tb("frn_lost_by_category"), 10)
show("LOST BY FIRST STOCK-OUT HOUR", tb("frn_lost_by_first_oos_hour"), 20)
