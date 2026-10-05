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
# MAGIC ### The story for a retail operations audience
# MAGIC 1. **"We lose X% of demand to empty shelves."** That is demand customers came for, not a forecast error.
# MAGIC 2. **"It is concentrated."** Show the share of lost demand in the top 20% of stores and the top categories. Start fixing there.
# MAGIC 3. **"It happens at a predictable time."** If the out-of-stock rate climbs through the evening (F1 chart), the fix is replenishment timing or a second delivery, not more stock all day.
# MAGIC 4. **"Our forecasts were learning the shortage."** Training on raw sales bakes the shortage into next week's order (F3, view C bias). Recovering demand first breaks that loop.
# MAGIC 5. **Why shelf monitoring matters:** without hourly stock-out flags (from shelf cameras or scans), none of this is measurable. The stock-out flag is what makes demand recovery possible.
