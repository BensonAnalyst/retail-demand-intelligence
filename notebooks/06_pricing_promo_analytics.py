# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Pricing & promotion analytics
# MAGIC
# MAGIC **Questions from the commercial team**
# MAGIC 1. How sensitive is each category to price? (elasticity)
# MAGIC 2. How much does a promo lift sales beyond the price cut, and how much does it steal from sibling SKUs?
# MAGIC 3. Which promo depths make money and which only move volume?
# MAGIC 4. Where are regular prices leaving margin on the table?
# MAGIC
# MAGIC **Method:** Poisson GLM per category with store-SKU fixed effects, SKU-level trends, day-of-week, month and holiday controls.
# MAGIC Because the data is synthetic we **know the true elasticities**, so the estimator is *validated* here rather than taken on trust.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from retail_ds import pricing
from retail_ds.data_gen import CANNIBALISATION

gold = read_table("gold_features")
gold["date"] = pd.to_datetime(gold["date"])
products = read_table("bronze_products")
cats = sorted(gold["category"].unique())

naive = pd.DataFrame([pricing.estimate_category_response(gold, c, control_trend=False) for c in cats])
resp = pd.DataFrame([pricing.estimate_category_response(gold, c, control_trend=True) for c in cats])
resp["elasticity_naive_no_trend"] = naive["elasticity"].values
truth = products.groupby("category")[["true_elasticity", "true_promo_uplift"]].mean().reset_index()
resp = resp.merge(truth, on="category")
resp["true_cannibalisation"] = CANNIBALISATION
resp["elasticity_error_pct"] = resp.elasticity / resp.true_elasticity - 1
write_table(resp, "price_response")
display(resp[["category", "elasticity_naive_no_trend", "elasticity", "elasticity_lo", "elasticity_hi", "true_elasticity",
              "promo_uplift", "true_promo_uplift", "cannibalisation_per_sibling", "true_cannibalisation"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lesson: confounding
# MAGIC Without SKU trends, the elasticity for some categories is badly understated. Regular prices creep up with inflation while some SKUs are growing and others declining, and the naive model credits that trend to price.
# MAGIC With trends controlled, estimates land within roughly 10% of the truth, but the confidence intervals **do not always cover it**, so some residual confounding remains.
# MAGIC
# MAGIC ➡️ **Recommendation to the business:** use these elasticities to *rank and shortlist* price moves, then **confirm with a store-level A/B test** (test stores vs matched controls) before a network roll-out.

# COMMAND ----------

last90 = gold[gold.date > gold.date.max() - pd.Timedelta(days=90)]
n_days, n_stores = last90.date.nunique(), gold.store_id.nunique()
sku = (last90[last90.on_promo == 0].groupby("sku_id")
       .agg(category=("category", "first"), regular_price=("regular_price", "median"),
            unit_cost=("unit_cost", "first"), daily_units=("demand_filled", "sum")).reset_index())
sku["daily_units"] /= n_days
recs = pricing.price_recommendations(sku, resp, band=0.10)
write_table(recs, "price_recommendations")
display(recs[["sku_id", "category", "regular_price", "rec_price", "price_change_pct", "elasticity", "annual_margin_gain"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promo depth economics (per store, per promo day, including cannibalisation)
# MAGIC The EDA in notebook 02 shows that deeper discounts sell more units. **This table shows whether they make more money.**

# COMMAND ----------

rows = []
for _, r in resp.iterrows():
    cs = sku[sku.category == r.category]
    n_sib = gold[gold.category == r.category].sku_id.nunique() - 1
    e = pricing.promo_depth_economics(
        r, cs.regular_price.mean(), cs.unit_cost.mean(), cs.daily_units.mean() / n_stores,
        ((cs.regular_price - cs.unit_cost) * cs.daily_units / n_stores).mean(), n_sib)
    e.insert(0, "category", r.category)
    rows.append(e)
promo = pd.concat(rows, ignore_index=True)
write_table(promo, "promo_economics")
display(promo.pivot(index="depth", columns="category", values="incremental_margin").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC **Story for the commercial team:** in *elastic* categories (Beverages, Snacks), shallow promos (10–15%) pay for themselves. In *inelastic* categories (Household, Personal Care), most promos only move volume. Their extra units do not cover the margin given away plus the sales taken from sibling SKUs.
# MAGIC Not modelled here (future work): post-promo dip / pantry loading, halo effects on other categories, and vendor funding. Vendor funding often changes the answer, so capture it in the promo calendar.
