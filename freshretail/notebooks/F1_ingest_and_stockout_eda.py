# Databricks notebook source
# MAGIC %md
# MAGIC # F1 · Ingest FreshRetailNet-50K and explore stock-outs
# MAGIC
# MAGIC **Why this dataset:** it is one of very few public retail datasets with **real, hourly stock-out records**:
# MAGIC 50,000 store-product series of fresh products from 898 stores in 18 cities (Dingdong, 2025, CC BY 4.0), with 90 days of training data plus 7 evaluation days.
# MAGIC In a real store you never see the demand you lost when the shelf was empty. This data shows *when* the shelf was empty, so we can estimate it.
# MAGIC
# MAGIC **Getting the data** (choose one):
# MAGIC 1. **Automatic:** this notebook downloads from Hugging Face into a Unity Catalog Volume. This needs outbound internet from your workspace.
# MAGIC 2. **Manual:** download `data/train.parquet` and `data/eval.parquet` from
# MAGIC    https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/tree/main/data and upload them in **Catalog → workspace → retail_ds → raw → Create directory `freshretail` → Upload**, naming them `train.parquet` and `eval.parquet`.

# COMMAND ----------

# MAGIC %pip install -q huggingface_hub lightgbm==4.*
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./F0_config

# COMMAND ----------

os.environ.setdefault("HF_HOME", "/tmp/hf")
if not (os.path.exists(f"{DATA_DIR}/train.parquet") and os.path.exists(f"{DATA_DIR}/eval.parquet")):
    from frn import data as frn_data
    try:
        print(frn_data.download(DATA_DIR))
    except Exception as e:
        raise RuntimeError(f"Download failed ({type(e).__name__}: {e}). Use the manual upload option above.") from e
print(os.listdir(DATA_DIR))

# COMMAND ----------

full, last_day, chk = load_data()
print("orientation check:", chk)
assert chk["share_rows_matching_stock_hour6_22_cnt"] > 0.95, "stock flags do not match stock_hour6_22_cnt; inspect the data"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data-quality check: what does `hours_stock_status = 1` mean?
# MAGIC The dataset card only says "hourly out-of-stock status". We verify the direction from the data: hours flagged as out of stock should show close to zero sales.
# MAGIC `pipeline.prepare` flips the flag automatically if the data says otherwise, and the result is printed above. The flags also match `stock_hour6_22_cnt` for almost every row.

# COMMAND ----------

from frn import recovery
from frn.data import OP_HOURS

df = full.df
clean = recovery.clean_days(full)
train_mask = (df["day"] <= last_day).to_numpy()
summary = pd.DataFrame([
    {"metric": "store-product-days (train)", "value": int(train_mask.sum())},
    {"metric": "days with any stock-out in 06:00-21:59", "value": float((~clean[train_mask]).mean())},
    {"metric": "availability (in-stock share of operating hours)", "value": float(1 - full.oos[train_mask][:, OP_HOURS].mean())},
    {"metric": "avg stock-out hours on stock-out days", "value": float(full.oos[train_mask & ~clean][:, OP_HOURS].sum(1).mean())},
    {"metric": "avg daily sales, stock-out days", "value": float(full.sales[train_mask & ~clean].sum(1).mean())},
    {"metric": "avg daily sales, in-stock days", "value": float(full.sales[train_mask & clean].sum(1).mean())},
])
write_table(summary, "frn_eda_summary")
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### When do shelves run empty?
# MAGIC Look for a rising stock-out rate through the day. That pattern is a **sell-out**: morning stock was not enough for the evening peak, a replenishment-timing problem that a forecast can fix.

# COMMAND ----------

import matplotlib.pyplot as plt

hours = np.arange(24)
oos_rate = full.oos[train_mask].mean(0)
sales_profile = full.sales[train_mask & clean].sum(0)
sales_profile = sales_profile / sales_profile.sum()
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(hours, sales_profile * 100, color="#9db7d5", label="share of daily sales (in-stock days)")
ax1.set_xlabel("hour of day"); ax1.set_ylabel("share of daily sales (%)")
ax2 = ax1.twinx()
ax2.plot(hours, oos_rate * 100, color="#c0392b", marker="o", label="out-of-stock rate")
ax2.set_ylabel("out-of-stock rate (%)")
ax1.set_title("Demand peaks vs when shelves are empty")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9)); ax1.grid(alpha=.3)
display(fig); plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The censoring problem in one picture
# MAGIC One product in one store, hour by hour. Red bands mark out-of-stock hours. The sales recorded there are not the demand.

# COMMAND ----------

g = df.groupby(["store_id", "product_id"]).ngroup().to_numpy()
oos_days = pd.Series(~clean).groupby(g).mean()
example = oos_days[(oos_days > 0.25) & (oos_days < 0.6)].index[0]
rows = np.where((g == example) & train_mask)[0][-7:]
s = full.sales[rows].ravel(); o = full.oos[rows].ravel()
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(s, color="#2c3e50", lw=1.2, label="hourly sales")
for i in np.where(o == 1)[0]:
    ax.axvspan(i - .5, i + .5, color="#e74c3c", alpha=.25, lw=0)
ax.set_xticks(range(0, 24 * 7, 24)); ax.set_xticklabels([d.strftime("%a %d %b") for d in df.loc[rows, "dt"]])
ax.set_title("Last 7 days of one series: red = out of stock"); ax.legend(); ax.grid(alpha=.3)
display(fig); plt.close(fig)
