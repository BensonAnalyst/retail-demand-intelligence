"""Build a single self-contained Kaggle notebook from src/frn (so the two never drift).

    python freshretail/scripts/build_kaggle_notebook.py   -> freshretail/kaggle/freshretailnet_censored_demand.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# When the shelf is empty, sales lie: recovering censored demand in fresh retail

**Dataset:** [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong, 2025, CC BY 4.0):
50,000 store-product series of fresh products, 898 stores, 90 days plus a 7-day evaluation split, with **hourly sales and hourly stock-out flags**.

**The problem.** On a stock-out day, recorded sales are only the demand from the hours the product was on the shelf.
Train a forecaster on raw sales and it learns the shortage: it under-forecasts, the store under-orders, and the shelf runs empty again.

**What this notebook does**
1. Explores **when** shelves run empty.
2. **Recovers latent demand** with two methods: hourly-profile scaling, and a *self-supervised* LightGBM trained by hiding hours on in-stock days using real stock-out patterns.
3. **Validates recovery** with a masking test on held-out series, where the hidden demand is known.
4. Trains the same 7-day forecaster on raw vs recovered demand, and evaluates it from **three views of the truth**, because the obvious evaluation (in-stock days only) is itself biased.
5. Sizes the **demand lost** to stock-outs, by category, store and hour.

*Paper:* [FreshRetailNet-50K, arXiv 2505.16319](https://arxiv.org/abs/2505.16319).
""")
code("""
!pip install -q lightgbm huggingface_hub
""")
md("## The `frn` package\nThe same code runs the Databricks version of this project.")
code("%%writefile frn/__init__.py\n# FreshRetailNet censored-demand toolkit\n".replace("%%writefile frn/__init__.py", "import os\nos.makedirs('frn', exist_ok=True)"))
for mod in ["data", "metrics", "recovery", "forecast", "pipeline"]:
    code(f"%%writefile frn/{mod}.py\n" + (ROOT / "src" / "frn" / f"{mod}.py").read_text())
code("""
open("frn/__init__.py", "w").write("")
import glob, os, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from frn import data, pipeline, recovery
from frn.data import OP_HOURS

# ---- settings
N_SERIES = 20000          # of 50,000 (0 = all; needs ~20+ GB RAM and a long run)
# data: a Kaggle input copy if attached, otherwise download from Hugging Face (Internet ON in settings)
found = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
if os.environ.get("FRN_DATA_DIR"):
    DATA_DIR = os.environ["FRN_DATA_DIR"]
elif found:
    DATA_DIR = os.path.dirname(found[0])
else:
    DATA_DIR = "/kaggle/working/frn_data"
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf")
    data.download(DATA_DIR)
print("data:", DATA_DIR)
keys = data.sample_series(f"{DATA_DIR}/train.parquet", N_SERIES or None, seed=42)
full, last_day, chk = pipeline.prepare(data.load_panel(f"{DATA_DIR}/train.parquet", keys),
                                       data.load_panel(f"{DATA_DIR}/eval.parquet", keys))
print(f"{len(full.df):,} rows | train days 0..{last_day} | orientation check: {chk}")
""")
md("""
## 1. When do shelves run empty?
First a data-quality check: hours flagged out-of-stock should show close to zero sales. The code verifies the flag's direction rather than assuming it.
""")
code("""
clean = recovery.clean_days(full)
tr = (full.df["day"] <= last_day).to_numpy()
print(f"stock-out days: {(~clean[tr]).mean():.1%} | availability 06-22h: {1-full.oos[tr][:, OP_HOURS].mean():.1%}")
prof = full.sales[tr & clean].sum(0); prof = prof / prof.sum()
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(range(24), prof * 100, color="#9db7d5", label="share of daily sales (in-stock days)")
ax2 = ax1.twinx(); ax2.plot(range(24), full.oos[tr].mean(0) * 100, "o-", color="#c0392b", label="out-of-stock rate")
ax1.set_xlabel("hour"); ax1.set_ylabel("% of daily sales"); ax2.set_ylabel("% out of stock")
ax1.set_title("Demand peaks vs when shelves are empty"); fig.legend(loc="upper left", bbox_to_anchor=(.1, .9)); plt.show()
""")
md("""
## 2. Recover latent demand, and prove it works
**Masking test:** on held-out series, take in-stock days, hide hours using *real* stock-out patterns copied from real stock-out days, recover, and compare with the demand we hid.
`observed` shows how much demand the stock-out patterns hide. A good method has bias ≈ 0 and the lowest WAPE.
""")
code("""
val = recovery.validate_recovery(full, last_day)
val
""")
md("""
## 3. Train the forecaster on raw vs recovered demand
The same global LightGBM (7-day horizon, lags ≥ 7, known discount, holiday and activity plan) is trained with three different targets, in two folds.

**Evaluating it is the subtle part.** In-stock days are not a random sample: high-demand days are the ones that sell out.
So **view A** (in-stock days only, the paper's protocol) skews towards low-demand days and flatters a model that under-forecasts.
**View C** scores all days against recovered demand, with **C2** using the simpler profile recovery as an independent check.
On a simulation where true demand was known, view C tracked the true bias within about 1 point, while view A reversed the ranking.
""")
code("""
results = []
for cutoff in (last_day - 7, last_day):
    res, rec, _ = pipeline.run_fold(full, cutoff)
    results.append(res)
results = pd.concat(results, ignore_index=True)
main = results[results.eval_view.str.match(r"^(A|B|C|C2)\\.")]
print("BIAS (WPE): negative = under-forecast")
display(main.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wpe_bias").round(4))
print("WAPE")
display(main.pivot_table(index=["cutoff_day", "training_target"], columns="eval_view", values="wape").round(4))
""")
code("""
results[~results.eval_view.str.match(r"^(A|B|C|C2)\\.")].round(4)
""")
md("## 4. How much demand do empty shelves cost?")
code("""
total, by_cat, by_store, by_hour = pipeline.lost_demand_tables(full, rec, last_day)
print({k: round(v, 4) for k, v in total.items()})
display(by_cat.round(4))
top = by_store.head(max(1, len(by_store) // 5))
print(f"top 20% of stores hold {top.lost.sum() / by_store.lost.sum():.0%} of lost demand")
""")
md("""
## Conclusions
*(Fill these in from your own run. The numbers depend on the data.)*

1. **Lost demand:** about __% of demand is invisible in sales because of stock-outs, concentrated in __ and in the __ hours.
2. **Recovery works:** on the masking test, self-supervised LightGBM recovers hidden demand with __% WAPE and __% bias, against __% bias with no recovery.
3. **Forecasts trained on raw sales under-forecast** by __% (view C). Training on recovered demand cuts that to __%.
4. **Evaluation matters:** scoring only in-stock days (view A) gives __, which is a selection effect, not a better model.

**Business takeaway:** a store that forecasts from raw sales keeps re-learning its own shortages. Hourly shelf-availability data is what makes the loop breakable.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "kaggle" / "freshretailnet_censored_demand.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
