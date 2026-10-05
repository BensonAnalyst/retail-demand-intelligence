# Censored Demand Recovery on FreshRetailNet-50K
**When the shelf is empty, sales lie.** This extension of the retail demand project uses **real** hourly stock-out data to (1) estimate the demand hidden by empty shelves, (2) prove the estimate works, and (3) show that forecasting from recovered demand removes the under-forecasting bias that raw sales create.

Data: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) by Dingdong (CC BY 4.0): 50,000 store-product series of fresh products, 898 stores, 18 cities, 90 days plus a 7-day evaluation split, with **hourly sales and hourly stock-out flags**. Paper: [arXiv 2505.16319](https://arxiv.org/abs/2505.16319).

## The problem in one loop
```
shelf empty at 6pm → evening demand not recorded → model learns "low demand"
      ↑                                                            ↓
more stock-outs  ←  store orders less  ←  next week's forecast is too low
```
The paper measures this trap at **−7.37% bias** for models trained on raw sales.

## What's in it
| Step | Notebook | Output table |
|---|---|---|
| Ingest + stock-out EDA (when do shelves run empty?) | `F1_ingest_and_stockout_eda` | `frn_eda_summary` |
| Demand recovery: profile scaling and **self-supervised LightGBM**, validated by a masking test on held-out series | `F2_demand_recovery` | `frn_recovery_validation`, `frn_recovered_demand` |
| Forecast with raw vs recovered targets, 2 folds, 4 evaluation views, MLflow | `F3_forecast_raw_vs_recovered` | `frn_forecast_results` |
| Business impact: lost demand by category, store and hour | `F4_business_impact` | `frn_business_kpis`, `frn_lost_by_*` |
| Same analysis as one Kaggle notebook | `kaggle/freshretailnet_censored_demand.ipynb` | – |

## Two ideas worth discussing
1. **Self-supervised recovery.** True demand on a stock-out day is never observed. So take *in-stock* days, hide hours using **real stock-out patterns copied from real stock-out days**, and train a model to rebuild the full day. The same trick on held-out series gives an honest accuracy and bias score for the recovery.
2. **The evaluation trap.** Scoring only on in-stock days, as the paper does, is a selected sample: high-demand days are the ones that sell out. On a simulation where true demand was known, that view **reversed the ranking** and made the under-forecasting model look best. Scoring all days against recovered demand (view C, with C2 as an independent check) tracked the true bias within about 1 point.

| Simulation check (true demand known) | Bias vs TRUE demand | View A (in-stock days) | View C (recovered truth) |
|---|---|---|---|
| Trained on raw sales | −10.5% | −0.2% ✗ looks fine | −11.5% ✓ |
| Trained on recovered demand (LightGBM) | −0.2% | +7.2% ✗ looks worse | −1.3% ✓ |

*Simulation only, used to choose the evaluation design. Real-data results come from your own run (below).*

## Results on the real data
*Fill in from F2, F3 and F4 after your run. Note the series sample size, e.g. 10,000 of 50,000.*

| Result | Value |
|---|---|
| Stock-out days / availability in operating hours | __ / __ |
| Recovery masking test: bias without recovery → with LightGBM recovery | __ → __ |
| Forecast bias, view C: raw-sales training → recovered training | __ → __ |
| Forecast WAPE, view C: raw → recovered | __ → __ |
| Demand lost to stock-outs (% of total) | __ |
| Share of lost demand in the top 20% of stores | __ |

## Run it
**Databricks** (same workspace as the main project)
1. Upload this `freshretail` folder to the GitHub repo root, then click **Pull** in your Databricks Git folder.
2. Open `freshretail/notebooks/F1_ingest_and_stockout_eda` and click **Run all**. It downloads the data into the Volume `workspace.retail_ds.raw`. If your workspace has no internet access, download `train.parquet` and `eval.parquet` from Hugging Face and upload them to `/Volumes/workspace/retail_ds/raw/freshretail/`.
3. Run F2, then F3 (the longest, about 15–30 minutes at 10,000 series), then F4.
4. `n_series` (default 10,000) controls the sample. Whole stores are sampled, with a fixed seed, so every notebook uses the same subset.

**Kaggle**
1. **Code → New Notebook → File → Import Notebook**, and upload `kaggle/freshretailnet_censored_demand.ipynb`.
2. **Settings → Internet: On** (needs a phone-verified account). Alternatively, upload the two parquet files as your own Kaggle Dataset (credit Dingdong, CC BY 4.0) and add it as input. The notebook finds `train.parquet` automatically.
3. **Run All**, check the results, fill in the Conclusions cell, then **Save Version → Save & Run All** and set the notebook to **Public**.

**Local / CI**
```bash
python freshretail/scripts/make_mock_data.py --out /tmp/frn_mock     # schema-identical mock, NOT real data
python freshretail/scripts/run_local.py --data /tmp/frn_mock
python freshretail/scripts/notebook_smoke_test.py /tmp/frn_mock/data
pytest -q freshretail/tests
python freshretail/scripts/build_kaggle_notebook.py                    # regenerate the Kaggle notebook from src/
```

## Limitations
- The dataset's sales are globally normalised, so results are in relative units, not dollars.
- Weather uses actual values as a stand-in for a weather forecast.
- Recovery treats lost demand as truly lost. Some shoppers substitute another product, so recovered demand is an upper bound on lost *sales*.
- The models are intentionally simple (LightGBM). The paper's deep-learning imputers (e.g. TimesNet) are the obvious next comparison.
