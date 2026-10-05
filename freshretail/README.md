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
| Ingest + stock-out EDA (when do shelves run empty? is demand trending?) | `F1_ingest_and_stockout_eda` | `frn_eda_summary`, `frn_weekly_trend` |
| Demand recovery: profile scaling and **self-supervised LightGBM**, validated by a masking test on held-out series | `F2_demand_recovery` | `frn_recovery_validation`, `frn_recovered_demand`, `frn_recovery_importance` |
| Forecast with raw vs recovered targets, walk-forward (warm-up + 2 folds), 4 evaluation views, bias calibration, error by days ahead, MLflow | `F3_forecast_raw_vs_recovered` | `frn_forecast_results`, `frn_forecast_by_horizon`, `frn_calibration_scales` |
| Business impact: lost demand by category, store and the hour the shelf first empties; sales-uplift scenario; one-cell text export | `F4_business_impact` | `frn_business_kpis`, `frn_lost_by_*` |
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
Sample: **10,000 of 50,000 series** (whole stores, fixed seed), 903,960 store-product-days of training data. Forecast results are the average of 2 folds (cutoff day 82 and the official evaluation week); both folds agree closely.

**1. How big is the problem?**
| Result | Value |
|---|---|
| Store-product-days with a stock-out (06:00–21:59) | **44.4%** |
| Shelf availability in operating hours | **80.0%** |
| Out-of-stock rate through the day | ~5% at 07:00 → ~41% at 21:00 (evening sell-out) |
| Avg daily sales, stock-out days vs in-stock days | 1.030 vs 0.973: **stock-outs hit the busier days** |
| Demand lost to stock-outs (% of total demand) | **18.6%** (≈ 23% on top of recorded sales) |
| Lost demand in the top 20% of stores | **36%**: spread across the network, so a *system* problem, not a few bad stores |
| Top 3 categories (of 28) | hold **53%** of lost demand; one category loses 26% of its demand at near-normal availability, because it empties at peak hours |

**2. Does recovery work?** Masking test on 100,568 held-out in-stock days, with 42% of demand hidden using real stock-out patterns:
| Method | WAPE | Bias |
|---|---|---|
| No recovery | 42.1% | −42.1% |
| Profile scaling | 25.5% | +0.9% |
| **Self-supervised LightGBM** | **18.7%** | **+0.1%** |

*(Paper's best deep-learning imputer, TimesNet: 27.6% / +1.4% on its own test design, so a reference point, not like-for-like.)*

**3. Does it fix the forecast?** Same 7-day LightGBM forecaster, different training target:
| Trained on | Bias, view A (in-stock days, paper protocol) | WAPE, A | Bias, view C (all days vs recovered demand) | WAPE, C |
|---|---|---|---|---|
| Raw sales | −12.8% | 32.0% | **−23.0%** | 34.0% |
| Recovered: profile | +5.1% | 32.4% | −4.3% | 29.1% |
| **Recovered: LightGBM** | **−0.8%** | **30.1%** | −10.1% | **28.5%** |

*(Paper: raw −7.37% → recovered +2.58% bias, WAPE 31.75% → 29.02%.)*

**What the numbers say**
- Raw-sales training under-forecasts on every view. Recovery cuts WAPE (view C) from **34.0% to 28.5%**.
- **The evaluation trap is real.** The paper's protocol (view A) shows raw training as 13% short; against recovered demand it is **23% short**. Scoring only in-stock days hides almost half the problem.
- **A shared level gap of ~10–12%** remains: every model is low even against its *own* target (raw vs raw sales −11%, LightGBM-trained vs LightGBM-recovered −10%). Censoring cannot explain that, so it is a separate level issue. `run_backtest` now adds a **walk-forward bias calibration** (multiplier learned on the previous week only) to address it; results appear as `+ calibration` rows.
- View C uses the LightGBM recovery as truth, which slightly favours the LightGBM-trained model. Against the profile truth (C2) it still beats raw sales: WAPE 34.9% vs 39.9%.
- Slow sellers are harder (WAPE ~36% vs ~26% for fast sellers). Profile-trained forecasts over-forecast slow sellers by ~15% on in-stock days, which for fresh food means waste.

## Run it
**Databricks** (same workspace as the main project)
1. Upload this `freshretail` folder to the GitHub repo root, then click **Pull** in your Databricks Git folder.
2. Open `freshretail/notebooks/F1_ingest_and_stockout_eda` and click **Run all**. It downloads the data into the Volume `workspace.retail_ds.raw`. If your workspace has no internet access, download `train.parquet` and `eval.parquet` from Hugging Face and upload them to `/Volumes/workspace/retail_ds/raw/freshretail/`.
3. Run F2, then F3 (the longest, about 20–30 minutes at 10,000 series, since it trains a warm-up fold plus 2 folds), then F4.
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
