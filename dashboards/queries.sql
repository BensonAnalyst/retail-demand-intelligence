-- AI/BI Dashboard datasets (Databricks SQL). Replace workspace.retail_ds with your catalog.schema.
-- Suggested pages: 1) Executive Before/After  2) Forecast accuracy  3) Inventory trade-off  4) Pricing & promo  5) Model health

-- 1. Executive summary (counter tiles + table)
SELECT kpi, before, after, note FROM workspace.retail_ds.exec_summary;

-- 2a. Model leaderboard at the replenishment grain
SELECT model, ROUND(wape*100,1) AS wape_pct, ROUND(bias*100,1) AS bias_pct, ROUND(vn1,3) AS vn1
FROM workspace.retail_ds.backtest_leaderboard WHERE grain = 'weekly_store_sku' ORDER BY vn1;

-- 2b. Actual vs forecast by week & category (line chart; the "does it catch 11.11 / 12.12?" view)
SELECT date_trunc('week', date) AS week, category, model,
       SUM(actual) AS actual, SUM(forecast) AS forecast
FROM workspace.retail_ds.backtest_forecasts
WHERE model IN ('lgbm_global', 'naive_dow_avg') AND actual IS NOT NULL
GROUP BY ALL ORDER BY week;

-- 2c. Where it's hard: accuracy by demand segment
SELECT model, segment, ROUND(wape*100,1) AS wape_pct FROM workspace.retail_ds.backtest_by_segment;

-- 3. Service vs working-capital frontier (scatter)
SELECT policy, target_service, fill_rate, avg_inventory_value, total_cost
FROM workspace.retail_ds.inventory_policy_results;

-- 4a. Price response with confidence intervals
SELECT category, elasticity, elasticity_lo, elasticity_hi, promo_uplift, cannibalisation_per_sibling
FROM workspace.retail_ds.price_response;

-- 4b. Promo depth heatmap: which promos make money
SELECT category, depth, incremental_margin FROM workspace.retail_ds.promo_economics;

-- 5. Model health traffic lights
SELECT week, category, status, ROUND(wape*100,1) AS wape_pct, ROUND(bias*100,1) AS bias_pct
FROM workspace.retail_ds.monitoring_weekly_health ORDER BY week DESC;

-- 6. Planner's action list: next week's order-up-to levels
SELECT store_id, sku_id, ROUND(demand_L_plus_R,1) AS expected_demand, ROUND(safety_stock,1) AS safety_stock, order_up_to
FROM workspace.retail_ds.replenishment_recommendation ORDER BY order_up_to DESC;
