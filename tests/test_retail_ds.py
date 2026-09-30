import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_ds import cleaning, data_gen, features, inventory, metrics, models, monitoring, pricing  # noqa: E402


# ---------------- metrics
def test_metrics_known_values():
    y, f = np.array([10, 0, 30]), np.array([12, 1, 24])
    assert metrics.wape(y, f) == pytest.approx(9 / 40)
    assert metrics.bias(y, f) == pytest.approx(-3 / 40)
    assert metrics.vn1_score(y, f) == pytest.approx(9 / 40 + 3 / 40)
    assert metrics.mape(y, f) == pytest.approx((0.2 + 0.2) / 2)  # zero actual excluded


def test_metrics_ignore_nan_actuals():
    assert metrics.wape([10, np.nan], [10, 999]) == 0


# ---------------- models
def test_croston_constant_series():
    y = np.array([0, 4, 0, 4, 0, 4] * 20, float)
    f = models.croston_sba(y, 7)
    assert np.allclose(f, f[0]) and 1.5 < f[0] < 2.1  # ~ 4 every 2 days, SBA-deflated


def test_naive_dow_avg_repeats_weekly_profile():
    y = np.tile(np.arange(7), 5).astype(float)
    assert np.allclose(models.naive_dow_avg(y, 14), np.tile(np.arange(7), 2))


# ---------------- features: leakage guard
@pytest.fixture(scope="module")
def gold():
    raw = data_gen.generate(n_stores=2, n_skus=4, start="2024-01-01", end="2024-12-31", inject_dq_issues=False)
    silver, _ = cleaning.clean_sales(raw["sales"])
    seg = cleaning.classify_demand(silver)
    return features.build_features(silver, raw["products"], raw["stores"], seg)


def test_lags_respect_horizon(gold):
    s = gold[(gold.store_id == "S00") & (gold.sku_id == "SKU000")].reset_index(drop=True)
    H = features.HORIZON
    assert s.loc[100, f"lag_{H}"] == s.loc[100 - H, "demand_filled"]
    # rolling mean at t must only use data up to t-H
    assert s.loc[100, "rmean_7"] == pytest.approx(s.loc[100 - H - 6:100 - H, "demand_filled"].mean())


def test_target_columns_not_in_features(gold):
    feats = features.feature_columns(gold)
    for leak in ["units_sold", "true_demand", "demand_target", "demand_filled", "stockout_flag"]:
        assert leak not in feats


# ---------------- cleaning
def test_cleaning_removes_duplicates_and_negatives():
    raw = data_gen.generate(n_stores=2, n_skus=4, start="2024-01-01", end="2024-06-30")
    silver, dq = cleaning.clean_sales(raw["sales"])
    assert not silver.duplicated(["store_id", "sku_id", "date"]).any()
    assert (silver["demand_target"].dropna() >= 0).all()
    assert silver.loc[silver.stockout_flag == 1, "demand_target"].isna().all()


# ---------------- inventory
def test_forward_sum():
    x = np.arange(10, dtype=float)[None, :]
    assert inventory.forward_sum(x, 3)[0, 0] == 0 + 1 + 2


def test_simulation_conserves_units():
    rng = np.random.default_rng(0)
    D = rng.poisson(5, (3, 50)).astype(float)
    S = np.full_like(D, 60.0)
    sim = inventory.simulate(D, S, inventory.InvParams(), np.full(3, 30.0))
    assert np.allclose(sim["sold"] + sim["lost"], D)
    assert (sim["on_hand"] >= 0).all()


def test_higher_service_level_means_more_stock():
    rng = np.random.default_rng(1)
    D = rng.poisson(10, (20, 84)).astype(float)
    F = np.full_like(D, 10.0)
    r = inventory.run_policies(D, F, np.full(20, 10.0), np.full(20, 3.0), np.full(20, 1.0), np.full(20, 2.0))
    fd = r[r.policy == "forecast_driven"]
    assert fd.avg_inventory_units.is_monotonic_increasing
    assert fd.fill_rate.iloc[-1] > fd.fill_rate.iloc[0]


# ---------------- pricing / monitoring
def test_optimal_price_elastic_and_guardrails():
    assert pricing.optimal_price(10, 5, -2.0, band=1.0) == pytest.approx(10)  # c*e/(1+e)=10
    assert pricing.optimal_price(10, 5, -0.5, band=0.1) == pytest.approx(11)  # inelastic -> cap


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert monitoring.psi(x, x) < 1e-6
