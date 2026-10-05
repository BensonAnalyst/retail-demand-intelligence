import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frn import forecast, metrics, recovery  # noqa: E402
from frn.data import Panel, check_oos_orientation  # noqa: E402


def _panel(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    shape = np.r_[np.zeros(6), np.ones(17), 0.0]
    shape[17:21] = 2.0
    shape /= shape.sum()
    rows, S, O = [], [], []
    for store in range(3):
        for prod in range(4):
            for d in range(n_days):
                demand = 10 * shape * rng.uniform(0.8, 1.2)
                oos = np.zeros(24, int)
                if rng.random() < 0.3:
                    oos[rng.integers(14, 21):23] = 1          # evening sell-out
                S.append(demand * (1 - oos)); O.append(oos)
                rows.append(dict(store_id=store, product_id=prod, third_category_id=prod % 2,
                                 dt=pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), day=d,
                                 discount=1.0, holiday_flag=0, activity_flag=0, avg_temperature=20.0, precpt=0.0))
    return Panel(pd.DataFrame(rows), np.array(S), np.array(O, dtype=np.int8))


def test_metrics_sign_convention():
    assert metrics.wpe([10, 10], [8, 8]) == pytest.approx(-0.2)     # under-forecast is negative
    assert metrics.vn1([10, 10], [8, 12]) == pytest.approx(0.2)


def test_orientation_flip():
    p = _panel()
    p.df["stock_hour6_22_cnt"] = p.oos[:, 6:23].sum(1)
    p.oos = (1 - p.oos).astype(np.int8)                              # wrong convention
    assert check_oos_orientation(p)["flipped"] is True


def test_profile_recovery_removes_censoring_bias():
    p = _panel()
    clean = recovery.clean_days(p)
    prof = recovery.hourly_profiles(p, clean)
    level = recovery.series_level(p, clean)
    rec = recovery.profile_recover(p.sales, p.oos, prof, level)
    truth_on_oos_days = 10.0                                          # demand scale built into _panel
    assert rec[~clean].mean() == pytest.approx(truth_on_oos_days, rel=0.08)
    assert p.sales[~clean].sum(1).mean() < 0.9 * truth_on_oos_days   # raw sales were censored


def test_forecast_lags_respect_horizon():
    p = _panel()
    fr = forecast.build_frame(p.df.assign(city_id=0, management_group_id=0, first_category_id=0,
                                          second_category_id=0, avg_humidity=50.0, avg_wind_level=1.0),
                              p.oos, p.sales.sum(1))
    s = fr[(fr.store_id == 0) & (fr.product_id == 0)].reset_index(drop=True)
    assert s.loc[20, "lag_7"] == s.loc[13, "y"]
    assert s.loc[20, "rmean_7"] == pytest.approx(s.loc[7:13, "y"].mean())


def test_feature_importance_shares_sum_to_100():
    p = _panel(n_days=30)
    _, model = recovery.recover(p, 29, return_model=True)
    imp = recovery.feature_importance(model)
    assert imp["gain_pct"].sum() == pytest.approx(100)
    assert set(imp["group"]) <= {"partial day", "profile hint", "series level", "promo", "calendar", "weather", "category", "other"}
