"""Forecast -> replenishment decision -> $ impact.

Periodic-review order-up-to policy (R, S) with lead time L, lost-sales setting:
    every R days:  order = max(0, S_t - inventory_position)
    S_t = forecast demand over (L + R) + safety stock
    safety stock = z(service level) * sigma_forecast_error * sqrt(L + R)

Two policies are simulated on the same TRUE demand:
  * current_state  : planner rule "keep N days of cover at last-4-week average"
  * forecast_driven: S from the ML forecast + error-calibrated safety stock
so the saving is attributable to the forecast, not to a different simulation.
All series are simulated at once with numpy (n_series x days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class InvParams:
    review_period: int = 7          # R: weekly ordering
    lead_time: int = 3              # L: DC -> store
    holding_rate_annual: float = 0.25  # cost of capital + storage + shrink, % of unit cost / yr
    cover_days_current: int = 14    # current-state rule of thumb


def simulate(demand: np.ndarray, order_up_to: np.ndarray, p: InvParams, start_inv: np.ndarray) -> dict:
    """demand, order_up_to: (n, T). Returns daily on-hand, sales, lost (n, T)."""
    n, T = demand.shape
    on_hand = start_inv.astype(float).copy()
    pipeline = np.zeros((n, T + p.lead_time + 1))
    oh_hist, sold_hist, lost_hist = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    for t in range(T):
        on_hand += pipeline[:, t]                      # receipts arrive start of day
        if t % p.review_period == 0:                   # review day
            inv_pos = on_hand + pipeline[:, t + 1:].sum(axis=1)
            order = np.maximum(0, np.ceil(order_up_to[:, t] - inv_pos))
            pipeline[:, t + p.lead_time] += order
        sold = np.minimum(on_hand, demand[:, t])
        lost_hist[:, t] = demand[:, t] - sold
        on_hand -= sold
        sold_hist[:, t], oh_hist[:, t] = sold, on_hand
    return {"on_hand": oh_hist, "sold": sold_hist, "lost": lost_hist}


def forward_sum(x: np.ndarray, window: int) -> np.ndarray:
    """sum of x[t : t+window] for each t = demand to cover from today until the
    NEXT order arrives (pads with the last value at the end of the horizon)."""
    n, T = x.shape
    pad = np.concatenate([x, np.repeat(x[:, -1:], window, axis=1)], axis=1)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(pad, axis=1)], axis=1)
    idx = np.arange(T)
    return cs[:, idx + window] - cs[:, idx]


def policy_costs(sim: dict, demand: np.ndarray, unit_cost: np.ndarray, price: np.ndarray, p: InvParams) -> dict:
    daily_hold = unit_cost[:, None] * p.holding_rate_annual / 365
    margin = (price - unit_cost)[:, None]
    holding = float((sim["on_hand"] * daily_hold).sum())
    lost_margin = float((sim["lost"] * margin).sum())
    return {
        "fill_rate": float(sim["sold"].sum() / max(demand.sum(), 1e-9)),
        "avg_inventory_units": float(sim["on_hand"].sum(axis=0).mean()),
        "avg_inventory_value": float((sim["on_hand"] * unit_cost[:, None]).sum(axis=0).mean()),
        "holding_cost": holding,
        "lost_margin": lost_margin,
        "total_cost": holding + lost_margin,
        "stockout_days_pct": float((sim["lost"] > 0).mean()),
    }


def run_policies(true_demand: np.ndarray, forecast: np.ndarray, hist_avg: np.ndarray,
                 sigma: np.ndarray, unit_cost: np.ndarray, price: np.ndarray,
                 service_levels=(0.85, 0.90, 0.95, 0.98, 0.99), p: InvParams | None = None) -> pd.DataFrame:
    """Compare current-state rule vs forecast-driven policy at several service levels."""
    p = p or InvParams()
    w = p.lead_time + p.review_period
    start = hist_avg * w
    rows = []
    # current state: N days of cover at recent average (static)
    S_cur = np.repeat((hist_avg * p.cover_days_current)[:, None], true_demand.shape[1], axis=1)
    rows.append({"policy": "current_state", "target_service": np.nan,
                 **policy_costs(simulate(true_demand, S_cur, p, start), true_demand, unit_cost, price, p)})
    fwd = forward_sum(forecast, w)
    for sl in service_levels:
        ss = norm.ppf(sl) * sigma * np.sqrt(w)
        S = fwd + ss[:, None]
        rows.append({"policy": "forecast_driven", "target_service": sl,
                     **policy_costs(simulate(true_demand, S, p, start), true_demand, unit_cost, price, p)})
    return pd.DataFrame(rows)
