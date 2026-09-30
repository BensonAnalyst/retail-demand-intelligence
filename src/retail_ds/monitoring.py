"""Post-deployment monitoring: is the model still earning its keep?

Three signals, each with an owner-actionable threshold:
  * accuracy drift  - weekly WAPE vs backtest baseline (alert if > +20% relative)
  * bias drift      - weekly bias outside +/-10% (systematic over/under-ordering)
  * feature drift   - PSI of key inputs vs training window (PSI > 0.2 = investigate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import bias, wape


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def weekly_health(fc: pd.DataFrame, model: str, baseline_wape: float,
                  wape_tol: float = 0.20, bias_tol: float = 0.10, by: str = "category") -> pd.DataFrame:
    d = fc[(fc["model"] == model) & fc["actual"].notna()].copy()
    d["week"] = d["date"].dt.to_period("W").dt.start_time
    out = (d.groupby(["week", by])
             .apply(lambda g: pd.Series({"wape": wape(g.actual, g.forecast), "bias": bias(g.actual, g.forecast),
                                         "units": g.actual.sum()}), include_groups=False)
             .reset_index())
    out["alert_accuracy"] = out["wape"] > baseline_wape * (1 + wape_tol)
    out["alert_bias"] = out["bias"].abs() > bias_tol
    out["status"] = np.select([out.alert_accuracy & out.alert_bias, out.alert_accuracy, out.alert_bias],
                              ["RED", "AMBER-accuracy", "AMBER-bias"], "GREEN")
    return out
