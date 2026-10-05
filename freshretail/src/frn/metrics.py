"""WAPE / WPE (bias) / VN1 - same definitions as the FreshRetailNet paper and the JD.
error = forecast - actual; WPE < 0 means UNDER-forecasting (the stock-out trap)."""
import numpy as np


def _c(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = ~(np.isnan(y) | np.isnan(f))
    return y[m], f[m]


def wape(y, f):
    y, f = _c(y, f)
    return float(np.abs(f - y).sum() / max(y.sum(), 1e-12))


def wpe(y, f):
    y, f = _c(y, f)
    return float((f - y).sum() / max(y.sum(), 1e-12))


def vn1(y, f):
    return wape(y, f) + abs(wpe(y, f))


def summary(y, f):
    return {"wape": wape(y, f), "wpe_bias": wpe(y, f), "vn1": vn1(y, f), "n": int(len(_c(y, f)[0]))}
