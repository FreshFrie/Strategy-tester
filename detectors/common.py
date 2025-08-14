import numpy as np


def _prev_close(cl):
    pc = np.empty_like(cl)
    pc[0] = cl[0]
    pc[1:] = cl[:-1]
    return pc


def atr_sma(hi, lo, cl, n=14):
    pc = _prev_close(cl)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    atr = np.full_like(tr, np.nan, dtype=float)
    if n <= 1:
        atr[:] = tr
        return atr
    c = np.convolve(tr, np.ones(n, dtype=float) / n, mode="valid")
    atr[n - 1 :] = c
    return atr


def body_ratio(op, hi, lo, cl):
    rng = np.maximum(1e-12, hi - lo)
    return np.abs(cl - op) / rng


def fractal_swings(hi, lo, lb=3):
    n = len(hi)
    is_h = np.zeros(n, dtype=bool)
    is_l = np.zeros(n, dtype=bool)
    for i in range(lb, n - lb):
        hwin = hi[i - lb : i + lb + 1]
        lwin = lo[i - lb : i + lb + 1]
        if hi[i] == np.max(hwin) and np.sum(hi[i] == hwin) == 1:
            is_h[i] = True
        if lo[i] == np.min(lwin) and np.sum(lo[i] == lwin) == 1:
            is_l[i] = True
    return is_h, is_l


def anchor_avwap_from(op, hi, lo, cl, start_idx, end_idx=None):
    """
    Unit-weighted anchored VWAP (AVWAP): cumulative mean of TP=(H+L+C)/3 from start_idx.
    Returns array same length as inputs, filled with np.nan before start_idx.
    """
    n = len(cl)
    tp = (hi + lo + cl) / 3.0
    out = np.full(n, np.nan, dtype=float)
    if start_idx is None or start_idx < 0 or start_idx >= n:
        return out
    if end_idx is None or end_idx > n:
        end_idx = n
    seg = tp[start_idx:end_idx]
    cum = np.cumsum(seg)
    denom = np.arange(1, len(seg) + 1, dtype=float)
    avwap_seg = cum / denom
    out[start_idx:end_idx] = avwap_seg
    return out
