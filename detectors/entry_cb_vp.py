import numpy as np
from .common import atr_sma, body_ratio


def detect_cb_vp(arr, day_ctx, params):
    op, hi, lo, cl = arr.op, arr.hi, arr.lo, arr.cl
    p0, p1 = day_ctx["p0"], day_ctx["p1"]
    l0, l1 = day_ctx["l0"], day_ctx["l1"]
    direction = day_ctx["direction"]
    pre_lo, pre_hi = day_ctx["pre_lo"], day_ctx["pre_hi"]

    # Guard for missing Pre-London
    if p0 is None or p1 is None or p1 <= p0:
        return []
    if not (np.isfinite(pre_lo) and np.isfinite(pre_hi)):
        return []

    n = int(params.get("atr_n", 14))
    atr = atr_sma(hi, lo, cl, n)
    pre_atr_mean = np.nanmean(atr[p0:p1]) if p0 is not None and p1 is not None and p0 < p1 else np.nan
    if not np.isfinite(pre_atr_mean) or pre_atr_mean <= 0:
        return []

    alpha_max = float(params.get("alpha_prelon_max", 0.6))
    comp = (pre_hi - pre_lo) / pre_atr_mean if pre_atr_mean > 0 else np.inf
    if not (comp <= alpha_max):
        return []

    beta = float(params.get("beta", 0.8))
    gamma = float(params.get("gamma", 0.7))
    eps = float(params.get("eps_atr", 0.1))
    R_retest = int(params.get("R_retest", 5))

    br = body_ratio(op, hi, lo, cl)
    out = []

    # scan KZ
    for i in range(max(l0, 1), l1):  # start at 1 for prev-close access
        if not np.isfinite(atr[i]):
            continue
        rng = hi[i] - lo[i]
        if rng < beta * atr[i] or br[i] < gamma:
            continue

        if direction == "LONG":
            if cl[i] <= pre_hi:
                continue
            # retest window
            j0, j1 = i + 1, min(l1, i + 1 + R_retest)
            touched = np.flatnonzero(lo[j0:j1] <= pre_hi + eps * atr[i])
            if touched.size == 0:
                continue
            j = j0 + int(touched[0])
            if cl[j] > max(cl[j - 1], hi[j - 1]):
                out.append({"entry_idx": j, "direction": "LONG", "stop_price": None, "note": "cb_vp", "features": {"comp_ratio": float(comp), "atr_mult": float((hi[i]-lo[i])/pre_atr_mean if pre_atr_mean>0 else 0.0), "retest_distance": float((cl[j]-pre_hi)/pre_atr_mean if pre_atr_mean>0 else 0.0)}, "score": None})
        else:
            if cl[i] >= pre_lo:
                continue
            j0, j1 = i + 1, min(l1, i + 1 + R_retest)
            touched = np.flatnonzero(hi[j0:j1] >= pre_lo - eps * atr[i])
            if touched.size == 0:
                continue
            j = j0 + int(touched[0])
            if cl[j] < min(cl[j - 1], lo[j - 1]):
                out.append({"entry_idx": j, "direction": "SHORT", "stop_price": None, "note": "cb_vp", "features": {"comp_ratio": float(comp), "atr_mult": float((hi[i]-lo[i])/pre_atr_mean if pre_atr_mean>0 else 0.0), "retest_distance": float((pre_lo-cl[j])/pre_atr_mean if pre_atr_mean>0 else 0.0)}, "score": None})

    # Stop: max(structural, ATR floor) per spec (min for LONG, max for SHORT yields farther stop)
    k_floor = float(params.get("k_floor", 1.25))
    if out:
        e = out[0]["entry_idx"]
        if np.isfinite(atr[e]):
            if direction == "LONG":
                structural = float(pre_hi - 0.1 * atr[e])
                atr_floor  = float(cl[e] - k_floor * atr[e])
                out[0]["stop_price"] = float(min(structural, atr_floor))
            else:
                structural = float(pre_lo + 0.1 * atr[e])
                atr_floor  = float(cl[e] + k_floor * atr[e])
                out[0]["stop_price"] = float(max(structural, atr_floor))
        else:
            out[0]["stop_price"] = None
    return out
