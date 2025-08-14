import numpy as np
from .common import atr_sma, anchor_avwap_from


def detect_svwap_sh(arr, day_ctx, params):
    op, hi, lo, cl = arr.op, arr.hi, arr.lo, arr.cl
    l0, l1 = day_ctx["l0"], day_ctx["l1"]
    vwap_anchor = day_ctx.get("vwap_anchor")
    direction = day_ctx["direction"]

    n = int(params.get("atr_n", 14))
    k_hold = int(params.get("k_hold", 3))
    k_s = int(params.get("k_slope", 10))
    slope_min = float(params.get("slope_min_atr", 0.1))
    eps = float(params.get("eps_atr", 0.1))

    atr = atr_sma(hi, lo, cl, n)

    if vwap_anchor is None:
        return []
    avwap = anchor_avwap_from(op, hi, lo, cl, vwap_anchor, end_idx=l1)

    out = []
    for t in range(max(l0 + k_hold, vwap_anchor + k_s), l1):
        if not np.isfinite(atr[t]) or not np.isfinite(avwap[t]) or not np.isfinite(avwap[t - k_s]):
            continue
        slope_up = (avwap[t] - avwap[t - k_s]) >= slope_min * atr[t]
        slope_down = (avwap[t - k_s] - avwap[t]) >= slope_min * atr[t]
        if direction == "LONG":
            hold = np.all(cl[t - k_hold + 1 : t + 1] > avwap[t - k_hold + 1 : t + 1])
            if not (hold and slope_up):
                continue
            j0 = max(l0, t - k_hold - 6)
            mask = np.isfinite(avwap[j0:t]) & np.isfinite(atr[j0:t])
            pulled = np.flatnonzero(mask & (lo[j0:t] <= (avwap[j0:t] + eps * atr[j0:t])))
            if pulled.size == 0:
                continue
            out.append({"entry_idx": t, "direction": "LONG", "stop_price": float(cl[t] - 1.25 * atr[t]), "note": "svwap_sh"})
        else:
            hold = np.all(cl[t - k_hold + 1 : t + 1] < avwap[t - k_hold + 1 : t + 1])
            if not (hold and slope_down):
                continue
            j0 = max(l0, t - k_hold - 6)
            mask = np.isfinite(avwap[j0:t]) & np.isfinite(atr[j0:t])
            pulled = np.flatnonzero(mask & (hi[j0:t] >= (avwap[j0:t] - eps * atr[j0:t])))
            if pulled.size == 0:
                continue
            out.append({"entry_idx": t, "direction": "SHORT", "stop_price": float(cl[t] + 1.25 * atr[t]), "note": "svwap_sh"})
    return out
