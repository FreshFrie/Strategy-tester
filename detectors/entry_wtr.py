import numpy as np
from .common import atr_sma


def detect_wtr(arr, day_ctx, params):
    op, hi, lo, cl = arr.op, arr.hi, arr.lo, arr.cl
    l0, l1 = day_ctx["l0"], day_ctx["l1"]
    direction = day_ctx["direction"]
    pre_lo, pre_hi = day_ctx["pre_lo"], day_ctx["pre_hi"]

    n = int(params.get("atr_n", 14))
    omega = float(params.get("omega", 0.55))
    tau = float(params.get("tau", 0.65))
    atr = atr_sma(hi, lo, cl, n)

    out = []
    for s in range(max(l0, 1), l1):
        if not np.isfinite(atr[s]) or (hi[s] - lo[s]) <= 0:
            continue
        if direction == "LONG":
            if lo[s] < pre_lo - omega * atr[s] and cl[s] > pre_lo:
                inward = (cl[s] - lo[s]) / max(1e-12, hi[s] - lo[s])
                if inward >= tau and s + 1 < l1:
                    c = s + 1
                    if cl[c] > (op[s] + 0.25 * (cl[s] - op[s])) and cl[c] > hi[c - 1]:
                        out.append({"entry_idx": c, "direction": "LONG", "stop_price": float(lo[s] - 0.1 * atr[s]), "note": "wtr"})
        else:
            if hi[s] > pre_hi + omega * atr[s] and cl[s] < pre_hi:
                inward = (hi[s] - cl[s]) / max(1e-12, hi[s] - lo[s])
                if inward >= tau and s + 1 < l1:
                    c = s + 1
                    if cl[c] < (op[s] - 0.25 * (op[s] - cl[s])) and cl[c] < lo[c - 1]:
                        out.append({"entry_idx": c, "direction": "SHORT", "stop_price": float(hi[s] + 0.1 * atr[s]), "note": "wtr"})
    return out
