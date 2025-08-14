import numpy as np
from .common import atr_sma, body_ratio


def detect_iib_c(arr, day_ctx, params):
    op, hi, lo, cl = arr.op, arr.hi, arr.lo, arr.cl
    l0, l1 = day_ctx["l0"], day_ctx["l1"]
    direction = day_ctx["direction"]
    n = int(params.get("atr_n", 14))
    beta = float(params.get("beta", 2.0))
    gamma = float(params.get("gamma", 0.7))
    atr = atr_sma(hi, lo, cl, n)
    br = body_ratio(op, hi, lo, cl)

    out = []
    for i in range(max(l0 + 1, 1), min(l1 - 2, len(cl) - 2)):
        if not np.isfinite(atr[i]):
            continue
        if (hi[i] - lo[i]) < beta * atr[i] or br[i] < gamma:
            continue
        # Inside bar at i+1
        if hi[i + 1] <= hi[i] and lo[i + 1] >= lo[i]:
            # Confirm with next bar (i+2) breaking impulse extreme by a small ATR buffer
            j = i + 2
            thr = 0.05 * atr[i]
            if direction == "LONG" and hi[j] >= hi[i] + thr and cl[j] >= hi[i]:
                out.append({"entry_idx": j, "direction": "LONG", "stop_price": float(lo[i] - 0.1 * atr[i]), "note": "iib_c"})
            elif direction == "SHORT" and lo[j] <= lo[i] - thr and cl[j] <= lo[i]:
                out.append({"entry_idx": j, "direction": "SHORT", "stop_price": float(hi[i] + 0.1 * atr[i]), "note": "iib_c"})
    return out
