import numpy as np
from .common import atr_sma, fractal_swings


def detect_mfvg_s(arr, day_ctx, params):
    hi, lo, cl = arr.hi, arr.lo, arr.cl
    l0, l1 = day_ctx["l0"], day_ctx["l1"]
    direction = day_ctx["direction"]
    n = int(params.get("atr_n", 14))
    atr = atr_sma(hi, lo, cl, n)
    gap_min = float(params.get("gap_min_atr", 0.6))
    f = float(params.get("fill_frac", 0.5))
    lb = int(params.get("lb_fractal", 3))

    # Compute fractal swings only on a local segment around London window to avoid full-series cost
    seg0 = max(0, l0 - 80)
    seg1 = min(len(cl), l1 + 80)
    is_h_seg, is_l_seg = fractal_swings(hi[seg0:seg1], lo[seg0:seg1], lb)
    out = []

    for m in range(max(l0 + 1, 2), l1 - 1):
        if direction == "LONG":
            if not (hi[m - 1] < lo[m + 1]):
                continue
            g = lo[m + 1] - hi[m - 1]
            if not (np.isfinite(atr[m]) and g >= gap_min * atr[m]):
                continue
            fill_level = lo[m + 1] - f * g
            k0, k1 = m + 1, l1
            touched = np.flatnonzero(lo[k0:k1] <= fill_level)
            if touched.size == 0:
                continue
            k = k0 + int(touched[0])
            w0 = max(l0, k - 20)
            # map to segment indices
            s_w0 = max(0, w0 - seg0)
            s_k = max(0, k - seg0)
            piv_hi_idx_s = np.flatnonzero(is_h_seg[s_w0:s_k]) + s_w0
            piv_hi_idx = (piv_hi_idx_s + seg0) if piv_hi_idx_s.size else piv_hi_idx_s
            if piv_hi_idx.size == 0:
                continue
            if cl[k] > hi[piv_hi_idx[-1]]:
                out.append({"entry_idx": k, "direction": "LONG", "stop_price": float(hi[m - 1] - 0.1 * atr[k]), "note": "mfvg_s"})
        else:
            if not (lo[m - 1] > hi[m + 1]):
                continue
            g = hi[m + 1] - lo[m - 1]
            if not (np.isfinite(atr[m]) and g >= gap_min * atr[m]):
                continue
            fill_level = hi[m + 1] + f * g
            k0, k1 = m + 1, l1
            touched = np.flatnonzero(hi[k0:k1] >= fill_level)
            if touched.size == 0:
                continue
            k = k0 + int(touched[0])
            w0 = max(l0, k - 20)
            s_w0 = max(0, w0 - seg0)
            s_k = max(0, k - seg0)
            piv_lo_idx_s = np.flatnonzero(is_l_seg[s_w0:s_k]) + s_w0
            piv_lo_idx = (piv_lo_idx_s + seg0) if piv_lo_idx_s.size else piv_lo_idx_s
            if piv_lo_idx.size == 0:
                continue
            if cl[k] < lo[piv_lo_idx[-1]]:
                out.append({"entry_idx": k, "direction": "SHORT", "stop_price": float(lo[m - 1] + 0.1 * atr[k]), "note": "mfvg_s"})
    return out
