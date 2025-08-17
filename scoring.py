def score_signal(features: dict, note: str) -> float:
    """
    Returns a heuristic score in [0,1] for a detector signal based on its features and detector name.
    Simple weighted sum with diminishing returns and small boolean bonuses.
    """
    s = 0.0
    # strength-like numeric features
    for k in ("wick_frac", "comp_ratio", "gap_atr", "impulse_atr", "atr_mult", "fill_frac", "retest_distance"):
        v = features.get(k)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        s += min(0.35, 0.12 * max(0.0, v))

    # vwap slope
    v = features.get("vwap_slope") or features.get("vwap_slope_atr")
    if v is not None:
        try:
            v = abs(float(v))
            s += min(0.2, 0.08 * v)
        except Exception:
            pass

    # boolean confirms
    for k in ("swing_confirm", "recent_pull", "confirm_break", "confirm"):
        if features.get(k):
            s += 0.06

    # small detector priors
    priors = {"cb_vp": 0.05, "mfvg_s": 0.03, "svwap_sh": 0.04, "iib_c": 0.03, "wtr": 0.03}
    s += priors.get(note, 0.0)

    # clamp
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return float(s)
