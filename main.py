"""
NAS100 Asia Killzone to London Backtester

Strategy:
- Uses Asia session (22:00-04:00 UTC) to identify killzone setups
- Asia Killzone: 00:00-02:00 UTC 
- Trades during London session (06:00-09:00 UTC) targeting Asia KZ levels
- Two scenarios based on whether Asia KZ made session high, low, or both

CSV format: [Time, Open, High, Low, Close] with Time in UTC
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import argparse
import time
import os
from zoneinfo import ZoneInfo

# =============================
# ======== CONFIG ============
# =============================

# Defaults (overridable by CLI)
CSV_PATH = ""  # If empty or missing, we'll auto-detect a dataset CSV in the workspace
DT_COL, O_COL, H_COL, L_COL, C_COL = "Time", "Open", "High", "Low", "Close"

# Instrument settings
TICK_SIZE = 0.25
TICK_VALUE = 0.5        # MNQ micro futures $/tick (5.0 for NQ, 1.0 for CFD-like)
POINT_VALUE = 1.0       # $ per 1 index point (CFD mode)
SIZING_MODE = "contracts"  # {"contracts", "cfd"}
POSITION_SIZE = 1.0
SLIPPAGE_TICKS = 1

# Strategy params
R_MULT = 1.5
START_CAPITAL = 10000.0
RISK_PCT_PER_TRADE = 0.01

# New takeout and execution settings
TAKEOUT_START_UTC = "01:00"
TAKEOUT_END_UTC = "08:00"
ENTRY_MODE = "break"  # {"break","retest"} relative to pre-London range
AMBIGUOUS_POLICY = "worst"  # {"worst","neutral","best"}
MAX_TRADES_PER_DAY = 2
COOLDOWN_MIN = 30
NEXT_BAR_EXECUTION = True

# Reports
ASIA_REPORT_PATH = "asia_kz_report.csv"

# Performance settings
READ_ENGINE: Optional[str] = None  # set to "pyarrow" if available
DOWNCAST_FLOATS = True

def parse_args():
    parser = argparse.ArgumentParser(description="Killzone Backtester")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV file (optional; auto-detect if omitted)")
    parser.add_argument("--engine", choices=["pandas", "pyarrow"], help="CSV read engine")
    parser.add_argument("--csv-format", choices=["auto","default", "mt"], default="auto", help="CSV schema: auto=sniff, default=Time,Open,High,Low,Close; mt=MetaTrader (date,time,OHLC,vol)")
    parser.add_argument("--data-utc-offset", type=int, default=0, help="Hours offset of data timestamps relative to UTC (e.g., -5 for UTC-5, -4 for UTC-4). Used if --data-tz not provided.")
    parser.add_argument("--data-tz", default="", help="IANA timezone of the data timestamps (e.g., America/New_York). If set, overrides --data-utc-offset and handles DST.")
    parser.add_argument("--sizing", choices=["contracts", "cfd"], default=SIZING_MODE, help="Position sizing mode")
    parser.add_argument("--tick-value", type=float, default=TICK_VALUE, help="$ per tick (contracts mode)")
    parser.add_argument("--point-value", type=float, default=POINT_VALUE, help="$ per point (CFD mode)")
    parser.add_argument("--risk-pct", type=float, default=RISK_PCT_PER_TRADE, help="Risk percentage per trade")
    parser.add_argument("--rr-ratio", type=float, default=R_MULT, help="Risk-to-Reward ratio (e.g., 1.5 for 1:1.5)")
    parser.add_argument("--takeout-start", default=TAKEOUT_START_UTC, help="Takeout window start time (HH:MM UTC)")
    parser.add_argument("--takeout-end", default=TAKEOUT_END_UTC, help="Takeout window end time (HH:MM UTC)")
    # Session windows (UTC)
    parser.add_argument("--asia-prev-start", default="00:00", help="Asia prev-day start (HH:MM UTC)")
    parser.add_argument("--asia-prev-end", default="00:00", help="Asia prev-day end (HH:MM UTC)")
    parser.add_argument("--asia-curr-start", default="17:00", help="Asia current-day start (HH:MM UTC)")
    parser.add_argument("--asia-curr-end", default="23:00", help="Asia current-day end (HH:MM UTC)")
    parser.add_argument("--kz-start", default="19:00", help="Asia Killzone start (HH:MM UTC)")
    parser.add_argument("--kz-end", default="21:00", help="Asia Killzone end (HH:MM UTC)")
    parser.add_argument("--pre-start", default="21:00", help="Pre-London start (HH:MM UTC)")
    parser.add_argument("--pre-end", default="23:00", help="Pre-London end (HH:MM UTC)")
    parser.add_argument("--london-start", default="01:00", help="London session start (HH:MM UTC)")
    parser.add_argument("--london-end", default="04:00", help="London session end (HH:MM UTC)")
    parser.add_argument("--entry-mode", choices=["break", "retest"], default=ENTRY_MODE, help="Entry mode relative to pre-London range")
    parser.add_argument("--ambiguous", choices=["worst", "neutral", "best"], default=AMBIGUOUS_POLICY, help="Policy for ambiguous TP/SL bars")
    parser.add_argument("--max-trades", type=int, default=MAX_TRADES_PER_DAY, help="Maximum trades per day")
    parser.add_argument("--cooldown-min", type=int, default=COOLDOWN_MIN, help="Cooldown minutes between trades")
    parser.add_argument("--early-takeout", choices=["on","off"], default="off", help="Apply the pre-London early-takeout day skip filter")
    parser.add_argument("--early-window", choices=["pre","kz-to-london"], default="kz-to-london", help="Window used for early-takeout detection")
    parser.add_argument("--no-write", action="store_true", help="Skip writing CSV outputs")
    return parser.parse_args()

# =============================
# ===== Asia KZ Analysis ======
# =============================

@dataclass
class AsiaKZInfo:
    session_high: float
    session_low: float
    kz_high: float
    kz_low: float
    high_made_in_kz: bool
    low_made_in_kz: bool
    kz_high_time: pd.Timestamp
    kz_low_time: pd.Timestamp

def build_asia_report(df: pd.DataFrame, out_path: str, config):
    """
    Fast Asia/KZ verification report using array day windows (O(n)).
    One row per current day with: date, session_high/low (+times), kz_high/low (+times), flags.
    Asia: 22:00 prev → 04:00 curr (UTC). KZ: 00:00–02:00 in current.
    """
    arr = build_arrays(df)
    rows = []
    n_days = len(arr.u_days)

    def _m(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h*60 + m

    for k in range(1, n_days):  # need previous day
        prev_i0, prev_i1 = arr.starts[k-1], arr.ends[k-1]
        i0, i1 = arr.starts[k], arr.ends[k]

        # Asia windows
        a0a, a1a = win_idx(arr.mins, prev_i0, prev_i1, _m(config.asia_prev_start), _m(config.asia_prev_end))
        a0b, a1b = win_idx(arr.mins, i0, i1, _m(config.asia_curr_start), _m(config.asia_curr_end))
        if (a0a is None or a1a is None) and (a0b is None or a1b is None):
            continue

        # Session extremes across both slices (earliest occurrence logic)
        segs = []
        if a0a is not None and a1a is not None and a0a < a1a:
            segs.append((a0a, a1a))
        if a0b is not None and a1b is not None and a0b < a1b:
            segs.append((a0b, a1b))

        # High
        sh_val, sh_idx = -np.inf, None
        for s0, s1 in segs:
            local_max = float(np.max(arr.hi[s0:s1]))
            rel = int(np.flatnonzero(arr.hi[s0:s1] == local_max)[0])
            cand = s0 + rel
            if (local_max > sh_val) or (np.isclose(local_max, sh_val) and (sh_idx is None or cand < sh_idx)):
                sh_val, sh_idx = local_max, cand

        # Low
        sl_val, sl_idx = np.inf, None
        for s0, s1 in segs:
            local_min = float(np.min(arr.lo[s0:s1]))
            rel = int(np.flatnonzero(arr.lo[s0:s1] == local_min)[0])
            cand = s0 + rel
            if (local_min < sl_val) or (np.isclose(local_min, sl_val) and (sl_idx is None or cand < sl_idx)):
                sl_val, sl_idx = local_min, cand

        sh_t = pd.Timestamp(arr.dt[sh_idx]) if sh_idx is not None else pd.NaT
        sl_t = pd.Timestamp(arr.dt[sl_idx]) if sl_idx is not None else pd.NaT

        # KZ in current day (00:00–02:00 UTC)
        kz0, kz1 = win_idx(arr.mins, i0, i1, _m(config.kz_start), _m(config.kz_end))
        if kz0 is None or kz1 is None or kz0 >= kz1:
            kh = kl = float("nan")
            kh_t = kl_t = pd.NaT
            high_in_kz = low_in_kz = False
        else:
            kz_hi = arr.hi[kz0:kz1]
            kz_lo = arr.lo[kz0:kz1]
            kh = float(np.max(kz_hi))
            kl = float(np.min(kz_lo))
            kh_rel = int(np.flatnonzero(kz_hi == kh)[0])
            kl_rel = int(np.flatnonzero(kz_lo == kl)[0])
            kh_t = pd.Timestamp(arr.dt[kz0 + kh_rel])
            kl_t = pd.Timestamp(arr.dt[kz0 + kl_rel])
            high_in_kz = (sh_idx is not None) and (kz0 <= sh_idx < kz1)
            low_in_kz  = (sl_idx is not None) and (kz0 <= sl_idx < kz1)

        rows.append({
            "date": pd.Timestamp(arr.u_days[k]),
            "session_high": float(sh_val), "session_high_time": sh_t,
            "session_low":  float(sl_val), "session_low_time":  sl_t,
            "kz_high": kh, "kz_high_time": kh_t,
            "kz_low":  kl, "kz_low_time":  kl_t,
            "high_made_in_kz": high_in_kz,
            "low_made_in_kz":  low_in_kz,
        })

    rep = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "date","session_high","session_high_time","session_low","session_low_time",
        "kz_high","kz_high_time","kz_low","kz_low_time","high_made_in_kz","low_made_in_kz"
    ])
    for col in ("date","session_high_time","session_low_time","kz_high_time","kz_low_time"):
        rep[col] = pd.to_datetime(rep[col]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    rep.to_csv(out_path, index=False)

def analyze_asia_session_arr(arr: Arr, asia_prev: Tuple[Optional[int], Optional[int]], asia_curr: Tuple[Optional[int], Optional[int]], config=None) -> Optional[AsiaKZInfo]:
    """Array-based Asia session analysis over prev[22-24] + curr[0-4] and KZ [0-2] within curr.
    Flags high/low_made_in_kz are set by checking whether the session extreme's index falls within the KZ window (not just value equality).
    """
    a0a, a1a = asia_prev
    a0b, a1b = asia_curr
    if (a0a is None or a1a is None) and (a0b is None or a1b is None):
        return None

    # Collect slices spanning session: prev and current
    segs = []
    if a0a is not None and a1a is not None and a0a < a1a:
        segs.append((a0a, a1a))
    if a0b is not None and a1b is not None and a0b < a1b:
        segs.append((a0b, a1b))
    if not segs:
        return None

    # Session high with earliest occurrence index
    sh_val, sh_idx = -np.inf, None
    for s0, s1 in segs:
        local_max = float(np.max(arr.hi[s0:s1]))
        rel = int(np.flatnonzero(arr.hi[s0:s1] == local_max)[0])
        cand = s0 + rel
        if (local_max > sh_val) or (np.isclose(local_max, sh_val) and (sh_idx is None or cand < sh_idx)):
            sh_val, sh_idx = local_max, cand

    # Session low with earliest occurrence index
    sl_val, sl_idx = np.inf, None
    for s0, s1 in segs:
        local_min = float(np.min(arr.lo[s0:s1]))
        rel = int(np.flatnonzero(arr.lo[s0:s1] == local_min)[0])
        cand = s0 + rel
        if (local_min < sl_val) or (np.isclose(local_min, sl_val) and (sl_idx is None or cand < sl_idx)):
            sl_val, sl_idx = local_min, cand

    session_high = float(sh_val)
    session_low = float(sl_val)

    # KZ window can be in prev or current, depending on config times
    def _m(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h*60 + m
    kz_s = _m(config.kz_start) if config is not None else 0
    kz_e = _m(config.kz_end) if config is not None else 120
    kz0 = kz1 = None
    # Try in current segment
    if a0b is not None and a1b is not None and a0b < a1b:
        kz0, kz1 = win_idx(arr.mins, a0b, a1b, kz_s, kz_e)
    # Fallback to previous segment
    if (kz0 is None or kz1 is None or kz0 >= kz1) and (a0a is not None and a1a is not None and a0a < a1a):
        kz0, kz1 = win_idx(arr.mins, a0a, a1a, kz_s, kz_e)
    if kz0 is None or kz1 is None or kz0 >= kz1:
        return None

    # KZ extremes and their first-occurrence timestamps
    kz_slice_hi = arr.hi[kz0:kz1]
    kz_slice_lo = arr.lo[kz0:kz1]
    kz_high = float(np.max(kz_slice_hi))
    kz_low = float(np.min(kz_slice_lo))
    idx_hi_rel = np.flatnonzero(kz_slice_hi == kz_high)
    idx_lo_rel = np.flatnonzero(kz_slice_lo == kz_low)
    if idx_hi_rel.size == 0 or idx_lo_rel.size == 0:
        return None
    kz_high_idx = kz0 + int(idx_hi_rel[0])
    kz_low_idx = kz0 + int(idx_lo_rel[0])
    kz_high_time = arr.dt[kz_high_idx]
    kz_low_time = arr.dt[kz_low_idx]

    # Flags: did the session extremes occur during KZ window?
    high_made_in_kz = (sh_idx is not None) and (kz0 <= sh_idx < kz1)
    low_made_in_kz = (sl_idx is not None) and (kz0 <= sl_idx < kz1)

    return AsiaKZInfo(
        session_high=session_high,
        session_low=session_low,
        kz_high=kz_high,
        kz_low=kz_low,
        high_made_in_kz=bool(high_made_in_kz),
        low_made_in_kz=bool(low_made_in_kz),
        kz_high_time=pd.Timestamp(kz_high_time),
        kz_low_time=pd.Timestamp(kz_low_time),
    )

def check_early_takeout(pre_london_df: pd.DataFrame, asia_info: AsiaKZInfo) -> bool:
    """Check if Asia KZ levels were taken out before London session (04:00-06:00 UTC)"""
    if pre_london_df.empty or asia_info is None:
        return False
        
    takeout_occurred = False
    
    # Check if KZ high was taken out (if it was significant)
    if asia_info.high_made_in_kz:
        if (pre_london_df['high'] >= asia_info.kz_high).any():
            takeout_occurred = True
            
    # Check if KZ low was taken out (if it was significant)  
    if asia_info.low_made_in_kz:
        if (pre_london_df['low'] <= asia_info.kz_low).any():
            takeout_occurred = True
            
    return takeout_occurred

def slice_utc_window(df: pd.DataFrame, day: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """Returns df rows for date==day in [start,end), both UTC"""
    start_hour, start_min = map(int, start_hhmm.split(":"))
    end_hour, end_min = map(int, end_hhmm.split(":"))
    
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    
    mask = (df["date_utc"] == day.date()) & \
           (df["minute_of_day_utc"] >= start_minutes) & \
           (df["minute_of_day_utc"] < end_minutes)
    
    return df[mask].copy()

def monitor_takeouts(window_df: pd.DataFrame, asia_info: AsiaKZInfo) -> list[Tuple[str, pd.Timestamp]]:
    """Monitor takeout window for KZ level takeouts (vectorized)"""
    takeouts = []
    
    if asia_info is None or window_df.empty:
        return takeouts
    
    # Check for high takeout (vectorized)
    if asia_info.high_made_in_kz:
        high_takeout_mask = window_df['high'] >= asia_info.kz_high
        if high_takeout_mask.any():
            first_takeout_idx = high_takeout_mask.idxmax()
            takeouts.append(("high_takeout", window_df.loc[first_takeout_idx, 'dt_utc']))
            
    # Check for low takeout (vectorized)
    if asia_info.low_made_in_kz:
        low_takeout_mask = window_df['low'] <= asia_info.kz_low
        if low_takeout_mask.any():
            first_takeout_idx = low_takeout_mask.idxmax()
            takeouts.append(("low_takeout", window_df.loc[first_takeout_idx, 'dt_utc']))
            
    # Sort by time, keep first by type
    takeouts.sort(key=lambda x: x[1])
    return takeouts

def monitor_takeouts_arr(arr: Arr, t0: int, t1: int, asia_info: AsiaKZInfo) -> list[Tuple[str, pd.Timestamp, int]]:
    """Array-based takeout detection. Returns (type, time, absolute_index)."""
    out: list[Tuple[str, pd.Timestamp, int]] = []
    if asia_info is None or t0 is None or t1 is None or t0 >= t1:
        return out
    if asia_info.high_made_in_kz:
        m = arr.hi[t0:t1] >= asia_info.kz_high
        idx = np.flatnonzero(m)
        if idx.size:
            i = t0 + int(idx[0])
            out.append(("high_takeout", pd.Timestamp(arr.dt[i]), i))
    if asia_info.low_made_in_kz:
        m = arr.lo[t0:t1] <= asia_info.kz_low
        idx = np.flatnonzero(m)
        if idx.size:
            i = t0 + int(idx[0])
            out.append(("low_takeout", pd.Timestamp(arr.dt[i]), i))
    out.sort(key=lambda x: x[1])
    return out

def pre_london_range(pre_df: pd.DataFrame) -> Tuple[float, float]:
    """Get pre-London range (04:00-06:00 UTC)"""
    if pre_df.empty:
        return 0.0, 0.0
    return float(pre_df['low'].min()), float(pre_df['high'].max())

def first_close_break(df: pd.DataFrame, level: float, direction: str) -> Optional[int]:
    """Find first close break above/below level"""
    if df.empty:
        return None
        
    if direction == "above":
        mask = df['close'] > level
    else:  # "below"
        mask = df['close'] < level
        
    return mask.idxmax() if mask.any() else None

def fill_next_open(idx: int, op: np.ndarray, default_level: float) -> float:
    """Fill at next bar's open, or default level if out of range"""
    j = idx + 1
    return float(op[j]) if j < len(op) else float(default_level)

def resolve_ambiguous(tp_idx: Optional[int], sl_idx: Optional[int], policy: str) -> Tuple[Optional[int], str]:
    """Resolve ambiguous TP/SL bars based on policy"""
    if tp_idx is None and sl_idx is None:
        return None, "NONE"
    elif tp_idx is None:
        return sl_idx, "SL"
    elif sl_idx is None:
        return tp_idx, "TP"
    elif tp_idx == sl_idx:  # Same bar - apply policy
        if policy == "worst":
            return sl_idx, "SL"  # Assume stop hit first
        elif policy == "best":
            return tp_idx, "TP"  # Assume target hit first
        else:  # "neutral"
            return tp_idx, "NEUTRAL"  # Exit at open
    else:
        # Different bars - take earliest
        if tp_idx < sl_idx:
            return tp_idx, "TP"
        else:
            return sl_idx, "SL"

@dataclass
class Trade:
    side: str
    session: str
    day: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    r_multiple: float
    pnl_currency: float
    qty: float
    risk_target_usd: float
    risk_allocated_usd: float
    per_unit_risk_usd: float


def _round_to_tick(x: float) -> float:
    return round(x / TICK_SIZE) * TICK_SIZE


def compute_qty(entry: float, stop: float, equity: float, risk_pct: float,
                sizing_mode: str, tick_size: float, tick_value: float, point_value: float) -> float:
    """Compute position size based on sizing mode"""
    risk_amount = equity * risk_pct
    per_unit_risk_points = max(tick_size, abs(entry - stop))
    
    if sizing_mode == "contracts":
        per_unit_risk_usd = (per_unit_risk_points / tick_size) * tick_value
        if per_unit_risk_usd <= 0:
            return 0
        qty_float = risk_amount / per_unit_risk_usd
        if not math.isfinite(qty_float):
            return 0
        qty = int(max(0, math.floor(qty_float)))
        # Add position size limits to prevent unrealistic sizes
        qty = min(qty, 10000)  # Maximum 10,000 contracts
    else:  # "cfd"
        per_unit_risk_usd = per_unit_risk_points * point_value
        if per_unit_risk_usd <= 0:
            return 0.0
        qty = max(0.0, risk_amount / per_unit_risk_usd)
        if not math.isfinite(qty):
            return 0.0
        # Add position size limits to prevent unrealistic sizes
        qty = min(qty, 1000000.0)  # Maximum 1M units
    
    return qty
def calculate_pnl(entry: float, exit: float, qty: float, side: str, 
                  sizing_mode: str, tick_size: float, tick_value: float, point_value: float) -> float:
    """Calculate PnL based on sizing mode"""
    if side == "LONG":
        price_move = exit - entry
    else:  # SHORT
        price_move = entry - exit
    
    if sizing_mode == "contracts":
        return (price_move / tick_size) * tick_value * qty
    else:  # "cfd"
        return price_move * point_value * qty


def load_data(path: str, engine: Optional[str] = None, csv_format: str = "default", data_utc_offset: int = 0, data_tz: str = "") -> pd.DataFrame:
    if csv_format == "default":
        usecols = [DT_COL, O_COL, H_COL, L_COL, C_COL]
        df = pd.read_csv(
            path,
            usecols=usecols,
            engine=engine,
        )
        # Parse timestamps (assumed naive)
        df.rename(columns={DT_COL: "dt", O_COL: "open", H_COL: "high", L_COL: "low", C_COL: "close"}, inplace=True)
        if data_tz:
            # Localize to provided timezone then convert to UTC
            local = pd.to_datetime(df["dt"], format="%m/%d/%Y %H:%M").dt.tz_localize(ZoneInfo(data_tz))
            df["dt"] = local.tz_convert("UTC")
        else:
            # Treat parsed as UTC then apply fixed offset (negative means data were behind UTC)
            df["dt"] = pd.to_datetime(df["dt"], format="%m/%d/%Y %H:%M", utc=True)
            if data_utc_offset != 0:
                df["dt"] = df["dt"] - pd.Timedelta(hours=data_utc_offset)
    else:  # MetaTrader format: date,time,open,high,low,close,vol (no header)
        df = pd.read_csv(
            path,
            header=None,
            names=["date","time","open","high","low","close","vol"],
            usecols=[0,1,2,3,4,5],
        )
        # Combine date and time like 'YYYY.MM.DD HH:MM'
        dt_str = df["date"].astype(str) + " " + df["time"].astype(str)
        if data_tz:
            local = pd.to_datetime(dt_str, format="%Y.%m.%d %H:%M").dt.tz_localize(ZoneInfo(data_tz))
            df["dt"] = local.tz_convert("UTC")
        else:
            df["dt"] = pd.to_datetime(dt_str, format="%Y.%m.%d %H:%M", utc=True)
            if data_utc_offset != 0:
                df["dt"] = df["dt"] - pd.Timedelta(hours=data_utc_offset)
        # Ensure numeric types
        for k in ("open","high","low","close"):
            df[k] = pd.to_numeric(df[k], errors="coerce")

    # Downcast floats to float32 to speed ops / reduce RAM
    if DOWNCAST_FLOATS:
        for k in ("open","high","low","close"):
            df[k] = pd.to_numeric(df[k], downcast="float")

    # Keep UTC time only
    df["dt_utc"] = df["dt"]
    df["date_utc"] = df["dt_utc"].dt.date
    df["hour"] = df["dt_utc"].dt.hour.astype(np.int16)
    df["minute_of_day_utc"] = df["dt_utc"].dt.hour.astype(np.int16) * 60 + df["dt_utc"].dt.minute.astype(np.int16)

    # Return only UTC columns plus OHLC
    return df[["dt","dt_utc","date_utc","hour","minute_of_day_utc","open","high","low","close"]]


class Arr:
    pass

def build_arrays(df: pd.DataFrame) -> Arr:
    """Build one-time NumPy arrays and day boundaries from dataframe."""
    arr = Arr()
    arr.dt = df["dt_utc"].to_numpy()
    arr.date = df["date_utc"].to_numpy()
    arr.mins = df["minute_of_day_utc"].to_numpy()
    arr.op = df["open"].to_numpy()
    arr.hi = df["high"].to_numpy()
    arr.lo = df["low"].to_numpy()
    arr.cl = df["close"].to_numpy()

    u_days, starts = np.unique(arr.date, return_index=True)
    ends = np.r_[starts[1:], len(arr.date)]
    arr.u_days, arr.starts, arr.ends = u_days, starts, ends
    return arr

def win_idx(mins: np.ndarray, i0: int, i1: int, start_min: int, end_min: int):
    m = (mins[i0:i1] >= start_min) & (mins[i0:i1] < end_min)
    if not np.any(m):
        return None, None
    rel = np.flatnonzero(m)
    return i0 + rel[0], i0 + rel[-1] + 1  # [start, end)


def _sniff_csv_format_and_path(path: Optional[str], requested_format: str) -> tuple[str, str]:
    """Return (path, csv_format) by sniffing when needed.
    - If path points to an existing file, honor it and requested_format (auto->sniff).
    - If path is empty/missing, try to find a CSV in the workspace and sniff it.
    Sniff rules:
      - If file has a header with 'Time,Open,High,Low,Close' -> default
      - If first line looks like 'YYYY.MM.DD,HH:MM,...' -> mt
    """
    import glob
    fmt = requested_format
    p = path
    if not p or not os.path.exists(p):
        # try common names in cwd
        candidates = []
        for pat in ("*.csv", "data/*.csv", "./*.csv"):
            candidates.extend(glob.glob(pat))
        # prefer files that look like MT or default
        # Choose the first stable-sorted candidate
        candidates = sorted(set(candidates))
        if not candidates:
            raise FileNotFoundError("No CSV file found. Provide --csv path.")
        p = candidates[0]
    # Sniff format if auto
    if fmt == "auto":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        if "," in first and first.lower().startswith("time,open,high,low,close"):
            fmt = "default"
        else:
            # try MT shape: 'YYYY.MM.DD,HH:MM,open,high,low,close,...'
            parts = first.split(",")
            if len(parts) >= 6 and len(parts[0].split(".")) == 3 and ":" in parts[1]:
                fmt = "mt"
            else:
                # fallback to default
                fmt = "default"
    return p, fmt


def iter_trading_days_arr(arr: Arr, config):
    """Iterate using precomputed array day windows (index-based)."""
    total_days = len(arr.u_days)
    for k, (i0, i1) in enumerate(zip(arr.starts, arr.ends)):
        if k % 500 == 0:
            print(f"Processing day {k + 1}/{total_days} ({k/total_days*100:.1f}%)", flush=True)
        if k == 0:
            continue  # need previous day

        prev_i0, prev_i1 = arr.starts[k - 1], arr.ends[k - 1]

        def _m(hhmm: str) -> int:
            h, m = map(int, hhmm.split(":"))
            return h * 60 + m

        # Asia: prev + curr (configurable)
        a0a, a1a = win_idx(arr.mins, prev_i0, prev_i1, _m(config.asia_prev_start), _m(config.asia_prev_end))
        a0b, a1b = win_idx(arr.mins, i0, i1, _m(config.asia_curr_start), _m(config.asia_curr_end))
        if a0a is None and a0b is None:
            continue

        # Pre-London
        p0, p1 = win_idx(arr.mins, i0, i1, _m(config.pre_start), _m(config.pre_end))

        # London (optional)
        l0, l1 = win_idx(arr.mins, i0, i1, _m(config.london_start), _m(config.london_end))

        # Takeout window (config)
        ts_h, ts_m = map(int, config.takeout_start.split(":"))
        te_h, te_m = map(int, config.takeout_end.split(":"))
        t0, t1 = win_idx(arr.mins, i0, i1, ts_h * 60 + ts_m, te_h * 60 + te_m)

        if (l0 is None) or (t0 is None):
            continue

        yield {
            "day": pd.Timestamp(arr.u_days[k]),
            "asia_prev": (a0a, a1a),
            "asia_curr": (a0b, a1b),
            "pre": (p0, p1),
            "london": (l0, l1),
            "takeout": (t0, t1),
            "range": (i0, i1),
        }   

def determine_trade_entries(asia_info: AsiaKZInfo, takeouts: list[Tuple[str, pd.Timestamp, int]],
                           arr: Arr, t0: int, t1: int, p0: Optional[int], p1: Optional[int], config) -> list[dict]:
    """Determine trade entries based on Asia KZ analysis with KZ levels as targets"""
    trades = []
    
    if asia_info is None or t0 is None or t1 is None or t0 >= t1:
        return trades
    
    # Get pre-London range
    if p0 is None or p1 is None or p0 >= p1:
        return trades
    pre_lo = float(np.min(arr.lo[p0:p1]))
    pre_hi = float(np.max(arr.hi[p0:p1]))
    if pre_lo == 0.0 and pre_hi == 0.0:
        return trades  # No valid pre-London range
    
    # Determine target based on KZ extremes
    target_price = None
    target_direction = None
    
    # Scenario 1: Only one extreme made in KZ (high OR low, not both)
    if (asia_info.high_made_in_kz and not asia_info.low_made_in_kz):
        target_price = asia_info.kz_high
        target_direction = "LONG"  # Target above current levels
    elif (asia_info.low_made_in_kz and not asia_info.high_made_in_kz):
        target_price = asia_info.kz_low  
        target_direction = "SHORT"  # Target below current levels
    
    # Scenario 2: Both extremes made in KZ - use takeout direction
    elif asia_info.high_made_in_kz and asia_info.low_made_in_kz:
        if not takeouts:
            return trades  # Need takeouts for this scenario
        # Use first takeout to determine bias
        first_takeout_type, takeout_time, _ = takeouts[0]
        if first_takeout_type == "high_takeout":
            target_price = asia_info.kz_low  # Target low after high takeout
            target_direction = "SHORT"
        else:
            target_price = asia_info.kz_high  # Target high after low takeout  
            target_direction = "LONG"
    
    if target_price is None or target_direction is None:
        return trades
    
    # Find entry trigger based on pre-London range and entry mode
    entry_candidates = []
    
    if target_direction == "LONG" and target_price > pre_hi:
        # Looking to go long towards a target above pre-London high
        if config.entry_mode == "break":
            # Enter on first close above pre-London high
            m = arr.cl[t0:t1] > pre_hi
            idx = np.flatnonzero(m)
            if idx.size:
                entry_candidates.append(t0 + int(idx[0]))
        elif config.entry_mode == "retest":
            # First find the break, then wait for retest
            m = arr.cl[t0:t1] > pre_hi
            idx = np.flatnonzero(m)
            if idx.size:
                br = t0 + int(idx[0])
                m2 = arr.lo[br+1:t1] <= pre_hi
                idx2 = np.flatnonzero(m2)
                if idx2.size:
                    entry_candidates.append(br + 1 + int(idx2[0]))
                        
    elif target_direction == "SHORT" and target_price < pre_lo:
        # Looking to go short towards a target below pre-London low
        if config.entry_mode == "break":
            # Enter on first close below pre-London low
            m = arr.cl[t0:t1] < pre_lo
            idx = np.flatnonzero(m)
            if idx.size:
                entry_candidates.append(t0 + int(idx[0]))
        elif config.entry_mode == "retest":
            # First find the break, then wait for retest
            m = arr.cl[t0:t1] < pre_lo
            idx = np.flatnonzero(m)
            if idx.size:
                br = t0 + int(idx[0])
                m2 = arr.hi[br+1:t1] >= pre_lo
                idx2 = np.flatnonzero(m2)
                if idx2.size:
                    entry_candidates.append(br + 1 + int(idx2[0]))
    
    # Create trade setups from valid entry candidates
    for entry_idx in entry_candidates[:config.max_trades]:  # Limit entries per day
        entry_time = pd.Timestamp(arr.dt[entry_idx])
        entry_price = float(arr.cl[entry_idx])  # Enter based on close break
        
        # Calculate stop based on RR ratio
        risk_points = abs(target_price - entry_price) / config.rr_ratio
        
        if target_direction == "LONG":
            stop_price = entry_price - risk_points
        else:
            stop_price = entry_price + risk_points
            
        stop_price = _round_to_tick(stop_price)
        target_price_rounded = _round_to_tick(target_price)
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'target_price': target_price_rounded,
            'stop_price': stop_price,
            'direction': target_direction,
            'scenario': 'kz_target',
            'entry_idx': entry_idx
        })
            
    return trades

def backtest(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity = START_CAPITAL
    trades: list[Trade] = []
    equity_points = []
    days_skipped_sizing = 0
    days_no_kz_setup = 0
    days_early_takeout = 0
    days_no_setups = 0
    days_with_trades = 0
    day_count = 0

    # Build arrays once
    arr = build_arrays(df)

    for day_data in iter_trading_days_arr(arr, config):
        day_count += 1
        if day_count % 500 == 0:  # Reduced progress reporting frequency
            print(f"Processed {day_count} days, {len(trades)} trades, equity: ${equity:,.0f}", flush=True)
            
        day_ts = day_data['day']
        a_prev = day_data['asia_prev']
        a_curr = day_data['asia_curr']
        p0, p1 = day_data['pre']
        l0, l1 = day_data['london']
        t0, t1 = day_data['takeout']

        # Step 1: Analyze Asia session for KZ setup
        asia_info = analyze_asia_session_arr(arr, a_prev, a_curr, config)
        if asia_info is None:
            days_no_kz_setup += 1
            equity_points.append({"date": day_ts, "equity": equity})
            continue
            
        # Step 2: Early takeout (optional): if KZ high/low taken out between KZ end and London KZ start, skip day
        early = False
        if config.early_takeout == "on":
            # Build an early window: from end of KZ (in its segment) to London KZ start (01:00) of current day
            def _m(hhmm: str) -> int:
                h, m = map(int, hhmm.split(":"))
                return h * 60 + m
            kz_end_min = _m(config.kz_end)
            london_kz_start_min = _m(config.london_start)

            # Determine where KZ lived (prev or curr) by checking times
            a_prev0, a_prev1 = a_prev
            a_curr0, a_curr1 = a_curr
            kz_seg = None  # ('prev' or 'curr')
            if a_curr0 is not None and a_curr1 is not None:
                k0, k1 = win_idx(arr.mins, a_curr0, a_curr1, _m(config.kz_start), kz_end_min)
                if k0 is not None and k1 is not None and k0 < k1:
                    kz_seg = ('curr', k0, k1)
            if kz_seg is None and a_prev0 is not None and a_prev1 is not None:
                k0, k1 = win_idx(arr.mins, a_prev0, a_prev1, _m(config.kz_start), kz_end_min)
                if k0 is not None and k1 is not None and k0 < k1:
                    kz_seg = ('prev', k0, k1)

            # Early window slices
            slices = []
            if config.early_window == 'pre':
                # Use pre-London window exactly
                if p0 is not None and p1 is not None and p0 < p1:
                    slices.append((p0, p1))
            else:
                # kz-to-london: KZ end → London KZ start (wrap-aware)
                if kz_seg is not None:
                    seg, k0, k1 = kz_seg
                    kz_end_idx = k1
                    if seg == 'curr':
                        # From kz_end_idx to london_kz_start in current day
                        ci0, ci1 = day_data['range']
                        w0, w1 = win_idx(arr.mins, ci0, ci1, kz_end_min, london_kz_start_min)
                        if w0 is not None and w1 is not None and w0 < w1:
                            slices.append((w0, w1))
                    else:
                        # KZ in prev day: from k1 to end of prev Asia window, plus from start of curr day to london_kz_start
                        ap0, ap1 = a_prev
                        if ap1 is not None and kz_end_idx < ap1:
                            slices.append((kz_end_idx, ap1))
                        ci0, ci1 = day_data['range']
                        w0, w1 = win_idx(arr.mins, ci0, ci1, 0, london_kz_start_min)
                        if w0 is not None and w1 is not None and w0 < w1:
                            slices.append((w0, w1))

            # Check takeout across slices
            if slices:
                if asia_info.high_made_in_kz:
                    if any(np.any(arr.hi[s0:s1] >= asia_info.kz_high) for s0, s1 in slices):
                        early = True
                if asia_info.low_made_in_kz:
                    if any(np.any(arr.lo[s0:s1] <= asia_info.kz_low) for s0, s1 in slices):
                        early = True
        if early:
            days_early_takeout += 1
            equity_points.append({"date": day_ts, "equity": equity})
            continue  # Skip day if early takeout occurred
            
        # Step 3: Get the takeout window (configurable)
        # Already have t0,t1 from iterator
        if t0 is None or t1 is None or t0 >= t1:
            equity_points.append({"date": day_ts, "equity": equity})
            continue
            
        # Step 4: Monitor takeout window for KZ level breaches
        takeouts = monitor_takeouts_arr(arr, t0, t1, asia_info)
        
        # Step 5: Determine trade entries based on scenario (KZ levels as targets)
        potential_trades = determine_trade_entries(asia_info, takeouts, arr, t0, t1, p0, p1, config)
        
        if not potential_trades:
            days_no_setups += 1
            equity_points.append({"date": day_ts, "equity": equity})
            continue
        
        # Step 6: Execute trades with realistic execution and re-entry limits
        day_trades = 0
        last_exit_time = None

        for trade_setup in potential_trades:
            # Check daily trade limit
            if day_trades >= config.max_trades:
                break

            # Check cooldown period
            if last_exit_time is not None:
                time_since_exit = (trade_setup['entry_time'] - last_exit_time).total_seconds() / 60
                if time_since_exit < config.cooldown_min:
                    continue

            entry_idx = trade_setup['entry_idx']
            entry_time = trade_setup['entry_time']
            target_price = trade_setup['target_price']
            direction = trade_setup['direction']

            # Realistic execution: fill at next bar's open
            entry_fill = fill_next_open(entry_idx, arr.op, trade_setup['entry_price'])

            # Recalculate stop/target based on actual fill (maintain RR ratio)
            if direction == "LONG":
                risk_points = abs(target_price - entry_fill) / config.rr_ratio
                stop_fill = entry_fill - risk_points
            else:
                risk_points = abs(target_price - entry_fill) / config.rr_ratio
                stop_fill = entry_fill + risk_points

            stop_fill = _round_to_tick(stop_fill)

            # Calculate position size using risk-first approach
            qty = compute_qty(entry_fill, stop_fill, equity, config.risk_pct,
                              config.sizing, TICK_SIZE, TICK_VALUE, POINT_VALUE)

            # Sizing audit values (for output diagnostics)
            per_unit_risk_points = max(TICK_SIZE, abs(entry_fill - stop_fill))
            if config.sizing == "contracts":
                per_unit_risk_usd = (per_unit_risk_points / TICK_SIZE) * config.tick_value
            else:
                per_unit_risk_usd = per_unit_risk_points * config.point_value
            risk_target_usd = equity * config.risk_pct
            risk_allocated_usd = per_unit_risk_usd * qty

            if qty <= 0:
                days_skipped_sizing += 1
                continue

            # Apply position limits
            if config.sizing == "contracts":
                MAX_POSITION_SIZE = 10000
                MAX_EQUITY_CAP = 100000000  # $100M
                qty = min(qty, MAX_POSITION_SIZE)
                if equity >= MAX_EQUITY_CAP:
                    days_skipped_sizing += 1
                    continue

            # Find exit using remaining bars after entry (realistic execution)
            r0 = entry_idx + 2
            if r0 >= t1:
                # No bars left for exit, close at target or last available price
                exit_time = pd.Timestamp(arr.dt[t1-1])
                exit_price = float(arr.cl[t1-1])
                exit_reason = "EOD"
            else:
                # Search for TP/SL hits in remaining bars
                hi_array = arr.hi[r0:t1]
                lo_array = arr.lo[r0:t1]
                op_array = arr.op[r0:t1]

                tp_idx = None
                sl_idx = None

                if direction == "LONG":
                    tp_mask = hi_array >= target_price
                    sl_mask = lo_array <= stop_fill

                    tp_hits = np.flatnonzero(tp_mask)
                    sl_hits = np.flatnonzero(sl_mask)
                    if tp_hits.size:
                        tp_idx = int(tp_hits[0])
                    if sl_hits.size:
                        sl_idx = int(sl_hits[0])
                else:  # SHORT
                    tp_mask = lo_array <= target_price
                    sl_mask = hi_array >= stop_fill

                    tp_hits = np.flatnonzero(tp_mask)
                    sl_hits = np.flatnonzero(sl_mask)
                    if tp_hits.size:
                        tp_idx = int(tp_hits[0])
                    if sl_hits.size:
                        sl_idx = int(sl_hits[0])

                # Resolve ambiguous bars and get exit
                exit_bar_idx, exit_reason = resolve_ambiguous(tp_idx, sl_idx, config.ambiguous)

                if exit_bar_idx is not None:
                    actual_exit_idx = r0 + exit_bar_idx
                    exit_time = pd.Timestamp(arr.dt[actual_exit_idx])

                    # Realistic exit fill: next bar's open with slippage on stops
                    if exit_reason == "TP":
                        # Fill TP exactly at target (no favorable slippage) to keep RR consistent
                        exit_price = float(target_price)
                    elif exit_reason == "SL":
                        # Stop fill: worst-of next open vs stop level, plus adverse slippage
                        open_next = float(fill_next_open(exit_bar_idx, op_array, stop_fill))
                        if direction == "LONG":
                            worst = min(open_next, float(stop_fill))
                            exit_price = worst - (SLIPPAGE_TICKS * TICK_SIZE)
                        else:  # SHORT
                            worst = max(open_next, float(stop_fill))
                            exit_price = worst + (SLIPPAGE_TICKS * TICK_SIZE)
                    else:  # NEUTRAL
                        exit_price = float(op_array[exit_bar_idx])
                else:
                    # No exit found, close at end of takeout window
                    exit_time = pd.Timestamp(arr.dt[t1-1])
                    exit_price = float(arr.cl[t1-1])
                    exit_reason = "EOD"

            # Calculate PnL and R-multiple
            pnl_currency = calculate_pnl(entry_fill, exit_price, qty, direction,
                                         config.sizing, TICK_SIZE, TICK_VALUE, POINT_VALUE)

            # Calculate R-multiple (actual return vs risk)
            risk_points = abs(entry_fill - stop_fill)
            if risk_points > 0:
                pnl_points = (exit_price - entry_fill) if direction == "LONG" else (entry_fill - exit_price)
                r_mult = pnl_points / risk_points
            else:
                r_mult = 0.0

            equity += pnl_currency
            last_exit_time = exit_time
            day_trades += 1

            # Create trade record
            trade = Trade(
                side=direction,
                session="Asia-London",
                day=day_ts,
                entry_time=entry_time,
                entry_price=float(entry_fill),
                stop_price=float(stop_fill),
                target_price=float(target_price),
                exit_time=exit_time,
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                r_multiple=float(r_mult),
                pnl_currency=float(pnl_currency),
                qty=float(qty),
                risk_target_usd=float(risk_target_usd),
                risk_allocated_usd=float(risk_allocated_usd),
                per_unit_risk_usd=float(per_unit_risk_usd),
            )
            trades.append(trade)

        # Track days that had potential trades
        if potential_trades:
            days_with_trades += 1

        equity_points.append({"date": day_ts, "equity": equity})

    # Create output dataframes (build once)
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
    eq_df = pd.DataFrame(equity_points).drop_duplicates(subset=["date"], keep="last")

    # Write outputs once
    if not config.no_write and not trades_df.empty:
        # Format timestamps as UTC strings
        trades_df_output = trades_df.copy()
        trades_df_output["entry_time"] = trades_df_output["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        trades_df_output["exit_time"] = trades_df_output["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        trades_df_output.to_csv("trades.csv", index=False)
    if not config.no_write:
        eq_df.to_csv("equity_curve.csv", index=False)

    # Summary calculations
    if trades_df.empty:
        print("No trades generated.")
        win_rate = 0.0
        net_pnl = 0.0
        total_trades = 0
        avg_trades_per_day = 0.0
    else:
        win_rate = (trades_df["exit_reason"] == "TP").mean()
        net_pnl = trades_df['pnl_currency'].sum()
        total_trades = len(trades_df)
        avg_trades_per_day = total_trades / max(1, days_with_trades)
    
    # Results display
    print()
    print("=== RESULTS ===")
    print(f"Risk-to-Reward Ratio: 1:{config.rr_ratio}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total PnL: ${net_pnl:.2f}")
    print(f"Total Trades: {total_trades:,}")
    print(f"Avg Trades/Day: {avg_trades_per_day:.2f}")
    print(f"Days skipped (sizing): {days_skipped_sizing}")
    
    # Sanity check warnings
    if avg_trades_per_day > 3:
        print(f"WARNING: High trade frequency ({avg_trades_per_day:.2f}/day) - consider reviewing config")
    if win_rate > 0.70 and total_trades > 100:
        print(f"WARNING: High win rate ({win_rate:.1%}) - check for overfitting")
    
    print()
    print("=== DEBUG STATS ===")
    print(f"Days processed: {day_count}")
    print(f"Days with no KZ setup: {days_no_kz_setup}")
    print(f"Days with early takeout: {days_early_takeout}")
    print(f"Days with no setups: {days_no_setups}")
    print(f"Days with trades: {days_with_trades}")

    return trades_df, eq_df


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    
    # Create config object for backtest
    class Config:
        def __init__(self, args):
            self.csv_path = args.csv
            self.engine = args.engine
            self.csv_format = args.csv_format
            self.sizing = args.sizing
            self.tick_value = args.tick_value
            self.point_value = args.point_value
            self.risk_pct = args.risk_pct
            self.rr_ratio = args.rr_ratio
            self.takeout_start = args.takeout_start
            self.takeout_end = args.takeout_end
            # Session windows
            self.asia_prev_start = args.asia_prev_start
            self.asia_prev_end = args.asia_prev_end
            self.asia_curr_start = args.asia_curr_start
            self.asia_curr_end = args.asia_curr_end
            self.kz_start = args.kz_start
            self.kz_end = args.kz_end
            self.pre_start = args.pre_start
            self.pre_end = args.pre_end
            self.london_start = args.london_start
            self.london_end = args.london_end
            self.entry_mode = args.entry_mode
            self.ambiguous = args.ambiguous
            self.max_trades = args.max_trades
            self.cooldown_min = args.cooldown_min
            self.no_write = args.no_write
            self.data_utc_offset = args.data_utc_offset
            self.data_tz = args.data_tz
            self.early_takeout = args.early_takeout
            self.early_window = args.early_window
    
    config = Config(args)
    # Resolve CSV path and format if needed
    resolved_csv, resolved_fmt = _sniff_csv_format_and_path(config.csv_path, args.csv_format)
    config.csv_path = resolved_csv
    config.csv_format = resolved_fmt
    
    print("=== NAS100 Killzone Backtester (Realistic Execution) ===")
    print(f"CSV: {config.csv_path}")
    print(f"Sizing: {config.sizing}, Tick Value: ${config.tick_value}, Point Value: ${config.point_value}")
    print(f"Risk: {config.risk_pct:.1%}, RR: 1:{config.rr_ratio}")
    print(f"Takeout Window: {config.takeout_start}-{config.takeout_end} UTC")
    print(f"Asia: prev {config.asia_prev_start}-{config.asia_prev_end}, curr {config.asia_curr_start}-{config.asia_curr_end}; KZ {config.kz_start}-{config.kz_end}; Pre {config.pre_start}-{config.pre_end}; London {config.london_start}-{config.london_end}")
    print(f"Entry Mode: {config.entry_mode}, Ambiguous Policy: {config.ambiguous}")
    print(f"Max Trades/Day: {config.max_trades}, Cooldown: {config.cooldown_min} min")
    if getattr(config, "data_tz", ""):
        print(f"Data timezone: {config.data_tz} (DST-aware ingest → UTC)")
    elif config.data_utc_offset:
        print(f"Data UTC offset: {config.data_utc_offset}h (fixed-offset ingest → UTC)")
    print()
    
    # Timing telemetry
    start_time = time.perf_counter()
    
    print("Loading data...")
    load_start = time.perf_counter()
    df = load_data(config.csv_path, config.engine, config.csv_format, config.data_utc_offset, config.data_tz)
    # Parquet caching
    base = os.path.splitext(config.csv_path)[0]
    tz_tag = (config.data_tz or "").replace("/", "_") or "none"
    CACHE = f"{base}.{config.csv_format}.offset{config.data_utc_offset}.tz{tz_tag}.parquet"
    try:
        if not os.path.exists(CACHE):
            df.to_parquet(CACHE)
        df = pd.read_parquet(CACHE)
    except Exception:
        # Fallback to in-memory df if parquet not available
        pass
    load_time = time.perf_counter() - load_start
    print(f"Data loaded: {len(df):,} rows in {load_time:.2f}s")

    # Always write fresh Asia/KZ verification report
    try:
        if os.path.exists(ASIA_REPORT_PATH):
            os.remove(ASIA_REPORT_PATH)
    except Exception:
        pass
    print(f"Writing Asia/KZ report → {ASIA_REPORT_PATH}")
    build_asia_report(df, ASIA_REPORT_PATH, config)
    
    print("Running backtest...")
    backtest_start = time.perf_counter()
    # Optional: wipe previous trades.csv to avoid overlap, only when writing is enabled
    if not config.no_write:
        try:
            if os.path.exists("trades.csv"):
                os.remove("trades.csv")
        except Exception:
            pass
    trades, equity = backtest(df, config)
    backtest_time = time.perf_counter() - backtest_start
    
    total_time = time.perf_counter() - start_time
    
    print()
    print("=== Timing Summary ===")
    print(f"Data load: {load_time:.2f}s")
    print(f"Backtest:  {backtest_time:.2f}s") 
    print(f"Total:     {total_time:.2f}s")
    print()
    print("Output files: trades.csv, equity_curve.csv")
