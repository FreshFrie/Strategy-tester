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

# =============================
# ======== CONFIG ============
# =============================

# Defaults (overridable by CLI)
CSV_PATH = "./NQ_5Years_8_11_2024.csv"
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
TAKEOUT_START_UTC = "06:00"
TAKEOUT_END_UTC = "12:00"
ENTRY_MODE = "break"  # {"break","retest"} relative to pre-London range
AMBIGUOUS_POLICY = "worst"  # {"worst","neutral","best"}
MAX_TRADES_PER_DAY = 2
COOLDOWN_MIN = 30
NEXT_BAR_EXECUTION = True

# Performance settings
READ_ENGINE: Optional[str] = None  # set to "pyarrow" if available
DOWNCAST_FLOATS = True

def parse_args():
    parser = argparse.ArgumentParser(description="NAS100 Killzone Backtester")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to CSV file")
    parser.add_argument("--engine", choices=["pandas", "pyarrow"], help="CSV read engine")
    parser.add_argument("--sizing", choices=["contracts", "cfd"], default=SIZING_MODE, help="Position sizing mode")
    parser.add_argument("--tick-value", type=float, default=TICK_VALUE, help="$ per tick (contracts mode)")
    parser.add_argument("--point-value", type=float, default=POINT_VALUE, help="$ per point (CFD mode)")
    parser.add_argument("--risk-pct", type=float, default=RISK_PCT_PER_TRADE, help="Risk percentage per trade")
    parser.add_argument("--rr-ratio", type=float, default=R_MULT, help="Risk-to-Reward ratio (e.g., 1.5 for 1:1.5)")
    parser.add_argument("--takeout-start", default=TAKEOUT_START_UTC, help="Takeout window start time (HH:MM UTC)")
    parser.add_argument("--takeout-end", default=TAKEOUT_END_UTC, help="Takeout window end time (HH:MM UTC)")
    parser.add_argument("--entry-mode", choices=["break", "retest"], default=ENTRY_MODE, help="Entry mode relative to pre-London range")
    parser.add_argument("--ambiguous", choices=["worst", "neutral", "best"], default=AMBIGUOUS_POLICY, help="Policy for ambiguous TP/SL bars")
    parser.add_argument("--max-trades", type=int, default=MAX_TRADES_PER_DAY, help="Maximum trades per day")
    parser.add_argument("--cooldown-min", type=int, default=COOLDOWN_MIN, help="Cooldown minutes between trades")
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

def analyze_asia_session(asia_df: pd.DataFrame) -> Optional[AsiaKZInfo]:
    """Analyze Asia session to determine if KZ made session extremes"""
    if asia_df.empty:
        return None
        
    # Get session extremes for entire 22:00-04:00 period
    session_high = asia_df['high'].max()
    session_low = asia_df['low'].min()
    
    # Extract Asia Killzone period (00:00-02:00 UTC)
    kz_mask = (asia_df['hour'] >= 0) & (asia_df['hour'] < 2)
    kz_df = asia_df[kz_mask]
    
    if kz_df.empty:
        return None
        
    # Get KZ extremes
    kz_high = kz_df['high'].max()
    kz_low = kz_df['low'].min()
    
    # Find exact times when extremes were made (with safety checks)
    kz_high_indices = kz_df[kz_df['high'] == kz_high].index
    kz_low_indices = kz_df[kz_df['low'] == kz_low].index
    
    if len(kz_high_indices) == 0 or len(kz_low_indices) == 0:
        return None
        
    kz_high_idx = kz_high_indices[0]
    kz_low_idx = kz_low_indices[0]
    kz_high_time = asia_df.loc[kz_high_idx, 'dt_utc']
    kz_low_time = asia_df.loc[kz_low_idx, 'dt_utc']
    
    # Check if session extremes occurred during KZ (robust comparison)
    high_made_in_kz = np.isclose(session_high, kz_high, atol=TICK_SIZE/2)
    low_made_in_kz = np.isclose(session_low, kz_low, atol=TICK_SIZE/2)
    
    return AsiaKZInfo(
        session_high=session_high,
        session_low=session_low,
        kz_high=kz_high,
        kz_low=kz_low,
        high_made_in_kz=high_made_in_kz,
        low_made_in_kz=low_made_in_kz,
        kz_high_time=kz_high_time,
        kz_low_time=kz_low_time
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


def load_data(path: str, engine: Optional[str] = None) -> pd.DataFrame:
    usecols = [DT_COL, O_COL, H_COL, L_COL, C_COL]
    df = pd.read_csv(
        path,
        usecols=usecols,
        engine=engine,
    )
    # Parse UTC datetimes (naive in file → set utc=True)
    df.rename(columns={DT_COL: "dt", O_COL: "open", H_COL: "high", L_COL: "low", C_COL: "close"}, inplace=True)
    df["dt"] = pd.to_datetime(df["dt"], format="%m/%d/%Y %H:%M", utc=True)

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


def iter_trading_days(df: pd.DataFrame):
    """Iterate through trading days, yielding Asia, Pre-London, London, and Takeout sessions"""
    all_days = np.array(sorted(pd.unique(df["date_utc"])))
    total_days = len(all_days)
    
    for day_idx, d in enumerate(all_days):
        if day_idx % 100 == 0:  # Reduced progress reporting frequency
            print(f"Processing day {day_idx + 1}/{total_days} ({day_idx/total_days*100:.1f}%)", flush=True)
            
        # Need to look at previous day for Asia session (22:00-04:00 next day)
        if day_idx == 0:
            continue  # Skip first day as we need previous Asia session
            
        prev_day = all_days[day_idx - 1]
        curr_day = d
        curr_day_ts = pd.Timestamp(curr_day)
        
        # Asia session: 22:00 previous day to 04:00 current day
        asia_mask = ((df["date_utc"] == prev_day) & (df["hour"] >= 22)) | \
                   ((df["date_utc"] == curr_day) & (df["hour"] < 4))
        asia_df = df[asia_mask].copy()
        
        # Pre-London: 04:00-06:00 current day  
        pre_london_df = slice_utc_window(df, curr_day_ts, "04:00", "06:00")
        
        # London session: 06:00-09:00 current day (for compatibility)
        london_df = slice_utc_window(df, curr_day_ts, "06:00", "09:00")
        
        # Only yield if we have all necessary sessions
        if not asia_df.empty and not london_df.empty:
            yield {
                'day': curr_day_ts,
                'asia_session': asia_df,
                'pre_london': pre_london_df, 
                'london_session': london_df
            }


def determine_trade_entries(asia_info: AsiaKZInfo, takeouts: list[Tuple[str, pd.Timestamp]], 
                           takeout_df: pd.DataFrame, pre_london_df: pd.DataFrame, config) -> list[dict]:
    """Determine trade entries based on Asia KZ analysis with KZ levels as targets"""
    trades = []
    
    if asia_info is None or takeout_df.empty:
        return trades
    
    # Get pre-London range
    pre_lo, pre_hi = pre_london_range(pre_london_df)
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
        first_takeout_type, takeout_time = takeouts[0]
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
            break_idx = first_close_break(takeout_df, pre_hi, "above")
            if break_idx is not None:
                entry_candidates.append(break_idx)
        elif config.entry_mode == "retest":
            # First find the break, then wait for retest
            break_idx = first_close_break(takeout_df, pre_hi, "above") 
            if break_idx is not None:
                # Look for retest (low back to pre_hi) after break
                retest_df = takeout_df.iloc[break_idx+1:]
                if not retest_df.empty:
                    retest_mask = retest_df['low'] <= pre_hi
                    if retest_mask.any():
                        retest_idx = retest_mask.idxmax()
                        entry_candidates.append(retest_idx)
                        
    elif target_direction == "SHORT" and target_price < pre_lo:
        # Looking to go short towards a target below pre-London low
        if config.entry_mode == "break":
            # Enter on first close below pre-London low
            break_idx = first_close_break(takeout_df, pre_lo, "below")
            if break_idx is not None:
                entry_candidates.append(break_idx)
        elif config.entry_mode == "retest":
            # First find the break, then wait for retest
            break_idx = first_close_break(takeout_df, pre_lo, "below")
            if break_idx is not None:
                # Look for retest (high back to pre_lo) after break
                retest_df = takeout_df.iloc[break_idx+1:]
                if not retest_df.empty:
                    retest_mask = retest_df['high'] >= pre_lo
                    if retest_mask.any():
                        retest_idx = retest_mask.idxmax()
                        entry_candidates.append(retest_idx)
    
    # Create trade setups from valid entry candidates
    for entry_idx in entry_candidates[:config.max_trades]:  # Limit entries per day
        entry_time = takeout_df.loc[entry_idx, 'dt_utc']
        entry_price = takeout_df.loc[entry_idx, 'close']  # Enter based on close break
        
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

    for day_data in iter_trading_days(df):
        day_count += 1
        if day_count % 200 == 0:  # Reduced progress reporting frequency
            print(f"Processed {day_count} days, {len(trades)} trades, equity: ${equity:,.0f}", flush=True)
            
        day_ts = day_data['day']
        asia_session = day_data['asia_session']
        pre_london = day_data['pre_london']
        london_session = day_data['london_session']
        
        # Step 1: Analyze Asia session for KZ setup
        asia_info = analyze_asia_session(asia_session)
        if asia_info is None:
            days_no_kz_setup += 1
            equity_points.append({"date": day_ts, "equity": equity})
            continue
            
        # Step 2: Check for early takeout (04:00-06:00 UTC)
        if check_early_takeout(pre_london, asia_info):
            days_early_takeout += 1
            equity_points.append({"date": day_ts, "equity": equity})
            continue  # Skip day if early takeout occurred
            
        # Step 3: Get the takeout window (configurable)
        takeout_df = slice_utc_window(df, day_ts, config.takeout_start, config.takeout_end)
        if takeout_df.empty:
            equity_points.append({"date": day_ts, "equity": equity})
            continue
            
        # Step 4: Monitor takeout window for KZ level breaches
        takeouts = monitor_takeouts(takeout_df, asia_info)
        
        # Step 5: Determine trade entries based on scenario (KZ levels as targets)
        potential_trades = determine_trade_entries(asia_info, takeouts, takeout_df, pre_london, config)
        
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
            stop_price = trade_setup['stop_price']
            direction = trade_setup['direction']
            
            # Realistic execution: fill at next bar's open
            op_array = takeout_df['open'].values
            entry_fill = fill_next_open(entry_idx, op_array, trade_setup['entry_price'])
            
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
            remaining_df = takeout_df.iloc[entry_idx + 2:]  # Start from 2 bars after entry signal
            if remaining_df.empty:
                # No bars left for exit, close at target or last available price
                exit_time = takeout_df.iloc[-1]['dt_utc']
                exit_price = takeout_df.iloc[-1]['close']
                exit_reason = "EOD"
            else:
                # Search for TP/SL hits in remaining bars
                hi_array = remaining_df['high'].values
                lo_array = remaining_df['low'].values
                op_array = remaining_df['open'].values
                
                tp_idx = None
                sl_idx = None
                
                if direction == "LONG":
                    # Find TP and SL hits
                    tp_mask = hi_array >= target_price
                    sl_mask = lo_array <= stop_fill
                    
                    if tp_mask.any():
                        tp_idx = np.argmax(tp_mask)
                    if sl_mask.any():
                        sl_idx = np.argmax(sl_mask)
                        
                else:  # SHORT
                    tp_mask = lo_array <= target_price
                    sl_mask = hi_array >= stop_fill
                    
                    if tp_mask.any():
                        tp_idx = np.argmax(tp_mask)
                    if sl_mask.any():
                        sl_idx = np.argmax(sl_mask)
                
                # Resolve ambiguous bars and get exit
                exit_bar_idx, exit_reason = resolve_ambiguous(tp_idx, sl_idx, config.ambiguous)
                
                if exit_bar_idx is not None:
                    actual_exit_idx = remaining_df.index[exit_bar_idx]
                    exit_time = remaining_df.iloc[exit_bar_idx]['dt_utc']
                    
                    # Realistic exit fill: next bar's open with slippage on stops
                    if exit_reason == "TP":
                        exit_price = fill_next_open(exit_bar_idx, op_array, target_price)
                    elif exit_reason == "SL":
                        # Apply adverse slippage on stops
                        base_exit = fill_next_open(exit_bar_idx, op_array, stop_fill)
                        if direction == "LONG":
                            exit_price = base_exit - (SLIPPAGE_TICKS * TICK_SIZE)  # Adverse slippage
                        else:
                            exit_price = base_exit + (SLIPPAGE_TICKS * TICK_SIZE)  # Adverse slippage
                    else:  # NEUTRAL
                        exit_price = remaining_df.iloc[exit_bar_idx]['open']
                else:
                    # No exit found, close at end of takeout window
                    exit_time = takeout_df.iloc[-1]['dt_utc']
                    exit_price = takeout_df.iloc[-1]['close']
                    exit_reason = "EOD"
                
            # Calculate PnL and R-multiple
            pnl_currency = calculate_pnl(entry_fill, exit_price, qty, direction, 
                                       config.sizing, TICK_SIZE, TICK_VALUE, POINT_VALUE)
            
            # Calculate R-multiple (actual return vs risk)
            risk_points = abs(entry_fill - stop_fill)
            if risk_points > 0:
                if direction == "LONG":
                    pnl_points = exit_price - entry_fill
                else:
                    pnl_points = entry_fill - exit_price
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
            )
            trades.append(trade)
            
        if potential_trades:  # Track days that had potential trades
            days_with_trades += 1
            
        equity_points.append({"date": day_ts, "equity": equity})

    # Create output dataframes (build once)
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
    eq_df = pd.DataFrame(equity_points).drop_duplicates(subset=["date"], keep="last")

    # Write outputs once
    if not trades_df.empty:
        # Format timestamps as UTC strings
        trades_df_output = trades_df.copy()
        trades_df_output["entry_time"] = trades_df_output["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        trades_df_output["exit_time"] = trades_df_output["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        trades_df_output.to_csv("trades.csv", index=False)
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
            self.sizing = args.sizing
            self.tick_value = args.tick_value
            self.point_value = args.point_value
            self.risk_pct = args.risk_pct
            self.rr_ratio = args.rr_ratio
            self.takeout_start = args.takeout_start
            self.takeout_end = args.takeout_end
            self.entry_mode = args.entry_mode
            self.ambiguous = args.ambiguous
            self.max_trades = args.max_trades
            self.cooldown_min = args.cooldown_min
    
    config = Config(args)
    
    print("=== NAS100 Killzone Backtester (Realistic Execution) ===")
    print(f"CSV: {config.csv_path}")
    print(f"Sizing: {config.sizing}, Tick Value: ${config.tick_value}, Point Value: ${config.point_value}")
    print(f"Risk: {config.risk_pct:.1%}, RR: 1:{config.rr_ratio}")
    print(f"Takeout Window: {config.takeout_start}-{config.takeout_end} UTC")
    print(f"Entry Mode: {config.entry_mode}, Ambiguous Policy: {config.ambiguous}")
    print(f"Max Trades/Day: {config.max_trades}, Cooldown: {config.cooldown_min} min")
    print()
    
    # Timing telemetry
    start_time = time.perf_counter()
    
    print("Loading data...")
    load_start = time.perf_counter()
    df = load_data(config.csv_path, config.engine)
    load_time = time.perf_counter() - load_start
    print(f"Data loaded: {len(df):,} rows in {load_time:.2f}s")
    
    print("Running backtest...")
    backtest_start = time.perf_counter()
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
