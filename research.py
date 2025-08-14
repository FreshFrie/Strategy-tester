import argparse
import os
import sys
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import main as bt


@dataclass
class RunMetrics:
    win_rate: float
    expectancy: float
    avg_R: float
    max_dd: float
    sharpe_like_R: float
    trades: int


def _anchor_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _anchored_trading_day_index(df: pd.DataFrame, asia_curr_start: str) -> pd.Series:
    # df must contain dt_utc column (UTC-aware or naive UTC)
    ts = pd.to_datetime(df["dt_utc"])  # Series[datetime64]
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    off_min = _anchor_minutes(asia_curr_start)
    td = (ts.dt.tz_convert("UTC") - pd.Timedelta(minutes=off_min))
    td = td.dt.floor("D")
    return td


def _slice_df_by_quarter(df: pd.DataFrame, asia_curr_start: str, year: int, q: int) -> pd.DataFrame:
    td = _anchored_trading_day_index(df, asia_curr_start)
    start_month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    end_month = start_month + 2
    q_start = pd.Timestamp(year=year, month=start_month, day=1, tz="UTC")
    q_end = (pd.Timestamp(year=year, month=end_month, day=1, tz="UTC") + pd.offsets.MonthEnd(1))
    mask = (td >= q_start.floor("D")) & (td <= q_end.floor("D"))
    return df.loc[mask]


def _build_config(base: Dict[str, Any]) -> SimpleNamespace:
    # Build a duck-typed config for bt.backtest
    return SimpleNamespace(**base)


def _compute_metrics(trades_df: pd.DataFrame, eq_df: pd.DataFrame) -> RunMetrics:
    if trades_df is None or trades_df.empty:
        return RunMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    win_rate = (trades_df["exit_reason"] == "TP").mean()
    expectancy = trades_df["pnl_currency"].mean()
    avg_R = trades_df["r_multiple"].mean()
    # max drawdown from equity curve
    if eq_df is None or eq_df.empty:
        max_dd = 0.0
    else:
        eq = eq_df.set_index("date")["equity"].astype(float)
        roll_max = eq.cummax()
        dd = (eq - roll_max)
        max_dd = float(dd.min())
    r_series = trades_df["r_multiple"].astype(float)
    sharpe_like = 0.0
    if r_series.std(ddof=0) > 1e-12:
        sharpe_like = float(r_series.mean() / r_series.std(ddof=0))
    return RunMetrics(float(win_rate), float(expectancy), float(avg_R), float(max_dd), float(sharpe_like), int(len(trades_df)))


def _default_base_config(csv_path: str, entry_strategy: str, detector_params: Dict[str, Any] | None = None,
                         london_start: str = "02:00", london_end: str = "04:00") -> Dict[str, Any]:
    return {
        "csv_path": csv_path,
        "engine": "pandas",
        "csv_format": "mt",
        "sizing": "cfd",
        "tick_size": bt.TICK_SIZE,
        "tick_value": bt.TICK_VALUE,
        "point_value": bt.POINT_VALUE,
        "risk_pct": 0.01,
        "rr_ratio": 1.5,
        "takeout_start": "01:00",
        "takeout_end": "08:00",
        "asia_prev_start": "00:00",
        "asia_prev_end": "00:00",
        "asia_curr_start": "17:00",
        "asia_curr_end": "23:00",
        "kz_start": "19:30",
        "kz_end": "21:30",
        "pre_start": "23:00",
        "pre_end": "02:00",
        "london_start": london_start,
        "london_end": london_end,
        "entry_mode": "break",
        "entry_strategy": entry_strategy,
        "ambiguous": "worst",
        "max_trades": 1,  # per protocol for research
        "cooldown_min": 30,
        "no_write": True,
    "data_utc_offset": -4,  # default for NY-local MT data; override via CLI as needed
        "data_tz": "",
        "out_dir": os.path.dirname(os.path.abspath(csv_path)),
        "detector_params": detector_params or {},
    }


def _param_grid(strategy: str) -> List[Dict[str, Any]]:
    # Minimal grids to keep runtime reasonable; expand as needed
    if strategy == "cb_vp":
        grid = []
        for alpha in [0.32, 0.40]:
            for beta in [0.8, 1.0]:
                for gamma in [0.6, 0.7]:
                    grid.append({"alpha_prelon_max": alpha, "beta": beta, "gamma": gamma, "atr_n": 14})
        return grid
    if strategy == "mfvg_s":
        return [{"gap_min_atr": g, "fill_frac": f, "atr_n": 14} for g in [0.6, 0.8] for f in [0.5, 0.66]]
    if strategy == "svwap_sh":
        return [{"k_hold": k, "k_slope": s, "slope_min_atr": 0.1, "atr_n": 14} for k in [2,3] for s in [5,10]]
    if strategy == "iib_c":
        return [{"beta": b, "gamma": g, "atr_n": 14} for b in [2.0] for g in [0.7, 0.8]]
    if strategy == "wtr":
        return [{"omega": o, "tau": t, "atr_n": 14} for o in [0.55] for t in [0.65, 0.75]]
    # baseline has no detector params
    return [{}]


def _apply_ablation_to_config(cfg: Dict[str, Any], ablation: str) -> Dict[str, Any]:
    cfg = dict(cfg)
    if ablation == "time_expand":
        cfg["london_start"] = "01:30"
        cfg["london_end"] = "04:30"
    return cfg


def _apply_ablation_to_params(params: Dict[str, Any], strategy: str, ablation: str) -> Dict[str, Any]:
    p = dict(params)
    if strategy == "cb_vp" and ablation == "no_compression":
        p["alpha_prelon_max"] = 1e9
    if ablation == "lower_body_ratio":
        if "gamma" in p:
            p["gamma"] = float(p["gamma"]) * 0.8
    if strategy == "mfvg_s" and ablation == "disable_mfvg_validation":
        p["gap_min_atr"] = 0.0
    return p


def run_fold(df: pd.DataFrame, base_cfg: Dict[str, Any], strategy: str, params: Dict[str, Any],
             ablation: str | None = None, cost_bump_pips: float | None = None,
             time_shift_min: int | None = None) -> Tuple[RunMetrics, pd.DataFrame, pd.DataFrame]:
    cfg_dict = dict(base_cfg)
    if ablation:
        cfg_dict = _apply_ablation_to_config(cfg_dict, ablation)
        params = _apply_ablation_to_params(params, strategy, ablation)
    # time shift of windows
    if time_shift_min:
        def shift(hhmm: str, m: int) -> str:
            h, mm = map(int, hhmm.split(":")); tot = (h*60+mm + m) % (24*60)
            return f"{tot//60:02d}:{tot%60:02d}"
        for key in ("pre_start","pre_end","london_start","london_end"):
            cfg_dict[key] = shift(cfg_dict[key], time_shift_min)
    cfg_dict["detector_params"] = params

    # Override module-level constants as needed
    old_cost = bt.ALL_IN_COST_PIPS
    if cost_bump_pips:
        bt.ALL_IN_COST_PIPS = float(old_cost) + float(cost_bump_pips)

    config = _build_config(cfg_dict)

    # Sync tick/point globals for backtest
    bt.TICK_SIZE = float(config.tick_size)
    bt.TICK_VALUE = float(config.tick_value)
    bt.POINT_VALUE = float(config.point_value)

    trades_df, eq_df = bt.backtest(df, config)

    # Restore costs
    bt.ALL_IN_COST_PIPS = old_cost

    return _compute_metrics(trades_df, eq_df), trades_df, eq_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--strategy", default="cb_vp", choices=["baseline","cb_vp","mfvg_s","svwap_sh","iib_c","wtr","mix"])
    ap.add_argument("--quick", action="store_true", help="Run a tiny grid for sanity")
    args = ap.parse_args()

    # Load data using the backtester's loader + parquet cache behavior
    df = bt.load_data(args.csv, engine="pandas", csv_format="mt", data_utc_offset=0, data_tz="")
    try:
        base = os.path.splitext(args.csv)[0]
        tz_tag = "none"
        cache = f"{base}.mt.offset0.tz{tz_tag}.parquet"
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
    except Exception:
        pass

    grids = _param_grid(args.strategy)
    if args.quick:
        grids = grids[:1]

    results_rows = []
    trades_all: list[pd.DataFrame] = []
    for q in [1,2,3,4]:
        dfq = _slice_df_by_quarter(df, "17:00", args.year, q)
        if dfq.empty:
            continue
        for params in grids:
            base_cfg = _default_base_config(args.csv, args.strategy, params)
            metrics, trades_df, eq_df = run_fold(dfq, base_cfg, args.strategy, params)
            results_rows.append({
                "fold": f"{args.year}Q{q}",
                "strategy": args.strategy,
                "params": params,
                "variant": "baseline",
                "variant_detail": "",
                "win_rate": metrics.win_rate,
                "expectancy": metrics.expectancy,
                "avg_R": metrics.avg_R,
                "max_dd": metrics.max_dd,
                "sharpe_like_R": metrics.sharpe_like_R,
                "trades": metrics.trades,
            })
            if trades_df is not None and not trades_df.empty:
                t = trades_df.copy()
                t["fold"] = f"{args.year}Q{q}"
                t["strategy"] = args.strategy
                t["params"] = str(params)
                t["variant"] = "baseline"
                trades_all.append(t)

            # Skip ablations/robustness in quick mode
            if args.quick:
                continue

            # Ablations
            ablations = []
            if args.strategy == "cb_vp":
                ablations += ["no_compression", "lower_body_ratio"]
            if args.strategy == "mfvg_s":
                ablations += ["disable_mfvg_validation"]
            ablations += ["time_expand"]
            for abl in ablations:
                abl_cfg = _default_base_config(args.csv, args.strategy, params)
                metrics, trades_df, eq_df = run_fold(dfq, abl_cfg, args.strategy, params, ablation=abl)
                results_rows.append({
                    "fold": f"{args.year}Q{q}",
                    "strategy": args.strategy,
                    "params": params,
                    "variant": "ablation",
                    "variant_detail": abl,
                    "win_rate": metrics.win_rate,
                    "expectancy": metrics.expectancy,
                    "avg_R": metrics.avg_R,
                    "max_dd": metrics.max_dd,
                    "sharpe_like_R": metrics.sharpe_like_R,
                    "trades": metrics.trades,
                })
                if trades_df is not None and not trades_df.empty:
                    t = trades_df.copy()
                    t["fold"] = f"{args.year}Q{q}"
                    t["strategy"] = args.strategy
                    t["params"] = str(params)
                    t["variant"] = f"ablation:{abl}"
                    trades_all.append(t)

            # Robustness: time shift ±5 and cost bump +0.3,+0.5; param perturbation ±15%
            for shift in [-5, 5]:
                metrics, trades_df, eq_df = run_fold(dfq, base_cfg, args.strategy, params, time_shift_min=shift)
                results_rows.append({
                    "fold": f"{args.year}Q{q}",
                    "strategy": args.strategy,
                    "params": params,
                    "variant": "robust",
                    "variant_detail": f"shift{shift}",
                    "win_rate": metrics.win_rate,
                    "expectancy": metrics.expectancy,
                    "avg_R": metrics.avg_R,
                    "max_dd": metrics.max_dd,
                    "sharpe_like_R": metrics.sharpe_like_R,
                    "trades": metrics.trades,
                })
            for bump in [0.3, 0.5]:
                metrics, trades_df, eq_df = run_fold(dfq, base_cfg, args.strategy, params, cost_bump_pips=bump)
                results_rows.append({
                    "fold": f"{args.year}Q{q}",
                    "strategy": args.strategy,
                    "params": params,
                    "variant": "robust",
                    "variant_detail": f"spread+{bump}",
                    "win_rate": metrics.win_rate,
                    "expectancy": metrics.expectancy,
                    "avg_R": metrics.avg_R,
                    "max_dd": metrics.max_dd,
                    "sharpe_like_R": metrics.sharpe_like_R,
                    "trades": metrics.trades,
                })
            # Param perturbation: scale numeric params by ±15%
            def _perturb(p: Dict[str, Any], scale: float) -> Dict[str, Any]:
                q = {}
                for k, v in p.items():
                    if isinstance(v, (int, float)):
                        q[k] = float(v) * (1.0 + scale)
                    else:
                        q[k] = v
                return q
            for sc in [-0.15, 0.15]:
                p2 = _perturb(params, sc)
                cfg2 = _default_base_config(args.csv, args.strategy, p2)
                metrics, trades_df, eq_df = run_fold(dfq, cfg2, args.strategy, p2)
                results_rows.append({
                    "fold": f"{args.year}Q{q}",
                    "strategy": args.strategy,
                    "params": p2,
                    "variant": "robust",
                    "variant_detail": f"param{sc:+.0%}",
                    "win_rate": metrics.win_rate,
                    "expectancy": metrics.expectancy,
                    "avg_R": metrics.avg_R,
                    "max_dd": metrics.max_dd,
                    "sharpe_like_R": metrics.sharpe_like_R,
                    "trades": metrics.trades,
                })
    res = pd.DataFrame(results_rows)
    outp = os.path.join(os.path.dirname(os.path.abspath(args.csv)), f"research_results_{args.strategy}.csv")
    if not res.empty:
        res.to_csv(outp, index=False)
        print(f"Saved results → {outp}")
        # Optional: save blotter if assembled
        if trades_all:
            blotter = pd.concat(trades_all, ignore_index=True)
            blot_out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), f"research_blotter_{args.strategy}.csv")
            blotter.to_csv(blot_out, index=False)
            print(f"Saved blotter → {blot_out}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
