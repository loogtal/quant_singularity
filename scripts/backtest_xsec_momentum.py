"""
Cross-Sectional Momentum Backtest (a market-neutral edge family)
================================================================
Time-series TA (≈0 edge) and single-asset trend (regime-dependent) both tested
weak. This tests cross-sectional momentum — a documented crypto edge that is
naturally market-NEUTRAL (long the strongest coins, short the weakest), which
directly serves the "minimise loss / survive years" goal by removing most beta.

  Each rebalance (every REBAL days): rank the basket by lookback return
  (skip the most recent SKIP days to avoid short-term reversal). Go LONG the top
  K, SHORT the bottom K, equal-weight and dollar-neutral. Hold to next rebalance.
  Costs: turnover × (taker + slippage) per leg per rebalance + funding.

Reports the market-neutral equity curve, walk-forward by quarter, and the spread
(long-leg vs short-leg) so you can see if the ranking has predictive power at all.

Usage:
    python scripts/backtest_xsec_momentum.py --months 18 --lookback 30 --k 3
    python scripts/backtest_xsec_momentum.py --months 18 --sweep
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from scripts.backtest_real_active import _fetch_paginated
from scripts.backtest_dual import TAKER_FEE, SLIPPAGE, FUNDING_8H

BASKET = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "SUI/USDT:USDT", "LTC/USDT:USDT", "DOT/USDT:USDT",
    "TRX/USDT:USDT", "BCH/USDT:USDT", "NEAR/USDT:USDT",
]


def fetch(symbols, months):
    from data.binance_client import BinanceClient
    ex = BinanceClient().get_exchange()
    data = {}
    for sym in symbols:
        try:
            df = _fetch_paginated(ex, sym, "1d", int(months * 30) + 60)
            if len(df) >= 60:
                data[sym] = df.reset_index(drop=True)
        except Exception as e:
            print(f"  fetch error {sym}: {e}")
    return data


def _align(data: dict):
    """Build a common timestamp axis and a close matrix {sym: aligned closes}."""
    common = None
    for sym, df in data.items():
        ts = set(int(t) for t in df["timestamp"].values)
        common = ts if common is None else (common & ts)
    axis = sorted(common)
    closes = {}
    for sym, df in data.items():
        m = {int(t): float(c) for t, c in zip(df["timestamp"].values, df["close"].values)}
        closes[sym] = np.array([m[t] for t in axis], dtype=float)
    return np.array(axis, dtype="int64"), closes


def run(data: dict, lookback: int, skip: int, rebal: int, k: int,
        start_ms=None, end_ms=None) -> dict:
    axis, closes = _align(data)
    syms = list(closes.keys())
    if len(axis) < lookback + rebal + 5 or len(syms) < 2 * k + 1:
        return {}
    cost_leg = (TAKER_FEE + SLIPPAGE)        # one side; rebalancing both legs costs more
    daily_fund = FUNDING_8H * 3
    eq = 1.0
    curve, curve_ts = [], []
    spread_hist = []

    i = lookback + skip
    while i < len(axis) - 1:
        now_ms = int(axis[i])
        if (start_ms and now_ms < start_ms) or (end_ms and now_ms > end_ms):
            i += rebal
            continue
        # momentum = return from i-lookback-skip .. i-skip (skip recent reversal window)
        mom = {}
        for s in syms:
            p_old = closes[s][i - lookback - skip]
            p_now = closes[s][i - skip]
            if p_old > 0:
                mom[s] = p_now / p_old - 1.0
        if len(mom) < 2 * k + 1:
            i += rebal
            continue
        ranked = sorted(mom, key=mom.get, reverse=True)
        longs, shorts = ranked[:k], ranked[-k:]

        # hold period return: i .. i+rebal (close to close)
        j = min(i + rebal, len(axis) - 1)
        long_ret = np.mean([closes[s][j] / closes[s][i] - 1.0 for s in longs])
        short_ret = np.mean([closes[s][j] / closes[s][i] - 1.0 for s in shorts])
        gross = (long_ret - short_ret) / 2.0          # dollar-neutral: half capital each leg
        hold_days = (int(axis[j]) - now_ms) / 86_400_000
        cost = cost_leg * 2 * 2 + daily_fund * max(0, hold_days)  # 2 legs × enter+exit
        net = gross - cost
        eq *= (1.0 + net)
        curve.append(eq); curve_ts.append(now_ms)
        spread_hist.append(long_ret - short_ret)
        i += rebal

    if not curve:
        return {}
    rets = [curve[0] - 1.0] + [curve[m] / curve[m - 1] - 1.0 for m in range(1, len(curve))]
    arr = np.array(rets)
    periods_per_year = 365 / rebal
    sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(periods_per_year))
    peak, mdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
    return {
        "rebalances": len(curve),
        "total_ret_pct": round((eq - 1.0) * 100, 2),
        "ann_ret_pct": round(((eq ** (periods_per_year / len(curve))) - 1.0) * 100, 2),
        "win_rebal_pct": round(float(np.mean(arr > 0)) * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(mdd * 100, 2),
        "avg_spread_pct": round(float(np.mean(spread_hist)) * 100, 2),
    }


def _fmt(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def main():
    p = argparse.ArgumentParser(description="Cross-sectional momentum backtest")
    p.add_argument("--symbols", nargs="+", default=BASKET)
    p.add_argument("--months", type=float, default=18.0)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--rebal", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print(f"[XSecMom] Fetching ~{args.months}mo for {len(args.symbols)} symbols …")
    data = fetch(args.symbols, args.months)
    print(f"  usable symbols: {len(data)}")
    if len(data) < 7:
        print("  too few symbols"); return

    if args.sweep:
        print(f"\n{'lookback':>8} {'rebal':>6} {'k':>3} {'total%':>8} {'ann%':>7} "
              f"{'win%':>6} {'sharpe':>7} {'maxDD%':>7} {'spread%':>8}")
        for lb in (14, 30, 60, 90):
            for rb in (7, 14):
                for k in (3, 4):
                    st = run(data, lb, args.skip, rb, k)
                    if st:
                        print(f"{lb:>8} {rb:>6} {k:>3} {st['total_ret_pct']:>8.2f} "
                              f"{st['ann_ret_pct']:>7.2f} {st['win_rebal_pct']:>6.1f} "
                              f"{st['sharpe']:>7.2f} {st['max_dd_pct']:>7.2f} {st['avg_spread_pct']:>8.2f}")
        return

    st = run(data, args.lookback, args.skip, args.rebal, args.k)
    print(f"\n{'=' * 56}\n  CROSS-SECTIONAL MOMENTUM "
          f"(lb={args.lookback} rebal={args.rebal} k={args.k})\n{'=' * 56}")
    for kk, vv in st.items():
        print(f"  {kk:<18} {vv}")
    print("=" * 56)

    # walk-forward
    axis, _ = _align(data)
    win = args.window_days
    starts = list(range(args.lookback + args.skip, len(axis) - win, win))
    print(f"\n  walk-forward ({win}d windows):")
    rets = []
    for s in starts:
        st = run(data, args.lookback, args.skip, args.rebal, args.k,
                 int(axis[s]), int(axis[min(s + win, len(axis) - 1)]))
        if st:
            rets.append(st["total_ret_pct"])
            print(f"    {_fmt(int(axis[s]))}  ret={st['total_ret_pct']:+6.2f}%  "
                  f"win={st['win_rebal_pct']:.0f}%  maxDD={st['max_dd_pct']:.1f}%")
    if rets:
        print(f"\n  TOTAL={sum(rets):+.2f}%  profitable_windows={sum(1 for r in rets if r>0)}/{len(rets)}")
    print()


if __name__ == "__main__":
    main()
