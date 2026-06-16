"""
Donchian Breakout Trend-Following Backtest (a DIFFERENT edge family)
====================================================================
The active TA-indicator signals (RSI/BB/MACD) tested as ≈0 edge, and passive's
EMA-pullback entry gets whipsawed. This tests the classic trend-following edge
that historically DOES work in crypto: Donchian channel breakouts (Turtle-style).

  Entry : close breaks above the N-day high (LONG) / below N-day low (SHORT)
  Exit  : opposite M-day channel (trailing) OR a hard ATR stop
  Costs : taker fee + slippage (breakouts cross the spread) + funding per day held

Walk-forward across the full period, per symbol and basket. Honest test with the
same realistic cost model as the rest of the validation stack.

Usage:
    python scripts/backtest_donchian.py --months 18 --entry 20 --exit 10
    python scripts/backtest_donchian.py --months 18 --sweep
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
from scripts.backtest_dual import TAKER_FEE, SLIPPAGE, FUNDING_8H, _max_drawdown

DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
CAPITAL = 10000.0
RISK_FRACTION = 0.20   # notional per position = 20% of capital (3-5 concurrent)


def _atr(highs, lows, closes, period=14):
    trs = [max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
           for j in range(1, len(closes))]
    return float(np.mean(trs[-period:])) if len(trs) >= period else 0.0


def replay(data: dict, symbols: list[str], entry_len: int, exit_len: int,
           atr_stop: float, start_ms=None, end_ms=None, allow_short=True) -> dict:
    cash = initial = CAPITAL
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve, equity_ts = [], []

    series = {sym: data[(sym, "1d")] for sym in symbols if (sym, "1d") in data}
    if not series:
        return {}
    base = series[symbols[0]]
    ts_axis = base["timestamp"].values.astype("int64")

    arr = {sym: {"ts": df["timestamp"].values.astype("int64"),
                 "h": df["high"].values.astype(float),
                 "l": df["low"].values.astype(float),
                 "c": df["close"].values.astype(float)}
           for sym, df in series.items()}
    idx_of = {sym: {int(t): k for k, t in enumerate(arr[sym]["ts"])} for sym in series}

    warmup = max(entry_len, 30) + 2
    for i in range(warmup, len(ts_axis)):
        now_ms = int(ts_axis[i])
        if start_ms is not None and now_ms < start_ms:
            continue
        if end_ms is not None and now_ms > end_ms:
            break

        for sym in series:
            a = arr[sym]; k = idx_of[sym].get(now_ms)
            if k is None or k < warmup:
                continue
            high, low, close = a["h"][k], a["l"][k], a["c"][k]

            # ── manage open position (trailing channel + ATR stop) ──
            if sym in positions:
                pos = positions[sym]
                exit_price = None
                if pos["side"] == "LONG":
                    chan_exit = np.min(a["l"][k - exit_len:k])      # M-day low
                    stop = max(pos["stop"], chan_exit)
                    pos["stop"] = stop
                    if low <= stop:
                        exit_price = stop
                else:
                    chan_exit = np.max(a["h"][k - exit_len:k])      # M-day high
                    stop = min(pos["stop"], chan_exit)
                    pos["stop"] = stop
                    if high >= stop:
                        exit_price = stop
                if exit_price is not None:
                    cash += _close(positions, trades, sym, exit_price, i)

            # ── entry on Donchian breakout ──
            if sym not in positions and len(positions) < 5:
                up = np.max(a["h"][k - entry_len:k])     # prior N-day high (excludes today)
                dn = np.min(a["l"][k - entry_len:k])
                side = None
                if close > up:
                    side = "LONG"
                elif allow_short and close < dn:
                    side = "SHORT"
                if side:
                    atrv = _atr(a["h"][:k + 1], a["l"][:k + 1], a["c"][:k + 1])
                    if atrv > 0:
                        value = min(CAPITAL * RISK_FRACTION, cash)
                        if value > 0:
                            cash -= value
                            stop = (close - atr_stop * atrv if side == "LONG"
                                    else close + atr_stop * atrv)
                            positions[sym] = {
                                "side": side, "entry": close, "size": value / close,
                                "position_value": value, "stop": stop, "open_bar": i,
                            }

        # equity snapshot
        unreal = 0.0
        for sym, pos in positions.items():
            k = idx_of[sym].get(now_ms)
            if k is None:
                continue
            p = arr[sym]["c"][k]
            unreal += ((p - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
                       else (pos["entry"] - p) * pos["size"])
        invested = sum(p["position_value"] for p in positions.values())
        equity_curve.append(cash + invested + unreal)
        equity_ts.append(now_ms)

    last = equity_ts[-1] if equity_ts else (end_ms or 0)
    for sym in list(positions.keys()):
        k = idx_of[sym].get(last)
        px = arr[sym]["c"][k] if k is not None else positions[sym]["entry"]
        _close(positions, trades, sym, px, len(ts_axis))

    return _stats(trades, equity_curve, initial)


def _close(positions, trades, sym, price, bar_idx) -> float:
    pos = positions.pop(sym)
    gross = ((price - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
             else (pos["entry"] - price) * pos["size"])
    hold_days = max(0, bar_idx - pos["open_bar"])
    cost = pos["position_value"] * (TAKER_FEE + SLIPPAGE) * 2 + pos["position_value"] * FUNDING_8H * 3 * hold_days
    pnl = gross - cost
    trades.append({"pnl": round(pnl, 4), "hold_days": hold_days})
    return pos["position_value"] + pnl


def _stats(trades, equity_curve, initial) -> dict:
    n = len(trades); wins = sum(1 for t in trades if t["pnl"] > 0)
    final = equity_curve[-1] if equity_curve else initial
    return {
        "trades": n, "winrate": round(wins / n, 4) if n else 0,
        "return_pct": round((final - initial) / initial * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(equity_curve, initial) * 100, 2),
        "avg_hold_days": round(sum(t["hold_days"] for t in trades) / n, 1) if n else 0,
    }


def fetch(symbols, months):
    from data.binance_client import BinanceClient
    ex = BinanceClient().get_exchange()
    data = {}
    for sym in symbols:
        try:
            df = _fetch_paginated(ex, sym, "1d", int(months * 30) + 60)
            data[(sym, "1d")] = df.reset_index(drop=True)
            print(f"  {sym} 1d: {len(df)} bars")
        except Exception as e:
            print(f"  fetch error {sym}: {e}")
    return data


def _fmt(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def main():
    p = argparse.ArgumentParser(description="Donchian breakout trend-following backtest")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--months", type=float, default=18.0)
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--atr-stop", type=float, default=2.5)
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print(f"[Donchian] Fetching ~{args.months}mo for {args.symbols} …")
    data = fetch(args.symbols, args.months)
    base = data.get((args.symbols[0], "1d"))
    if base is None:
        print("  no data"); return
    ts = base["timestamp"].values.astype("int64")

    if args.sweep:
        print("\nentry/exit sweep (full period, all symbols pooled):")
        print(f"{'entry':>6} {'exit':>5} {'ret%':>8} {'win%':>6} {'maxDD%':>7} {'trades':>7}")
        for e in (10, 20, 30, 55):
            for x in (5, 10, 20):
                if x >= e: continue
                st = replay(data, args.symbols, e, x, args.atr_stop)
                print(f"{e:>6} {x:>5} {st['return_pct']:>8.2f} {st['winrate']*100:>6.1f} "
                      f"{st['max_drawdown_pct']:>7.2f} {st['trades']:>7}")
        return

    win = args.window_days
    starts = list(range(max(args.entry, 30) + 2, len(ts) - win, win))
    print(f"\nDonchian({args.entry}/{args.exit}) walk-forward: {len(starts)} × {win}d\n")
    hdr = f"{'window':<16} {'ret%':>8} {'win%':>6} {'maxDD%':>7} {'trades':>7}"
    print(hdr); print("-" * len(hdr))
    rets = []
    for s in starts:
        st = replay(data, args.symbols, args.entry, args.exit, args.atr_stop,
                    int(ts[s]), int(ts[min(s + win, len(ts) - 1)]))
        if not st: continue
        rets.append(st["return_pct"])
        print(f"{_fmt(int(ts[s]))}/{_fmt(int(ts[min(s+win,len(ts)-1)])):<7} "
              f"{st['return_pct']:>8.2f} {st['winrate']*100:>6.1f} {st['max_drawdown_pct']:>7.2f} {st['trades']:>7}")
    if rets:
        pos = sum(1 for r in rets if r > 0)
        print(f"\n  TOTAL ret={sum(rets):+.2f}%  avg/win={np.mean(rets):+.2f}%  "
              f"profitable_windows={pos}/{len(rets)}")
    print()


if __name__ == "__main__":
    main()
