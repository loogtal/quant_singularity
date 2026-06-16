"""
Real-Strategy Passive Backtest (walk-forward capable)
=====================================================
Replays the ACTUAL PassiveStrategy.generate_signal over historical DAILY bars via
the same point-in-time adapter used for active. Passive has no wall-clock session
gate, so no clock patching is needed — only the market data must be time-correct.

Validates the long-term "อยู่ยาว" wealth engine, which had never been tested with
the real signal logic (backtest_dual.py only tests a simplified EMA50/200 proxy).

Usage:
    python scripts/backtest_real_passive.py --months 15
    python scripts/backtest_real_passive.py --months 15 --window-days 90
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

import strategy.passive_strategy as PS
from core.trade_manager import TradeManager
from scripts.backtest_real_active import PointInTimeMarket, _fetch_paginated
from scripts.backtest_dual import TAKER_FEE, SLIPPAGE, FUNDING_8H, _daily_metrics, _max_drawdown
from meta.regime_classifier import RegimeClassifier
from config.dual_settings import (
    PASSIVE_CAPITAL, PASSIVE_MAX_POSITION_SIZE, PASSIVE_MAX_POSITIONS,
)

MAX_HOLD_DAYS = 14   # matches live passive max-hold


def fetch_data(symbols: list[str], months: float) -> dict:
    from data.binance_client import BinanceClient
    ex = BinanceClient().get_exchange()
    days = int(months * 30)
    data: dict = {}
    for sym in symbols:
        for tf, n in (("1d", days + 60), ("4h", (days + 60) * 6), ("1w", days // 7 + 20)):
            try:
                df = _fetch_paginated(ex, sym, tf, n)
                data[(sym, tf)] = df.reset_index(drop=True)
                print(f"  {sym} {tf}: {len(df)} bars")
            except Exception as e:
                print(f"  fetch error {sym} {tf}: {e}")
    return data


def replay(data: dict, symbols: list[str],
           start_ms: int | None = None, end_ms: int | None = None,
           use_tm: bool = True) -> dict:
    base = data.get((symbols[0], "1d"))
    if base is None or len(base) < 120:
        return {}
    ts_axis = base["timestamp"].values.astype("int64")

    adapter = PointInTimeMarket(data)
    strat = PS.PassiveStrategy()
    strat.market = adapter
    # TradeManager simulates the live exits (trailing stop, breakeven-lock after
    # +activation profit, partial-TP at TP1 → extend to TP2). It reads ATR via the
    # same point-in-time adapter, so its stop logic is time-correct (no lookahead).
    tm = TradeManager()
    tm.market = adapter

    cash = initial = float(PASSIVE_CAPITAL)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_ts: list[int] = []

    rows_by_ts = {
        sym: {int(r.timestamp): r for r in data[(sym, "1d")].itertuples()}
        for sym in symbols if (sym, "1d") in data
    }
    rc = RegimeClassifier()
    warmup = 100   # need 1d history for EMA/ADX/regime

    for i in range(warmup, len(ts_axis)):
        now_ms = int(ts_axis[i])
        if start_ms is not None and now_ms < start_ms:
            continue
        if end_ms is not None and now_ms > end_ms:
            break
        adapter.now_ms = now_ms

        # regime from 1d closes up to now
        closes_now = base[base["timestamp"] <= now_ms]["close"].values
        regime = rc.classify(np.asarray(closes_now[-250:], dtype=float)) if len(closes_now) >= 100 else "sideways"
        market_state = {"regime": regime}

        # 1. Manage positions — TradeManager simulates trailing/breakeven/partial-TP
        for sym in list(positions.keys()):
            r = rows_by_ts.get(sym, {}).get(now_ms)
            if r is None:
                continue
            high, low, close = float(r.high), float(r.low), float(r.close)
            pos = positions[sym]
            pos["current_price"] = close

            # Use the stop as of BEFORE this bar's trail update — trailing now then
            # testing this same bar's low/high would be intrabar lookahead.
            prev_sl = pos["stop_loss"]
            if use_tm:
                action = tm.tick(pos)   # trails pos["stop_loss"]/["take_profit"] for NEXT bar
                if action.get("action") == "close_partial" and not pos.get("_halved"):
                    cash += _close_half(pos, trades, sym, close, i)

            sl, tp = prev_sl, pos["take_profit"]
            exit_price = reason = None
            if pos["side"] == "LONG":
                if low <= sl:    exit_price, reason = sl, "STOP"
                elif high >= tp: exit_price, reason = tp, "TP"
            else:
                if high >= sl:   exit_price, reason = sl, "STOP"
                elif low <= tp:  exit_price, reason = tp, "TP"
            if exit_price is None and (i - pos["open_bar"]) >= MAX_HOLD_DAYS:
                exit_price, reason = close, "MAX_HOLD"
            if exit_price is not None:
                cash += _close(positions, trades, sym, exit_price, reason, i)
                if use_tm:
                    tm.release(sym)

        # 2. Open via the REAL passive strategy (mode=auto → respects regime blacklist)
        for sym in symbols:
            if len(positions) >= PASSIVE_MAX_POSITIONS:
                break
            if sym in positions:
                continue
            try:
                sig = strat.generate_signal(sym, market_state, mode="auto")
            except Exception:
                continue
            if sig.get("side", "HOLD") == "HOLD":
                continue
            price = float(sig["entry_price"])
            value = min(PASSIVE_MAX_POSITION_SIZE, PASSIVE_CAPITAL / PASSIVE_MAX_POSITIONS, cash)
            if value <= 0:
                continue
            cash -= value
            positions[sym] = {
                "symbol": sym, "strategy": "passive",
                "side": sig["side"], "entry": price, "entry_price": price,
                "size": value / price, "position_value": value,
                "tp": float(sig["take_profit"]), "sl": float(sig["stop_loss"]),
                # TradeManager reads/updates these in place:
                "take_profit": float(sig["take_profit"]),
                "stop_loss": float(sig["stop_loss"]),
                "open_bar": i,
            }
            if use_tm:
                tm.register(sym, price, sig["side"])

        # 3. Snapshot equity
        unreal = 0.0
        for sym, pos in positions.items():
            r = rows_by_ts.get(sym, {}).get(now_ms)
            if r is None:
                continue
            p = float(r.close)
            unreal += ((p - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
                       else (pos["entry"] - p) * pos["size"])
        invested = sum(p["position_value"] for p in positions.values())
        equity_curve.append(cash + invested + unreal)
        equity_ts.append(now_ms)

    last_ms = equity_ts[-1] if equity_ts else (end_ms or 0)
    for sym in list(positions.keys()):
        r = rows_by_ts.get(sym, {}).get(last_ms)
        px = float(r.close) if r is not None else positions[sym]["entry"]
        _close(positions, trades, sym, px, "WINDOW_END", len(ts_axis))

    return _stats(trades, equity_curve, equity_ts, initial)


def _close_half(pos, trades, sym, price, bar_idx) -> float:
    """Realise half the position at TP1 (partial profit lock), keep the rest running."""
    half_val  = pos["position_value"] / 2
    half_size = pos["size"] / 2
    gross = ((price - pos["entry"]) * half_size if pos["side"] == "LONG"
             else (pos["entry"] - price) * half_size)
    hold_days = max(0, bar_idx - pos["open_bar"])
    cost = half_val * (TAKER_FEE + SLIPPAGE) * 2 + half_val * FUNDING_8H * 3 * hold_days
    pnl = gross - cost
    pos["position_value"] -= half_val
    pos["size"] -= half_size
    pos["_halved"] = True
    trades.append({"symbol": sym, "side": pos["side"], "pnl": round(pnl, 4),
                   "reason": "TP1_PARTIAL", "hold_days": hold_days})
    return half_val + pnl


def _close(positions, trades, sym, price, reason, bar_idx) -> float:
    pos = positions.pop(sym)
    gross = ((price - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
             else (pos["entry"] - price) * pos["size"])
    hold_days = max(0, bar_idx - pos["open_bar"])
    cost = pos["position_value"] * (TAKER_FEE + SLIPPAGE) * 2
    cost += pos["position_value"] * FUNDING_8H * 3 * hold_days   # 3 funding windows/day
    pnl = gross - cost
    trades.append({"symbol": sym, "side": pos["side"], "pnl": round(pnl, 4),
                   "reason": reason, "hold_days": hold_days})
    return pos["position_value"] + pnl


def _stats(trades, equity_curve, equity_ts, initial) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)
    final_eq = equity_curve[-1] if equity_curve else initial
    dm = _daily_metrics(equity_ts, equity_curve)
    hold_avg = round(sum(t["hold_days"] for t in trades) / n, 1) if n else 0
    return {
        "trades": n, "wins": wins, "winrate": round(wins / n, 4) if n else 0,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final_eq - initial) / initial * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(equity_curve, initial) * 100, 2),
        "final_equity": round(final_eq, 2), "avg_hold_days": hold_avg,
        "days": dm["days"], "profitable_days_pct": dm["profitable_days_pct"],
        "daily_sharpe": dm["daily_sharpe"],
    }


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    p = argparse.ArgumentParser(description="Backtest the REAL PassiveStrategy")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    p.add_argument("--months", type=float, default=15.0)
    p.add_argument("--window-days", type=float, default=0.0,
                   help="if >0, run walk-forward in windows of this many days")
    args = p.parse_args()

    print(f"[RealPassive] Fetching ~{args.months}mo for {args.symbols} …")
    data = fetch_data(args.symbols, args.months)
    base = data.get((args.symbols[0], "1d"))
    if base is None or len(base) < 160:
        print("  Not enough 1d data — aborting")
        return
    ts = base["timestamp"].values.astype("int64")

    if args.window_days and args.window_days > 0:
        win = int(args.window_days)
        starts = list(range(100, len(ts) - win, win))
        print(f"[RealPassive] walk-forward: {len(starts)} windows × {win} days\n")
        hdr = f"{'window':<24} {'ret%':>7} {'win%':>6} {'maxDD%':>7} {'trades':>7}"
        print(hdr); print("-" * len(hdr))
        rets = []
        for s in starts:
            st = replay(data, args.symbols, int(ts[s]), int(ts[min(s + win, len(ts) - 1)]))
            if not st:
                continue
            rets.append(st["return_pct"])
            lbl = f"{_fmt(int(ts[s]))}/{_fmt(int(ts[min(s+win,len(ts)-1)]))}"
            print(f"{lbl:<24} {st['return_pct']:>7.2f} {st['winrate']*100:>6.1f} "
                  f"{st['max_drawdown_pct']:>7.2f} {st['trades']:>7}")
        if rets:
            pos = sum(1 for r in rets if r > 0)
            print(f"\n  TOTAL ret={sum(rets):+.2f}%  avg/win={np.mean(rets):+.2f}%  "
                  f"profitable_windows={pos}/{len(rets)}")
        print()
    else:
        st = replay(data, args.symbols)
        print(f"\n{'=' * 52}\n  REAL PassiveStrategy — full period\n{'=' * 52}")
        for k, v in st.items():
            print(f"  {k:<22} {v}")
        print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
