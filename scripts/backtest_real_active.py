"""
Real-Strategy Active Backtest
=============================
Unlike scripts/backtest_dual.py (which simulates a *simplified proxy*), this
harness replays the ACTUAL ActiveStrategy.generate_signal() over historical bars
via a point-in-time market adapter — so it validates the logic that really trades.

How it stays honest:
  - PointInTimeMarket only ever returns bars with timestamp <= the simulated "now",
    so the strategy cannot see the future (no lookahead).
  - The wall-clock session gate (_session_quality) is patched to read the simulated
    bar's UTC hour instead of the real clock.
  - Funding is stubbed neutral (0.0) so funding_arb stays out of the way.
  - Realistic costs (taker fee + slippage + funding) reuse backtest_dual's model.

Usage:
    python scripts/backtest_real_active.py
    python scripts/backtest_real_active.py --symbols SOL/USDT:USDT BNB/USDT:USDT --mode mean_reversion
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

import strategy.active_strategy as AS
from data.market_data import MarketData
from config.dual_settings import (
    ACTIVE_CAPITAL, ACTIVE_MAX_POSITION_SIZE, ACTIVE_MAX_POSITIONS,
)
from scripts.backtest_dual import (
    TAKER_FEE, SLIPPAGE, FUNDING_8H, _daily_metrics, _max_drawdown,
)

MAX_HOLD_BARS = 96   # 96×15m = 24h safety cap (active is intraday)


# ── point-in-time market adapter ────────────────────────────────────────────────

class PointInTimeMarket:
    """Serves only bars at or before `now_ms` — no lookahead."""

    def __init__(self, data: dict):
        self._data = data          # {(symbol, timeframe): full DataFrame}
        self.now_ms = 0

    def _slice(self, symbol: str, timeframe: str, limit: int):
        df = self._data.get((symbol, timeframe))
        if df is None:
            return None
        sub = df[df["timestamp"] <= self.now_ms]
        return sub.tail(limit).reset_index(drop=True)

    def get_ohlcv_df(self, symbol: str, timeframe: str = "15m", limit: int = 120):
        sub = self._slice(symbol, timeframe, limit)
        if sub is None or len(sub) == 0:
            raise ValueError(f"no data for {symbol} {timeframe}")
        return sub

    def get_price(self, symbol: str) -> float:
        sub = self._slice(symbol, "15m", 1)
        if sub is None or len(sub) == 0:
            raise ValueError(f"no price for {symbol}")
        return float(sub["close"].iloc[-1])

    def get_atr(self, symbol: str, timeframe: str = "15m", period: int = 14) -> float:
        sub = self._slice(symbol, timeframe, period + 10)
        if sub is None or len(sub) < 2:
            return 0.0
        h, l, c = sub["high"].values, sub["low"].values, sub["close"].values
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
               for i in range(1, len(c))]
        return float(np.mean(trs[-period:])) if trs else 0.0


class _StubFunding:
    """Neutral funding so funding signals stay out of the way during replay."""
    def get_funding(self, symbol: str) -> dict:
        return {"funding_rate": 0.0, "symbol": symbol}


# ── simulated clock for the session-quality gate ────────────────────────────────

_SIM = {"utc_hour": 12}   # London/NY overlap default (neutral)


def _patched_session_quality() -> float:
    return AS._SESSION_QUALITY_MAP.get(_SIM["utc_hour"], 0.60)


# ── backtest ────────────────────────────────────────────────────────────────────

_TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000,
          "1d": 86_400_000, "1w": 604_800_000}


def _fetch_paginated(ex, symbol: str, timeframe: str, total_bars: int) -> "pd.DataFrame":
    """Walk `since` forward in 1000-bar batches to gather multi-month history."""
    import time as _time
    from data.indicators import ohlcv_to_df
    tf_ms = _TF_MS[timeframe]
    since = ex.milliseconds() - total_bars * tf_ms
    rows: list = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000 or len(rows) >= total_bars:
            break
        _time.sleep(ex.rateLimit / 1000)
    # Dedupe by timestamp, keep order
    seen, uniq = set(), []
    for c in rows:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    return ohlcv_to_df(uniq)


def fetch_data(symbols: list[str], months: float = 3.0) -> dict:
    """Paginated multi-month OHLCV for 15m/1h/4h, returned as {(symbol, tf): df}."""
    from data.binance_client import BinanceClient
    ex = BinanceClient().get_exchange()
    bars_15m = int(months * 30 * 96)
    data: dict = {}
    for sym in symbols:
        for tf, n in (("15m", bars_15m), ("1h", bars_15m // 4), ("4h", bars_15m // 16)):
            try:
                df = _fetch_paginated(ex, sym, tf, n)
                data[(sym, tf)] = df.reset_index(drop=True)
                print(f"  {sym} {tf}: {len(df)} bars")
            except Exception as e:
                print(f"  fetch error {sym} {tf}: {e}")
    return data


def replay(data: dict, symbols: list[str], mode: str,
           start_ms: int | None = None, end_ms: int | None = None,
           warmup: int = 80, min_conf: float = 0.0, use_tm: bool = True,
           cooldown: int = 0) -> dict:
    """
    Replay the REAL ActiveStrategy over [start_ms, end_ms] (default: full range).
    The adapter still serves pre-window history for warmup, so no window starts blind.
    Any position still open at end is marked-to-market closed for window isolation.
    """
    base = data.get((symbols[0], "15m"))
    if base is None or len(base) < 120:
        return {}
    ts_axis = base["timestamp"].values

    adapter = PointInTimeMarket(data)
    AS._session_quality = _patched_session_quality      # patch wall-clock gate
    strat = AS.ActiveStrategy()
    strat.market = adapter
    strat.funding = _StubFunding()
    # Simulate live exits (trailing/breakeven/partial-TP) via TradeManager on 15m ATR.
    from core.trade_manager import TradeManager
    tm = TradeManager()
    tm.market = adapter

    cash = initial = float(ACTIVE_CAPITAL)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_ts: list[int] = []
    market_state = {"regime": "sideways", "volatility": 0.5}
    last_close: dict[str, int] = {}   # sym → bar index of last close (re-entry cooldown)

    # 15m row lookup per symbol for O(1) bar access
    rows_by_ts = {
        sym: {int(r.timestamp): r for r in data[(sym, "15m")].itertuples()}
        for sym in symbols if (sym, "15m") in data
    }

    for i in range(warmup, len(ts_axis)):
        now_ms = int(ts_axis[i])
        if start_ms is not None and now_ms < start_ms:
            continue
        if end_ms is not None and now_ms > end_ms:
            break
        adapter.now_ms = now_ms
        _SIM["utc_hour"] = (now_ms // 3_600_000) % 24

        # 1. Manage open positions — TradeManager simulates trailing/breakeven/partial-TP
        for sym in list(positions.keys()):
            r = rows_by_ts.get(sym, {}).get(now_ms)
            if r is None:
                continue
            high, low, close = float(r.high), float(r.low), float(r.close)
            pos = positions[sym]
            pos["current_price"] = close

            # Exit is checked against the stop as it stood BEFORE this bar's trail
            # update — trailing now then testing this same bar's low/high would be
            # intrabar lookahead (the trailed level didn't exist when the low printed).
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
            if exit_price is None and (i - pos["open_bar"]) >= MAX_HOLD_BARS:
                exit_price, reason = close, "MAX_HOLD"
            if exit_price is not None:
                cash += _close(positions, trades, sym, exit_price, reason, i)
                last_close[sym] = i
                if use_tm:
                    tm.release(sym)

        # 2. Open new positions via the REAL strategy
        for sym in symbols:
            if len(positions) >= ACTIVE_MAX_POSITIONS:
                break
            if sym in positions:
                continue
            if i - last_close.get(sym, -10**9) < cooldown:
                continue   # re-entry cooldown — suppress churn after a close
            try:
                sig = strat.generate_signal(sym, market_state, mode=mode)
            except Exception:
                continue
            if sig.get("side", "HOLD") == "HOLD":
                continue
            if float(sig.get("confidence", 0.0)) < min_conf:
                continue   # conviction filter — fewer, higher-quality trades
            price = float(sig["entry_price"])
            value = min(ACTIVE_MAX_POSITION_SIZE, ACTIVE_CAPITAL / ACTIVE_MAX_POSITIONS, cash)
            if value <= 0:
                continue
            cash -= value
            positions[sym] = {
                "symbol": sym, "strategy": "active",
                "side": sig["side"], "entry": price, "entry_price": price,
                "size": value / price, "position_value": value,
                "tp": float(sig["take_profit"]), "sl": float(sig["stop_loss"]),
                "take_profit": float(sig["take_profit"]),
                "stop_loss": float(sig["stop_loss"]), "open_bar": i,
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

    # Force-close leftovers at the last seen price (window isolation)
    last_ms = equity_ts[-1] if equity_ts else (end_ms or 0)
    for sym in list(positions.keys()):
        r = rows_by_ts.get(sym, {}).get(last_ms)
        px = float(r.close) if r is not None else positions[sym]["entry"]
        _close(positions, trades, sym, px, "WINDOW_END", len(ts_axis))

    return _stats(mode, trades, equity_curve, equity_ts, initial)


def run(symbols: list[str], mode: str, months: float = 3.0) -> dict:
    print(f"[RealActive] Fetching ~{months}mo history for {symbols} …")
    data = fetch_data(symbols, months)
    if not data:
        print("  No data — aborting")
        return {}
    return replay(data, symbols, mode)


def _close_half(pos, trades, sym, price, bar_idx) -> float:
    """Realise half at the partial-profit lock, keep the rest running."""
    half_val  = pos["position_value"] / 2
    half_size = pos["size"] / 2
    gross = ((price - pos["entry"]) * half_size if pos["side"] == "LONG"
             else (pos["entry"] - price) * half_size)
    hold_hours = max(0, (bar_idx - pos["open_bar"]) * 15 // 60)
    cost = half_val * (TAKER_FEE + SLIPPAGE) * 2 + half_val * FUNDING_8H * (hold_hours // 8)
    pnl = gross - cost
    pos["position_value"] -= half_val
    pos["size"] -= half_size
    pos["_halved"] = True
    trades.append({"symbol": sym, "side": pos["side"], "pnl": round(pnl, 4), "reason": "PARTIAL_TP"})
    return half_val + pnl


def _close(positions, trades, sym, price, reason, bar_idx) -> float:
    pos = positions.pop(sym)
    gross = ((price - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
             else (pos["entry"] - price) * pos["size"])
    hold_hours = max(0, (bar_idx - pos["open_bar"]) * 15 // 60)
    cost = pos["position_value"] * (TAKER_FEE + SLIPPAGE) * 2
    cost += pos["position_value"] * FUNDING_8H * (hold_hours // 8)
    pnl = gross - cost
    trades.append({"symbol": sym, "side": pos["side"], "pnl": round(pnl, 4), "reason": reason})
    return pos["position_value"] + pnl


def _stats(mode, trades, equity_curve, equity_ts, initial) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)
    final_eq = equity_curve[-1] if equity_curve else initial
    dm = _daily_metrics(equity_ts, equity_curve)
    return {
        "mode": mode, "trades": n, "wins": wins,
        "winrate": round(wins / n, 4) if n else 0,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final_eq - initial) / initial * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(equity_curve, initial) * 100, 2),
        "final_equity": round(final_eq, 2),
        "days": dm["days"], "profitable_days_pct": dm["profitable_days_pct"],
        "avg_daily_pnl": dm["avg_daily_pnl"], "worst_day_pnl": dm["worst_day_pnl"],
        "best_day_pnl": dm["best_day_pnl"], "daily_sharpe": dm["daily_sharpe"],
    }


def main():
    p = argparse.ArgumentParser(description="Backtest the REAL ActiveStrategy")
    p.add_argument("--symbols", nargs="+", default=["SOL/USDT:USDT", "BNB/USDT:USDT"])
    p.add_argument("--mode", default="mean_reversion",
                   choices=["mean_reversion", "momentum", "vwap_reversal", "breakout", "funding_arb"])
    p.add_argument("--months", type=float, default=3.0)
    args = p.parse_args()

    stats = run(args.symbols, args.mode, months=args.months)
    if not stats:
        return
    print(f"\n{'=' * 52}")
    print(f"  REAL ActiveStrategy — mode={args.mode}")
    print(f"{'=' * 52}")
    labels = [
        ("trades", "Trades"), ("wins", "Wins"), ("winrate", "Win rate"),
        ("total_pnl", "Total P&L"), ("return_pct", "Return %"),
        ("max_drawdown_pct", "Max drawdown %"), ("final_equity", "Final equity"),
        ("days", "Days measured"), ("profitable_days_pct", "Profitable days %"),
        ("avg_daily_pnl", "Avg daily P&L"), ("worst_day_pnl", "Worst day P&L"),
        ("best_day_pnl", "Best day P&L"), ("daily_sharpe", "Daily Sharpe"),
    ]
    for k, lbl in labels:
        v = stats[k]
        if k == "winrate":
            print(f"  {lbl:<22} {v:.1%}")
        elif k == "profitable_days_pct":
            print(f"  {lbl:<22} {v:.1f}%")
        else:
            print(f"  {lbl:<22} {v}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
