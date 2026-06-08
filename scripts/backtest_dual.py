"""
Dual Strategy Backtest
======================
Simulates passive (1D EMA 50/200) and active (1H momentum)
on separate capital pools with conflict checks.

Usage:
    python scripts/backtest_dual.py
    python scripts/backtest_dual.py --passive BTC/USDT:USDT ETH/USDT:USDT
    python scripts/backtest_dual.py --active SOL/USDT:USDT BNB/USDT:USDT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

TAKER_FEE = 0.0004   # Binance USDT-M taker fee (0.04%) — charged on open AND close

from config.dual_settings import (
    ACTIVE_CAPITAL,
    ACTIVE_MAX_DAILY_LOSS,
    ACTIVE_MAX_POSITION_SIZE,
    ACTIVE_MAX_POSITIONS,
    ACTIVE_STOP_LOSS,
    ACTIVE_TAKE_PROFIT,
    ACTIVE_TRADING_END,
    ACTIVE_TRADING_START,
    KILL_SWITCH_EQUITY,
    PASSIVE_CAPITAL,
    PASSIVE_MAX_POSITION_SIZE,
    PASSIVE_MAX_POSITIONS,
    PASSIVE_STOP_LOSS,
    PASSIVE_TAKE_PROFIT,
)
from data.market_data import MarketData

# ── shared helpers ────────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains))  if gains.any()  else 0.0
    avg_l  = float(np.mean(losses)) if losses.any() else 1e-9
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))


def _bollinger(closes: np.ndarray, period: int = 20) -> tuple[float, float, float]:
    """Returns (upper, mid, lower)."""
    if len(closes) < period:
        mid = float(closes[-1])
        return mid, mid, mid
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid + 2.0 * std, mid, mid - 2.0 * std


def _bar_hour_utc7(ts_ms: int) -> int:
    """Return the UTC+7 hour for a millisecond timestamp."""
    return (int(ts_ms) // 3_600_000 + 7) % 24


# UTC-hour session quality map (matches active_strategy.py)
_SESSION_QUALITY: dict[int, float] = {
    0: 0.55, 1: 0.50, 2: 0.50, 3: 0.55, 4: 0.60, 5: 0.60,
    6: 0.65, 7: 0.70, 8: 0.80, 9: 0.85, 10: 0.85, 11: 0.88,
    12: 0.92, 13: 1.00, 14: 1.00, 15: 1.00, 16: 0.95, 17: 0.88,
    18: 0.80, 19: 0.75, 20: 0.70, 21: 0.68, 22: 0.65, 23: 0.58,
}
_MIN_SESSION_QUALITY = 0.70   # Gate: skip entry in low-quality hours


def _session_quality(ts_ms: int) -> float:
    hour_utc = (int(ts_ms) // 3_600_000) % 24
    return _SESSION_QUALITY.get(hour_utc, 0.60)


# ── passive strategy backtest ─────────────────────────────────────────────────

class PassiveBacktest:
    """
    Bar-by-bar simulation on daily candles.
    Enters when EMA50/200 agree on direction across all tracked symbols.
    Exits at TP (10%), SL (5%), or 14-day max-hold.
    """

    def __init__(self):
        self.cash = float(PASSIVE_CAPITAL)
        self.initial = float(PASSIVE_CAPITAL)
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.held_symbols: set[str] = set()      # expose to conflict checker

    # ─── internal ────────────────────────────────────────────────────────────

    def _close(self, sym: str, pnl: float, reason: str, hold_days: int) -> None:
        pos = self.positions.pop(sym)
        self.held_symbols.discard(sym)
        fee = pos["position_value"] * TAKER_FEE * 2  # open + close
        net_pnl = pnl - fee
        self.cash += pos["position_value"] + net_pnl
        self.trades.append({
            "symbol": sym,
            "side": pos["side"],
            "pnl": round(net_pnl, 4),
            "reason": reason,
            "hold_days": hold_days,
        })

    def _check_exit(self, sym: str, high: float, low: float, close: float, bar_idx: int) -> None:
        pos = self.positions[sym]
        pnl = 0.0
        closed = False
        reason = ""
        if pos["side"] == "LONG":
            if low <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["size"]
                closed, reason = True, "SL"
            elif high >= pos["tp"]:
                pnl = (pos["tp"] - pos["entry"]) * pos["size"]
                closed, reason = True, "TP"
        else:
            if high >= pos["sl"]:
                pnl = (pos["entry"] - pos["sl"]) * pos["size"]
                closed, reason = True, "SL"
            elif low <= pos["tp"]:
                pnl = (pos["entry"] - pos["tp"]) * pos["size"]
                closed, reason = True, "TP"

        hold_days = bar_idx - pos["open_bar"]
        if not closed and hold_days >= 14:
            pnl = (
                (close - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
                else (pos["entry"] - close) * pos["size"]
            )
            closed, reason = True, "MAX_HOLD"

        if closed:
            self._close(sym, pnl, reason, hold_days)

    def _open(self, sym: str, side: str, price: float, bar_idx: int) -> None:
        slot_cap = PASSIVE_CAPITAL / PASSIVE_MAX_POSITIONS
        value = min(PASSIVE_MAX_POSITION_SIZE, slot_cap, self.cash)
        if value <= 0:
            return
        self.cash -= value
        size = value / price
        tp = price * (1 + PASSIVE_TAKE_PROFIT if side == "LONG" else 1 - PASSIVE_TAKE_PROFIT)
        sl = price * (1 - PASSIVE_STOP_LOSS if side == "LONG" else 1 + PASSIVE_STOP_LOSS)
        self.positions[sym] = {
            "side": side,
            "entry": price,
            "size": size,
            "position_value": value,
            "tp": round(tp, 6),
            "sl": round(sl, 6),
            "open_bar": bar_idx,
        }
        self.held_symbols.add(sym)

    # ─── public ──────────────────────────────────────────────────────────────

    def run(self, dfs: dict, min_len: int) -> dict:
        """
        dfs: {symbol: DataFrame with timestamp/open/high/low/close/volume}
        min_len: aligned length across all symbols
        """
        symbols = list(dfs.keys())
        warmup = 210   # need 200+ bars before EMA200 is reliable

        for i in range(warmup, min_len):
            # 1. Manage open positions
            for sym in list(self.positions.keys()):
                df = dfs[sym]
                self._check_exit(
                    sym,
                    float(df["high"].iloc[i]),
                    float(df["low"].iloc[i]),
                    float(df["close"].iloc[i]),
                    i,
                )

            # 2. Try to open new positions
            for sym in symbols:
                if sym in self.positions or len(self.positions) >= PASSIVE_MAX_POSITIONS:
                    break
                closes = dfs[sym]["close"].values[: i + 1]
                ema50 = _ema(closes, 50)
                ema200 = _ema(closes, 200)
                if ema50 > ema200:
                    side = "LONG"
                elif ema50 < ema200:
                    side = "SHORT"
                else:
                    continue
                self._open(sym, side, float(dfs[sym]["close"].iloc[i]), i)

            # 3. Snapshot equity
            unrealized = 0.0
            for sym, pos in self.positions.items():
                p = float(dfs[sym]["close"].iloc[i])
                unrealized += (
                    (p - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
                    else (pos["entry"] - p) * pos["size"]
                )
            invested = sum(p["position_value"] for p in self.positions.values())
            self.equity_curve.append(self.cash + invested + unrealized)

        return self._stats()

    def _stats(self) -> dict:
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        n = len(self.trades)
        total_pnl = sum(t["pnl"] for t in self.trades)
        final_eq = self.equity_curve[-1] if self.equity_curve else self.cash
        max_dd = _max_drawdown(self.equity_curve, self.initial)
        hold_avg = (
            round(sum(t["hold_days"] for t in self.trades) / n, 1) if n else 0
        )
        return {
            "strategy": "passive",
            "trades": n,
            "wins": wins,
            "winrate": round(wins / n, 4) if n else 0,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round((final_eq - self.initial) / self.initial * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_equity": round(final_eq, 2),
            "avg_hold_days": hold_avg,
        }


# ── active strategy backtest ──────────────────────────────────────────────────

class ActiveBacktest:
    """
    Bar-by-bar simulation on 1H candles.
    Only opens trades during UTC+7 09:00–22:00.
    Exits at TP (1.5%), SL (0.7%), or forced close at EOD (22:00 UTC+7).
    Daily loss is capped at ACTIVE_MAX_DAILY_LOSS.
    """

    def __init__(self, blocked_symbols: set[str] | None = None):
        self.cash = float(ACTIVE_CAPITAL)
        self.initial = float(ACTIVE_CAPITAL)
        self.blocked = blocked_symbols or set()
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.daily_loss: float = 0.0
        self.current_day: int = -1

    # ─── internal ────────────────────────────────────────────────────────────

    def _refresh_day(self, ts_ms: int) -> None:
        day = int(ts_ms) // 86_400_000
        if day != self.current_day:
            self.daily_loss = 0.0
            self.current_day = day

    def _signal(self, closes: np.ndarray, ts_ms: int) -> str:
        """
        Realistic mean-reversion signal matching the live strategy:
          RSI(14) < 30 + price below BB lower  → LONG
          RSI(14) > 70 + price above BB upper  → SHORT
        Gated by:
          - Session quality (skip low-quality hours)
          - 1H trend bias (EMA50 vs EMA200) to block counter-trend fades
        """
        if len(closes) < 30:
            return "HOLD"
        if _session_quality(ts_ms) < _MIN_SESSION_QUALITY:
            return "HOLD"

        rsi = _rsi(closes)
        upper, _, lower = _bollinger(closes)
        price = float(closes[-1])

        # 1H trend bias — approximates 4H gate used in live strategy
        trend = "NEUTRAL"
        if len(closes) >= 200:
            ema50  = _ema(closes, 50)
            ema200 = _ema(closes, 200)
            if ema50 > ema200 and price > ema50:
                trend = "BULL"
            elif ema50 < ema200 and price < ema50:
                trend = "BEAR"

        if rsi < 30 and price < lower and trend != "BEAR":
            return "LONG"
        if rsi > 70 and price > upper and trend != "BULL":
            return "SHORT"
        return "HOLD"

    def _close(self, sym: str, price: float, reason: str, bar_idx: int) -> None:
        pos = self.positions.pop(sym)
        gross_pnl = (
            (price - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
            else (pos["entry"] - price) * pos["size"]
        )
        fee = pos["position_value"] * TAKER_FEE * 2  # open + close
        pnl = gross_pnl - fee
        self.cash += pos["position_value"] + pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)
        self.trades.append({
            "symbol": sym,
            "side": pos["side"],
            "pnl": round(pnl, 4),
            "reason": reason,
        })

    # ─── public ──────────────────────────────────────────────────────────────

    def run(self, dfs: dict, min_len: int) -> dict:
        symbols = [s for s in dfs if s not in self.blocked]
        warmup = 210   # need EMA200 to be reliable

        for i in range(warmup, min_len):
            # Pick timestamp from first symbol
            ts_ms = int(dfs[symbols[0]]["timestamp"].iloc[i]) if symbols else 0
            hour = _bar_hour_utc7(ts_ms)
            in_window = ACTIVE_TRADING_START <= hour < ACTIVE_TRADING_END
            self._refresh_day(ts_ms)

            # 1. Manage positions
            for sym in list(self.positions.keys()):
                df = dfs[sym]
                high = float(df["high"].iloc[i])
                low = float(df["low"].iloc[i])
                price = float(df["close"].iloc[i])
                pos = self.positions[sym]

                # Force close at EOD (22:00 UTC+7 = hour >= ACTIVE_TRADING_END)
                if hour >= ACTIVE_TRADING_END:
                    self._close(sym, price, "EOD", i)
                    continue

                if pos["side"] == "LONG":
                    if low <= pos["sl"]:
                        self._close(sym, pos["sl"], "SL", i)
                    elif high >= pos["tp"]:
                        self._close(sym, pos["tp"], "TP", i)
                else:
                    if high >= pos["sl"]:
                        self._close(sym, pos["sl"], "SL", i)
                    elif low <= pos["tp"]:
                        self._close(sym, pos["tp"], "TP", i)

            # 2. Open if within window and daily loss OK
            if in_window and self.daily_loss < ACTIVE_MAX_DAILY_LOSS:
                for sym in symbols:
                    if sym in self.positions or len(self.positions) >= ACTIVE_MAX_POSITIONS:
                        break
                    closes = dfs[sym]["close"].values[: i + 1]
                    side = self._signal(closes, ts_ms)
                    if side == "HOLD":
                        continue
                    price = float(dfs[sym]["close"].iloc[i])
                    value = min(ACTIVE_MAX_POSITION_SIZE, ACTIVE_CAPITAL / ACTIVE_MAX_POSITIONS, self.cash)
                    if value <= 0:
                        continue
                    self.cash -= value
                    size = value / price
                    tp = price * (1 + ACTIVE_TAKE_PROFIT if side == "LONG" else 1 - ACTIVE_TAKE_PROFIT)
                    sl = price * (1 - ACTIVE_STOP_LOSS if side == "LONG" else 1 + ACTIVE_STOP_LOSS)
                    self.positions[sym] = {
                        "side": side,
                        "entry": price,
                        "size": size,
                        "position_value": value,
                        "tp": round(tp, 6),
                        "sl": round(sl, 6),
                    }

            # 3. Snapshot equity
            unrealized = 0.0
            for sym, pos in self.positions.items():
                p = float(dfs[sym]["close"].iloc[i])
                unrealized += (
                    (p - pos["entry"]) * pos["size"] if pos["side"] == "LONG"
                    else (pos["entry"] - p) * pos["size"]
                )
            invested = sum(p["position_value"] for p in self.positions.values())
            self.equity_curve.append(self.cash + invested + unrealized)

        return self._stats()

    def _stats(self) -> dict:
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        n = len(self.trades)
        total_pnl = sum(t["pnl"] for t in self.trades)
        final_eq = self.equity_curve[-1] if self.equity_curve else self.cash
        max_dd = _max_drawdown(self.equity_curve, self.initial)
        daily_loss_hits = sum(1 for t in self.trades if t["reason"] == "EOD" and t["pnl"] < 0)
        return {
            "strategy": "active",
            "trades": n,
            "wins": wins,
            "winrate": round(wins / n, 4) if n else 0,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round((final_eq - self.initial) / self.initial * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_equity": round(final_eq, 2),
            "eod_closes": daily_loss_hits,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _max_drawdown(equity_curve: list[float], initial: float) -> float:
    if not equity_curve:
        return 0.0
    peak = initial
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _fetch_aligned(market: MarketData, symbols: list[str], timeframe: str, limit: int) -> tuple[dict, int]:
    """Fetch OHLCV for each symbol, align to shortest common length."""
    dfs = {}
    for sym in symbols:
        try:
            df = market.get_ohlcv_df(sym, timeframe=timeframe, limit=limit)
            if len(df) >= 50:
                dfs[sym] = df.reset_index(drop=True)
        except Exception as e:
            print(f"  [Backtest] {sym} fetch error: {e}")
    if not dfs:
        return {}, 0
    min_len = min(len(df) for df in dfs.values())
    for sym in dfs:
        dfs[sym] = dfs[sym].tail(min_len).reset_index(drop=True)
    return dfs, min_len


def _print_section(title: str, stats: dict) -> None:
    w = 52
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    labels = {
        "trades": "Trades",
        "wins": "Wins",
        "winrate": "Win rate",
        "total_pnl": "Total P&L",
        "return_pct": "Return %",
        "max_drawdown_pct": "Max drawdown %",
        "final_equity": "Final equity",
        "avg_hold_days": "Avg hold (days)",
        "eod_closes": "EOD force-closes",
    }
    for k, label in labels.items():
        if k not in stats:
            continue
        val = stats[k]
        if k == "winrate":
            print(f"  {label:<22} {val:.1%}")
        elif isinstance(val, float):
            print(f"  {label:<22} {val:,.2f}")
        else:
            print(f"  {label:<22} {val}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    passive_symbols: list[str],
    active_symbols: list[str],
) -> dict:
    market = MarketData()

    print(f"\n[DualBacktest] Fetching passive data (1D, ~12 months)…")
    passive_dfs, passive_len = _fetch_aligned(market, passive_symbols, "1d", 600)

    print(f"[DualBacktest] Fetching active data (1H, ~90 days)…")
    active_dfs, active_len = _fetch_aligned(market, active_symbols, "1h", 2200)

    if not passive_dfs:
        print("  No passive data — aborting passive backtest")
    if not active_dfs:
        print("  No active data — aborting active backtest")

    # Run passive
    passive_bt = PassiveBacktest()
    passive_stats: dict = {}
    if passive_dfs:
        print(f"\n[PassiveBacktest] Running on {list(passive_dfs)} ({passive_len} bars)…")
        passive_stats = passive_bt.run(passive_dfs, passive_len)

    # Run active (conflict: block symbols held by passive at end of passive run)
    active_bt = ActiveBacktest(blocked_symbols=passive_bt.held_symbols)
    active_stats: dict = {}
    if active_dfs:
        print(f"[ActiveBacktest]  Running on {list(active_dfs)} ({active_len} bars)…")
        active_stats = active_bt.run(active_dfs, active_len)

    # Combined summary
    passive_eq = passive_stats.get("final_equity", PASSIVE_CAPITAL)
    active_eq = active_stats.get("final_equity", ACTIVE_CAPITAL)
    total_initial = PASSIVE_CAPITAL + ACTIVE_CAPITAL
    total_eq = passive_eq + active_eq
    total_pnl = passive_stats.get("total_pnl", 0) + active_stats.get("total_pnl", 0)
    total_return = round((total_eq - total_initial) / total_initial * 100, 2)
    kill_triggered = total_eq < KILL_SWITCH_EQUITY

    combined = {
        "passive_equity": round(passive_eq, 2),
        "active_equity": round(active_eq, 2),
        "total_equity": round(total_eq, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": total_return,
        "kill_switch_triggered": kill_triggered,
    }

    _print_section("PASSIVE STRATEGY", passive_stats)
    _print_section("ACTIVE STRATEGY", active_stats)

    w = 52
    print(f"\n{'=' * w}")
    print(f"  COMBINED PORTFOLIO")
    print(f"{'=' * w}")
    print(f"  {'Passive equity':<22} {passive_eq:,.2f}")
    print(f"  {'Active equity':<22} {active_eq:,.2f}")
    print(f"  {'Total equity':<22} {total_eq:,.2f}  (initial {total_initial:,.0f})")
    print(f"  {'Total P&L':<22} {total_pnl:+,.2f}")
    print(f"  {'Total return':<22} {total_return:+.2f}%")
    kc = "YES — would halt" if kill_triggered else "No"
    print(f"  {'Kill switch':<22} {kc}  (threshold {KILL_SWITCH_EQUITY:,.0f})")
    print(f"{'=' * w}\n")

    return {"passive": passive_stats, "active": active_stats, "combined": combined}


def main():
    p = argparse.ArgumentParser(description="Dual strategy backtest")
    p.add_argument("--passive", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    p.add_argument("--active", nargs="+", default=["SOL/USDT:USDT", "BNB/USDT:USDT"])
    args = p.parse_args()
    run(passive_symbols=args.passive, active_symbols=args.active)


if __name__ == "__main__":
    main()
