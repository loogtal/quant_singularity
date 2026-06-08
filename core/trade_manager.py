"""
Position lifecycle manager — trailing stops and partial profit-taking.

Logic per strategy type:

Passive (multi-day holds):
  - Activates trailing once position is +PASSIVE_TRAIL_ACTIVATION in profit
  - Trails at PASSIVE_TRAIL_MULT × ATR(4H) below running peak (LONG) / above trough (SHORT)
  - After activation, stop never retreats past entry (breakeven lock)
  - On first touch of original TP: close 50%, trail the other 50% to TP2

Active (intraday):
  - Activates trailing once position is +ACTIVE_TRAIL_ACTIVATION in profit
  - Trails at ACTIVE_TRAIL_MULT × ATR(15M)
  - Breakeven stop: when unrealised PnL reaches +1.0x initial risk, move SL to entry ±0.1%
  - Partial profit lock: when unrealised PnL reaches +1.5x initial risk, return close_partial

Actions returned:
  {"action": "hold"}
  {"action": "move_stop", "new_stop": <float>}
  {"action": "close_partial", "reason": str}
  {"action": "close", "reason": str}
"""

import time
import numpy as np

from data.market_data import MarketData


# ── ATR helper ─────────────────────────────────────────────────────────────────

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return float(closes[-1]) * 0.02
    tr_list = []
    for i in range(1, len(closes)):
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    n = min(period, len(tr_list))
    return float(np.mean(tr_list[-n:])) if tr_list else float(closes[-1]) * 0.015


# ── configuration ──────────────────────────────────────────────────────────────

# Passive: wide trail to let multi-day trends breathe
PASSIVE_TRAIL_MULT       = 2.5   # 2.5×ATR trailing distance
PASSIVE_TRAIL_ACTIVATION = 0.04  # activate after +4% unrealised profit
PASSIVE_PARTIAL_MULT     = 2.0   # TP2 = entry ± 2 × original TP distance

# Active: tight trail for intraday protection
ACTIVE_TRAIL_MULT        = 1.5   # 1.5×ATR trailing distance
ACTIVE_TRAIL_ACTIVATION  = 0.012 # activate after +1.2% unrealised profit

# Smart exit thresholds (multiples of initial risk = SL distance)
BREAKEVEN_RISK_MULT    = 1.0   # move SL to breakeven+0.1% once profit = 1× initial risk
PARTIAL_PROFIT_MULT    = 1.5   # close half once profit = 1.5× initial risk


class TradeManager:
    """
    Manages trailing stops and partial exits for all open positions.

    Usage in DualEngine:
        tm = TradeManager()
        # each cycle, for every position:
        action = tm.tick(position)
        if action["action"] == "close":
            _close_position(portfolio, position)
        elif action["action"] == "close_partial":
            _close_half_position(portfolio, position)
    """

    def __init__(self):
        self.market = MarketData()
        # Running peak / trough price seen while position is open
        self._peak:   dict[str, float] = {}
        self._trough: dict[str, float] = {}
        # Closed trades log for daily report (capped at last 500)
        self._closed_trades: list[dict] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def register(self, symbol: str, entry_price: float, side: str) -> None:
        """Call when a new position is opened."""
        if side == "LONG":
            self._peak[symbol]   = entry_price
        else:
            self._trough[symbol] = entry_price

    def release(self, symbol: str) -> None:
        """Call when a position is fully closed."""
        self._peak.pop(symbol, None)
        self._trough.pop(symbol, None)

    def record_closed(self, position: dict, pnl: float) -> None:
        """Record a closed trade for daily reporting."""
        self._closed_trades.append({
            "symbol":      position.get("symbol", ""),
            "side":        position.get("side", ""),
            "pnl":         pnl,
            "signal_mode": position.get("signal_mode", position.get("passive_mode", "")),
            "confidence":  position.get("confidence", 0.0),
            "regime":      position.get("regime", ""),
            "strategy":    position.get("strategy", ""),
            "closed_at":   time.time(),
        })
        # Keep only the last 500 trades
        if len(self._closed_trades) > 500:
            self._closed_trades = self._closed_trades[-500:]

    def get_recent_trades(self, hours: float = 24.0) -> list[dict]:
        """Return trades closed within the last `hours` hours."""
        cutoff = time.time() - hours * 3600
        return [t for t in self._closed_trades if t.get("closed_at", 0) >= cutoff]

    def tick(self, position: dict) -> dict:
        """
        Evaluate a position for the current cycle.

        Updates position["stop_loss"] in-place if trailing stop improves.
        Returns an action dict.
        """
        symbol   = position["symbol"]
        side     = position["side"]
        strategy = position.get("strategy", "passive")
        current  = position.get("current_price", position["entry_price"])
        entry    = position["entry_price"]

        if strategy == "passive":
            trail_mult   = PASSIVE_TRAIL_MULT
            activation   = PASSIVE_TRAIL_ACTIVATION
            timeframe    = "4h"
        else:
            trail_mult   = ACTIVE_TRAIL_MULT
            activation   = ACTIVE_TRAIL_ACTIVATION
            timeframe    = "15m"

        atr = self._get_atr(symbol, timeframe)
        if atr <= 0:
            return {"action": "hold"}

        if side == "LONG":
            return self._tick_long(position, current, entry, strategy, atr, trail_mult, activation)
        else:
            return self._tick_short(position, current, entry, strategy, atr, trail_mult, activation)

    # ── internal ───────────────────────────────────────────────────────────────

    def _get_atr(self, symbol: str, timeframe: str, period: int = 14) -> float:
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe=timeframe, limit=period + 10)
            if len(df) < 5:
                return 0.0
            return _atr(df["high"].values, df["low"].values, df["close"].values, period)
        except Exception:
            return 0.0

    @staticmethod
    def _initial_risk(position: dict) -> float:
        """Return the initial risk (SL distance in price) for a position.
        Falls back to a small fraction of entry if stop_loss is missing.
        """
        entry = position["entry_price"]
        sl    = position.get("stop_loss")
        if sl and sl > 0:
            return abs(entry - sl)
        # Fallback: 1% of entry as default risk distance
        return entry * 0.01

    def _tick_long(
        self,
        position: dict,
        current: float,
        entry: float,
        strategy: str,
        atr: float,
        trail_mult: float,
        activation: float,
    ) -> dict:
        symbol = position["symbol"]

        # Update running peak
        peak = self._peak.get(symbol, current)
        if current > peak:
            peak = current
            self._peak[symbol] = peak

        profit_pct = (peak - entry) / entry

        # ── partial exit at original TP (passive only) ──────────────────────
        if (
            strategy == "passive"
            and not position.get("partial_done", False)
            and current >= position.get("take_profit", float("inf"))
        ):
            position["partial_done"] = True   # persisted in position dict — survives restart
            # Extend TP to TP2 = entry + 2× original TP distance
            original_tp_dist = position["take_profit"] - entry
            tp2 = round(entry + PASSIVE_PARTIAL_MULT * original_tp_dist, 6)
            position["take_profit"] = tp2
            return {"action": "close_partial", "reason": f"TP1_PARTIAL tp2={tp2:.4f}"}

        # ── Smart exit: active positions only ──────────────────────────────
        if strategy == "active":
            initial_risk = self._initial_risk(position)
            unrealised   = current - entry

            # Breakeven stop: lock in trade once profit >= 1× initial risk
            if (
                unrealised >= BREAKEVEN_RISK_MULT * initial_risk
                and not position.get("breakeven_done", False)
            ):
                be_stop = round(entry * 1.001, 6)   # entry + 0.1%
                current_sl = position.get("stop_loss", 0.0)
                if be_stop > current_sl:
                    position["stop_loss"]      = be_stop
                    position["trailing_active"] = True
                position["breakeven_done"] = True

            # Partial profit lock: close half once profit >= 1.5× initial risk
            if (
                unrealised >= PARTIAL_PROFIT_MULT * initial_risk
                and not position.get("half_closed", False)
            ):
                position["half_closed"] = True
                return {"action": "close_partial", "reason": "PARTIAL_TP"}

        # ── trailing stop ───────────────────────────────────────────────────
        if profit_pct >= activation:
            trail_stop = peak - trail_mult * atr
            # Breakeven lock: don't let stop fall below entry after activation
            trail_stop = max(trail_stop, entry * 1.0005)

            current_sl = position.get("stop_loss", 0.0)
            if trail_stop > current_sl:
                position["stop_loss"] = round(trail_stop, 6)
                position["trailing_active"] = True

            if current <= position["stop_loss"]:
                return {
                    "action": "close",
                    "reason": (
                        f"TRAIL_STOP peak={round(peak, 4)} "
                        f"stop={round(position['stop_loss'], 4)}"
                    ),
                }

        return {"action": "hold"}

    def _tick_short(
        self,
        position: dict,
        current: float,
        entry: float,
        strategy: str,
        atr: float,
        trail_mult: float,
        activation: float,
    ) -> dict:
        symbol = position["symbol"]

        # Update running trough
        trough = self._trough.get(symbol, current)
        if current < trough:
            trough = current
            self._trough[symbol] = trough

        profit_pct = (entry - trough) / entry

        # ── partial exit at original TP (passive only) ──────────────────────
        if (
            strategy == "passive"
            and not position.get("partial_done", False)
            and current <= position.get("take_profit", 0.0)
        ):
            position["partial_done"] = True   # persisted in position dict — survives restart
            original_tp_dist = entry - position["take_profit"]
            tp2 = round(entry - PASSIVE_PARTIAL_MULT * original_tp_dist, 6)
            position["take_profit"] = tp2
            return {"action": "close_partial", "reason": f"TP1_PARTIAL tp2={tp2:.4f}"}

        # ── Smart exit: active positions only ──────────────────────────────
        if strategy == "active":
            initial_risk = self._initial_risk(position)
            unrealised   = entry - current   # profit for SHORT = entry - current

            # Breakeven stop: lock in trade once profit >= 1× initial risk
            if (
                unrealised >= BREAKEVEN_RISK_MULT * initial_risk
                and not position.get("breakeven_done", False)
            ):
                be_stop = round(entry * 0.999, 6)   # entry - 0.1%
                current_sl = position.get("stop_loss", float("inf"))
                if be_stop < current_sl:
                    position["stop_loss"]      = be_stop
                    position["trailing_active"] = True
                position["breakeven_done"] = True

            # Partial profit lock: close half once profit >= 1.5× initial risk
            if (
                unrealised >= PARTIAL_PROFIT_MULT * initial_risk
                and not position.get("half_closed", False)
            ):
                position["half_closed"] = True
                return {"action": "close_partial", "reason": "PARTIAL_TP"}

        # ── trailing stop ───────────────────────────────────────────────────
        if profit_pct >= activation:
            trail_stop = trough + trail_mult * atr
            # Breakeven lock
            trail_stop = min(trail_stop, entry * 0.9995)

            current_sl = position.get("stop_loss", float("inf"))
            if trail_stop < current_sl:
                position["stop_loss"] = round(trail_stop, 6)
                position["trailing_active"] = True

            if current >= position["stop_loss"]:
                return {
                    "action": "close",
                    "reason": (
                        f"TRAIL_STOP trough={round(trough, 4)} "
                        f"stop={round(position['stop_loss'], 4)}"
                    ),
                }

        return {"action": "hold"}
