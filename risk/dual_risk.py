"""Risk manager for the dual strategy engine."""

from collections import deque
from datetime import datetime, timezone
from typing import Optional

from config.dual_settings import (
    ACTIVE_CAPITAL,
    ACTIVE_MAX_DAILY_LOSS,
    ACTIVE_MAX_POSITION_SIZE,
    ACTIVE_STOP_LOSS,
    ACTIVE_TAKE_PROFIT,
    DUAL_ALLOW_SAME_SYMBOL_STACKING,
    KILL_SWITCH_EQUITY,
    MAX_PORTFOLIO_DRAWDOWN,
    PASSIVE_CAPITAL,
    PASSIVE_MAX_DRAWDOWN,
    PASSIVE_MAX_POSITION_SIZE,
    PASSIVE_STOP_LOSS,
    PASSIVE_TAKE_PROFIT,
)
from risk.kelly import KellyCriterion


class DualConflictChecker:
    """Prevent passive and active strategies from opening incompatible exposure."""

    def check(self, strategy: str, signal: dict, passive_positions: list, active_positions: list) -> dict:
        side = signal.get("side")
        symbol = signal.get("symbol")
        if side == "HOLD" or not symbol:
            return {
                "allow_trade": True,
                "reason": "OK",
            }

        other_strategy = "active" if strategy == "passive" else "passive"
        other_positions = active_positions if strategy == "passive" else passive_positions

        for position in other_positions:
            if position.get("symbol") != symbol:
                continue
            other_side = position.get("side")
            if other_side != side:
                return {
                    "allow_trade": False,
                    "reason": (
                        f"{strategy} {side} conflicts with "
                        f"{other_strategy} {other_side} on {symbol}"
                    ),
                }
            if not DUAL_ALLOW_SAME_SYMBOL_STACKING:
                return {
                    "allow_trade": False,
                    "reason": f"{strategy} duplicate exposure with {other_strategy} on {symbol}",
                }

        return {
            "allow_trade": True,
            "reason": "OK",
        }


class DualRiskManager:
    """Manage combined and strategy-specific risk for passive and active allocations."""

    def __init__(self):
        self.daily_loss_active = 0.0
        self.daily_loss_date = self._current_date()
        self.peak_equity = PASSIVE_CAPITAL + ACTIVE_CAPITAL
        self.conflict_checker = DualConflictChecker()
        self.kelly = KellyCriterion()
        # Rolling win-rate: last 50 trades per strategy for more responsive Kelly sizing
        self._trade_history: dict[str, deque] = {
            "passive": deque(maxlen=50),
            "active":  deque(maxlen=50),
        }
        self.strategy_stats = {
            "passive": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
            },
            "active": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
            },
        }
        # Kill-switch hysteresis: only halt after 3 consecutive checks below threshold
        self._kill_switch_count: int = 0
        # Track minimum ever-seen equity so the baseline persists across restarts
        self._min_ever_equity: float = float("inf")

    def _current_date(self):
        return datetime.now(timezone.utc).date()

    def refresh_daily(self):
        current_date = self._current_date()
        if current_date != self.daily_loss_date:
            self.daily_loss_active = 0.0
            self.daily_loss_date = current_date

    def check_conflict(self, strategy: str, signal: dict, passive_positions: list, active_positions: list) -> dict:
        return self.conflict_checker.check(strategy, signal, passive_positions, active_positions)

    def allow_passive(self, portfolio, signal, market_state, max_position_value: Optional[float] = None):
        self.refresh_daily()
        if portfolio.current_drawdown() >= PASSIVE_MAX_DRAWDOWN:
            return {
                "allow_trade": False,
                "reason": "PASSIVE DRAWDOWN LIMIT",
            }
        # SIDEWAYS block removed — PassiveStrategy's ADX + EMA + pullback filters
        # are sufficient quality gates; hard-blocking sideways wastes $700 of capital.
        if portfolio.cash <= 0:
            return {
                "allow_trade": False,
                "reason": "PASSIVE NO CASH",
            }
        # Kelly-adjusted cap using rolling win rate (last 50 trades)
        stats   = self.strategy_stats["passive"]
        history = self._trade_history["passive"]
        winrate = sum(history) / len(history) if history else 0.5
        kelly_value = self.kelly.position_value(
            winrate=winrate,
            win_pct=PASSIVE_TAKE_PROFIT,
            loss_pct=PASSIVE_STOP_LOSS,
            equity=portfolio.equity,
            trades_so_far=stats["trades"],
            default_fraction=PASSIVE_MAX_POSITION_SIZE / max(portfolio.equity, 1),
        )
        base_cap = PASSIVE_MAX_POSITION_SIZE if max_position_value is None else max_position_value
        position_cap = min(base_cap, kelly_value)
        return {
            "allow_trade": True,
            "max_position_value": min(position_cap, portfolio.cash),
        }

    def allow_active(self, portfolio, signal, market_state, max_position_value: Optional[float] = None):
        self.refresh_daily()
        if self.daily_loss_active >= ACTIVE_MAX_DAILY_LOSS:
            return {
                "allow_trade": False,
                "reason": "ACTIVE DAILY LOSS LIMIT",
            }
        if portfolio.cash <= 0:
            return {
                "allow_trade": False,
                "reason": "ACTIVE NO CASH",
            }
        if not signal.get("intraday", False):
            return {
                "allow_trade": False,
                "reason": "ACTIVE OUTSIDE TRADING WINDOW",
            }
        # Kelly-adjusted cap using rolling win rate (last 50 trades)
        stats   = self.strategy_stats["active"]
        history = self._trade_history["active"]
        winrate = sum(history) / len(history) if history else 0.5
        kelly_value = self.kelly.position_value(
            winrate=winrate,
            win_pct=ACTIVE_TAKE_PROFIT,
            loss_pct=ACTIVE_STOP_LOSS,
            equity=portfolio.equity,
            trades_so_far=stats["trades"],
            default_fraction=ACTIVE_MAX_POSITION_SIZE / max(portfolio.equity, 1),
        )
        base_cap = ACTIVE_MAX_POSITION_SIZE if max_position_value is None else max_position_value
        position_cap = min(base_cap, kelly_value)
        return {
            "allow_trade": True,
            "max_position_value": min(position_cap, portfolio.cash),
        }

    def record_trade(self, strategy: str, pnl: float):
        stats = self.strategy_stats.setdefault(
            strategy,
            {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
            },
        )
        stats["trades"] += 1
        stats["pnl"] += pnl
        if pnl >= 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        self._trade_history[strategy].append(1 if pnl >= 0 else 0)

        if strategy == "active" and pnl < 0:
            self.refresh_daily()
            self.daily_loss_active += abs(pnl)

    def get_strategy_stats(self) -> dict:
        result = {}
        for strategy, stats in self.strategy_stats.items():
            d = dict(stats)
            d["expected_value"] = self.expected_value(strategy)
            d["kelly_fraction"] = self._kelly_fraction(strategy)
            result[strategy] = d
        return result

    def expected_value(self, strategy: str) -> float:
        """
        EV per trade as a fraction of position size.
        EV = winrate × tp_pct  − (1−winrate) × sl_pct
        Positive EV = system has edge; negative = no edge.
        """
        stats = self.strategy_stats.get(strategy, {})
        n = stats.get("trades", 0)
        if n < 5:
            return 0.0
        winrate = stats.get("wins", 0) / n
        if strategy == "passive":
            tp, sl = PASSIVE_TAKE_PROFIT, PASSIVE_STOP_LOSS
        else:
            tp, sl = ACTIVE_TAKE_PROFIT, ACTIVE_STOP_LOSS
        ev = round(winrate * tp - (1 - winrate) * sl, 6)
        return ev

    def _kelly_fraction(self, strategy: str) -> float:
        history = self._trade_history.get(strategy, deque())
        if len(history) < self.kelly.MIN_TRADES:
            return 0.0
        winrate = sum(history) / len(history)
        if strategy == "passive":
            tp, sl = PASSIVE_TAKE_PROFIT, PASSIVE_STOP_LOSS
        else:
            tp, sl = ACTIVE_TAKE_PROFIT, ACTIVE_STOP_LOSS
        return self.kelly.half_kelly_fraction(winrate, tp, sl)

    def check_portfolio(self, passive_equity: float, active_equity: float) -> dict:
        total_equity = passive_equity + active_equity
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = max(0.0, (self.peak_equity - total_equity) / self.peak_equity)

        # Kill-switch with hysteresis: require 3 consecutive checks below threshold
        # before halting.  A single bad tick (e.g. rate-limit → stale price) cannot
        # trigger a false halt; the counter resets as soon as equity recovers.
        if total_equity < KILL_SWITCH_EQUITY:
            self._kill_switch_count += 1
            if self._kill_switch_count >= 3:
                return {
                    "halt": True,
                    "reason": f"KILL SWITCH EQUITY (count={self._kill_switch_count})",
                }
            # Below threshold but not yet 3 consecutive — warn only
            return {
                "halt": False,
                "reason": f"KILL_SWITCH_WARN count={self._kill_switch_count}/3",
            }
        else:
            self._kill_switch_count = 0   # reset counter when equity is healthy

        if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
            return {
                "halt": True,
                "reason": "TOTAL DRAWDOWN LIMIT",
            }
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        return {
            "halt": False,
            "reason": "OK",
        }
