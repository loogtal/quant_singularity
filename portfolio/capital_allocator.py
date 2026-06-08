"""Dynamic capital allocation for dual strategy mode."""

from typing import Optional

from config.dual_settings import (
    ACTIVE_CAPITAL,
    ACTIVE_MAX_CAPITAL_FRACTION,
    ACTIVE_MAX_POSITION_SIZE,
    ACTIVE_MIN_CAPITAL_FRACTION,
    ALLOCATION_MIN_TRANSFER,
    ALLOCATION_REBALANCE_CYCLES,
    ALLOCATION_SCORE_THRESHOLD,
    ALLOCATION_STEP,
    DYNAMIC_ALLOCATION,
    PASSIVE_CAPITAL,
    PASSIVE_MAX_POSITION_SIZE,
    TOTAL_CAPITAL,
)


class DualCapitalAllocator:
    """Shift idle capital toward the strategy with better recent results."""

    def __init__(self):
        configured_total = PASSIVE_CAPITAL + ACTIVE_CAPITAL
        starting_total = max(
            configured_total if configured_total > 0 else TOTAL_CAPITAL,
            1.0,
        )
        self.active_weight = self._clamp_active_weight(ACTIVE_CAPITAL / starting_total)
        self.last_rebalance_cycle = 0
        self.history: list[dict] = []

    def _clamp_active_weight(self, weight: float) -> float:
        floor = max(0.0, min(ACTIVE_MIN_CAPITAL_FRACTION, 1.0))
        ceiling = max(floor, min(ACTIVE_MAX_CAPITAL_FRACTION, 1.0))
        return min(max(float(weight), floor), ceiling)

    # Minimum trades before a strategy's score is trusted for rebalancing.
    # Below this, its score is treated as 0 (neutral) to avoid
    # draining a starting strategy based on noise.
    _MIN_TRADES_FOR_SCORE = 10

    def _score(self, portfolio, stats: Optional[dict]) -> float:
        stats = stats or {}
        trades = int(stats.get("trades", 0) or 0)
        wins = int(stats.get("wins", 0) or 0)

        # Not enough trades to evaluate: return neutral score
        if trades < self._MIN_TRADES_FOR_SCORE:
            return 0.0

        winrate = wins / trades
        pnl = float(portfolio.realized_pnl + portfolio.unrealized_pnl)
        base = max(float(getattr(portfolio, "initial_cash", 0.0)), 1.0)
        pnl_return = pnl / base
        confidence = min(trades, 20) / 20
        winrate_score = (winrate - 0.5) * 0.08 * confidence
        drawdown_penalty = portfolio.current_drawdown() * 0.35
        return round(float(pnl_return + winrate_score - drawdown_penalty), 6)

    def _target_weight(self, passive_score: float, active_score: float) -> float:
        diff = active_score - passive_score
        if abs(diff) < ALLOCATION_SCORE_THRESHOLD:
            return self.active_weight
        direction = 1 if diff > 0 else -1
        return self._clamp_active_weight(self.active_weight + direction * ALLOCATION_STEP)

    def _transfer_idle_cash(self, source, destination, amount: float) -> None:
        amount = round(float(amount), 2)
        if amount <= 0:
            return
        source.cash -= amount
        destination.cash += amount
        source.initial_cash = max(1.0, source.initial_cash - amount)
        destination.initial_cash += amount
        source.update_equity()
        destination.update_equity()
        # Adjust peak_equity so intentional transfers don't trigger DrawdownGuard,
        # then call update_equity again so self.drawdown reflects the corrected peak.
        if source._peak_equity > source.equity:
            source._peak_equity = source.equity
            source.update_equity()
        if destination._peak_equity > destination.equity:
            destination._peak_equity = destination.equity
            destination.update_equity()

    def _build_status(
        self,
        passive_portfolio,
        active_portfolio,
        passive_score: float,
        active_score: float,
        transfer: Optional[dict] = None,
    ) -> dict:
        total_equity = passive_portfolio.equity + active_portfolio.equity
        active_target = total_equity * self.active_weight
        passive_target = total_equity - active_target
        return {
            "enabled": DYNAMIC_ALLOCATION,
            "passive_weight": round(1 - self.active_weight, 4),
            "active_weight": round(self.active_weight, 4),
            "passive_target_capital": round(passive_target, 2),
            "active_target_capital": round(active_target, 2),
            "passive_score": passive_score,
            "active_score": active_score,
            "last_transfer": transfer or {},
            "history": self.history[:10],
        }

    def snapshot(
        self,
        passive_portfolio,
        active_portfolio,
        stats: Optional[dict] = None,
    ) -> dict:
        stats = stats or {}
        passive_score = self._score(passive_portfolio, stats.get("passive"))
        active_score = self._score(active_portfolio, stats.get("active"))
        return self._build_status(
            passive_portfolio,
            active_portfolio,
            passive_score,
            active_score,
        )

    def position_cap(self, strategy: str, max_positions: int, total_equity: float) -> float:
        if max_positions <= 0:
            return 0.0
        weight = 1 - self.active_weight if strategy == "passive" else self.active_weight
        target_capital = max(0.0, total_equity * weight)
        if strategy == "passive":
            base_capital = PASSIVE_CAPITAL
            base_cap = PASSIVE_MAX_POSITION_SIZE
        else:
            base_capital = ACTIVE_CAPITAL
            base_cap = ACTIVE_MAX_POSITION_SIZE
        if base_capital <= 0:
            return target_capital / max_positions
        scaled_cap = base_cap * (target_capital / base_capital)
        slot_cap = target_capital / max_positions
        return max(0.0, min(scaled_cap, slot_cap))

    def rebalance(
        self,
        cycle: int,
        passive_portfolio,
        active_portfolio,
        stats: Optional[dict] = None,
    ) -> dict:
        stats = stats or {}
        passive_score = self._score(passive_portfolio, stats.get("passive"))
        active_score = self._score(active_portfolio, stats.get("active"))
        transfer: dict = {}

        if not DYNAMIC_ALLOCATION:
            return self._build_status(
                passive_portfolio,
                active_portfolio,
                passive_score,
                active_score,
                transfer,
            )

        if cycle - self.last_rebalance_cycle < ALLOCATION_REBALANCE_CYCLES:
            return self._build_status(
                passive_portfolio,
                active_portfolio,
                passive_score,
                active_score,
                transfer,
            )

        self.last_rebalance_cycle = cycle
        self.active_weight = self._target_weight(passive_score, active_score)

        total_equity = passive_portfolio.equity + active_portfolio.equity
        active_target = total_equity * self.active_weight
        passive_target = total_equity - active_target
        active_gap = active_target - active_portfolio.equity

        PASSIVE_EQUITY_FLOOR = 600.0  # never drain passive below this
        if active_gap >= ALLOCATION_MIN_TRANSFER:
            transferable = max(0.0, passive_portfolio.equity - max(passive_target, PASSIVE_EQUITY_FLOOR))
            amount = min(active_gap, transferable, passive_portfolio.cash)
            if amount >= ALLOCATION_MIN_TRANSFER:
                self._transfer_idle_cash(passive_portfolio, active_portfolio, amount)
                transfer = {
                    "from": "passive",
                    "to": "active",
                    "amount": round(float(amount), 2),
                    "reason": "active outperformed passive",
                }
        elif active_gap <= -ALLOCATION_MIN_TRANSFER:
            amount = min(
                abs(active_gap),
                max(0.0, active_portfolio.equity - active_target),
                active_portfolio.cash,
            )
            if amount >= ALLOCATION_MIN_TRANSFER:
                self._transfer_idle_cash(active_portfolio, passive_portfolio, amount)
                transfer = {
                    "from": "active",
                    "to": "passive",
                    "amount": round(float(amount), 2),
                    "reason": "passive outperformed active",
                }

        if transfer:
            self.history.insert(0, transfer)
            self.history = self.history[:25]

        return self._build_status(
            passive_portfolio,
            active_portfolio,
            passive_score,
            active_score,
            transfer,
        )
