"""Central halt logic — drawdown, daily loss, volatility, reflection."""

from config.settings import MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN, VOLATILITY_CUTOFF
from config.constants import ACTION_INCREASE, ACTION_REDUCE


class KillSwitch:
    def check(
        self,
        portfolio,
        perf: dict,
        market_state: dict,
        reflection: dict | None = None,
    ) -> dict:
        """Returns {halt: bool, reason: str, action: str}."""
        reflection = reflection or {}

        if portfolio.drawdown >= MAX_DRAWDOWN:
            return {
                "halt": True,
                "reason": f"Max drawdown {portfolio.drawdown:.1%}",
                "action": "manage_only",
            }

        if perf.get("daily_pnl_pct", 0) <= -MAX_DAILY_LOSS_PCT:
            return {
                "halt": True,
                "reason": f"Daily loss limit ({perf['daily_pnl_pct']:.2%})",
                "action": "cooldown",
            }

        if market_state.get("volatility", 0) >= VOLATILITY_CUTOFF:
            return {
                "halt": True,
                "reason": "Volatility too high",
                "action": "skip_new",
            }

        return {"halt": False, "reason": "", "action": "trade"}

    def size_multiplier(self, meta: dict, reflection: dict) -> float:
        mult = meta.get("size_multiplier", 1.0)
        if reflection.get("action") == ACTION_REDUCE:
            mult *= 0.5
        elif reflection.get("action") == ACTION_INCREASE:
            mult = min(mult * 1.15, 1.25)
        return round(mult, 2)
