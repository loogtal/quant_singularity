"""
Meta controller — adjusts aggression from market + portfolio performance.
"""


class MetaBrain:
    def evaluate(self, market_state: dict, portfolio_metrics: dict | None = None) -> dict:
        regime = market_state.get("regime", "sideways")
        volatility = market_state.get("volatility", 0.5)
        risk_on = market_state.get("risk_on", True)

        mode = "neutral"
        size_mult = 1.0
        min_confidence = 0.55 if regime == "sideways" else 0.62

        if regime == "bull" and risk_on and volatility < 0.7:
            mode = "aggressive"
            size_mult = 1.15
            min_confidence = 0.58
        elif regime == "bear" or not risk_on:
            mode = "defensive"
            size_mult = 0.65
            min_confidence = 0.72
        elif volatility > 0.75:
            mode = "cautious"
            size_mult = 0.5
            min_confidence = 0.75

        if portfolio_metrics:
            dd = portfolio_metrics.get("drawdown", 0)
            daily_pnl_pct = portfolio_metrics.get("daily_pnl_pct", 0)

            if dd > 0.08:
                mode = "defensive"
                size_mult *= 0.5
                min_confidence = max(min_confidence, 0.75)
            elif daily_pnl_pct < -0.02:
                mode = "recovery"
                size_mult *= 0.4
                min_confidence = 0.78
            elif daily_pnl_pct > 0.01 and dd < 0.05:
                size_mult = min(size_mult * 1.1, 1.2)

        return {
            "mode": mode,
            "size_multiplier": round(size_mult, 2),
            "min_confidence": round(min_confidence, 2),
        }
