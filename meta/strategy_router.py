"""Route and ensemble sub-strategies by market regime."""

from config.constants import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    STRATEGY_BREAKOUT,
    STRATEGY_FUNDING,
    STRATEGY_MEAN_REVERSION,
    STRATEGY_MOMENTUM,
    STRATEGY_TREND,
)
from config.settings import USE_FUNDING_ARB
from strategy.trend_following import TrendFollowingStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.momentum import MomentumStrategy
from strategy.funding_arb import FundingArbStrategy


class StrategyRouter:
    REGIME_WEIGHTS = {
        REGIME_BULL: {
            STRATEGY_TREND: 0.40,
            STRATEGY_MOMENTUM: 0.35,
            STRATEGY_BREAKOUT: 0.25,
        },
        REGIME_BEAR: {
            STRATEGY_TREND: 0.35,
            STRATEGY_MEAN_REVERSION: 0.35,
            STRATEGY_BREAKOUT: 0.30,
        },
        REGIME_SIDEWAYS: {
            STRATEGY_MEAN_REVERSION: 0.30,
            STRATEGY_BREAKOUT: 0.30,
            STRATEGY_TREND: 0.25,
            STRATEGY_FUNDING: 0.15,
        },
    }

    def __init__(self):
        self._strategies = {
            STRATEGY_TREND: TrendFollowingStrategy(),
            STRATEGY_MEAN_REVERSION: MeanReversionStrategy(),
            STRATEGY_BREAKOUT: BreakoutStrategy(),
            STRATEGY_MOMENTUM: MomentumStrategy(),
        }
        if USE_FUNDING_ARB:
            self._strategies[STRATEGY_FUNDING] = FundingArbStrategy()
            for regime in self.REGIME_WEIGHTS:
                self.REGIME_WEIGHTS[regime].setdefault(STRATEGY_FUNDING, 0.10)

    def ensemble_signal(
        self,
        symbol: str,
        df,
        market_state: dict,
        alpha: dict,
    ) -> dict:
        regime = market_state.get("regime", REGIME_SIDEWAYS)
        weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS[REGIME_SIDEWAYS])

        long_score = 0.0
        short_score = 0.0
        total_w = 0.0

        for name, weight in weights.items():
            strat = self._strategies.get(name)
            if not strat or weight <= 0:
                continue
            sig = strat.generate(symbol, df, market_state, alpha)
            conf = sig.get("confidence", 0.5) * weight
            total_w += weight
            if sig["side"] == "LONG":
                long_score += conf
            elif sig["side"] == "SHORT":
                short_score += conf

        if total_w <= 0:
            side, confidence = "HOLD", 0.0
        elif long_score > short_score and long_score / total_w > 0.45:
            side = "LONG"
            confidence = long_score / total_w
        elif short_score > long_score and short_score / total_w > 0.45:
            side = "SHORT"
            confidence = short_score / total_w
        else:
            side = "HOLD"
            confidence = max(long_score, short_score) / max(total_w, 1)

        alpha_dir = alpha.get("direction", "HOLD")
        alpha_conf = float(alpha.get("confidence", 0))

        if side != "HOLD" and alpha_dir != "HOLD" and side != alpha_dir:
            side = "HOLD"
            confidence *= 0.7

        # Ensemble unclear but alpha + scanner agree — follow alpha
        if side == "HOLD" and alpha_dir in ("LONG", "SHORT") and alpha_conf >= 0.52:
            side = alpha_dir
            confidence = max(confidence, alpha_conf * 0.95)

        return {
            "symbol": symbol,
            "side": side,
            "direction": side,
            "confidence": round(float(min(confidence, 1.0)), 4),
            "score": round(float(min(confidence, 1.0)), 4),
            "regime": regime,
            "strategies_used": list(weights.keys()),
        }
