"""
Strategy engine — ensembles sub-strategies via StrategyRouter,
then applies market and meta filters.
"""

from data.market_data import MarketData
from meta.strategy_router import StrategyRouter


class StrategyEngine:
    def __init__(self):
        self.market = MarketData()
        self.router = StrategyRouter()
        self.last_signal = None

    def generate_signal(self, alpha, market_state=None, meta=None, **kwargs):
        symbol = alpha.get("symbol")
        regime = alpha.get("regime", "sideways")

        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            signal = self.router.ensemble_signal(symbol, df, market_state or {}, alpha)
        except Exception as e:
            print(f"[Strategy] {symbol} error: {e}")
            signal = {
                "symbol": symbol,
                "side": "HOLD",
                "direction": "HOLD",
                "confidence": 0.0,
                "score": 0.0,
                "regime": regime,
            }

        direction = signal["side"]
        confidence = float(signal.get("confidence", 0))

        if confidence < 0.48:
            direction = "HOLD"

        if market_state:
            volatility = market_state.get("volatility", 0)
            regime_state = market_state.get("regime", "sideways")

            if volatility > 0.90:
                direction = "HOLD"
            if regime_state == "bear" and direction == "LONG":
                direction = "HOLD"
            if regime_state == "bull" and direction == "SHORT":
                direction = "HOLD"

        if meta:
            mode = meta.get("mode", "neutral")
            if mode == "defensive" and direction == "LONG":
                confidence *= 0.85
            elif mode == "aggressive":
                confidence = min(confidence * 1.08, 1.0)

        signal = {
            "symbol": symbol,
            "side": direction,
            "direction": direction,
            "score": round(confidence, 4),
            "confidence": round(confidence, 4),
            "regime": regime,
        }
        self.last_signal = signal
        return signal

    def get_last_signal(self):
        return self.last_signal
