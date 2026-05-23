"""Higher-timeframe trend confirmation — reduces false signals on 15m."""

import numpy as np

from data.market_data import MarketData


class HTFFilter:
    """Check if 1h trend agrees with the 15m signal direction."""

    def __init__(self):
        self.market = MarketData()

    def _ema(self, values: np.ndarray, period: int) -> float:
        if len(values) < period:
            return float(values[-1])
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return float(np.convolve(values[-period:], weights, mode="valid")[-1])

    def htf_trend(self, symbol: str) -> str:
        """Returns LONG / SHORT / NEUTRAL based on 1h EMA alignment."""
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="1h", limit=60)
            closes = df["close"].values
            if len(closes) < 30:
                return "NEUTRAL"

            ema8 = self._ema(closes, 8)
            ema21 = self._ema(closes, 21)
            price = closes[-1]

            if price > ema8 > ema21:
                return "LONG"
            if price < ema8 < ema21:
                return "SHORT"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def confirms(self, symbol: str, direction: str) -> bool:
        """True if the 1h trend matches or is neutral."""
        htf = self.htf_trend(symbol)
        if htf == "NEUTRAL":
            return True
        return htf == direction

    def confidence_adjust(self, symbol: str, direction: str, confidence: float) -> float:
        """Boost confidence if HTF agrees, penalize if it disagrees."""
        htf = self.htf_trend(symbol)
        if htf == direction:
            return min(confidence * 1.15, 1.0)
        if htf == "NEUTRAL":
            return confidence
        return confidence * 0.6
