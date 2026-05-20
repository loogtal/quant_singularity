"""Regime classification from price series."""

import numpy as np

from config.constants import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


class RegimeClassifier:
    @staticmethod
    def _ema(values: np.ndarray, period: int) -> float:
        if len(values) < period:
            return float(values[-1])
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return float(np.convolve(values[-period:], weights, mode="valid")[-1])

    def classify(self, closes: np.ndarray) -> str:
        if len(closes) < 100:
            return REGIME_SIDEWAYS

        price = closes[-1]
        ema50 = self._ema(closes, 50)
        ema200 = self._ema(closes, min(200, len(closes)))
        change_20 = (closes[-1] - closes[-20]) / closes[-20]

        if price > ema50 > ema200 and change_20 > 0.02:
            return REGIME_BULL
        if price < ema50 < ema200 and change_20 < -0.02:
            return REGIME_BEAR
        return REGIME_SIDEWAYS

    def risk_on(self, regime: str, volatility: float) -> bool:
        return regime != REGIME_BEAR and volatility < 0.75
