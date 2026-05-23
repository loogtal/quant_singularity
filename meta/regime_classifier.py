"""Regime classification from price series — EMA + ADX + BB Width."""

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

    @staticmethod
    def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average Directional Index — measures trend strength (0–100)."""
        if len(closes) < period + 2:
            return 0.0
        n = len(closes)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)
        for i in range(1, n):
            h_diff = highs[i] - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            plus_dm[i] = max(h_diff, 0) if h_diff > l_diff else 0
            minus_dm[i] = max(l_diff, 0) if l_diff > h_diff else 0
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

        alpha = 1.0 / period
        atr_s = tr[1]
        plus_di_s = plus_dm[1]
        minus_di_s = minus_dm[1]
        dx_vals = []
        for i in range(2, n):
            atr_s = atr_s * (1 - alpha) + tr[i] * alpha
            plus_di_s = plus_di_s * (1 - alpha) + plus_dm[i] * alpha
            minus_di_s = minus_di_s * (1 - alpha) + minus_dm[i] * alpha
            if atr_s > 0:
                pdi = plus_di_s / atr_s * 100
                mdi = minus_di_s / atr_s * 100
                denom = pdi + mdi
                dx = abs(pdi - mdi) / denom * 100 if denom > 0 else 0
                dx_vals.append(dx)
        if len(dx_vals) < period:
            return float(np.mean(dx_vals)) if dx_vals else 0.0
        return float(np.mean(dx_vals[-period:]))

    @staticmethod
    def _bb_width(closes: np.ndarray, period: int = 20) -> float:
        """Bollinger Band width as % of middle band — measures volatility expansion."""
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        mid = float(np.mean(window))
        if mid <= 0:
            return 0.0
        std = float(np.std(window))
        return (2 * std * 2) / mid

    def classify(self, closes: np.ndarray, highs: np.ndarray | None = None, lows: np.ndarray | None = None) -> str:
        if len(closes) < 100:
            return REGIME_SIDEWAYS

        price = closes[-1]
        ema50 = self._ema(closes, 50)
        ema200 = self._ema(closes, min(200, len(closes)))
        change_20 = (closes[-1] - closes[-20]) / closes[-20]

        adx = 0.0
        if highs is not None and lows is not None:
            adx = self._adx(highs, lows, closes)

        bb_w = self._bb_width(closes)

        ema_bull = price > ema50 > ema200
        ema_bear = price < ema50 < ema200
        strong_trend = adx > 25
        expanding_vol = bb_w > 0.04

        if ema_bull and change_20 > 0.01 and (strong_trend or expanding_vol):
            return REGIME_BULL
        if ema_bear and change_20 < -0.01 and (strong_trend or expanding_vol):
            return REGIME_BEAR

        if ema_bull and change_20 > 0.02:
            return REGIME_BULL
        if ema_bear and change_20 < -0.02:
            return REGIME_BEAR

        return REGIME_SIDEWAYS

    def trend_strength(self, closes: np.ndarray, highs: np.ndarray | None = None, lows: np.ndarray | None = None) -> float:
        """0–1 score of how strong the current trend is."""
        if len(closes) < 50:
            return 0.0
        adx = 0.0
        if highs is not None and lows is not None:
            adx = self._adx(highs, lows, closes)
        ema_diff = abs(self._ema(closes, 20) - self._ema(closes, 50)) / closes[-1]
        return float(np.clip(adx / 50 * 0.6 + ema_diff * 20 * 0.4, 0, 1))

    def risk_on(self, regime: str, volatility: float) -> bool:
        return regime != REGIME_BEAR and volatility < 0.75
