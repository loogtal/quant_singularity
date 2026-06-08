"""Regime classification from price series."""

import numpy as np

from config.constants import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, REGIME_VOLATILE


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

        # Volatile: sustained wide swings (AND not OR) OR a single violent spike
        recent_rets = np.diff(closes[-24:]) / closes[-24:-1]
        ret_std = float(np.std(recent_rets)) if len(recent_rets) >= 5 else 0.0
        change_48h = abs((closes[-1] - closes[-48]) / closes[-48]) if len(closes) >= 48 else 0.0
        # Condition A: sustained volatility + big 48h move
        if ret_std > 0.035 and change_48h > 0.08:
            return REGIME_VOLATILE
        # Condition B: single-bar spike > 5% anywhere in the 24-bar window
        if len(recent_rets) >= 1 and float(np.max(np.abs(recent_rets))) > 0.05:
            return REGIME_VOLATILE

        if price > ema50 > ema200 and change_20 > 0.02:
            return REGIME_BULL
        if price < ema50 < ema200 and change_20 < -0.02:
            return REGIME_BEAR
        return REGIME_SIDEWAYS

    def classify_with_intel(self, closes: np.ndarray, intel: dict) -> str:
        """
        Fuse EMA-based classification with macro market signals.

        The EMA classifier is slow (EMA200 lags by months); market intel signals
        like fear_greed, aggregate_funding, and dominant_trend react in hours.
        Fusing them catches early regime shifts that pure-price classifiers miss.

        Override rules (requires 2+ signals to agree before overriding):
          SIDEWAYS → BEAR  : dominant_trend=bear + (fear_greed<25 OR funding<-0.0002)
          SIDEWAYS → BULL  : dominant_trend=bull + fear_greed>65 + funding>0
          SIDEWAYS → BEAR  : extreme panic (fear_greed<15) overrides alone
        """
        base = self.classify(closes)
        if not intel:
            return base

        fg      = float(intel.get("fear_greed",        50))
        funding = float(intel.get("aggregate_funding", 0.0))
        trend   = intel.get("dominant_trend", "mixed")
        breadth = float(intel.get("breadth", 50))

        # Extreme panic alone is sufficient to call BEAR
        if fg < 15:
            return REGIME_BEAR

        if base == REGIME_SIDEWAYS:
            bear_signals = sum([
                trend == "bear",
                fg < 25,
                funding < -0.0002,
                breadth < 30,
            ])
            if bear_signals >= 2:
                return REGIME_BEAR

            bull_signals = sum([
                trend == "bull",
                fg > 65,
                funding > 0.0001,
                breadth > 65,
            ])
            if bull_signals >= 3:
                return REGIME_BULL

        return base

    def risk_on(self, regime: str, volatility: float) -> bool:
        return regime not in (REGIME_BEAR, REGIME_VOLATILE) and volatility < 0.75
