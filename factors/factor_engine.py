"""Unified factor computation for scanner, alpha, and strategies."""


import numpy as np

from factors.alpha_factors import AlphaFactors


class FactorEngine:
    def __init__(self):
        self._factors = AlphaFactors()

    def compute(self, closes: np.ndarray, volumes: np.ndarray | None = None) -> dict[str, float]:
        momentum = self._factors.momentum_factor(closes)
        volatility = self._factors.volatility_factor(closes)
        trend = self._factors.trend_strength(closes)

        vol_usdt = 0.0
        avg_vol = 0.0
        if volumes is not None and len(volumes) >= 20:
            vol_usdt = float(volumes[-1] * closes[-1])
            avg_vol = float(np.mean(volumes[-20:] * closes[-20:]))

        return {
            "momentum": float(momentum),
            "volatility": float(volatility),
            "trend": float(trend),
            "volume_usdt": vol_usdt,
            "avg_volume_usdt": avg_vol,
        }

    @staticmethod
    def normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.5
        return float(np.clip((value - low) / (high - low), 0, 1))

    def composite_score(self, factors: dict[str, float]) -> float:
        mom = self.normalize(factors["momentum"], -0.05, 0.05)
        trend = self.normalize(factors["trend"], -0.03, 0.03)
        vol_penalty = self.normalize(factors["volatility"], 0, 0.03)
        vol_score = self.normalize(
            factors.get("avg_volume_usdt", 0),
            0,
            max(factors.get("volume_usdt", 1) * 10, 1),
        )
        return round(
            mom * 0.35 + trend * 0.35 + vol_score * 0.20 + (1 - vol_penalty) * 0.10,
            4,
        )

    def direction_from_factors(self, factors: dict[str, float]) -> str:
        m, t = factors["momentum"], factors["trend"]
        if m > 0.005 and t > 0:
            return "LONG"
        if m < -0.005 and t < 0:
            return "SHORT"
        return "HOLD"
