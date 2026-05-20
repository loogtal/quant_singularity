"""
Factor-based alpha — no random signals.
Combines momentum, trend, RSI, and regime alignment.
"""

from typing import Any

import numpy as np
import ta

from config.settings import USE_ML
from data.market_data import MarketData
from factors.alpha_factors import AlphaFactors
from models.predictor import MLPredictor


class AlphaEngine:
    def __init__(self):
        self.market = MarketData()
        self.factors = AlphaFactors()
        self.ml = MLPredictor() if USE_ML else None
        self.min_confidence = 0.55

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.5
        return float(np.clip((value - low) / (high - low), 0, 1))

    def _rsi_signal(self, closes: np.ndarray) -> float:
        if len(closes) < 20:
            return 0.5
        import pandas as pd
        series = pd.Series(closes)
        rsi = ta.momentum.RSIIndicator(series, window=14).rsi().iloc[-1]
        if np.isnan(rsi):
            return 0.5
        if rsi < 30:
            return 0.75
        if rsi > 70:
            return 0.25
        return 0.5 + (50 - rsi) / 100

    def generate_alpha(
        self,
        symbol: str,
        market_state: dict[str, Any],
        market_data: dict[str, Any] | None = None,
        scanner_score: float | None = None,
    ) -> dict[str, Any]:
        regime = market_state.get("regime", "sideways")
        volatility = market_state.get("volatility", 0.5)

        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            closes = df["close"].values

            momentum = self.factors.momentum_factor(closes)
            vol_f = self.factors.volatility_factor(closes)
            trend = self.factors.trend_strength(closes)

            factors = {
                "momentum": float(np.clip(momentum * 10, -1, 1)),
                "trend": float(np.clip(trend * 20, -1, 1)),
                "volatility": float(np.clip(vol_f * 50, 0, 1)),
            }

            # Same 0–1 scale as CoinScanner so confidence matches ranking
            mom_score = self._normalize(momentum, -0.05, 0.05)
            trend_score = self._normalize(trend, -0.03, 0.03)
            rsi_adj = self._rsi_signal(closes)
            score = mom_score * 0.35 + trend_score * 0.35 + rsi_adj * 0.30

            # Direction from factor alignment
            if momentum > 0.005 and trend > 0:
                direction = "LONG"
            elif momentum < -0.005 and trend < 0:
                direction = "SHORT"
            else:
                direction = "HOLD"

            # Regime filter
            if regime == "bull" and direction == "SHORT":
                score *= 0.7
                if score < 0.55:
                    direction = "HOLD"
            elif regime == "bear" and direction == "LONG":
                score *= 0.7
                if score < 0.55:
                    direction = "HOLD"

            if volatility > 0.8:
                score *= 0.85

            # Align with scanner when direction is clear
            if scanner_score is not None and direction != "HOLD":
                score = max(score, scanner_score * 0.92)

            ml_info = {}
            if self.ml is not None:
                vols = df["volume"].values if "volume" in df else None
                ml_pred = self.ml.predict(closes, vols)
                ml_info = ml_pred
                if ml_pred["direction"] != "HOLD":
                    if direction == "HOLD":
                        direction = ml_pred["direction"]
                    if direction == ml_pred["direction"]:
                        score = 0.55 * score + 0.45 * ml_pred["confidence"]
                    else:
                        score *= 0.85

            confidence = round(float(np.clip(score, 0, 1)), 2)

            return {
                "symbol": symbol,
                "score": confidence,
                "confidence": confidence,
                "direction": direction,
                "regime": regime,
                "factors": factors,
                "ml": ml_info,
            }

        except Exception as e:
            print(f"[Alpha] {symbol} error: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "confidence": 0.0,
                "direction": "HOLD",
                "regime": regime,
            }
