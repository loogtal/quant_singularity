"""Breakout — price vs recent range."""

import numpy as np

from factors.factor_engine import FactorEngine


class BreakoutStrategy:
    def __init__(self):
        self.factors = FactorEngine()

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        closes = df["close"].values
        if len(closes) < 40:
            return {"symbol": symbol, "side": "HOLD", "confidence": 0.0, "regime": market_state.get("regime")}

        high_20 = float(np.max(closes[-20:]))
        low_20 = float(np.min(closes[-20:]))
        price = closes[-1]
        rng = high_20 - low_20
        if rng <= 0:
            return {"symbol": symbol, "side": "HOLD", "confidence": 0.0, "regime": market_state.get("regime")}

        pos_in_range = (price - low_20) / rng
        side = "HOLD"
        confidence = 0.0

        if pos_in_range > 0.92:
            side, confidence = "LONG", pos_in_range
        elif pos_in_range < 0.08:
            side, confidence = "SHORT", 1 - pos_in_range

        vol = market_state.get("volatility", 0.5)
        if vol > 0.85:
            confidence *= 0.7

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(float(confidence), 4),
            "regime": market_state.get("regime"),
        }
