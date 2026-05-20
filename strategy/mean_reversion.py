"""Mean reversion — RSI extremes in range-bound markets."""

import numpy as np
import pandas as pd
import ta

from factors.factor_engine import FactorEngine


class MeanReversionStrategy:
    def __init__(self):
        self.factors = FactorEngine()

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        closes = df["close"].values
        regime = market_state.get("regime", "sideways")

        side = "HOLD"
        confidence = 0.0

        if len(closes) >= 20:
            rsi = ta.momentum.RSIIndicator(pd.Series(closes), window=14).rsi().iloc[-1]
            if not np.isnan(rsi):
                if rsi < 32 and regime != "bear":
                    side, confidence = "LONG", (35 - rsi) / 35
                elif rsi > 68 and regime != "bull":
                    side, confidence = "SHORT", (rsi - 65) / 35

        if regime == "bull" and side == "SHORT":
            side, confidence = "HOLD", 0.0
        if regime == "bear" and side == "LONG":
            side, confidence = "HOLD", 0.0

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(float(np.clip(confidence, 0, 1)), 4),
            "regime": regime,
        }
