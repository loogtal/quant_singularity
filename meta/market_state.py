"""
Market regime detection from real BTC futures data.
"""

import time

import numpy as np

from config.constants import REF_SYMBOL as BTC_REF_SYMBOL
from data.market_data import MarketData
from meta.regime_classifier import RegimeClassifier


class MarketState:
    REF_SYMBOL = BTC_REF_SYMBOL

    def __init__(self):
        self.market = MarketData()
        self.regime_classifier = RegimeClassifier()
        self.current_state = {
            "timestamp": time.time(),
            "regime": "sideways",
            "volatility": 0.5,
            "risk_on": True,
            "btc_change_24h": 0.0,
        }

    def detect_regime(self, closes: np.ndarray) -> str:
        return self.regime_classifier.classify(closes)

    def calculate_volatility(self, closes: np.ndarray, atr: float) -> float:
        if len(closes) < 2 or closes[-1] <= 0:
            return 0.5
        atr_pct = atr / closes[-1]
        returns = np.diff(closes[-30:]) / closes[-30:-1]
        ret_vol = float(np.std(returns)) if len(returns) else 0.01
        combined = atr_pct * 0.6 + ret_vol * 0.4
        return round(float(np.clip(combined / 0.05, 0.05, 0.99)), 4)

    def update(self) -> dict:
        try:
            df = self.market.get_ohlcv_df(self.REF_SYMBOL, timeframe="1h", limit=250)
            closes = df["close"].values
            atr = self.market.get_atr(self.REF_SYMBOL, timeframe="1h", period=14)

            change_24h = 0.0
            if len(closes) >= 24:
                change_24h = (closes[-1] - closes[-24]) / closes[-24]

            regime = self.detect_regime(closes)
            volatility = self.calculate_volatility(closes, atr)
            risk_on = self.regime_classifier.risk_on(regime, volatility)

            self.current_state = {
                "timestamp": time.time(),
                "regime": regime,
                "volatility": volatility,
                "risk_on": risk_on,
                "btc_change_24h": round(float(change_24h), 4),
            }
        except Exception as e:
            print(f"[MarketState] fallback: {e}")
            self.current_state["timestamp"] = time.time()

        return self.current_state

    def get_market_state(self) -> dict:
        return self.update()
