import numpy as np

from config.settings import (
    MAX_DRAWDOWN,
    MAX_POSITION_FRACTION,
    MIN_CONFIDENCE,
    RISK_PER_TRADE,
)
from data.market_data import MarketData


class RiskEngine:

    def __init__(self):

        self.market_data = MarketData()

        self.max_drawdown = MAX_DRAWDOWN

        self.max_position_fraction = MAX_POSITION_FRACTION

        self.base_risk_per_trade = RISK_PER_TRADE

    # ==========================================================
    # MAIN
    # ==========================================================

    def evaluate(
        self,
        signal,
        portfolio,
        market_state,
        min_confidence=None,
    ):

        side = signal["side"]

        confidence = signal["confidence"]

        # ======================================================
        # NO SIGNAL
        # ======================================================

        if side == "HOLD":

            return {

                "allow_trade": False,

                "reason": "NO SIGNAL"

            }

        # ======================================================
        # DRAWDOWN PROTECTION
        # ======================================================

        drawdown = portfolio.current_drawdown()

        if drawdown >= self.max_drawdown:

            return {

                "allow_trade": False,

                "reason": "MAX DRAWDOWN"

            }

        # ======================================================
        # MARKET FILTER
        # ======================================================

        volatility = market_state["volatility"]

        regime = market_state["regime"]

        # Choppy sideways + very high vol only
        if regime == "sideways" and volatility > 0.78:

            return {

                "allow_trade": False,

                "reason": "CHOPPY MARKET"

            }

        # ======================================================
        # CONFIDENCE FILTER
        # ======================================================

        threshold = min_confidence if min_confidence is not None else MIN_CONFIDENCE

        if confidence < threshold:

            return {

                "allow_trade": False,

                "reason": f"LOW CONFIDENCE ({confidence:.2f} < {threshold:.2f})"

            }

        # ======================================================
        # ATR
        # ======================================================

        symbol = signal["symbol"]

        atr = self.market_data.get_atr(symbol)

        if atr <= 0:

            return {

                "allow_trade": False,

                "reason": "INVALID ATR"

            }

        # ======================================================
        # RISK MODE
        # ======================================================

        risk_per_trade = self.base_risk_per_trade

        risk_level = "LOW"

        if confidence > 0.85:

            risk_per_trade *= 1.5

            risk_level = "HIGH"

        elif confidence > 0.75:

            risk_per_trade *= 1.2

            risk_level = "NORMAL"

        # volatility สูง = ลด risk
        if volatility > 0.7:

            risk_per_trade *= 0.5

        # ======================================================
        # PRICE
        # ======================================================

        price = self.market_data.get_price(symbol)

        # ======================================================
        # STOP DISTANCE
        # ======================================================

        stop_distance = atr * 2.0

        # ======================================================
        # ACCOUNT RISK
        # ======================================================

        equity = portfolio.equity

        risk_amount = equity * risk_per_trade

        raw_size = risk_amount / stop_distance

        # ======================================================
        # MAX POSITION CAP
        # ======================================================

        max_position_value = equity * self.max_position_fraction

        max_size = max_position_value / price

        final_size = min(raw_size, max_size)

        # ======================================================
        # SAFETY
        # ======================================================

        if final_size <= 0:

            return {

                "allow_trade": False,

                "reason": "INVALID SIZE"

            }

        final_size = round(float(final_size), 4)

        return {

            "allow_trade": True,

            "risk_level": risk_level,

            "position_size": final_size,

            "atr": round(float(atr), 4),

            "stop_distance": round(float(stop_distance), 4)

        }