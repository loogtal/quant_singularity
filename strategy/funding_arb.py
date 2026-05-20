"""Funding rate arbitrage bias — fade crowded funding."""

from data.funding_data import FundingData


class FundingArbStrategy:
    EXTREME_NEG = -0.0003
    EXTREME_POS = 0.0005

    def __init__(self):
        self.funding = FundingData()

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        fd = self.funding.get_funding(symbol)
        rate = fd["funding_rate"]
        side = "HOLD"
        confidence = 0.0

        if rate <= self.EXTREME_NEG:
            side, confidence = "LONG", min(abs(rate) / 0.001, 0.85)
        elif rate >= self.EXTREME_POS:
            side, confidence = "SHORT", min(rate / 0.001, 0.85)

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(confidence, 4),
            "funding_rate": rate,
            "regime": market_state.get("regime"),
        }

    def score_adjustment(self, symbol: str) -> float:
        """Scanner bonus: -0.05 to +0.05."""
        fd = self.funding.get_funding(symbol)
        rate = fd["funding_rate"]
        if rate <= self.EXTREME_NEG:
            return 0.05
        if rate >= self.EXTREME_POS:
            return -0.03
        return 0.0
