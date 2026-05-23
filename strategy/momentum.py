"""Momentum strategy — ride short-term impulse."""


from factors.factor_engine import FactorEngine


class MomentumStrategy:
    def __init__(self):
        self.factors = FactorEngine()

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        closes = df["close"].values
        f = self.factors.compute(closes, df["volume"].values if "volume" in df else None)
        mom_n = self.factors.normalize(f["momentum"], -0.08, 0.08)

        side = "HOLD"
        if mom_n > 0.62 and f["momentum"] > 0.01:
            side = "LONG"
        elif mom_n < 0.38 and f["momentum"] < -0.01:
            side = "SHORT"

        confidence = abs(mom_n - 0.5) * 2
        if market_state.get("volatility", 0) > 0.8:
            confidence *= 0.8

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(float(confidence), 4),
            "regime": market_state.get("regime"),
        }
