"""Trend following — EMA alignment + slope."""


from factors.factor_engine import FactorEngine


class TrendFollowingStrategy:
    def __init__(self):
        self.factors = FactorEngine()

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        closes = df["close"].values
        f = self.factors.compute(closes)
        trend_n = self.factors.normalize(f["trend"], -0.05, 0.05)

        side = "HOLD"
        if trend_n > 0.58 and f["trend"] > 0:
            side = "LONG"
        elif trend_n < 0.42 and f["trend"] < 0:
            side = "SHORT"

        confidence = abs(trend_n - 0.5) * 2
        regime = market_state.get("regime", "sideways")
        if regime == "bear" and side == "LONG":
            side, confidence = "HOLD", confidence * 0.5
        if regime == "bull" and side == "SHORT":
            side, confidence = "HOLD", confidence * 0.5

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(float(confidence), 4),
            "regime": regime,
        }
