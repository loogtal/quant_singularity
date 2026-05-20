"""
Autonomous coin selection — ranks Binance USDT futures by
momentum, volume, trend, and liquidity using real market data.
"""

from typing import Any

import numpy as np

from config.settings import MIN_24H_VOLUME_USDT, SCAN_TOP_N, USE_FUNDING_ARB
from data.binance_client import BinanceClient
from data.market_data import MarketData
from factors.alpha_factors import AlphaFactors
from strategy.funding_arb import FundingArbStrategy


class CoinScanner:
    def __init__(self):
        self.client = BinanceClient()
        self.market = MarketData()
        self.factors = AlphaFactors()
        self.funding_arb = FundingArbStrategy() if USE_FUNDING_ARB else None

    def _normalize(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.5
        return float(np.clip((value - low) / (high - low), 0, 1))

    def score_symbol(self, symbol: str) -> dict[str, Any] | None:
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            if len(df) < 50:
                return None

            closes = df["close"].values
            volumes = df["volume"].values

            momentum = self.factors.momentum_factor(closes)
            vol_factor = self.factors.volatility_factor(closes)
            trend = self.factors.trend_strength(closes)

            vol_usdt = float(volumes[-1] * closes[-1])
            avg_vol = float(np.mean(volumes[-20:] * closes[-20:]))

            mom_score = self._normalize(momentum, -0.05, 0.05)
            trend_score = self._normalize(trend, -0.03, 0.03)
            volume_score = self._normalize(avg_vol, 0, max(vol_usdt * 10, 1))
            vol_penalty = self._normalize(vol_factor, 0, 0.03)

            # Higher momentum + trend + volume; lower raw volatility
            score = round(
                mom_score * 0.35
                + trend_score * 0.35
                + volume_score * 0.20
                + (1 - vol_penalty) * 0.10,
                4,
            )
            if self.funding_arb:
                score = round(min(1.0, max(0.0, score + self.funding_arb.score_adjustment(symbol))), 4)

            return {
                "symbol": symbol,
                "score": score,
                "momentum": round(float(momentum), 4),
                "volume": round(volume_score, 4),
                "trend": round(float(trend), 4),
                "alpha": round(score, 4),
                "volatility": round(float(vol_factor), 4),
                "liquidity": round(volume_score, 4),
            }
        except Exception:
            return None

    def scan_market(self) -> list[dict[str, Any]]:
        symbols = self.client.list_usdt_futures()
        if not symbols:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

        coins = []
        for symbol in symbols[:SCAN_TOP_N]:
            data = self.score_symbol(symbol)
            if data and data["score"] > 0.35:
                coins.append(data)

        coins.sort(key=lambda x: x["score"], reverse=True)
        return coins
