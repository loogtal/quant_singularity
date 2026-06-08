"""Binance USDT-M perpetual funding rates."""

import time
from typing import Any

from data.binance_client import BinanceClient


class FundingData:
    # Class-level cache — shared across ALL instances so new FundingData() calls
    # never refetch within the TTL window (funding rates update every 8 hours)
    CACHE_TTL = 3600   # 1 hour — safe since Binance funding settles every 8h
    _cache: dict[str, tuple[float, dict]] = {}
    _last_warn: dict[str, float] = {}   # rate-limit log spam guard

    def __init__(self):
        self.client = BinanceClient()

    def get_funding(self, symbol: str) -> dict[str, Any]:
        now = time.time()
        cached = FundingData._cache.get(symbol)
        if cached and (now - cached[0]) < self.CACHE_TTL:
            return cached[1]

        ex = self.client.get_exchange()
        rate = 0.0
        if ex:
            try:
                info = ex.fetch_funding_rate(symbol)
                rate = float(info.get("fundingRate") or 0)
            except Exception:
                # Silently fall back to last cached value or 0
                if cached:
                    return cached[1]

        data = {
            "symbol": symbol,
            "funding_rate": rate,
            "funding_pct": round(rate * 100, 4),
            "bias": self._bias_from_rate(rate),
        }
        FundingData._cache[symbol] = (now, data)
        return data

    @staticmethod
    def _bias_from_rate(rate: float) -> str:
        if rate <= -0.0003:
            return "LONG"
        if rate >= 0.0005:
            return "SHORT"
        return "NEUTRAL"

    def scan_extremes(self, symbols: list[str], top_n: int = 10) -> list[dict]:
        rows = []
        for sym in symbols[:30]:
            try:
                rows.append(self.get_funding(sym))
            except Exception:
                continue
        rows.sort(key=lambda x: abs(x["funding_rate"]), reverse=True)
        return rows[:top_n]
