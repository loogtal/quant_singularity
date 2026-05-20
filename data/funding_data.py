"""Binance USDT-M perpetual funding rates."""

import time
from typing import Any

from data.binance_client import BinanceClient


class FundingData:
    CACHE_TTL = 300  # 5 min

    def __init__(self):
        self.client = BinanceClient()
        self._cache: dict[str, tuple[float, dict]] = {}

    def get_funding(self, symbol: str) -> dict[str, Any]:
        now = time.time()
        if symbol in self._cache:
            ts, data = self._cache[symbol]
            if now - ts < self.CACHE_TTL:
                return data

        ex = self.client.get_exchange()
        rate = 0.0
        if ex:
            try:
                info = ex.fetch_funding_rate(symbol)
                rate = float(info.get("fundingRate") or 0)
            except Exception as e:
                print(f"[Funding] {symbol}: {e}")

        data = {
            "symbol": symbol,
            "funding_rate": rate,
            "funding_pct": round(rate * 100, 4),
            "bias": self._bias_from_rate(rate),
        }
        self._cache[symbol] = (now, data)
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
