import time

import numpy as np

from data.binance_client import BinanceClient
from data.indicators import ohlcv_to_df

# Cache TTL per timeframe — longer candles change less often
_TTL_MAP = {
    "1m":  20,
    "3m":  30,
    "5m":  45,
    "15m": 60,
    "30m": 90,
    "1h":  120,
    "4h":  300,
    "1d":  600,
    "1w":  1800,
}
_DEFAULT_TTL = 60


class MarketData:
    def __init__(self):
        self.client = BinanceClient()
        self._cache: dict = {}   # key → (ts, df, candles)

    def _ttl(self, timeframe: str) -> int:
        return _TTL_MAP.get(timeframe, _DEFAULT_TTL)

    def _cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        return f"{symbol}:{timeframe}:{limit}"

    def get_ohlcv_df(self, symbol: str, timeframe: str = "15m", limit: int = 120):
        key = self._cache_key(symbol, timeframe, limit)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[0]) < self._ttl(timeframe):
            return cached[1]
        try:
            candles = self.client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = ohlcv_to_df(candles)
            self._cache[key] = (now, df, candles)
            return df
        except Exception:
            # Return stale cache rather than crashing
            if cached:
                return cached[1]
            raise

    def get_price(self, symbol: str) -> float:
        ticker = self.client.get_ticker(symbol)
        return float(ticker["last"])

    def get_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 100):
        return self.client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def get_atr(self, symbol: str, timeframe: str = "15m", period: int = 14) -> float:
        candles = self.get_ohlcv(symbol, timeframe=timeframe, limit=period + 10)
        highs = np.array([x[2] for x in candles])
        lows = np.array([x[3] for x in candles])
        closes = np.array([x[4] for x in candles])
        trs = []
        for i in range(1, len(candles)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        return float(np.mean(trs[-period:]))
