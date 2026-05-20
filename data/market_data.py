import numpy as np

from data.binance_client import BinanceClient
from data.indicators import ohlcv_to_df


class MarketData:
    def __init__(self):
        self.client = BinanceClient()
        self._cache: dict = {}
        self._cache_ttl = 30

    def _cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        return f"{symbol}:{timeframe}:{limit}"

    def get_ohlcv_df(self, symbol: str, timeframe: str = "15m", limit: int = 120):
        key = self._cache_key(symbol, timeframe, limit)
        now = __import__("time").time()
        if key in self._cache:
            ts, df = self._cache[key]
            if now - ts < self._cache_ttl:
                return df
        candles = self.client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = ohlcv_to_df(candles)
        self._cache[key] = (now, df)
        return df

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
