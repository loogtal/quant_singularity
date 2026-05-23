"""Correlation filter — prevent opening highly correlated positions."""

import numpy as np

from data.market_data import MarketData

CORR_THRESHOLD = 0.85


class CorrelationFilter:
    def __init__(self):
        self.market = MarketData()

    def _get_returns(self, symbol: str, bars: int = 50) -> np.ndarray | None:
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=bars + 1)
            closes = df["close"].values
            if len(closes) < 20:
                return None
            return np.diff(closes) / closes[:-1]
        except Exception:
            return None

    def correlation(self, sym_a: str, sym_b: str) -> float:
        """Pearson correlation between two symbols' returns."""
        ret_a = self._get_returns(sym_a)
        ret_b = self._get_returns(sym_b)
        if ret_a is None or ret_b is None:
            return 0.0
        n = min(len(ret_a), len(ret_b))
        if n < 10:
            return 0.0
        corr = float(np.corrcoef(ret_a[-n:], ret_b[-n:])[0, 1])
        return corr if not np.isnan(corr) else 0.0

    def is_too_correlated(self, candidate: str, open_symbols: list[str]) -> bool:
        """True if candidate is highly correlated with any open position."""
        for sym in open_symbols:
            corr = self.correlation(candidate, sym)
            if abs(corr) >= CORR_THRESHOLD:
                return True
        return False
