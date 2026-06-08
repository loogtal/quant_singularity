"""
Correlation Filter — prevent opening highly-correlated positions simultaneously.

BTC and ETH typically move together (ρ > 0.85). Opening both as passive positions
doubles concentration risk without true diversification benefit.

Usage:
    cf = CorrelationFilter()
    safe, reason = cf.is_safe("ETH/USDT:USDT", existing_symbols, market)
    if not safe:
        log.info(f"CORR BLOCK: {reason}")
        continue
"""

import time

import numpy as np

# Symbols known to be structurally correlated — skip expensive API call
_KNOWN_CORR: dict[frozenset, float] = {
    frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT"}): 0.88,
    frozenset({"BTC/USDT:USDT", "BNB/USDT:USDT"}): 0.82,
    frozenset({"ETH/USDT:USDT", "BNB/USDT:USDT"}): 0.80,
}

CORR_THRESHOLD = 0.78   # block if correlation above this
CACHE_TTL = 600         # 10 minutes


class CorrelationFilter:
    """
    Checks whether a new symbol is too correlated with already-open positions.
    Uses a 2-layer approach:
      1. Known pairs table (instant, no API call)
      2. Live 1H return correlation for unknown pairs (cached 10 min)
    """

    def __init__(self):
        self._cache: dict[frozenset, tuple[float, float]] = {}

    # ── public ───────────────────────────────────────────────────────────────

    def is_safe(
        self,
        new_symbol: str,
        existing_symbols: list[str],
        market,
    ) -> tuple[bool, str]:
        """
        Returns (True, "OK") if it is safe to open new_symbol alongside
        existing_symbols, or (False, reason) if it would be too correlated.
        """
        for sym in existing_symbols:
            if sym == new_symbol:
                continue
            corr = self._correlation(new_symbol, sym, market)
            if corr >= CORR_THRESHOLD:
                return False, (
                    f"{new_symbol} ρ={corr:.2f} with {sym} "
                    f"(threshold {CORR_THRESHOLD})"
                )
        return True, "OK"

    # ── internals ─────────────────────────────────────────────────────────────

    def _correlation(self, sym1: str, sym2: str, market) -> float:
        key = frozenset({sym1, sym2})

        # Fast path: known pairs
        if key in _KNOWN_CORR:
            return _KNOWN_CORR[key]

        # Cache hit
        now = time.time()
        if key in self._cache:
            ts, corr = self._cache[key]
            if now - ts < CACHE_TTL:
                return corr

        # Live calculation on 50 hourly returns
        corr = self._live_corr(sym1, sym2, market)
        self._cache[key] = (now, corr)
        return corr

    def _live_corr(self, sym1: str, sym2: str, market) -> float:
        try:
            df1 = market.get_ohlcv_df(sym1, "1h", 55)
            df2 = market.get_ohlcv_df(sym2, "1h", 55)
            r1 = np.diff(df1["close"].values[-51:]) / df1["close"].values[-51:-1]
            r2 = np.diff(df2["close"].values[-51:]) / df2["close"].values[-51:-1]
            n = min(len(r1), len(r2))
            if n < 20:
                return 0.0
            corr = float(np.corrcoef(r1[-n:], r2[-n:])[0, 1])
            return round(corr, 4)
        except Exception:
            return 0.0
