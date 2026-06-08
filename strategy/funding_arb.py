"""
Funding Rate Arbitrage — structural income from crowded positions.

How it works:
  Binance perpetual futures settle funding every 8 hours.
  When funding_rate > 0 → longs pay shorts (shorts collect income)
  When funding_rate < 0 → shorts pay longs (longs collect income)

  This is STRUCTURAL EDGE — not speculative. If rate is extreme,
  the opposite side collects guaranteed income every 8 hours.

Thresholds (per 8h settlement):
  |rate| > 0.05%  → moderate signal  (annualised ~54%)
  |rate| > 0.10%  → strong signal    (annualised ~109%)
  |rate| > 0.20%  → extreme signal   (annualised ~219%)

Strategy:
  1. Scan all pairs every 30 min for extreme funding rates
  2. Enter opposite direction to the crowded side
  3. Exit when rate normalises OR after holding through settlement
  4. Position size proportional to rate magnitude (higher rate = bigger size)

Why this works:
  - Funding is paid regardless of price movement
  - Extreme rates attract arbitrageurs → mean-reverts quickly
  - Even if price moves against us slightly, funding income compensates
"""

from __future__ import annotations

import time

from data.funding_data import FundingData

# Funding rate thresholds (per 8h settlement period)
THRESHOLD_MODERATE = 0.0005   # 0.05% per 8h
THRESHOLD_STRONG   = 0.0010   # 0.10% per 8h
THRESHOLD_EXTREME  = 0.0020   # 0.20% per 8h

# Confidence levels per threshold
CONF_MODERATE = 0.68
CONF_STRONG   = 0.78
CONF_EXTREME  = 0.88

# Exit when rate falls below this (position no longer earning meaningful income)
EXIT_THRESHOLD = 0.0002   # 0.02%

_SCAN_CACHE_TTL = 1800   # re-scan universe every 30 min


class FundingArbStrategy:

    def __init__(self) -> None:
        self.funding = FundingData()
        self._universe_cache: list[dict] = []
        self._universe_ts: float = 0.0

    # ── signal generation (per symbol) ───────────────────────────────────────

    def generate(self, symbol: str, df, market_state: dict, alpha: dict) -> dict:
        fd   = self.funding.get_funding(symbol)
        rate = fd["funding_rate"]
        side, confidence = self._signal_from_rate(rate)

        return {
            "symbol":       symbol,
            "side":         side,
            "confidence":   round(confidence, 4),
            "funding_rate": rate,
            "funding_pct":  round(rate * 100, 4),
            "regime":       market_state.get("regime"),
            "signal_mode":  "funding_arb",
        }

    def _signal_from_rate(self, rate: float) -> tuple[str, float]:
        """Returns (side, confidence) from funding rate."""
        abs_rate = abs(rate)
        if abs_rate < THRESHOLD_MODERATE:
            return "HOLD", 0.0
        if abs_rate >= THRESHOLD_EXTREME:
            conf = CONF_EXTREME
        elif abs_rate >= THRESHOLD_STRONG:
            conf = CONF_STRONG
        else:
            conf = CONF_MODERATE
        # Positive rate = longs crowded → SHORT (collect from longs)
        # Negative rate = shorts crowded → LONG (collect from shorts)
        side = "SHORT" if rate > 0 else "LONG"
        return side, conf

    # ── universe scanner ──────────────────────────────────────────────────────

    def scan_universe(self, symbols: list[str], top_n: int = 10) -> list[dict]:
        """
        Scan all symbols for extreme funding rates.
        Returns top_n by |rate| descending.
        Cached for 30 min to avoid hammering the API.
        """
        now = time.time()
        if self._universe_cache and (now - self._universe_ts) < _SCAN_CACHE_TTL:
            return self._universe_cache[:top_n]

        results = []
        for sym in symbols:
            try:
                fd   = self.funding.get_funding(sym)
                rate = fd["funding_rate"]
                if abs(rate) >= THRESHOLD_MODERATE:
                    side, conf = self._signal_from_rate(rate)
                    results.append({
                        "symbol":       sym,
                        "funding_rate": rate,
                        "funding_pct":  round(rate * 100, 4),
                        "side":         side,
                        "confidence":   conf,
                        "abs_rate":     abs(rate),
                        "tier":         self._tier(abs(rate)),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x["abs_rate"], reverse=True)
        self._universe_cache = results
        self._universe_ts    = now
        return results[:top_n]

    @staticmethod
    def _tier(abs_rate: float) -> str:
        if abs_rate >= THRESHOLD_EXTREME:
            return "extreme"
        if abs_rate >= THRESHOLD_STRONG:
            return "strong"
        return "moderate"

    # ── position management helpers ───────────────────────────────────────────

    def should_exit(self, symbol: str) -> bool:
        """
        Returns True when funding rate has normalised enough to exit.
        Call this every cycle for open funding_arb positions.
        """
        try:
            fd   = self.funding.get_funding(symbol)
            rate = fd["funding_rate"]
            return abs(rate) < EXIT_THRESHOLD
        except Exception:
            return False

    # ── scanner score adjustment (for CoinScanner) ───────────────────────────

    def score_adjustment(self, symbol: str) -> float:
        """Score bonus for CoinScanner: higher rate = bigger bonus."""
        try:
            rate = self.funding.get_funding(symbol)["funding_rate"]
            abs_r = abs(rate)
            if abs_r >= THRESHOLD_EXTREME:
                return 0.10
            if abs_r >= THRESHOLD_STRONG:
                return 0.07
            if abs_r >= THRESHOLD_MODERATE:
                return 0.04
        except Exception:
            pass
        return 0.0
