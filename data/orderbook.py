"""
Order book imbalance signals for active intraday trading.

Fetches the top N bid/ask levels from Binance and computes:

  imbalance_ratio  — bid_qty / (bid_qty + ask_qty) within ±DEPTH_PCT of mid price
                     > 0.60 = buy pressure  (bullish)
                     < 0.40 = sell pressure (bearish)
                     0.40-0.60 = balanced

  wall_side        — "bid" if a large bid wall is within WALL_DIST_PCT of price
                     "ask" if a large ask wall is within WALL_DIST_PCT of price
                     None if no wall detected

  spread_bps       — bid/ask spread in basis points (quality metric)

These signals are consumed by ActiveStrategy to:
  1. Boost confidence when order book pressure agrees with signal direction
  2. Reduce confidence (or veto) when pressure opposes direction
  3. Skip entry when spread is too wide (illiquid moment)
"""

from __future__ import annotations

import time
from typing import Optional

DEPTH_PCT    = 0.005   # look at bids/asks within 0.5% of mid price
WALL_DIST_PCT = 0.003  # wall must be within 0.3% of current price
WALL_MULT    = 5.0     # order is a "wall" if it's WALL_MULT × median order size
MAX_SPREAD_BPS = 20.0  # skip entry if spread > 20 bps
CACHE_TTL    = 10      # seconds — order book is stale after 10s


class OrderBookSignal:
    """
    Fetches and caches order book data, computes imbalance and wall signals.
    Thread-safe for use inside the scanner thread pool.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}   # symbol → {"ts", "data"}

    def _fetch(self, exchange, symbol: str) -> Optional[dict]:
        if exchange is None:
            return None
        try:
            ob = exchange.fetch_order_book(symbol, limit=20)
            return ob
        except Exception:
            return None

    def get_signal(self, exchange, symbol: str) -> dict:
        """
        Returns imbalance dict.  Uses cached result if fresh enough.
        Falls back to neutral (0.50) when data unavailable.
        """
        neutral = {
            "imbalance_ratio": 0.50,
            "imbalance_label": "balanced",
            "wall_side":       None,
            "spread_bps":      0.0,
            "available":       False,
        }

        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached["ts"]) < CACHE_TTL:
            return cached["data"]

        raw = self._fetch(exchange, symbol)
        if not raw:
            return neutral

        bids = raw.get("bids", [])  # [[price, size], ...]
        asks = raw.get("asks", [])
        if not bids or not asks:
            return neutral

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid      = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return neutral

        spread_bps = round((best_ask - best_bid) / mid * 10000, 2)

        # Aggregate qty within DEPTH_PCT of mid
        bid_cutoff = mid * (1 - DEPTH_PCT)
        ask_cutoff = mid * (1 + DEPTH_PCT)

        bid_qty = sum(float(s) for p, s in bids if float(p) >= bid_cutoff)
        ask_qty = sum(float(s) for p, s in asks if float(p) <= ask_cutoff)

        total = bid_qty + ask_qty
        if total <= 0:
            return neutral

        imbalance = round(bid_qty / total, 4)
        if imbalance > 0.60:
            label = "buy_pressure"
        elif imbalance < 0.40:
            label = "sell_pressure"
        else:
            label = "balanced"

        # Wall detection
        wall_side = None
        if bids:
            all_bid_sizes = [float(s) for _, s in bids]
            median_bid    = float(_median(all_bid_sizes)) if all_bid_sizes else 1.0
            for price, size in bids:
                p, s = float(price), float(size)
                if abs(p - mid) / mid <= WALL_DIST_PCT and s >= median_bid * WALL_MULT:
                    wall_side = "bid"
                    break
        if asks and wall_side is None:
            all_ask_sizes = [float(s) for _, s in asks]
            median_ask    = float(_median(all_ask_sizes)) if all_ask_sizes else 1.0
            for price, size in asks:
                p, s = float(price), float(size)
                if abs(p - mid) / mid <= WALL_DIST_PCT and s >= median_ask * WALL_MULT:
                    wall_side = "ask"
                    break

        result = {
            "imbalance_ratio": imbalance,
            "imbalance_label": label,
            "wall_side":       wall_side,
            "spread_bps":      spread_bps,
            "available":       True,
        }
        self._cache[symbol] = {"ts": now, "data": result}
        return result

    def confidence_adjustment(self, signal: dict, trade_side: str) -> float:
        """
        Returns a confidence delta to add/subtract based on order book state.

          trade_side = "LONG"  or "SHORT"

        Buy pressure on LONG → +0.05
        Sell pressure on SHORT → +0.05
        Opposite pressure → −0.08
        Wall opposing direction → −0.05
        Spread too wide → −0.10 (discourages entry)
        """
        if not signal.get("available"):
            return 0.0

        delta = 0.0
        label = signal.get("imbalance_label", "balanced")
        wall  = signal.get("wall_side")
        spread = signal.get("spread_bps", 0.0)

        if spread > MAX_SPREAD_BPS:
            delta -= 0.10

        if trade_side == "LONG":
            if label == "buy_pressure":
                delta += 0.05
            elif label == "sell_pressure":
                delta -= 0.08
            if wall == "ask":
                delta -= 0.05   # ask wall above → resistance
        elif trade_side == "SHORT":
            if label == "sell_pressure":
                delta += 0.05
            elif label == "buy_pressure":
                delta -= 0.08
            if wall == "bid":
                delta -= 0.05   # bid wall below → support

        return round(delta, 4)


def _median(arr: list[float]) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
