"""
Market-wide intelligence layer.

Provides macro context that informs both strategies:

  market_breadth()     → % of top-N coins above their EMA50 (0-100)
  fear_greed_proxy()   → 0-100 score derived purely from price action
  aggregate_funding()  → average funding rate across the universe
  dominant_trend()     → "bull" | "bear" | "mixed"
  btc_cycle()          → BTC halving-cycle phase, 200wMA position
  full_context()       → combined dict (cached 10 min to avoid spamming API)

Used by DualRegimeRouter and DualEngine to gate position sizing:
  - breadth < 30  → bear breadth → shrink passive sizes
  - breadth > 70  → bull breadth → allow full passive sizes
  - fear_greed > 80 → euphoria → fade with active SHORT / mean reversion
  - fear_greed < 20 → panic → fade with active LONG / mean reversion
  - btc_cycle.phase = bear/accumulation → prefer mean_reversion + funding_arb
  - btc_cycle.phase = bull/early_bull   → allow momentum + larger passive sizes
"""

import datetime
import json
import time
import urllib.request
from typing import Optional

import numpy as np

from data.market_data import MarketData

_UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
    "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "DOT/USDT:USDT",
]

_CACHE_TTL = 600  # seconds


def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


class MarketIntelligence:
    """Aggregates market-wide signals across a 10-coin universe."""

    def __init__(self):
        self.market = MarketData()
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0

    def _fetch_real_fear_greed(self) -> float | None:
        """
        Pull the live Crypto Fear & Greed Index from alternative.me (free, no key needed).
        Updates once per day; falls back silently to the price-action proxy on failure.
        """
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = json.loads(resp.read())
                return float(data["data"][0]["value"])
        except Exception:
            return None

    def full_context(self) -> dict:
        now = time.time()
        if self._cache and (now - self._cache_ts) < _CACHE_TTL:
            return self._cache

        breadth  = self._compute_breadth()
        funding  = self._compute_aggregate_funding()
        real_fg  = self._fetch_real_fear_greed()          # live index (preferred)
        fg       = real_fg if real_fg is not None else self._compute_fear_greed(breadth, funding)
        trend    = self._dominant_trend(breadth)
        btc_rs   = self._compute_btc_relative_strength()
        btc_cycle = self._compute_btc_cycle()

        result = {
            "breadth":           round(breadth, 1),
            "fear_greed":        round(fg, 1),
            "aggregate_funding": round(funding, 6),
            "dominant_trend":    trend,
            "bull_breadth":      breadth >= 60,
            "bear_breadth":      breadth <= 35,
            "euphoria":          fg >= 78,
            "panic":             fg <= 22,
            "btc_rs_score":      btc_rs["btc_rs_score"],
            "alt_season":        btc_rs["alt_season"],
            "btc_dominant":      btc_rs["btc_dominant"],
            "fear_greed_source": "live" if real_fg is not None else "proxy",
            "btc_cycle":         btc_cycle,
        }
        self._cache = result
        self._cache_ts = now
        return result

    # ── breadth ───────────────────────────────────────────────────────────────

    def _compute_breadth(self) -> float:
        """% of universe coins where price > EMA50 on 4H."""
        above = 0
        total = 0
        for sym in _UNIVERSE:
            try:
                df = self.market.get_ohlcv_df(sym, timeframe="4h", limit=60)
                if len(df) < 55:
                    continue
                closes = df["close"].values
                ema50 = _ema(closes, 50)
                if closes[-1] > ema50:
                    above += 1
                total += 1
            except Exception:
                continue
        if total == 0:
            return 50.0
        return round(100.0 * above / total, 1)

    # ── aggregate funding ─────────────────────────────────────────────────────

    def _compute_aggregate_funding(self) -> float:
        """Mean funding rate across universe (positive = longs crowded)."""
        try:
            from data.funding_data import FundingData
            fd = FundingData()
            rates = []
            for sym in _UNIVERSE[:5]:   # sample top-5 to limit API calls
                try:
                    data = fd.get_funding(sym)
                    rate = float(data.get("funding_rate", 0))
                    rates.append(rate)
                except Exception:
                    continue
            return float(np.mean(rates)) if rates else 0.0
        except Exception:
            return 0.0

    # ── fear & greed proxy ────────────────────────────────────────────────────

    def _compute_fear_greed(self, breadth: float, funding: float) -> float:
        """
        Proxy F&G index from price action:
          - breadth component:  0-50 → fear; 50-100 → greed
          - BTC momentum:       recent 7D return of BTC
          - funding component:  extreme positive = greed, extreme negative = fear
        Combined to 0-100.
        """
        # Breadth sub-score (0-100)
        breadth_score = breadth   # already 0-100

        # BTC momentum sub-score (0-100)
        btc_score = 50.0
        try:
            df = self.market.get_ohlcv_df("BTC/USDT:USDT", timeframe="1d", limit=10)
            if len(df) >= 8:
                ret_7d = (float(df["close"].iloc[-1]) - float(df["close"].iloc[-8])) / float(df["close"].iloc[-8])
                btc_score = float(np.clip(50 + ret_7d * 500, 0, 100))  # ±10% → 0-100
        except Exception:
            pass

        # Funding sub-score (0-100)
        # funding 0 → 50, +0.001 → 80, -0.001 → 20
        funding_score = float(np.clip(50 + funding * 30_000, 0, 100))

        # Weighted average
        fg = breadth_score * 0.45 + btc_score * 0.35 + funding_score * 0.20
        return float(np.clip(fg, 0, 100))

    # ── BTC relative strength ─────────────────────────────────────────────────

    def _compute_btc_relative_strength(self) -> dict:
        """
        Compare BTC's 7-day return vs the altcoin universe average.

        btc_rs_score > 65 → BTC outperforming (rotate to BTC, reduce alt exposure)
        btc_rs_score < 35 → alts outperforming (alt-season, favour active long alts)
        """
        try:
            alts = [s for s in _UNIVERSE if s != "BTC/USDT:USDT"]
            df_btc = self.market.get_ohlcv_df("BTC/USDT:USDT", timeframe="1d", limit=10)
            if len(df_btc) < 8:
                return {"btc_rs_score": 50.0, "alt_season": False, "btc_dominant": False}
            btc_ret = (
                float(df_btc["close"].iloc[-1]) - float(df_btc["close"].iloc[-8])
            ) / float(df_btc["close"].iloc[-8])

            alt_rets = []
            for sym in alts[:6]:
                try:
                    df = self.market.get_ohlcv_df(sym, timeframe="1d", limit=10)
                    if len(df) >= 8:
                        r = (
                            float(df["close"].iloc[-1]) - float(df["close"].iloc[-8])
                        ) / float(df["close"].iloc[-8])
                        alt_rets.append(r)
                except Exception:
                    continue

            if not alt_rets:
                return {"btc_rs_score": 50.0, "alt_season": False, "btc_dominant": False}

            diff = btc_ret - float(np.mean(alt_rets))
            btc_rs_score = float(np.clip(50.0 + diff * 500, 0, 100))
            return {
                "btc_rs_score": round(btc_rs_score, 1),
                "alt_season":   btc_rs_score < 35,
                "btc_dominant": btc_rs_score > 65,
            }
        except Exception:
            return {"btc_rs_score": 50.0, "alt_season": False, "btc_dominant": False}

    # ── BTC halving cycle ─────────────────────────────────────────────────────

    def _compute_btc_cycle(self) -> dict:
        """
        Detect BTC market cycle phase using halving calendar + 200-week MA.

        Known halvings:
          2024-04-20 (last)
          2028-04-20 (next, approximate)

        Historical post-halving pattern (rough guide, not gospel):
          0–20 %  of cycle → early_bull  (accumulation into rally)
          20–45%           → bull        (markup phase)
          45–65%           → distribution (euphoria / top)
          65–85%           → bear        (markdown)
          85–100%          → accumulation (pre-halving recovery)

        200-week MA: price below = historically deep value zone.
        """
        try:
            last_halving = datetime.date(2024, 4, 20)
            next_halving = datetime.date(2028, 4, 20)
            today        = datetime.date.today()

            cycle_len            = (next_halving - last_halving).days
            days_since_halving   = max(0, (today - last_halving).days)
            days_to_next_halving = max(0, (next_halving - today).days)
            cycle_progress       = min(1.0, days_since_halving / cycle_len)

            if cycle_progress < 0.20:
                phase = "early_bull"
            elif cycle_progress < 0.45:
                phase = "bull"
            elif cycle_progress < 0.65:
                phase = "distribution"
            elif cycle_progress < 0.85:
                phase = "bear"
            else:
                phase = "accumulation"

            # 200-week MA (requires ~1400 daily bars ≈ 200 weeks)
            price_vs_200wma = 1.0
            above_200wma    = True
            try:
                df = self.market.get_ohlcv_df("BTC/USDT:USDT", timeframe="1w", limit=210)
                if len(df) >= 50:
                    price       = float(df["close"].iloc[-1])
                    n           = min(200, len(df))
                    ma200w      = float(np.mean(df["close"].values[-n:]))
                    price_vs_200wma = round(price / ma200w, 3)
                    above_200wma    = price > ma200w
            except Exception:
                pass

            return {
                "cycle_phase":           phase,
                "cycle_progress":        round(cycle_progress, 3),
                "days_since_halving":    days_since_halving,
                "days_to_next_halving":  days_to_next_halving,
                "price_vs_200wma":       price_vs_200wma,
                "above_200wma":          above_200wma,
            }
        except Exception:
            return {
                "cycle_phase":           "unknown",
                "cycle_progress":        0.5,
                "days_since_halving":    0,
                "days_to_next_halving":  0,
                "price_vs_200wma":       1.0,
                "above_200wma":          True,
            }

    # ── dominant trend ────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_trend(breadth: float) -> str:
        if breadth >= 65:
            return "bull"
        if breadth <= 35:
            return "bear"
        return "mixed"
