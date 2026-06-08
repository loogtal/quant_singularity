"""
Crypto News & Event Detection — CryptoPanic free API + RSS fallback.

Detects high-impact events that move markets:
  - Exchange hacks / security incidents
  - Regulatory actions (SEC, CFTC)
  - Major protocol upgrades / forks
  - Macro events (Fed, CPI)
  - Large liquidation cascades

Outputs:
  - news_sentiment: "positive" | "negative" | "neutral"
  - impact_level:   "high" | "medium" | "low"
  - affected_coins: ["BTC", "ETH", ...]
  - should_pause:   True if market-moving event detected (pause new entries)
  - fear_spike:     True if sudden fear event (possible flash crash incoming)

Free API: CryptoPanic (no key required for basic feed)
Fallback: RSS feeds from CoinDesk, Cointelegraph
Cache: 10 min (news doesn't change that fast)
"""

from __future__ import annotations

import json
import time
from typing import Optional
from urllib.request import urlopen, Request

_CACHE_TTL = 600   # 10 min

# ONLY truly market-moving events that justify pausing ALL new entries.
# Must be 2+ words to avoid false positives (e.g. "hack" alone is too broad).
# These are exchange-level or systemic events, not project-specific news.
_PAUSE_KEYWORDS = [
    "exchange hack", "exchange exploit", "exchange breach",
    "exchange halted", "exchange suspended", "exchange shutdown",
    "ftx collapse", "binance hack", "coinbase hack",
    "flash crash", "market circuit breaker", "trading halted",
    "sec emergency", "doj seizure", "government ban crypto",
    "billion liquidated", "billion liquidation",
    "tether depegged", "usdt depegged", "usdc depegged",
    "systemic risk", "contagion",
]

# Keywords that adjust confidence (NOT pause) — just shift probability
_BEARISH_KEYWORDS = [
    "regulation crackdown", "major lawsuit", "criminal charges",
    "sell-off", "market crash", "rate hike surprise",
    "inflation spike", "recession fears",
]

_BULLISH_KEYWORDS = [
    "etf approved", "spot etf", "institutional adoption",
    "rate cut", "fed pivot", "major partnership",
    "bitcoin reserve", "strategic reserve",
]


class NewsFeed:
    """
    Lightweight news monitor for crypto markets.
    Uses CryptoPanic free API (no key needed for latest headlines).
    """

    def __init__(self) -> None:
        self._cache:         Optional[dict] = None
        self._cache_ts:      float = 0.0
        self._last_warn_ts:  float = 0.0   # rate-limit error logging

    def get_news_signal(self) -> dict:
        """
        Returns news-based market signal.
        Cached 10 minutes — call every cycle, API only hit every 10 min.
        """
        neutral = {
            "news_sentiment":  "neutral",
            "impact_level":    "low",
            "affected_coins":  [],
            "should_pause":    False,
            "fear_spike":      False,
            "top_headline":    "",
            "available":       False,
        }

        now = time.time()
        if self._cache and (now - self._cache_ts) < _CACHE_TTL:
            return self._cache

        headlines = self._fetch_headlines()
        if not headlines:
            return neutral

        result = self._analyze(headlines)
        result["available"] = True
        self._cache    = result
        self._cache_ts = now
        return result

    def _fetch_headlines(self) -> list[str]:
        """Fetch latest crypto headlines. Returns list of headline strings."""
        # Try CryptoPanic free API (public, no key needed)
        try:
            req = Request(
                "https://cryptopanic.com/api/free/v1/posts/?auth_token=free&public=true&kind=news&limit=20",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                headlines = []
                for r in results:
                    title = r.get("title", "").lower()
                    if title:
                        headlines.append(title)
                return headlines
        except Exception:
            pass

        # Fallback: parse CoinDesk RSS
        try:
            req = Request(
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urlopen(req, timeout=5) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                import re
                titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", content)
                return [t.lower() for t in titles[:20]]
        except Exception:
            pass

        return []

    def _analyze(self, headlines: list[str]) -> dict:
        """Analyze headlines for market impact."""
        bearish_count = 0
        bullish_count = 0
        pause_triggered = False
        affected_coins: set[str] = set()
        top_headline = headlines[0] if headlines else ""

        # Common coin mentions
        coin_keywords = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth": "ETH",
            "solana": "SOL", "sol": "SOL",
            "ripple": "XRP", "xrp": "XRP",
            "bnb": "BNB", "binance": "BNB",
        }

        for headline in headlines:
            # Check for pause-worthy events
            for kw in _PAUSE_KEYWORDS:
                if kw in headline:
                    pause_triggered = True
                    break

            # Count sentiment
            for kw in _BEARISH_KEYWORDS:
                if kw in headline:
                    bearish_count += 1
            for kw in _BULLISH_KEYWORDS:
                if kw in headline:
                    bullish_count += 1

            # Detect affected coins
            for kw, coin in coin_keywords.items():
                if kw in headline:
                    affected_coins.add(coin)

        # Determine sentiment
        if bearish_count > bullish_count * 1.5:
            sentiment = "negative"
        elif bullish_count > bearish_count * 1.5:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        # Impact level
        total_signals = bearish_count + bullish_count
        if total_signals >= 6 or pause_triggered:
            impact = "high"
        elif total_signals >= 3:
            impact = "medium"
        else:
            impact = "low"

        return {
            "news_sentiment":  sentiment,
            "impact_level":    impact,
            "affected_coins":  list(affected_coins),
            "should_pause":    pause_triggered and impact == "high",
            "fear_spike":      pause_triggered and bearish_count > 3,
            "top_headline":    top_headline[:100],
            "bearish_signals": bearish_count,
            "bullish_signals": bullish_count,
        }

    def confidence_adjustment(self, signals: dict, trade_side: str, symbol: str) -> float:
        """
        Returns confidence delta based on news sentiment.
        Only applies when coin is directly mentioned in news.
        """
        if not signals.get("available"):
            return 0.0

        coin = symbol.split("/")[0] if "/" in symbol else symbol[:3]
        is_affected = (
            coin in signals.get("affected_coins", [])
            or not signals.get("affected_coins")   # if no specific coin, global news
        )

        if not is_affected:
            return 0.0

        sentiment = signals.get("news_sentiment", "neutral")
        impact    = signals.get("impact_level", "low")

        multiplier = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(impact, 0.5)

        delta = 0.0
        if trade_side == "SHORT" and sentiment == "negative":
            delta = 0.06 * multiplier
        elif trade_side == "LONG" and sentiment == "positive":
            delta = 0.06 * multiplier
        elif trade_side == "SHORT" and sentiment == "positive":
            delta = -0.08 * multiplier
        elif trade_side == "LONG" and sentiment == "negative":
            delta = -0.08 * multiplier

        return round(delta, 4)
