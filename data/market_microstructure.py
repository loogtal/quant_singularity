"""
Market Microstructure Signals — free data from Binance USDT-M futures API.

Four signals that professional quant funds use:

1. OPEN INTEREST (OI)
   - Rising OI + rising price = new LONG money entering = bull confirmation
   - Rising OI + falling price = new SHORT money entering = bear confirmation
   - Falling OI + price move = move is just liquidations, not conviction
   - OI change > 5% in 1h = significant repositioning

2. LONG/SHORT RATIO
   - > 1.5 = retail is very long = contrarian bearish signal
   - < 0.7 = retail is very short = contrarian bullish signal
   - Retail traders are WRONG at extremes (documented phenomenon)
   - Use as contrarian filter, not direction predictor

3. TAKER BUY/SELL VOLUME
   - Taker = aggressive market orders (willing to pay spread)
   - Buy ratio > 0.55 = buyers are aggressive = bullish pressure
   - Buy ratio < 0.45 = sellers are aggressive = bearish pressure
   - 15-min window shows short-term order flow direction

4. LIQUIDATION PRESSURE
   - Large liquidations = forced sellers/buyers = volatility spike incoming
   - Long liquidations dominant = market still has shorts to fill = bear continues
   - Short liquidations dominant = squeeze potential = be careful with shorts

Combined CONVICTION SCORE (0-1):
   - Aggregates all 4 signals relative to current trade direction
   - HIGH conviction (> 0.7) = increase position size
   - LOW conviction (< 0.3) = reduce or skip
"""

from __future__ import annotations

import time
from typing import Optional

_CACHE_TTL = 300   # 5 min — microstructure data changes slowly enough


class MarketMicrostructure:
    """
    Fetches and caches microstructure signals for a symbol.
    All data is free via Binance USDT-M futures API.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}   # symbol → {"ts": float, "data": dict}

    def get_signals(self, exchange, symbol: str) -> dict:
        """
        Returns microstructure signal dict for a symbol.
        Falls back to neutral signals if data unavailable.
        """
        neutral = self._neutral()
        if exchange is None:
            return neutral

        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached["ts"]) < _CACHE_TTL:
            return cached["data"]

        data = dict(neutral)

        # 1. Open Interest
        try:
            oi_now  = exchange.fetch_open_interest(symbol)
            oi_hist = exchange.fetch_open_interest_history(symbol, "1h", limit=3)
            if oi_now and oi_hist and len(oi_hist) >= 2:
                oi_current = float(oi_now.get("openInterestValue") or oi_now.get("openInterest") or 0)
                oi_prev    = float(oi_hist[-2].get("openInterestValue") or oi_hist[-2].get("openInterest") or 1)
                if oi_prev > 0:
                    data["oi_change_pct"] = round((oi_current - oi_prev) / oi_prev * 100, 2)
                    data["oi_available"] = True
        except Exception:
            pass

        # 2. Long/Short Ratio
        try:
            ls = exchange.fetch_long_short_ratio(symbol, "1h", limit=1)
            if ls:
                row = ls[-1] if isinstance(ls, list) else ls
                ratio = float(row.get("longShortRatio") or row.get("longAccount") or 0)
                if ratio > 0:
                    data["long_short_ratio"] = round(ratio, 4)
                    data["ls_available"] = True
        except Exception:
            pass

        # 3. Taker Buy/Sell Volume
        try:
            # Binance provides taker buy/sell via fetch_contract_ohlcv or custom endpoint
            raw = exchange.fapiDataGetTakerBuySellVol({
                "symbol": symbol.replace("/USDT:USDT", "USDT").replace("/", ""),
                "period": "15m",
                "limit":  4,
            })
            if raw:
                total_buy  = sum(float(r.get("buyVol",  0)) for r in raw)
                total_sell = sum(float(r.get("sellVol", 0)) for r in raw)
                total = total_buy + total_sell
                if total > 0:
                    data["taker_buy_ratio"] = round(total_buy / total, 4)
                    data["taker_available"]  = True
        except Exception:
            pass

        # 4. Recent Liquidations (approximate from OI + price move)
        # True liquidation data requires a paid feed; we approximate:
        # if OI dropped sharply while price moved = liquidation event
        if data["oi_available"]:
            oi_chg = data["oi_change_pct"]
            if oi_chg < -3.0:
                data["liquidation_signal"] = "long_liq"   # longs being flushed
            elif oi_chg > 3.0:
                data["liquidation_signal"] = "short_liq"  # shorts being squeezed
            else:
                data["liquidation_signal"] = "none"

        # Compute conviction score
        data["conviction"] = self._conviction_score(data)

        self._cache[symbol] = {"ts": now, "data": data}
        return data

    def _conviction_score(self, d: dict) -> float:
        """
        Aggregate signal into 0-1 conviction score.
        0.5 = neutral. > 0.65 = strong. < 0.35 = weak.
        Note: This is not directional — it measures signal clarity.
        Caller provides trade direction to interpret correctly.
        """
        score = 0.5
        count = 0

        # OI change: large move = conviction (regardless of direction)
        if d.get("oi_available"):
            oi_chg = abs(d.get("oi_change_pct", 0))
            if oi_chg > 5.0:
                score += 0.15; count += 1
            elif oi_chg > 2.0:
                score += 0.07; count += 1
            elif oi_chg < 0.5:
                score -= 0.07; count += 1   # no conviction

        # Taker ratio: extreme = conviction
        if d.get("taker_available"):
            tbr = d.get("taker_buy_ratio", 0.5)
            if tbr > 0.58 or tbr < 0.42:
                score += 0.12; count += 1
            elif 0.48 < tbr < 0.52:
                score -= 0.05; count += 1   # balanced = low conviction

        if count == 0:
            return 0.5
        return round(min(0.9, max(0.1, score)), 4)

    def directional_boost(self, signals: dict, trade_side: str) -> float:
        """
        Returns a confidence delta based on microstructure alignment with trade direction.

        trade_side = "LONG" or "SHORT"
        Returns +0.05 to +0.12 (aligned) or -0.08 to -0.15 (opposed)
        """
        if not any([signals.get("oi_available"), signals.get("ls_available"),
                    signals.get("taker_available")]):
            return 0.0

        delta = 0.0

        # Taker flow
        tbr = signals.get("taker_buy_ratio", 0.5)
        if signals.get("taker_available"):
            if trade_side == "LONG"  and tbr > 0.55:  delta += 0.07
            if trade_side == "SHORT" and tbr < 0.45:  delta += 0.07
            if trade_side == "LONG"  and tbr < 0.42:  delta -= 0.08
            if trade_side == "SHORT" and tbr > 0.58:  delta -= 0.08

        # Long/short ratio (contrarian)
        lsr = signals.get("long_short_ratio", 1.0)
        if signals.get("ls_available"):
            if trade_side == "SHORT" and lsr > 1.5:  delta += 0.06  # retail too long = SHORT confirmed
            if trade_side == "LONG"  and lsr < 0.7:  delta += 0.06  # retail too short = LONG confirmed
            if trade_side == "LONG"  and lsr > 2.0:  delta -= 0.05  # extreme long = caution
            if trade_side == "SHORT" and lsr < 0.5:  delta -= 0.05  # extreme short = squeeze risk

        # OI direction
        oi_chg = signals.get("oi_change_pct", 0)
        if signals.get("oi_available"):
            if trade_side == "SHORT" and oi_chg > 3.0: delta += 0.05  # OI rising while price falls
            if trade_side == "LONG"  and oi_chg > 3.0 and delta < 0: delta += 0.03

        return round(delta, 4)

    @staticmethod
    def _neutral() -> dict:
        return {
            "oi_change_pct":     0.0,
            "long_short_ratio":  1.0,
            "taker_buy_ratio":   0.5,
            "liquidation_signal":"none",
            "conviction":        0.5,
            "oi_available":      False,
            "ls_available":      False,
            "taker_available":   False,
        }
