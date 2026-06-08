"""
Autonomous coin selection — ranks Binance USDT-M futures separately
for the passive (trend) and active (intraday) strategies.

Passive scan  (scan_for_passive):
  Uses 1D bars. Rewards strong sustained trends with good ADX.
  Penalises overextended coins and high recent volatility.

Active scan   (scan_for_active):
  Uses 15M bars. Rewards RSI extremes and volume surges.
  Prefers coins near VWAP extremes (mean reversion candidates).

General scan  (scan_market — backward compat for single-strategy use).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from config.settings import MIN_24H_VOLUME_USDT, SCAN_TOP_N, USE_FUNDING_ARB
from data.binance_client import BinanceClient
from data.market_data import MarketData
from factors.alpha_factors import AlphaFactors
from strategy.funding_arb import FundingArbStrategy

_SCAN_WORKERS = 4   # parallel threads — keep below Binance rate limit


# ── shared technical helpers ────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains))  if gains.any()  else 0.0
    avg_l  = float(np.mean(losses)) if losses.any() else 1e-9
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))


def _atr(highs, lows, closes, period=14) -> float:
    if len(closes) < 2:
        return float(closes[-1]) * 0.02
    tr = []
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return float(np.mean(tr[-period:])) if tr else float(closes[-1]) * 0.015


def _adx_simple(highs, lows, closes, period=14) -> float:
    """Simplified ADX estimate (0-100)."""
    if len(closes) < period + 5:
        return 15.0
    dm_plus, dm_minus, tr_list = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i - 1]
        ld = lows[i - 1] - lows[i]
        dm_plus.append(hd if hd > ld and hd > 0 else 0.0)
        dm_minus.append(ld if ld > hd and ld > 0 else 0.0)
        tr_list.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))

    def _wilder(arr, p):
        s = float(np.sum(arr[:p]))
        result = [s]
        for x in arr[p:]:
            s = s - s / p + x
            result.append(s)
        return result

    sm_tr  = _wilder(tr_list,  period)
    sm_dmp = _wilder(dm_plus,  period)
    sm_dmn = _wilder(dm_minus, period)
    dx_list = []
    for tr, dmp, dmn in zip(sm_tr, sm_dmp, sm_dmn):
        if tr <= 0:
            continue
        di_p = 100 * dmp / tr
        di_n = 100 * dmn / tr
        denom = di_p + di_n
        if denom > 0:
            dx_list.append(100 * abs(di_p - di_n) / denom)
    return float(np.mean(dx_list[-period:])) if len(dx_list) >= period else 15.0


def _vwap(df) -> float:
    tp  = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    vol = df["volume"].values
    total = float(np.sum(vol))
    if total <= 0:
        return float(df["close"].iloc[-1])
    return float(np.sum(tp * vol) / total)


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# ── scanner class ───────────────────────────────────────────────────────────────

_PASSIVE_CACHE_TTL = 300   # 5 min — 1D bars change slowly
_ACTIVE_CACHE_TTL  = 120   # 2 min — 15M setups shift faster


class CoinScanner:
    """Ranks Binance USDT-M futures for passive (trend) and active (intraday) use."""

    def __init__(self):
        self.client      = BinanceClient()
        self.market      = MarketData()
        self.factors     = AlphaFactors()
        self.funding_arb = FundingArbStrategy() if USE_FUNDING_ARB else None
        self._cache_passive: list = []
        self._cache_active:  list = []
        self._ts_passive: float = 0.0
        self._ts_active:  float = 0.0
        self._cache_anomaly: list = []
        self._ts_anomaly:  float = 0.0

    # ── passive scoring (1D trend quality) ─────────────────────────────────────

    def _score_passive(self, symbol: str) -> dict[str, Any] | None:
        """
        Score a symbol for the passive (multi-day trend) strategy using 1D bars.

        Rewards:
          - Strong ADX (clear directional trend)
          - EMA50 / EMA200 aligned (trend direction)
          - Trend has been in place for multiple days (duration bonus)
          - Reasonable ATR (not too explosive, not dead)
        Penalises:
          - Overextended price (RSI > 80 for LONG or < 20 for SHORT)
          - Very high recent volatility (unpredictable)
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="1d", limit=220)
            if len(df) < 60:
                return None

            closes  = df["close"].values
            highs   = df["high"].values
            lows    = df["low"].values
            volumes = df["volume"].values

            # Liquidity filter: daily volume in USDT
            daily_vol_usdt = float(closes[-1] * volumes[-1])
            if daily_vol_usdt < MIN_24H_VOLUME_USDT * 0.5:
                return None

            ema50  = _ema(closes, 50)
            ema200 = _ema(closes, 200)
            trend  = "LONG" if ema50 > ema200 else "SHORT"

            adx = _adx_simple(highs, lows, closes, period=14)

            rsi = _rsi(closes, 14)
            # Penalise overextension in the trend direction
            if trend == "LONG"  and rsi > 80: return None
            if trend == "SHORT" and rsi < 20:  return None

            atr_pct = _atr(highs, lows, closes, 14) / closes[-1]

            # Trend duration: how many days has EMA50 been above/below EMA200?
            ema50_series  = [_ema(closes[:i + 1], 50)  for i in range(max(0, len(closes) - 30), len(closes))]
            ema200_series = [_ema(closes[:i + 1], 200) for i in range(max(0, len(closes) - 30), len(closes))]
            if trend == "LONG":
                duration = sum(1 for f, s in zip(ema50_series, ema200_series) if f > s)
            else:
                duration = sum(1 for f, s in zip(ema50_series, ema200_series) if f < s)

            # Component scores
            adx_score      = _normalize(adx, 15, 50)           # 15→0, 50→1
            duration_score = _normalize(duration, 0, 30)       # 0→0, 30→1
            atr_score      = 1 - _normalize(atr_pct, 0, 0.06)  # lower atr = more predictable = better
            volume_score   = _normalize(daily_vol_usdt, MIN_24H_VOLUME_USDT * 0.5, MIN_24H_VOLUME_USDT * 10)
            # RSI sweet zone: 40-65 for LONG, 35-60 for SHORT
            if trend == "LONG":
                rsi_score = _normalize(rsi, 35, 65) if rsi < 65 else max(0, 1 - (rsi - 65) / 20)
            else:
                rsi_score = _normalize(60 - rsi, -5, 30) if rsi > 35 else max(0, 1 - (35 - rsi) / 15)

            score = round(
                adx_score      * 0.30
                + duration_score * 0.25
                + volume_score   * 0.20
                + rsi_score      * 0.15
                + atr_score      * 0.10,
                4,
            )
            return {
                "symbol":    symbol,
                "score":     score,
                "strategy":  "passive",
                "trend":     trend,
                "adx":       round(adx, 1),
                "rsi":       round(rsi, 1),
                "atr_pct":   round(atr_pct * 100, 2),
                "duration":  duration,
                "volume_usdt": round(daily_vol_usdt / 1e6, 1),
            }
        except Exception:
            return None

    # ── active scoring (15M mean reversion / momentum) ─────────────────────────

    @staticmethod
    def _breakout_score(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                        lookback: int = 48) -> float:
        """
        Score how close price is to a breakout from its recent range.
        A breakout = price exceeding max(highs[-lookback:]) or below min(lows[-lookback:])
        within the last 2 bars. Score 1.0 at confirmed breakout, 0.5 at range midpoint.
        """
        if len(closes) < lookback + 2:
            return 0.5
        price      = float(closes[-1])
        range_high = float(np.max(highs[-(lookback + 1):-1]))
        range_low  = float(np.min(lows[-(lookback + 1):-1]))
        range_size = range_high - range_low
        if range_size <= 0:
            return 0.5
        # Normalised position 0 (at low) → 1 (at high); breakout = outside [0, 1]
        pos = (price - range_low) / range_size
        # Reward extremes: either near the top (bull breakout) or near the bottom (bear)
        return float(np.clip(max(abs(pos - 0.5) * 2, 0), 0, 1))

    @staticmethod
    def _volume_trend_score(volumes: np.ndarray, closes: np.ndarray, lookback: int = 12) -> float:
        """
        Is volume trending up or down with price?
        Rising volume with rising price (or falling with falling) = directional conviction.
        Returns 0-1; 1 = strong directional volume trend.
        """
        if len(volumes) < lookback + 1 or len(closes) < lookback + 1:
            return 0.5
        price_ret  = (float(closes[-1]) - float(closes[-lookback])) / float(closes[-lookback])
        vol_change = (float(np.mean(volumes[-3:])) - float(np.mean(volumes[-lookback:-3]))) / \
                     (float(np.mean(volumes[-lookback:-3])) + 1e-9)
        # Same direction = conviction; opposite = divergence (fade setup)
        conviction = np.clip(abs(vol_change) * (1 if price_ret * vol_change >= 0 else -1), -1, 1)
        return float(np.clip((conviction + 1) / 2, 0, 1))

    def _score_active(self, symbol: str) -> dict[str, Any] | None:
        """
        Score a symbol for the active (intraday) strategy using 15M bars.

        Rewards:
          - Extreme RSI (< 30 or > 70) — mean reversion opportunity
          - Price far from VWAP — snap-back potential
          - Recent volume surge with directional conviction
          - Good liquidity
          - Breakout from recent range (momentum setup)
        Penalises:
          - Very low volatility (nothing moves)
          - Price mid-range with no clear setup
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            if len(df) < 50:
                return None

            closes  = df["close"].values
            highs   = df["high"].values
            lows    = df["low"].values
            volumes = df["volume"].values
            price   = float(closes[-1])

            # Liquidity filter: 15M turnover in USDT
            recent_vol_usdt = float(np.mean(volumes[-12:] * closes[-12:]))
            if recent_vol_usdt < 500_000:   # $500k per 15M bar minimum
                return None

            rsi      = _rsi(closes, 14)
            vwap_val = _vwap(df)

            # RSI extreme score: 0 at RSI=50, 1 at RSI=20 or RSI=80
            rsi_extreme = max(0, (50 - rsi) / 30) if rsi < 50 else max(0, (rsi - 50) / 30)
            rsi_extreme = float(np.clip(rsi_extreme, 0, 1))

            # VWAP deviation score
            vwap_dev   = abs(price - vwap_val) / vwap_val if vwap_val > 0 else 0.0
            vwap_score = _normalize(vwap_dev, 0, 0.025)

            # Volume surge + directional conviction
            avg_vol      = float(np.mean(volumes[-21:-1]))
            vol_ratio    = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0
            volume_score = _normalize(vol_ratio, 0.8, 2.5)
            vol_trend    = self._volume_trend_score(volumes, closes)

            # ATR / volatility — need some movement to trade intraday
            atr_pct          = _atr(highs, lows, closes, 14) / price
            volatility_score = _normalize(atr_pct, 0.002, 0.012)

            # Breakout score (bonus for momentum setups)
            breakout = self._breakout_score(closes, highs, lows)

            # Funding arb bonus
            funding_bonus = 0.0
            if self.funding_arb:
                try:
                    funding_bonus = self.funding_arb.score_adjustment(symbol)
                except Exception:
                    pass

            score = round(
                rsi_extreme      * 0.30
                + vwap_score     * 0.20
                + volume_score   * 0.15
                + vol_trend      * 0.10
                + volatility_score * 0.10
                + breakout       * 0.10
                + funding_bonus  * 0.05,
                4,
            )
            score = float(np.clip(score, 0, 1))
            return {
                "symbol":    symbol,
                "score":     score,
                "strategy":  "active",
                "rsi":       round(rsi, 1),
                "vwap_dev":  round(vwap_dev * 100, 2),
                "vol_ratio": round(vol_ratio, 2),
                "vol_trend": round(vol_trend, 3),
                "breakout":  round(breakout, 3),
                "atr_pct":   round(atr_pct * 100, 3),
            }
        except Exception:
            return None

    # ── backward-compatible general scorer ─────────────────────────────────────

    def score_symbol(self, symbol: str) -> dict[str, Any] | None:
        """General-purpose scorer (legacy single-strategy mode)."""
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            if len(df) < 50:
                return None

            closes  = df["close"].values
            volumes = df["volume"].values

            momentum     = self.factors.momentum_factor(closes)
            vol_factor   = self.factors.volatility_factor(closes)
            trend        = self.factors.trend_strength(closes)

            vol_usdt = float(volumes[-1] * closes[-1])
            avg_vol  = float(np.mean(volumes[-20:] * closes[-20:]))

            mom_score    = _normalize(momentum,   -0.05, 0.05)
            trend_score  = _normalize(trend,      -0.03, 0.03)
            volume_score = _normalize(avg_vol,    0,     max(vol_usdt * 10, 1))
            vol_penalty  = _normalize(vol_factor, 0,     0.03)

            score = round(
                mom_score    * 0.35
                + trend_score  * 0.35
                + volume_score * 0.20
                + (1 - vol_penalty) * 0.10,
                4,
            )
            if self.funding_arb:
                try:
                    score = float(np.clip(score + self.funding_arb.score_adjustment(symbol), 0, 1))
                except Exception:
                    pass

            return {
                "symbol":     symbol,
                "score":      round(score, 4),
                "momentum":   round(float(momentum), 4),
                "volume":     round(float(volume_score), 4),
                "trend":      round(float(trend), 4),
                "alpha":      round(score, 4),
                "volatility": round(float(vol_factor), 4),
                "liquidity":  round(float(volume_score), 4),
            }
        except Exception:
            return None

    # ── public scanning methods ─────────────────────────────────────────────────

    def _get_universe(self) -> list[str]:
        symbols = self.client.list_usdt_futures()
        if not symbols:
            return ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                    "BNB/USDT:USDT", "XRP/USDT:USDT"]
        # Verify symbols are actually active in the exchange markets dict
        ex = self.client.get_exchange()
        if ex and hasattr(ex, "markets"):
            symbols = [s for s in symbols if ex.markets.get(s, {}).get("active", True)]
        return symbols[:SCAN_TOP_N]

    def _parallel_score(
        self,
        symbols: list[str],
        scorer,
        min_score: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Score symbols in parallel using a thread pool."""
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
            futures = {pool.submit(scorer, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    data = fut.result(timeout=15)
                    if data and data.get("score", 0) >= min_score:
                        results.append(data)
                except Exception:
                    pass
        return results

    def scan_for_passive(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return top-N symbols suited for the passive (trend) strategy (cached 5 min)."""
        import time
        if self._cache_passive and (time.time() - self._ts_passive) < _PASSIVE_CACHE_TTL:
            return self._cache_passive[:top_n]
        symbols = self._get_universe()
        results = self._parallel_score(symbols, self._score_passive, min_score=0.40)
        results.sort(key=lambda x: x["score"], reverse=True)
        self._cache_passive = results
        self._ts_passive = time.time()
        return results[:top_n]

    def scan_for_active(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return top-N symbols suited for the active (intraday) strategy (cached 2 min)."""
        import time
        if self._cache_active and (time.time() - self._ts_active) < _ACTIVE_CACHE_TTL:
            return self._cache_active[:top_n]
        symbols = self._get_universe()
        results = self._parallel_score(symbols, self._score_active, min_score=0.40)
        results.sort(key=lambda x: x["score"], reverse=True)
        self._cache_active = results
        self._ts_active = time.time()
        return results[:top_n]

    def scan_market(self) -> list[dict[str, Any]]:
        """Backward-compatible general scan (used by legacy engine)."""
        symbols = self._get_universe()
        results = self._parallel_score(symbols, self.score_symbol, min_score=0.35)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def scan_volume_anomaly(self, top_n: int = 5) -> list[dict[str, Any]]:
        """
        Detect symbols with unusual 24h volume vs their 7-day average.

        For each symbol in the universe (first 50 by volume):
          - Fetch 1h OHLCV for last 48 bars
          - Compute 24h volume vs 7-day average daily volume
          - If ratio >= 2.5x: anomaly detected
          - Return top_n symbols sorted by anomaly ratio

        Returns list of dicts with keys:
            symbol, volume_ratio, direction (LONG if price up, SHORT if down)

        Result is cached for 15 minutes.
        """
        import time as _time
        _ANOMALY_TTL = 900   # 15 min

        if self._cache_anomaly and (_time.time() - self._ts_anomaly) < _ANOMALY_TTL:
            return self._cache_anomaly[:top_n]

        symbols  = self._get_universe()[:50]
        anomalies: list[dict[str, Any]] = []

        def _score_anomaly(symbol: str) -> dict[str, Any] | None:
            try:
                df = self.market.get_ohlcv_df(symbol, timeframe="1h", limit=48)
                if len(df) < 24:
                    return None

                vols   = df["volume"].values
                closes = df["close"].values

                # 24h volume = sum of last 24 hourly bars
                vol_24h = float(np.sum(vols[-24:]))
                # 7-day average daily volume = mean of 24-bar sums across the 48-bar window
                # We use the whole 48 bars split into two 24-bar windows
                if len(vols) >= 48:
                    vol_prior_24h = float(np.sum(vols[-48:-24]))
                else:
                    vol_prior_24h = vol_24h  # not enough data

                # 7-day avg = use available prior bars / number of full days
                # Simplified: compare 24h vs the prior 24h (best proxy with 48 bars)
                avg_daily = vol_prior_24h if vol_prior_24h > 0 else vol_24h
                ratio = vol_24h / avg_daily if avg_daily > 0 else 1.0

                if ratio < 2.5:
                    return None

                # Direction: price change over last 24h
                price_now  = float(closes[-1])
                price_prev = float(closes[-24]) if len(closes) >= 24 else float(closes[0])
                direction  = "LONG" if price_now >= price_prev else "SHORT"

                return {
                    "symbol":       symbol,
                    "volume_ratio": round(ratio, 3),
                    "direction":    direction,
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
            futures = {pool.submit(_score_anomaly, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    result = fut.result(timeout=15)
                    if result is not None:
                        anomalies.append(result)
                except Exception:
                    pass

        anomalies.sort(key=lambda x: x["volume_ratio"], reverse=True)
        self._cache_anomaly = anomalies
        self._ts_anomaly    = _time.time()
        return anomalies[:top_n]
