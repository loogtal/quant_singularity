"""
Passive trend strategy — multi-day EMA crossover with quality filters.

Entry requirements (all must pass):
  1. EMA50 / EMA200 crossover agrees on both 4H and 1D
  2. ADX > ADX_MIN_THRESHOLD (confirms trend strength, rejects choppy markets)
  3. Entry volume >= VOLUME_CONFIRM_MULT × 20-bar average (institutional participation)
  4. Ichimoku cloud confirms direction (price above cloud for LONG, below for SHORT)
     — price inside cloud reduces confidence; doesn't block entry on strong ADX

TP / SL use ATR-based distances instead of fixed percentages:
  TP = entry ± ATR_TP_MULT × ATR(1D)
  SL = entry ± ATR_SL_MULT × ATR(1D)
  This auto-widens stops in high-volatility conditions and tightens them in calm markets.
"""

import numpy as np

from config.dual_settings import (
    PASSIVE_MAX_POSITIONS,
    PASSIVE_TIMEFRAMES,
    PASSIVE_TAKE_PROFIT,
    PASSIVE_STOP_LOSS,
    PASSIVE_MIN_HOLD_HOURS,
    PASSIVE_MAX_HOLD_HOURS,
    PASSIVE_REGIME_BLACKLIST,
)
from data.market_data import MarketData


# ── quality-filter constants ───────────────────────────────────────────────────

ADX_MIN_THRESHOLD    = 14     # require at least moderate trend strength
ADX_MIN_THRESHOLD_TREND = 12  # relaxed threshold when mode forces a direction (e.g. short in BEAR)
VOLUME_CONFIRM_MULT  = 1.1    # volume must be 1.1× the 20-bar average

# ATR multipliers for TP and SL
ATR_TP_MULT = 3.0   # TP = entry ± 3 × ATR(1D)
ATR_SL_MULT = 1.5   # SL = entry ± 1.5 × ATR(1D)

# If ATR-derived TP/SL are outside the evolved bounds, cap to evolved values
MAX_TP_CAP = 0.25   # never set TP higher than 25%
MIN_SL_FLOOR = 0.02  # never set SL tighter than 2%

# Pullback entry: price must not have extended more than this % from EMA50
PULLBACK_EMA_PCT = 0.05   # 5% — avoids chasing moves already far from the average


# ── technical helpers ──────────────────────────────────────────────────────────

def _ema_weighted(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


def _ichimoku_cloud(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
    """
    Ichimoku Kinko Hyo for daily bars.
    Tenkan (9), Kijun (26), Senkou A, Senkou B (52).

    Returns cloud position relative to current price:
      'above'  = price above cloud → bullish macro structure
      'below'  = price below cloud → bearish macro structure
      'inside' = price inside cloud → caution, reduce confidence

    Also returns tk_cross:
      'bull' = Tenkan > Kijun (bullish momentum)
      'bear' = Tenkan < Kijun (bearish momentum)
    """
    def mid(h, l, n):
        if len(h) < n:
            return (float(h[-1]) + float(l[-1])) / 2.0
        return (float(np.max(h[-n:])) + float(np.min(l[-n:]))) / 2.0

    tenkan   = mid(highs, lows, 9)
    kijun    = mid(highs, lows, 26)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = mid(highs, lows, 52)

    cloud_top    = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    price = float(closes[-1])

    if price > cloud_top:
        pos = "above"
    elif price < cloud_bottom:
        pos = "below"
    else:
        pos = "inside"

    return {
        "position":  pos,
        "tk_cross":  "bull" if tenkan > kijun else "bear",
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
    }


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return float(closes[-1]) * 0.02
    tr_list = []
    for i in range(1, len(closes)):
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    n = min(period, len(tr_list))
    return float(np.mean(tr_list[-n:])) if tr_list else float(closes[-1]) * 0.015


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """
    Average Directional Index (0-100).
    > 25 = strong trend, < 20 = choppy/ranging.
    """
    if len(closes) < period + 5:
        return 15.0   # not enough data → assume weak trend

    dm_plus, dm_minus, tr_list = [], [], []
    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i - 1]
        low_diff  = lows[i - 1] - lows[i]
        dm_plus.append(high_diff if high_diff > low_diff and high_diff > 0 else 0.0)
        dm_minus.append(low_diff if low_diff > high_diff and low_diff > 0 else 0.0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    # Use simple rolling sum for smoothing (Wilder's method approximation)
    def _smooth(arr, p):
        result = []
        s = float(np.sum(arr[:p]))
        result.append(s)
        for x in arr[p:]:
            s = s - s / p + x
            result.append(s)
        return result

    n = len(tr_list)
    if n < period:
        return 15.0

    sm_tr   = _smooth(tr_list,   period)
    sm_dmp  = _smooth(dm_plus,   period)
    sm_dmn  = _smooth(dm_minus,  period)

    dx_list = []
    for tr, dmp, dmn in zip(sm_tr, sm_dmp, sm_dmn):
        if tr <= 0:
            continue
        di_plus  = 100 * dmp / tr
        di_minus = 100 * dmn / tr
        denom = di_plus + di_minus
        if denom <= 0:
            continue
        dx_list.append(100 * abs(di_plus - di_minus) / denom)

    if not dx_list:
        return 15.0

    # ADX = smoothed DX over last `period` DX values
    recent = dx_list[-period:]
    return float(np.mean(recent))


# ── main class ─────────────────────────────────────────────────────────────────

class PassiveStrategy:
    """
    Passive allocation strategy for multi-day trend positions.

    Signal improvements vs. basic version:
      - ADX filter: skips choppy markets
      - Volume confirmation: entry only on high-participation bars
      - ATR-based TP/SL: dynamically sized to current volatility
      - Confidence decay: confidence penalised when market is fragile
    """

    # 4H uses shorter EMAs for entry timing (more frequent signals)
    # 1D uses the evolved longer EMAs for long-term trend confirmation
    _EMA_FAST_4H = 20
    _EMA_SLOW_4H = 50

    def __init__(self):
        self.market    = MarketData()
        self.timeframes = PASSIVE_TIMEFRAMES
        self.take_profit = PASSIVE_TAKE_PROFIT
        self.stop_loss   = PASSIVE_STOP_LOSS
        self.min_hold_hours = PASSIVE_MIN_HOLD_HOURS
        self.max_hold_hours = PASSIVE_MAX_HOLD_HOURS
        self._ema_fast  = 50    # 1D trend EMA (evolved)
        self._ema_slow  = 200   # 1D trend EMA (evolved)

    # ── evolution hook ─────────────────────────────────────────────────────────

    def update_params(self, params: dict) -> None:
        if "take_profit"   in params: self.take_profit   = float(params["take_profit"])
        if "stop_loss"     in params: self.stop_loss     = float(params["stop_loss"])
        if "ema_fast"      in params: self._ema_fast     = int(params["ema_fast"])
        if "ema_slow"      in params: self._ema_slow     = int(params["ema_slow"])
        if "max_hold_days" in params: self.max_hold_hours = int(params["max_hold_days"]) * 24

    # ── internal helpers ───────────────────────────────────────────────────────

    def _trend_alignment(self, closes: np.ndarray) -> str:
        """1D long-term trend: evolved EMA50/200."""
        ema_f = _ema_weighted(closes, self._ema_fast)
        ema_s = _ema_weighted(closes, self._ema_slow)
        if ema_f > ema_s:
            return "LONG"
        if ema_f < ema_s:
            return "SHORT"
        return "HOLD"

    def _weekly_trend(self, symbol: str) -> str:
        """
        Weekly EMA10/30 trend direction.
        Acts as the highest-timeframe filter:
          LONG  = weekly uptrend → allow passive LONGs; SHORT only with strong signal
          SHORT = weekly downtrend → allow passive SHORTs; LONGs need extra confirmation
          HOLD  = weekly neutral
        Cached per call (1W bars change slowly).
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="1w", limit=40)
            if len(df) < 15:
                return "HOLD"
            closes = df["close"].values
            ema10 = _ema_weighted(closes, 10)
            ema30 = _ema_weighted(closes, min(30, len(closes)))
            if ema10 > ema30:
                return "LONG"
            if ema10 < ema30:
                return "SHORT"
        except Exception:
            pass
        return "HOLD"

    def _trend_alignment_4h(self, closes: np.ndarray) -> str:
        """
        4H entry timing: EMA20/50.
        Faster than 1D EMA50/200 — confirms that the coin is moving in
        the 1D direction right NOW, not just at some point in history.
        """
        ema_f = _ema_weighted(closes, self._EMA_FAST_4H)
        ema_s = _ema_weighted(closes, self._EMA_SLOW_4H)
        if ema_f > ema_s:
            return "LONG"
        if ema_f < ema_s:
            return "SHORT"
        return "HOLD"

    def _pullback_ok(self, closes_4h: np.ndarray, side: str) -> bool:
        """
        Pullback entry filter: only enter when price is within PULLBACK_EMA_PCT
        of EMA50.  Prevents chasing breakouts that have already run far from the
        moving average — waits for a re-test instead.
        """
        ema50 = _ema_weighted(closes_4h, self._ema_fast)
        price = float(closes_4h[-1])
        if ema50 <= 0:
            return True
        pct_dev = (price - ema50) / ema50
        if side == "LONG":
            return pct_dev <= PULLBACK_EMA_PCT   # not more than 5% above EMA50
        return pct_dev >= -PULLBACK_EMA_PCT      # not more than 5% below EMA50

    def _volume_confirms(self, volumes: np.ndarray) -> bool:
        """
        Passive volume check: ensure the market is not completely dead.

        Passive is a multi-day hold — it does NOT need a volume surge at entry.
        The coin scanner (MIN_24H_VOLUME_USDT) already ensures we only trade
        liquid markets. Here we only block if the last 5 bars average is below
        10% of the 30-bar baseline (genuine illiquidity / halt event).
        """
        if len(volumes) < 10:
            return True
        n_recent   = min(5, len(volumes) - 1)
        n_baseline = min(30, len(volumes) - n_recent)
        recent   = float(np.mean(volumes[-(n_recent + 1):-1]))
        baseline = float(np.mean(volumes[-(n_recent + n_baseline + 1):-(n_recent + 1)]))
        if baseline <= 0:
            return True
        return recent >= baseline * 0.10   # only block if < 10% normal volume

    def _compute_atr_levels(self, highs, lows, closes, price, side):
        """
        Compute ATR-based TP and SL prices.
        Falls back to evolved fixed-% if ATR is abnormally small or large.
        """
        daily_atr = _atr(highs, lows, closes, period=14)
        atr_pct = daily_atr / price if price > 0 else self.stop_loss

        # Clip ATR-derived SL to [MIN_SL_FLOOR, evolved stop_loss × 2]
        sl_pct = float(np.clip(ATR_SL_MULT * atr_pct, MIN_SL_FLOOR, self.stop_loss * 2))
        tp_pct = float(np.clip(ATR_TP_MULT * atr_pct, sl_pct * 1.5, MAX_TP_CAP))

        if side == "LONG":
            return (
                round(price * (1 + tp_pct), 6),
                round(price * (1 - sl_pct), 6),
                round(tp_pct, 6),
                round(sl_pct, 6),
            )
        else:
            return (
                round(price * (1 - tp_pct), 6),
                round(price * (1 + sl_pct), 6),
                round(tp_pct, 6),
                round(sl_pct, 6),
            )

    def _build_signal(self, symbol, side, price, regime, tp=None, sl=None, confidence=0.85) -> dict:
        if side == "HOLD":
            return {"symbol": symbol, "side": "HOLD", "confidence": 0.0, "regime": regime}
        return {
            "symbol":        symbol,
            "side":          side,
            "confidence":    round(confidence, 4),
            "regime":        regime,
            "entry_price":   round(price, 6),
            "take_profit":   tp,
            "stop_loss":     sl,
            "min_hold_hours": self.min_hold_hours,
            "max_hold_hours": self.max_hold_hours,
        }

    # ── public entry point ─────────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        market_state: dict,
        mode: str = "auto",
        market_intelligence: dict | None = None,
    ) -> dict:
        """
        Generate trade signal based on EMA trend alignment + quality filters.

        mode : "long"  → only emit LONG signals
               "short" → only emit SHORT signals
               "skip"  → always HOLD
               "auto"  → no restriction
        """
        regime = market_state.get("regime", "sideways")

        if mode == "skip":
            return self._build_signal(symbol, "HOLD", 0.0, regime)

        if regime in PASSIVE_REGIME_BLACKLIST and mode == "auto":
            return self._build_signal(symbol, "HOLD", 0.0, regime)

        try:
            df_4h = self.market.get_ohlcv_df(symbol, timeframe="4h", limit=260)
            df_1d = self.market.get_ohlcv_df(symbol, timeframe="1d", limit=260)
            if len(df_4h) < 60 or len(df_1d) < 60:
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            closes_4h = df_4h["close"].values
            closes_1d = df_1d["close"].values
            highs_1d  = df_1d["high"].values
            lows_1d   = df_1d["low"].values

            side_4h = self._trend_alignment_4h(closes_4h)
            side_1d = self._trend_alignment(closes_1d)
            if side_1d not in ("LONG", "SHORT"):
                return self._build_signal(symbol, "HOLD", 0.0, regime)
            if side_4h not in (side_1d, "HOLD"):
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            side = side_1d

            # Weekly trend check: hard block only when weekly strongly opposes AND ADX is weak.
            # When ADX is strong (>25), allow the trade but reduce confidence.
            weekly = self._weekly_trend(symbol)

            if mode == "long"  and side != "LONG":
                return self._build_signal(symbol, "HOLD", 0.0, regime)
            if mode == "short" and side != "SHORT":
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            # In BEAR regime with extreme fear, prices may already be 10-20% below EMA50.
            # Relaxing the pullback filter lets us SHORT strongly trending coins
            # rather than waiting for a bounce that may never come.
            _bear_mode = regime in ("bear", "bear_trend") or mode == "short"
            _intel_panic = (market_intelligence or {}).get("panic", False)
            if _bear_mode and mode == "short":
                pullback_pct = 0.20 if _intel_panic else 0.15
            elif mode in ("long", "short"):
                pullback_pct = 0.08
            else:
                pullback_pct = PULLBACK_EMA_PCT
            ema50 = _ema_weighted(closes_4h, self._EMA_SLOW_4H)
            price_now = float(closes_4h[-1])
            pct_dev = (price_now - ema50) / ema50 if ema50 > 0 else 0.0
            if side == "LONG" and pct_dev > pullback_pct:
                return self._build_signal(symbol, "HOLD", 0.0, regime)
            if side == "SHORT" and pct_dev < -pullback_pct:
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            adx = _adx(highs_1d, lows_1d, closes_1d, period=14)
            adx_min = ADX_MIN_THRESHOLD_TREND if mode in ("long", "short") else ADX_MIN_THRESHOLD
            if adx < adx_min:
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            volumes_4h = df_4h["volume"].values
            if not self._volume_confirms(volumes_4h):
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            # ── Ichimoku cloud (1D bars) ──────────────────────────────────────
            # Price in cloud = unclear structure; price vs cloud direction = macro bias
            ichi = _ichimoku_cloud(highs_1d, lows_1d, closes_1d)

            # Hard block: strong ADX in cloud means no clear trend — skip unless
            # ADX is very strong (> 30), which overrides the "inside" hesitation.
            if ichi["position"] == "inside" and adx < 30:
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            # Direction conflict: price above cloud but we want SHORT (or vice versa)
            if ichi["position"] == "above" and side == "SHORT" and adx < 25:
                return self._build_signal(symbol, "HOLD", 0.0, regime)
            if ichi["position"] == "below" and side == "LONG" and adx < 25:
                return self._build_signal(symbol, "HOLD", 0.0, regime)

            price = float(closes_4h[-1])

            # ── ATR-based TP / SL ────────────────────────────────────────────
            tp, sl, tp_pct, sl_pct = self._compute_atr_levels(
                highs_1d, lows_1d, closes_1d, price, side
            )

            # ── confidence scoring ───────────────────────────────────────────
            # Base 0.85; bonus for strong ADX + Ichimoku; penalty for conflict
            confidence = 0.85
            confidence += min(0.08, (adx - ADX_MIN_THRESHOLD) / 100)

            # Weekly alignment: boost when all 3 TFs agree, penalise when they conflict.
            # Never hard-block here — let confidence decide (ADX is the hard gate above).
            if weekly == side:
                confidence += 0.07   # 3-timeframe alignment = strong conviction
            elif weekly not in ("HOLD",) and weekly != side:
                confidence -= 0.12   # weekly opposes — reduce conviction significantly

            # Ichimoku bonuses / penalties
            if ichi["position"] == "above" and side == "LONG":
                confidence += 0.06   # price above cloud = bullish confirmation
                if ichi["tk_cross"] == "bull":
                    confidence += 0.03  # TK bullish cross adds further conviction
            elif ichi["position"] == "below" and side == "SHORT":
                confidence += 0.06
                if ichi["tk_cross"] == "bear":
                    confidence += 0.03
            elif ichi["position"] == "inside":
                confidence -= 0.08   # inside cloud = conflicting signals

            if market_intelligence:
                if market_intelligence.get("bear_breadth") and side == "LONG":
                    confidence -= 0.10
                if market_intelligence.get("bull_breadth") and side == "LONG":
                    confidence += 0.05
                if market_intelligence.get("panic") and side == "LONG":
                    confidence += 0.06
                if market_intelligence.get("euphoria") and side == "SHORT":
                    confidence += 0.06
                # BTC cycle phase: boost confidence in early_bull/bull phases for LONG
                cycle = market_intelligence.get("btc_cycle", {})
                if cycle.get("cycle_phase") in ("early_bull", "bull") and side == "LONG":
                    confidence += 0.04
                if cycle.get("cycle_phase") in ("bear", "distribution") and side == "LONG":
                    confidence -= 0.05

            confidence = round(float(np.clip(confidence, 0.50, 0.95)), 4)

            return self._build_signal(symbol, side, price, regime, tp, sl, confidence)

        except Exception as exc:
            # Only log if not a routine data-unavailable error (rate limit noise)
            msg = str(exc)
            if "No OHLCV" not in msg and "No ticker" not in msg:
                print(f"[PassiveStrategy] {symbol} error: {exc}")

        return self._build_signal(symbol, "HOLD", 0.0, regime)
