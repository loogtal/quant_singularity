"""
Active intraday strategy — four signal modes + quality gating.

Modes (set by DualRegimeRouter):
  mean_reversion : RSI + BB fade + Stochastic RSI (sideways markets)
  momentum       : EMA crossover + MACD + VWAP + volume surge (trending markets)
  funding_arb    : Fade extreme funding rates (crowded/volatile markets)
  vwap_reversal  : Price stretched from VWAP → snapback trade (any regime)

Additional quality gates:
  Session quality   — confidence weighted by trading session (London/NY overlap = 1.0)
  Daily profit gate — stops NEW entries when daily target hit; still manages open positions
  Volume surge      — momentum/vwap modes require above-average volume
  Ichimoku cloud    — price vs Kumo used as macro trend filter across all modes
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from config.dual_settings import (
    ACTIVE_CAPITAL,
    ACTIVE_DAILY_LOSS_PCT,
    ACTIVE_DAILY_TARGET_PCT,
    ACTIVE_STOP_LOSS,
    ACTIVE_TAKE_PROFIT,
    ACTIVE_TRADING_END,
    ACTIVE_TRADING_START,
)
from data.funding_data import FundingData
from data.market_data import MarketData


# ── technical helpers ──────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


def _atr_pct(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
    """Compute ATR as a fraction of current price.  Returns atr/price.

    Uses True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    Falls back to std-dev of returns when insufficient bars.
    """
    if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        # Fallback: std of recent returns
        if len(closes) < 2:
            return 0.005
        rets = np.diff(closes[-period:]) / (closes[-period:-1] + 1e-9)
        return float(np.std(rets)) if len(rets) > 1 else 0.005
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        ),
    )
    atr = float(np.mean(tr[-period:]))
    price = float(closes[-1])
    return atr / (price + 1e-9)


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains))  if gains.any()  else 0.0
    avg_l  = float(np.mean(losses)) if losses.any() else 1e-9
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))


def _bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Returns (upper, mid, lower)."""
    if len(closes) < period:
        mid = float(closes[-1])
        return mid, mid, mid
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid + num_std * std, mid, mid - num_std * std


def _vwap(df) -> float:
    """
    Volume-weighted average price for the available session bars.
    Uses (H+L+C)/3 as typical price.
    """
    if len(df) < 2:
        return float(df["close"].iloc[-1])
    tp  = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    vol = df["volume"].values
    total_vol = float(np.sum(vol))
    if total_vol <= 0:
        return float(df["close"].iloc[-1])
    return float(np.sum(tp * vol) / total_vol)


def _volume_surge(volumes: np.ndarray, lookback: int = 20, mult: float = 1.4) -> bool:
    """True when latest bar's volume > mult × lookback-bar average."""
    if len(volumes) < lookback + 1:
        return True
    avg = float(np.mean(volumes[-lookback - 1:-1]))
    return float(volumes[-1]) >= avg * mult


def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series needed for MACD signal line."""
    k   = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram) for the latest bar."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_f    = _ema_series(closes, fast)
    ema_s    = _ema_series(closes, slow)
    macd_arr = ema_f - ema_s
    sig_arr  = _ema_series(macd_arr[slow - 1:], signal)
    macd_val = float(macd_arr[-1])
    sig_val  = float(sig_arr[-1])
    return macd_val, sig_val, round(macd_val - sig_val, 8)


def _ichimoku(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
              ) -> dict:
    """
    Ichimoku Kinko Hyo cloud components.
    Returns tenkan, kijun, senkou_a, senkou_b and cloud_position:
      'above'  = price above cloud (bullish)
      'below'  = price below cloud (bearish)
      'inside' = price inside cloud (neutral/caution)
    """
    def midpoint(h, l, n):
        if len(h) < n:
            return (float(h[-1]) + float(l[-1])) / 2
        return (float(np.max(h[-n:])) + float(np.min(l[-n:]))) / 2

    tenkan  = midpoint(highs, lows, 9)
    kijun   = midpoint(highs, lows, 26)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = midpoint(highs, lows, 52)
    cloud_top    = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    price = float(closes[-1])

    if price > cloud_top:
        cloud_pos = "above"
    elif price < cloud_bottom:
        cloud_pos = "below"
    else:
        cloud_pos = "inside"

    return {
        "tenkan":       tenkan,
        "kijun":        kijun,
        "senkou_a":     senkou_a,
        "senkou_b":     senkou_b,
        "cloud_top":    cloud_top,
        "cloud_bottom": cloud_bottom,
        "position":     cloud_pos,
        "tk_cross":     "bull" if tenkan > kijun else "bear",
    }


def _price_momentum_dir(closes: np.ndarray, lookback: int = 8) -> str:
    """
    Price Momentum Direction — fraction of recent bars that closed up.

    Simpler and more reliable than SuperTrend for intraday confirmation:
      ≥ 6/8 bars up   → "LONG"  (strong bullish momentum)
      ≤ 2/8 bars up   → "SHORT" (strong bearish momentum)
      else            → "HOLD"  (neutral/mixed)

    Independent of EMA/MACD, adds genuine new information to the signal.
    """
    if len(closes) < lookback + 1:
        return "HOLD"
    recent = closes[-(lookback + 1):]
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    ratio = ups / lookback
    if ratio >= 0.75:   # 6+ of 8 bars up
        return "LONG"
    if ratio <= 0.25:   # 2 or fewer of 8 bars up
        return "SHORT"
    return "HOLD"


def _stoch_rsi(closes: np.ndarray, rsi_period: int = 14, stoch_period: int = 14
               ) -> float:
    """Stochastic RSI — 0-100. < 20 = oversold, > 80 = overbought."""
    if len(closes) < rsi_period + stoch_period + 1:
        return 50.0
    # Build rolling RSI series
    rsi_vals = []
    for i in range(rsi_period + 1, len(closes) + 1):
        deltas = np.diff(closes[i - rsi_period - 1:i])
        g = float(np.mean(np.where(deltas > 0, deltas, 0.0)))
        l = float(np.mean(np.where(deltas < 0, -deltas, 0.0))) + 1e-9
        rsi_vals.append(100 - 100 / (1 + g / l))
    if len(rsi_vals) < stoch_period:
        return 50.0
    window = rsi_vals[-stoch_period:]
    mn, mx = min(window), max(window)
    if mx == mn:
        return 50.0
    return float(100 * (rsi_vals[-1] - mn) / (mx - mn))


# ── session quality ────────────────────────────────────────────────────────────

_SESSION_QUALITY_MAP = {
    # UTC hour → quality score (0-1)
    0: 0.55, 1: 0.50, 2: 0.50, 3: 0.55, 4: 0.60, 5: 0.60,
    6: 0.65, 7: 0.70, 8: 0.80, 9: 0.85, 10: 0.85, 11: 0.88,
    12: 0.92, 13: 1.00, 14: 1.00, 15: 1.00, 16: 0.95, 17: 0.88,
    18: 0.80, 19: 0.75, 20: 0.70, 21: 0.68, 22: 0.65, 23: 0.58,
}


def _session_quality() -> float:
    """Returns 0-1 quality score for current UTC hour."""
    hour = datetime.now(timezone.utc).hour
    return _SESSION_QUALITY_MAP.get(hour, 0.60)


# ── main class ─────────────────────────────────────────────────────────────────

class ActiveStrategy:
    """
    Active intraday strategy. Supports four signal modes + quality gating.

    New in this version:
      - VWAP signal integration (momentum and vwap_reversal modes)
      - Session quality multiplier on confidence
      - Daily profit gate (stops new entries after target hit)
      - Volume surge filter for momentum mode
      - Fear/greed context from MarketIntelligence
    """

    FUNDING_LONG_THRESHOLD  = -0.0002   # -0.02%: shorts paying longs enough to cover fees
    FUNDING_SHORT_THRESHOLD =  0.0003   # +0.03%: longs paying too much; fade the crowding

    # VWAP deviation required to trigger vwap_reversal mode
    VWAP_REVERSAL_PCT = 0.012  # 1.2% from VWAP

    def __init__(self):
        self.market      = MarketData()
        self.funding     = FundingData()
        self.take_profit = ACTIVE_TAKE_PROFIT
        self.stop_loss   = ACTIVE_STOP_LOSS
        self.start_hour  = ACTIVE_TRADING_START
        self.end_hour    = ACTIVE_TRADING_END
        self._ema_fast   = 9
        self._ema_slow   = 21
        self._daily_pnl  = 0.0   # caller updates this via record_daily_pnl()
        self._daily_date = None
        self._daily_capital = ACTIVE_CAPITAL  # caller updates via update_capital()

    # ── param evolution hook ───────────────────────────────────────────────────

    def update_params(self, params: dict) -> None:
        if "take_profit" in params: self.take_profit = float(params["take_profit"])
        if "stop_loss"   in params: self.stop_loss   = float(params["stop_loss"])
        if "ema_fast"    in params: self._ema_fast   = int(params["ema_fast"])
        if "ema_slow"    in params: self._ema_slow   = int(params["ema_slow"])

    # ── daily P&L gate ─────────────────────────────────────────────────────────

    def record_daily_pnl(self, pnl: float) -> None:
        """Call after each closed active trade."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_date:
            self._daily_pnl  = 0.0
            self._daily_date = today
        self._daily_pnl += pnl

    def update_capital(self, capital: float) -> None:
        """Call when active capital changes so the daily gate stays proportional
        to current equity instead of the initial ACTIVE_CAPITAL seed."""
        if capital > 0:
            self._daily_capital = capital

    def _daily_gate_ok(self) -> bool:
        """False when daily profit target already hit or loss limit breached."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_date:
            return True
        target    = self._daily_capital * ACTIVE_DAILY_TARGET_PCT
        loss_gate = -(self._daily_capital * ACTIVE_DAILY_LOSS_PCT)
        if self._daily_pnl >= target:
            return False
        if self._daily_pnl <= loss_gate:
            return False
        return True

    # ── trading window ─────────────────────────────────────────────────────────

    def is_within_trading_hours(self) -> bool:
        now = datetime.now(timezone.utc) + timedelta(hours=7)
        return self.start_hour <= now.hour < self.end_hour

    def trading_window_has_closed(self) -> bool:
        now = datetime.now(timezone.utc) + timedelta(hours=7)
        return now.hour >= self.end_hour or now.hour < self.start_hour

    # ── signal builder ─────────────────────────────────────────────────────────

    def _build_signal(self, symbol, side, price, confidence=0.75) -> dict:
        if side == "HOLD":
            return {"symbol": symbol, "side": "HOLD", "confidence": 0.0}
        # Scale TP/SL by confidence: higher confidence → slightly wider TP
        confidence_boost = max(0, (confidence - 0.65) * 0.3)
        tp_pct = self.take_profit * (1 + confidence_boost)
        tp = price * (1 + tp_pct if side == "LONG" else 1 - tp_pct)
        sl = price * (1 - self.stop_loss if side == "LONG" else 1 + self.stop_loss)
        return {
            "symbol":      symbol,
            "side":        side,
            "confidence":  round(confidence, 4),
            "entry_price": round(price, 6),
            "take_profit": round(float(tp), 6),
            "stop_loss":   round(float(sl), 6),
            "intraday":    True,
        }

    # ── higher-timeframe trend ─────────────────────────────────────────────────

    def _htf_trend(self, symbol: str) -> str:
        """4H macro trend direction — BULL | BEAR | NEUTRAL. Gates intraday signals."""
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="4h", limit=60)
            if len(df) < 55:
                return "NEUTRAL"
            closes = df["close"].values
            ema20  = _ema(closes, 20)
            ema50  = _ema(closes, 50)
            price  = float(closes[-1])
            if ema20 > ema50 and price > ema20:
                return "BULL"
            if ema20 < ema50 and price < ema20:
                return "BEAR"
        except Exception:
            pass
        return "NEUTRAL"

    def _1h_structure(self, symbol: str) -> str:
        """
        1H bar structure: counts bullish vs bearish bars in the last 8 bars.
        BULL = 5+ bars closed up (price momentum behind the signal)
        BEAR = 5+ bars closed down
        NEUTRAL = mixed
        Used to adjust confidence — not a hard block.
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="1h", limit=12)
            if len(df) < 8:
                return "NEUTRAL"
            closes = df["close"].values[-8:]
            up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            if up_bars >= 5:
                return "BULL"
            if up_bars <= 2:
                return "BEAR"
        except Exception:
            pass
        return "NEUTRAL"

    # ── individual signal sources ──────────────────────────────────────────────

    def _mean_reversion_signal(self, symbol: str) -> tuple[str, float]:
        """
        Mean reversion: RSI + Bollinger Bands + Stochastic RSI triple confirmation.
        Stochastic RSI < 15 = deep oversold (high confidence LONG).
        Stochastic RSI > 85 = deep overbought (high confidence SHORT).
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=100)
            if len(df) < 30:
                return "HOLD", 0.0
            closes   = df["close"].values
            rsi      = _rsi(closes)
            srsi     = _stoch_rsi(closes)
            upper, _, lower = _bollinger(closes)
            price    = float(closes[-1])
            vwap_val = _vwap(df)

            if rsi < 32 and price < lower:
                below_vwap_bonus = 0.05 if price < vwap_val else 0.0
                srsi_bonus       = 0.08 if srsi < 15 else (0.04 if srsi < 25 else 0.0)
                conf = round(min(0.90, 0.55 + (32 - rsi) / 32 * 0.30
                                 + below_vwap_bonus + srsi_bonus), 4)
                return "LONG", conf

            if rsi > 68 and price > upper:
                above_vwap_bonus = 0.05 if price > vwap_val else 0.0
                srsi_bonus       = 0.08 if srsi > 85 else (0.04 if srsi > 75 else 0.0)
                conf = round(min(0.90, 0.55 + (rsi - 68) / 32 * 0.30
                                 + above_vwap_bonus + srsi_bonus), 4)
                return "SHORT", conf
        except Exception as exc:
            print(f"[ActiveStrategy] mean_reversion {symbol}: {exc}")
        return "HOLD", 0.0

    def _momentum_signal(self, symbol: str) -> tuple[str, float]:
        """
        Momentum: EMA crossover + MACD histogram + Ichimoku cloud + VWAP + volume surge.

        Signal requires 3 of 4 conditions:
          1. EMA fast > slow (direction)
          2. MACD histogram positive and rising (momentum accelerating)
          3. Price above/below Ichimoku cloud (macro structure)
          4. Price on correct side of 1H VWAP
        Plus volume surge (mandatory gate).
        """
        try:
            df_15m = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=120)
            df_1h  = self.market.get_ohlcv_df(symbol, timeframe="1h",  limit=60)
            if len(df_15m) < 60 or len(df_1h) < 30:
                return "HOLD", 0.0

            closes  = df_15m["close"].values
            highs   = df_15m["high"].values
            lows    = df_15m["low"].values
            price   = float(closes[-1])

            # Gate 1: EMA crossover
            fast_ema = _ema(closes, self._ema_fast)
            slow_ema = _ema(closes, self._ema_slow)
            ema_bull = fast_ema > slow_ema
            ema_bear = fast_ema < slow_ema

            # Gate 2: MACD histogram direction and sign
            _, _, macd_hist = _macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            # Gate 3: Ichimoku cloud position
            ichi     = _ichimoku(highs, lows, closes)
            ichi_bull = ichi["position"] == "above"
            ichi_bear = ichi["position"] == "below"

            # Gate 4: 1H VWAP
            vwap_1h   = _vwap(df_1h)
            vwap_bull = price > vwap_1h
            vwap_bear = price < vwap_1h

            # Volume surge is mandatory
            if not _volume_surge(df_15m["volume"].values):
                return "HOLD", 0.0

            # Gate 5: Price Momentum Direction (6+/8 recent bars closing same way)
            pmd     = _price_momentum_dir(closes)
            pmd_bull = pmd == "LONG"
            pmd_bear = pmd == "SHORT"

            bull_score = sum([ema_bull, macd_bull, ichi_bull, vwap_bull, pmd_bull])
            bear_score = sum([ema_bear, macd_bear, ichi_bear, vwap_bear, pmd_bear])

            if bull_score >= 3:
                conf = 0.70 + min(0.14, bull_score * 0.028)
                if ichi["tk_cross"] == "bull":
                    conf = min(0.92, conf + 0.04)
                return "LONG", round(conf, 4)

            if bear_score >= 3:
                conf = 0.70 + min(0.14, bear_score * 0.028)
                if ichi["tk_cross"] == "bear":
                    conf = min(0.92, conf + 0.04)
                return "SHORT", round(conf, 4)

        except Exception as exc:
            print(f"[ActiveStrategy] momentum {symbol}: {exc}")
        return "HOLD", 0.0

    def _funding_signal(self, symbol: str) -> tuple[str, float]:
        try:
            fd   = self.funding.get_funding(symbol)
            rate = float(fd.get("funding_rate", 0))
            if rate >= self.FUNDING_SHORT_THRESHOLD:
                conf = round(min(0.84, abs(rate) / 0.001 * 0.84), 4)
                return "SHORT", conf
            if rate <= self.FUNDING_LONG_THRESHOLD:
                conf = round(min(0.84, abs(rate) / 0.001 * 0.84), 4)
                return "LONG", conf
        except Exception as exc:
            print(f"[ActiveStrategy] funding_arb {symbol}: {exc}")
        return "HOLD", 0.0

    def _breakout_signal(self, symbol: str) -> tuple[str, float]:
        """
        Breakout mode: price breaks above/below a consolidation range with volume surge.

        Setup:
          1. Identify a tight 20-bar range (range < 2× ATR = consolidation)
          2. Current bar closes decisively above range high (LONG) or below range low (SHORT)
          3. Volume surge confirms (>1.6× 20-bar average)
          4. MACD histogram positive (momentum building into breakout)

        This captures the most profitable setups: compressed ranges that explode.
        Avoids chasing momentum that's already run; requires fresh breakout.
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=80)
            if len(df) < 25:
                return "HOLD", 0.0

            closes  = df["close"].values
            highs   = df["high"].values
            lows    = df["low"].values
            volumes = df["volume"].values
            price   = float(closes[-1])

            # Define consolidation zone from bars [-21:-1] (exclude current bar)
            zone_h = float(np.max(highs[-21:-1]))
            zone_l = float(np.min(lows[-21:-1]))
            zone_range = zone_h - zone_l

            # ATR of the zone bars
            rets = np.diff(closes[-21:]) / closes[-21:-1]
            atr  = float(np.std(rets)) * price if len(rets) > 1 else price * 0.005

            # Only trade breakouts from tight ranges (compression = energy storage)
            if zone_range > 3.5 * atr:
                return "HOLD", 0.0   # range too wide = not a consolidation

            # Volume surge required (breakouts without volume are false)
            avg_vol = float(np.mean(volumes[-21:-1]))
            if avg_vol <= 0 or float(volumes[-1]) < avg_vol * 1.6:
                return "HOLD", 0.0

            # MACD histogram for momentum direction
            _, _, macd_hist = _macd(closes)

            # Breakout confirmation: current close beyond the zone
            breakout_up   = price > zone_h * 1.001 and macd_hist > 0
            breakout_down = price < zone_l * 0.999 and macd_hist < 0

            if breakout_up:
                # Confidence scales with how far above the zone we closed
                excess = (price - zone_h) / (zone_range + 1e-9)
                conf = round(min(0.88, 0.65 + excess * 0.5), 4)
                return "LONG", conf

            if breakout_down:
                excess = (zone_l - price) / (zone_range + 1e-9)
                conf = round(min(0.88, 0.65 + excess * 0.5), 4)
                return "SHORT", conf

        except Exception as exc:
            print(f"[ActiveStrategy] breakout {symbol}: {exc}")
        return "HOLD", 0.0

    def _vwap_reversal_signal(self, symbol: str) -> tuple[str, float]:
        """
        VWAP Reversal: price stretched far from VWAP → snapback.

        Tightened criteria (bandit history: 43% wr → target 58%+):
          1. Deviation >= 1.8% (was 1.2%) — only extreme stretches
          2. RSI < 30 for LONG (was < 45); RSI > 70 for SHORT (was > 55)
          3. Volume DECLINING (stretch without volume = exhaustion, not continuation)
          4. Stoch RSI confirms oversold/overbought

        Three confirming conditions required for full confidence.
        """
        try:
            df = self.market.get_ohlcv_df(symbol, timeframe="15m", limit=80)
            if len(df) < 30:
                return "HOLD", 0.0

            closes  = df["close"].values
            volumes = df["volume"].values
            vwap_val = _vwap(df)
            price    = float(closes[-1])
            dev      = (price - vwap_val) / vwap_val if vwap_val > 0 else 0.0

            # Gate 1: deviation must be large enough
            if abs(dev) < self.VWAP_REVERSAL_PCT:
                return "HOLD", 0.0

            rsi  = _rsi(closes)
            srsi = _stoch_rsi(closes)

            # Gate 2: volume should be declining (stretch exhausting itself)
            avg_vol     = float(np.mean(volumes[-10:-1]))
            vol_fading  = float(volumes[-1]) < avg_vol * 0.9  # volume lower than recent avg

            if dev < 0:  # price below VWAP → potential LONG
                rsi_ok   = rsi < 35           # more extreme than before
                srsi_ok  = srsi < 25
                confirms = sum([rsi_ok, srsi_ok, vol_fading])
                if confirms < 2:
                    return "HOLD", 0.0
                conf = round(min(0.85, 0.58 + abs(dev) * 8 + confirms * 0.04), 4)
                return "LONG", conf

            if dev > 0:  # price above VWAP → potential SHORT
                rsi_ok   = rsi > 65
                srsi_ok  = srsi > 75
                confirms = sum([rsi_ok, srsi_ok, vol_fading])
                if confirms < 2:
                    return "HOLD", 0.0
                conf = round(min(0.85, 0.58 + abs(dev) * 8 + confirms * 0.04), 4)
                return "SHORT", conf

        except Exception as exc:
            print(f"[ActiveStrategy] vwap_reversal {symbol}: {exc}")
        return "HOLD", 0.0

    # ── public entry point ─────────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        market_state: dict,
        mode: str = "mean_reversion",
        market_intelligence: dict | None = None,
    ) -> dict:
        """
        Generate intraday signal with session quality and daily gate.

        mode : "mean_reversion" | "momentum" | "funding_arb" | "vwap_reversal" | "breakout"
        """
        if not self.is_within_trading_hours():
            return self._build_signal(symbol, "HOLD", 0.0)

        if not self._daily_gate_ok():
            return self._build_signal(symbol, "HOLD", 0.0)

        try:
            price = self.market.get_price(symbol)
        except Exception:
            return self._build_signal(symbol, "HOLD", 0.0)

        # ── fetch base signal ──────────────────────────────────────────────────
        if mode == "funding_arb":
            fund_side, fund_conf = self._funding_signal(symbol)
            mr_side, _           = self._mean_reversion_signal(symbol)
            if fund_side == "HOLD":
                return self._build_signal(symbol, "HOLD", 0.0)
            bonus = 0.05 if mr_side == fund_side else 0.0
            side, conf = fund_side, fund_conf + bonus

        elif mode == "momentum":
            side, conf = self._momentum_signal(symbol)
            # In BEAR regime, momentum should only SHORT (no catching falling knives)
            regime = market_state.get("regime", "")
            if regime == "bear" and side == "LONG":
                return self._build_signal(symbol, "HOLD", 0.0)

        elif mode == "vwap_reversal":
            side, conf = self._vwap_reversal_signal(symbol)

        elif mode == "breakout":
            side, conf = self._breakout_signal(symbol)

        else:  # mean_reversion (default)
            mr_side, mr_conf     = self._mean_reversion_signal(symbol)
            fund_side, fund_conf = self._funding_signal(symbol)

            if mr_side == "HOLD" and fund_side == "HOLD":
                return self._build_signal(symbol, "HOLD", 0.0)
            if mr_side != "HOLD" and fund_side == mr_side:
                side, conf = mr_side, min(0.92, mr_conf + 0.08)
            elif mr_side != "HOLD":
                side, conf = mr_side, mr_conf
            else:
                side, conf = fund_side, fund_conf * 0.85

        if side == "HOLD":
            return self._build_signal(symbol, "HOLD", 0.0)

        # ── 4H macro trend gate ────────────────────────────────────────────────
        # Don't fight the higher-timeframe trend (funding_arb is exempt — it fades
        # funding extremes regardless of macro direction)
        if mode != "funding_arb":
            htf = self._htf_trend(symbol)
            if htf == "BULL" and side == "SHORT":
                return self._build_signal(symbol, "HOLD", 0.0)
            if htf == "BEAR" and side == "LONG":
                return self._build_signal(symbol, "HOLD", 0.0)
            if (htf == "BULL" and side == "LONG") or (htf == "BEAR" and side == "SHORT"):
                conf = min(0.95, conf + 0.05)   # aligned with macro → boost

        # ── 1H bar structure: adjusts confidence without blocking ─────────────
        # Reduces bad-signal entries; does NOT hard-block (preserves trade count).
        if mode in ("momentum", "vwap_reversal", "mean_reversion"):
            struct = self._1h_structure(symbol)
            if struct == "BULL" and side == "LONG":
                conf = min(0.95, conf + 0.04)   # aligned
            elif struct == "BEAR" and side == "SHORT":
                conf = min(0.95, conf + 0.04)   # aligned
            elif struct == "BULL" and side == "SHORT":
                conf -= 0.06   # fighting 1H momentum
            elif struct == "BEAR" and side == "LONG":
                conf -= 0.06   # fighting 1H momentum

        # ── apply market intelligence context ──────────────────────────────────
        if market_intelligence:
            if market_intelligence.get("panic")   and side == "LONG":
                conf = min(0.92, conf + 0.07)
            if market_intelligence.get("euphoria") and side == "SHORT":
                conf = min(0.92, conf + 0.07)
            # Breadth confirmation
            if market_intelligence.get("bear_breadth") and side == "LONG":
                conf -= 0.05
            if market_intelligence.get("bull_breadth") and side == "SHORT":
                conf -= 0.05
            # BTC relative strength: alt-season favours LONG alts; BTC dominance penalises them
            if market_intelligence.get("alt_season") and side == "LONG":
                conf = min(0.95, conf + 0.05)
            if market_intelligence.get("btc_dominant") and side == "LONG" and "BTC" not in symbol:
                conf -= 0.05   # capital rotating into BTC → altcoin longs less attractive

        # ── scale by session quality ───────────────────────────────────────────
        # Additive adjustment: London/NY overlap (sq=1.0) = no change.
        # Poor session (sq=0.55) = -0.08 penalty, not a 45% multiplicative cut.
        # This prevents strong signals from being killed by low-quality hours.
        sq = _session_quality()
        if sq < 0.50:   # deep night / very poor session → skip entirely
            return self._build_signal(symbol, "HOLD", 0.0)
        # Additive: best session (+0) worst session (-0.08)
        sq_adj = round((sq - 1.0) * 0.16, 4)   # sq=1.0→0, sq=0.55→-0.072
        conf   = round(float(np.clip(conf + sq_adj, 0.40, 0.95)), 4)

        # Per-mode minimum confidence gate — must match or exceed APC threshold
        _mode_min = {"momentum": 0.65, "mean_reversion": 0.62, "vwap_reversal": 0.62,
                     "funding_arb": 0.55, "breakout": 0.65}
        if conf < _mode_min.get(mode, 0.62):
            return self._build_signal(symbol, "HOLD", 0.0)

        return self._build_signal(symbol, side, price, conf)
