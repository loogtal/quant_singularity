"""
SIDEWAYS + High-Breadth Edge Test
==================================
Tests the hypothesis that "extreme fear + bullish breadth" (regime=SIDEWAYS,
breadth>65% of coins above EMA50) is actually a capitulation/dip-buy setup,
not a true range-bound market — and that momentum-LONG (and passive long)
has positive edge there, unlike regular SIDEWAYS.

Compares momentum-LONG-only (TP=1.5%/SL=0.7%) and forward returns in:
  bucket A: regime=SIDEWAYS (BTC) & breadth > 0.65   (the "dip in uptrend" case)
  bucket B: regime=SIDEWAYS (BTC) & breadth <= 0.65  (true range-bound — current router default)

Usage:
    python scripts/backtest_sideways_breadth.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import ccxt
import numpy as np

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "LINK/USDT:USDT",
    "AVAX/USDT:USDT", "LTC/USDT:USDT", "TRX/USDT:USDT", "DOT/USDT:USDT",
]
TIMEFRAME = "15m"
LIMIT = 1500
TAKER_FEE = 0.0004
TP, SL = 0.015, 0.007
SLOPE_THRESH = 0.003
BREADTH_THRESH = 0.65


def build_exchange() -> ccxt.Exchange:
    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30_000})
    ex.load_markets()
    return ex


def fetch_closes(ex, symbol) -> np.ndarray | None:
    try:
        raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        if not raw or len(raw) < 200:
            print(f"  [skip] {symbol}: insufficient bars")
            return None
        return np.array(raw, dtype=float)
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None


def ema_series(closes: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.empty_like(closes)
    result[0] = closes[0]
    for i in range(1, len(closes)):
        result[i] = alpha * closes[i] + (1 - alpha) * result[i - 1]
    return result


def slope_series(closes: np.ndarray, window: int = 20) -> np.ndarray:
    ema50 = ema_series(closes, 50)
    slopes = np.zeros_like(ema50)
    for i in range(window, len(ema50)):
        slopes[i] = (ema50[i] - ema50[i - window]) / (ema50[i - window] + 1e-12)
    return slopes


def simulate_momentum_long(ohlcv: np.ndarray, mask: np.ndarray) -> list[float]:
    """LONG-only momentum: enter LONG whenever mask is True, TP=1.5%/SL=0.7%."""
    closes = ohlcv[:, 4]
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    trades: list[float] = []
    i = 0
    while i < len(ohlcv) - 1:
        if not mask[i]:
            i += 1
            continue
        entry = closes[i]
        j = i + 1
        while j < len(ohlcv):
            if highs[j] >= entry * (1 + TP):
                trades.append(TP - 2 * TAKER_FEE)
                break
            if lows[j] <= entry * (1 - SL):
                trades.append(-SL - 2 * TAKER_FEE)
                break
            j += 1
        else:
            last = closes[-1]
            trades.append((last - entry) / entry - 2 * TAKER_FEE)
        i = j + 1
    return trades


def forward_returns(closes: np.ndarray, mask: np.ndarray, horizon: int) -> list[float]:
    rets = []
    for i in range(len(closes) - horizon):
        if mask[i]:
            rets.append((closes[i + horizon] - closes[i]) / closes[i])
    return rets


def summarize(trades: list[float], label: str) -> None:
    if not trades:
        print(f"  {label}: 0 trades")
        return
    arr = np.array(trades)
    wr = (arr > 0).mean() * 100
    total = arr.sum()
    sharpe = arr.mean() / (arr.std() + 1e-12) * np.sqrt(len(arr))
    print(f"  {label}: n={len(arr):4d}  WR={wr:5.1f}%  total_pnl={total:+.4f}  "
          f"sharpe={sharpe:+.2f}  avg={arr.mean():+.5f}")


def summarize_returns(rets: list[float], label: str) -> None:
    if not rets:
        print(f"  {label}: 0 samples")
        return
    arr = np.array(rets)
    print(f"  {label}: n={len(arr):4d}  mean={arr.mean():+.4%}  "
          f"pos%={100*(arr>0).mean():.1f}%  median={np.median(arr):+.4%}")


def main():
    ex = build_exchange()
    print(f"Fetching {TIMEFRAME} OHLCV for {len(SYMBOLS)} symbols (limit={LIMIT})...")

    data = {}
    for sym in SYMBOLS:
        arr = fetch_closes(ex, sym)
        if arr is not None:
            data[sym] = arr

    min_len = min(len(v) for v in data.values())
    print(f"Loaded {len(data)} symbols, aligned length={min_len}\n")

    closes = {sym: arr[-min_len:, 4] for sym, arr in data.items()}
    ema50 = {sym: ema_series(c, 50) for sym, c in closes.items()}

    # Breadth: fraction of symbols with close > EMA50 at each bar
    breadth = np.zeros(min_len)
    for sym in closes:
        breadth += (closes[sym] > ema50[sym]).astype(float)
    breadth /= len(closes)

    # Reference regime via BTC EMA50 slope
    ref = "BTC/USDT:USDT"
    btc_slope = slope_series(closes[ref])
    sideways_mask = np.abs(btc_slope) <= SLOPE_THRESH

    bucket_a = sideways_mask & (breadth > BREADTH_THRESH)   # SIDEWAYS + bullish breadth
    bucket_b = sideways_mask & (breadth <= BREADTH_THRESH)  # SIDEWAYS + normal/bearish breadth

    print(f"SIDEWAYS bars: {sideways_mask.sum()} / {min_len}")
    print(f"  bucket A (breadth>{BREADTH_THRESH:.0%}): {bucket_a.sum()} bars")
    print(f"  bucket B (breadth<={BREADTH_THRESH:.0%}): {bucket_b.sum()} bars\n")

    print("=== Forward returns (does price drift up after this bar?) ===")
    for horizon, hlabel in [(4, "1h"), (16, "4h"), (32, "8h")]:
        for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
            rets_a = forward_returns(closes[sym], bucket_a, horizon)
            rets_b = forward_returns(closes[sym], bucket_b, horizon)
            summarize_returns(rets_a, f"{sym:18s} A({hlabel})")
            summarize_returns(rets_b, f"{sym:18s} B({hlabel})")
        print()

    print("=== Momentum LONG-only simulation (TP=1.5% SL=0.7%) ===")
    all_trades_a, all_trades_b = [], []
    for sym, ohlcv in data.items():
        ohlcv_aligned = ohlcv[-min_len:]
        trades_a = simulate_momentum_long(ohlcv_aligned, bucket_a)
        trades_b = simulate_momentum_long(ohlcv_aligned, bucket_b)
        all_trades_a += trades_a
        all_trades_b += trades_b

    summarize(all_trades_a, "Bucket A (SIDEWAYS+breadth>65, momentum LONG)")
    summarize(all_trades_b, "Bucket B (SIDEWAYS+breadth<=65, momentum LONG)")


if __name__ == "__main__":
    main()
