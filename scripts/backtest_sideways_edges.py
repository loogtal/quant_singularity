"""
Sideways Edge Search
=====================
The active engine is currently idle ~50% of the time because, in SIDEWAYS
regime, only funding_arb is allowed (and funding rates are currently too
small to qualify). This script searches for a genuine edge during true
sideways/ranging conditions using tighter-TP/SL variants of mean-reversion
and Bollinger Band fades — the existing mean_reversion mode (RSI 35/65,
TP=1.0%/SL=0.5%) tested at Sharpe -1.43 in sideways; this tests whether
TIGHTER targets (matching the smaller realised moves in chop) do better.

Usage:
    python scripts/backtest_sideways_edges.py
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
TOTAL_BARS = 4000   # ~41 days at 15m
TAKER_FEE = 0.0004
SLOPE_THRESH = 0.003


def build_exchange() -> ccxt.Exchange:
    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30_000})
    ex.load_markets()
    return ex


def fetch_ohlcv(ex, symbol) -> np.ndarray | None:
    """Paginate backwards to assemble ~TOTAL_BARS candles."""
    try:
        all_rows = []
        end_ts = None
        ms_per_bar = 15 * 60 * 1000
        while len(all_rows) < TOTAL_BARS:
            params = {"endTime": end_ts} if end_ts else {}
            raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT, params=params)
            if not raw:
                break
            all_rows = raw + all_rows
            end_ts = raw[0][0] - ms_per_bar
        if len(all_rows) < 200:
            return None
        arr = np.array(all_rows, dtype=float)
        # de-dup by timestamp, keep sorted
        _, idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[np.sort(idx)]
        return arr[-TOTAL_BARS:]
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None


def ema_series(closes, period):
    alpha = 2.0 / (period + 1)
    result = np.empty_like(closes)
    result[0] = closes[0]
    for i in range(1, len(closes)):
        result[i] = alpha * closes[i] + (1 - alpha) * result[i - 1]
    return result


def slope_series(closes, window=20):
    ema50 = ema_series(closes, 50)
    slopes = np.zeros_like(ema50)
    for i in range(window, len(ema50)):
        slopes[i] = (ema50[i] - ema50[i - window]) / (ema50[i - window] + 1e-12)
    return slopes


def rsi_series(closes, period=14):
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l != 0 else 100.0
        rsi[i + 1] = 100 - 100 / (1 + rs)
    return rsi


def bollinger(closes, period=20, mult=2.0):
    n = len(closes)
    mid = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(period, n):
        window = closes[i - period:i]
        m = window.mean()
        s = window.std()
        mid[i] = m
        upper[i] = m + mult * s
        lower[i] = m - mult * s
    return mid, upper, lower


def simulate_reversion(ohlcv, mask, entry_signal, tp, sl, max_hold=96):
    """Generic fade simulation: entry_signal[i] in {-1,0,1} (direction), TP/SL in pct."""
    closes, highs, lows = ohlcv[:, 4], ohlcv[:, 2], ohlcv[:, 3]
    trades = []
    i = 0
    n = len(ohlcv)
    while i < n - 1:
        if not mask[i] or entry_signal[i] == 0:
            i += 1
            continue
        direction = entry_signal[i]
        entry = closes[i]
        j = i + 1
        limit = min(n, i + 1 + max_hold)
        while j < limit:
            if direction == 1:
                if highs[j] >= entry * (1 + tp):
                    trades.append(tp - 2 * TAKER_FEE)
                    break
                if lows[j] <= entry * (1 - sl):
                    trades.append(-sl - 2 * TAKER_FEE)
                    break
            else:
                if lows[j] <= entry * (1 - tp):
                    trades.append(tp - 2 * TAKER_FEE)
                    break
                if highs[j] >= entry * (1 + sl):
                    trades.append(-sl - 2 * TAKER_FEE)
                    break
            j += 1
        else:
            last = closes[min(j, n - 1)]
            trades.append(direction * (last - entry) / entry - 2 * TAKER_FEE)
        i = j + 1
    return trades


def summarize(trades, label):
    if not trades:
        print(f"  {label:55s}: 0 trades")
        return
    arr = np.array(trades)
    wr = (arr > 0).mean() * 100
    total = arr.sum()
    sharpe = arr.mean() / (arr.std() + 1e-12) * np.sqrt(len(arr))
    print(f"  {label:55s}: n={len(arr):4d}  WR={wr:5.1f}%  total={total:+.4f}  "
          f"sharpe={sharpe:+.2f}  avg={arr.mean()*100:+.4f}%")


def main():
    ex = build_exchange()
    print(f"Fetching {TIMEFRAME} OHLCV for {len(SYMBOLS)} symbols (limit={LIMIT})...")

    data = {}
    for sym in SYMBOLS:
        arr = fetch_ohlcv(ex, sym)
        if arr is not None:
            data[sym] = arr

    min_len = min(len(v) for v in data.values())
    print(f"Loaded {len(data)} symbols, aligned length={min_len}\n")

    aligned = {sym: arr[-min_len:] for sym, arr in data.items()}
    closes_d = {sym: arr[:, 4] for sym, arr in aligned.items()}

    ref = "BTC/USDT:USDT"
    btc_slope = slope_series(closes_d[ref])
    sideways_mask = np.abs(btc_slope) <= SLOPE_THRESH
    print(f"SIDEWAYS bars (BTC slope): {sideways_mask.sum()} / {min_len}\n")

    candidates = {
        "RSI(14) 35/65, TP1.0%/SL0.5% [current mean_reversion]": dict(rsi_lo=35, rsi_hi=65, tp=0.010, sl=0.005),
        "RSI(14) 35/65, TP0.4%/SL0.25% [tighter]":               dict(rsi_lo=35, rsi_hi=65, tp=0.004, sl=0.0025),
        "RSI(14) 25/75, TP0.4%/SL0.25% [tighter+extreme]":        dict(rsi_lo=25, rsi_hi=75, tp=0.004, sl=0.0025),
        "RSI(14) 25/75, TP0.6%/SL0.3% [extreme, wider TP]":       dict(rsi_lo=25, rsi_hi=75, tp=0.006, sl=0.003),
    }

    print("=== RSI mean-reversion variants (sideways bars only) ===")
    for label, p in candidates.items():
        all_trades = []
        for sym, arr in aligned.items():
            closes = closes_d[sym]
            rsi = rsi_series(closes, 14)
            sig = np.zeros(min_len, dtype=int)
            sig[rsi < p["rsi_lo"]] = 1
            sig[rsi > p["rsi_hi"]] = -1
            trades = simulate_reversion(arr, sideways_mask, sig, p["tp"], p["sl"])
            all_trades += trades
        summarize(all_trades, label)
    print()

    print("=== Bollinger Band (20, 2.0) fade variants (sideways bars only) ===")
    bb_candidates = {
        "BB touch outer, TP=mid-revert(0.4% cap)/SL0.25%": dict(tp=0.004, sl=0.0025),
        "BB touch outer, TP0.6%/SL0.3%":                   dict(tp=0.006, sl=0.003),
        "BB touch outer, TP0.3%/SL0.2% [very tight]":      dict(tp=0.003, sl=0.002),
    }
    for label, p in bb_candidates.items():
        all_trades = []
        for sym, arr in aligned.items():
            closes = closes_d[sym]
            mid, upper, lower = bollinger(closes, 20, 2.0)
            sig = np.zeros(min_len, dtype=int)
            sig[(closes <= lower) & (lower > 0)] = 1   # touched lower band -> fade LONG
            sig[(closes >= upper) & (upper > 0)] = -1  # touched upper band -> fade SHORT
            trades = simulate_reversion(arr, sideways_mask, sig, p["tp"], p["sl"])
            all_trades += trades
        summarize(all_trades, label)


if __name__ == "__main__":
    main()
