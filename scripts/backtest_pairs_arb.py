"""
Pairs / Stat-Arb Edge Test (true SIDEWAYS bars only)
=====================================================
Market-neutral candidate for the idle active-engine slot in SIDEWAYS regime:
rolling z-score of log(price_a/price_b) between correlated majors. Fade the
spread when |z| > entry_z, exit on reversion to exit_z or TP/SL on the spread.

Run after scripts/backtest_sideways_edges.py confirmed RSI/Bollinger fades are
deeply negative (fee drag dominates at tight TPs in 15m chop).

Usage:
    python scripts/backtest_pairs_arb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import itertools
import ccxt
import numpy as np

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "LTC/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
]
TIMEFRAME = "15m"
LIMIT = 1500
TOTAL_BARS = 4000
TAKER_FEE = 0.0004
SLOPE_THRESH = 0.003


def build_exchange() -> ccxt.Exchange:
    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30_000})
    ex.load_markets()
    return ex


def fetch_ohlcv(ex, symbol) -> np.ndarray | None:
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
            if len(raw) < LIMIT:
                break
        if len(all_rows) < 200:
            return None
        arr = np.array(all_rows, dtype=float)
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


def simulate_pairs(closes_a, closes_b, mask, lookback, entry_z, exit_z, tp, sl, max_hold=96):
    n = len(closes_a)
    spread = np.log(closes_a) - np.log(closes_b)
    trades = []
    i = lookback
    while i < n - 1:
        if not mask[i]:
            i += 1
            continue
        window = spread[i - lookback:i]
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-9:
            i += 1
            continue
        z = (spread[i] - mu) / sigma
        if abs(z) < entry_z:
            i += 1
            continue
        direction = -1 if z > 0 else 1  # fade: spread too high -> expect it to fall
        entry_spread = spread[i]
        j = i + 1
        limit = min(n, i + 1 + max_hold)
        while j < limit:
            cur_spread = spread[j]
            pnl = direction * (cur_spread - entry_spread)
            cur_z = (cur_spread - mu) / sigma
            if pnl >= tp:
                trades.append(tp - 4 * TAKER_FEE); break
            if pnl <= -sl:
                trades.append(-sl - 4 * TAKER_FEE); break
            if abs(cur_z) < exit_z:
                trades.append(pnl - 4 * TAKER_FEE); break
            j += 1
        else:
            last = spread[min(j, n - 1)]
            trades.append(direction * (last - entry_spread) - 4 * TAKER_FEE)
        i = j + 1
    return trades


def summarize(trades, label):
    if not trades:
        print(f"  {label:60s}: 0 trades")
        return
    arr = np.array(trades)
    wr = (arr > 0).mean() * 100
    total = arr.sum()
    sharpe = arr.mean() / (arr.std() + 1e-12) * np.sqrt(len(arr))
    print(f"  {label:60s}: n={len(arr):4d}  WR={wr:5.1f}%  total={total:+.4f}  "
          f"sharpe={sharpe:+.2f}  avg={arr.mean()*100:+.4f}%")


def main():
    ex = build_exchange()
    print(f"Fetching {TIMEFRAME} OHLCV for {len(SYMBOLS)} symbols ({TOTAL_BARS} bars each)...")
    data = {}
    for sym in SYMBOLS:
        arr = fetch_ohlcv(ex, sym)
        if arr is not None:
            data[sym] = arr
    min_len = min(len(v) for v in data.values())
    closes = {s: v[-min_len:, 4] for s, v in data.items()}
    print(f"Loaded {len(data)} symbols, aligned length={min_len}\n")

    ref = "BTC/USDT:USDT"
    sideways_mask = np.abs(slope_series(closes[ref])) <= SLOPE_THRESH
    print(f"SIDEWAYS bars: {sideways_mask.sum()} / {min_len}\n")

    pairs = list(itertools.combinations(closes.keys(), 2))
    print(f"Testing {len(pairs)} pairs\n")

    for lookback in [50, 100]:
        for entry_z, exit_z, tp, sl in [(2.0, 0.5, 0.02, 0.015), (1.5, 0.5, 0.015, 0.012), (2.5, 1.0, 0.025, 0.018)]:
            all_trades = []
            for a, b in pairs:
                all_trades += simulate_pairs(closes[a], closes[b], sideways_mask,
                                              lookback, entry_z, exit_z, tp, sl)
            summarize(all_trades, f"lookback={lookback} z_entry={entry_z} z_exit={exit_z} tp={tp:.1%} sl={sl:.1%}")
        print()


if __name__ == "__main__":
    main()
