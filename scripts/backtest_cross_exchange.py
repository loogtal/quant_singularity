"""
Cross-Exchange Divergence Scan (Binance vs Gate.io perps)
=========================================================
Tests the only remaining retail-accessible edge family reachable from here:
cross-exchange price divergence / stat-arb. Measures, on aligned 1-minute bars,
how often the two venues' prices diverge by MORE than a round-trip cost — i.e.
whether there is any capturable spread after fees.

Reality check up front: liquid-pair cross-exchange spreads are tiny and closed by
HFT in milliseconds. This quantifies whether a 1-minute-resolution retail bot
could capture anything at all.

Usage:
    python scripts/backtest_cross_exchange.py --symbols BTC/USDT:USDT ETH/USDT:USDT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import ccxt

# Round-trip cost to capture a cross-exchange divergence: you trade on BOTH venues
# (taker each) and unwind both. ~0.04%×4 legs ≈ 0.16%; use 0.12% optimistic (some maker).
ROUNDTRIP_COST = 0.0012


def fetch_1m(ex, symbol, limit=1000):
    o = ex.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
    return {int(r[0]): float(r[4]) for r in o}   # ts -> close


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    binance = ccxt.binanceusdm()
    gate = ccxt.gateio({"options": {"defaultType": "swap"}})

    print(f"[CrossEx] Binance vs Gate.io, 1m bars, cost/round={ROUNDTRIP_COST:.2%}\n")
    hdr = (f"{'symbol':<16} {'bars':>5} {'spread_std%':>11} {'mean|sp|%':>10} "
           f"{'max%':>7} {'>cost%':>7} {'capturable/day':>15}")
    print(hdr); print("-" * len(hdr))

    for sym in args.symbols:
        try:
            a = fetch_1m(binance, sym, args.limit)
            b = fetch_1m(gate, sym, args.limit)
        except Exception as e:
            print(f"{sym:<16} fetch error: {str(e)[:50]}")
            continue
        common = sorted(set(a) & set(b))
        if len(common) < 50:
            print(f"{sym:<16} too few aligned bars ({len(common)})")
            continue
        spread = np.array([(a[t] - b[t]) / ((a[t] + b[t]) / 2) for t in common])
        abs_sp = np.abs(spread)
        over = abs_sp > ROUNDTRIP_COST
        # net capturable per opportunity = (|spread| - cost), only when > cost
        net = np.where(over, abs_sp - ROUNDTRIP_COST, 0.0)
        minutes = len(common)
        capturable_per_day = float(np.sum(net)) / minutes * 1440 * 100   # %/day if you caught every one
        print(f"{sym:<16} {minutes:>5} {np.std(spread)*100:>11.4f} {np.mean(abs_sp)*100:>10.4f} "
              f"{np.max(abs_sp)*100:>7.4f} {np.mean(over)*100:>7.2f} {capturable_per_day:>14.3f}%")

    print("\n  Interpretation: if '>cost%' ≈ 0 and capturable/day ≈ 0, there is NO")
    print("  retail-capturable cross-exchange edge at 1-minute resolution (HFT eats it).")
    print()


if __name__ == "__main__":
    main()
