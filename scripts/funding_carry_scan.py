"""
Funding-Carry Opportunity Scan
==============================
Measures the REAL market-neutral funding-harvest edge: hold a delta-neutral
position (long spot + short perp) and collect the 8h funding. Price PnL ≈ 0, so
the equity curve ≈ cumulative funding − costs — a low-variance "daily income"
stream, which is the closest thing to the user's "profit every day" goal.

Uses historical funding rates (ccxt fetch_funding_rate_history). Reports, per
symbol and for an equal-weight basket:
  - avg funding / 8h and annualised (×3×365)
  - % of periods with positive funding
  - STATIC short-perp neutral: net = Σ funding_rate (you receive when +, pay when −)
  - HARVEST: only hold when funding ≥ threshold (collect +), flat otherwise,
    minus a round-trip cost each time you enter/exit (2 legs).

Usage:
    python scripts/funding_carry_scan.py --months 6
"""

import argparse
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

# Round-trip cost to set up + tear down a neutral pair (perp + spot), both legs.
# Maker on both legs ≈ 0.02%×2 legs×2 sides = 0.08%. Use 0.10% to be safe.
PAIR_ROUNDTRIP_COST = 0.0010

DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "SUI/USDT:USDT",
]


def fetch_funding(ex, symbol: str, periods: int) -> np.ndarray:
    """Paginated funding-rate history (8h cadence). Returns array of rates."""
    tf_ms = 8 * 3_600_000
    since = ex.milliseconds() - periods * tf_ms
    rates: list = []
    seen = set()
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        for r in batch:
            ts = r["timestamp"]
            if ts not in seen:
                seen.add(ts)
                rates.append(float(r["fundingRate"]))
        since = batch[-1]["timestamp"] + tf_ms
        if len(batch) < 1000 or len(rates) >= periods:
            break
        _time.sleep(ex.rateLimit / 1000)
    return np.array(rates, dtype=float)


def harvest_net(rates: np.ndarray, threshold: float) -> tuple[float, int]:
    """
    Hold neutral (collect funding) only on periods where rate >= threshold; flat
    otherwise. Pay PAIR_ROUNDTRIP_COST each time we transition flat→held (and the
    final exit). Returns (net_return_fraction, num_entries).
    """
    held = False
    net = 0.0
    entries = 0
    for r in rates:
        want = r >= threshold
        if want and not held:
            net -= PAIR_ROUNDTRIP_COST   # enter (pay round-trip up front)
            held = True
            entries += 1
        if held:
            net += r                     # collect this period's funding
        if not want and held:
            held = False                 # exit (cost already booked at entry)
    return net, entries


def main():
    p = argparse.ArgumentParser(description="Funding-carry opportunity scan")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--months", type=float, default=6.0)
    p.add_argument("--threshold", type=float, default=0.00005,
                   help="min 8h funding to bother harvesting (default 0.005%%)")
    args = p.parse_args()

    from data.binance_client import BinanceClient
    ex = BinanceClient().get_exchange()
    periods = int(args.months * 30 * 3)   # 3 funding windows/day

    print(f"[FundingScan] {len(args.symbols)} symbols, ~{args.months}mo "
          f"({periods} funding periods), cost/round={PAIR_ROUNDTRIP_COST:.2%}\n")
    hdr = (f"{'symbol':<16} {'avg/8h%':>9} {'annual%':>9} {'pos%':>6} "
           f"{'static_an%':>11} {'harvest_an%':>12} {'entries':>8}")
    print(hdr); print("-" * len(hdr))

    basket_annual_static = []
    basket_annual_harvest = []
    for sym in args.symbols:
        try:
            rates = fetch_funding(ex, sym, periods)
        except Exception as e:
            print(f"{sym:<16} fetch error: {e}")
            continue
        if len(rates) < 30:
            print(f"{sym:<16} too few periods ({len(rates)})")
            continue

        n = len(rates)
        yrs = n / (3 * 365)
        avg = float(np.mean(rates))
        annual = avg * 3 * 365
        pos_pct = float(np.mean(rates > 0)) * 100
        static_net = float(np.sum(rates))                     # static short-perp neutral
        static_an = static_net / yrs
        harv_net, entries = harvest_net(rates, args.threshold)
        harv_an = harv_net / yrs

        basket_annual_static.append(static_an)
        basket_annual_harvest.append(harv_an)
        print(f"{sym:<16} {avg*100:>9.4f} {annual*100:>9.2f} {pos_pct:>6.1f} "
              f"{static_an*100:>11.2f} {harv_an*100:>12.2f} {entries:>8}")

    if basket_annual_harvest:
        print("\n" + "=" * 64)
        print(f"  EQUAL-WEIGHT BASKET (annualised, net of cost)")
        print("=" * 64)
        print(f"  static short-perp neutral : {np.mean(basket_annual_static)*100:+.2f}%/yr")
        print(f"  harvest (≥{args.threshold:.4%} only): {np.mean(basket_annual_harvest)*100:+.2f}%/yr")
        print(f"  (note: harvest needs long-spot + short-perp; collects POSITIVE funding only)")
    print()


if __name__ == "__main__":
    main()
