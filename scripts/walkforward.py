#!/usr/bin/env python3
"""python scripts/walkforward.py [--symbol BTC/USDT:USDT] [--bars 800]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from backtest.walkforward import WalkForwardBacktest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--bars", type=int, default=800)
    args = p.parse_args()
    r = WalkForwardBacktest().run(args.symbol, limit=args.bars)
    print("\n=== WALK-FORWARD ===")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
