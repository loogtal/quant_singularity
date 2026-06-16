"""
Train LightGBM direction predictor across multiple symbols and regimes.

Usage:
    python scripts/train_lgbm.py
    python scripts/train_lgbm.py --symbols BTC/USDT:USDT ETH/USDT:USDT --bars 1500
    python scripts/train_lgbm.py --regime bull

Trains one model per regime using data labeled by the regime classifier,
then saves to storage/models/lgbm/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np

DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
]
DEFAULT_BARS = 1000
TIMEFRAME    = "15m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM predictor")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--bars",    type=int,   default=DEFAULT_BARS)
    parser.add_argument("--regime",  default=None, help="Train only this regime (or 'global')")
    args = parser.parse_args()

    from data.market_data import MarketData
    from meta.regime_classifier import RegimeClassifier
    from models.lgbm_predictor import LGBMPredictor

    market     = MarketData()
    classifier = RegimeClassifier()
    lgbm       = LGBMPredictor()

    print(f"Training LightGBM on {len(args.symbols)} symbols × {args.bars} bars ({TIMEFRAME})")

    # Per-symbol data lists for train_multi(); also grouped by regime
    global_data: list[tuple[np.ndarray, np.ndarray]] = []
    regime_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for sym in args.symbols:
        print(f"  Fetching {sym}...", end=" ", flush=True)
        try:
            df = market.get_ohlcv_df(sym, timeframe=TIMEFRAME, limit=args.bars)
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

        closes  = df["close"].values
        volumes = df["volume"].values
        global_data.append((closes, volumes))

        # Classify using the last 200+ bars of closes (NOT a dict — classifier needs array)
        regime = classifier.classify(closes)
        regime_data.setdefault(regime, []).append((closes, volumes))
        print(f"regime={regime}  bars={len(closes)}")

    print()
    regimes_to_train = [args.regime] if args.regime else (["global"] + list(regime_data.keys()))

    for regime in regimes_to_train:
        if regime == "global":
            if not global_data:
                print("[global] no data — skip")
                continue
            data = global_data
        else:
            if regime not in regime_data:
                print(f"[{regime}] no data — skip")
                continue
            data = regime_data[regime]

        n_bars = sum(len(c) for c, _ in data)
        print(f"[{regime}] training on {n_bars:,} bars from {len(data)} symbols...", end=" ", flush=True)
        acc = lgbm.train_multi(data, regime=regime)
        if acc > 0:
            print(f"accuracy={acc:.3f}  {'OK' if acc >= 0.52 else 'BELOW THRESHOLD — consider more data'}")
        else:
            print("FAILED (too few samples or import error)")

    print("\nDone. Model files saved to storage/models/lgbm/")
    print("Accuracy summary:")
    for r in regimes_to_train:
        a = lgbm.get_accuracy(r)
        if a > 0:
            flag = "✓" if a >= 0.52 else "✗"
            print(f"  {flag} {r}: {a:.3f}")


if __name__ == "__main__":
    main()
