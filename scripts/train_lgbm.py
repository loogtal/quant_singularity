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

    # Accumulate closes per regime across all symbols
    regime_closes:  dict[str, list[np.ndarray]] = {}
    regime_volumes: dict[str, list[np.ndarray]] = {}
    global_closes:  list[np.ndarray] = []
    global_volumes: list[np.ndarray] = []

    for sym in args.symbols:
        print(f"  Fetching {sym}...", end=" ", flush=True)
        try:
            df = market.get_ohlcv_df(sym, timeframe=TIMEFRAME, limit=args.bars)
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

        closes  = df["close"].values
        volumes = df["volume"].values
        global_closes.append(closes)
        global_volumes.append(volumes)

        # Classify the last bar's regime
        sample = {
            "close":       float(closes[-1]),
            "ema50":       float(np.mean(closes[-50:])) if len(closes) >= 50 else float(closes[-1]),
            "ema200":      float(np.mean(closes[-200:])) if len(closes) >= 200 else float(closes[-1]),
            "volatility":  float(np.std(np.diff(closes[-21:]) / closes[-21:-1])) if len(closes) >= 21 else 0.0,
            "change_24h":  float((closes[-1] - closes[-96]) / closes[-96]) if len(closes) >= 96 else 0.0,
        }
        regime = classifier.classify(sample)
        regime_closes.setdefault(regime,  []).append(closes)
        regime_volumes.setdefault(regime, []).append(volumes)
        print(f"regime={regime}  bars={len(closes)}")

    print()
    regimes_to_train = [args.regime] if args.regime else (["global"] + list(regime_closes.keys()))

    for regime in regimes_to_train:
        if regime == "global":
            if not global_closes:
                print("[global] no data — skip")
                continue
            closes_arr  = np.concatenate(global_closes)
            volumes_arr = np.concatenate(global_volumes)
        else:
            if regime not in regime_closes:
                print(f"[{regime}] no data — skip")
                continue
            closes_arr  = np.concatenate(regime_closes[regime])
            volumes_arr = np.concatenate(regime_volumes[regime])

        print(f"[{regime}] training on {len(closes_arr):,} bars...", end=" ", flush=True)
        acc = lgbm.train(closes_arr, volumes_arr, regime=regime)
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
