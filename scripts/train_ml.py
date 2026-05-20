#!/usr/bin/env python3
"""python scripts/train_ml.py [--symbol BTC/USDT:USDT]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from models.trainer import ModelTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    r = ModelTrainer().train(args.symbol, epochs=args.epochs)
    print("\n=== ML TRAIN ===")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
