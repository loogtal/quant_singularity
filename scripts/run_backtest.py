#!/usr/bin/env python3
"""python scripts/backtest.py [--symbol BTC/USDT:USDT] [--bars 500]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from backtest.quick import main

if __name__ == "__main__":
    main()
