#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()
from config.settings import INITIAL_CAPITAL
from research.daily_report import DailyReport

if __name__ == "__main__":
    r = DailyReport().generate(INITIAL_CAPITAL, 0.0, run_backtest=True)
    DailyReport().print_report(r)
