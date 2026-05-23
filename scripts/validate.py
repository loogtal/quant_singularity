#!/usr/bin/env python3
"""Run all validation: preflight + backtest + walkforward + readiness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()

from config.settings import INITIAL_CAPITAL
from core.preflight import Preflight
from meta.live_readiness import LiveReadiness
from self_evolve.performance_tracker import PerformanceTracker
from backtest.quick import QuickBacktest
from backtest.walkforward import WalkForwardBacktest


def main():
    print("\n=== VALIDATE ALL ===\n")
    pf = Preflight().run()
    Preflight().print_report(pf)

    print("--- Quick backtest BTC ---")
    print(QuickBacktest().run("BTC/USDT:USDT", limit=500))

    print("\n--- Walk-forward BTC ---")
    print(WalkForwardBacktest().run("BTC/USDT:USDT", limit=600))

    perf = PerformanceTracker().get_metrics(INITIAL_CAPITAL, INITIAL_CAPITAL)
    ready = LiveReadiness().score(perf, 0.0)
    print(f"\n--- Live readiness: {ready['score']}/100 {ready['verdict']} ---")
    for n in ready["notes"]:
        print(f"  - {n}")

    ok = pf["ok"] and ready["score"] >= 40
    print(f"\nOverall: {'PASS (keep paper)' if ok else 'FIX ISSUES FIRST'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
