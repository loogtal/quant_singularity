"""
Walk-Forward Validation of the REAL ActiveStrategy
==================================================
Splits multi-month history into sequential windows, labels each window's regime,
and replays the ACTUAL ActiveStrategy in each mode through every window. Answers
the real question: is mean_reversion's edge robust across regimes, or was the
10-day result luck? And does the live router's regime→mode map actually hold up?

Usage:
    python scripts/walkforward_real.py --months 3 --window-days 10
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from scripts.backtest_real_active import fetch_data, replay
from meta.regime_classifier import RegimeClassifier

ALL_MODES = ["mean_reversion", "momentum", "vwap_reversal", "breakout"]


def _regime_at(data: dict, symbol: str, end_ms: int) -> str:
    """Classify regime using 1h closes up to window end (label = as-of window end)."""
    df = data.get((symbol, "1h"))
    if df is None:
        return "unknown"
    closes = df[df["timestamp"] <= end_ms]["close"].values
    if len(closes) < 100:
        return "unknown"
    return RegimeClassifier().classify(np.asarray(closes[-250:], dtype=float))


def _fmt_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d")


def main():
    p = argparse.ArgumentParser(description="Walk-forward the REAL ActiveStrategy")
    p.add_argument("--symbols", nargs="+", default=["SOL/USDT:USDT", "BNB/USDT:USDT"])
    p.add_argument("--months", type=float, default=3.0)
    p.add_argument("--window-days", type=float, default=10.0)
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="only open trades with signal confidence >= this (conviction filter)")
    p.add_argument("--cooldown", type=int, default=0,
                   help="bars to wait before re-opening a symbol after a close (anti-churn)")
    args = p.parse_args()
    modes = args.modes

    print(f"[WalkForward] Fetching ~{args.months}mo for {args.symbols} …")
    data = fetch_data(args.symbols, args.months)
    base = data.get((args.symbols[0], "15m"))
    if base is None or len(base) < 500:
        print("  Not enough data — aborting")
        return

    ts = base["timestamp"].values.astype("int64")
    win_bars = int(args.window_days * 96)
    warmup = 80
    starts = list(range(warmup, len(ts) - win_bars, win_bars))
    print(f"[WalkForward] {len(starts)} windows × {len(modes)} modes "
          f"({args.window_days:.0f} days each)\n")

    header = f"{'window':<13} {'regime':<9} {'mode':<15} {'ret%':>7} {'win%':>6} {'pday%':>6} {'trades':>7}"
    print(header); print("-" * len(header))

    # Aggregate per (regime, mode) and per mode
    agg: dict = defaultdict(lambda: {"ret": [], "pday": [], "trades": 0, "wins": 0, "n": 0})
    per_mode: dict = defaultdict(lambda: {"ret": [], "pday": []})

    for s in starts:
        start_ms = int(ts[s]); end_ms = int(ts[min(s + win_bars, len(ts) - 1)])
        regime = _regime_at(data, args.symbols[0], end_ms)
        win_lbl = f"{_fmt_day(start_ms)}/{_fmt_day(end_ms)}"
        for mode in modes:
            st = replay(data, args.symbols, mode, start_ms=start_ms, end_ms=end_ms,
                        min_conf=args.min_conf, cooldown=args.cooldown)
            if not st:
                continue
            ret, win, pday, n = (st["return_pct"], st["winrate"] * 100,
                                 st["profitable_days_pct"], st["trades"])
            print(f"{win_lbl:<13} {regime:<9} {mode:<15} {ret:>7.2f} {win:>6.1f} {pday:>6.1f} {n:>7}")
            a = agg[(regime, mode)]
            a["ret"].append(ret); a["pday"].append(pday)
            a["trades"] += n; a["wins"] += st["wins"]; a["n"] += 1
            per_mode[mode]["ret"].append(ret); per_mode[mode]["pday"].append(pday)
        print()

    # ── summaries ──────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  PER (REGIME × MODE)  — avg return, avg profitable-days, win rate")
    print("=" * 64)
    for (regime, mode), a in sorted(agg.items()):
        wr = a["wins"] / a["trades"] * 100 if a["trades"] else 0
        print(f"  {regime:<9} {mode:<15} "
              f"avg_ret={np.mean(a['ret']):+6.2f}%  "
              f"avg_pday={np.mean(a['pday']):5.1f}%  "
              f"wr={wr:4.1f}%  windows={a['n']}  trades={a['trades']}")

    print("\n" + "=" * 64)
    print("  PER MODE (all regimes)")
    print("=" * 64)
    for mode in modes:
        r = per_mode[mode]["ret"]
        if not r:
            continue
        total = float(np.sum(r)); avg = float(np.mean(r))
        pos = sum(1 for x in r if x > 0)
        print(f"  {mode:<15} total_ret={total:+7.2f}%  avg/win={avg:+6.2f}%  "
              f"profitable_windows={pos}/{len(r)}  avg_pday={np.mean(per_mode[mode]['pday']):.1f}%")
    print()


if __name__ == "__main__":
    main()
