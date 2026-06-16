"""
Edge Gate — GO / NO-GO deployment decision
===========================================
The hard lesson from validation: every strategy tested has ≈0 or negative net
edge in the current regime, and deploying capital into a no-edge strategy is how
accounts die. This gate operationalises "think for itself → decide NOT to trade
when there is no edge" — the single most important control for the stated goals
("minimise loss, survive years").

It re-runs the REAL-strategy walk-forwards (active + passive) with realistic costs
and emits a per-engine GO/NO-GO with reasons. Intended to be run before enabling
live capital (and could later gate the live loop directly).

A GO requires ALL of:
  - walk-forward TOTAL return > MIN_TOTAL_RET
  - profitable windows >= MIN_WIN_FRACTION of windows
  - worst single-window drawdown < MAX_DD

Usage:
    python scripts/edge_gate.py                  # default quick check
    python scripts/edge_gate.py --active-months 3 --passive-months 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

# GO thresholds — deliberately strict; capital preservation first.
MIN_TOTAL_RET   = 0.0     # walk-forward total must be positive net of cost
MIN_WIN_FRACTION = 0.55   # >55% of windows profitable (consistency, not one lucky window)
MAX_DD          = 25.0    # no single window worse than -25% peak-to-trough


def _windows(ts, start_idx, win_bars):
    return list(range(start_idx, len(ts) - win_bars, win_bars))


def _verdict(name: str, rets: list[float], dds: list[float]) -> bool:
    if not rets:
        print(f"  {name:<28} NO-GO  (no data)")
        return False
    total = float(np.sum(rets))
    win_frac = sum(1 for r in rets if r > 0) / len(rets)
    worst_dd = max(dds) if dds else 0.0
    reasons = []
    if total <= MIN_TOTAL_RET:        reasons.append(f"total {total:+.1f}%<=0")
    if win_frac < MIN_WIN_FRACTION:   reasons.append(f"win {win_frac:.0%}<{MIN_WIN_FRACTION:.0%}")
    if worst_dd >= MAX_DD:            reasons.append(f"DD {worst_dd:.0f}%>={MAX_DD:.0f}%")
    ok = not reasons
    tag = "GO   " if ok else "NO-GO"
    detail = "all criteria met" if ok else "; ".join(reasons)
    print(f"  {name:<28} {tag}  (total={total:+.1f}% win={win_frac:.0%} "
          f"worstDD={worst_dd:.0f}% — {detail})")
    return ok


def gate_active(months: float, window_days: float) -> bool:
    from scripts.backtest_real_active import fetch_data, replay
    symbols = ["SOL/USDT:USDT", "BNB/USDT:USDT"]
    print(f"\n[ACTIVE] fetching ~{months}mo …")
    data = fetch_data(symbols, months)
    base = data.get((symbols[0], "15m"))
    if base is None:
        print("  ACTIVE NO-GO (no data)"); return False
    ts = base["timestamp"].values.astype("int64")
    win = int(window_days * 96)
    starts = _windows(ts, 80, win)
    any_go = False
    for mode in ("momentum", "mean_reversion"):
        rets, dds = [], []
        for s in starts:
            st = replay(data, symbols, mode, int(ts[s]), int(ts[min(s + win, len(ts) - 1)]))
            if st:
                rets.append(st["return_pct"]); dds.append(st["max_drawdown_pct"])
        any_go |= _verdict(f"active/{mode}", rets, dds)
    return any_go


def gate_passive(months: float, window_days: float) -> bool:
    from scripts.backtest_real_passive import fetch_data, replay
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    print(f"\n[PASSIVE] fetching ~{months}mo …")
    data = fetch_data(symbols, months)
    base = data.get((symbols[0], "1d"))
    if base is None:
        print("  PASSIVE NO-GO (no data)"); return False
    ts = base["timestamp"].values.astype("int64")
    win = int(window_days)
    starts = _windows(ts, 100, win)
    rets, dds = [], []
    for s in starts:
        st = replay(data, symbols, int(ts[s]), int(ts[min(s + win, len(ts) - 1)]))
        if st:
            rets.append(st["return_pct"]); dds.append(st["max_drawdown_pct"])
    return _verdict("passive", rets, dds)


def main():
    p = argparse.ArgumentParser(description="GO/NO-GO edge gate")
    p.add_argument("--active-months", type=float, default=3.0)
    p.add_argument("--active-window-days", type=float, default=10.0)
    p.add_argument("--passive-months", type=float, default=12.0)
    p.add_argument("--passive-window-days", type=float, default=90.0)
    args = p.parse_args()

    print("=" * 64)
    print("  EDGE GATE — deploy real capital only on GO")
    print(f"  criteria: total>0, win-windows>={MIN_WIN_FRACTION:.0%}, worstDD<{MAX_DD:.0f}%")
    print("=" * 64)

    active_go  = gate_active(args.active_months, args.active_window_days)
    passive_go = gate_passive(args.passive_months, args.passive_window_days)

    print("\n" + "=" * 64)
    print(f"  ACTIVE : {'GO' if active_go else 'NO-GO'}")
    print(f"  PASSIVE: {'GO' if passive_go else 'NO-GO'}")
    print("=" * 64)
    if not (active_go or passive_go):
        print("  → DO NOT deploy real capital. No engine has validated edge.")
    else:
        print("  → Only deploy the GO engine(s); keep the rest in paper mode.")
    print()
    sys.exit(0 if (active_go or passive_go) else 2)


if __name__ == "__main__":
    main()
