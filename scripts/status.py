#!/usr/bin/env python3
"""
Quick status check — shows current portfolio state, today's trades,
and performance metrics without restarting the bot.

Usage:
    python scripts/status.py
    python scripts/status.py --json
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STORAGE = ROOT / "storage"


def _color(text: str, code: str) -> str:
    if "--json" in sys.argv:
        return text
    return f"\033[{code}m{text}\033[0m"

green  = lambda t: _color(t, "92")
red    = lambda t: _color(t, "91")
yellow = lambda t: _color(t, "93")
bold   = lambda t: _color(t, "1")
cyan   = lambda t: _color(t, "96")


def main():
    now = datetime.now(timezone.utc)
    data = {}

    # ── System state ─────────────────────────────────────────────────────────
    state_file = STORAGE / "system_state.json"
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            pass

    dual   = state.get("dual", {})
    port   = state.get("portfolio", {})
    market = state.get("market", {})
    intel  = state.get("market_intelligence", {})

    passive_eq = dual.get("passive_equity", port.get("equity", 0) * 0.6)
    active_eq  = dual.get("active_equity",  port.get("equity", 0) * 0.4)
    total_eq   = passive_eq + active_eq
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        initial = float(os.getenv("QS_TOTAL_CAPITAL", "5000"))
    except Exception:
        initial = 5000.0

    pnl_pct  = (total_eq - initial) / initial * 100 if initial else 0
    drawdown = port.get("drawdown", 0) * 100

    # ── Trade DB ─────────────────────────────────────────────────────────────
    today_trades = all_trades = clean_trades = 0
    today_pnl = today_wr = all_wr = clean_wr = 0.0
    all_pnl = clean_pnl = 0.0
    by_mode: dict = {}
    daily_avg_pnl = 0.0

    db_path = STORAGE / "trades.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))

            # Today
            row = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), ROUND(SUM(pnl),2)
                FROM trades WHERE timestamp > datetime('now','-24 hours')
            """).fetchone()
            today_trades, today_wins, today_pnl = row[0] or 0, row[1] or 0, row[2] or 0.0
            today_wr = today_wins / today_trades if today_trades else 0

            # All time
            row2 = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), ROUND(SUM(pnl),2)
                FROM trades
            """).fetchone()
            all_trades, all_wins, all_pnl = row2[0] or 0, row2[1] or 0, row2[2] or 0.0
            all_wr = all_wins / all_trades if all_trades else 0

            # "Clean" WR: since Jun 1 2026 (after system fixes)
            row3 = conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), ROUND(SUM(pnl),2)
                FROM trades WHERE timestamp >= '2026-06-01'
            """).fetchone()
            clean_trades, clean_wins, clean_pnl = row3[0] or 0, row3[1] or 0, row3[2] or 0.0
            clean_wr = clean_wins / clean_trades if clean_trades else 0

            # Daily average (days with trades since Jun 1)
            row4 = conn.execute("""
                SELECT COUNT(DISTINCT date(timestamp)), ROUND(SUM(pnl),2)
                FROM trades WHERE timestamp >= '2026-06-01'
            """).fetchone()
            trade_days = row4[0] or 1
            daily_avg_pnl = round((row4[1] or 0.0) / max(trade_days, 1), 2)

            # By mode
            rows = conn.execute("""
                SELECT signal_mode, COUNT(*),
                       SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END),
                       ROUND(SUM(pnl),2)
                FROM trades WHERE signal_mode IS NOT NULL AND signal_mode != ''
                  AND timestamp >= '2026-06-01'
                GROUP BY signal_mode ORDER BY COUNT(*) DESC
            """).fetchall()
            for r in rows:
                by_mode[r[0]] = {"trades": r[1], "wins": r[2], "pnl": r[3]}

            conn.close()
        except Exception as e:
            pass

    # ── Evolved params ───────────────────────────────────────────────────────
    evolved = {}
    ev_file = STORAGE / "evolved_params.json"
    if ev_file.exists():
        try:
            evolved = json.loads(ev_file.read_text())
        except Exception:
            pass

    a_params = evolved.get("active", {})
    p_params = evolved.get("passive", {})

    # ── Output ───────────────────────────────────────────────────────────────
    if "--json" in sys.argv:
        out = {
            "timestamp": now.isoformat(),
            "portfolio": {
                "total_equity": round(total_eq, 2),
                "passive_equity": round(passive_eq, 2),
                "active_equity": round(active_eq, 2),
                "pnl_pct": round(pnl_pct, 2),
                "drawdown_pct": round(drawdown, 2),
            },
            "today": {
                "trades": today_trades,
                "win_rate": round(today_wr, 4),
                "pnl": round(today_pnl, 2),
            },
            "all_time": {
                "trades": all_trades,
                "win_rate": round(all_wr, 4),
            },
        }
        print(json.dumps(out, indent=2))
        return

    print()
    print(bold("╔══════════════════════════════════════════════════╗"))
    print(bold("║        QUANT SINGULARITY — LIVE STATUS          ║"))
    print(bold("╚══════════════════════════════════════════════════╝"))
    print(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    # Portfolio
    pnl_str = green(f"+{pnl_pct:.2f}%") if pnl_pct >= 0 else red(f"{pnl_pct:.2f}%")
    dd_str  = red(f"{drawdown:.1f}%") if drawdown > 5 else yellow(f"{drawdown:.1f}%") if drawdown > 2 else green(f"{drawdown:.1f}%")
    print(bold("  PORTFOLIO"))
    print(f"    Total:    {bold(f'${total_eq:,.2f}')}  {pnl_str}")
    print(f"    Passive:  ${passive_eq:,.2f}  ({passive_eq/total_eq*100:.0f}%)" if total_eq else "    Passive:  -")
    print(f"    Active:   ${active_eq:,.2f}  ({active_eq/total_eq*100:.0f}%)" if total_eq else "    Active:   -")
    print(f"    Drawdown: {dd_str}")
    print()

    # Market
    regime  = market.get("regime", intel.get("dominant_trend", "?"))
    fg      = intel.get("fear_greed", 50)
    breadth = intel.get("breadth", 50)
    fg_str  = red(f"EXTREME FEAR ({fg:.0f})") if fg < 25 else red(f"FEAR ({fg:.0f})") if fg < 40 else green(f"GREED ({fg:.0f})") if fg > 60 else f"NEUTRAL ({fg:.0f})"
    print(bold("  MARKET"))
    print(f"    Regime:      {yellow(regime.upper()) if regime else '?'}")
    print(f"    Fear/Greed:  {fg_str}")
    print(f"    Breadth:     {breadth:.0f}% above EMA50")
    print()

    # ── Daily income machine ──────────────────────────────────────────────────
    thb_rate = 35.0
    try:
        import os; from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
        thb_rate = float(os.getenv("QS_THB_PER_USD", "35.0"))
    except Exception:
        pass
    active_cap    = state.get("dual", {}).get("active", {}).get("equity", active_eq) or active_eq
    target_usdt   = round(active_cap * 0.02, 2)
    target_thb    = round(target_usdt * thb_rate, 0)
    withdrawable  = max(0.0, round(today_pnl * 0.80, 2))   # 80% withdrawable, 20% compound buffer
    withdraw_thb  = round(withdrawable * thb_rate, 0)
    progress_pct  = min(100, round(today_pnl / target_usdt * 100, 0)) if target_usdt else 0
    bar_filled    = int(progress_pct / 10)
    bar           = "█" * bar_filled + "░" * (10 - bar_filled)

    avg_thb_day   = round(daily_avg_pnl * thb_rate, 0)
    cum_pnl_str   = green(f"+${clean_pnl:.2f}") if clean_pnl >= 0 else red(f"${clean_pnl:.2f}")

    print(bold("  DAILY INCOME MACHINE"))
    pnl_today_str = green(f"+${today_pnl:.2f}") if today_pnl >= 0 else red(f"${today_pnl:.2f}")
    print(f"    Today:      {pnl_today_str}  ({progress_pct:.0f}% of target)")
    print(f"    Progress:   [{bar}]  target=${target_usdt:.0f}  ({target_thb:.0f} THB)")
    if withdrawable > 0:
        print(f"    Withdrawable: {green(f'${withdrawable:.2f}')}  ({withdraw_thb:.0f} THB) ← can take out now")
    print(f"    Avg/day:    ${daily_avg_pnl:.2f}  ({avg_thb_day:.0f} THB/day)  |  Cumulative: {cum_pnl_str}")
    print()

    # Today's trades
    wr_str = green(f"{today_wr:.0%}") if today_wr >= 0.40 else red(f"{today_wr:.0%}") if today_wr < 0.30 else yellow(f"{today_wr:.0%}")
    print(bold("  TODAY TRADES"))
    print(f"    Count: {today_trades}  |  WR: {wr_str}")
    print()

    # Performance (clean: since Jun 1)
    cwr_str = green(f"{clean_wr:.0%}") if clean_wr >= 0.50 else red(f"{clean_wr:.0%}") if clean_wr < 0.35 else yellow(f"{clean_wr:.0%}")
    awr_str = green(f"{all_wr:.0%}") if all_wr >= 0.40 else red(f"{all_wr:.0%}") if all_wr < 0.30 else yellow(f"{all_wr:.0%}")
    print(bold("  PERFORMANCE"))
    print(f"    Since fixes : {clean_trades:3} trades  WR={cwr_str}  PnL=${clean_pnl:+.2f}  ← real signal")
    print(f"    All-time    : {all_trades:3} trades  WR={awr_str}  (includes legacy garbage trades)")
    if by_mode:
        print(f"    By mode (since Jun 1):")
        for mode, s in by_mode.items():
            mode_wr = s['wins']/s['trades'] if s['trades'] else 0
            mwr_str = green(f"{mode_wr:.0%}") if mode_wr >= 0.50 else red(f"{mode_wr:.0%}") if mode_wr < 0.35 else yellow(f"{mode_wr:.0%}")
            print(f"      {mode:16}: {s['trades']:3} trades  WR={mwr_str}  PnL=${s['pnl']:+.2f}")
    print()

    # Params
    tp_a = a_params.get("take_profit", 0)
    sl_a = a_params.get("stop_loss", 0)
    rr_a = round(tp_a/sl_a, 2) if sl_a else 0
    rr_str = green(f"{rr_a}x") if rr_a >= 2.5 else red(f"{rr_a}x")
    print(bold("  EVOLVED PARAMS"))
    print(f"    Passive: TP={p_params.get('take_profit')} SL={p_params.get('stop_loss')} sharpe={p_params.get('sharpe')}")
    print(f"    Active:  TP={tp_a} SL={sl_a} R:R={rr_str} sharpe={a_params.get('sharpe')}")
    print()
    print(f"  Dashboard: {cyan('http://localhost:8788')}")
    print()


if __name__ == "__main__":
    main()
