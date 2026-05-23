"""Daily report — session stats, backtest snapshot, live readiness."""

import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import INITIAL_CAPITAL, STORAGE_DIR
from meta.live_readiness import LiveReadiness
from self_evolve.performance_tracker import PerformanceTracker


class DailyReport:
    REPORT_DIR = STORAGE_DIR / "reports"

    def generate(self, equity: float, drawdown: float, run_backtest: bool = True) -> dict:
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        perf_tracker = PerformanceTracker()
        perf = perf_tracker.get_metrics(equity, INITIAL_CAPITAL)
        perf["drawdown"] = drawdown

        readiness = LiveReadiness().score(perf, drawdown)
        backtests = []

        if run_backtest:
            backtests = self._quick_backtests(["BTC/USDT:USDT", "ETH/USDT:USDT"])

        report = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "equity": round(equity, 2),
            "initial_capital": INITIAL_CAPITAL,
            "performance": perf,
            "readiness": readiness,
            "backtests": backtests,
        }

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.REPORT_DIR / f"{day}.json"
        path.write_text(json.dumps(report, indent=2))
        return report

    def _quick_backtests(self, symbols: list[str]) -> list[dict]:
        results = []
        try:
            from backtest.quick import QuickBacktest
            for sym in symbols:
                try:
                    r = QuickBacktest().run(sym, limit=400)
                    results.append(r)
                except Exception as e:
                    results.append({"symbol": sym, "error": str(e)})
        except ImportError:
            pass
        return results

    def print_report(self, report: dict) -> None:
        print("\n========== DAILY REPORT ==========")
        print(f"  {report['date']}")
        p = report["performance"]
        print(f"  Equity: {report['equity']} | Daily: {p['daily_pnl_pct']:.2%} | Total PnL: {p['pnl']}")
        print(f"  Trades: {p['total_trades']} | Winrate: {p['winrate']:.1%}")
        r = report["readiness"]
        print(f"  Live readiness: {r['score']}/100 — {r['verdict']}")
        for n in r.get("notes", [])[:4]:
            print(f"    - {n}")
        for bt in report.get("backtests", []):
            if "error" in bt:
                print(f"  BT {bt.get('symbol')}: error")
            else:
                print(
                    f"  BT {bt['symbol']}: {bt['trades']} trades "
                    f"wr={bt['winrate']:.0%} pnl={bt['total_pnl']}"
                )
        print("==================================\n")
