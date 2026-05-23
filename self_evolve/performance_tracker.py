"""
Tracks performance from trade history for self-improvement.
"""

import json
from datetime import datetime, timezone

from config.settings import RESET_STATS_ON_START, STORAGE_DIR


class PerformanceTracker:
    def __init__(self):
        self.storage = STORAGE_DIR
        self.storage.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.storage / "performance.json"
        if RESET_STATS_ON_START:
            self.reset_all()
        else:
            self._load()

    def reset_all(self) -> None:
        self.metrics = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "last_reset_day": self._today(),
            "strategies": {},
        }
        self._save()

    def _load(self):
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "last_reset_day": "",
                "strategies": {},
            }

    def _save(self):
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def reset_daily_if_needed(self):
        today = self._today()
        if self.metrics.get("last_reset_day") != today:
            self.metrics["daily_pnl"] = 0.0
            self.metrics["last_reset_day"] = today
            self._save()

    def record_trade(self, pnl: float, strategy: str = "ensemble"):
        self.reset_daily_if_needed()
        self.metrics["total_trades"] += 1
        self.metrics["total_pnl"] += pnl
        self.metrics["daily_pnl"] += pnl
        if pnl >= 0:
            self.metrics["wins"] += 1
        else:
            self.metrics["losses"] += 1

        strat = self.metrics["strategies"].setdefault(
            strategy, {"trades": 0, "pnl": 0.0, "wins": 0}
        )
        strat["trades"] += 1
        strat["pnl"] += pnl
        if pnl >= 0:
            strat["wins"] += 1
        self._save()

    def get_metrics(self, equity: float, initial_capital: float) -> dict:
        self.reset_daily_if_needed()
        total = self.metrics["total_trades"]
        wins = self.metrics["wins"]
        winrate = wins / total if total else 0.5

        daily_pnl_pct = (
            self.metrics["daily_pnl"] / initial_capital if initial_capital else 0
        )

        # Simple Sharpe proxy from win rate and avg trade
        avg_pnl = self.metrics["total_pnl"] / total if total else 0
        sharpe = (winrate * 2 - 1) + (avg_pnl / max(initial_capital * 0.01, 1))

        return {
            "winrate": round(winrate, 4),
            "sharpe": round(sharpe, 4),
            "pnl": round(self.metrics["total_pnl"], 2),
            "daily_pnl": round(self.metrics["daily_pnl"], 2),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "total_trades": total,
            "equity": equity,
        }
