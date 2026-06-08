"""
Performance time series tracker.

Records daily snapshots: equity, PnL, drawdown, win rates, Sharpe, LGBM accuracy.
AI Brain reads this to detect patterns like declining win rate or accuracy drop.

Persisted to storage/perf_timeseries.json (rolling 90 days).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from config.settings import STORAGE_DIR

_PERF_FILE = STORAGE_DIR / "perf_timeseries.json"
_MAX_DAYS  = 90


class PerformanceTracker:
    """Daily snapshot recorder and rolling analytics."""

    def __init__(self):
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if _PERF_FILE.exists():
            try:
                self._records = json.loads(_PERF_FILE.read_text())
            except Exception:
                self._records = []

    def _save(self) -> None:
        try:
            _PERF_FILE.write_text(
                json.dumps(self._records[-_MAX_DAYS:], indent=2)
            )
        except Exception:
            pass

    def record_daily(
        self,
        equity: float,
        pnl_today: float,
        drawdown: float,
        active_trades: int,
        active_wins: int,
        passive_trades: int,
        passive_wins: int,
        lgbm_accuracy: float,
        regime: str,
    ) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._records and self._records[-1].get("date") == date_str:
            return
        a_wr = active_wins  / max(active_trades, 1)
        p_wr = passive_wins / max(passive_trades, 1)
        self._records.append({
            "date":            date_str,
            "equity":          round(equity, 2),
            "pnl_today":       round(pnl_today, 2),
            "drawdown":        round(drawdown, 4),
            "active_trades":   active_trades,
            "active_winrate":  round(a_wr, 3),
            "passive_trades":  passive_trades,
            "passive_winrate": round(p_wr, 3),
            "lgbm_accuracy":   round(lgbm_accuracy, 4),
            "regime":          regime,
        })
        self._save()

    def rolling_sharpe(self, days: int = 14) -> float:
        if len(self._records) < max(3, days // 2):
            return 0.0
        window = self._records[-days:]
        returns = []
        for i in range(1, len(window)):
            prev = window[i - 1]["equity"]
            curr = window[i]["equity"]
            if prev > 0:
                returns.append((curr - prev) / prev)
        if len(returns) < 2:
            return 0.0
        mu  = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mu) ** 2 for r in returns) / (len(returns) - 1))
        if std < 1e-9:
            return 0.0
        return round(mu / std * math.sqrt(365), 3)

    def rolling_winrate(self, strategy: str = "active", days: int = 7) -> float:
        window = self._records[-days:]
        trades = sum(r.get(f"{strategy}_trades", 0) for r in window)
        wins   = sum(
            round(r.get(f"{strategy}_trades", 0) * r.get(f"{strategy}_winrate", 0))
            for r in window
        )
        return wins / max(trades, 1)

    def trend(self, field: str, days: int = 5) -> str:
        window = self._records[-days:]
        if len(window) < 3:
            return "stable"
        vals = [r.get(field, 0) for r in window]
        n = len(vals)
        x_mean = (n - 1) / 2
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den < 1e-9:
            return "stable"
        slope = num / den
        threshold = abs(y_mean) * 0.02 if y_mean != 0 else 0.001
        if slope > threshold:
            return "improving"
        if slope < -threshold:
            return "declining"
        return "stable"

    def summary_for_ai(self) -> dict:
        return {
            "days_tracked":       len(self._records),
            "sharpe_14d":         self.rolling_sharpe(14),
            "sharpe_7d":          self.rolling_sharpe(7),
            "active_winrate_7d":  round(self.rolling_winrate("active",  7), 3),
            "passive_winrate_7d": round(self.rolling_winrate("passive", 7), 3),
            "equity_trend_5d":    self.trend("equity", 5),
            "accuracy_trend_5d":  self.trend("lgbm_accuracy", 5),
            "winrate_trend_5d":   self.trend("active_winrate", 5),
            "last_7d":            self._records[-7:] if self._records else [],
        }
