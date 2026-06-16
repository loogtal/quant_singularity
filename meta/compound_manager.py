"""
Auto-compound Manager — tracks and enforces profit compounding rules.

Two compound schedules:
  1. Daily micro-compound (active bot)
     After hitting daily target, lock 20% of profit into capital growth.
     The remaining 80% is withdrawable.

  2. Weekly macro-compound (both bots)
     Every 7 days: review total equity growth and rebalance capital between
     passive/active according to the best-performing engine.
     Also logs the compound growth rate (CAGR estimate).

Usage (in DualEngine):
    compound = CompoundManager(passive_portfolio, active_portfolio)
    compound.daily_check(daily_profit_engine)   # call once per cycle
    compound.weekly_check()                     # call once per cycle (checks internally)
    compound.snapshot()                         # for dashboard

The manager does NOT move real funds on Binance — it adjusts the internal
capital bookkeeping so that PortfolioEngine sizing grows accordingly.
Real fund rebalancing is handled separately by DualCapitalAllocator.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from config.settings import STORAGE_DIR

_COMPOUND_FILE = STORAGE_DIR / "compound_state.json"

DAILY_COMPOUND_RATE   = 0.20   # keep 20% of daily profit in capital
WEEKLY_CHECK_INTERVAL = 7 * 86400  # 7 days
MIN_WEEKLY_EQUITY_GAIN = 0.005     # only rebalance if equity grew > 0.5%


class CompoundManager:
    """Manages the compound growth loop for both bots."""

    def __init__(self, passive_portfolio, active_portfolio) -> None:
        self._passive   = passive_portfolio
        self._active    = active_portfolio
        self._start_equity: float = 0.0
        self._last_weekly_ts: float = 0.0
        self._compound_log: list[dict] = []
        self._daily_compounded_today: float = 0.0
        self._today_ordinal: int = 0
        self._peak_equity: float = 0.0
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _COMPOUND_FILE.exists():
            self._start_equity = (
                self._passive.initial_cash + self._active.initial_cash
            )
            self._peak_equity  = self._start_equity
            return
        try:
            data = json.loads(_COMPOUND_FILE.read_text())
            self._start_equity        = float(data.get("start_equity", 0.0))
            self._last_weekly_ts      = float(data.get("last_weekly_ts", 0.0))
            self._peak_equity         = float(data.get("peak_equity", self._start_equity))
            self._compound_log        = data.get("compound_log", [])[-90:]
            # Correct a stale start_equity written when capital config was much smaller
            # (e.g. default 1000 before .env was set to 5000). If start_equity is less
            # than 50% of actual initial capital, it's a legacy artefact — reset it so
            # CAGR/growth displays reflect the real baseline.
            actual_initial = self._passive.initial_cash + self._active.initial_cash
            if actual_initial > 0 and self._start_equity < actual_initial * 0.5:
                self._start_equity = actual_initial
                self._save()   # persist corrected baseline immediately
        except Exception:
            self._start_equity = self._passive.initial_cash + self._active.initial_cash
            self._peak_equity  = self._start_equity

    def _save(self) -> None:
        try:
            _COMPOUND_FILE.write_text(json.dumps({
                "start_equity":    round(self._start_equity, 2),
                "last_weekly_ts":  self._last_weekly_ts,
                "peak_equity":     round(self._peak_equity, 2),
                "compound_log":    self._compound_log[-90:],
                "saved_at":        datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── daily compound ────────────────────────────────────────────────────────

    def daily_check(self, daily_profit_engine) -> float:
        """
        Called every cycle. When DailyProfitEngine transitions to LOCKED today,
        compute the compound amount and add it to the active portfolio's initial
        capital so future position sizing reflects the growth.

        Returns the amount compounded (0.0 if nothing happened).
        """
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).toordinal()
        if today == self._today_ordinal:
            return 0.0

        phase = daily_profit_engine.get_phase()
        if phase != "LOCKED":
            return 0.0

        # First LOCKED detection for today
        self._today_ordinal = today
        pnl = daily_profit_engine._pnl
        if pnl <= 0:
            return 0.0

        compound_amount = round(pnl * DAILY_COMPOUND_RATE, 2)
        if compound_amount < 0.01:
            return 0.0

        # Grow active portfolio's initial capital (so compound_size_factor reflects it)
        self._active.initial_cash = round(
            self._active.initial_cash + compound_amount, 2
        )
        self._daily_compounded_today = compound_amount

        self._compound_log.append({
            "ts":     datetime.now(timezone.utc).isoformat(),
            "type":   "daily_active",
            "amount": compound_amount,
            "pnl":    round(pnl, 2),
        })

        new_equity = self._passive.equity + self._active.equity
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity

        self._save()
        return compound_amount

    # ── weekly compound ───────────────────────────────────────────────────────

    def weekly_check(self) -> dict:
        """
        Called every cycle (checks internally if 7 days have passed).
        Reviews growth and logs a compound snapshot.
        Returns a dict with compound metrics or empty dict if not time yet.
        """
        now = time.time()
        if now - self._last_weekly_ts < WEEKLY_CHECK_INTERVAL:
            return {}

        self._last_weekly_ts = now
        total_equity    = self._passive.equity + self._active.equity
        equity_growth   = (total_equity - self._start_equity) / max(self._start_equity, 1.0)

        if equity_growth < MIN_WEEKLY_EQUITY_GAIN:
            self._save()
            return {"equity_growth": round(equity_growth, 4), "action": "none"}

        # Compound passive portfolio initial capital proportionally
        passive_growth  = (self._passive.equity - self._passive.initial_cash)
        if passive_growth > 0:
            passive_compound = round(passive_growth * DAILY_COMPOUND_RATE, 2)
            self._passive.initial_cash = round(
                self._passive.initial_cash + passive_compound, 2
            )
            self._compound_log.append({
                "ts":     datetime.now(timezone.utc).isoformat(),
                "type":   "weekly_passive",
                "amount": passive_compound,
                "equity": round(total_equity, 2),
            })

        # Update peak
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity

        self._save()
        return {
            "equity_growth": round(equity_growth, 4),
            "total_equity":  round(total_equity, 2),
            "action":        "compounded",
        }

    # ── analytics ─────────────────────────────────────────────────────────────

    def cagr_estimate(self) -> float:
        """
        Annualised growth rate from start to now.
        Only meaningful after at least 30 days of trading.
        """
        if self._start_equity <= 0:
            return 0.0
        total_equity = self._passive.equity + self._active.equity
        ratio = total_equity / self._start_equity
        # Assume records span from first compound log entry
        if self._compound_log:
            try:
                from datetime import datetime, timezone
                first_ts = datetime.fromisoformat(self._compound_log[0]["ts"])
                days     = (datetime.now(timezone.utc) - first_ts).total_seconds() / 86400
                if days >= 7:
                    return round((ratio ** (365 / days) - 1), 4)
            except Exception:
                pass
        return 0.0

    def snapshot(self) -> dict:
        total_equity = self._passive.equity + self._active.equity
        total_growth = (total_equity - self._start_equity) / max(self._start_equity, 1.0)
        return {
            "total_equity":           round(total_equity, 2),
            "start_equity":           round(self._start_equity, 2),
            "total_growth_pct":       round(total_growth * 100, 2),
            "peak_equity":            round(self._peak_equity, 2),
            "cagr_estimate":          round(self.cagr_estimate() * 100, 2),
            "daily_compounded_today": round(self._daily_compounded_today, 2),
            "compound_log_count":     len(self._compound_log),
            "last_weekly_ts":         self._last_weekly_ts,
        }
