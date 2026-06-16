"""
Daily profit tracking engine for the active bot.
Targets a configurable USDT amount per day (default: ~2.86% of active capital ≈ 300 THB/day).
Displays progress in both USDT and THB.

Phases (based on % of daily target achieved):
  HUNTING    <30%  → aggressive: 1.25x sizing, full entry mode
  ON_TRACK   30-70% → normal:    1.00x sizing
  PROTECTING 70-100% → cautious: 0.70x sizing, tighter SL preference
  LOCKED     ≥100%  → no new entries (protect gains)

Resets at UTC midnight.

Withdrawal logic:
  withdrawable_usdt() = today's realized PnL minus a 20% compound buffer
  log_withdrawal(amount) records the withdrawal and reduces available balance
  All withdrawals are persisted to storage/withdrawal_log.json for auditing
"""

import json
from datetime import datetime, timezone
from config.settings import _env_float, STORAGE_DIR

_WITHDRAWAL_LOG = STORAGE_DIR / "withdrawal_log.json"
_STATE_FILE     = STORAGE_DIR / "daily_profit_state.json"
_COMPOUND_BUFFER_PCT = 0.20   # keep 20% of daily profits for compounding

# THB/USD exchange rate for display (update via QS_THB_PER_USD env var)
THB_PER_USD = _env_float("QS_THB_PER_USD", 35.0)

# Override with a fixed USDT target (0 = derive from % of capital)
DAILY_TARGET_USDT_OVERRIDE = _env_float("QS_DAILY_TARGET_USDT", 0.0)

# Fraction of active capital = default daily target (~2.86% of 300 USDT ≈ 8.57 USD = 300 THB)
DAILY_TARGET_PCT = _env_float("QS_ACTIVE_DAILY_TARGET_PCT", 0.0286)

# Phase boundaries (fraction of daily target)
_THRESHOLDS = (0.30, 0.70, 1.00)

# Aggression multipliers per phase
_PHASE_MULT = {
    "HUNTING":    1.25,
    "ON_TRACK":   1.00,
    "PROTECTING": 0.70,
    "LOCKED":     0.00,
}


def _utc_today() -> int:
    return datetime.now(timezone.utc).toordinal()


class DailyProfitEngine:
    """
    Tracks today's realized PnL for the active bot.
    Adjusts entry sizing automatically based on how close we are to the daily target.
    """

    def __init__(self, active_capital: float = 300.0):
        self._capital   = active_capital
        self._pnl       = 0.0        # today's realized PnL (USDT)
        self._day       = _utc_today()
        self._target    = self._compute_target(active_capital)
        # Restore today's PnL so the LOCKED/HUNTING phase survives a mid-day restart.
        self._load()

    # ── internals ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _STATE_FILE.exists():
            return
        try:
            data = json.loads(_STATE_FILE.read_text())
            # Only restore if the saved snapshot is from the current UTC day;
            # a stale day means the daily window already rolled over → start at 0.
            if int(data.get("day", 0)) == self._day:
                self._pnl = float(data.get("pnl", 0.0))
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _STATE_FILE.write_text(json.dumps({
                "day":      self._day,
                "pnl":      round(self._pnl, 4),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    @staticmethod
    def _compute_target(capital: float) -> float:
        if DAILY_TARGET_USDT_OVERRIDE > 0:
            return DAILY_TARGET_USDT_OVERRIDE
        return round(capital * DAILY_TARGET_PCT, 4)

    def _maybe_reset(self) -> None:
        today = _utc_today()
        if today != self._day:
            self._pnl = 0.0
            self._day = today
            self._save()

    # ── public API ────────────────────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        """Call when active capital changes (e.g. after rebalance)."""
        self._capital = new_capital
        if DAILY_TARGET_USDT_OVERRIDE <= 0:
            self._target = self._compute_target(new_capital)

    def compound_profit(self, realized_pnl: float) -> float:
        """
        Calculate how much to compound back into capital vs withdraw.

        Rule: keep 80% of today's profits as compound fuel (already reflected
        in _COMPOUND_BUFFER_PCT = 0.20 → 20% reserved, 80% compoundable).

        Returns the suggested reinvestment amount (not automatically applied;
        DualEngine calls capital_allocator.rebalance() which handles the transfer).
        """
        self._maybe_reset()
        if self._pnl <= 0:
            return 0.0
        # Compound amount = profits minus the withdrawable portion
        compound = max(0.0, round(self._pnl - self.withdrawable_usdt(), 2))
        return compound

    def record_pnl(self, pnl: float) -> None:
        """Call after every active trade close."""
        self._maybe_reset()
        self._pnl += pnl
        self._save()

    def get_phase(self) -> str:
        self._maybe_reset()
        if self._target <= 0:
            return "ON_TRACK"
        ratio = self._pnl / self._target
        if ratio >= _THRESHOLDS[2]:
            return "LOCKED"
        if ratio >= _THRESHOLDS[1]:
            return "PROTECTING"
        if ratio >= _THRESHOLDS[0]:
            return "ON_TRACK"
        return "HUNTING"

    def get_aggression_mult(self) -> float:
        """Sizing multiplier for new active entries. Returns 0.0 when LOCKED."""
        return _PHASE_MULT[self.get_phase()]

    def should_open_new(self) -> bool:
        """False when daily target is hit (LOCKED phase)."""
        return self.get_phase() != "LOCKED"

    def progress_pct(self) -> float:
        self._maybe_reset()
        if self._target <= 0:
            return 0.0
        return round(min(self._pnl / self._target * 100, 999.9), 1)

    def withdrawable_usdt(self) -> float:
        """
        Amount safely available to withdraw today without disrupting compounding.
        Only positive once daily target is hit.
        Reserves _COMPOUND_BUFFER_PCT of daily profit as compound fuel.
        """
        self._maybe_reset()
        if self._pnl <= 0:
            return 0.0
        buffer = self._target * _COMPOUND_BUFFER_PCT
        return max(0.0, round(self._pnl - buffer, 2))

    def log_withdrawal(self, amount: float) -> None:
        """
        Record a user withdrawal.  Deducts from today's tracked PnL and
        appends to the persistent withdrawal audit log.
        """
        amount = max(0.0, min(float(amount), self.withdrawable_usdt()))
        if amount <= 0:
            return
        self._pnl = max(0.0, self._pnl - amount)
        self._save()
        try:
            records = json.loads(_WITHDRAWAL_LOG.read_text()) if _WITHDRAWAL_LOG.exists() else []
            records.append({
                "ts":           datetime.now(timezone.utc).isoformat(),
                "amount_usdt":  round(amount, 2),
                "amount_thb":   round(amount * THB_PER_USD, 0),
            })
            _WITHDRAWAL_LOG.write_text(json.dumps(records[-180:], indent=2))   # keep 180 records
        except Exception:
            pass

    def withdrawal_history(self) -> list[dict]:
        """Return the last 30 withdrawal records."""
        try:
            return json.loads(_WITHDRAWAL_LOG.read_text())[-30:]
        except Exception:
            return []

    def snapshot(self) -> dict:
        """Dict for dashboard / state persistence."""
        self._maybe_reset()
        phase = self.get_phase()
        withdrable = self.withdrawable_usdt()
        return {
            "pnl_today_usdt":     round(self._pnl, 2),
            "target_usdt":        round(self._target, 2),
            "pnl_today_thb":      round(self._pnl  * THB_PER_USD, 0),
            "target_thb":         round(self._target * THB_PER_USD, 0),
            "progress_pct":       self.progress_pct(),
            "phase":              phase,
            "aggression_mult":    self.get_aggression_mult(),
            "withdrawable_usdt":  withdrable,
            "withdrawable_thb":   round(withdrable * THB_PER_USD, 0),
            "thb_per_usd":        THB_PER_USD,
        }
