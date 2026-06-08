"""
Profit Vault — separates locked daily profits from trading capital.

When DailyProfitEngine transitions to LOCKED:
  1. The withdrawable profit is moved into the Vault (a separate counter).
  2. The active trading capital is capped at the pre-LOCKED level.
  3. The Vault balance accumulates across days until the user withdraws.

This prevents the system from re-risking today's profit after hitting the target:
  - LOCKED phase already blocks new entries (DailyProfitEngine.should_open_new())
  - Vault provides a clear, auditable register of what can be withdrawn

On withdraw:
    vault.withdraw(amount)   → records withdrawal, reduces vault balance

Vault balance persists across restarts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from config.settings import STORAGE_DIR
from meta.daily_profit_engine import THB_PER_USD

_VAULT_FILE = STORAGE_DIR / "profit_vault.json"


class ProfitVault:
    """Accumulates daily locked profits until manually withdrawn."""

    def __init__(self) -> None:
        self._balance: float    = 0.0
        self._total_earned: float = 0.0
        self._total_withdrawn: float = 0.0
        self._log: list[dict]   = []
        self._today_vaulted: float = 0.0
        self._today_ordinal: int   = 0
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _VAULT_FILE.exists():
            return
        try:
            data = json.loads(_VAULT_FILE.read_text())
            self._balance          = float(data.get("balance", 0.0))
            self._total_earned     = float(data.get("total_earned", 0.0))
            self._total_withdrawn  = float(data.get("total_withdrawn", 0.0))
            self._log              = data.get("log", [])[-180:]
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _VAULT_FILE.write_text(json.dumps({
                "balance":         round(self._balance, 2),
                "total_earned":    round(self._total_earned, 2),
                "total_withdrawn": round(self._total_withdrawn, 2),
                "log":             self._log[-180:],
                "saved_at":        datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def deposit_daily_profit(self, daily_profit_engine) -> float:
        """
        Called each cycle. When today's profit is LOCKED, move the withdrawable
        amount into the vault (once per day only).

        Returns the deposited amount (0.0 if nothing deposited).
        """
        today = datetime.now(timezone.utc).toordinal()
        if today == self._today_ordinal:
            return 0.0
        if daily_profit_engine.get_phase() != "LOCKED":
            return 0.0

        withdrawable = daily_profit_engine.withdrawable_usdt()
        if withdrawable <= 0:
            return 0.0

        self._today_ordinal   = today
        self._today_vaulted   = withdrawable
        self._balance         = round(self._balance + withdrawable, 2)
        self._total_earned    = round(self._total_earned + withdrawable, 2)

        self._log.append({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "type":    "deposit",
            "amount":  round(withdrawable, 2),
            "balance": round(self._balance, 2),
        })
        self._save()
        return withdrawable

    def withdraw(self, amount: float) -> float:
        """
        Withdraw from vault.  Amount is capped at available balance.
        Returns actual withdrawn amount.
        """
        amount = max(0.0, min(float(amount), self._balance))
        if amount <= 0:
            return 0.0

        self._balance         = round(self._balance - amount, 2)
        self._total_withdrawn = round(self._total_withdrawn + amount, 2)

        self._log.append({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "type":    "withdraw",
            "amount":  round(amount, 2),
            "amount_thb": round(amount * THB_PER_USD, 0),
            "balance": round(self._balance, 2),
        })
        self._save()
        return amount

    def snapshot(self) -> dict:
        return {
            "vault_balance_usdt":    round(self._balance, 2),
            "vault_balance_thb":     round(self._balance * THB_PER_USD, 0),
            "total_earned_usdt":     round(self._total_earned, 2),
            "total_withdrawn_usdt":  round(self._total_withdrawn, 2),
            "today_vaulted_usdt":    round(self._today_vaulted, 2),
            "log_count":             len(self._log),
        }

    def recent_log(self, n: int = 10) -> list[dict]:
        return self._log[-n:]
