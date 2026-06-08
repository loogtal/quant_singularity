"""
Adaptive Performance Controller (APC)

Monitors live trading performance and automatically adjusts system behaviour
to maximise edge while protecting capital.

Rules (all based on rolling windows of recent trades):

CONFIDENCE THRESHOLD
  WR_last20 < 30%  → raise MIN_CONFIDENCE to 0.72 (be more selective)
  WR_last20 > 55%  → lower MIN_CONFIDENCE to 0.62 (take more setups)
  Default          → 0.65

CAPITAL SCALING (auto-grow active capital when performing well)
  After 20 trades with WR > 40% AND realized_pnl > +2%:
    → Add 10% of passive profit to active capital
    → Cap: active never > 40% of total portfolio
  When WR < 30% over last 20 trades:
    → Reduce active capital by 10% (protect capital)

DAILY TARGET ADAPTATION
  If daily target not reached by 20:00 UTC:
    → Allow one extra scan with 0.7x size (last-hour push)
    → Only if daily_loss < 1% so far (not in drawdown)

FUNDING ARB PRIORITY
  If funding_arb WR > 70% (bandit) AND mode is available:
    → Boost funding_arb scan frequency (every 15 min vs 30 min)
    → Only in sideways/volatile regimes
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from config.settings import STORAGE_DIR, MIN_CONFIDENCE

_APC_FILE = STORAGE_DIR / "apc_state.json"

_WR_HIGH     = 0.55   # WR above this = relax confidence threshold
_WR_LOW      = 0.30   # WR below this = tighten confidence threshold
_WR_SCALE_UP = 0.40   # WR above this = eligible for capital scale-up
_WINDOW      = 20     # rolling window for WR calculation
_SCALE_COOLDOWN = 86400  # wait 24h between capital adjustments


class AdaptiveController:
    """
    Reads live trade outcomes and dynamically adjusts:
      - confidence threshold
      - active capital allocation
      - funding arb scan frequency
    """

    def __init__(self, active_portfolio, passive_portfolio) -> None:
        self._active  = active_portfolio
        self._passive = passive_portfolio
        self._recent_trades: deque[dict] = deque(maxlen=_WINDOW)
        self._last_scale_ts: float = 0.0
        self._last_daily_push_day: int = -1
        self._dynamic_confidence: float = MIN_CONFIDENCE
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _APC_FILE.exists():
            return
        try:
            data = json.loads(_APC_FILE.read_text())
            self._last_scale_ts      = float(data.get("last_scale_ts", 0.0))
            self._dynamic_confidence = float(data.get("dynamic_confidence", MIN_CONFIDENCE))
            for t in data.get("recent_trades", []):
                self._recent_trades.append(t)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _APC_FILE.write_text(json.dumps({
                "last_scale_ts":      self._last_scale_ts,
                "dynamic_confidence": round(self._dynamic_confidence, 4),
                "recent_trades":      list(self._recent_trades)[-_WINDOW:],
                "saved_at":           datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def record_trade(self, pnl: float, mode: str, regime: str) -> None:
        """Call after every active trade closes."""
        self._recent_trades.append({
            "pnl":    round(pnl, 4),
            "won":    pnl >= 0,
            "mode":   mode,
            "regime": regime,
            "ts":     time.time(),
        })
        self._update_confidence()
        self._maybe_scale_capital()
        self._save()

    def get_min_confidence(self) -> float:
        """Current dynamic confidence threshold."""
        return self._dynamic_confidence

    def should_do_late_push(self, daily_pnl: float, daily_loss_limit: float) -> bool:
        """
        Returns True at 20:00 UTC if daily target not yet met and we're not in loss.
        Allows one extra aggressive scan with lower confidence.
        """
        now = datetime.now(timezone.utc)
        today = now.toordinal()
        if now.hour != 20:
            return False
        if today == self._last_daily_push_day:
            return False
        if daily_pnl <= 0 and abs(daily_pnl) > daily_loss_limit * 0.5:
            return False
        self._last_daily_push_day = today
        return True

    def funding_arb_scan_interval(self, bandit_wr: float) -> int:
        """
        Returns funding arb scan interval in seconds.
        If bandit WR > 70%: scan every 15 min. Else: every 30 min.
        """
        return 900 if bandit_wr > 0.70 else 1800

    def snapshot(self) -> dict:
        wr = self._rolling_wr()
        return {
            "rolling_wr_20":         round(wr, 4),
            "dynamic_confidence":    round(self._dynamic_confidence, 4),
            "active_capital":        round(self._active.initial_cash, 2),
            "passive_capital":       round(self._passive.initial_cash, 2),
            "recent_trades":         len(self._recent_trades),
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _rolling_wr(self) -> float:
        if not self._recent_trades:
            return 0.50
        return sum(1 for t in self._recent_trades if t["won"]) / len(self._recent_trades)

    def _update_confidence(self) -> None:
        if len(self._recent_trades) < 10:
            return   # not enough data
        wr = self._rolling_wr()
        if wr > _WR_HIGH:
            new_conf = max(0.62, MIN_CONFIDENCE - 0.03)
        elif wr < _WR_LOW:
            new_conf = min(0.75, MIN_CONFIDENCE + 0.07)
        else:
            new_conf = MIN_CONFIDENCE

        if abs(new_conf - self._dynamic_confidence) >= 0.01:
            self._dynamic_confidence = round(new_conf, 4)

    def _maybe_scale_capital(self) -> None:
        now = time.time()
        if now - self._last_scale_ts < _SCALE_COOLDOWN:
            return
        if len(self._recent_trades) < _WINDOW:
            return

        wr = self._rolling_wr()
        total = self._active.initial_cash + self._passive.initial_cash
        active_pct = self._active.initial_cash / max(total, 1.0)
        realized_gain = (self._active.equity - self._active.initial_cash) / max(self._active.initial_cash, 1.0)

        if wr >= _WR_SCALE_UP and realized_gain >= 0.02 and active_pct < 0.40:
            # Scale up: transfer 10% of passive profit to active
            passive_profit = max(0.0, self._passive.equity - self._passive.initial_cash)
            transfer = round(passive_profit * 0.10, 2)
            if transfer >= 5.0:
                self._passive.initial_cash = round(self._passive.initial_cash - transfer, 2)
                self._active.initial_cash  = round(self._active.initial_cash  + transfer, 2)
                self._last_scale_ts = now

        elif wr < _WR_LOW and active_pct > 0.20:
            # Scale down: reduce active capital by 10%
            reduction = round(self._active.initial_cash * 0.10, 2)
            if reduction >= 5.0:
                self._active.initial_cash  = round(self._active.initial_cash  - reduction, 2)
                self._passive.initial_cash = round(self._passive.initial_cash + reduction, 2)
                self._last_scale_ts = now
