"""
Daily Equity Snapshot Tracker

Records portfolio equity once per hour and once per day.
Enables charting equity curve and computing real CAGR.

Called from DualEngine hourly report.
Data stored in storage/equity_curve.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from config.settings import STORAGE_DIR

_EQUITY_FILE = STORAGE_DIR / "equity_curve.json"
_MAX_HOURLY  = 168 * 7   # 7 weeks of hourly snapshots
_MAX_DAILY   = 365 * 3   # 3 years of daily snapshots


class EquityTracker:

    def __init__(self) -> None:
        self._hourly: list[dict] = []
        self._daily:  list[dict] = []
        self._last_daily_ordinal: int = 0
        self._load()

    def _load(self) -> None:
        if not _EQUITY_FILE.exists():
            return
        try:
            data = json.loads(_EQUITY_FILE.read_text())
            self._hourly = data.get("hourly", [])[-_MAX_HOURLY:]
            self._daily  = data.get("daily",  [])[-_MAX_DAILY:]
            if self._daily:
                try:
                    last_date = self._daily[-1].get("date", "")
                    dt = datetime.fromisoformat(last_date)
                    self._last_daily_ordinal = dt.toordinal()
                except Exception:
                    pass
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _EQUITY_FILE.write_text(json.dumps({
                "hourly": self._hourly[-_MAX_HOURLY:],
                "daily":  self._daily[-_MAX_DAILY:],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    def record(
        self,
        passive_equity: float,
        active_equity: float,
        regime: str = "",
        fear_greed: float = 50,
    ) -> None:
        now     = datetime.now(timezone.utc)
        total   = round(passive_equity + active_equity, 2)
        snap    = {
            "ts":      now.isoformat(),
            "total":   total,
            "passive": round(passive_equity, 2),
            "active":  round(active_equity, 2),
            "regime":  regime,
            "fg":      round(fear_greed, 1),
        }

        # Hourly snapshot (always)
        self._hourly.append(snap)

        # Daily snapshot (once per UTC day)
        today_ord = now.toordinal()
        if today_ord != self._last_daily_ordinal:
            self._last_daily_ordinal = today_ord
            self._daily.append({**snap, "date": now.strftime("%Y-%m-%d")})

        self._save()

    # ── analytics ─────────────────────────────────────────────────────────────

    def cagr(self) -> float:
        if len(self._daily) < 7:
            return 0.0
        first = self._daily[0]["total"]
        last  = self._daily[-1]["total"]
        if first <= 0:
            return 0.0
        try:
            first_dt = datetime.fromisoformat(self._daily[0]["ts"])
            last_dt  = datetime.fromisoformat(self._daily[-1]["ts"])
            days = max(7, (last_dt - first_dt).days)
            raw = (last / first) ** (365.0 / days) - 1
            return round(min(raw, 99.0), 4)  # cap at 9900% to avoid display overflow
        except Exception:
            return 0.0

    def max_drawdown(self) -> float:
        """Historical max drawdown from equity curve."""
        if len(self._hourly) < 2:
            return 0.0
        peak = self._hourly[0]["total"]
        mdd  = 0.0
        for snap in self._hourly:
            t = snap["total"]
            if t > peak:
                peak = t
            dd = (peak - t) / peak if peak > 0 else 0
            if dd > mdd:
                mdd = dd
        return round(mdd, 4)

    def daily_returns(self) -> list[float]:
        """List of daily return percentages."""
        if len(self._daily) < 2:
            return []
        rets = []
        for i in range(1, len(self._daily)):
            prev = self._daily[i - 1]["total"]
            curr = self._daily[i]["total"]
            if prev > 0:
                rets.append(round((curr - prev) / prev, 6))
        return rets

    def snapshot(self) -> dict:
        import numpy as np
        total  = self._daily[-1]["total"] if self._daily else 0.0
        start  = self._daily[0]["total"]  if self._daily else total
        growth = round((total - start) / start * 100, 2) if start else 0

        rets = self.daily_returns()
        sharpe = 0.0
        if len(rets) >= 7:
            arr    = np.array(rets)
            sharpe = round(
                float(np.mean(arr) / (np.std(arr) + 1e-9)) * (252 ** 0.5), 3
            )

        return {
            "days_tracked":   len(self._daily),
            "start_equity":   round(start, 2),
            "current_equity": round(total, 2),
            "total_growth_pct": growth,
            "cagr_pct":       round(self.cagr() * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "sharpe_ratio":   sharpe,
            "hourly_points":  len(self._hourly),
        }
