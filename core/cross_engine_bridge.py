"""
Cross-Engine Learning Bridge — Passive ↔ Active shared intelligence.

Enables the two bots to learn from each other without coupling their logic:

  1. HOT COINS REGISTRY
     Active scanner finds coins with strong momentum/mean-reversion setups.
     Passive strategy checks if those same coins also have a good 1D trend.
     → Active insights can fast-track passive entries on the right coins.

  2. SYMBOLS TO AVOID (shared blacklist)
     When Active gets stopped out on a coin, that coin enters a shared
     avoid list for both bots for a configurable period.
     Passive avoids opening new positions on those coins.

  3. REGIME CONFIDENCE SHARING
     Active strategy processes 15M bars constantly — it builds a "bottom-up"
     regime signal from individual coin behaviour.
     If ≥60% of active coins are trending short, bridge signals "bear pressure"
     to the passive side, which can tighten stops or skip counter-trend entries.

  4. DAILY CROSS-PERFORMANCE LOG
     Tracks which engine made more profit each day → informs weekly capital
     rebalance in CompoundManager.

All state is persisted so insights survive restarts.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from config.settings import STORAGE_DIR

_BRIDGE_FILE = STORAGE_DIR / "cross_engine_state.json"

AVOID_DURATION_ACTIVE  = 1800    # 30 min after active stop-out
AVOID_DURATION_PASSIVE = 14400   # 4 h after passive stop-out
HOT_COINS_TTL          = 300     # 5 min — hot coins refresh
BEAR_PRESSURE_THRESH   = 0.60    # fraction of short coins to signal bear pressure
BULL_PRESSURE_THRESH   = 0.60    # fraction of long coins to signal bull pressure


class CrossEngineBridge:
    """
    Singleton-style bridge: both DualEngine bots share one instance.
    """

    def __init__(self) -> None:
        self._avoid: dict[str, float] = {}          # symbol → expiry ts
        self._hot_passive: list[str]  = []          # symbols active found trending
        self._hot_active:  list[str]  = []          # symbols passive found trending
        self._hot_ts: float           = 0.0
        self._active_sides: dict[str, str] = {}     # symbol → "LONG"/"SHORT" (active positions)
        self._daily_pnl: dict[str, float]  = defaultdict(float)  # "passive"/"active" → today pnl
        self._daily_ordinal: int = 0
        self._perf_log: list[dict]    = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _BRIDGE_FILE.exists():
            return
        try:
            data = json.loads(_BRIDGE_FILE.read_text())
            self._avoid    = {k: float(v) for k, v in data.get("avoid", {}).items()}
            self._perf_log = data.get("perf_log", [])[-90:]
        except Exception:
            pass

    def _save(self) -> None:
        try:
            # Prune expired entries before saving
            now = time.time()
            active_avoid = {k: v for k, v in self._avoid.items() if v > now}
            _BRIDGE_FILE.write_text(json.dumps({
                "avoid":    active_avoid,
                "perf_log": self._perf_log[-90:],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── shared avoid list ─────────────────────────────────────────────────────

    def flag_avoid(self, symbol: str, source: str) -> None:
        """
        Mark a symbol to avoid for both engines.
        source = "active" or "passive"
        """
        duration = AVOID_DURATION_ACTIVE if source == "active" else AVOID_DURATION_PASSIVE
        self._avoid[symbol] = time.time() + duration
        self._save()

    def is_avoided(self, symbol: str) -> bool:
        expiry = self._avoid.get(symbol, 0.0)
        if time.time() < expiry:
            return True
        if symbol in self._avoid:
            del self._avoid[symbol]
        return False

    def avoided_symbols(self) -> list[str]:
        now = time.time()
        return [s for s, exp in self._avoid.items() if exp > now]

    # ── hot coins registry ─────────────────────────────────────────────────────

    def register_active_scan(self, coins: list[dict]) -> None:
        """
        Active scanner just found good intraday setups.
        Extract the trending ones and share with passive.
        coins = list of {"symbol": ..., "score": ..., "rsi": ...}
        """
        self._hot_passive = [
            c["symbol"] for c in coins
            if c.get("score", 0) >= 0.60
        ]
        self._hot_ts = time.time()

        # Track active sides from RSI (>60 = bullish momentum, <40 = bearish)
        for c in coins:
            sym = c["symbol"]
            rsi = c.get("rsi", 50)
            if rsi > 60:
                self._active_sides[sym] = "LONG"
            elif rsi < 40:
                self._active_sides[sym] = "SHORT"

    def register_passive_scan(self, coins: list[dict]) -> None:
        """
        Passive scanner just found good trend setups.
        Share trending coins with active as trend-following targets.
        coins = list of {"symbol": ..., "score": ..., "trend": ...}
        """
        self._hot_active = [
            c["symbol"] for c in coins
            if c.get("score", 0) >= 0.55
        ]

    def hot_coins_for_passive(self) -> list[str]:
        """Coins with strong active momentum that passive should check first."""
        if time.time() - self._hot_ts > HOT_COINS_TTL:
            return []
        return list(self._hot_passive)

    def hot_coins_for_active(self) -> list[str]:
        """Coins in a confirmed passive trend — active can ride the intraday swings."""
        return list(self._hot_active)

    # ── regime pressure (bottom-up from active) ───────────────────────────────

    def get_regime_pressure(self) -> Optional[str]:
        """
        Derive a regime signal from active positions' directional bias.
        Returns "bear_pressure", "bull_pressure", or None.
        """
        if not self._active_sides:
            return None
        total  = len(self._active_sides)
        shorts = sum(1 for s in self._active_sides.values() if s == "SHORT")
        longs  = total - shorts
        if shorts / total >= BEAR_PRESSURE_THRESH:
            return "bear_pressure"
        if longs / total >= BULL_PRESSURE_THRESH:
            return "bull_pressure"
        return None

    def update_active_position(self, symbol: str, side: str) -> None:
        """Track live active positions for regime pressure calculation."""
        self._active_sides[symbol] = side

    def clear_active_position(self, symbol: str) -> None:
        self._active_sides.pop(symbol, None)

    # ── cross-performance tracking ────────────────────────────────────────────

    def record_pnl(self, engine: str, pnl: float) -> None:
        """Record PnL for passive or active engine. Call on each trade close."""
        today = datetime.now(timezone.utc).toordinal()
        if today != self._daily_ordinal:
            if self._daily_ordinal > 0:
                self._perf_log.append({
                    "date":    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "passive": round(self._daily_pnl["passive"], 4),
                    "active":  round(self._daily_pnl["active"],  4),
                    "winner":  "passive" if self._daily_pnl["passive"] > self._daily_pnl["active"] else "active",
                })
                self._save()
            self._daily_pnl  = defaultdict(float)
            self._daily_ordinal = today
        self._daily_pnl[engine] += pnl

    def better_engine_7d(self) -> str:
        """Returns 'passive', 'active', or 'tied' based on last 7 day pnl."""
        if len(self._perf_log) < 3:
            return "tied"
        recent = self._perf_log[-7:]
        p_total = sum(d["passive"] for d in recent)
        a_total = sum(d["active"]  for d in recent)
        if p_total > a_total * 1.1:
            return "passive"
        if a_total > p_total * 1.1:
            return "active"
        return "tied"

    def snapshot(self) -> dict:
        return {
            "avoided_symbols":    self.avoided_symbols(),
            "hot_for_passive":    self.hot_coins_for_passive(),
            "hot_for_active":     self.hot_coins_for_active(),
            "regime_pressure":    self.get_regime_pressure(),
            "better_engine_7d":   self.better_engine_7d(),
            "today_passive_pnl":  round(self._daily_pnl["passive"], 2),
            "today_active_pnl":   round(self._daily_pnl["active"],  2),
        }
