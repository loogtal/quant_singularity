"""
Position Sync — restores open Binance positions into PortfolioEngine on startup.

Problem: if the bot crashes or is restarted, it loses its in-memory position tracking.
The positions on Binance remain open but the bot has no knowledge of them.
This leads to:
  - Missing TP/SL monitoring (positions drift unmanaged)
  - Opening duplicate positions (bot thinks it has no positions)

Solution: on startup, fetch all open positions from Binance and restore them into
the appropriate portfolio (passive or active) based on the persisted state file.

Priority order for restoring:
  1. portfolio_snapshot in system_state.json (has full position details)
  2. Binance API (position size + entry price, but no TP/SL)
  3. Skip if neither available
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from config.settings import LIVE_MODE
from core.logger import get_logger

_log = get_logger()


def sync_positions_from_state(
    state_file: Path,
    passive_portfolio,
    active_portfolio,
) -> int:
    """
    Restore open positions from the last saved state file into portfolio engines.
    Returns number of positions restored.

    Safe to call even if state file is missing or corrupt — returns 0 in that case.
    """
    import json

    if not state_file.exists():
        return 0

    try:
        state = json.loads(state_file.read_text())
    except Exception:
        return 0

    snap = state.get("portfolio_snapshot", {})
    passive_snap = snap.get("passive", {})
    active_snap  = snap.get("active",  {})

    restored = 0

    for pos_list, portfolio, label in [
        (passive_snap.get("open_positions", []), passive_portfolio, "passive"),
        (active_snap.get("open_positions",  []), active_portfolio,  "active"),
    ]:
        for pos in pos_list:
            sym = pos.get("symbol")
            if not sym:
                continue
            if portfolio.has_open_position(sym):
                continue   # already tracked

            # Restore into portfolio's position list
            pos_copy = dict(pos)
            pos_copy.setdefault("strategy",       label)
            pos_copy.setdefault("opened_at",      time.time())
            pos_copy.setdefault("trailing_active", False)
            pos_copy.setdefault("leverage",        1)
            pos_copy.setdefault("signal_mode",     "")
            pos_copy.setdefault("passive_mode",    "")
            pos_copy["_restored"] = True   # tag so we know it was restored

            try:
                portfolio.positions.append(pos_copy)
                leverage = float(pos_copy.get("leverage", 1)) or 1.0
                margin = pos_copy.get("margin")
                if margin is None:
                    margin = pos_copy.get("position_value", 0.0) / leverage
                portfolio.cash = max(0.0, portfolio.cash - margin)
                restored += 1
                _log.info(
                    f"[PositionSync] restored {label} {sym} {pos_copy.get('side')} "
                    f"entry={pos_copy.get('entry_price')} sl={pos_copy.get('stop_loss')}"
                )
            except Exception as e:
                _log.warning(f"[PositionSync] failed to restore {sym}: {e}")

    return restored


def sync_positions_from_binance(
    exchange,
    passive_portfolio,
    active_portfolio,
    state_file: Optional[Path] = None,
) -> int:
    """
    Fallback: fetch actual open positions from Binance API and restore them.
    Used when state file doesn't have position details.

    Note: Binance only gives us size + entry price, not TP/SL.
    Assigns positions to active portfolio (intraday assumption).
    Returns number of positions restored.
    """
    if not LIVE_MODE or exchange is None:
        return 0

    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        _log.warning(f"[PositionSync] Binance fetch failed: {e}")
        return 0

    restored = 0
    for pos in positions:
        contracts = float(pos.get("contracts") or pos.get("positionAmt") or 0)
        if abs(contracts) < 1e-9:
            continue   # no position

        sym        = pos.get("symbol") or pos.get("info", {}).get("symbol", "")
        side_raw   = pos.get("side") or ("long" if contracts > 0 else "short")
        side       = side_raw.upper()
        entry      = float(pos.get("entryPrice") or pos.get("info", {}).get("entryPrice") or 0)
        notional   = abs(contracts) * entry

        # Skip if already tracked
        if (passive_portfolio.has_open_position(sym) or
                active_portfolio.has_open_position(sym)):
            continue

        pos_dict = {
            "symbol":          sym,
            "side":            side,
            "size":            abs(contracts),
            "entry_price":     entry,
            "position_value":  round(notional, 2),
            "current_price":   entry,
            "stop_loss":       None,
            "take_profit":     None,
            "strategy":        "active",
            "regime":          "unknown",
            "opened_at":       time.time(),
            "min_hold_hours":  0,
            "max_hold_hours":  0,
            "trailing_active": False,
            "leverage":        1,
            "signal_mode":     "restored",
            "passive_mode":    "",
            "_restored":       True,
            "_no_sl":          True,   # flag: no TP/SL, manage manually
        }

        try:
            active_portfolio.positions.append(pos_dict)
            active_portfolio.cash = max(0.0, active_portfolio.cash - notional)
            restored += 1
            _log.info(
                f"[PositionSync] Binance restored {sym} {side} "
                f"size={abs(contracts)} entry={entry:.4f} (no TP/SL)"
            )
        except Exception as e:
            _log.warning(f"[PositionSync] failed to restore {sym}: {e}")

    return restored
