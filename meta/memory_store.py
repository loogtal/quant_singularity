"""Lightweight memory — symbol edge and recent outcomes."""

import json
import time
from pathlib import Path

from config.settings import STORAGE_DIR

MEMORY_FILE = STORAGE_DIR / "memory.json"
MAX_ENTRIES = 200


class MemoryStore:
    def __init__(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            try:
                return json.loads(MEMORY_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"symbols": {}, "events": []}

    def _save(self) -> None:
        MEMORY_FILE.write_text(json.dumps(self._data, indent=2))

    def record_trade(self, symbol: str, pnl: float, side: str) -> None:
        sym = self._data.setdefault("symbols", {}).setdefault(
            symbol, {"trades": 0, "wins": 0, "pnl": 0.0}
        )
        sym["trades"] += 1
        sym["pnl"] = round(sym["pnl"] + pnl, 2)
        if pnl >= 0:
            sym["wins"] += 1

        self._data.setdefault("events", []).append(
            {"ts": time.time(), "symbol": symbol, "side": side, "pnl": round(pnl, 2)}
        )
        self._data["events"] = self._data["events"][-MAX_ENTRIES:]
        self._save()

    def symbol_bias(self, symbol: str) -> float:
        """0.85–1.15 multiplier from historical symbol performance."""
        sym = self._data.get("symbols", {}).get(symbol)
        if not sym or sym["trades"] < 3:
            return 1.0
        wr = sym["wins"] / sym["trades"]
        if wr >= 0.6:
            return 1.1
        if wr <= 0.25:
            return 0.85
        return 1.0
