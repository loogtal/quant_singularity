import json
import time
from pathlib import Path

import numpy as np

from config.settings import STATE_FILE, STORAGE_DIR


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class StateManager:
    def __init__(self):
        self.state_path = str(STATE_FILE)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def default_state(self):

        return {

            "created_at": time.time(),

            "last_update": time.time(),

            "daily_pnl": 0,

            "total_trades": 0,

            "winning_trades": 0,

            "losing_trades": 0,

            "peak_equity": 10000,

            "max_drawdown": 0,

            "strategy_performance": {},

            "regime_memory": {},

            "active_strategies": [],

            "market_bias": "neutral"
        }

    def load_state(self):

        if not STATE_FILE.exists():
            state = self.default_state()
            self._write_state_file(state)
            return state

        try:
            with open(self.state_path, "r") as f:
                loaded = json.load(f)
        except json.JSONDecodeError as e:
            backup = STATE_FILE.with_suffix(
                f".corrupt.{int(time.time())}.json"
            )
            STATE_FILE.rename(backup)
            print(
                f"[State] Corrupt state file backed up to {backup.name} "
                f"({e}); starting fresh."
            )
            state = self.default_state()
            self._write_state_file(state)
            return state

        defaults = self.default_state()
        for key, value in defaults.items():
            if key not in loaded:
                loaded[key] = value

        return loaded

    def _write_state_file(self, state: dict) -> None:
        tmp = Path(self.state_path).with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(_json_safe(state), f, indent=4)
        tmp.replace(self.state_path)

    def save_state(self, state=None):

        if state is not None:

            if hasattr(self, "state"):
                self.state.update(state)
            else:
                self.state = state

        self.state["last_update"] = time.time()

        self._write_state_file(self.state)

    def update_metric(self, key, value):

        self.state[key] = value

        self.save_state()

    def increment_trade(self, won=False):

        self.state["total_trades"] += 1

        if won:
            self.state["winning_trades"] += 1
        else:
            self.state["losing_trades"] += 1

        self.save_state()

    def update_equity(self, equity):

        peak = max(self.state.get("peak_equity", equity), equity)
        self.state["peak_equity"] = peak

        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > self.state.get("max_drawdown", 0):
                self.state["max_drawdown"] = round(drawdown, 4)

        self.save_state()