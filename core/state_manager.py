import json
import time

from config.settings import STATE_FILE, STORAGE_DIR


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
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4)
            return state

        with open(self.state_path, "r") as f:

            loaded = json.load(f)

        defaults = self.default_state()
        for key, value in defaults.items():
            if key not in loaded:
                loaded[key] = value

        return loaded

    def save_state(self, state=None):

        if state is not None:

            if hasattr(self, "state"):
                self.state.update(state)
            else:
                self.state = state

        self.state["last_update"] = time.time()

        with open(self.state_path, "w") as f:

            json.dump(self.state, f, indent=4)

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