"""Auto-tune min_confidence and risk from recent session performance."""

import json

from config.settings import MIN_CONFIDENCE, STORAGE_DIR

TUNING_FILE = STORAGE_DIR / "tuning.json"


class AutoTuner:
    BOUNDS = (0.50, 0.72)

    def __init__(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.params = self._load()
        self._last_tune_trade_count = self.params.get("last_tune_trade_count", 0)

    def _load(self) -> dict:
        if TUNING_FILE.exists():
            try:
                data = json.loads(TUNING_FILE.read_text())
                data.setdefault("last_tune_trade_count", 0)
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "min_confidence": MIN_CONFIDENCE,
            "risk_multiplier": 1.0,
            "last_tune_trade_count": 0,
        }

    def _save(self) -> None:
        TUNING_FILE.write_text(json.dumps(self.params, indent=2))

    def apply_to_meta(self, meta: dict, perf: dict) -> dict:
        meta = dict(meta)
        regime_min = meta.get("min_confidence", MIN_CONFIDENCE)
        tuned = self.params.get("min_confidence", MIN_CONFIDENCE)
        daily_pct = perf.get("daily_pnl_pct", 0)
        dd = perf.get("drawdown", 0)

        # On a flat/up day with low drawdown, don't let stale tuning block regime defaults
        if daily_pct >= 0 and dd < 0.05:
            meta["min_confidence"] = min(tuned, regime_min)
        else:
            meta["min_confidence"] = max(tuned, regime_min)

        meta["size_multiplier"] = round(
            meta.get("size_multiplier", 1.0) * self.params.get("risk_multiplier", 1.0),
            2,
        )
        return meta

    def tune(self, perf: dict) -> dict:
        total = perf.get("total_trades", 0)
        winrate = perf.get("winrate", 0.5)
        daily_pct = perf.get("daily_pnl_pct", 0)
        dd = perf.get("drawdown", 0)

        min_conf = self.params.get("min_confidence", MIN_CONFIDENCE)
        risk_mult = self.params.get("risk_multiplier", 1.0)

        if total >= 5 and total > self._last_tune_trade_count:
            if winrate < 0.35:
                min_conf = min(min_conf + 0.02, self.BOUNDS[1])
                risk_mult = max(risk_mult * 0.85, 0.5)
            elif winrate > 0.55 and daily_pct >= 0:
                min_conf = max(min_conf - 0.01, self.BOUNDS[0])
                risk_mult = min(risk_mult * 1.05, 1.2)
            self._last_tune_trade_count = total

        # Relax toward baseline on neutral days so old losses don't lock out entries
        if daily_pct >= 0 and dd < 0.05 and min_conf > MIN_CONFIDENCE:
            min_conf = max(min_conf - 0.01, MIN_CONFIDENCE)

        if daily_pct <= -0.02:
            risk_mult = max(risk_mult * 0.7, 0.4)

        self.params = {
            "min_confidence": round(min_conf, 3),
            "risk_multiplier": round(risk_mult, 3),
            "last_tune_trade_count": self._last_tune_trade_count,
        }
        self._save()
        return self.params
