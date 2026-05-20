"""Auto-tune min_confidence and risk from recent session performance."""

import json
from pathlib import Path

from config.settings import MIN_CONFIDENCE, STORAGE_DIR

TUNING_FILE = STORAGE_DIR / "tuning.json"


class AutoTuner:
    BOUNDS = (0.50, 0.72)

    def __init__(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.params = self._load()

    def _load(self) -> dict:
        if TUNING_FILE.exists():
            try:
                return json.loads(TUNING_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "min_confidence": MIN_CONFIDENCE,
            "risk_multiplier": 1.0,
        }

    def _save(self) -> None:
        TUNING_FILE.write_text(json.dumps(self.params, indent=2))

    def apply_to_meta(self, meta: dict, perf: dict) -> dict:
        meta = dict(meta)
        meta["min_confidence"] = self.params.get("min_confidence", MIN_CONFIDENCE)
        meta["size_multiplier"] = round(
            meta.get("size_multiplier", 1.0) * self.params.get("risk_multiplier", 1.0),
            2,
        )
        return meta

    def tune(self, perf: dict) -> dict:
        total = perf.get("total_trades", 0)
        winrate = perf.get("winrate", 0.5)
        daily_pct = perf.get("daily_pnl_pct", 0)

        min_conf = self.params.get("min_confidence", MIN_CONFIDENCE)
        risk_mult = self.params.get("risk_multiplier", 1.0)

        if total >= 5:
            if winrate < 0.35:
                min_conf = min(min_conf + 0.02, self.BOUNDS[1])
                risk_mult = max(risk_mult * 0.85, 0.5)
            elif winrate > 0.55 and daily_pct >= 0:
                min_conf = max(min_conf - 0.01, self.BOUNDS[0])
                risk_mult = min(risk_mult * 1.05, 1.2)

        if daily_pct <= -0.02:
            risk_mult = max(risk_mult * 0.7, 0.4)

        self.params = {
            "min_confidence": round(min_conf, 3),
            "risk_multiplier": round(risk_mult, 3),
        }
        self._save()
        return self.params
