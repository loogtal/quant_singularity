"""Merge meta, settings, and reflection into one confidence gate."""

from config.settings import MIN_CONFIDENCE
from config.constants import ACTION_INCREASE


class ConfidenceManager:
    def resolve(
        self,
        meta: dict,
        reflection: dict,
        signal_confidence: float,
    ) -> tuple[float, bool, str]:
        min_conf = meta.get("min_confidence", MIN_CONFIDENCE)

        # REDUCE only shrinks position size (kill_switch) — do not raise the confidence bar
        if reflection.get("action") == ACTION_INCREASE:
            min_conf = max(min_conf - 0.02, 0.52)

        if signal_confidence < min_conf:
            return min_conf, False, f"conf {signal_confidence:.2f} < {min_conf:.2f}"

        return min_conf, True, "ok"
