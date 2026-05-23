"""Volume confirmation — avoid entering on low-volume moves."""

import numpy as np


class VolumeFilter:
    MIN_REL_VOLUME = 0.8
    SPIKE_THRESHOLD = 1.5

    @staticmethod
    def relative_volume(volumes: np.ndarray, lookback: int = 20) -> float:
        """Current volume / average volume over lookback period."""
        if len(volumes) < lookback + 1:
            return 1.0
        avg = float(np.mean(volumes[-(lookback + 1):-1]))
        if avg <= 0:
            return 1.0
        return float(volumes[-1] / avg)

    @staticmethod
    def has_volume_spike(volumes: np.ndarray, threshold: float = 1.5, lookback: int = 20) -> bool:
        """True if recent volume is above threshold × average."""
        if len(volumes) < lookback + 1:
            return False
        avg = float(np.mean(volumes[-(lookback + 1):-1]))
        return avg > 0 and volumes[-1] > avg * threshold

    @staticmethod
    def volume_trend(volumes: np.ndarray, period: int = 10) -> float:
        """Positive = increasing volume, negative = decreasing."""
        if len(volumes) < period * 2:
            return 0.0
        recent = float(np.mean(volumes[-period:]))
        older = float(np.mean(volumes[-period * 2:-period]))
        if older <= 0:
            return 0.0
        return (recent - older) / older

    def confirms_entry(self, volumes: np.ndarray) -> bool:
        """True if volume conditions support a new entry."""
        rvol = self.relative_volume(volumes)
        return rvol >= self.MIN_REL_VOLUME

    def confidence_adjust(self, volumes: np.ndarray, confidence: float) -> float:
        """Adjust confidence based on volume profile."""
        rvol = self.relative_volume(volumes)
        if rvol >= 2.0:
            return min(confidence * 1.1, 1.0)
        if rvol >= 1.2:
            return confidence
        if rvol >= 0.8:
            return confidence * 0.9
        return confidence * 0.7
