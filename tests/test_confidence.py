"""Tests for meta/confidence_manager.py."""

from meta.confidence_manager import ConfidenceManager


class TestConfidenceManager:
    def setup_method(self):
        self.cm = ConfidenceManager()

    def test_passes_above_threshold(self):
        min_conf, ok, reason = self.cm.resolve(
            meta={"min_confidence": 0.55},
            reflection={},
            signal_confidence=0.60,
        )
        assert ok is True
        assert reason == "ok"

    def test_fails_below_threshold(self):
        min_conf, ok, reason = self.cm.resolve(
            meta={"min_confidence": 0.55},
            reflection={},
            signal_confidence=0.50,
        )
        assert ok is False
        assert "conf" in reason

    def test_increase_lowers_threshold(self):
        min_conf, ok, reason = self.cm.resolve(
            meta={"min_confidence": 0.55},
            reflection={"action": "INCREASE_ALLOCATION"},
            signal_confidence=0.54,
        )
        assert ok is True
        assert min_conf == 0.53

    def test_threshold_floor(self):
        min_conf, _, _ = self.cm.resolve(
            meta={"min_confidence": 0.52},
            reflection={"action": "INCREASE_ALLOCATION"},
            signal_confidence=0.80,
        )
        assert min_conf >= 0.52
