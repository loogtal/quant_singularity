"""Tests for confidence manager."""

import pytest

from meta.confidence_manager import ConfidenceManager


@pytest.fixture
def cm():
    return ConfidenceManager()


class TestResolve:
    def test_high_confidence_passes(self, cm):
        meta = {"min_confidence": 0.55}
        reflection = {"status": "NEUTRAL"}
        _, ok, _ = cm.resolve(meta, reflection, 0.7)
        assert ok is True

    def test_low_confidence_blocked(self, cm):
        meta = {"min_confidence": 0.65}
        reflection = {"status": "NEUTRAL"}
        _, ok, reason = cm.resolve(meta, reflection, 0.4)
        assert ok is False
        assert "CONF" in reason.upper()

    def test_increase_allocation_lowers_threshold(self, cm):
        meta = {"min_confidence": 0.65}
        reflection = {"status": "STRONG", "action": "INCREASE_ALLOCATION"}
        _, ok, _ = cm.resolve(meta, reflection, 0.64)
        assert ok is True

    def test_reduce_does_not_block(self, cm):
        meta = {"min_confidence": 0.55}
        reflection = {"status": "WEAK", "action": "REDUCE_ALLOCATION"}
        _, ok, _ = cm.resolve(meta, reflection, 0.58)
        assert ok is True
