"""Tests for the enhanced regime classifier."""

import numpy as np
import pytest

from meta.regime_classifier import RegimeClassifier


@pytest.fixture
def rc():
    return RegimeClassifier()


def _rising(n=200, start=100.0, step=0.5):
    return np.array([start + i * step for i in range(n)])


def _falling(n=200, start=200.0, step=0.5):
    return np.array([start - i * step for i in range(n)])


def _flat(n=200, base=100.0):
    np.random.seed(42)
    return base + np.random.randn(n) * 0.3


class TestClassify:
    def test_bull_regime(self, rc):
        closes = _rising()
        assert rc.classify(closes) == "bull"

    def test_bear_regime(self, rc):
        closes = _falling()
        assert rc.classify(closes) == "bear"

    def test_sideways_regime(self, rc):
        closes = _flat()
        assert rc.classify(closes) == "sideways"

    def test_short_series_sideways(self, rc):
        assert rc.classify(np.array([100.0] * 50)) == "sideways"

    def test_bull_with_hlc(self, rc):
        closes = _rising()
        highs = closes + 1
        lows = closes - 1
        assert rc.classify(closes, highs, lows) == "bull"

    def test_bear_with_hlc(self, rc):
        closes = _falling()
        highs = closes + 1
        lows = closes - 1
        assert rc.classify(closes, highs, lows) == "bear"


class TestRiskOn:
    def test_bull_low_vol(self, rc):
        assert rc.risk_on("bull", 0.3) is True

    def test_bear(self, rc):
        assert rc.risk_on("bear", 0.3) is False

    def test_high_vol(self, rc):
        assert rc.risk_on("bull", 0.8) is False


class TestADX:
    def test_adx_trending(self, rc):
        closes = _rising()
        highs = closes + 2
        lows = closes - 1
        adx = rc._adx(highs, lows, closes)
        assert adx > 0

    def test_adx_short_series(self, rc):
        closes = np.array([1.0, 2.0])
        assert rc._adx(closes, closes, closes) == 0.0


class TestBBWidth:
    def test_bb_width_positive(self, rc):
        closes = np.arange(100.0, 120.0)
        assert rc._bb_width(closes) > 0

    def test_bb_width_short(self, rc):
        assert rc._bb_width(np.array([1.0])) == 0.0


class TestTrendStrength:
    def test_strong_trend(self, rc):
        closes = _rising()
        highs = closes + 2
        lows = closes - 1
        ts = rc.trend_strength(closes, highs, lows)
        assert 0 <= ts <= 1
        assert ts > 0

    def test_flat_trend(self, rc):
        closes = _flat()
        ts = rc.trend_strength(closes)
        assert 0 <= ts <= 1
