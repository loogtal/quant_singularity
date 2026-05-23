"""Tests for alpha factors."""

import numpy as np
import pytest

from factors.alpha_factors import AlphaFactors


@pytest.fixture
def af():
    return AlphaFactors()


def _series(n=100, start=100.0, step=0.0):
    return np.array([start + i * step for i in range(n)])


class TestMomentum:
    def test_positive(self, af):
        closes = _series(step=0.5)
        assert af.momentum_factor(closes) > 0

    def test_negative(self, af):
        closes = _series(step=-0.5)
        assert af.momentum_factor(closes) < 0

    def test_flat(self, af):
        closes = _series()
        assert abs(af.momentum_factor(closes)) < 0.01


class TestVolatility:
    def test_returns_positive(self, af):
        closes = _series(step=0.5)
        assert af.volatility_factor(closes) >= 0


class TestTrendStrength:
    def test_strong_uptrend(self, af):
        closes = _series(step=1.0)
        assert af.trend_strength(closes) > 0

    def test_strong_downtrend(self, af):
        closes = _series(step=-1.0)
        assert af.trend_strength(closes) < 0

    def test_flat_trend(self, af):
        closes = _series()
        assert abs(af.trend_strength(closes)) < 0.01
