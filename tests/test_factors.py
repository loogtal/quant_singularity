"""Tests for factors/alpha_factors.py."""

import numpy as np

from factors.alpha_factors import AlphaFactors


class TestAlphaFactors:
    def setup_method(self):
        self.af = AlphaFactors()

    def test_momentum_positive(self):
        closes = np.linspace(100, 110, 20)
        mom = self.af.momentum_factor(closes)
        assert mom > 0

    def test_momentum_negative(self):
        closes = np.linspace(110, 100, 20)
        mom = self.af.momentum_factor(closes)
        assert mom < 0

    def test_momentum_flat(self):
        closes = np.full(20, 100.0)
        mom = self.af.momentum_factor(closes)
        assert mom == 0.0

    def test_volatility_factor_positive(self):
        closes = np.array([100 + (i % 2) * 5 for i in range(50)], dtype=float)
        vol = self.af.volatility_factor(closes)
        assert vol > 0

    def test_volatility_factor_zero_for_constant(self):
        closes = np.full(50, 100.0)
        vol = self.af.volatility_factor(closes)
        assert vol == 0.0

    def test_trend_strength_uptrend(self):
        closes = np.linspace(80, 120, 100)
        trend = self.af.trend_strength(closes)
        assert trend > 0

    def test_trend_strength_downtrend(self):
        closes = np.linspace(120, 80, 100)
        trend = self.af.trend_strength(closes)
        assert trend < 0
