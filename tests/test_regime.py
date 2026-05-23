"""Tests for meta/regime_classifier.py."""

import numpy as np

from meta.regime_classifier import RegimeClassifier


class TestRegimeClassifier:
    def setup_method(self):
        self.rc = RegimeClassifier()

    def test_insufficient_data_returns_sideways(self):
        closes = np.linspace(100, 110, 50)
        assert self.rc.classify(closes) == "sideways"

    def test_strong_uptrend_returns_bull(self):
        closes = np.linspace(80, 120, 200)
        result = self.rc.classify(closes)
        assert result == "bull"

    def test_strong_downtrend_returns_bear(self):
        closes = np.linspace(120, 80, 200)
        result = self.rc.classify(closes)
        assert result == "bear"

    def test_flat_returns_sideways(self):
        closes = np.full(200, 100.0)
        result = self.rc.classify(closes)
        assert result == "sideways"

    def test_risk_on_bull_low_vol(self):
        assert self.rc.risk_on("bull", 0.3) is True

    def test_risk_off_bear(self):
        assert self.rc.risk_on("bear", 0.3) is False

    def test_risk_off_high_vol(self):
        assert self.rc.risk_on("bull", 0.8) is False

    def test_risk_on_sideways_low_vol(self):
        assert self.rc.risk_on("sideways", 0.5) is True
