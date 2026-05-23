"""Tests for enhanced ML features."""

import numpy as np
import pytest

from models.predictor import MLPredictor


@pytest.fixture
def ml():
    return MLPredictor()


class TestExtractFeatures:
    def test_feature_count(self, ml):
        closes = np.linspace(100, 120, 100)
        volumes = np.ones(100) * 1000
        feats = ml.extract_features(closes, volumes)
        assert len(feats) == 9

    def test_features_bounded(self, ml):
        closes = np.linspace(100, 120, 100)
        volumes = np.ones(100) * 1000
        feats = ml.extract_features(closes, volumes)
        for f in feats:
            assert 0 <= f <= 1


class TestRSIFeature:
    def test_rising_prices(self, ml):
        closes = np.linspace(100, 130, 30)
        rsi = ml._rsi_feature(closes)
        assert rsi > 0.5

    def test_falling_prices(self, ml):
        closes = np.linspace(130, 100, 30)
        rsi = ml._rsi_feature(closes)
        assert rsi < 0.5

    def test_short_series(self, ml):
        assert ml._rsi_feature(np.array([100.0])) == 0.5


class TestMACDFeature:
    def test_uptrend(self, ml):
        closes = np.linspace(100, 130, 50)
        macd = ml._macd_feature(closes)
        assert macd > 0.5

    def test_short_series(self, ml):
        assert ml._macd_feature(np.array([100.0] * 10)) == 0.5


class TestBBPosition:
    def test_at_upper(self, ml):
        closes = np.array([100.0] * 19 + [120.0])
        pos = ml._bb_position(closes)
        assert pos > 0.8

    def test_at_lower(self, ml):
        closes = np.array([100.0] * 19 + [80.0])
        pos = ml._bb_position(closes)
        assert pos < 0.2


class TestOBVTrend:
    def test_rising_volume(self, ml):
        closes = np.linspace(100, 120, 30)
        volumes = np.ones(30) * 1000
        obv = ml._obv_trend(closes, volumes)
        assert obv > 0.5

    def test_no_volumes(self, ml):
        closes = np.linspace(100, 120, 30)
        assert ml._obv_trend(closes, None) == 0.5


class TestPredict:
    def test_output_keys(self, ml):
        closes = np.linspace(100, 120, 50)
        result = ml.predict(closes)
        assert "direction" in result
        assert "confidence" in result
        assert "prob_up" in result

    def test_short_series_hold(self, ml):
        result = ml.predict(np.array([100.0] * 10))
        assert result["direction"] == "HOLD"
