"""Tests for volume filter."""

import numpy as np
import pytest

from data.volume_filter import VolumeFilter


@pytest.fixture
def vf():
    return VolumeFilter()


class TestRelativeVolume:
    def test_normal_volume(self, vf):
        vols = np.array([100.0] * 25)
        assert abs(vf.relative_volume(vols) - 1.0) < 0.01

    def test_high_volume(self, vf):
        vols = np.array([100.0] * 24 + [300.0])
        assert vf.relative_volume(vols) > 2.0

    def test_low_volume(self, vf):
        vols = np.array([100.0] * 24 + [20.0])
        assert vf.relative_volume(vols) < 0.5

    def test_short_series(self, vf):
        assert vf.relative_volume(np.array([100.0])) == 1.0


class TestVolumeSpike:
    def test_spike_detected(self, vf):
        vols = np.array([100.0] * 24 + [200.0])
        assert vf.has_volume_spike(vols)

    def test_no_spike(self, vf):
        vols = np.array([100.0] * 25)
        assert not vf.has_volume_spike(vols)


class TestConfirmsEntry:
    def test_allows_normal(self, vf):
        vols = np.array([100.0] * 25)
        assert vf.confirms_entry(vols) is True

    def test_blocks_low(self, vf):
        vols = np.array([100.0] * 24 + [10.0])
        assert vf.confirms_entry(vols) is False


class TestConfidenceAdjust:
    def test_high_volume_boosts(self, vf):
        vols = np.array([100.0] * 24 + [250.0])
        assert vf.confidence_adjust(vols, 0.7) > 0.7

    def test_low_volume_penalizes(self, vf):
        vols = np.array([100.0] * 24 + [30.0])
        assert vf.confidence_adjust(vols, 0.7) < 0.7
