"""Tests for kill switch."""

import pytest

from risk.kill_switch import KillSwitch


@pytest.fixture
def ks():
    return KillSwitch()


class _FakePortfolio:
    def __init__(self, dd=0.0):
        self.drawdown = dd


class TestCheck:
    def test_no_halt(self, ks):
        port = _FakePortfolio(dd=0.01)
        perf = {"winrate": 0.5, "daily_pnl_pct": 0, "total_trades": 5}
        result = ks.check(port, perf, {"volatility": 0.3}, {"status": "NEUTRAL"})
        assert result["halt"] is False

    def test_drawdown_halt(self, ks):
        port = _FakePortfolio(dd=0.15)
        perf = {"winrate": 0.5, "daily_pnl_pct": 0, "total_trades": 5}
        result = ks.check(port, perf, {"volatility": 0.3}, {"status": "NEUTRAL"})
        assert result["halt"] is True

    def test_daily_loss_halt(self, ks):
        port = _FakePortfolio(dd=0.01)
        perf = {"winrate": 0.5, "daily_pnl_pct": -0.04, "total_trades": 5}
        result = ks.check(port, perf, {"volatility": 0.3}, {"status": "NEUTRAL"})
        assert result["halt"] is True

    def test_high_vol_halt(self, ks):
        port = _FakePortfolio(dd=0.01)
        perf = {"winrate": 0.5, "daily_pnl_pct": 0, "total_trades": 5}
        result = ks.check(port, perf, {"volatility": 0.95}, {"status": "NEUTRAL"})
        assert result["halt"] is True


class TestSizeMultiplier:
    def test_normal(self, ks):
        meta = {"mode": "neutral"}
        ref = {"status": "NEUTRAL"}
        m = ks.size_multiplier(meta, ref)
        assert 0 < m <= 1.5

    def test_defensive(self, ks):
        meta = {"mode": "defensive"}
        ref = {"status": "WEAK", "action": "REDUCE_ALLOCATION"}
        m = ks.size_multiplier(meta, ref)
        assert m < 1.0

    def test_strong(self, ks):
        meta = {"mode": "aggressive"}
        ref = {"status": "STRONG", "action": "INCREASE_ALLOCATION"}
        m = ks.size_multiplier(meta, ref)
        assert m >= 1.0
