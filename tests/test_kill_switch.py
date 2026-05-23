"""Tests for risk/kill_switch.py."""

from unittest.mock import MagicMock

from risk.kill_switch import KillSwitch


class TestKillSwitch:
    def setup_method(self):
        self.ks = KillSwitch()

    def _make_portfolio(self, drawdown=0.0):
        p = MagicMock()
        p.drawdown = drawdown
        return p

    def test_normal_conditions(self):
        result = self.ks.check(
            portfolio=self._make_portfolio(0.0),
            perf={"daily_pnl_pct": 0.01},
            market_state={"volatility": 0.3},
        )
        assert result["halt"] is False

    def test_max_drawdown_halt(self):
        result = self.ks.check(
            portfolio=self._make_portfolio(0.20),
            perf={"daily_pnl_pct": 0.0},
            market_state={"volatility": 0.3},
        )
        assert result["halt"] is True
        assert "drawdown" in result["reason"].lower()

    def test_daily_loss_halt(self):
        result = self.ks.check(
            portfolio=self._make_portfolio(0.0),
            perf={"daily_pnl_pct": -0.05},
            market_state={"volatility": 0.3},
        )
        assert result["halt"] is True
        assert "loss" in result["reason"].lower()

    def test_high_volatility_halt(self):
        result = self.ks.check(
            portfolio=self._make_portfolio(0.0),
            perf={"daily_pnl_pct": 0.0},
            market_state={"volatility": 0.90},
        )
        assert result["halt"] is True
        assert "volatility" in result["reason"].lower()

    def test_size_multiplier_reduce(self):
        mult = self.ks.size_multiplier(
            meta={"size_multiplier": 1.0},
            reflection={"action": "REDUCE_ALLOCATION"},
        )
        assert mult == 0.5

    def test_size_multiplier_increase(self):
        mult = self.ks.size_multiplier(
            meta={"size_multiplier": 1.0},
            reflection={"action": "INCREASE_ALLOCATION"},
        )
        assert mult == 1.15

    def test_size_multiplier_increase_cap(self):
        mult = self.ks.size_multiplier(
            meta={"size_multiplier": 1.2},
            reflection={"action": "INCREASE_ALLOCATION"},
        )
        assert mult <= 1.25

    def test_size_multiplier_no_action(self):
        mult = self.ks.size_multiplier(
            meta={"size_multiplier": 0.8},
            reflection={"action": "MONITOR"},
        )
        assert mult == 0.8
