"""Tests for portfolio engine."""

import pytest

from portfolio.portfolio_engine import PortfolioEngine


@pytest.fixture
def pe():
    return PortfolioEngine(initial_cash=10000)


class TestAddPosition:
    def test_add(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "position_value": 5000})
        assert len(pe.positions) == 1
        assert pe.cash < 10000

    def test_duplicate_check(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "position_value": 5000})
        assert pe.has_open_position("BTC/USDT:USDT")
        assert not pe.has_open_position("ETH/USDT:USDT")


class TestClosePosition:
    def test_close_profit(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "position_value": 5000})
        pe.close_position("BTC/USDT:USDT", 100)
        assert len(pe.positions) == 0
        assert pe.realized_pnl == 100

    def test_close_loss(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "position_value": 5000})
        pe.close_position("BTC/USDT:USDT", -200)
        assert pe.realized_pnl == -200


class TestUpdatePrice:
    def test_update(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "side": "LONG", "position_value": 5000})
        pe.update_position_price("BTC/USDT:USDT", 51000)
        pos = pe.positions[0]
        assert pos["current_price"] == 51000


class TestDrawdown:
    def test_no_drawdown_initial(self, pe):
        assert pe.drawdown == 0

    def test_drawdown_after_loss(self, pe):
        pe.add_position({"symbol": "BTC/USDT:USDT", "size": 0.1, "entry_price": 50000, "position_value": 5000})
        pe.close_position("BTC/USDT:USDT", -1000)
        assert pe.drawdown > 0


class TestRestore:
    def test_restore(self, pe):
        pe.restore(cash=8000, positions=[{"symbol": "X", "size": 1, "position_value": 500, "entry_price": 500}], realized_pnl=50)
        assert pe.cash == 8000
        assert len(pe.positions) == 1
        assert pe.realized_pnl == 50


class TestStatus:
    def test_status_keys(self, pe):
        s = pe.status()
        assert "cash" in s
        assert "equity" in s
