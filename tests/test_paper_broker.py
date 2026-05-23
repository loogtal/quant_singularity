"""Tests for paper broker."""

import pytest

from execution.paper_broker import PaperBroker


class _FakeFeed:
    def get_price(self, symbol):
        return 50000.0


@pytest.fixture
def broker():
    return PaperBroker(price_feed=_FakeFeed())


class TestGetPrice:
    def test_returns_float(self, broker):
        assert broker.get_price("BTC/USDT:USDT") == 50000.0


class TestExecuteOrder:
    def test_buy(self, broker):
        result = broker.execute_order("BTC/USDT:USDT", "LONG", 0.1)
        assert result is not None
        assert result["side"] == "LONG"
        assert result["price"] > 0

    def test_sell(self, broker):
        result = broker.execute_order("BTC/USDT:USDT", "SHORT", 0.1)
        assert result is not None
        assert result["side"] == "SHORT"


class TestClosePosition:
    def test_long_profit(self, broker):
        pos = {"symbol": "BTC/USDT:USDT", "side": "LONG", "size": 0.1, "entry_price": 49000}
        pnl = broker.close_position(pos)
        assert pnl > 0

    def test_short_profit(self, broker):
        pos = {"symbol": "BTC/USDT:USDT", "side": "SHORT", "size": 0.1, "entry_price": 51000}
        pnl = broker.close_position(pos)
        assert pnl > 0
