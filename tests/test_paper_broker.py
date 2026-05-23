"""Tests for execution/paper_broker.py."""

from unittest.mock import MagicMock

from execution.paper_broker import PaperBroker


class TestPaperBroker:
    def _make_broker(self, price=50000):
        feed = MagicMock()
        feed.get_price.return_value = price
        return PaperBroker(price_feed=feed)

    def test_get_price(self):
        broker = self._make_broker(price=50000)
        assert broker.get_price("BTC/USDT:USDT") == 50000

    def test_execute_order_returns_fill(self):
        broker = self._make_broker(price=50000)
        result = broker.execute_order("BTC/USDT:USDT", "LONG", 0.01)
        assert result["symbol"] == "BTC/USDT:USDT"
        assert result["side"] == "LONG"
        assert result["size"] == 0.01
        assert result["status"] == "FILLED"
        assert "price" in result
        assert "timestamp" in result
        assert abs(result["price"] - 50000) < 100

    def test_close_position_long_profit(self):
        broker = self._make_broker(price=51000)
        pos = {
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
        }
        pnl = broker.close_position(pos)
        assert pnl == (51000 - 50000) * 0.01

    def test_close_position_short_profit(self):
        broker = self._make_broker(price=49000)
        pos = {
            "symbol": "BTC/USDT:USDT",
            "side": "SHORT",
            "entry_price": 50000,
            "size": 0.01,
        }
        pnl = broker.close_position(pos)
        assert pnl == (50000 - 49000) * 0.01

    def test_close_position_long_loss(self):
        broker = self._make_broker(price=49000)
        pos = {
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
        }
        pnl = broker.close_position(pos)
        assert pnl < 0
