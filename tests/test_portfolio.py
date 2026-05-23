"""Tests for portfolio/portfolio_engine.py."""

from portfolio.portfolio_engine import PortfolioEngine


class TestPortfolioEngine:
    def _make_engine(self, capital=10000):
        return PortfolioEngine(initial_cash=capital)

    def test_initial_state(self):
        pe = self._make_engine()
        assert pe.cash == 10000
        assert pe.equity == 10000
        assert pe.positions == []
        assert pe.realized_pnl == 0
        assert pe.drawdown == 0

    def test_add_position_reduces_cash(self):
        pe = self._make_engine()
        pe.add_position({
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
            "position_value": 500,
        })
        assert pe.cash == 9500
        assert len(pe.positions) == 1

    def test_has_open_position(self):
        pe = self._make_engine()
        assert pe.has_open_position("BTC/USDT:USDT") is False
        pe.add_position({
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
            "position_value": 500,
        })
        assert pe.has_open_position("BTC/USDT:USDT") is True
        assert pe.has_open_position("ETH/USDT:USDT") is False

    def test_close_position_profit(self):
        pe = self._make_engine(capital=10000)
        pe.add_position({
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
            "position_value": 500,
        })
        pe.close_position("BTC/USDT:USDT", pnl=50)
        assert pe.realized_pnl == 50
        assert pe.cash == 10050
        assert len(pe.positions) == 0

    def test_close_position_loss(self):
        pe = self._make_engine(capital=10000)
        pe.add_position({
            "symbol": "ETH/USDT:USDT",
            "side": "SHORT",
            "entry_price": 3000,
            "size": 0.1,
            "position_value": 300,
        })
        pe.close_position("ETH/USDT:USDT", pnl=-30)
        assert pe.realized_pnl == -30
        assert pe.cash == 9970

    def test_close_nonexistent_position(self):
        pe = self._make_engine()
        pe.close_position("FAKE/USDT:USDT", pnl=100)
        assert pe.realized_pnl == 0

    def test_update_position_price_long(self):
        pe = self._make_engine()
        pe.add_position({
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
            "position_value": 500,
        })
        pe.update_position_price("BTC/USDT:USDT", 51000)
        pos = pe.get_position("BTC/USDT:USDT")
        assert pos["unrealized_pnl"] == (51000 - 50000) * 0.01

    def test_update_position_price_short(self):
        pe = self._make_engine()
        pe.add_position({
            "symbol": "ETH/USDT:USDT",
            "side": "SHORT",
            "entry_price": 3000,
            "size": 0.1,
            "position_value": 300,
        })
        pe.update_position_price("ETH/USDT:USDT", 2900)
        pos = pe.get_position("ETH/USDT:USDT")
        assert pos["unrealized_pnl"] == (3000 - 2900) * 0.1

    def test_drawdown_calculation(self):
        pe = self._make_engine(capital=10000)
        pe.add_position({
            "symbol": "BTC/USDT:USDT",
            "side": "LONG",
            "entry_price": 50000,
            "size": 0.01,
            "position_value": 500,
        })
        pe.update_position_price("BTC/USDT:USDT", 40000)
        assert pe.drawdown > 0

    def test_status_output(self):
        pe = self._make_engine()
        status = pe.status()
        assert "cash" in status
        assert "equity" in status
        assert "positions" in status
        assert "realized_pnl" in status
        assert "unrealized_pnl" in status
        assert "drawdown" in status

    def test_restore(self):
        pe = self._make_engine()
        positions = [{"symbol": "BTC/USDT:USDT", "side": "LONG",
                       "entry_price": 50000, "size": 0.01,
                       "position_value": 500, "unrealized_pnl": 10}]
        pe.restore(cash=9500, positions=positions, realized_pnl=100)
        assert pe.cash == 9500
        assert len(pe.positions) == 1
        assert pe.realized_pnl == 100
