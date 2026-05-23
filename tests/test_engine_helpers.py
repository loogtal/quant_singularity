"""Tests for helper functions in core/engine.py."""

from core.engine import stop_or_tp_hit, build_risk_levels, update_trailing_stop


class TestStopOrTpHit:
    def test_long_stop_loss(self):
        pos = {"side": "LONG", "stop_loss": 100, "take_profit": 120}
        assert stop_or_tp_hit(pos, 99) == "STOP LOSS"

    def test_long_take_profit(self):
        pos = {"side": "LONG", "stop_loss": 100, "take_profit": 120}
        assert stop_or_tp_hit(pos, 121) == "TAKE PROFIT"

    def test_long_between(self):
        pos = {"side": "LONG", "stop_loss": 100, "take_profit": 120}
        assert stop_or_tp_hit(pos, 110) is None

    def test_short_stop_loss(self):
        pos = {"side": "SHORT", "stop_loss": 120, "take_profit": 100}
        assert stop_or_tp_hit(pos, 121) == "STOP LOSS"

    def test_short_take_profit(self):
        pos = {"side": "SHORT", "stop_loss": 120, "take_profit": 100}
        assert stop_or_tp_hit(pos, 99) == "TAKE PROFIT"

    def test_short_between(self):
        pos = {"side": "SHORT", "stop_loss": 120, "take_profit": 100}
        assert stop_or_tp_hit(pos, 110) is None

    def test_missing_levels(self):
        pos = {"side": "LONG", "stop_loss": None, "take_profit": None}
        assert stop_or_tp_hit(pos, 100) is None

    def test_no_keys(self):
        pos = {"side": "LONG"}
        assert stop_or_tp_hit(pos, 100) is None


class TestBuildRiskLevels:
    def test_long_levels(self):
        sl, tp = build_risk_levels("LONG", 100, 5)
        assert sl < 100
        assert tp > 100
        assert sl == 100 - 5 * 2.0
        assert tp == 100 + 5 * 3.5

    def test_short_levels(self):
        sl, tp = build_risk_levels("SHORT", 100, 5)
        assert sl > 100
        assert tp < 100
        assert sl == 100 + 5 * 2.0
        assert tp == 100 - 5 * 3.5


class TestUpdateTrailingStop:
    def test_long_trailing_moves_up(self):
        pos = {
            "side": "LONG",
            "current_price": 110,
            "stop_loss": 90,
            "atr": 5,
        }
        update_trailing_stop(pos)
        expected = round(110 - 5 * 1.5, 6)
        assert pos["stop_loss"] == expected

    def test_long_trailing_no_move_down(self):
        pos = {
            "side": "LONG",
            "current_price": 95,
            "stop_loss": 90,
            "atr": 5,
        }
        update_trailing_stop(pos)
        assert pos["stop_loss"] == 90

    def test_short_trailing_moves_down(self):
        pos = {
            "side": "SHORT",
            "current_price": 90,
            "stop_loss": 110,
            "atr": 5,
        }
        update_trailing_stop(pos)
        expected = round(90 + 5 * 1.5, 6)
        assert pos["stop_loss"] == expected

    def test_short_trailing_no_move_up(self):
        pos = {
            "side": "SHORT",
            "current_price": 108,
            "stop_loss": 110,
            "atr": 5,
        }
        update_trailing_stop(pos)
        assert pos["stop_loss"] == 110

    def test_zero_atr_noop(self):
        pos = {
            "side": "LONG",
            "current_price": 110,
            "stop_loss": 90,
            "atr": 0,
        }
        update_trailing_stop(pos)
        assert pos["stop_loss"] == 90
