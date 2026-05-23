"""Tests for engine helper functions."""


from core.engine import (
    build_risk_levels,
    stop_or_tp_hit,
    update_trailing_stop,
    check_partial_tp,
    should_time_exit,
)


class TestBuildRiskLevels:
    def test_long_basic(self):
        sl, tp, ptp = build_risk_levels("LONG", 100, 2.0)
        assert sl < 100
        assert tp > 100
        assert ptp > 100
        assert sl < ptp < tp

    def test_short_basic(self):
        sl, tp, ptp = build_risk_levels("SHORT", 100, 2.0)
        assert sl > 100
        assert tp < 100
        assert ptp < 100
        assert sl > ptp > tp

    def test_bull_long_wider_tp(self):
        _, tp_bull, _ = build_risk_levels("LONG", 100, 2.0, regime="bull")
        _, tp_side, _ = build_risk_levels("LONG", 100, 2.0, regime="sideways")
        assert tp_bull > tp_side

    def test_sideways_tighter_than_bull(self):
        _, tp_side, _ = build_risk_levels("LONG", 100, 2.0, regime="sideways")
        _, tp_bull, _ = build_risk_levels("LONG", 100, 2.0, regime="bull")
        assert (tp_side - 100) < (tp_bull - 100)

    def test_trend_strength_extends_tp(self):
        _, tp_low, _ = build_risk_levels("LONG", 100, 2.0, trend_strength=0.3)
        _, tp_high, _ = build_risk_levels("LONG", 100, 2.0, trend_strength=0.8)
        assert tp_high > tp_low

    def test_returns_three_values(self):
        result = build_risk_levels("LONG", 100, 2.0)
        assert len(result) == 3


class TestStopOrTpHit:
    def test_long_stop_hit(self):
        pos = {"side": "LONG", "entry_price": 100, "stop_loss": 95, "take_profit": 110}
        assert stop_or_tp_hit(pos, 94) == "STOP LOSS"

    def test_long_tp_hit(self):
        pos = {"side": "LONG", "entry_price": 100, "stop_loss": 95, "take_profit": 110}
        assert stop_or_tp_hit(pos, 111) == "TAKE PROFIT"

    def test_short_stop_hit(self):
        pos = {"side": "SHORT", "entry_price": 100, "stop_loss": 105, "take_profit": 90}
        assert stop_or_tp_hit(pos, 106) == "STOP LOSS"

    def test_short_tp_hit(self):
        pos = {"side": "SHORT", "entry_price": 100, "stop_loss": 105, "take_profit": 90}
        assert stop_or_tp_hit(pos, 89) == "TAKE PROFIT"

    def test_no_hit(self):
        pos = {"side": "LONG", "entry_price": 100, "stop_loss": 95, "take_profit": 110}
        assert stop_or_tp_hit(pos, 100) is None


class TestTrailingStop:
    def test_long_trail_up(self):
        pos = {"side": "LONG", "entry_price": 100, "stop_loss": 95, "atr": 2, "current_price": 108}
        update_trailing_stop(pos)
        assert pos["stop_loss"] > 95

    def test_short_trail_down(self):
        pos = {"side": "SHORT", "entry_price": 100, "stop_loss": 105, "atr": 2, "current_price": 92}
        update_trailing_stop(pos)
        assert pos["stop_loss"] < 105


class TestPartialTP:
    def test_partial_tp_hit_long(self):
        pos = {"side": "LONG", "partial_tp": 105, "partial_closed": False}
        assert check_partial_tp(pos, 106) is True

    def test_partial_tp_not_hit(self):
        pos = {"side": "LONG", "partial_tp": 105, "partial_closed": False}
        assert check_partial_tp(pos, 103) is False

    def test_already_closed(self):
        pos = {"side": "LONG", "partial_tp": 105, "partial_closed": True}
        assert check_partial_tp(pos, 106) is False

    def test_no_partial_tp_key(self):
        pos = {"side": "LONG"}
        assert check_partial_tp(pos, 110) is False

    def test_short_partial_tp(self):
        pos = {"side": "SHORT", "partial_tp": 95, "partial_closed": False}
        assert check_partial_tp(pos, 94) is True


class TestTimeExit:
    def test_should_exit(self):
        assert should_time_exit({"opened_at": 1000}, 1000 + 3600 * 5) is True

    def test_should_not_exit(self):
        assert should_time_exit({"opened_at": 1000}, 1000 + 3600) is False

    def test_no_opened_at(self):
        assert should_time_exit({}, 1000) is False
