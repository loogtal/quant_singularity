"""
Core-logic regression tests.

Covers the money-critical logic and locks in the bugs fixed this cycle so they
cannot silently return. Runs with plain `python tests/test_core_logic.py`
(no pytest needed) and is also pytest-compatible (test_* functions).

Only modules that import without network are tested (portfolio, kelly, risk,
daily-profit, compound, symbol-edge, evolver cost model).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── portfolio engine ────────────────────────────────────────────────────────

def test_portfolio_leverage_zero_guard():
    """add_position must not divide by zero when leverage is 0/missing."""
    from portfolio.portfolio_engine import PortfolioEngine
    pe = PortfolioEngine(initial_cash=1000)
    pe.add_position({"symbol": "X", "side": "LONG", "position_value": 100,
                     "leverage": 0, "entry_price": 10, "size": 10})
    assert pe.positions[0]["margin"] == 100  # leverage 0 -> treated as 1.0


def test_portfolio_drawdown_no_div_zero():
    from portfolio.portfolio_engine import PortfolioEngine
    pe = PortfolioEngine(initial_cash=1000)
    pe._peak_equity = 0.0          # pathological
    pe.equity = 0.0
    pe.update_equity()             # must not raise ZeroDivisionError
    assert pe.drawdown >= 0.0


def test_portfolio_cash_accounting():
    from portfolio.portfolio_engine import PortfolioEngine
    pe = PortfolioEngine(initial_cash=1000)
    pe.add_position({"symbol": "X", "side": "LONG", "position_value": 200,
                     "leverage": 1, "entry_price": 10, "size": 20})
    assert abs(pe.cash - 800) < 1e-6
    pe.close_position("X", pnl=50)
    assert abs(pe.cash - 1050) < 1e-6     # margin 200 returned + 50 pnl


# ── kelly ──────────────────────────────────────────────────────────────────

def test_kelly_fraction_bounds():
    from risk.kelly import KellyCriterion
    k = KellyCriterion()
    for wr in (0.0, 0.3, 0.5, 0.9, 1.0):
        f = k.half_kelly_fraction(wr, 0.02, 0.01)
        assert k.MIN_FRACTION <= f <= k.MAX_FRACTION


def test_kelly_default_before_min_trades():
    from risk.kelly import KellyCriterion
    k = KellyCriterion()
    v = k.position_value(0.6, 0.02, 0.01, equity=1000,
                         trades_so_far=0, default_fraction=0.08)
    assert abs(v - 80) < 1e-6     # uses default fraction until MIN_TRADES


# ── symbol edge ──────────────────────────────────────────────────────────────

def test_symbol_edge_mult_bounds_and_neutral():
    from risk.symbol_edge import SymbolEdgeTracker
    t = SymbolEdgeTracker()
    assert t.edge_mult("NOPE/USDT") == 1.0          # no data -> neutral
    for won, pnl in [(True, 0.01)] * 6 + [(False, -0.005)] * 4:
        t.record("BTC/USDT", won, pnl)
    assert 0.40 <= t.edge_mult("BTC/USDT") <= 1.40   # always clamped


# ── daily profit engine: persistence (bug fixed this cycle) ──────────────────

def test_daily_profit_persistence_same_and_stale_day():
    import meta.daily_profit_engine as dpe
    with tempfile.TemporaryDirectory() as d:
        dpe._STATE_FILE = Path(d) / "dp.json"
        e = dpe.DailyProfitEngine(active_capital=300)
        e.record_pnl(50.0)
        same = dpe.DailyProfitEngine(active_capital=300)
        assert abs(same._pnl - 50.0) < 1e-6          # same-day restore
        # stale day -> reset
        import json
        data = json.loads(dpe._STATE_FILE.read_text())
        data["day"] = dpe._utc_today() - 1
        dpe._STATE_FILE.write_text(json.dumps(data))
        fresh = dpe.DailyProfitEngine(active_capital=300)
        assert fresh._pnl == 0.0


def test_daily_profit_locked_phase():
    import meta.daily_profit_engine as dpe
    with tempfile.TemporaryDirectory() as d:
        dpe._STATE_FILE = Path(d) / "dp.json"
        e = dpe.DailyProfitEngine(active_capital=300)
        e.record_pnl(e._target + 1)
        assert e.get_phase() == "LOCKED"
        assert e.should_open_new() is False


# ── compound manager: no double-compound across restart (bug fixed) ──────────

def test_compound_no_double_compound_on_restart():
    import meta.compound_manager as cm
    import meta.daily_profit_engine as dpe
    from portfolio.portfolio_engine import PortfolioEngine
    with tempfile.TemporaryDirectory() as d:
        cm._COMPOUND_FILE = Path(d) / "compound.json"
        dpe._STATE_FILE = Path(d) / "dp.json"
        passive, active = PortfolioEngine(700), PortfolioEngine(300)
        dp = dpe.DailyProfitEngine(active_capital=300)
        dp.record_pnl(dp._target + 5)            # hit target -> LOCKED
        mgr = cm.CompoundManager(passive, active)
        first = mgr.daily_check(dp)
        assert first > 0                          # compounds once
        # simulate restart: new manager loads persisted today_ordinal
        mgr2 = cm.CompoundManager(passive, active)
        second = mgr2.daily_check(dp)
        assert second == 0.0                      # must NOT double-compound same day


# ── evolver realistic cost (bug fixed this cycle) ────────────────────────────

def test_evolver_cost_is_charged():
    from self_evolve.strategy_evolver import _trade_cost
    assert _trade_cost(1000, 0) > 0               # fee+slippage always charged
    assert _trade_cost(1000, 10) > _trade_cost(1000, 0)  # funding grows with hold


# ── dual risk: daily-loss gate blocks active ─────────────────────────────────

def test_dual_risk_daily_loss_gate():
    from risk.dual_risk import DualRiskManager
    rm = DualRiskManager()
    rm.active_daily_loss_cap = 10.0
    rm.daily_loss_active = 11.0                    # already over the cap
    sig = {"symbol": "X", "side": "LONG", "intraday": True}

    class _P:
        cash = 1000; equity = 1000
    d = rm.allow_active(_P(), sig, {})
    assert d["allow_trade"] is False


# ── runner ───────────────────────────────────────────────────────────────────

def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({len(tests)} total)")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
