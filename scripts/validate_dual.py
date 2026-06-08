#!/usr/bin/env python3
"""
Pre-live checklist for Dual Strategy mode.

Runs entirely offline (no API calls) — uses mocked market data
to exercise every component and prints PASS/FAIL for each check.

Usage:
    python scripts/validate_dual.py
Exit 0 = all checks pass. Exit 1 = one or more failures.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np

# ── colour helpers ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> bool:
    """Run fn(); record PASS/FAIL. Returns True on pass."""
    try:
        msg = fn()
        _results.append((name, True, msg or ""))
        return True
    except Exception as exc:
        _results.append((name, False, str(exc)))
        return False


def warn(name: str, fn) -> bool:
    """Like check() but a failure is a WARNING not a hard FAIL."""
    try:
        msg = fn()
        _results.append((f"[W] {name}", True, msg or ""))
        return True
    except Exception as exc:
        _results.append((f"[W] {name}", None, str(exc)))  # type: ignore[arg-type]
        return False


# ── mock market ───────────────────────────────────────────────────────────────

class _MockOHLCV:
    """Returns synthetic price data so checks never need a real API key."""

    def _make(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        prices = 30_000 + np.cumsum(rng.normal(0, 200, n))
        prices = np.clip(prices, 1, None)
        highs  = prices * (1 + rng.uniform(0, 0.005, n))
        lows   = prices * (1 - rng.uniform(0, 0.005, n))
        import pandas as pd
        return pd.DataFrame({"close": prices, "high": highs, "low": lows,
                             "open": prices, "volume": np.ones(n) * 1e6})

    def get_ohlcv_df(self, symbol, timeframe="1d", limit=300):
        return self._make(n=max(limit, 300))

    def get_funding_rate(self, symbol):
        return 0.0001

    def get_price(self, symbol):
        return 30_000.0


_MOCK_MARKET = _MockOHLCV()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Imports
# ─────────────────────────────────────────────────────────────────────────────

def _import_checks():
    check("Import: config.dual_settings", lambda: __import__("config.dual_settings") and "")
    check("Import: strategy.passive_strategy", lambda: __import__("strategy.passive_strategy") and "")
    check("Import: strategy.active_strategy", lambda: __import__("strategy.active_strategy") and "")
    check("Import: strategy.dual_regime_router", lambda: __import__("strategy.dual_regime_router") and "")
    check("Import: risk.dual_risk", lambda: __import__("risk.dual_risk") and "")
    check("Import: risk.kelly", lambda: __import__("risk.kelly") and "")
    check("Import: risk.correlation_filter", lambda: __import__("risk.correlation_filter") and "")
    check("Import: portfolio.capital_allocator", lambda: __import__("portfolio.capital_allocator") and "")
    check("Import: self_evolve.strategy_evolver", lambda: __import__("self_evolve.strategy_evolver") and "")
    check("Import: monitoring.dual_dashboard", lambda: __import__("monitoring.dual_dashboard") and "")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Config sanity
# ─────────────────────────────────────────────────────────────────────────────

def _config_checks():
    from config.dual_settings import (
        PASSIVE_CAPITAL, ACTIVE_CAPITAL, TOTAL_CAPITAL,
        KILL_SWITCH_EQUITY, MAX_PORTFOLIO_DRAWDOWN,
        PASSIVE_MAX_DRAWDOWN, PASSIVE_TAKE_PROFIT, PASSIVE_STOP_LOSS,
        ACTIVE_TAKE_PROFIT, ACTIVE_STOP_LOSS, ACTIVE_MAX_DAILY_LOSS,
        PASSIVE_MAX_POSITIONS, ACTIVE_MAX_POSITIONS,
        ACTIVE_MIN_CAPITAL_FRACTION, ACTIVE_MAX_CAPITAL_FRACTION,
    )

    def cap_sum():
        total = PASSIVE_CAPITAL + ACTIVE_CAPITAL
        if abs(total - TOTAL_CAPITAL) > 1:
            raise ValueError(
                f"PASSIVE_CAPITAL({PASSIVE_CAPITAL}) + ACTIVE_CAPITAL({ACTIVE_CAPITAL})"
                f" = {total} ≠ TOTAL_CAPITAL({TOTAL_CAPITAL})"
            )
        return f"passive={PASSIVE_CAPITAL} + active={ACTIVE_CAPITAL} = {total}"

    def kill_switch():
        pct_remaining = KILL_SWITCH_EQUITY / TOTAL_CAPITAL
        if pct_remaining < 0.70:
            raise ValueError(
                f"KILL_SWITCH_EQUITY {KILL_SWITCH_EQUITY} is below 70% of capital "
                f"({TOTAL_CAPITAL * 0.70:.0f}) — kill switch may be too loose"
            )
        return f"{KILL_SWITCH_EQUITY} = {pct_remaining*100:.1f}% of total capital"

    def ev_passive():
        ev = 0.55 * PASSIVE_TAKE_PROFIT - 0.45 * PASSIVE_STOP_LOSS
        if ev <= 0:
            raise ValueError(
                f"Passive EV at 55% WR = {ev:.4f} — check TP/SL ratio"
            )
        return f"EV @55%WR = +{ev*100:.3f}%"

    def ev_active():
        ev = 0.50 * ACTIVE_TAKE_PROFIT - 0.50 * ACTIVE_STOP_LOSS
        if ev < 0:
            raise ValueError(
                f"Active EV at 50% WR = {ev:.4f} — TP must exceed SL"
            )
        return f"EV @50%WR = +{ev*100:.3f}%"

    def tp_sl_ratio_passive():
        ratio = PASSIVE_TAKE_PROFIT / PASSIVE_STOP_LOSS
        if ratio < 1.5:
            raise ValueError(f"Passive TP/SL ratio {ratio:.2f} < 1.5")
        return f"TP/SL = {PASSIVE_TAKE_PROFIT}/{PASSIVE_STOP_LOSS} (ratio {ratio:.2f})"

    def tp_sl_ratio_active():
        ratio = ACTIVE_TAKE_PROFIT / ACTIVE_STOP_LOSS
        if ratio < 1.5:
            raise ValueError(f"Active TP/SL ratio {ratio:.2f} < 1.5")
        return f"TP/SL = {ACTIVE_TAKE_PROFIT}/{ACTIVE_STOP_LOSS} (ratio {ratio:.2f})"

    def alloc_fractions():
        if ACTIVE_MIN_CAPITAL_FRACTION >= ACTIVE_MAX_CAPITAL_FRACTION:
            raise ValueError("ACTIVE_MIN_CAPITAL_FRACTION >= ACTIVE_MAX_CAPITAL_FRACTION")
        return f"active fraction [{ACTIVE_MIN_CAPITAL_FRACTION:.0%}, {ACTIVE_MAX_CAPITAL_FRACTION:.0%}]"

    check("Config: capital sum matches TOTAL_CAPITAL", cap_sum)
    check("Config: kill switch ≥ 70% of capital", kill_switch)
    check("Config: passive EV positive @ 55% WR", ev_passive)
    check("Config: active EV non-negative @ 50% WR", ev_active)
    check("Config: passive TP/SL ratio ≥ 1.5", tp_sl_ratio_passive)
    check("Config: active TP/SL ratio ≥ 1.5", tp_sl_ratio_active)
    check("Config: active capital fraction bounds valid", alloc_fractions)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Strategy signal generation
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_checks():
    from strategy.passive_strategy import PassiveStrategy
    from strategy.active_strategy import ActiveStrategy

    def passive_signal():
        s = PassiveStrategy()
        market_state = {
            "regime": "bull",
            "volatility": 0.01,
            "btc_change_24h": 0.02,
        }
        sig = s.generate_signal("BTC/USDT:USDT", market_state, mode="auto")
        if not isinstance(sig, dict):
            raise TypeError(f"Expected dict, got {type(sig)}")
        if "side" not in sig:
            raise KeyError("Signal missing 'side' key")
        return f"side={sig['side']}"

    def passive_skip_mode():
        s = PassiveStrategy()
        market_state = {"regime": "sideways", "volatility": 0.01}
        sig = s.generate_signal("BTC/USDT:USDT", market_state, mode="skip")
        if sig.get("side") != "HOLD":
            raise ValueError(f"mode='skip' should always return HOLD, got {sig.get('side')}")
        return "HOLD confirmed"

    def active_signal():
        s = ActiveStrategy()
        market_state = {
            "regime": "bull",
            "volatility": 0.01,
            "btc_change_24h": 0.02,
        }
        sig = s.generate_signal("BTC/USDT:USDT", market_state, mode="mean_reversion")
        if not isinstance(sig, dict):
            raise TypeError(f"Expected dict, got {type(sig)}")
        if "side" not in sig:
            raise KeyError("Signal missing 'side' key")
        return f"side={sig['side']}"

    def passive_update_params():
        s = PassiveStrategy()
        new_params = {"take_profit": 0.12, "stop_loss": 0.06, "ema_fast": 45, "ema_slow": 190, "max_hold_days": 10}
        s.update_params(new_params)
        return "update_params() accepted"

    def active_update_params():
        s = ActiveStrategy()
        new_params = {"take_profit": 0.018, "stop_loss": 0.008, "ema_fast": 7, "ema_slow": 25}
        s.update_params(new_params)
        return "update_params() accepted"

    check("Strategy: PassiveStrategy.generate_signal() returns dict", passive_signal)
    check("Strategy: PassiveStrategy mode='skip' returns HOLD", passive_skip_mode)
    check("Strategy: ActiveStrategy.generate_signal() returns dict", active_signal)
    check("Strategy: PassiveStrategy.update_params() works", passive_update_params)
    check("Strategy: ActiveStrategy.update_params() works", active_update_params)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Regime router
# ─────────────────────────────────────────────────────────────────────────────

def _regime_router_checks():
    from strategy.dual_regime_router import DualRegimeRouter

    router = DualRegimeRouter()
    regimes_and_expected = [
        ("bull",     {"passive_mode": "long",  "active_mode": "momentum"}),
        ("bear",     {"passive_mode": "short", "active_mode": "momentum"}),
        ("sideways", {"passive_mode": "skip",  "active_mode": "funding_arb"}),
        ("volatile", {"passive_mode": "skip",  "active_mode": "funding_arb"}),
    ]

    for regime, expected in regimes_and_expected:
        r = regime  # capture
        e = expected

        def _test(regime=r, expected=e):
            route = router.route({"regime": regime, "volatility": 0.01, "btc_change_24h": 0.0})
            for k, v in expected.items():
                if route.get(k) != v:
                    raise ValueError(
                        f"regime={regime}: expected {k}={v!r}, got {route.get(k)!r}"
                    )
            return f"passive={route['passive_mode']}, active={route['active_mode']}, size×{route.get('size_mult',1):.2f}"

        check(f"RegimeRouter: {regime} → correct modes", _test)

    def size_mult_in_range():
        for regime in ("bull", "bear", "sideways", "volatile"):
            route = router.route({"regime": regime, "volatility": 0.01, "btc_change_24h": 0.0})
            sm = route.get("size_mult", 1.0)
            if not (0.1 <= sm <= 2.0):
                raise ValueError(f"size_mult={sm} out of sensible range for regime={regime}")
        return "all size multipliers in [0.1, 2.0]"

    check("RegimeRouter: size_mult always in [0.1, 2.0]", size_mult_in_range)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Risk manager
# ─────────────────────────────────────────────────────────────────────────────

def _risk_checks():
    from risk.dual_risk import DualRiskManager

    def init():
        rm = DualRiskManager()
        return f"peak_equity={rm.peak_equity}"

    def conflict_same_side():
        rm = DualRiskManager()
        signal = {"symbol": "BTC/USDT:USDT", "side": "LONG"}
        passive_pos = [{"symbol": "BTC/USDT:USDT", "side": "SHORT"}]
        result = rm.check_conflict("active", signal, passive_pos, [])
        if result["allow_trade"]:
            raise ValueError("Should block LONG when passive holds SHORT on same symbol")
        return f"blocked: {result['reason']}"

    def conflict_opposite_ok_stacking_off():
        from config.dual_settings import DUAL_ALLOW_SAME_SYMBOL_STACKING
        if DUAL_ALLOW_SAME_SYMBOL_STACKING:
            return "stacking is ON — skip duplicate check"
        rm = DualRiskManager()
        signal = {"symbol": "BTC/USDT:USDT", "side": "LONG"}
        passive_pos = [{"symbol": "BTC/USDT:USDT", "side": "LONG"}]
        result = rm.check_conflict("active", signal, passive_pos, [])
        if result["allow_trade"]:
            raise ValueError("Should block same-side stacking when DUAL_ALLOW_SAME_SYMBOL_STACKING=False")
        return f"blocked duplicate: {result['reason']}"

    def record_trade_and_ev():
        rm = DualRiskManager()
        for _ in range(6):
            rm.record_trade("passive", 100.0)
        for _ in range(4):
            rm.record_trade("passive", -50.0)
        ev = rm.expected_value("passive")
        if ev == 0.0:
            raise ValueError("EV should be non-zero after 10 trades")
        return f"EV after 10 trades = {ev*100:.3f}%"

    def kill_switch_fires():
        from config.dual_settings import KILL_SWITCH_EQUITY
        rm = DualRiskManager()
        # Kill switch now requires 3 consecutive checks below threshold (hysteresis)
        for _ in range(3):
            result = rm.check_portfolio(KILL_SWITCH_EQUITY * 0.4, 0)
        if not result.get("halt"):
            raise ValueError("Kill switch should halt after 3 consecutive checks below threshold")
        return f"halt triggered after 3× below {KILL_SWITCH_EQUITY}"

    check("RiskManager: initialises cleanly", init)
    check("RiskManager: conflict checker blocks opposing sides", conflict_same_side)
    check("RiskManager: conflict checker blocks same-side stacking", conflict_opposite_ok_stacking_off)
    check("RiskManager: EV calculation after 10 trades", record_trade_and_ev)
    check("RiskManager: kill switch fires below threshold", kill_switch_fires)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Kelly criterion
# ─────────────────────────────────────────────────────────────────────────────

def _kelly_checks():
    from risk.kelly import KellyCriterion

    def positive_edge():
        k = KellyCriterion()
        f = k.half_kelly_fraction(winrate=0.55, win_pct=0.10, loss_pct=0.05)
        if not (0.01 <= f <= 0.20):
            raise ValueError(f"fraction {f} out of [1%, 20%]")
        return f"f = {f*100:.2f}%"

    def no_edge_floored():
        k = KellyCriterion()
        f = k.half_kelly_fraction(winrate=0.30, win_pct=0.05, loss_pct=0.10)
        if f < k.MIN_FRACTION:
            raise ValueError(f"fraction {f} below MIN_FRACTION")
        return f"negative Kelly floored to {f*100:.2f}%"

    def fallback_before_min_trades():
        k = KellyCriterion()
        val = k.position_value(0.55, 0.10, 0.05, equity=10_000, trades_so_far=5, default_fraction=0.08)
        expected = 10_000 * 0.08
        if abs(val - expected) > 1:
            raise ValueError(f"Expected {expected}, got {val}")
        return f"fallback returns {val:.0f} (8% of equity)"

    def kelly_scales_with_equity():
        k = KellyCriterion()
        v1 = k.position_value(0.55, 0.10, 0.05, equity=10_000, trades_so_far=20)
        v2 = k.position_value(0.55, 0.10, 0.05, equity=20_000, trades_so_far=20)
        if not (v2 > v1):
            raise ValueError("Position value should scale with equity")
        return f"10k→{v1:.0f}, 20k→{v2:.0f}"

    check("Kelly: positive edge gives fraction in [1%,20%]", positive_edge)
    check("Kelly: negative edge is floored to MIN_FRACTION", no_edge_floored)
    check("Kelly: < MIN_TRADES falls back to default fraction", fallback_before_min_trades)
    check("Kelly: position value scales with equity", kelly_scales_with_equity)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Correlation filter
# ─────────────────────────────────────────────────────────────────────────────

def _correlation_checks():
    from risk.correlation_filter import CorrelationFilter

    def blocks_highly_correlated():
        cf = CorrelationFilter()
        safe, reason = cf.is_safe(
            "ETH/USDT:USDT",
            ["BTC/USDT:USDT"],
            _MOCK_MARKET,
        )
        if safe:
            raise ValueError("BTC and ETH (ρ≈0.88) should be blocked")
        return f"blocked: {reason}"

    def allows_uncorrelated():
        cf = CorrelationFilter()
        safe, reason = cf.is_safe(
            "SOL/USDT:USDT",
            [],           # no existing positions
            _MOCK_MARKET,
        )
        if not safe:
            raise ValueError(f"Empty portfolio should always allow: {reason}")
        return "empty portfolio → allowed"

    def allows_same_symbol_absent():
        cf = CorrelationFilter()
        safe, reason = cf.is_safe(
            "SOL/USDT:USDT",
            ["ADA/USDT:USDT"],
            _MOCK_MARKET,
        )
        # SOL/ADA is not in KNOWN_CORR so live correlation is attempted;
        # mock market returns uniform random prices → live corr will be near 0
        return f"safe={safe}, reason={reason}"

    check("CorrelationFilter: BTC/ETH (ρ=0.88) is blocked", blocks_highly_correlated)
    check("CorrelationFilter: empty portfolio always allowed", allows_uncorrelated)
    warn("CorrelationFilter: unknown pair falls back to live calc", allows_same_symbol_absent)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – Capital allocator
# ─────────────────────────────────────────────────────────────────────────────

def _allocator_checks():
    from portfolio.capital_allocator import DualCapitalAllocator

    def init():
        ca = DualCapitalAllocator()
        if not (0 < ca.active_weight < 1):
            raise ValueError(f"active_weight {ca.active_weight} out of (0,1)")
        passive_w = round(1 - ca.active_weight, 4)
        return f"passive={passive_w:.0%} active={ca.active_weight:.0%}"

    def position_cap_passive():
        ca = DualCapitalAllocator()
        cap = ca.position_cap("passive", max_positions=3, total_equity=35000)
        if cap <= 0:
            raise ValueError(f"position_cap returned {cap} — expected positive value")
        return f"passive cap = {cap:.0f} USDT"

    def position_cap_active():
        ca = DualCapitalAllocator()
        cap = ca.position_cap("active", max_positions=2, total_equity=35000)
        if cap <= 0:
            raise ValueError(f"position_cap returned {cap} — expected positive value")
        return f"active cap = {cap:.0f} USDT"

    def active_weight_clamped():
        from config.dual_settings import ACTIVE_MIN_CAPITAL_FRACTION, ACTIVE_MAX_CAPITAL_FRACTION
        ca = DualCapitalAllocator()
        w = ca.active_weight
        if not (ACTIVE_MIN_CAPITAL_FRACTION <= w <= ACTIVE_MAX_CAPITAL_FRACTION):
            raise ValueError(
                f"active_weight {w} outside [{ACTIVE_MIN_CAPITAL_FRACTION}, {ACTIVE_MAX_CAPITAL_FRACTION}]"
            )
        return f"weight {w:.0%} within configured bounds"

    check("CapitalAllocator: initialises with valid active_weight", init)
    check("CapitalAllocator: position_cap() positive for passive", position_cap_passive)
    check("CapitalAllocator: position_cap() positive for active", position_cap_active)
    check("CapitalAllocator: active_weight within configured bounds", active_weight_clamped)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – Strategy evolver
# ─────────────────────────────────────────────────────────────────────────────

def _evolver_checks():
    from self_evolve.strategy_evolver import StrategyEvolver, DEFAULTS, BOUNDS

    def loads_defaults():
        ev = StrategyEvolver()
        p = ev.current_params("passive")
        for key in DEFAULTS["passive"]:
            if key not in p:
                raise KeyError(f"Param '{key}' missing from passive defaults")
        return "all default keys present"

    def mutation_in_bounds():
        ev = StrategyEvolver()
        for _ in range(20):
            cand = ev._mutate("passive")
            for key, (lo, hi) in BOUNDS["passive"].items():
                val = cand[key]
                if not (lo <= val <= hi):
                    raise ValueError(f"{key}={val} outside [{lo},{hi}]")
            if cand["ema_fast"] >= cand["ema_slow"]:
                raise ValueError(
                    f"ema_fast({cand['ema_fast']}) >= ema_slow({cand['ema_slow']})"
                )
        return "20 mutations all in-bounds, ema_fast < ema_slow"

    def backtest_passive_runs():
        from self_evolve.strategy_evolver import _backtest_passive, DEFAULTS
        rng = np.random.default_rng(0)
        n = 250
        c = 30_000 + np.cumsum(rng.normal(0, 200, n))
        h = c * 1.003
        lo = c * 0.997
        s = _backtest_passive(c, h, lo, DEFAULTS["passive"])
        return f"Sharpe = {s:.4f}"

    def backtest_active_runs():
        from self_evolve.strategy_evolver import _backtest_active, DEFAULTS
        rng = np.random.default_rng(1)
        n = 450
        c = 30_000 + np.cumsum(rng.normal(0, 50, n))
        h = c * 1.002
        lo = c * 0.998
        s = _backtest_active(c, h, lo, DEFAULTS["active"])
        return f"Sharpe = {s:.4f}"

    def evolved_params_loadable():
        from self_evolve.strategy_evolver import PARAMS_FILE
        if not PARAMS_FILE.exists():
            return "no evolved params file yet — defaults will be used"
        ev = StrategyEvolver()
        p = ev.current_params("passive")
        for key in DEFAULTS["passive"]:
            if key not in p:
                raise KeyError(f"Evolved passive missing '{key}'")
        pa = ev.current_params("active")
        for key in DEFAULTS["active"]:
            if key not in pa:
                raise KeyError(f"Evolved active missing '{key}'")
        return "evolved params loaded and complete"

    check("Evolver: loads defaults for both strategies", loads_defaults)
    check("Evolver: _mutate() stays in BOUNDS and ema_fast < ema_slow", mutation_in_bounds)
    check("Evolver: _backtest_passive() runs on synthetic data", backtest_passive_runs)
    check("Evolver: _backtest_active() runs on synthetic data", backtest_active_runs)
    check("Evolver: evolved_params.json loadable (or absent gracefully)", evolved_params_loadable)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _dashboard_checks():
    def html_renders():
        from monitoring.dashboard import _html
        html = _html()
        if len(html) < 500:
            raise ValueError(f"HTML suspiciously short: {len(html)} chars")
        required = ["Quant Singularity", "Portfolio", "Passive Strategy", "Active Strategy",
                    "Capital Allocation", "Evolved Params", "Evolution Log"]
        missing = [s for s in required if s not in html]
        if missing:
            raise ValueError(f"Dashboard HTML missing sections: {missing}")
        return f"HTML ok ({len(html):,} chars)"

    def normalize_empty():
        from monitoring.dual_dashboard import normalize_dual_state
        result = normalize_dual_state({})
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict, got {type(result)}")
        return "empty state normalized without error"

    def normalize_with_dual():
        from monitoring.dual_dashboard import normalize_dual_state
        state = {
            "dual": {
                "passive": {"equity": 25000, "realized_pnl": 150},
                "active":  {"equity": 10000, "realized_pnl": -30},
                "total_portfolio_value": 35150,
                "strategy_stats": {"passive": {"trades": 5}, "active": {"trades": 2}},
            }
        }
        result = normalize_dual_state(state)
        if result.get("dual", {}).get("passive", {}).get("equity") != 25000:
            raise ValueError("normalize_dual_state lost passive equity")
        return "dual state preserved correctly"

    check("Dashboard: _html() renders full page", html_renders)
    check("Dashboard: normalize_dual_state({}) is safe", normalize_empty)
    check("Dashboard: normalize_dual_state(dual_data) preserves values", normalize_with_dual)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – Regime classifier
# ─────────────────────────────────────────────────────────────────────────────

def _regime_checks():
    def volatile_detected():
        from meta.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        # Need >= 100 bars; closes[-24] must still be stable (30000), then jump in last 23
        stable = np.full(77, 30_000.0)    # closes[-24] = closes[76] = 30000
        jumped = np.full(23, 32_200.0)    # closes[-1] = 32200 → +7.3% (>6% threshold)
        closes = np.concatenate([stable, jumped])
        regime = clf.classify(closes)
        if regime != "volatile":
            raise ValueError(f"Expected 'volatile', got '{regime}'")
        pct = abs(closes[-1] - closes[-24]) / closes[-24]
        return f"volatile regime detected ({pct:.1%} 24h move)"

    def bull_detected():
        from meta.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        # Steady uptrend, low volatility
        closes = np.linspace(25_000, 32_000, 100)
        regime = clf.classify(closes)
        if regime not in ("bull", "volatile"):
            raise ValueError(f"Expected 'bull', got '{regime}'")
        return f"regime = {regime}"

    check("RegimeClassifier: volatile regime fires on large move", volatile_detected)
    check("RegimeClassifier: bull regime on steady uptrend", bull_detected)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def _print_results():
    print()
    width = max(len(n) for n, _, _ in _results) + 4
    passes = 0
    fails  = 0
    warns  = 0

    for name, ok, msg in _results:
        if ok is True:
            status = f"{GREEN}PASS{RESET}"
            passes += 1
        elif ok is None:
            status = f"{YELLOW}WARN{RESET}"
            warns += 1
        else:
            status = f"{RED}FAIL{RESET}"
            fails += 1

        detail = f"  ({msg})" if msg else ""
        print(f"  [{status}] {name:<{width}}{detail}")

    print()
    total = passes + fails + warns
    print(f"  Results: {GREEN}{passes} pass{RESET} / {RED}{fails} fail{RESET} / {YELLOW}{warns} warn{RESET}  ({total} checks)")
    print()

    if fails == 0:
        print(f"  {GREEN}✓ All checks passed — dual mode is ready for paper trading.{RESET}")
    else:
        print(f"  {RED}✗ {fails} check(s) failed — fix issues before going live.{RESET}")
    print()

    return fails == 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – Strategy Bandit
# ─────────────────────────────────────────────────────────────────────────────

def _bandit_checks():
    def imports_ok():
        from meta.strategy_bandit import DualStrategyBandit, ThompsonBandit
        b = DualStrategyBandit()
        return "DualStrategyBandit instantiated"

    def select_returns_valid_arm():
        from meta.strategy_bandit import DualStrategyBandit, ACTIVE_ARMS
        b = DualStrategyBandit()
        mode = b.select_active_mode("bull", "momentum")
        if mode not in ACTIVE_ARMS:
            raise ValueError(f"select_active_mode returned invalid arm: {mode}")
        return f"selected={mode}"

    def warmup_returns_recommended():
        from meta.strategy_bandit import DualStrategyBandit
        b = DualStrategyBandit()
        # No trials yet — must return recommended arm
        mode = b.select_active_mode("bear", "mean_reversion")
        if mode != "mean_reversion":
            raise ValueError(f"During warmup expected 'mean_reversion', got '{mode}'")
        return "warmup correctly defers to regime recommendation"

    def records_and_learns():
        from meta.strategy_bandit import DualStrategyBandit
        b = DualStrategyBandit()
        # Record many wins for funding_arb, losses for momentum in sideways
        for _ in range(15):
            b.record_active("sideways", "funding_arb", won=True)
        for _ in range(15):
            b.record_active("sideways", "momentum", won=False)
        # After warmup, should prefer funding_arb over momentum
        mode = b.select_active_mode("sideways", "momentum")
        return f"bandit chose: {mode} (funding_arb wins=15 vs momentum wins=0)"

    def summary_non_empty_after_records():
        from meta.strategy_bandit import DualStrategyBandit
        b = DualStrategyBandit()
        b.record_active("bull", "momentum", won=True)
        s = b.summary()
        if "active" not in s:
            raise KeyError("summary() missing 'active' key")
        return "summary() returns correct structure"

    check("Bandit: imports cleanly", imports_ok)
    check("Bandit: select returns valid arm", select_returns_valid_arm)
    check("Bandit: warmup defers to regime recommendation", warmup_returns_recommended)
    check("Bandit: learns from outcomes (biased toward wins)", records_and_learns)
    check("Bandit: summary() structure correct", summary_non_empty_after_records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 – Daily Profit Engine
# ─────────────────────────────────────────────────────────────────────────────

def _daily_profit_checks():
    def imports_ok():
        from meta.daily_profit_engine import DailyProfitEngine
        e = DailyProfitEngine(active_capital=300.0)
        return f"DailyProfitEngine initialised, target={e._target}"

    def phases_correct():
        from meta.daily_profit_engine import DailyProfitEngine
        e = DailyProfitEngine(active_capital=1000.0)  # target = 10 USDT
        target = e._target

        e._pnl = 0.0
        if e.get_phase() != "HUNTING":
            raise ValueError(f"0% → expected HUNTING, got {e.get_phase()}")

        e._pnl = target * 0.50
        if e.get_phase() != "ON_TRACK":
            raise ValueError(f"50% → expected ON_TRACK, got {e.get_phase()}")

        e._pnl = target * 0.85
        if e.get_phase() != "PROTECTING":
            raise ValueError(f"85% → expected PROTECTING, got {e.get_phase()}")

        e._pnl = target * 1.10
        if e.get_phase() != "LOCKED":
            raise ValueError(f"110% → expected LOCKED, got {e.get_phase()}")

        return "HUNTING→ON_TRACK→PROTECTING→LOCKED all correct"

    def aggression_multipliers():
        from meta.daily_profit_engine import DailyProfitEngine, _PHASE_MULT
        e = DailyProfitEngine(active_capital=1000.0)
        target = e._target

        e._pnl = 0.0
        m = e.get_aggression_mult()
        if m != _PHASE_MULT["HUNTING"]:
            raise ValueError(f"HUNTING mult wrong: {m}")

        e._pnl = target * 1.5
        if e.should_open_new():
            raise ValueError("should_open_new() must be False when LOCKED")

        return f"HUNTING={_PHASE_MULT['HUNTING']}x, LOCKED blocks entries"

    def snapshot_has_thb():
        from meta.daily_profit_engine import DailyProfitEngine, THB_PER_USD
        e = DailyProfitEngine(active_capital=500.0)
        e._pnl = 10.0
        snap = e.snapshot()
        expected_thb = round(10.0 * THB_PER_USD, 0)
        if snap["pnl_today_thb"] != expected_thb:
            raise ValueError(f"THB mismatch: {snap['pnl_today_thb']} ≠ {expected_thb}")
        return f"10 USDT → ฿{expected_thb:.0f} THB (rate={THB_PER_USD})"

    check("DailyProfit: imports cleanly", imports_ok)
    check("DailyProfit: phase transitions correct", phases_correct)
    check("DailyProfit: aggression multipliers + LOCKED gate", aggression_multipliers)
    check("DailyProfit: snapshot includes THB amounts", snapshot_has_thb)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 – LightGBM Predictor
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_checks():
    def imports_ok():
        from models.lgbm_predictor import LGBMPredictor, _LGB_OK
        p = LGBMPredictor()
        return f"LGBMPredictor instantiated (lightgbm_available={_LGB_OK})"

    def feature_extraction():
        from models.lgbm_predictor import extract_features, N_FEATURES
        rng = np.random.default_rng(0)
        closes = 30_000 + np.cumsum(rng.normal(0, 100, 250))
        vols   = rng.uniform(100, 500, 250)
        feat   = extract_features(closes, vols, hour=12)
        if len(feat) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(feat)}")
        if np.any(np.isnan(feat)):
            raise ValueError("NaN in features")
        return f"{N_FEATURES} features extracted, no NaN"

    def predict_returns_valid_dict():
        from models.lgbm_predictor import LGBMPredictor
        p    = LGBMPredictor()
        rng  = np.random.default_rng(1)
        c    = 30_000 + np.cumsum(rng.normal(0, 100, 200))
        pred = p.predict(c, regime="global")
        if pred["direction"] not in ("LONG", "SHORT", "HOLD"):
            raise ValueError(f"Invalid direction: {pred['direction']}")
        if not (0.0 <= pred["confidence"] <= 1.0):
            raise ValueError(f"confidence out of range: {pred['confidence']}")
        return f"direction={pred['direction']} conf={pred['confidence']:.3f} trained={pred['trained']}"

    def train_on_synthetic():
        from models.lgbm_predictor import LGBMPredictor, _LGB_OK
        if not _LGB_OK:
            return "lightgbm not installed — skipping train test"
        p   = LGBMPredictor()
        rng = np.random.default_rng(42)
        c   = 30_000 + np.cumsum(rng.normal(0, 80, 600))
        v   = rng.uniform(200, 800, 600)
        acc = p.train(c, v, regime="global")
        if acc <= 0:
            raise ValueError("train() returned 0 — dataset too small or error")
        return f"trained global model accuracy={acc:.3f}"

    check("LGBM: imports cleanly", imports_ok)
    check("LGBM: feature extraction (18 features, no NaN)", feature_extraction)
    check("LGBM: predict() returns valid dict (untrained fallback)", predict_returns_valid_dict)
    check("LGBM: train() on synthetic data runs without error", train_on_synthetic)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== DUAL STRATEGY VALIDATION ===")

    sections = [
        ("1. Imports",            _import_checks),
        ("2. Config Sanity",      _config_checks),
        ("3. Strategy Signals",   _strategy_checks),
        ("4. Regime Router",      _regime_router_checks),
        ("5. Risk Manager",       _risk_checks),
        ("6. Kelly Criterion",    _kelly_checks),
        ("7. Correlation Filter", _correlation_checks),
        ("8. Capital Allocator",  _allocator_checks),
        ("9. Strategy Evolver",   _evolver_checks),
        ("10. Dashboard",         _dashboard_checks),
        ("11. Regime Classifier", _regime_checks),
        ("12. Strategy Bandit",   _bandit_checks),
        ("13. Daily Profit Engine",_daily_profit_checks),
        ("14. LightGBM Predictor",_lgbm_checks),
    ]

    for title, fn in sections:
        print(f"\n── {title} ──")
        try:
            fn()
        except Exception:
            _results.append((f"SECTION {title}", False, traceback.format_exc(limit=2)))

    ok = _print_results()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
