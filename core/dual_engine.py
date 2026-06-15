"""Dual strategy engine — passive (long-term wealth builder) + active (daily profit machine)."""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from collections import deque

from config.dual_settings import (
    ACTIVE_CAPITAL,
    ACTIVE_LEVERAGE,
    ACTIVE_MAX_POSITIONS,
    PASSIVE_CAPITAL,
    PASSIVE_LEVERAGE,
    PASSIVE_MAX_POSITIONS,
)
from config.settings import DASHBOARD_ENABLED, LOOP_DELAY_SECONDS
from core.broker import create_broker
from core.logger import get_logger
from core.state_manager import StateManager
from core.trade_manager import TradeManager
from meta.market_intelligence import MarketIntelligence
from meta.market_state import MarketState
from models.online_learner import OnlineLearner
from models.predictor import MLPredictor
from monitoring.alerting import Alerting
from monitoring.dashboard import DashboardServer
from portfolio.capital_allocator import DualCapitalAllocator
from portfolio.portfolio_engine import PortfolioEngine
from research.coin_scanner import CoinScanner
from risk.correlation_filter import CorrelationFilter
from risk.dual_risk import DualRiskManager
from meta.daily_profit_engine import DailyProfitEngine
from meta.strategy_bandit import DualStrategyBandit
from models.lgbm_predictor import LGBMPredictor
from models.drift_detector import DriftDetector
from self_evolve.strategy_evolver import StrategyEvolver
from meta.ai_brain import AIBrain
from risk.symbol_edge import SymbolEdgeTracker
from self_evolve.strategy_lab import StrategyLab
from research.performance_tracker import PerformanceTracker
from strategy.active_strategy import ActiveStrategy
from strategy.dual_regime_router import DualRegimeRouter
from strategy.passive_strategy import PassiveStrategy
from meta.compound_manager import CompoundManager
from meta.profit_vault import ProfitVault
from core.cross_engine_bridge import CrossEngineBridge
from data.orderbook import OrderBookSignal
from data.market_microstructure import MarketMicrostructure
from data.news_feed import NewsFeed
from meta.adaptive_controller import AdaptiveController
from research.equity_tracker import EquityTracker
from core.position_sync import sync_positions_from_state, sync_positions_from_binance


CYCLE_PRINT_INTERVAL = 20


class DualEngine:
    """
    Coordinates the two autonomous agents:

    Agent 1 — Passive (wealth builder)
      Holds multi-day trend positions, dynamically selected from a scanned universe.
      Uses ATR-based stops, ADX trend filter, volume confirmation, trailing stops,
      and partial profit-taking. Evolves its own parameters via strategy evolver.

    Agent 2 — Active (daily profit machine)
      Opens and closes intraday positions across four signal modes.
      Uses VWAP, session quality gating, daily profit target, and confidence-weighted sizing.
      Learns from every closed trade via online gradient updates.
    """

    def __init__(self):
        self.log             = get_logger()
        self.market_state    = MarketState()
        self.scanner         = CoinScanner()
        self.passive_strategy = PassiveStrategy()
        self.active_strategy  = ActiveStrategy()
        self.risk_manager    = DualRiskManager()
        self.capital_allocator = DualCapitalAllocator()
        self.broker          = create_broker(price_feed=self.market_state.market)
        self.passive_portfolio = PortfolioEngine(initial_cash=PASSIVE_CAPITAL)
        self.active_portfolio  = PortfolioEngine(initial_cash=ACTIVE_CAPITAL)
        self.state_manager   = StateManager()
        self.regime_router   = DualRegimeRouter()
        self.correlation_filter = CorrelationFilter()
        self.strategy_evolver   = StrategyEvolver()
        self.trade_manager      = TradeManager()
        self.market_intel       = MarketIntelligence()
        self.online_learner     = OnlineLearner()
        self.ml_predictor       = MLPredictor()
        self.lgbm               = LGBMPredictor()
        self.strategy_bandit    = DualStrategyBandit()
        self.daily_profit       = DailyProfitEngine(ACTIVE_CAPITAL)
        self._last_lgbm_retrain: float = self._load_lgbm_retrain_ts()

        # Seed strategies with any previously evolved params at startup
        self.passive_strategy.update_params(self.strategy_evolver.current_params("passive"))
        self.active_strategy.update_params(self.strategy_evolver.current_params("active"))

        self.edge_tracker    = SymbolEdgeTracker()
        self.ai_brain        = AIBrain()
        self.strategy_lab    = StrategyLab(self.strategy_evolver)
        self._last_lgbm_emergency: float = 0.0   # for 4-h emergency retrain path

        self.alerting          = Alerting()
        self.perf_tracker      = PerformanceTracker()
        self.drift_detector    = DriftDetector()
        self.compound_manager  = CompoundManager(self.passive_portfolio, self.active_portfolio)
        self.profit_vault      = ProfitVault()
        self.cross_bridge      = CrossEngineBridge()
        self.ob_signal         = OrderBookSignal()
        self.microstructure    = MarketMicrostructure()
        self.news_feed         = NewsFeed()
        self.apc               = AdaptiveController(self.active_portfolio, self.passive_portfolio)
        self.equity_tracker    = EquityTracker()
        self._last_news:       dict = {}
        self._news_check_ts:   float = 0.0
        self._last_alert_day: int = -1
        self._last_report_day: int = -1   # daily autopilot report — fires once per UTC day
        self._active_paused_until: float = 0.0   # AI Brain or circuit breaker can pause
        self._consec_losses: int = 0              # consecutive active losses counter
        # Adaptive mode: track win/loss per active signal mode (last 20 trades each)
        self._mode_stats: dict[str, deque] = {
            mode: deque(maxlen=20)
            for mode in ("momentum", "mean_reversion", "funding_arb", "vwap_reversal")
        }
        self.conflict_log:  list[dict] = []
        self._last_route:   dict = {}
        self._last_intel:   dict = {}
        self._cooldowns:    dict[str, float] = {}   # f"{strategy}:{symbol}" → expiry ts
        self._conflict_logged: dict[str, float] = {}  # symbol → last log ts (rate-limit spam)
        self._prev_regime:      str = ""
        self._regime_candidate: str = ""
        self._regime_streak:    int = 0
        self._active_idle_cycles: int = 0
        self.cycle = 0
        self.last_report_at = time.time()

        # ── Phase A/B/C enhancements ───────────────────────────────────────────
        # C1: Minimum trade spacing — prevent 3 positions opening in rapid succession
        self._last_active_open_ts: float = 0.0
        # C2: Post-circuit-breaker elevated confidence window
        self._elevated_confidence_until: float = 0.0
        # C3: Daily PnL gates (configurable via env vars; otherwise scale with
        # current active equity so the gates don't stay pinned to the initial
        # $300 seed as the account compounds — see _daily_pause_loss_threshold
        # and _daily_protect_profit_threshold below)
        self._daily_pause_loss_override   = float(os.getenv("QS_DAILY_PAUSE_LOSS", "0"))
        self._daily_protect_profit_override = float(os.getenv("QS_DAILY_PROTECT_PROFIT", "0"))
        self._active_daily_paused_today: bool = False
        self._active_protect_mode_until: float = 0.0
        self._protect_mode_date = None
        # B4: Per-mode/regime performance log
        self._mode_regime_stats: dict[str, dict] = {}

        # ── Position sync on startup ──────────────────────────────────────────
        # Restore open positions from last saved state so bot can manage them
        # after a crash or restart without re-opening duplicates on Binance.
        try:
            from config.settings import STATE_FILE
            n = sync_positions_from_state(
                STATE_FILE,
                self.passive_portfolio,
                self.active_portfolio,
            )
            if n > 0:
                self.log.info(f"[PositionSync] restored {n} positions from state file")
            else:
                # Fallback: sync directly from Binance API
                from data.binance_client import BinanceClient
                ex = BinanceClient().get_exchange()
                n2 = sync_positions_from_binance(
                    ex,
                    self.passive_portfolio,
                    self.active_portfolio,
                )
                if n2 > 0:
                    self.log.info(
                        f"[PositionSync] restored {n2} positions from Binance API "
                        f"(no TP/SL — monitor manually)"
                    )

            # After sync, reset _peak_equity to actual equity from saved state so
            # the drawdown gauge starts from a sensible baseline, not initial_cash.
            # Without this, restoring an equity below initial_cash (e.g. after
            # capital was reallocated to the other sleeve in a prior session)
            # would show false drawdown on every restart, triggering survival mode
            # for reasons unrelated to this session's trading performance.
            try:
                import json as _j
                from config.settings import STATE_FILE as _sf
                _saved = _j.loads(_sf.read_text())
                _d = _saved.get("dual", {})
                _p_eq = _d.get("passive", {}).get("equity", 0.0) or 0.0
                _a_eq = _d.get("active",  {}).get("equity", 0.0) or 0.0
                if _p_eq > 0:
                    self.passive_portfolio._peak_equity = float(_p_eq)
                if _a_eq > 0:
                    self.active_portfolio._peak_equity  = float(_a_eq)
                if _a_eq > 0:
                    # Seed the capital-dependent gates with the restored active
                    # equity so they're correct from cycle 1, not just after the
                    # first _rebalance_capital() call.
                    self.daily_profit.update_capital(_a_eq)
                    self.active_strategy.update_capital(_a_eq)
                    self.risk_manager.update_capital(_a_eq)

                # capital_allocator.active_weight/history are in-memory only and
                # would otherwise reset to the initial 30/70 split on every
                # restart, discarding any rebalancing drift. Restore from the
                # last persisted snapshot.
                _alloc = _d.get("capital_allocation", {})
                _aw = _alloc.get("active_weight")
                if _aw is not None:
                    self.capital_allocator.active_weight = self.capital_allocator._clamp_active_weight(float(_aw))
                if _alloc.get("history"):
                    self.capital_allocator.history = _alloc["history"]
            except Exception:
                pass

            # risk_manager.peak_equity is statically initialized to
            # PASSIVE_CAPITAL + ACTIVE_CAPITAL on every restart, which would
            # silently reset the 15% portfolio drawdown kill-switch baseline
            # below the true historical high-water mark once equity has
            # compounded past that seed value. compound_manager already
            # persists the real high-water mark to compound_state.json, so
            # sync risk_manager off of it here.
            try:
                self.risk_manager.peak_equity = max(
                    self.risk_manager.peak_equity,
                    self.compound_manager._peak_equity,
                )
            except Exception:
                pass

        except Exception as e:
            self.log.warning(f"[PositionSync] startup sync failed: {e}")

        if DASHBOARD_ENABLED:
            DashboardServer().start()

    # ── helpers ────────────────────────────────────────────────────────────────

    # ── A2: Session-based position limit ──────────────────────────────────────

    @staticmethod
    def _session_max_active_positions() -> int:
        """Return max active positions allowed for the current UTC session.
        Asian session (00:00-06:59 UTC): lower liquidity → cap at 1 position.
        London/NY overlap (07:00-23:59 UTC): normal cap of 3.
        """
        hour = datetime.now(timezone.utc).hour
        return 1 if 0 <= hour <= 6 else 3

    # ── A3: Correlation guard (same-direction count across both portfolios) ────

    def _same_direction_count(self, side: str) -> int:
        """Count how many open positions across BOTH portfolios are in *side* direction."""
        count = 0
        for pos in self.passive_portfolio.positions:
            if pos.get("side") == side:
                count += 1
        for pos in self.active_portfolio.positions:
            if pos.get("side") == side:
                count += 1
        return count

    # ── B1: EV gate — skip modes with proven negative EV (≥15 trades) ─────────

    def _mode_ev(self, mode: str, regime: str) -> float:
        """Compute EV for an active mode using per-mode win stats.
        EV = winrate × TP − (1−winrate) × SL.
        Returns 0.0 when insufficient data.
        """
        from config.dual_settings import ACTIVE_TAKE_PROFIT, ACTIVE_STOP_LOSS
        hist = self._mode_stats.get(mode, [])
        n = len(hist)
        if n < 15:
            return 0.0   # not enough data
        winrate = sum(hist) / n
        ev = winrate * ACTIVE_TAKE_PROFIT - (1 - winrate) * ACTIVE_STOP_LOSS
        return round(ev, 6)

    @staticmethod
    def _load_lgbm_retrain_ts() -> float:
        try:
            from config.settings import STORAGE_DIR as SD
            p = SD / "lgbm_retrain_ts.json"
            if p.exists():
                return float(json.loads(p.read_text()).get("ts", 0.0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _save_lgbm_retrain_ts(ts: float) -> None:
        try:
            from config.settings import STORAGE_DIR as SD
            (SD / "lgbm_retrain_ts.json").write_text(json.dumps({"ts": ts}))
        except Exception:
            pass

    def _should_print_cycle(self) -> bool:
        return self.cycle <= 3 or self.cycle % CYCLE_PRINT_INTERVAL == 0

    def _print_cycle_status(self) -> None:
        if not self._should_print_cycle():
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[DualEngine] cycle={self.cycle} ts={ts}")

    def _record_cooldown(self, symbol: str, strategy: str) -> None:
        """Block re-entry on same symbol after a stop-loss. Active: 30 min, Passive: 4 h."""
        duration = 1800 if strategy == "active" else 14400
        self._cooldowns[f"{strategy}:{symbol}"] = time.time() + duration

    def _in_cooldown(self, symbol: str, strategy: str) -> bool:
        return time.time() < self._cooldowns.get(f"{strategy}:{symbol}", 0.0)

    def _handle_regime_shift(self, old_regime: str, new_regime: str) -> None:
        """On shift to volatile/bear: breakeven-lock all open passive positions."""
        if new_regime not in {"volatile", "bear"} or old_regime == new_regime:
            return
        locked = 0
        for pos in self.passive_portfolio.positions:
            entry = pos["entry_price"]
            if pos["side"] == "LONG":
                new_sl = round(entry * 1.0005, 6)
                if (pos.get("stop_loss") or 0.0) < new_sl:
                    pos["stop_loss"]       = new_sl
                    pos["trailing_active"] = True
                    locked += 1
            else:
                new_sl = round(entry * 0.9995, 6)
                if (pos.get("stop_loss") or float("inf")) > new_sl:
                    pos["stop_loss"]       = new_sl
                    pos["trailing_active"] = True
                    locked += 1
        if locked:
            self.log.info(
                f"[RegimeShift] {old_regime}→{new_regime}: "
                f"breakeven-locked {locked} passive positions"
            )

    def _record_conflict(self, symbol: str, reason: str) -> None:
        self.conflict_log.insert(0, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason,
        })
        self.conflict_log = self.conflict_log[:25]

    def _position_data(self, signal: dict, fill: dict, strategy: str) -> dict:
        leverage = ACTIVE_LEVERAGE if strategy == "active" else PASSIVE_LEVERAGE
        return {
            "symbol":         fill["symbol"],
            "side":           fill["side"],
            "size":           fill["size"],
            "entry_price":    fill["price"],
            "position_value": round(fill["price"] * fill["size"], 2),
            "stop_loss":      signal.get("stop_loss"),
            "take_profit":    signal.get("take_profit"),
            "strategy":       strategy,
            "regime":         signal.get("regime"),
            "opened_at":      time.time(),
            "min_hold_hours": signal.get("min_hold_hours", 0),
            "max_hold_hours": signal.get("max_hold_hours", 0),
            "current_price":  fill["price"],
            "trailing_active": False,
            "leverage":       leverage,
            "signal_mode":    signal.get("signal_mode", ""),
            "passive_mode":   signal.get("passive_mode", ""),
            "confidence":     round(signal.get("confidence", 0.0), 4),
        }

    def _update_position_price(self, portfolio: PortfolioEngine, position: dict) -> None:
        try:
            current_price = self.broker.get_price(position["symbol"])
            portfolio.update_position_price(position["symbol"], current_price)
            position["current_price"] = current_price
            position["_price_fail_count"] = 0
        except Exception:
            # Price fetch failed — keep using last known price, never remove position
            # Force-close is dangerous during rate limiting: ETH/BTC are never delisted
            count = position.get("_price_fail_count", 0) + 1
            position["_price_fail_count"] = count
            if count == 1 or count % 40 == 0:   # log once, then every ~10 min
                self.log.warning(
                    f"[PriceFeed] cannot fetch {position['symbol']} "
                    f"(fail #{count}) — holding last known price "
                    f"{position.get('current_price', '?')}"
                )

    def _extract_features_now(self, symbol: str):
        """Extract ML features from CURRENT market state (used at close time)."""
        try:
            df = self.market_state.market.get_ohlcv_df(symbol, timeframe="15m", limit=50)
            if len(df) < 30:
                return None
            return self.ml_predictor.extract_features(df["close"].values, df["volume"].values)
        except Exception:
            return None

    # ── position close helpers ─────────────────────────────────────────────────

    def _close_position(
        self,
        portfolio: PortfolioEngine,
        position: dict,
        reason: str = "",
    ) -> float:
        pnl = self.broker.close_position(position)
        # Deduct Binance taker fee (0.04% of notional value, paid on both open and close)
        fee = round(position.get("position_value", 0.0) * 0.0008, 4)   # 2 × 0.04%
        pnl = round(pnl - fee, 4)
        portfolio.close_position(position["symbol"], pnl)
        self.risk_manager.record_trade(position["strategy"], pnl)
        # Pass pnl_pct so SymbolEdgeTracker can use Kelly criterion sizing
        pos_val = position.get("position_value", 1.0) or 1.0
        pnl_pct = round(pnl / pos_val, 6)
        self.edge_tracker.record(position["symbol"], pnl >= 0, pnl_pct=pnl_pct)

        # Persist trade to database for analytics and backtesting
        try:
            from research.trade_logger import TradeLogger
            TradeLogger().log_trade({
                "symbol":      position["symbol"],
                "side":        position["side"],
                "entry_price": position.get("entry_price", 0),
                "exit_price":  position.get("current_price", 0),
                "qty":         position.get("size", 0),
                "pnl":         pnl,
                "strategy":    position["strategy"],
                "regime":      position.get("regime", "unknown"),
                "signal_mode": position.get("signal_mode", ""),
                "confidence":  position.get("confidence", 0),
                "reason":      reason,
            })
        except Exception:
            pass

        self.alerting.trade_close(
            f"[{position['strategy']}] {position['symbol']}",
            reason, pnl, portfolio.equity,
        )

        # Adaptive mode: record outcome for active trades
        if position.get("strategy") == "active":
            mode = position.get("signal_mode", "")
            if mode in self._mode_stats:
                self._mode_stats[mode].append(1 if pnl >= 0 else 0)
            # B4: per-mode/regime performance log for future EV gating
            if mode:
                regime_close = position.get("regime", "sideways") or "sideways"
                key = f"{mode}_{regime_close}"
                entry = self._mode_regime_stats.setdefault(key, {"trades": 0, "wins": 0})
                entry["trades"] += 1
                if pnl >= 0:
                    entry["wins"] += 1

        # Re-entry cooldown after any stop-out (STOP_LOSS or trailing stop)
        if "STOP" in reason or "TRAIL" in reason:
            self._record_cooldown(position["symbol"], position["strategy"])

        # Consecutive loss circuit breaker for active strategy
        if position["strategy"] == "active":
            if pnl < 0:
                self._consec_losses += 1
                # 3 consecutive losses → pause active for 1h, alert user
                if self._consec_losses >= 3:
                    pause_secs = 3600
                    self._active_paused_until = time.time() + pause_secs
                    self._consec_losses = 0
                    # C2: After circuit breaker fires, require higher confidence for 2h
                    self._elevated_confidence_until = time.time() + 7200
                    self.log.warning(
                        f"[CircuitBreaker] 3 consecutive active losses — "
                        f"pausing active trading for {pause_secs // 60} min, "
                        f"elevated confidence threshold (0.72) for 2h"
                    )
                    self.alerting.halt(
                        f"Circuit breaker: 3 consecutive losses\n"
                        f"Active paused 60 min. Equity={round(self.active_portfolio.equity,2)}"
                    )
            else:
                self._consec_losses = 0  # reset on any win

        # Per-strategy post-close hooks
        if position["strategy"] == "active":
            self.active_strategy.record_daily_pnl(pnl)
            self.daily_profit.record_pnl(pnl)
            self.cross_bridge.clear_active_position(position["symbol"])
            # APC: record outcome to adapt confidence threshold + capital sizing
            try:
                self.apc.record_trade(
                    pnl,
                    mode=position.get("signal_mode", "unknown"),
                    regime=position.get("regime", "unknown"),
                )
            except Exception:
                pass

        # Cross-engine: record PnL per engine + flag avoid on stop-outs
        self.cross_bridge.record_pnl(position["strategy"], pnl)
        if "STOP" in reason or "TRAIL" in reason:
            self.cross_bridge.flag_avoid(position["symbol"], position["strategy"])

        # Bandit feedback: record outcome for the mode that was used
        regime = position.get("regime", "sideways") or "sideways"
        if position["strategy"] == "active":
            mode = position.get("signal_mode", "")
            if mode:
                self.strategy_bandit.record_active(regime, mode, won=(pnl >= 0))
        elif position["strategy"] == "passive":
            mode = position.get("passive_mode", "long")
            self.strategy_bandit.record_passive(regime, mode, won=(pnl >= 0))

        # Online learning: extract current-time features and update weights
        features = self._extract_features_now(position["symbol"])
        if features is not None:
            try:
                self.online_learner.update(features, won=(pnl >= 0))
            except Exception:
                pass

        # Release trade manager tracking
        self.trade_manager.release(position["symbol"])

        self._persist_state()
        self.log.info(
            f"{position['strategy'].upper()} EXIT | {position['symbol']} | "
            f"{position['side']} | PnL={round(pnl, 2)}"
            + (f" [{reason}]" if reason else "")
        )
        return pnl

    def _close_half_position(
        self,
        portfolio: PortfolioEngine,
        position: dict,
        reason: str = "",
    ) -> None:
        """Close ~50% of the position size, leave the rest to trail."""
        half_size = round(position["size"] / 2, 8)
        if half_size <= 0:
            return

        # Compute PnL on half the size by temporarily patching position["size"]
        original_size = position["size"]
        position["size"] = half_size
        pnl = self.broker.close_position(position)
        position["size"] = round(original_size - half_size, 8)

        # Return half the margin + PnL to cash WITHOUT removing the position
        half_value  = round(position["position_value"] / 2, 2)
        leverage    = float(position.get("leverage", 1)) or 1.0
        margin      = position.get("margin", position["position_value"] / leverage)
        position["position_value"] = half_value
        half_margin = round(margin / 2, 4)
        position["margin"] = half_margin

        portfolio.cash         += half_margin + pnl
        portfolio.realized_pnl += pnl
        portfolio.update_equity()

        self.risk_manager.record_trade(position["strategy"], pnl)
        if position["strategy"] == "active":
            self.active_strategy.record_daily_pnl(pnl)

        self.log.info(
            f"{position['strategy'].upper()} PARTIAL EXIT | {position['symbol']} | "
            f"{position['side']} | half_pnl={round(pnl, 2)} [{reason}]"
        )

    def _strategy_position_cap(self, strategy: str) -> float:
        total = self.passive_portfolio.equity + self.active_portfolio.equity
        max_pos = PASSIVE_MAX_POSITIONS if strategy == "passive" else ACTIVE_MAX_POSITIONS
        return self.capital_allocator.position_cap(strategy, max_pos, total)

    def _check_strategy_conflict(self, strategy: str, signal: dict) -> bool:
        decision = self.risk_manager.check_conflict(
            strategy, signal,
            self.passive_portfolio.positions,
            self.active_portfolio.positions,
        )
        if decision["allow_trade"]:
            return False
        symbol = signal.get("symbol", "-")
        self._record_conflict(symbol, decision["reason"])
        # Rate-limit: log each symbol at most once every 10 minutes
        now = time.time()
        if now - self._conflict_logged.get(symbol, 0) >= 600:
            self.log.info(f"CONFLICT BLOCK | {decision['reason']}")
            self._conflict_logged[symbol] = now
        return True

    def _best_active_mode(self, regime_mode: str) -> str:
        """
        Return the best-performing recent active mode.
        Sticks with the regime recommendation unless it's clearly losing AND
        another mode has significantly better recent win rate.
        Requires ≥ 5 trades in a mode before considering it as an alternative.
        """
        history = self._mode_stats.get(regime_mode, deque())
        if len(history) < 5:
            return regime_mode   # not enough data → trust regime

        regime_wr = sum(history) / len(history)
        if regime_wr >= 0.40:
            return regime_mode   # regime mode holding up → keep it

        best_mode, best_wr = regime_mode, regime_wr
        for mode, hist in self._mode_stats.items():
            if mode == regime_mode or len(hist) < 5:
                continue
            wr = sum(hist) / len(hist)
            if wr > best_wr:
                best_wr, best_mode = wr, mode

        if best_mode != regime_mode:
            self.log.info(
                f"[AdaptiveMode] {regime_mode}({regime_wr:.0%}) → {best_mode}({best_wr:.0%})"
            )
        return best_mode

    def _compound_size_factor(self) -> float:
        """
        Grows position sizes as portfolio equity compounds above initial capital.

        This is the engine of exponential growth:
          equity = initial × 1.0  →  factor = 1.00  (no change at start)
          equity = initial × 1.5  →  factor = 1.22  (+22% size after +50% gain)
          equity = initial × 2.0  →  factor = 1.41  (+41% size after doubling)
          equity = initial × 4.0  →  factor = 2.00  (cap — never more than 2×)

        Uses square-root scaling so growth is sustainable, not runaway.
        Only scales UP (never below 1.0); DrawdownGuard handles the downside.

        BTC cycle phase adjustment applied on top:
          bull/early_bull  → up to +15% extra (upside participation)
          distribution     → -10% (risk-off as market may top)
          bear             → -20% (capital preservation)
          accumulation     → -5%  (slight caution near cycle bottom)
        """
        initial = max(1.0, self.passive_portfolio.initial_cash + self.active_portfolio.initial_cash)
        equity  = self.passive_portfolio.equity + self.active_portfolio.equity
        ratio   = equity / initial
        base    = 1.0 if ratio <= 1.0 else round(min(2.0, ratio ** 0.5), 3)

        # BTC cycle multiplier
        cycle = (self._last_intel or {}).get("btc_cycle", {})
        phase = cycle.get("cycle_phase", "unknown")
        cycle_mult = {
            "early_bull":   1.15,
            "bull":         1.10,
            "distribution": 0.90,
            "bear":         0.80,
            "accumulation": 0.95,
        }.get(phase, 1.0)

        return round(base * cycle_mult, 3)

    def _drawdown_size_factor(self) -> float:
        """
        Reduce new position sizes automatically when portfolios are in drawdown.
        Uses each portfolio's own high-water mark drawdown.
          >15% drawdown → 50% of normal size
          10-15%        → 70%
           5-10%        → 85%
          <5%           → no reduction
        """
        dd = max(
            self.passive_portfolio.current_drawdown(),
            self.active_portfolio.current_drawdown(),
        )
        if dd >= 0.15:
            return 0.50
        if dd >= 0.10:
            return 0.70
        if dd >= 0.05:
            return 0.85
        return 1.0

    def _apply_drawdown_healing(self, route: dict, dd_factor: float) -> dict:
        """
        Self-healing: restrict active modes and passive entries in drawdown.

        dd_factor reflects portfolio drawdown:
          1.00 → <5%  drawdown  (normal)
          0.85 → 5-10% drawdown (mild)
          0.70 → 10-15% drawdown (moderate — healing mode)
          0.50 → ≥15% drawdown  (severe — survival mode)

        In moderate+ drawdown, force the active strategy into its two safest modes
        (funding_arb + mean_reversion) since these have defined-edge setups and
        don't require strong directional LGBM predictions.
        In severe drawdown, also skip the passive strategy to stop capital bleed.
        """
        if dd_factor > 0.85:
            return route  # <5% drawdown — no healing needed

        route = dict(route)
        safe_active = {"funding_arb", "mean_reversion"}

        if dd_factor <= 0.70:  # 10-15% drawdown
            if route.get("active_mode") not in safe_active:
                old_mode = route["active_mode"]
                route["active_mode"] = "mean_reversion"
                if self._should_print_cycle():
                    self.log.info(
                        f"[DrawdownHeal] dd≥10% — overriding {old_mode} → mean_reversion"
                    )

        if dd_factor <= 0.50:  # ≥15% drawdown — survival mode
            if route.get("active_mode") not in safe_active:
                route["active_mode"] = "funding_arb"
            route["passive_mode"] = "skip"
            if self._should_print_cycle():
                self.log.warning(
                    "[DrawdownHeal] dd≥15% SURVIVAL MODE — passive=skip, active=funding_arb only"
                )

        return route

    def _daily_pause_loss_threshold(self) -> float:
        """Daily active P&L floor that pauses new entries for the rest of the day.

        Defaults to -1% of current active equity (env override takes priority),
        so the gate scales with compounding instead of staying pinned to the
        $300 seed capital.
        """
        if self._daily_pause_loss_override < 0:
            return self._daily_pause_loss_override
        return round(-self.active_portfolio.equity * 0.01, 2)

    def _daily_protect_profit_threshold(self) -> float:
        """Daily active P&L ceiling that switches to funding_arb-only "protect" mode.

        Defaults to +2% of current active equity (env override takes priority),
        so the gate scales with compounding instead of staying pinned to the
        $300 seed capital.
        """
        if self._daily_protect_profit_override > 0:
            return self._daily_protect_profit_override
        return round(self.active_portfolio.equity * 0.02, 2)

    def _enforce_active_daily_loss_protection(self) -> None:
        """
        When the daily active loss cap is hit, tighten all open active stops to
        near-breakeven.  This prevents existing losers from digging the hole deeper
        — the risk manager already blocks NEW entries, but this closes the back door.
        """
        if self.risk_manager.daily_loss_active < self.risk_manager.active_daily_loss_cap:
            return
        tightened = 0
        for pos in self.active_portfolio.positions:
            entry = pos["entry_price"]
            if pos["side"] == "LONG":
                new_sl = round(entry * 0.9998, 6)   # 0.02% below entry (covers spread)
                if (pos.get("stop_loss") or 0.0) < new_sl:
                    pos["stop_loss"] = new_sl
                    tightened += 1
            else:
                new_sl = round(entry * 1.0002, 6)   # 0.02% above entry
                if (pos.get("stop_loss") or float("inf")) > new_sl:
                    pos["stop_loss"] = new_sl
                    tightened += 1
        if tightened:
            self.log.warning(
                f"[DailyLoss] cap hit (loss={self.risk_manager.daily_loss_active:.2f} "
                f">= {self.risk_manager.active_daily_loss_cap:.2f}) — "
                f"tightened {tightened} active stop(s) to breakeven"
            )

    def _rebalance_capital(self) -> None:
        status = self.capital_allocator.rebalance(
            self.cycle,
            self.passive_portfolio,
            self.active_portfolio,
            self.risk_manager.get_strategy_stats(),
        )
        transfer = status.get("last_transfer", {})
        if transfer:
            self.log.info(
                f"CAPITAL REALLOCATED | {transfer['from']} -> {transfer['to']} "
                f"| amount={transfer['amount']}"
            )
        # Keep daily profit engine, active strategy's daily gate, and the
        # risk manager's daily-loss backstop in sync with current active capital
        self.daily_profit.update_capital(self.active_portfolio.equity)
        self.active_strategy.update_capital(self.active_portfolio.equity)
        self.risk_manager.update_capital(self.active_portfolio.equity)

    # ── position management ─────────────────────────────────────────────────────

    def _should_close_position(self, position: dict) -> tuple[bool, str]:
        """Returns (should_close, reason)."""
        current = position.get("current_price", position["entry_price"])
        side    = position["side"]

        if side == "LONG":
            if position.get("stop_loss") and current <= position["stop_loss"]:
                return True, "STOP_LOSS"
            if position.get("take_profit") and current >= position["take_profit"]:
                if position.get("trailing_active"):
                    return False, ""   # TradeManager already moved TP to TP2
                return True, "TAKE_PROFIT"
        else:
            if position.get("stop_loss") and current >= position["stop_loss"]:
                return True, "STOP_LOSS"
            if position.get("take_profit") and current <= position["take_profit"]:
                if position.get("trailing_active"):
                    return False, ""
                return True, "TAKE_PROFIT"

        age_hours = (time.time() - position["opened_at"]) / 3600
        max_hold  = position.get("max_hold_hours", 0)
        if max_hold > 0 and age_hours >= max_hold:
            return True, "MAX_HOLD"

        if (position["strategy"] == "active"
                and self.active_strategy.trading_window_has_closed()):
            return True, "EOD_CLOSE"

        return False, ""

    def _manage_open_positions(self) -> None:
        """Update prices, apply trailing stops, and close positions as needed."""
        for portfolio in (self.passive_portfolio, self.active_portfolio):
            for position in list(portfolio.positions):
                self._update_position_price(portfolio, position)

                # Skip if force-closed inside _update_position_price
                if position not in portfolio.positions:
                    continue

                # Stale passive: open > 5 days and price moved < 1% from entry
                if position.get("strategy") == "passive":
                    age_hours = (time.time() - position["opened_at"]) / 3600
                    if age_hours >= 120:
                        move_pct = abs(
                            position.get("current_price", position["entry_price"])
                            - position["entry_price"]
                        ) / position["entry_price"]
                        if move_pct < 0.01:
                            self._close_position(portfolio, position, reason="STALE")
                            continue

                # TradeManager: trailing stop + partial profit
                action = self.trade_manager.tick(position)
                if action["action"] == "close":
                    self._close_position(portfolio, position, reason=action.get("reason", "TRAIL"))
                    continue
                if action["action"] == "close_partial":
                    self._close_half_position(portfolio, position, reason=action.get("reason", "TP1"))
                    # Position is still open (halved) — continue managing it

                # Funding arb exit: close when rate normalises
                if (position.get("signal_mode") == "funding_arb"
                        and position.get("strategy") == "active"):
                    try:
                        if self.scanner.funding_arb.should_exit(position["symbol"]):
                            self._close_position(
                                portfolio, position, reason="FUNDING_NORMALISED"
                            )
                            continue
                    except Exception:
                        pass

                # Classic TP/SL/MaxHold/EOD check
                should_close, reason = self._should_close_position(position)
                if should_close:
                    self._close_position(portfolio, position, reason=reason)

    # ── position opening ────────────────────────────────────────────────────────

    def _build_position_size(
        self, portfolio: PortfolioEngine, max_size_value: float, price: float
    ) -> float:
        if portfolio.cash <= 0 or price <= 0:
            return 0.0
        return round(min(portfolio.cash, max_size_value) / price, 4)

    def _open_position(
        self,
        portfolio: PortfolioEngine,
        signal: dict,
        strategy: str,
    ) -> bool:
        price = signal["entry_price"]
        size  = self._build_position_size(
            portfolio, signal.get("max_position_value", 0.0), price
        )
        if size <= 0:
            return False

        fill = self.broker.execute_order(signal["symbol"], signal["side"], size)
        if fill is None:
            return False

        position = self._position_data(signal, fill, strategy)

        # Register with TradeManager for trailing stop tracking
        self.trade_manager.register(position["symbol"], position["entry_price"], position["side"])

        portfolio.add_position(position)
        self.log.info(
            f"{strategy.upper()} OPEN | {signal['symbol']} | {signal['side']} | "
            f"size={size} | entry={fill['price']} | conf={signal.get('confidence', 0)}"
        )
        self.alerting.trade_open(
            f"[{strategy}] {signal['symbol']}",
            signal["side"], fill["price"], size, portfolio.equity,
        )
        return True

    def _open_passive_trades(
        self,
        symbols: list[str],
        market_state: dict,
        passive_mode: str = "auto",
        size_mult: float = 1.0,
    ) -> None:
        if len(self.passive_portfolio.positions) >= PASSIVE_MAX_POSITIONS:
            return
        if passive_mode == "skip":
            return

        intel = self._last_intel

        for symbol in symbols:
            if self.passive_portfolio.has_open_position(symbol):
                continue
            if self._in_cooldown(symbol, "passive"):
                continue

            signal = self.passive_strategy.generate_signal(
                symbol, market_state,
                mode=passive_mode,
                market_intelligence=intel,
            )
            signal["passive_mode"] = passive_mode   # tag for bandit feedback
            if signal["side"] == "HOLD":
                continue

            if self._check_strategy_conflict("passive", signal):
                continue

            # A3: Correlation guard — don't pile into 3+ positions in the same direction
            if self._same_direction_count(signal["side"]) >= 3:
                self._record_conflict(symbol, f"DIR_CROWD: already 3+ {signal['side']} across both portfolios")
                continue

            held = [p["symbol"] for p in self.passive_portfolio.positions]
            safe, corr_reason = self.correlation_filter.is_safe(
                symbol, held, self.market_state.market
            )
            if not safe:
                self._record_conflict(symbol, f"CORR: {corr_reason}")
                continue

            # ML confirmation: LGBM primary, logistic fallback
            # Only applied when model accuracy >= 0.50; below that it's noise.
            try:
                df  = self.market_state.market.get_ohlcv_df(symbol, timeframe="1d", limit=200)
                regime_key = market_state.get("regime", "global") or "global"
                if self.lgbm.trained:
                    ml = self.lgbm.predict(df["close"].values, df["volume"].values, regime=regime_key)
                else:
                    ml = self.ml_predictor.predict(df["close"].values, df["volume"].values)
                if ml.get("accuracy", 0.0) >= 0.50:
                    if ml["direction"] not in ("HOLD", signal["side"]):
                        signal["confidence"] = round(signal["confidence"] * (1.0 - ml["confidence"] * 0.3), 4)
                    elif ml["direction"] == signal["side"] and ml["confidence"] > 0.55:
                        signal["confidence"] = round(min(signal["confidence"] * 1.05, 0.95), 4)
            except Exception:
                pass

            base_cap  = self._strategy_position_cap("passive")
            edge_mult = self.edge_tracker.edge_mult(symbol)
            decision  = self.risk_manager.allow_passive(
                self.passive_portfolio, signal, market_state,
                max_position_value=base_cap * size_mult * edge_mult,
            )
            if not decision["allow_trade"]:
                continue

            signal["max_position_value"] = decision.get("max_position_value", 0.0)
            self._open_position(self.passive_portfolio, signal, "passive")
            if len(self.passive_portfolio.positions) >= PASSIVE_MAX_POSITIONS:
                break

    @staticmethod
    def _market_health_ok(market_state: dict) -> bool:
        """
        Skip all new entries when macro conditions are dangerous:
          - Volatility extreme (>85th pctile) AND fear/greed < 15 = panic crash
          - BTC 24h drop > 8% = black-swan move, wait for stabilisation
        """
        vol = float(market_state.get("volatility", 0.5))
        btc_change = float(market_state.get("btc_change_24h", 0.0))
        if vol > 0.85 and btc_change < -0.05:
            return False
        if btc_change < -0.08:   # BTC -8% in 24h = extreme event
            return False
        return True

    def _open_active_trades(
        self,
        symbols: list[str],
        market_state: dict,
        active_mode: str = "mean_reversion",
        size_mult: float = 1.0,
    ) -> None:
        # A2: Session-based position cap (Asian session: max 1 active position)
        session_cap = self._session_max_active_positions()
        if len(self.active_portfolio.positions) >= session_cap:
            return
        if active_mode == "skip":
            return

        # Market health gate: hold off during extreme macro events
        if not self._market_health_ok(market_state):
            if self._should_print_cycle():
                self.log.info("[MarketHealth] extreme conditions — skipping new active entries")
            return

        # AI Brain pause gate
        if time.time() < self._active_paused_until:
            return

        # C1: Minimum trade spacing — wait at least 2 min after last active open
        if time.time() - self._last_active_open_ts < 120:
            return

        # C3: Daily PnL gates
        _today = datetime.now(timezone.utc).date()
        if self._protect_mode_date != _today:
            # New day: reset daily flags AND active strategy's pnl counter
            self._active_daily_paused_today = False
            self._protect_mode_date = _today
            if hasattr(self.active_strategy, "_daily_pnl"):
                self.active_strategy._daily_pnl  = 0.0
                self.active_strategy._daily_date = _today
        _today_pnl = self.active_strategy._daily_pnl if hasattr(self.active_strategy, "_daily_pnl") else 0.0
        _pause_loss = self._daily_pause_loss_threshold()
        if _today_pnl <= _pause_loss:
            # C3a: daily loss ≥1% of active equity → pause active for rest of day
            if not self._active_daily_paused_today:
                self._active_daily_paused_today = True
                self.log.warning(
                    f"[DailyPnLGate] daily_pnl={_today_pnl:.2f} ≤ {_pause_loss:.2f} "
                    f"— pausing active for rest of day"
                )
            return
        _protect_profit = self._daily_protect_profit_threshold()
        _protect_only_arb = (
            _today_pnl >= _protect_profit
            and active_mode != "funding_arb"
        )
        if _protect_only_arb:
            # C3b: daily profit ≥2% of active equity → protect mode, only funding_arb allowed
            if self._should_print_cycle():
                self.log.info(
                    f"[DailyPnLGate] daily_pnl={_today_pnl:.2f} ≥ {_protect_profit:.2f} "
                    f"— protect mode, only funding_arb allowed"
                )
            return

        # LGBM accuracy gate: when model is worse than ~random, only allow
        # funding_arb (which relies on funding extremes, not direction prediction).
        regime_key = market_state.get("regime", "global") or "global"
        lgbm_acc = self.lgbm.get_accuracy(regime_key) or self.lgbm.get_accuracy("global")
        if lgbm_acc > 0 and lgbm_acc < 0.45 and active_mode != "funding_arb":
            if self._should_print_cycle():
                self.log.info(
                    f"[LGBM] acc={lgbm_acc:.3f} < 0.45 — pausing {active_mode} "
                    f"(funding_arb still allowed)"
                )
            return

        # Daily profit engine gate: stop new entries when target reached
        if not self.daily_profit.should_open_new():
            if self._should_print_cycle():
                self.log.info("[DailyProfit] target LOCKED — no new active entries today")
            return
        size_mult = round(size_mult * self.daily_profit.get_aggression_mult(), 3)

        intel = self._last_intel
        new_this_cycle = 0
        MAX_NEW_PER_CYCLE = 2  # prevent opening 4 correlated positions at once

        for symbol in symbols:
            if new_this_cycle >= MAX_NEW_PER_CYCLE:
                break
            if self.active_portfolio.has_open_position(symbol):
                continue
            if self._in_cooldown(symbol, "active"):
                continue

            signal = self.active_strategy.generate_signal(
                symbol, market_state,
                mode=active_mode,
                market_intelligence=intel,
            )
            signal["signal_mode"] = active_mode   # tag for adaptive tracking
            if signal["side"] == "HOLD":
                continue

            # B1: EV gate — skip modes with proven negative EV (≥15 trades)
            ev = self._mode_ev(active_mode, market_state.get("regime", ""))
            if ev < -0.002:
                if self._should_print_cycle():
                    self.log.info(
                        f"[EVGate] mode={active_mode} EV={ev:.4f} < -0.002 — skipping"
                    )
                continue

            # APC: use dynamic confidence threshold (adapts based on rolling win rate)
            _apc_min_conf = self.apc.get_min_confidence()
            if signal.get("confidence", 0) < _apc_min_conf:
                continue

            # C2: Elevated confidence gate after circuit-breaker pause
            if time.time() < self._elevated_confidence_until:
                if signal.get("confidence", 0) < 0.72:
                    continue   # require higher bar while in recovery window

            # A3: Correlation guard — don't open a 4th same-direction position
            if self._same_direction_count(signal["side"]) >= 3:
                self._record_conflict(signal["symbol"], f"DIR_CROWD: already 3+ {signal['side']} across both portfolios")
                continue

            # Regime direction guard: block counter-trend signals
            _regime_str = (market_state.get("regime") or "").lower()
            _bear_regime = "bear" in _regime_str or "sideways" in _regime_str
            _bull_regime = "bull" in _regime_str
            if active_mode == "momentum":
                if _bear_regime and signal["side"] == "LONG":
                    continue   # no momentum LONGs in bear/sideways
                if _bull_regime and signal["side"] == "SHORT":
                    continue   # no momentum SHORTs in bull
            if active_mode == "breakout":
                if _bear_regime and signal["side"] == "LONG":
                    continue   # breakout LONGs in bear frequently fail
                if _bull_regime and signal["side"] == "SHORT":
                    continue
            if active_mode == "funding_arb":
                # Funding arb counter-trend LONGs in bear are risky unless rate is
                # STRONG+ (>= 0.10%/8h). At MODERATE (0.05%), funding income won't
                # cover a stop-loss in a trending bear market.
                if _bear_regime and signal["side"] == "LONG":
                    from strategy.funding_arb import THRESHOLD_STRONG
                    fr = abs(signal.get("funding_rate", 0.0))
                    if fr < THRESHOLD_STRONG:
                        continue   # too small to justify counter-trend LONG
                if _bull_regime and signal["side"] == "SHORT":
                    from strategy.funding_arb import THRESHOLD_STRONG
                    fr = abs(signal.get("funding_rate", 0.0))
                    if fr < THRESHOLD_STRONG:
                        continue   # too small to justify counter-trend SHORT

            if self._check_strategy_conflict("active", signal):
                continue

            # ML confidence filter: LGBM on 15M bars (same timeframe as active signals)
            # Only applied when model accuracy >= 0.50; below that it's noise.
            try:
                df_ml = self.market_state.market.get_ohlcv_df(symbol, timeframe="15m", limit=100)
                if self.lgbm.trained:
                    ml = self.lgbm.predict(df_ml["close"].values, df_ml["volume"].values, regime=regime_key)
                    # Feed live features into drift detector.
                    # Don't use signal agreement as "correct" — that compares model vs strategy,
                    # not model vs actual price. Only update features (not accuracy) here.
                    # Accuracy is tracked separately when positions close.
                    from models.lgbm_predictor import extract_features
                    feats = extract_features(df_ml["close"].values, df_ml["volume"].values)
                    self.drift_detector.update(feats, correct=None)
                else:
                    ml = self.ml_predictor.predict(df_ml["close"].values, df_ml["volume"].values)
                if ml.get("accuracy", 0.0) >= 0.50:
                    if ml["direction"] not in ("HOLD", signal["side"]):
                        signal["confidence"] = round(signal["confidence"] * (1.0 - ml["confidence"] * 0.3), 4)
                    elif ml["direction"] == signal["side"] and ml["confidence"] > 0.55:
                        signal["confidence"] = round(min(signal["confidence"] * 1.05, 0.95), 4)
            except Exception:
                pass

            # Multi-timeframe 1H confirmation: agree = +8% confidence, disagree = -12%
            try:
                df_1h = self.market_state.market.get_ohlcv_df(symbol, timeframe="1h", limit=120)
                if len(df_1h) >= 60 and self.lgbm.trained:
                    ml_1h = self.lgbm.predict(df_1h["close"].values, df_1h["volume"].values, regime=regime_key)
                    if ml_1h.get("accuracy", 0.0) >= 0.50 and ml_1h["direction"] != "HOLD":
                        if ml_1h["direction"] == signal["side"]:
                            signal["confidence"] = round(min(signal["confidence"] * 1.08, 0.95), 4)
                        else:
                            signal["confidence"] = round(signal["confidence"] * 0.88, 4)
            except Exception:
                pass

            # B3: Multi-timeframe 5m confluence — if 5m EMAs disagree with signal
            # direction AND confidence < 0.75, reduce confidence by 15%.
            try:
                import numpy as _np5m
                df_5m = self.market_state.market.get_ohlcv_df(symbol, timeframe="5m", limit=60)
                if len(df_5m) >= 30:
                    closes_5m = df_5m["close"].values
                    # Weighted EMA helper (same as active_strategy._ema)
                    def _ema_weighted(vals, period):
                        if len(vals) < period:
                            return float(vals[-1])
                        weights = _np5m.exp(_np5m.linspace(-1.0, 0.0, period))
                        weights /= weights.sum()
                        return float(_np5m.convolve(vals[-period:], weights, mode="valid")[-1])
                    ema_fast_5m = _ema_weighted(closes_5m, 9)
                    ema_slow_5m = _ema_weighted(closes_5m, 21)
                    five_m_bull = ema_fast_5m > ema_slow_5m
                    five_m_bear = ema_fast_5m < ema_slow_5m
                    sig_long = signal["side"] == "LONG"
                    sig_short = signal["side"] == "SHORT"
                    # Disagree = 5m says opposite direction
                    disagrees = (sig_long and five_m_bear) or (sig_short and five_m_bull)
                    if disagrees and signal.get("confidence", 1.0) < 0.75:
                        signal["confidence"] = round(signal["confidence"] * 0.85, 4)
            except Exception:
                pass

            # ── Momentum Quality Gates (A, B, C) — only for momentum mode ──────
            # Thresholds tuned for bear-market SHORT signals:
            # Gate A: 1.3x volume (above avg, not requiring a surge — SHORTs often don't spike)
            # Gate B: 3.0 ATR from EMA21 (widened — bear moves can extend further)
            # Gate C: 1h alignment penalty (-0.12 conf), hard-skip only if final conf < 0.65
            if active_mode == "momentum":
                try:
                    import numpy as _npq
                    df_q = self.market_state.market.get_ohlcv_df(symbol, timeframe="15m", limit=30)
                    if len(df_q) >= 22:
                        _closes_q = df_q["close"].values
                        _vols_q   = df_q["volume"].values

                        # Gate A — Volume Confirmation (1.3x avg)
                        _vol_avg = float(_npq.mean(_vols_q[-21:-1]))
                        _vol_cur = float(_vols_q[-1])
                        if _vol_avg > 0 and _vol_cur < 1.3 * _vol_avg:
                            continue

                        # Gate B — Not Overextended (3.0 ATR from EMA21)
                        _price_q = float(_closes_q[-1])
                        _ema21_w = _npq.exp(_npq.linspace(-1.0, 0.0, 21))
                        _ema21_w /= _ema21_w.sum()
                        _ema21_q = float(_npq.convolve(_closes_q[-21:], _ema21_w, mode="valid")[-1])
                        _rets_q  = _npq.diff(_closes_q[-15:]) / (_closes_q[-15:-1] + 1e-9)
                        _atr_q   = float(_npq.std(_rets_q)) * _price_q if len(_rets_q) > 1 else 0.0
                        if _atr_q > 0 and abs(_price_q - _ema21_q) > 3.0 * _atr_q:
                            continue

                        # Gate C — 1h Trend Alignment (penalty, not hard skip)
                        try:
                            _df_1h_q = self.market_state.market.get_ohlcv_df(symbol, "1h", 50)
                            if len(_df_1h_q) >= 21:
                                _c1h  = _df_1h_q["close"].values
                                _w9   = _npq.exp(_npq.linspace(-1.0, 0.0, 9));  _w9  /= _w9.sum()
                                _w21q = _npq.exp(_npq.linspace(-1.0, 0.0, 21)); _w21q /= _w21q.sum()
                                _ema9_1h  = float(_npq.convolve(_c1h[-9:],  _w9,   mode="valid")[-1])
                                _ema21_1h = float(_npq.convolve(_c1h[-21:], _w21q, mode="valid")[-1])
                                _1h_disagrees = (
                                    (signal["side"] == "LONG"  and _ema9_1h < _ema21_1h)
                                    or (signal["side"] == "SHORT" and _ema9_1h > _ema21_1h)
                                )
                                if _1h_disagrees:
                                    signal["confidence"] = round(
                                        signal.get("confidence", 0.65) - 0.12, 4
                                    )
                                    if signal["confidence"] < 0.65:
                                        continue   # after penalty below MIN_CONFIDENCE
                        except Exception:
                            pass  # 1h fetch failed — let signal through
                except Exception:
                    pass  # gate calc failed — don't block the trade

            # Order book imbalance: adjust confidence based on bid/ask pressure
            try:
                exchange = self.broker._exchange if hasattr(self.broker, "_exchange") else None
                if exchange is None:
                    from data.binance_client import BinanceClient
                    exchange = BinanceClient().get_exchange()
                ob = self.ob_signal.get_signal(exchange, symbol)
                ob_delta = self.ob_signal.confidence_adjustment(ob, signal["side"])
                if ob_delta != 0.0:
                    signal["confidence"] = round(
                        float(min(max(signal["confidence"] + ob_delta, 0.30), 0.95)), 4
                    )
                # Update cross-bridge active position tracking
                if signal["side"] in ("LONG", "SHORT"):
                    self.cross_bridge.update_active_position(symbol, signal["side"])
            except Exception:
                pass

            # ── Market Microstructure: OI + Long/Short Ratio + Taker Flow ────────
            # These are the signals professional quant funds use.
            # All free via Binance USDT-M API.
            try:
                exchange_ms = self.broker._exchange if hasattr(self.broker, "_exchange") else None
                if exchange_ms is None:
                    from data.binance_client import BinanceClient
                    exchange_ms = BinanceClient().get_exchange()
                ms = self.microstructure.get_signals(exchange_ms, symbol)
                ms_delta = self.microstructure.directional_boost(ms, signal["side"])
                if ms_delta != 0.0:
                    signal["confidence"] = round(
                        float(min(max(signal["confidence"] + ms_delta, 0.30), 0.95)), 4
                    )
                    signal["microstructure"] = ms
            except Exception:
                pass

            # ── News/Event Detection ──────────────────────────────────────────────
            # Check once per 10 min (cached), pause trading if high-impact event.
            # Log the warning only ONCE per news update, not every cycle.
            try:
                now_ns = time.time()
                if now_ns - self._news_check_ts > 600:
                    prev_pause = (self._last_news or {}).get("should_pause", False)
                    self._last_news     = self.news_feed.get_news_signal()
                    self._news_check_ts = now_ns
                    # Log only on transition to pause state
                    if self._last_news.get("should_pause") and not prev_pause:
                        self.log.warning(
                            f"[News] High-impact event — pausing new entries: "
                            f"{self._last_news.get('top_headline','')[:80]}"
                        )
                    elif not self._last_news.get("should_pause") and prev_pause:
                        self.log.info("[News] Event cleared — resuming new entries")
                if self._last_news:
                    if self._last_news.get("should_pause"):
                        continue   # silent skip — already logged on transition
                    news_delta = self.news_feed.confidence_adjustment(
                        self._last_news, signal["side"], symbol
                    )
                    if news_delta != 0.0:
                        signal["confidence"] = round(
                            float(min(max(signal["confidence"] + news_delta, 0.30), 0.95)), 4
                        )
            except Exception:
                pass

            # Funding rate cross-check: fade crowded side, boost contrarian edge
            try:
                from data.funding_data import FundingData
                fr = float(FundingData().get_funding(symbol).get("funding_rate", 0.0))
                if signal["side"] == "LONG":
                    if fr > 0.001:    # longs very crowded → fade long confidence
                        signal["confidence"] = round(signal["confidence"] * 0.88, 4)
                    elif fr < -0.0005:  # shorts crowded → long has funding edge
                        signal["confidence"] = round(min(signal["confidence"] * 1.10, 0.95), 4)
                else:  # SHORT
                    if fr < -0.0005:  # shorts very crowded → fade short confidence
                        signal["confidence"] = round(signal["confidence"] * 0.88, 4)
                    elif fr > 0.001:   # longs crowded → short has funding edge
                        signal["confidence"] = round(min(signal["confidence"] * 1.10, 0.95), 4)
            except Exception:
                pass

            # R:R gate — never enter a trade where reward < 2x risk
            # At momentum bear WR=40%: need TP > 1.5x SL to break even.
            # 2.0x gives margin of safety. Below this, expected value is negative.
            sig_tp = signal.get("take_profit") or 0
            sig_sl = signal.get("stop_loss")   or 0
            if sig_tp > 0 and sig_sl > 0:
                sig_entry = signal.get("entry_price") or 1.0
                tp_pct = abs(sig_tp - sig_entry) / sig_entry
                sl_pct = abs(sig_sl - sig_entry) / sig_entry
                rr = tp_pct / sl_pct if sl_pct > 0 else 0
                if rr < 2.0:
                    continue   # negative EV at current win rates

            base_cap  = self._strategy_position_cap("active")
            edge_mult = self.edge_tracker.edge_mult(symbol)
            decision  = self.risk_manager.allow_active(
                self.active_portfolio, signal, market_state,
                max_position_value=base_cap * size_mult * edge_mult,
            )
            if not decision["allow_trade"]:
                continue

            # Confidence-weighted position size: scale ±20% based on signal quality
            max_val   = decision.get("max_position_value", 0.0)
            conf      = signal.get("confidence", 0.65)
            conf_mult = 0.70 + (conf - 0.50) * 1.5   # 0.5 conf → 0.75×, 0.9 conf → 1.30×
            max_val   = round(max_val * float(min(1.35, max(0.60, conf_mult))), 2)

            # ATR-normalized sizing: halve position in high-vol environments
            # so risk-per-trade stays constant regardless of market turbulence.
            try:
                df_atr = self.market_state.market.get_ohlcv_df(symbol, "15m", limit=30)
                if len(df_atr) >= 15:
                    import numpy as _np
                    _rets = _np.diff(df_atr["close"].values[-21:]) / df_atr["close"].values[-21:-1]
                    atr_now = float(_np.std(_rets)) if len(_rets) > 1 else 0.005
                    # Baseline ATR for crypto 15m ≈ 0.005 (0.5%)
                    # If ATR is 2× baseline → halve position; if 0.5× → can scale up slightly
                    atr_mult = float(_np.clip(0.005 / (atr_now + 1e-9), 0.50, 1.20))
                    max_val  = round(max_val * atr_mult, 2)
            except Exception:
                pass

            # B2: ATR-based adaptive stop-loss for active strategy
            # adaptive_sl = min(fixed_sl, max(0.003, atr_pct * 1.5))
            # Widens stops in high-vol, tightens in low-vol
            try:
                import numpy as _npb2
                df_b2 = self.market_state.market.get_ohlcv_df(symbol, timeframe="15m", limit=20)
                if len(df_b2) >= 15:
                    closes_b2 = df_b2["close"].values
                    highs_b2  = df_b2["high"].values
                    lows_b2   = df_b2["low"].values
                    price_b2  = float(closes_b2[-1])
                    # True range ATR
                    tr_arr = _npb2.maximum(
                        highs_b2[1:] - lows_b2[1:],
                        _npb2.abs(highs_b2[1:] - closes_b2[:-1]),
                        _npb2.abs(lows_b2[1:]  - closes_b2[:-1]),
                    )
                    atr_pct = float(_npb2.mean(tr_arr[-14:])) / (price_b2 + 1e-9)
                    from config.dual_settings import ACTIVE_STOP_LOSS as _ASL
                    adaptive_sl = min(_ASL, max(0.003, atr_pct * 1.5))
                    # Recompute stop price from signal entry using adaptive SL
                    entry = signal.get("entry_price", price_b2)
                    if signal["side"] == "LONG":
                        signal["stop_loss"] = round(entry * (1 - adaptive_sl), 6)
                    else:
                        signal["stop_loss"] = round(entry * (1 + adaptive_sl), 6)
            except Exception:
                pass

            signal["max_position_value"] = max_val * size_mult
            opened = self._open_position(self.active_portfolio, signal, "active")
            if opened:
                # C1: Record timestamp to enforce minimum spacing between active opens
                self._last_active_open_ts = time.time()
            new_this_cycle += 1
            if len(self.active_portfolio.positions) >= session_cap:
                break

    # ── LGBM auto-retrain ───────────────────────────────────────────────────────

    def _maybe_retrain_lgbm(self, regime: str) -> None:
        """
        Retrain LGBM on two schedules:
          • Normal:    once per 24 h (scheduled refresh)
          • Emergency: triggered when accuracy < 0.50 (matches the gate in ML filter),
                       but backs off to 24 h if a retrain produced no meaningful improvement.
        """
        now = time.time()
        current_acc = self.lgbm.get_accuracy(regime) or self.lgbm.get_accuracy("global")
        EMERGENCY_THRESHOLD = 0.50   # same as gate in ML filter blocks

        is_normal    = now - self._last_lgbm_retrain >= 86400
        is_emergency = (
            current_acc > 0
            and current_acc < EMERGENCY_THRESHOLD
            and now - self._last_lgbm_emergency >= 14400
        )
        # Drift detector can independently trigger a retrain
        is_drift = self.drift_detector.should_retrain()

        if not is_normal and not is_emergency and not is_drift:
            return

        if is_drift and not is_emergency:
            drift_snap = self.drift_detector.snapshot()
            self.log.warning(
                f"[DriftDetector] feature drift detected — PSI={drift_snap['psi']:.3f} "
                f"rolling_acc={drift_snap['rolling_accuracy']:.3f} — triggering retrain"
            )
            self._last_lgbm_emergency = now

        if is_emergency:
            self.log.warning(
                f"[LGBM] accuracy low — acc={current_acc:.3f} < {EMERGENCY_THRESHOLD} "
                f"— retraining with more data"
            )
            self._last_lgbm_emergency = now
        else:
            self._last_lgbm_retrain = now
            self._save_lgbm_retrain_ts(now)

        try:
            # Multi-symbol training: BTC+ETH+SOL+BNB — 2000 bars each (~20 days of 15m)
            # Training on 4 symbols gives the model exposure to different volatility
            # regimes and coin behaviors, improving accuracy on all active trades.
            _TRAIN_SYMBOLS = [
                "BTC/USDT:USDT", "ETH/USDT:USDT",
                "SOL/USDT:USDT", "BNB/USDT:USDT",
            ]
            market = self.market_state.market
            multi_data = []
            for sym in _TRAIN_SYMBOLS:
                try:
                    df = market.get_ohlcv_df(sym, timeframe="15m", limit=2000)
                    if len(df) >= 300:
                        multi_data.append((df["close"].values, df["volume"].values))
                except Exception:
                    pass

            if multi_data:
                new_acc = self.lgbm.train_multi(multi_data, regime=regime)
                self.log.info(
                    f"[LGBM] retrained regime={regime} on {len(multi_data)} symbols "
                    f"accuracy={new_acc:.3f}"
                )
                # Set reference distribution so drift detector has a fresh baseline
                try:
                    X_ref = []
                    for closes, vols in multi_data:
                        from models.lgbm_predictor import extract_features
                        for i in range(200, len(closes) - 6, 10):
                            X_ref.append(extract_features(closes[:i], vols[:i] if vols is not None else None))
                    if len(X_ref) > 50:
                        import numpy as np
                        self.drift_detector.set_reference(np.stack(X_ref))
                        self.drift_detector.reset()
                except Exception:
                    pass
                if regime != "global":
                    new_acc_g = self.lgbm.train_multi(multi_data, regime="global")
                    self.log.info(f"[LGBM] global retrained accuracy={new_acc_g:.3f}")
                    new_acc = max(new_acc, new_acc_g)
            else:
                # Fallback to BTC-only if multi-data fetch failed
                df = market.get_ohlcv_df("BTC/USDT:USDT", timeframe="15m", limit=2000)
                new_acc = self.lgbm.train(df["close"].values, df["volume"].values, regime=regime)
                self.log.info(f"[LGBM] retrained (BTC-only fallback) accuracy={new_acc:.3f}")

            # If retrain didn't help, back off emergency cooldown to 24h
            if is_emergency and new_acc < current_acc + 0.02:
                self._last_lgbm_emergency = now + 72000   # skip for extra 20h (= 24h total)
                self.log.info(
                    f"[LGBM] no improvement ({current_acc:.3f}→{new_acc:.3f}) — "
                    f"backing off emergency retrain for 24 h"
                )
        except Exception as exc:
            self.log.warning(f"[LGBM] retrain failed: {exc}")

    # ── strategy evolution ──────────────────────────────────────────────────────

    def _maybe_evolve_strategies(self) -> None:
        market = self.market_state.market
        ref    = "BTC/USDT:USDT"

        # Minimum R:R = 2.0 for passive (ATR stops allow some flexibility)
        # Minimum R:R = 2.5 for active (intraday needs strong reward/risk)
        # At 40% WR (momentum bear): need TP > 1.5x SL to have positive EV
        # At 37% WR (momentum bull): need TP > 1.7x SL to have positive EV
        # 2.0x and 2.5x provide a safety margin above the break-even requirement.
        new_p = self.strategy_evolver.maybe_evolve("passive", ref, market)
        if new_p:
            tp = new_p.get("take_profit") or 0
            sl = new_p.get("stop_loss") or 0
            ratio = tp / sl if sl > 0 else 0
            if ratio < 2.0:
                self.log.warning(
                    f"[Evolution] passive params rejected: TP/SL={ratio:.2f} < 2.0 "
                    f"(TP={tp:.4f} SL={sl:.4f})"
                )
            else:
                self.passive_strategy.update_params(new_p)
                self.log.info(
                    f"[Evolution] passive promoted: "
                    f"TP={new_p.get('take_profit')} SL={new_p.get('stop_loss')} "
                    f"EMA={new_p.get('ema_fast')}/{new_p.get('ema_slow')} "
                    f"sharpe={new_p.get('sharpe')}"
                )

        new_a = self.strategy_evolver.maybe_evolve("active", ref, market)
        if new_a:
            tp = new_a.get("take_profit") or 0
            sl = new_a.get("stop_loss") or 0
            ratio = tp / sl if sl > 0 else 0
            if ratio < 2.5:
                self.log.warning(
                    f"[Evolution] active params rejected: TP/SL={ratio:.2f} < 2.5 "
                    f"(TP={tp:.4f} SL={sl:.4f})"
                )
            else:
                self.active_strategy.update_params(new_a)
                self.log.info(
                    f"[Evolution] active promoted: "
                    f"TP={new_a.get('take_profit')} SL={new_a.get('stop_loss')} "
                    f"EMA={new_a.get('ema_fast')}/{new_a.get('ema_slow')} "
                    f"sharpe={new_a.get('sharpe')}"
                )

    # ── R:R guardian ────────────────────────────────────────────────────────────

    def _enforce_rr_guardian(self) -> None:
        """Every 50 cycles: ensure evolved params always satisfy minimum R:R ratios.
        Active must be ≥ 2.5x, Passive must be ≥ 2.0x. Auto-fixes silently."""
        if self.cycle % 50 != 0:
            return
        try:
            import json as _json
            from pathlib import Path as _Path
            ev_file = _Path("storage/evolved_params.json")
            if not ev_file.exists():
                return
            ev = _json.loads(ev_file.read_text())
            changed = False

            a = ev.get("active", {})
            a_sl = a.get("stop_loss", 0)
            a_tp = a.get("take_profit", 0)
            if a_sl > 0 and a_tp > 0:
                rr_a = a_tp / a_sl
                if rr_a < 2.5:
                    ev["active"]["take_profit"] = round(a_sl * 3.0, 5)
                    self.log.warning(
                        f"[RRGuardian] Active R:R={rr_a:.2f}x < 2.5x — auto-fixed to 3.0x "
                        f"(TP={ev['active']['take_profit']})"
                    )
                    changed = True

            p = ev.get("passive", {})
            p_sl = p.get("stop_loss", 0)
            p_tp = p.get("take_profit", 0)
            if p_sl > 0 and p_tp > 0:
                rr_p = p_tp / p_sl
                if rr_p < 2.0:
                    ev["passive"]["take_profit"] = round(p_sl * 3.0, 5)
                    self.log.warning(
                        f"[RRGuardian] Passive R:R={rr_p:.2f}x < 2.0x — auto-fixed to 3.0x "
                        f"(TP={ev['passive']['take_profit']})"
                    )
                    changed = True

            if changed:
                ev_file.write_text(_json.dumps(ev, indent=2))
                # Also update live strategy objects so they take effect immediately
                self.active_strategy.update_params(ev.get("active", {}))
                self.passive_strategy.update_params(ev.get("passive", {}))
        except Exception:
            pass

    # ── AI Brain + Strategy Lab ─────────────────────────────────────────────────

    def _maybe_run_ai_brain(self, market_state: dict) -> None:
        """
        Once per ~24 h, ask Claude to analyse the market and suggest param tweaks.
        Each suggestion is paper-tested in StrategyLab before being promoted.
        Silently skips if ANTHROPIC_API_KEY is not set.
        """
        if not self.ai_brain.is_available():
            return
        if not self.ai_brain.needs_run():
            return

        totals = self._portfolio_totals()
        stats  = self.risk_manager.get_strategy_stats()

        output = self.ai_brain.analyze(
            regime=self._prev_regime or "sideways",
            market_state=market_state,
            portfolio_status={
                "equity":       round(totals["equity"], 2),
                "realized_pnl": round(totals["realized_pnl"], 2),
                "drawdown":     round(totals["drawdown"], 4),
                "positions":    totals["positions"],
            },
            strategy_stats=stats,
            evolved_params={
                "passive": self.strategy_evolver.current_params("passive"),
                "active":  self.strategy_evolver.current_params("active"),
            },
            market_intel=self._last_intel or {},
            lgbm_accuracy={
                r: self.lgbm.get_accuracy(r)
                for r in ("global", "bull", "bear", "sideways", "volatile")
                if self.lgbm.get_accuracy(r) > 0
            },
            performance_history=self.perf_tracker.summary_for_ai(),
        )

        if not output or "error" in output:
            self.log.warning(f"[AIBrain] error: {output.get('error', 'no output')}")
            return

        confidence = float(output.get("confidence", 0.0))
        self.log.info(
            f"[AIBrain] regime_override={output.get('regime_override')} "
            f"confidence={confidence:.2f} "
            f"mode_priority={output.get('active_mode_priority')} "
            f"drawdown_action={output.get('drawdown_action')}"
        )

        if confidence < 0.50:
            self.log.info("[AIBrain] confidence too low — no param changes")
            return

        market = self.market_state.market

        # Apply regime override (updates market_state in-place for this cycle)
        regime_override = output.get("regime_override")
        if regime_override and regime_override != self._prev_regime:
            self.log.info(
                f"[AIBrain] regime override: {self._prev_regime} → {regime_override} "
                f"({output.get('regime_override_reason', '')})"
            )
            market_state = dict(market_state)
            market_state["regime"] = regime_override
            self._prev_regime = regime_override

        # Apply mode priority to bandit so it learns from AI's suggestion
        mode_priority = output.get("active_mode_priority")
        if mode_priority and isinstance(mode_priority, list):
            self.strategy_bandit.apply_priority_hint(
                self._prev_regime or "sideways", mode_priority
            )

        # Drawdown action
        dd_action = output.get("drawdown_action", "none")
        if dd_action == "pause_active":
            self.log.warning("[AIBrain] drawdown_action=pause_active — blocking new active trades")
            self._active_paused_until = time.time() + 3600  # pause 1 h
        elif dd_action == "close_passive_weakest":
            weakest = min(
                self.passive_portfolio.positions,
                key=lambda p: p.get("current_price", p["entry_price"]) / p["entry_price"]
                if p["side"] == "LONG"
                else p["entry_price"] / p.get("current_price", p["entry_price"]),
                default=None,
            )
            if weakest:
                self._close_position(self.passive_portfolio, weakest, reason="AI_DRAWDOWN")
                self.log.info(f"[AIBrain] closed weakest passive position: {weakest['symbol']}")

        # Test passive suggestions through StrategyLab
        passive_adj = output.get("passive_adjustments") or {}
        if passive_adj:
            promoted, delta = self.strategy_lab.test_and_promote("passive", passive_adj, market)
            if promoted:
                self.passive_strategy.update_params(
                    self.strategy_evolver.current_params("passive")
                )
                self.log.info(
                    f"[StrategyLab] passive AI params promoted Δsharpe={delta:.3f}"
                )
            else:
                self.log.info(
                    f"[StrategyLab] passive AI params rejected Δsharpe={delta:.3f}"
                )

        # Test active suggestions through StrategyLab
        active_adj = output.get("active_adjustments") or {}
        if active_adj:
            # Pre-validate R:R before passing to StrategyLab
            tp_adj = active_adj.get("take_profit") or 0
            sl_adj = active_adj.get("stop_loss")   or 0
            rr_adj = tp_adj / sl_adj if sl_adj > 0 else 0
            if rr_adj > 0 and rr_adj < 2.5:
                self.log.warning(
                    f"[StrategyLab] active AI params rejected before test: "
                    f"TP/SL={rr_adj:.2f} < 2.5 (TP={tp_adj} SL={sl_adj})"
                )
            else:
                promoted, delta = self.strategy_lab.test_and_promote("active", active_adj, market)
                if promoted:
                    # Also verify R:R of what was actually stored
                    stored = self.strategy_evolver.current_params("active")
                    tp_s = stored.get("take_profit") or 0
                    sl_s = stored.get("stop_loss")   or 0
                    rr_s = tp_s / sl_s if sl_s > 0 else 0
                    if rr_s >= 2.5:
                        self.active_strategy.update_params(stored)
                        self.log.info(f"[StrategyLab] active AI params promoted Δsharpe={delta:.3f}")
                    else:
                        self.log.warning(
                            f"[StrategyLab] active AI stored params rejected: "
                            f"TP/SL={rr_s:.2f} < 2.5"
                        )
                else:
                    self.log.info(f"[StrategyLab] active AI params rejected Δsharpe={delta:.3f}")

    # ── reporting ───────────────────────────────────────────────────────────────

    def _report(self) -> None:
        now = time.time()
        if now - self.last_report_at < 3600:
            return
        self.last_report_at = now
        ps  = self.passive_portfolio.status()
        as_ = self.active_portfolio.status()
        total = ps["equity"] + as_["equity"]
        intel = self._last_intel or {}
        self.log.info("[DualEngine] hourly summary")
        self.log.info(f"  Passive  equity={ps['equity']}  positions={ps['positions']}")
        self.log.info(f"  Active   equity={as_['equity']} positions={as_['positions']}")
        self.log.info(f"  Total portfolio value={total}")
        if intel:
            self.log.info(
                f"  MarketIntel breadth={intel.get('breadth')} "
                f"fear_greed={intel.get('fear_greed')} "
                f"trend={intel.get('dominant_trend')}"
            )

        # Record equity snapshot (hourly + daily)
        try:
            self.equity_tracker.record(
                passive_equity=ps["equity"],
                active_equity=as_["equity"],
                regime=self._prev_regime or "",
                fear_greed=float(intel.get("fear_greed", 50) or 50),
            )
            snap = self.equity_tracker.snapshot()
            if snap["days_tracked"] >= 2:
                self.log.info(
                    f"  CAGR={snap['cagr_pct']:+.1f}%  "
                    f"MaxDD={snap['max_drawdown_pct']:.1f}%  "
                    f"Sharpe={snap['sharpe_ratio']:.2f}  "
                    f"Growth={snap['total_growth_pct']:+.2f}%"
                )
        except Exception:
            pass

        # Notify when daily target reached (once per UTC day)
        dp = self.daily_profit.snapshot()
        if dp.get("phase") == "LOCKED":
            self.alerting.daily_target_reached(
                pnl_usdt=dp.get("pnl_today_usdt", 0.0),
                withdrawable_usdt=dp.get("withdrawable_usdt", 0.0),
                thb_per_usd=dp.get("thb_per_usd", 35.0),
            )
            # Move withdrawable profit into vault (once per day)
            vaulted = self.profit_vault.deposit_daily_profit(self.daily_profit)
            if vaulted > 0:
                from meta.daily_profit_engine import THB_PER_USD
                self.log.info(
                    f"[ProfitVault] deposited {vaulted:.2f} USDT "
                    f"({round(vaulted * THB_PER_USD, 0):.0f} THB) — "
                    f"vault balance={self.profit_vault.snapshot()['vault_balance_usdt']:.2f} USDT"
                )

        # Daily compound check (active bot)
        compounded = self.compound_manager.daily_check(self.daily_profit)
        if compounded > 0:
            self.log.info(f"[Compound] {compounded:.2f} USDT added to active capital")

        # Weekly compound check (passive + active)
        weekly_result = self.compound_manager.weekly_check()
        if weekly_result.get("action") == "compounded":
            self.log.info(
                f"[Compound] weekly review — equity_growth={weekly_result['equity_growth']:.2%} "
                f"total={weekly_result['total_equity']:.2f}"
            )

        # Once-daily Telegram/Discord summary (fires on the first report of each UTC day)
        today = datetime.now(timezone.utc).toordinal()
        if today != self._last_alert_day:
            self._last_alert_day = today
            stats        = self.risk_manager.get_strategy_stats()
            total_trades = sum(s.get("trades", 0) for s in stats.values())
            initial      = self.passive_portfolio.initial_cash + self.active_portfolio.initial_cash
            daily_pct    = (total - initial) / initial if initial else 0.0

            p_stats = stats.get("passive", {})
            a_stats = stats.get("active",  {})
            p_wr = p_stats.get("wins", 0) / max(p_stats.get("trades", 1), 1)
            a_wr = a_stats.get("wins", 0) / max(a_stats.get("trades", 1), 1)

            # Record daily snapshot for AI Brain performance analytics
            totals = self._portfolio_totals()
            self.perf_tracker.record_daily(
                equity=total,
                pnl_today=dp.get("pnl_today_usdt", 0.0),
                drawdown=totals["drawdown"],
                active_trades=a_stats.get("trades", 0),
                active_wins=a_stats.get("wins", 0),
                passive_trades=p_stats.get("trades", 0),
                passive_wins=p_stats.get("wins", 0),
                lgbm_accuracy=self.lgbm.get_accuracy("global"),
                regime=self._prev_regime or "unknown",
            )

            cycle = intel.get("btc_cycle", {})
            self.alerting.daily_summary(
                equity=total,
                daily_pct=daily_pct,
                trades=total_trades,
                passive_equity=ps["equity"],
                active_equity=as_["equity"],
                pnl_today_usdt=dp.get("pnl_today_usdt", 0.0),
                withdrawable_usdt=dp.get("withdrawable_usdt", 0.0),
                thb_per_usd=dp.get("thb_per_usd", 35.0),
                lgbm_accuracy=self.lgbm.get_accuracy("global"),
                regime=self._prev_regime or "",
                btc_cycle_phase=cycle.get("cycle_phase", ""),
                active_winrate=a_wr,
                passive_winrate=p_wr,
            )

        # Daily autopilot report (fires once per day at market close ~23:00 UTC)
        hour = datetime.now(timezone.utc).hour
        today_ord = datetime.now(timezone.utc).toordinal()
        if hour == 23 and today_ord != self._last_report_day:
            self._last_report_day = today_ord
            from research.daily_report import DailyAutopilot
            try:
                recent_trades = list(self.trade_manager.get_recent_trades())
                report = DailyAutopilot().generate(
                    trade_history=recent_trades,
                    market_intel=self._last_intel or {},
                    evolved_params={
                        "passive": self.strategy_evolver.current_params("passive"),
                        "active":  self.strategy_evolver.current_params("active"),
                    },
                )
                self.log.info(f"[DailyReport] {report['summary']}")
                self.log.info(
                    f"[DailyReport] Winners: {report['winning_modes']} "
                    f"| Losers: {report['losing_modes']}"
                )
                self.log.info(f"[DailyReport] Tomorrow watch: {report['tomorrow_focus']}")
                if report.get("action_items"):
                    for item in report["action_items"][:3]:
                        self.log.info(f"[DailyReport] Action: {item}")
            except Exception:
                pass

    # ── state persistence ───────────────────────────────────────────────────────

    def _persist_state(self, market_state: Optional[dict] = None) -> None:
        totals         = self._portfolio_totals()
        strategy_stats = self.risk_manager.get_strategy_stats()
        payload = {
            "dual":               self._dual_status(totals, strategy_stats),
            "portfolio":          self._combined_portfolio_status(totals),
            "performance":        self._performance_status(totals, strategy_stats),
            "portfolio_snapshot": self._portfolio_snapshot(),
        }
        if market_state is not None:
            payload["market"] = market_state
        if self._last_intel:
            payload["market_intelligence"] = self._last_intel

        self.state_manager.save_state(payload)
        self.state_manager.update_equity(totals["equity"])

    def _portfolio_totals(self) -> dict:
        p_invested = sum(p["position_value"] for p in self.passive_portfolio.positions)
        a_invested = sum(p["position_value"] for p in self.active_portfolio.positions)
        equity     = self.passive_portfolio.equity + self.active_portfolio.equity
        cash       = self.passive_portfolio.cash   + self.active_portfolio.cash
        realized   = self.passive_portfolio.realized_pnl + self.active_portfolio.realized_pnl
        unrealized = self.passive_portfolio.unrealized_pnl + self.active_portfolio.unrealized_pnl
        initial    = self.passive_portfolio.initial_cash + self.active_portfolio.initial_cash
        drawdown   = max(0.0, (initial - equity) / initial) if initial > 0 else 0.0
        return {
            "active_invested": a_invested,
            "cash":            cash,
            "drawdown":        drawdown,
            "equity":          equity,
            "initial":         initial,
            "passive_invested": p_invested,
            "positions":       len(self.passive_portfolio.positions) + len(self.active_portfolio.positions),
            "realized_pnl":    realized,
            "unrealized_pnl":  unrealized,
        }

    def _combined_portfolio_status(self, totals: dict) -> dict:
        return {
            "cash":           round(totals["cash"], 2),
            "equity":         round(totals["equity"], 2),
            "positions":      totals["positions"],
            "realized_pnl":   round(totals["realized_pnl"], 2),
            "unrealized_pnl": round(totals["unrealized_pnl"], 2),
            "drawdown":       round(totals["drawdown"], 4),
        }

    def _performance_status(self, totals: dict, strategy_stats: dict) -> dict:
        total_trades = sum(s.get("trades", 0) for s in strategy_stats.values())
        total_wins   = sum(s.get("wins",   0) for s in strategy_stats.values())
        winrate = total_wins / total_trades if total_trades else 0.0
        pnl     = totals["realized_pnl"] + totals["unrealized_pnl"]
        return {
            "winrate":       round(winrate, 4),
            "total_trades":  total_trades,
            "pnl":           round(pnl, 2),
            "daily_pnl":     round(pnl, 2),
            "daily_pnl_pct": round(pnl / totals["initial"], 4) if totals["initial"] else 0,
            "equity":        round(totals["equity"], 2),
            "drawdown":      round(totals["drawdown"], 4),
        }

    def _dual_status(self, totals: dict, strategy_stats: dict) -> dict:
        intel = self._last_intel
        return {
            "passive": {
                "cash":          round(self.passive_portfolio.cash, 2),
                "equity":        round(self.passive_portfolio.equity, 2),
                "positions":     len(self.passive_portfolio.positions),
                "realized_pnl":  round(self.passive_portfolio.realized_pnl, 2),
                "unrealized_pnl": round(self.passive_portfolio.unrealized_pnl, 2),
            },
            "active": {
                "cash":          round(self.active_portfolio.cash, 2),
                "equity":        round(self.active_portfolio.equity, 2),
                "positions":     len(self.active_portfolio.positions),
                "realized_pnl":  round(self.active_portfolio.realized_pnl, 2),
                "unrealized_pnl": round(self.active_portfolio.unrealized_pnl, 2),
            },
            "total_portfolio_value": round(totals["equity"], 2),
            "capital_utilization": {
                "passive": round(
                    totals["passive_invested"]
                    / max(self.passive_portfolio.cash + totals["passive_invested"], 1), 4
                ),
                "active": round(
                    totals["active_invested"]
                    / max(self.active_portfolio.cash + totals["active_invested"], 1), 4
                ),
            },
            "capital_allocation": self.capital_allocator.snapshot(
                self.passive_portfolio, self.active_portfolio, strategy_stats,
            ),
            "strategy_stats": strategy_stats,
            "conflict_log":   self.conflict_log[:10],
            "regime_route":   self._last_route,
            "evolution_log":  self.strategy_evolver.evolution_log()[:5],
            "evolved_params": {
                "passive": self.strategy_evolver.current_params("passive"),
                "active":  self.strategy_evolver.current_params("active"),
            },
            "market_intelligence": intel,
            "daily_target": {
                "passive": "on track" if self.passive_portfolio.realized_pnl >= 0 else "behind",
                "active":  self.daily_profit.snapshot(),
            },
            "bandit": self.strategy_bandit.summary(),
            "bandit_top_modes": self.strategy_bandit.top_modes(
                self._prev_regime or "sideways"
            ),
            "lgbm_accuracy": {
                regime: self.lgbm.get_accuracy(regime)
                for regime in ("global", "bull", "bear", "sideways", "volatile")
                if self.lgbm.get_accuracy(regime) > 0
            },
            "bo_status":       self.strategy_evolver.bo_status(),
            "symbol_edge":     self.edge_tracker.summary(),
            "ai_brain":        self.ai_brain.last_output(),
        }

    def _portfolio_snapshot(self) -> dict:
        return {
            "mode": "dual",
            "passive": {
                "cash":          self.passive_portfolio.cash,
                "realized_pnl":  self.passive_portfolio.realized_pnl,
                "open_positions": self.passive_portfolio.positions,
            },
            "active": {
                "cash":          self.active_portfolio.cash,
                "realized_pnl":  self.active_portfolio.realized_pnl,
                "open_positions": self.active_portfolio.positions,
            },
        }

    # ── main loop ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            try:
                self.cycle += 1
                self._print_cycle_status()

                # ── market intelligence first (cached 10 min) ───────────────────
                # Fetched before market_state so regime classifier can use it.
                try:
                    self._last_intel = self.market_intel.full_context()
                except Exception as exc:
                    self.log.warning(f"[MarketIntel] {exc}")

                # Pass intel so regime classifier can fuse macro signals
                market_state = self.market_state.get_market_state(intel=self._last_intel)

                # ── manage existing positions (trailing stops, TP/SL, EOD close) ──
                self._manage_open_positions()

                # ── daily loss protection: tighten active stops if cap hit ───────
                self._enforce_active_daily_loss_protection()

                # ── portfolio-level risk check ──────────────────────────────────
                portfolio_check = self.risk_manager.check_portfolio(
                    self.passive_portfolio.equity,
                    self.active_portfolio.equity,
                )
                if portfolio_check["halt"]:
                    self.log.warning(f"[DualEngine] halt: {portfolio_check['reason']}")
                    self.alerting.halt(portfolio_check["reason"])
                    self._persist_state(market_state)
                    # Auto-reset kill switch if Binance actual balance is healthy
                    # (protects against false triggers from rate-limit-induced tracking errors)
                    try:
                        actual_balance = self.broker._client.fetch_balance_usdt() if hasattr(self.broker, "_client") else 0.0
                        if actual_balance <= 0:
                            from data.binance_client import BinanceClient
                            actual_balance = BinanceClient().fetch_balance_usdt()
                        from config.dual_settings import KILL_SWITCH_EQUITY
                        if actual_balance > 0 and actual_balance >= KILL_SWITCH_EQUITY:
                            self.log.info(
                                f"[KillSwitch] Binance balance={actual_balance:.2f} >= "
                                f"threshold={KILL_SWITCH_EQUITY:.2f} — resetting internal equity"
                            )
                            # Re-sync portfolio initial_cash to actual balance
                            total = self.passive_portfolio.initial_cash + self.active_portfolio.initial_cash
                            ratio = actual_balance / max(total, 1.0)
                            self.passive_portfolio.initial_cash = round(self.passive_portfolio.initial_cash * ratio, 2)
                            self.active_portfolio.initial_cash  = round(self.active_portfolio.initial_cash  * ratio, 2)
                    except Exception:
                        pass
                    time.sleep(300)
                    continue

                # ── capital rebalance ───────────────────────────────────────────
                self._rebalance_capital()

                # ── strategy evolution ──────────────────────────────────────────
                self._maybe_evolve_strategies()

                # ── R:R guardian (every 50 cycles) ─────────────────────────────
                self._enforce_rr_guardian()

                # ── AI Brain (every 4h, silently skips if no API key) ───────────
                self._maybe_run_ai_brain(market_state)

                # ── regime routing ──────────────────────────────────────────────
                route = self.regime_router.route(market_state)
                raw_regime = market_state.get("regime", "")
                # Debounce: require 3 consecutive same-regime reads before flipping.
                # Prevents single-candle spikes from halting all trading.
                if raw_regime == self._regime_candidate:
                    self._regime_streak += 1
                else:
                    self._regime_candidate = raw_regime
                    self._regime_streak = 1
                if self._regime_streak >= 3 or not self._prev_regime:
                    current_regime = raw_regime
                else:
                    current_regime = self._prev_regime
                if current_regime != self._prev_regime and self._prev_regime:
                    self._handle_regime_shift(self._prev_regime, current_regime)
                    self.alerting.regime_change(self._prev_regime, current_regime)
                self._prev_regime = current_regime

                # LGBM retrain check (now that current_regime is known)
                self._maybe_retrain_lgbm(current_regime or "global")

                # Save regime-recommended mode before bandit can override it (used as fallback)
                regime_active_mode = route["active_mode"]

                # Bandit: autonomously select best passive + active modes per regime
                bandit_passive = self.strategy_bandit.select_passive_mode(
                    current_regime, route["passive_mode"]
                )
                bandit_active  = self.strategy_bandit.select_active_mode(
                    current_regime, route["active_mode"]
                )
                if bandit_active != route["active_mode"] or bandit_passive != route["passive_mode"]:
                    route = dict(route)
                    route["active_mode"]  = bandit_active
                    route["passive_mode"] = bandit_passive

                # Legacy adaptive mode fallback (still used if bandit is in warmup)
                adaptive = self._best_active_mode(route["active_mode"])
                if adaptive != route["active_mode"]:
                    route = dict(route)
                    route["active_mode"] = adaptive

                self._last_route = route
                if self._should_print_cycle():
                    self.log.info(
                        f"[Regime] {route['reason']} | "
                        f"passive={route['passive_mode']} "
                        f"active={route['active_mode']} "
                        f"size×{route['size_mult']}"
                    )

                # ── scan — separate universe per strategy ───────────────────────
                try:
                    passive_scan = self.scanner.scan_for_passive(top_n=8)
                    active_scan  = self.scanner.scan_for_active(top_n=15)
                    passive_symbols = [x["symbol"] for x in passive_scan]
                    active_symbols  = [x["symbol"] for x in active_scan]
                    # Funding arb: always prepend high-rate symbols regardless of scanner
                    # These have STRUCTURAL edge — always worth checking first
                    if route.get("active_mode") in ("funding_arb", "mean_reversion"):
                        try:
                            all_syms = self.scanner._get_universe()
                            funding_hits = self.scanner.funding_arb.scan_universe(
                                all_syms, top_n=8
                            )
                            funding_syms = [x["symbol"] for x in funding_hits]
                            active_symbols = list(dict.fromkeys(funding_syms + active_symbols))
                        except Exception:
                            pass
                    # Cross-engine: register scan results with bridge
                    self.cross_bridge.register_active_scan(active_scan)
                    self.cross_bridge.register_passive_scan(passive_scan)
                    # Prepend hot coins from cross-bridge (keep deduped order)
                    hot_passive = self.cross_bridge.hot_coins_for_passive()
                    hot_active  = self.cross_bridge.hot_coins_for_active()
                    passive_symbols = list(dict.fromkeys(hot_passive + passive_symbols))
                    active_symbols  = list(dict.fromkeys(hot_active  + active_symbols))
                    # Remove globally avoided symbols (stop-loss flagged by either engine)
                    avoided = set(self.cross_bridge.avoided_symbols())
                    passive_symbols = [s for s in passive_symbols if s not in avoided]
                    active_symbols  = [s for s in active_symbols  if s not in avoided]

                    # Volume anomaly scan: prepend high-volume breakout symbols when in momentum mode
                    if route.get("active_mode") == "momentum":
                        try:
                            anomaly_hits = self.scanner.scan_volume_anomaly(top_n=5)
                            anomaly_syms = [x["symbol"] for x in anomaly_hits if x["symbol"] not in avoided]
                            if anomaly_syms:
                                active_symbols = list(dict.fromkeys(anomaly_syms + active_symbols))
                        except Exception:
                            pass

                    # BEAR priority: always check the most liquid SHORT candidates first.
                    # In BEAR+extreme-fear, BTC/ETH/SOL/BNB have the cleanest downtrends
                    # and best momentum signal quality. Prepend them so they're always
                    # evaluated even when the scanner scores other coins higher.
                    _bear_regime = "bear" in (current_regime or "").lower()
                    if _bear_regime and route.get("active_mode") == "momentum":
                        _priority = [
                            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                            "BNB/USDT:USDT", "XRP/USDT:USDT", "LINK/USDT:USDT",
                        ]
                        _priority = [s for s in _priority if s not in avoided]
                        active_symbols = list(dict.fromkeys(_priority + active_symbols))

                    if _bear_regime and route.get("passive_mode") == "short":
                        _pass_priority = [
                            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                            "BNB/USDT:USDT",
                        ]
                        _pass_priority = [s for s in _pass_priority if s not in avoided]
                        passive_symbols = list(dict.fromkeys(_pass_priority + passive_symbols))
                except Exception:
                    # Fallback to general scan
                    general = [x["symbol"] for x in self.scanner.scan_market()]
                    passive_symbols = general
                    active_symbols  = general

                # ── open new trades ─────────────────────────────────────────────
                dd_factor       = self._drawdown_size_factor()
                compound_factor = self._compound_size_factor()
                eff_mult        = round(route["size_mult"] * dd_factor * compound_factor, 3)
                if (dd_factor < 1.0 or compound_factor > 1.0) and self._should_print_cycle():
                    self.log.info(
                        f"[SizingFactors] dd={dd_factor:.2f} compound={compound_factor:.2f} "
                        f"regime_mult={route['size_mult']} → eff={eff_mult}"
                    )

                # ── self-healing drawdown mode ──────────────────────────────────
                # When drawdown >= 10%, restrict active to safest modes only.
                # When drawdown >= 15%, also cut position sizes to 50% and boost
                # passive hold threshold (STALE check tightens automatically via
                # dd_factor = 0.5 already, so we only need mode restriction here).
                route = self._apply_drawdown_healing(route, dd_factor)
                self._open_passive_trades(
                    passive_symbols, market_state,
                    passive_mode=route["passive_mode"],
                    size_mult=eff_mult,
                )
                positions_before = len(self.active_portfolio.positions)
                self._open_active_trades(
                    active_symbols, market_state,
                    active_mode=route["active_mode"],
                    size_mult=eff_mult,
                )
                # Fallback: if bandit chose a non-regime mode and found no signals, try regime mode
                # Skip momentum fallback in sideways/bear — it opens LONGs into a downtrend
                _regime_fallback_ok = not (
                    current_regime in ("sideways", "bear")
                    and regime_active_mode == "momentum"
                )
                if (
                    len(self.active_portfolio.positions) == positions_before
                    and route["active_mode"] != regime_active_mode
                    and regime_active_mode not in ("skip",)
                    and _regime_fallback_ok
                ):
                    self._open_active_trades(
                        active_symbols, market_state,
                        active_mode=regime_active_mode,
                        size_mult=eff_mult,
                    )

                # Multi-mode idle fallback: if active has had 0 positions for ≥5 cycles,
                # sweep remaining modes at 0.8× size — but respect regime direction.
                if len(self.active_portfolio.positions) == 0:
                    self._active_idle_cycles += 1
                    if self._active_idle_cycles >= 5:
                        if self._active_idle_cycles == 5 or self._active_idle_cycles % 100 == 0:
                            self.log.info(
                                f"[ActiveIdle] {self._active_idle_cycles} idle cycles — "
                                f"trying regime-safe modes at 0.8× size"
                            )
                        # Backtest-validated fallback order:
                        # Only use modes with confirmed positive edge per regime.
                        # mean_reversion: Sharpe -6 to -1 across ALL regimes → BANNED
                        # vwap_reversal:  Sharpe -6 to -3 across ALL regimes → BANNED
                        # momentum: edge in bull+bear only (direction-guarded below)
                        # funding_arb: structural, always safe to try
                        _bear_like = current_regime in ("bear", "bear_trend", "sideways")
                        _bull_like = current_regime in ("bull", "bull_trend")
                        for _fb_mode in ("funding_arb", "momentum", "breakout"):
                            if _fb_mode == route["active_mode"]:
                                continue
                            # mean_reversion and vwap_reversal permanently banned
                            if _bear_like and _fb_mode == "momentum":
                                pass   # momentum SHORT is fine in bear (direction guard handles LONG block)
                            if _bull_like and _fb_mode == "momentum":
                                pass   # momentum LONG is fine in bull
                            self._open_active_trades(
                                active_symbols, market_state,
                                active_mode=_fb_mode,
                                size_mult=round(eff_mult * 0.8, 3),
                            )
                            if len(self.active_portfolio.positions) > 0:
                                self._active_idle_cycles = 0
                                break
                else:
                    self._active_idle_cycles = 0

                # ── hourly report + state persist ───────────────────────────────
                self._report()
                self._persist_state(market_state)
                time.sleep(LOOP_DELAY_SECONDS)

            except Exception as exc:
                self.log.error(f"[DualEngine] runtime error: {exc}")
                time.sleep(LOOP_DELAY_SECONDS)
