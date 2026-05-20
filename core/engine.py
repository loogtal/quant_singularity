"""
Trading engine — autonomous loop orchestrating all subsystems.
"""

import time
import traceback
from datetime import datetime, timezone

from config.settings import (
    AUTO_TUNE_ENABLED,
    AUTO_TRAIN_ML,
    COOLDOWN_SECONDS,
    DAILY_TARGET_PCT,
    DASHBOARD_ENABLED,
    INITIAL_CAPITAL,
    LIVE_CONFIRM,
    LIVE_MODE,
    LOOP_DELAY_SECONDS,
    MAX_POSITIONS,
    PERSIST_PORTFOLIO,
    USE_ML,
)
from core.logger import get_logger
from core.metrics import CycleMetrics
from meta.market_state import MarketState
from meta.meta_brain import MetaBrain
from meta.confidence_manager import ConfidenceManager
from meta.memory_store import MemoryStore
from research.coin_scanner import CoinScanner
from alpha.alpha_engine import AlphaEngine
from strategy.strategy_engine import StrategyEngine
from risk.risk_engine import RiskEngine
from risk.kill_switch import KillSwitch
from portfolio.portfolio_engine import PortfolioEngine
from execution.paper_broker import PaperBroker
from research.trade_logger import TradeLogger
from core.state_manager import StateManager
from self_evolve.performance_tracker import PerformanceTracker
from self_evolve.evolution_engine import EvolutionEngine
from self_evolve.auto_tuner import AutoTuner
from monitoring.alerting import Alerting
from monitoring.dashboard import DashboardServer
from data.realtime_feed import RealtimeFeed
from models.online_learner import OnlineLearner
from models.trainer import ModelTrainer


def _print_header():
    print("\n" + "=" * 70)


def stop_or_tp_hit(position, current_price):
    side = position["side"]
    stop_loss = position.get("stop_loss")
    take_profit = position.get("take_profit")
    if stop_loss is None or take_profit is None:
        return None
    if side == "LONG":
        if current_price <= stop_loss:
            return "STOP LOSS"
        if current_price >= take_profit:
            return "TAKE PROFIT"
    elif side == "SHORT":
        if current_price >= stop_loss:
            return "STOP LOSS"
        if current_price <= take_profit:
            return "TAKE PROFIT"
    return None


def build_risk_levels(side, entry_price, atr):
    stop_distance = atr * 2.0
    take_distance = atr * 3.5
    if side == "LONG":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + take_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - take_distance
    return round(stop_loss, 6), round(take_profit, 6)


def update_trailing_stop(position):
    current_price = position["current_price"]
    atr = position.get("atr", 0)
    if atr <= 0:
        return
    if position["side"] == "LONG":
        new_sl = current_price - atr * 1.5
        if new_sl > position["stop_loss"]:
            position["stop_loss"] = round(float(new_sl), 6)
    else:
        new_sl = current_price + atr * 1.5
        if new_sl < position["stop_loss"]:
            position["stop_loss"] = round(float(new_sl), 6)


def create_broker(price_feed=None):
    if LIVE_MODE:
        from execution.live_broker import LiveBroker
        return LiveBroker(price_feed=price_feed)
    return PaperBroker(price_feed=price_feed)


class TradingEngine:
    """Autonomous trading loop — scan, alpha, ensemble strategy, risk, execute."""

    def __init__(self):
        self.log = get_logger()
        self.market = MarketState()
        self.meta_brain = MetaBrain()
        self.scanner = CoinScanner()
        self.alpha_engine = AlphaEngine()
        self.strategy_engine = StrategyEngine()
        self.risk_engine = RiskEngine()
        self.kill_switch = KillSwitch()
        self.confidence_mgr = ConfidenceManager()
        self.memory = MemoryStore()
        self.realtime = RealtimeFeed()
        self.portfolio = PortfolioEngine(initial_cash=INITIAL_CAPITAL)
        self.broker = create_broker(price_feed=self.realtime)
        self.trade_logger = TradeLogger()
        self.state_manager = StateManager()
        self.perf_tracker = PerformanceTracker()
        self.evolution = EvolutionEngine()
        self.auto_tuner = AutoTuner()
        self.alerts = Alerting()
        self.online_learner = OnlineLearner() if USE_ML else None
        self.last_trade_time: dict[str, float] = {}
        self.cycle = 0
        self._load_persisted_portfolio()
        if AUTO_TRAIN_ML and USE_ML:
            result = ModelTrainer().train("BTC/USDT:USDT")
            print(f"[ML] Auto-train: {result}")
        if DASHBOARD_ENABLED:
            DashboardServer().start()

    def manage_open_positions(self):
        for pos in list(self.portfolio.positions):
            current_price = self.broker.get_price(pos["symbol"])
            self.portfolio.update_position_price(pos["symbol"], current_price)
            pos["current_price"] = current_price
            update_trailing_stop(pos)
            hit = stop_or_tp_hit(pos, current_price)
            if hit:
                pnl = self.broker.close_position(pos)
                self.portfolio.close_position(pos["symbol"], pnl)
                self.perf_tracker.record_trade(pnl)
                self.memory.record_trade(pos["symbol"], pnl, pos["side"])
                self.trade_logger.log_trade({
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "qty": pos["size"],
                    "pnl": pnl,
                    "strategy": "ensemble",
                    "regime": pos.get("regime", ""),
                })
                msg = f"{hit} | {pos['symbol']} | PnL={round(pnl, 2)}"
                print(f"\n{msg}")
                self.log.info(msg)
                self.alerts.trade_close(
                    pos["symbol"], hit, pnl, self.portfolio.equity
                )
                if self.online_learner and pos.get("ml_features") is not None:
                    self.online_learner.update(
                        pos["ml_features"], won=(pnl >= 0)
                    )

    def _print_open_positions(self) -> None:
        if not self.portfolio.positions:
            return
        print("\nOpen positions:")
        for pos in self.portfolio.positions:
            px = pos.get("current_price", pos["entry_price"])
            upnl = pos.get("unrealized_pnl", 0)
            print(
                f"  {pos['side']} {pos['symbol']} | entry={pos['entry_price']} "
                f"now={round(px, 6)} | uPnL={round(upnl, 2)} | "
                f"SL={pos.get('stop_loss')} TP={pos.get('take_profit')}"
            )

    def run_cycle(self) -> None:
        self.cycle += 1
        _print_header()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"Cycle #{self.cycle} | {ts}")

        market_state = self.market.get_market_state()
        print(f"Market: {market_state}")

        perf = self.perf_tracker.get_metrics(self.portfolio.equity, INITIAL_CAPITAL)
        perf["drawdown"] = self.portfolio.drawdown
        meta = self.meta_brain.evaluate(market_state, perf)
        if AUTO_TUNE_ENABLED:
            self.auto_tuner.tune(perf)
            meta = self.auto_tuner.apply_to_meta(meta, perf)
        reflection = self.evolution.reflect_and_adjust(perf)
        size_mult = self.kill_switch.size_multiplier(meta, reflection)
        meta["size_multiplier"] = size_mult

        print(f"Meta: {meta}")
        print(f"Reflection: {reflection}")

        halt = self.kill_switch.check(self.portfolio, perf, market_state, reflection)
        self.manage_open_positions()

        if halt["halt"]:
            print(f"\nHALT: {halt['reason']}")
            self.alerts.halt(halt["reason"])
            print(self.portfolio.status())
            self._persist(market_state, perf, meta)
            delay = 300 if halt["action"] == "cooldown" else LOOP_DELAY_SECONDS
            time.sleep(delay)
            return

        if len(self.portfolio.positions) >= MAX_POSITIONS:
            print("\nMax positions reached")
            print(self.portfolio.status())
            self._persist(market_state, perf, meta)
            time.sleep(LOOP_DELAY_SECONDS)
            return

        ranked = self.scanner.scan_market()
        if not ranked:
            print("\nNo tradeable coins found")
            time.sleep(LOOP_DELAY_SECONDS)
            return

        print("\nTop coins:")
        for coin in ranked[:5]:
            print(f"  {coin['symbol']} score={coin['score']:.3f}")

        traded = False
        skip_reason = None

        for candidate in ranked[:8]:
            symbol = candidate["symbol"]
            now = time.time()

            if self.portfolio.has_open_position(symbol):
                continue
            if symbol in self.last_trade_time and (now - self.last_trade_time[symbol]) < COOLDOWN_SECONDS:
                continue

            alpha = self.alpha_engine.generate_alpha(
                symbol=symbol,
                market_state=market_state,
                scanner_score=candidate["score"],
            )
            signal = self.strategy_engine.generate_signal(
                alpha=alpha, market_state=market_state, meta=meta
            )

            if signal["side"] == "HOLD":
                if skip_reason is None:
                    skip_reason = (
                        f"{symbol}: HOLD (alpha {alpha['direction']} conf "
                        f"{signal['confidence']:.2f}, scanner {candidate['score']:.2f})"
                    )
                continue

            effective_conf = max(
                signal["confidence"],
                candidate["score"] * 0.92,
            )
            _, ok, reason = self.confidence_mgr.resolve(
                meta, reflection, effective_conf
            )
            if not ok:
                if skip_reason is None:
                    skip_reason = f"{symbol}: {reason}"
                continue

            bias = self.memory.symbol_bias(symbol)
            signal["confidence"] = round(min(effective_conf * bias, 1.0), 4)

            risk = self.risk_engine.evaluate(
                signal=signal,
                portfolio=self.portfolio,
                market_state=market_state,
                min_confidence=meta.get("min_confidence"),
            )
            if not risk["allow_trade"]:
                if skip_reason is None:
                    skip_reason = f"{symbol}: risk — {risk.get('reason', 'blocked')}"
                continue

            size = risk["position_size"] * meta.get("size_multiplier", 1.0)
            if LIVE_MODE:
                order = self.broker.execute_order(
                    symbol=signal["symbol"],
                    side=signal["side"],
                    size=size,
                    equity=self.portfolio.equity,
                )
            else:
                order = self.broker.execute_order(
                    symbol=signal["symbol"], side=signal["side"], size=size
                )
            if order is None:
                continue

            entry_price = order["price"]
            stop_loss, take_profit = build_risk_levels(
                signal["side"], entry_price, risk["atr"]
            )

            ml_features = None
            if USE_ML and self.alpha_engine.ml:
                try:
                    df_ml = self.alpha_engine.market.get_ohlcv_df(symbol, "15m", 120)
                    ml_features = self.alpha_engine.ml.extract_features(
                        df_ml["close"].values,
                        df_ml["volume"].values if "volume" in df_ml else None,
                    )
                except Exception:
                    pass

            self.portfolio.add_position({
                "symbol": signal["symbol"],
                "side": signal["side"],
                "size": size,
                "entry_price": entry_price,
                "current_price": entry_price,
                "position_value": entry_price * size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": risk["atr"],
                "regime": market_state["regime"],
                "opened_at": now,
                "ml_features": ml_features,
            })
            self.last_trade_time[symbol] = now
            self.trade_logger.log_trade({
                "symbol": signal["symbol"],
                "side": signal["side"],
                "entry_price": entry_price,
                "exit_price": None,
                "qty": size,
                "pnl": None,
                "strategy": "ensemble",
                "regime": market_state["regime"],
            })
            print(f"\nOPEN {signal['side']} {symbol} @ {entry_price} size={size}")
            self.log.info(f"OPEN {signal['side']} {symbol} @ {entry_price} size={size}")
            self.alerts.trade_open(
                symbol, signal["side"], entry_price, size, self.portfolio.equity
            )
            traded = True
            break

        if not traded:
            print("\nNo trade this cycle")
            if self.portfolio.positions:
                held = ", ".join(p["symbol"] for p in self.portfolio.positions)
                print(f"  (holding {len(self.portfolio.positions)}: {held})")
            if skip_reason:
                print(f"  ({skip_reason})")

        self._print_open_positions()
        print(f"\nPortfolio: {self.portfolio.status()}")
        sess = self.perf_tracker.metrics
        if sess["total_trades"]:
            wr = sess["wins"] / sess["total_trades"]
            print(
                f"Session: {sess['total_trades']} trades | "
                f"winrate={wr:.1%} | PnL={sess['total_pnl']:.2f}"
            )
        print(f"Performance: winrate={perf['winrate']:.1%} daily={perf['daily_pnl_pct']:.2%}")
        if perf["daily_pnl_pct"] >= DAILY_TARGET_PCT:
            print(f"  Daily target hit: {perf['daily_pnl_pct']:.2%} >= {DAILY_TARGET_PCT:.2%}")

        m = CycleMetrics(
            cycle=self.cycle,
            equity=self.portfolio.equity,
            drawdown=self.portfolio.drawdown,
            positions=len(self.portfolio.positions),
            winrate=perf["winrate"],
            daily_pnl_pct=perf["daily_pnl_pct"],
            regime=market_state["regime"],
            meta_mode=meta.get("mode", "neutral"),
        )
        self.log.info(f"cycle_metrics {m.to_dict()}")
        self._persist(market_state, perf, meta)

    def _load_persisted_portfolio(self) -> None:
        if not PERSIST_PORTFOLIO:
            return
        snap = self.state_manager.state.get("portfolio_snapshot")
        if not snap:
            return
        positions = snap.get("open_positions", [])
        if positions:
            self.portfolio.restore(
                cash=snap.get("cash", INITIAL_CAPITAL),
                positions=positions,
                realized_pnl=snap.get("realized_pnl", 0),
            )
            print(f"Restored {len(positions)} position(s) from disk")

    def _persist(self, market_state, perf, meta):
        payload = {
            "market": market_state,
            "portfolio": self.portfolio.status(),
            "performance": perf,
            "meta": meta,
        }
        if PERSIST_PORTFOLIO:
            payload["portfolio_snapshot"] = {
                "cash": self.portfolio.cash,
                "realized_pnl": self.portfolio.realized_pnl,
                "open_positions": self.portfolio.positions,
            }
        self.state_manager.save_state(payload)
        self.state_manager.update_equity(self.portfolio.equity)

    def run(self):
        _print_header()
        print("QUANT SINGULARITY — Autonomous AI Trader")
        print(f"Mode: {'LIVE' if LIVE_MODE else 'PAPER'}")
        print(f"Capital: {INITIAL_CAPITAL} USDT")
        if LIVE_MODE and not LIVE_CONFIRM:
            print("WARNING: QS_LIVE_CONFIRM=false — live orders BLOCKED until enabled")
        if self.alerts.enabled:
            print("Alerts: ON (Discord/Telegram)")
        if AUTO_TUNE_ENABLED:
            print(f"Auto-tune: min_conf={self.auto_tuner.params.get('min_confidence')}")
        print("=" * 70)
        self.log.info("Trading engine started")

        while True:
            try:
                self.run_cycle()
                time.sleep(LOOP_DELAY_SECONDS)
            except KeyboardInterrupt:
                print("\nStopped by user")
                self.log.info("Stopped by user")
                break
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()
                self.log.exception(str(e))
                time.sleep(10)
