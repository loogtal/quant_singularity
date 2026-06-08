"""
Live Binance USDT-M Futures execution.

Production-grade order execution with:
  - Automatic leverage setting per symbol before first order
  - Lot-size precision rounding (avoids LOT_SIZE filter errors)
  - Retry logic (up to 3 attempts on network/rate-limit errors)
  - Fill confirmation: reads actual fill price from order response
  - Bid-ask spread check: skips if spread > MAX_SPREAD_PCT
  - Audit log: every order written to storage/orders.jsonl
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LIVE_MODE, STORAGE_DIR
from data.binance_client import BinanceClient
from data.market_data import MarketData
from risk.live_safety import LiveSafety

_ORDER_LOG = STORAGE_DIR / "orders.jsonl"

MAX_SPREAD_PCT  = 0.0008   # skip if spread > 0.08%
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2.0


class LiveBroker:
    """Production Binance USDT-M futures broker."""

    def __init__(self, price_feed=None):
        if not LIVE_MODE:
            raise RuntimeError("LiveBroker requires QS_LIVE_MODE=true")
        self.client  = BinanceClient()
        self.market  = price_feed or MarketData()
        self.safety  = LiveSafety()
        self._ex     = self.client.get_exchange()
        if not self._ex or not getattr(self._ex, "apiKey", None):
            raise RuntimeError("LiveBroker requires BINANCE_API_KEY and BINANCE_API_SECRET")

        self._leverage_set: set[str] = set()   # symbols where leverage already applied
        self._precision:    dict[str, dict]  = {}  # cached lot-size info

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        return self.market.get_price(symbol)

    def _log_order(self, data: dict) -> None:
        try:
            with _ORDER_LOG.open("a") as f:
                f.write(json.dumps({**data, "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
        except Exception:
            pass

    def _get_precision(self, symbol: str) -> dict:
        """Cache lot-size and price-step from Binance market info."""
        if symbol in self._precision:
            return self._precision[symbol]
        try:
            markets = self._ex.load_markets()
            info = markets.get(symbol, {})
            precision = {
                "amount": info.get("precision", {}).get("amount", 3),
                "price":  info.get("precision", {}).get("price",  2),
                "min_qty": float(info.get("limits", {}).get("amount", {}).get("min", 0.001) or 0.001),
            }
        except Exception:
            precision = {"amount": 3, "price": 2, "min_qty": 0.001}
        self._precision[symbol] = precision
        return precision

    def _round_size(self, symbol: str, size: float) -> float:
        prec = self._get_precision(symbol)
        decimals = int(prec["amount"])
        rounded = round(size, decimals)
        return max(rounded, prec["min_qty"])

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage once per symbol per session."""
        if symbol in self._leverage_set:
            return
        try:
            self._ex.set_leverage(leverage, symbol)
            self._leverage_set.add(symbol)
        except Exception as e:
            print(f"[LiveBroker] leverage {leverage}x for {symbol}: {e}")

    def _check_spread(self, symbol: str) -> bool:
        """Return True if spread is acceptable, False if too wide."""
        try:
            ob = self._ex.fetch_order_book(symbol, limit=1)
            bid = ob["bids"][0][0] if ob["bids"] else 0.0
            ask = ob["asks"][0][0] if ob["asks"] else 0.0
            if bid <= 0 or ask <= 0:
                return True  # no data → allow
            spread_pct = (ask - bid) / bid
            if spread_pct > MAX_SPREAD_PCT:
                print(f"[LiveBroker] {symbol} spread {spread_pct:.4%} > {MAX_SPREAD_PCT:.4%} — skip")
                return False
        except Exception:
            pass
        return True

    def _execute_with_retry(self, symbol: str, order_side: str,
                             size: float, params: dict) -> dict | None:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                order = self._ex.create_market_order(
                    symbol=symbol, side=order_side, amount=size, params=params
                )
                return order
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    print(f"[LiveBroker] retry {attempt}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY_SEC)
        print(f"[LiveBroker] all retries failed: {last_err}")
        return None

    # ── public API ────────────────────────────────────────────────────────────

    def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        equity: float | None = None,
        leverage: int = 2,
    ) -> dict | None:
        price = self.get_price(symbol)
        eq    = equity if equity is not None else 0.0

        ok, reason = self.safety.validate_order(symbol, side, size, price, eq)
        if not ok:
            print(f"[LiveBroker] safety blocked: {reason}")
            return None

        if not self._check_spread(symbol):
            return None

        self._set_leverage(symbol, leverage)
        size        = self._round_size(symbol, size)
        order_side  = "buy" if side == "LONG" else "sell"
        order       = self._execute_with_retry(
            symbol, order_side, size, {"reduceOnly": False}
        )
        if order is None:
            self._log_order({"symbol": symbol, "side": side, "size": size,
                             "status": "FAILED", "error": "retries exhausted"})
            return None

        fill = float(order.get("average") or order.get("price") or price)
        result = {
            "symbol":    symbol,
            "side":      side,
            "size":      float(order.get("filled", size)),
            "price":     round(fill, 8),
            "status":    order.get("status", "FILLED"),
            "timestamp": time.time(),
            "order_id":  order.get("id"),
        }
        self._log_order({**result, "direction": "OPEN"})
        return result

    def close_position(self, position: dict) -> float:
        symbol     = position["symbol"]
        size       = self._round_size(symbol, position["size"])
        close_side = "sell" if position["side"] == "LONG" else "buy"
        entry      = position["entry_price"]

        order = self._execute_with_retry(
            symbol, close_side, size, {"reduceOnly": True}
        )
        if order is None:
            # Emergency: fetch current price for PnL calculation even if close failed
            exit_price = self.get_price(symbol)
            print(f"[LiveBroker] WARN: close failed for {symbol} — using market price for PnL")
        else:
            exit_price = float(order.get("average") or order.get("price") or self.get_price(symbol))

        self._log_order({
            "symbol": symbol, "side": position["side"],
            "size": size, "exit_price": exit_price,
            "direction": "CLOSE", "status": order.get("status") if order else "FAILED",
        })

        if position["side"] == "LONG":
            return (exit_price - entry) * size
        return (entry - exit_price) * size
