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

from config.settings import LIVE_MODE, STORAGE_DIR, _env_bool, _env_float
from data.binance_client import BinanceClient
from data.market_data import MarketData
from risk.live_safety import LiveSafety

_ORDER_LOG = STORAGE_DIR / "orders.jsonl"

MAX_SPREAD_PCT  = 0.0008   # skip if spread > 0.08%
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2.0

# Maker (post-only limit) entries — OFF by default. Validation showed execution
# cost (taker fee + slippage) dominates the thin edge; maker fills cut that, but a
# resting limit can MISS (price runs away), so this is opt-in and TESTNET-FIRST.
# On a miss we SKIP the trade rather than fall back to taker (that would defeat it).
MAKER_ENTRY        = _env_bool("QS_MAKER_ENTRY", False)
MAKER_TIMEOUT_SEC  = _env_float("QS_MAKER_TIMEOUT_SEC", 8.0)
MAKER_POLL_SEC     = 1.0


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

    def _maker_entry(self, symbol: str, order_side: str, size: float,
                     params: dict) -> dict | None:
        """
        Place a post-only limit at the near touch and wait up to MAKER_TIMEOUT_SEC.
        Returns the filled order, or None if it never fills (caller SKIPS the trade —
        we do NOT fall back to a taker market order, which would erase the cost saving).
        """
        try:
            ob = self._ex.fetch_order_book(symbol, limit=1)
            bid = ob["bids"][0][0] if ob["bids"] else 0.0
            ask = ob["asks"][0][0] if ob["asks"] else 0.0
            if bid <= 0 or ask <= 0:
                return None
            # Rest on our own side so the order is a maker (won't cross the spread).
            px = bid if order_side == "buy" else ask
            px = float(self._ex.price_to_precision(symbol, px))
            order = self._ex.create_order(
                symbol, "limit", order_side, size, px,
                params={**params, "postOnly": True},
            )
        except Exception as e:
            print(f"[LiveBroker] maker entry rejected for {symbol}: {e}")
            return None

        oid = order.get("id")
        waited = 0.0
        while waited < MAKER_TIMEOUT_SEC:
            time.sleep(MAKER_POLL_SEC)
            waited += MAKER_POLL_SEC
            try:
                o = self._ex.fetch_order(oid, symbol)
            except Exception:
                continue
            status = (o.get("status") or "").lower()
            if status in ("closed", "filled") and float(o.get("filled", 0)) > 0:
                return o
            if status in ("canceled", "cancelled", "rejected", "expired"):
                return None
        # Timed out unfilled → cancel and skip
        self.cancel_order(symbol, oid)
        print(f"[LiveBroker] maker entry for {symbol} unfilled in {MAKER_TIMEOUT_SEC:.0f}s — skip")
        return None

    # ── stop-order management (catastrophic backstop) ──────────────────────────
    #
    # We place ONE reduceOnly STOP_MARKET order at the position's initial
    # stop-loss level so an open position is never fully naked on the exchange
    # during a bot/network outage. The app-level trailing stop in
    # TradeManager remains the primary, tighter exit mechanism and runs every
    # cycle — this backstop is a dead-man's-switch at the original level, not
    # kept in sync with every trailing update (that would mean cancel/replace
    # every ~15s, which is its own source of risk).

    def place_stop_order(self, symbol: str, side: str, size: float,
                          stop_price: float | None) -> str | None:
        if not stop_price:
            return None
        close_side = "sell" if side == "LONG" else "buy"
        size = self._round_size(symbol, size)
        try:
            order = self._ex.create_order(
                symbol, "STOP_MARKET", close_side, size,
                params={
                    "stopPrice":  round(stop_price, 8),
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            self._log_order({
                "symbol": symbol, "side": side, "size": size,
                "stop_price": stop_price, "direction": "STOP_PLACED",
                "order_id": order.get("id"), "status": order.get("status", "NEW"),
            })
            return order.get("id")
        except Exception as e:
            print(f"[LiveBroker] failed to place stop order for {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str | None) -> None:
        if not order_id:
            return
        try:
            self._ex.cancel_order(order_id, symbol)
            self._log_order({"symbol": symbol, "order_id": order_id, "direction": "STOP_CANCELLED"})
        except Exception as e:
            print(f"[LiveBroker] failed to cancel order {order_id} for {symbol}: {e}")

    # ── public API ────────────────────────────────────────────────────────────

    def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        equity: float | None = None,
        leverage: int = 2,
        stop_loss: float | None = None,
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
        if MAKER_ENTRY:
            order = self._maker_entry(symbol, order_side, size, {"reduceOnly": False})
            if order is None:
                # Maker miss = skip (no taker fallback). Not a failure — just no fill.
                self._log_order({"symbol": symbol, "side": side, "size": size,
                                 "status": "SKIPPED", "error": "maker entry unfilled"})
                return None
        else:
            order = self._execute_with_retry(
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

        # Catastrophic backstop: reduceOnly STOP_MARKET at the initial stop level
        result["sl_order_id"] = self.place_stop_order(symbol, side, result["size"], stop_loss)
        return result

    def close_position(self, position: dict) -> float:
        symbol     = position["symbol"]
        self.cancel_order(symbol, position.get("sl_order_id"))
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
