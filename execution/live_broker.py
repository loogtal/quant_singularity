"""Live Binance futures execution — QS_LIVE_MODE=true + API keys required."""

import time

from config.settings import LIVE_MODE
from data.binance_client import BinanceClient
from data.market_data import MarketData
from risk.live_safety import LiveSafety


class LiveBroker:
    def __init__(self, price_feed=None):
        if not LIVE_MODE:
            raise RuntimeError("LiveBroker requires QS_LIVE_MODE=true")
        self.client = BinanceClient()
        self.market = price_feed or MarketData()
        self.safety = LiveSafety()
        ex = self.client.get_exchange()
        if not ex or not getattr(ex, "apiKey", None):
            raise RuntimeError("LiveBroker requires BINANCE_API_KEY and BINANCE_API_SECRET")

    def get_price(self, symbol: str) -> float:
        return self.market.get_price(symbol)

    def execute_order(
        self, symbol: str, side: str, size: float, equity: float | None = None
    ) -> dict | None:
        ex = self.client.get_exchange()
        if not ex:
            return None
        price = self.get_price(symbol)
        eq = equity if equity is not None else 0.0
        ok, reason = self.safety.validate_order(symbol, side, size, price, eq)
        if not ok:
            print(f"[LiveBroker] Blocked: {reason}")
            return None
        order_side = "buy" if side == "LONG" else "sell"
        try:
            order = ex.create_market_order(
                symbol=symbol,
                side=order_side,
                amount=size,
                params={"reduceOnly": False},
            )
            fill = float(order.get("average") or order.get("price") or self.get_price(symbol))
            return {
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": round(fill, 6),
                "status": order.get("status", "FILLED"),
                "timestamp": time.time(),
                "order_id": order.get("id"),
            }
        except Exception as e:
            print(f"[LiveBroker] Order failed: {e}")
            return None

    def close_position(self, position: dict) -> float:
        ex = self.client.get_exchange()
        if not ex:
            return 0.0
        symbol = position["symbol"]
        size = position["size"]
        close_side = "sell" if position["side"] == "LONG" else "buy"
        try:
            ex.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=size,
                params={"reduceOnly": True},
            )
        except Exception as e:
            print(f"[LiveBroker] Close failed: {e}")
        exit_price = self.get_price(symbol)
        entry = position["entry_price"]
        if position["side"] == "LONG":
            return (exit_price - entry) * size
        return (entry - exit_price) * size
