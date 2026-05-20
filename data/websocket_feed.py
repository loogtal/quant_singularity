"""Binance futures WebSocket — live mark prices (background thread)."""

import json
import threading
import time

from config.settings import WS_SYMBOLS_LIMIT

try:
    import websocket
except ImportError:
    websocket = None


class WebSocketFeed:
    """Subscribes to !markPrice@arr stream for fast price updates."""

    STREAM_URL = "wss://fstream.binance.com/ws/!markPrice@arr"

    def __init__(self):
        self._prices: dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_msg = 0.0

    @property
    def connected(self) -> bool:
        return self._running and (time.time() - self._last_msg) < 60

    def start(self) -> bool:
        if websocket is None:
            print("[WS] Install websocket-client: pip install websocket-client")
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    self.STREAM_URL,
                    on_message=self._on_message,
                    on_error=lambda ws, e: None,
                    on_close=lambda ws, c, m: None,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[WS] reconnect: {e}")
            if self._running:
                time.sleep(3)

    def _on_message(self, ws, message: str) -> None:
        self._last_msg = time.time()
        try:
            payload = json.loads(message)
            items = payload if isinstance(payload, list) else [payload]
            with self._lock:
                for item in items:
                    sym = item.get("s", "")
                    if not sym.endswith("USDT"):
                        continue
                    ccxt_sym = f"{sym.replace('USDT', '')}/USDT:USDT"
                    price = float(item.get("p") or item.get("markPrice") or 0)
                    if price > 0:
                        self._prices[ccxt_sym] = price
                        if len(self._prices) > WS_SYMBOLS_LIMIT * 2:
                            break
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    def get_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._prices.get(symbol)

    def get_all_prices(self) -> dict[str, float]:
        with self._lock:
            return dict(self._prices)
