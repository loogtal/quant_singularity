"""Realtime prices — WebSocket first, REST fallback."""

from data.market_data import MarketData
from data.websocket_feed import WebSocketFeed
from config.settings import USE_WEBSOCKET


class RealtimeFeed:
    def __init__(self):
        self.market = MarketData()
        self.ws = WebSocketFeed()
        self._ws_started = False

    def ensure_started(self) -> None:
        if USE_WEBSOCKET and not self._ws_started:
            self._ws_started = self.ws.start()
            if self._ws_started:
                print("[Realtime] WebSocket feed started")

    def get_price(self, symbol: str) -> float:
        self.ensure_started()
        if USE_WEBSOCKET:
            px = self.ws.get_price(symbol)
            if px is not None:
                return px
        return self.market.get_price(symbol)
