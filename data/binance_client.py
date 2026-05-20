"""Binance USDT-M futures client via CCXT."""

import os
import time
from typing import Optional

import ccxt

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TESTNET,
    LIVE_MODE,
)


class BinanceClient:
    _exchange = None

    def __init__(self):
        if BinanceClient._exchange is None:
            BinanceClient._exchange = self._connect()

    def _connect(self):
        opts = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if LIVE_MODE and BINANCE_API_KEY and BINANCE_API_SECRET:
            opts["apiKey"] = BINANCE_API_KEY
            opts["secret"] = BINANCE_API_SECRET
        if LIVE_MODE and BINANCE_TESTNET:
            opts["sandbox"] = True

        try:
            print("\n[Binance] Loading USDT-M futures markets...")
            exchange = ccxt.binanceusdm(opts)
            exchange.load_markets()
            print("[Binance] Connected — USDT-M futures")
            if LIVE_MODE and BINANCE_API_KEY:
                print("[Binance] API keys loaded — live mode enabled")
            return exchange
        except Exception as e:
            print(f"\n[Binance] Connection failed: {e}")
            return None

    @property
    def connected(self) -> bool:
        return BinanceClient._exchange is not None

    def get_exchange(self):
        return BinanceClient._exchange

    def get_ticker(self, symbol: str) -> dict:
        ex = BinanceClient._exchange
        if ex:
            try:
                return ex.fetch_ticker(symbol)
            except Exception:
                pass
        raise ConnectionError(f"No ticker for {symbol}")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 100,
    ) -> list:
        ex = BinanceClient._exchange
        if ex:
            try:
                return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception:
                pass
        raise ConnectionError(f"No OHLCV for {symbol}")

    def list_usdt_futures(self) -> list[str]:
        """Active USDT perpetual symbols sorted by 24h quote volume."""
        ex = BinanceClient._exchange
        if not ex:
            return [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "SOL/USDT:USDT",
                "BNB/USDT:USDT",
                "XRP/USDT:USDT",
            ]

        symbols = []
        for sym, market in ex.markets.items():
            if not market.get("active"):
                continue
            if market.get("quote") != "USDT":
                continue
            if not market.get("linear", True):
                continue
            symbols.append(sym)

        try:
            tickers = ex.fetch_tickers(symbols[:80])
            ranked = sorted(
                tickers.items(),
                key=lambda x: float(x[1].get("quoteVolume") or 0),
                reverse=True,
            )
            return [s for s, _ in ranked[:50]]
        except Exception:
            return symbols[:30]

    def fetch_balance_usdt(self) -> float:
        ex = BinanceClient._exchange
        if not ex or not LIVE_MODE:
            return 0.0
        try:
            bal = ex.fetch_balance()
            return float(bal.get("USDT", {}).get("free", 0))
        except Exception as e:
            print(f"[Binance] Balance fetch failed: {e}")
            return 0.0
