"""Walk-forward backtest — rolling train/test windows."""

import numpy as np

from data.market_data import MarketData
from factors.factor_engine import FactorEngine
from config.settings import INITIAL_CAPITAL, RISK_PER_TRADE


class WalkForwardBacktest:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital = initial_capital
        self.factors = FactorEngine()

    def run(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 800,
        train_bars: int = 200,
        test_bars: int = 80,
        step_bars: int = 80,
    ) -> dict:
        market = MarketData()
        df = market.get_ohlcv_df(symbol, timeframe=timeframe, limit=limit)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        windows = []
        start = train_bars
        while start + test_bars < len(closes):
            train_slice = slice(start - train_bars, start)
            test_slice = slice(start, start + test_bars)
            windows.append((train_slice, test_slice))
            start += step_bars

        if not windows:
            return {"symbol": symbol, "windows": 0, "error": "not enough bars"}

        total_pnl = 0.0
        total_trades = 0
        wins = 0
        window_results = []

        for train_sl, test_sl in windows:
            train_c = closes[train_sl]
            momentum = self.factors.momentum_factor(train_c)
            trend = self.factors.trend_strength(train_c)
            bias = "LONG" if momentum > 0 and trend > 0 else "SHORT" if momentum < 0 and trend < 0 else "HOLD"

            pnl, trades, w = self._simulate_segment(
                closes[test_sl], highs[test_sl], lows[test_sl], bias
            )
            total_pnl += pnl
            total_trades += trades
            wins += w
            window_results.append({"pnl": round(pnl, 2), "trades": trades})

        return {
            "symbol": symbol,
            "windows": len(windows),
            "total_trades": total_trades,
            "winrate": round(wins / total_trades, 4) if total_trades else 0,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / self.capital * 100, 2),
            "avg_pnl_per_window": round(total_pnl / len(windows), 2),
            "windows_detail": window_results[-5:],
        }

    def _simulate_segment(self, closes, highs, lows, bias: str) -> tuple[float, int, int]:
        equity = self.capital
        position = None
        trades = wins = 0
        pnl_sum = 0.0

        for i in range(5, len(closes)):
            price = closes[i]
            if position:
                side = position["side"]
                if side == "LONG":
                    if lows[i] <= position["sl"]:
                        pnl = (position["sl"] - position["entry"]) * position["size"]
                        pnl_sum += pnl
                        trades += 1
                        wins += int(pnl > 0)
                        position = None
                    elif highs[i] >= position["tp"]:
                        pnl = (position["tp"] - position["entry"]) * position["size"]
                        pnl_sum += pnl
                        trades += 1
                        wins += int(pnl > 0)
                        position = None
                elif side == "SHORT":
                    if highs[i] >= position["sl"]:
                        pnl = (position["entry"] - position["sl"]) * position["size"]
                        pnl_sum += pnl
                        trades += 1
                        wins += int(pnl > 0)
                        position = None
                    elif lows[i] <= position["tp"]:
                        pnl = (position["entry"] - position["tp"]) * position["size"]
                        pnl_sum += pnl
                        trades += 1
                        wins += int(pnl > 0)
                        position = None

            if position is None and bias in ("LONG", "SHORT"):
                atr = max(price * 0.01, 0.0001)
                size = (equity * RISK_PER_TRADE) / (atr * 2)
                if bias == "LONG":
                    sl, tp = price - atr * 2, price + atr * 3.5
                else:
                    sl, tp = price + atr * 2, price - atr * 3.5
                position = {"side": bias, "entry": price, "size": size, "sl": sl, "tp": tp}

        return pnl_sum, trades, wins
