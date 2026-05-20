"""Quick backtest — factor signals + ATR stops on historical bars."""

import argparse

import numpy as np

from config.settings import INITIAL_CAPITAL, RISK_PER_TRADE
from data.market_data import MarketData
from factors.factor_engine import FactorEngine


class QuickBacktest:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital = initial_capital
        self.equity = initial_capital
        self.factors = FactorEngine()
        self.trades: list[dict] = []

    def run(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> dict:
        df = MarketData().get_ohlcv_df(symbol, timeframe=timeframe, limit=limit)
        closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
        position = None
        equity_curve = [self.equity]

        for i in range(100, len(closes)):
            f = self.factors.compute(closes[: i + 1])
            direction = self.factors.direction_from_factors(f)
            price = closes[i]

            if position:
                side = position["side"]
                if side == "LONG":
                    if lows[i] <= position["sl"]:
                        self._close((position["sl"] - position["entry"]) * position["size"])
                        position = None
                    elif highs[i] >= position["tp"]:
                        self._close((position["tp"] - position["entry"]) * position["size"])
                        position = None
                elif highs[i] >= position["sl"]:
                    self._close((position["entry"] - position["sl"]) * position["size"])
                    position = None
                elif lows[i] <= position["tp"]:
                    self._close((position["entry"] - position["tp"]) * position["size"])
                    position = None

            if position is None and direction in ("LONG", "SHORT"):
                atr = self._atr(highs[: i + 1], lows[: i + 1], closes[: i + 1])
                if atr <= 0:
                    continue
                size = (self.equity * RISK_PER_TRADE) / (atr * 2)
                if direction == "LONG":
                    sl, tp = price - atr * 2, price + atr * 3.5
                else:
                    sl, tp = price + atr * 2, price - atr * 3.5
                position = {"side": direction, "entry": price, "size": size, "sl": sl, "tp": tp}

            equity_curve.append(self.equity)

        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        total = len(self.trades)
        total_pnl = sum(t["pnl"] for t in self.trades)
        rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = (
            (np.mean(rets) / np.std(rets) * np.sqrt(252 * 24 * 4))
            if len(rets) > 1 and np.std(rets) > 0
            else 0
        )
        return {
            "symbol": symbol,
            "trades": total,
            "wins": wins,
            "winrate": wins / total if total else 0,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round((self.equity - self.capital) / self.capital * 100, 2),
            "sharpe": round(float(sharpe), 2),
            "final_equity": round(self.equity, 2),
        }

    def _close(self, pnl: float) -> None:
        self.equity += pnl
        self.trades.append({"pnl": pnl})

    @staticmethod
    def _atr(highs, lows, closes, period: int = 14) -> float:
        trs = [
            max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
            for j in range(1, len(closes))
        ]
        return float(np.mean(trs[-period:])) if len(trs) >= period else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--bars", type=int, default=500)
    args = p.parse_args()
    r = QuickBacktest().run(args.symbol, limit=args.bars)
    print("\n=== BACKTEST ===")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
