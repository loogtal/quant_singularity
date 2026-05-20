"""Train logistic model on historical OHLCV bars."""

import numpy as np

from data.market_data import MarketData
from models.predictor import MLPredictor


class ModelTrainer:
    def __init__(self):
        self.predictor = MLPredictor()
        self.market = MarketData()

    def build_dataset(
        self, symbol: str, timeframe: str = "15m", limit: int = 800
    ) -> tuple[np.ndarray, np.ndarray] | None:
        df = self.market.get_ohlcv_df(symbol, timeframe=timeframe, limit=limit)
        closes = df["close"].values
        volumes = df["volume"].values if "volume" in df else None

        xs, ys = [], []
        for i in range(50, len(closes) - 5):
            window = closes[: i + 1]
            vol_w = volumes[: i + 1] if volumes is not None else None
            x = self.predictor.extract_features(window, vol_w)
            future_ret = (closes[i + 5] - closes[i]) / closes[i]
            y = 1.0 if future_ret > 0.001 else 0.0
            xs.append(x)
            ys.append(y)

        if len(xs) < 80:
            return None
        return np.array(xs), np.array(ys)

    def train(
        self,
        symbol: str = "BTC/USDT:USDT",
        epochs: int = 200,
        lr: float = 0.05,
    ) -> dict:
        data = self.build_dataset(symbol)
        if data is None:
            return {"ok": False, "reason": "insufficient data"}

        X, y = data
        n_features = X.shape[1]
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(epochs):
            z = X @ w + b
            pred = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
            error = pred - y
            w -= lr * (X.T @ error) / len(y)
            b -= lr * float(np.mean(error))

        self.predictor.save_weights(w, b)

        pred = 1 / (1 + np.exp(-np.clip(X @ w + b, -20, 20)))
        acc = float(np.mean((pred > 0.5) == y))

        return {
            "ok": True,
            "symbol": symbol,
            "samples": len(y),
            "accuracy": round(acc, 4),
            "weights": w.tolist(),
        }
