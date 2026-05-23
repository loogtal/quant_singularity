"""ML direction predictor — logistic regression (numpy), no sklearn required."""

import json

import numpy as np

from config.settings import MODEL_DIR
from factors.factor_engine import FactorEngine

MODEL_FILE = MODEL_DIR / "logistic_weights.json"


class MLPredictor:
    def __init__(self):
        self.factors = FactorEngine()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.trained = False
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not MODEL_FILE.exists():
            return
        try:
            data = json.loads(MODEL_FILE.read_text())
            self.weights = np.array(data["weights"], dtype=float)
            self.bias = float(data.get("bias", 0))
            self.trained = True
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def extract_features(self, closes: np.ndarray, volumes: np.ndarray | None = None) -> np.ndarray:
        f = self.factors.compute(closes, volumes)
        mom_n = self.factors.normalize(f["momentum"], -0.05, 0.05)
        trend_n = self.factors.normalize(f["trend"], -0.03, 0.03)
        vol_n = self.factors.normalize(f["volatility"], 0, 0.03)
        ret = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0
        ret_n = self.factors.normalize(ret, -0.03, 0.03)

        rsi_n = self._rsi_feature(closes)
        macd_n = self._macd_feature(closes)
        bb_pos = self._bb_position(closes)
        obv_trend = self._obv_trend(closes, volumes)

        return np.array([mom_n, trend_n, vol_n, ret_n, rsi_n, macd_n, bb_pos, obv_trend, 1.0])

    @staticmethod
    def _rsi_feature(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.5
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(np.clip(rsi / 100, 0, 1))

    @staticmethod
    def _macd_feature(closes: np.ndarray) -> float:
        if len(closes) < 30:
            return 0.5
        ema12 = float(np.mean(closes[-12:]))
        ema26 = float(np.mean(closes[-26:]))
        price = closes[-1]
        if price <= 0:
            return 0.5
        macd_pct = (ema12 - ema26) / price
        return float(np.clip(macd_pct / 0.02 + 0.5, 0, 1))

    @staticmethod
    def _bb_position(closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.5
        window = closes[-period:]
        mid = float(np.mean(window))
        std = float(np.std(window))
        if std <= 0:
            return 0.5
        upper = mid + 2 * std
        lower = mid - 2 * std
        return float(np.clip((closes[-1] - lower) / (upper - lower), 0, 1))

    @staticmethod
    def _obv_trend(closes: np.ndarray, volumes: np.ndarray | None, lookback: int = 20) -> float:
        if volumes is None or len(volumes) < lookback or len(closes) < lookback:
            return 0.5
        direction = np.sign(np.diff(closes[-lookback:]))
        obv = np.cumsum(direction * volumes[-(lookback - 1):])
        if len(obv) < 5:
            return 0.5
        slope = (obv[-1] - obv[0]) / max(abs(obv).max(), 1)
        return float(np.clip(slope + 0.5, 0, 1))

    def predict_proba(self, features: np.ndarray) -> float:
        if self.weights is None or len(self.weights) != len(features):
            # Untrained: momentum heuristic
            return float(np.clip(0.5 + (features[0] - 0.5) * 0.8, 0.05, 0.95))
        z = float(np.dot(self.weights, features) + self.bias)
        return float(1 / (1 + np.exp(-np.clip(z, -20, 20))))

    def predict(self, closes: np.ndarray, volumes: np.ndarray | None = None) -> dict:
        if len(closes) < 30:
            return {"direction": "HOLD", "confidence": 0.0, "prob_up": 0.5}

        x = self.extract_features(closes, volumes)
        prob = self.predict_proba(x)

        if prob > 0.58:
            direction = "LONG"
            confidence = prob
        elif prob < 0.42:
            direction = "SHORT"
            confidence = 1 - prob
        else:
            direction = "HOLD"
            confidence = abs(prob - 0.5) * 2

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "prob_up": round(prob, 4),
            "trained": self.trained,
        }

    def save_weights(self, weights: np.ndarray, bias: float) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_FILE.write_text(
            json.dumps({
                "weights": weights.tolist(),
                "bias": bias,
            }, indent=2)
        )
        self.weights = weights
        self.bias = bias
        self.trained = True


# Backward compat
Predictor = MLPredictor
