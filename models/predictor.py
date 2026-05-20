"""ML direction predictor — logistic regression (numpy), no sklearn required."""

import json
from pathlib import Path

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
        return np.array([mom_n, trend_n, vol_n, ret_n, 1.0])

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
