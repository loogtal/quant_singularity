"""Online weight nudge from closed trade outcomes."""

import numpy as np

from models.predictor import MLPredictor


class OnlineLearner:
    def __init__(self, lr: float = 0.02):
        self.predictor = MLPredictor()
        self.lr = lr

    def update(self, features: np.ndarray, won: bool) -> None:
        if self.predictor.weights is None or len(features) != len(self.predictor.weights):
            return

        target = 1.0 if won else 0.0
        prob = self.predictor.predict_proba(features)
        error = prob - target

        self.predictor.weights -= self.lr * error * features
        self.predictor.bias -= self.lr * error
        self.predictor.save_weights(self.predictor.weights, self.predictor.bias)
