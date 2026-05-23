"""Online weight nudge from closed trade outcomes."""

import numpy as np

from models.predictor import MLPredictor


class OnlineLearner:
    def __init__(self, lr: float = 0.02):
        self.predictor = MLPredictor()
        self.lr = lr

    def update(self, features, won: bool) -> None:
        if self.predictor.weights is None:
            return

        feats = np.asarray(features, dtype=float)
        if len(feats) != len(self.predictor.weights):
            return

        target = 1.0 if won else 0.0
        prob = self.predictor.predict_proba(feats)
        error = prob - target

        self.predictor.weights -= self.lr * error * feats
        self.predictor.bias -= self.lr * error
        self.predictor.save_weights(self.predictor.weights, self.predictor.bias)
