"""
Feature Drift Detector for LightGBM predictor.

Detects when the live feature distribution has shifted from the training
distribution using Population Stability Index (PSI).

PSI < 0.10  → stable          (green)
PSI 0.10-0.25 → moderate drift (yellow — monitor)
PSI > 0.25  → major drift      (red — auto-retrain)

Also tracks rolling prediction accuracy over a sliding window:
  - accuracy < ACCURACY_THRESHOLD for WINDOW_SIZE predictions → trigger retrain

Both signals can independently trigger retrain.  The detector is conservative:
it requires N consecutive drift signals before firing to avoid false positives
from single-candle anomalies.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import STORAGE_DIR

_DRIFT_FILE = STORAGE_DIR / "drift_state.json"

PSI_YELLOW          = 0.10   # moderate drift
PSI_RED             = 0.25   # major drift → retrain
ACCURACY_THRESHOLD  = 0.50   # rolling accuracy threshold
WINDOW_SIZE         = 50     # rolling accuracy window
MIN_CONSECUTIVE     = 3      # consecutive red signals before retrain fires
RETRAIN_COOLDOWN    = 4 * 3600   # minimum seconds between retrains (4 h)


def _psi_bin(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between reference and current distributions.
    Returns PSI value.  Higher = more drift.
    """
    combined = np.concatenate([reference, current])
    bins     = np.linspace(np.min(combined), np.max(combined), n_bins + 1)
    bins[0]  -= 1e-9
    bins[-1] += 1e-9

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current,   bins=bins)

    ref_pct = (ref_hist + 0.5) / (len(reference) + 0.5 * n_bins)
    cur_pct = (cur_hist + 0.5) / (len(current)   + 0.5 * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct + 1e-9)))
    return max(0.0, psi)


class DriftDetector:
    """
    Monitors feature distribution and prediction accuracy to trigger retrains.

    Usage:
        detector = DriftDetector()
        detector.set_reference(X_train)           # called once after training
        psi = detector.update(X_new, prediction)   # called each bar
        if detector.should_retrain():
            lgbm.train_multi(...)
            detector.reset()
    """

    def __init__(self) -> None:
        self._reference: Optional[np.ndarray] = None    # training feature matrix
        self._recent:    list[np.ndarray]     = []      # last N feature vectors
        self._recent_maxlen                   = 500

        # Rolling accuracy window
        self._predictions: deque[int]   = deque(maxlen=WINDOW_SIZE)  # 1 correct, 0 wrong
        self._psi_history: deque[float] = deque(maxlen=100)

        self._consecutive_red: int  = 0
        self._last_retrain_ts: float = 0.0
        self._last_psi: float        = 0.0
        self._drift_status: str      = "unknown"

        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _DRIFT_FILE.exists():
            return
        try:
            data = json.loads(_DRIFT_FILE.read_text())
            self._last_retrain_ts = float(data.get("last_retrain_ts", 0.0))
            self._consecutive_red = int(data.get("consecutive_red", 0))
            ref = data.get("reference_stats")
            if ref:
                self._reference_mean = np.array(ref["mean"], dtype=np.float32)
                self._reference_std  = np.array(ref["std"],  dtype=np.float32)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            payload: dict = {
                "last_retrain_ts": self._last_retrain_ts,
                "consecutive_red": self._consecutive_red,
                "last_psi":        round(self._last_psi, 4),
                "drift_status":    self._drift_status,
                "saved_at":        datetime.now(timezone.utc).isoformat(),
            }
            if self._reference is not None:
                payload["reference_stats"] = {
                    "mean": self._reference.mean(axis=0).tolist(),
                    "std":  self._reference.std(axis=0).tolist(),
                }
            _DRIFT_FILE.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def set_reference(self, X: np.ndarray) -> None:
        """Store training feature distribution as the reference baseline."""
        if len(X) < 50:
            return
        self._reference = np.array(X, dtype=np.float32)
        self._save()

    def update(self, features: np.ndarray, correct: Optional[bool] = None) -> float:
        """
        Add a new live feature vector.  Optionally pass whether last prediction was correct.
        Returns current PSI across all features.
        """
        if features is None or len(features) == 0:
            return self._last_psi

        self._recent.append(np.array(features, dtype=np.float32).flatten())
        if len(self._recent) > self._recent_maxlen:
            self._recent = self._recent[-self._recent_maxlen:]

        if correct is not None:
            self._predictions.append(1 if correct else 0)

        if self._reference is None or len(self._recent) < 50:
            return 0.0

        # Compute per-feature PSI and take mean
        recent_arr = np.stack(self._recent[-100:], axis=0)
        psi_vals   = []
        n_features = min(self._reference.shape[1], recent_arr.shape[1])
        for i in range(n_features):
            ref_col = self._reference[:, i]
            cur_col = recent_arr[:, i]
            if np.std(ref_col) < 1e-9:
                continue
            psi_vals.append(_psi_bin(ref_col, cur_col))

        psi = float(np.mean(psi_vals)) if psi_vals else 0.0
        self._last_psi = psi
        self._psi_history.append(psi)

        # Update drift status
        if psi > PSI_RED:
            self._consecutive_red += 1
            self._drift_status = "red"
        elif psi > PSI_YELLOW:
            self._consecutive_red = max(0, self._consecutive_red - 1)
            self._drift_status = "yellow"
        else:
            self._consecutive_red = 0
            self._drift_status = "green"

        return psi

    def rolling_accuracy(self) -> float:
        if not self._predictions:
            return 1.0
        return float(np.mean(list(self._predictions)))

    def should_retrain(self) -> bool:
        """Returns True when drift is severe enough and cooldown has passed."""
        if time.time() - self._last_retrain_ts < RETRAIN_COOLDOWN:
            return False
        # PSI-based trigger
        if self._consecutive_red >= MIN_CONSECUTIVE:
            return True
        # Accuracy-based trigger (only after enough predictions)
        if len(self._predictions) >= WINDOW_SIZE:
            if self.rolling_accuracy() < ACCURACY_THRESHOLD:
                return True
        return False

    def reset(self) -> None:
        """Call after a successful retrain."""
        self._last_retrain_ts  = time.time()
        self._consecutive_red  = 0
        self._recent.clear()
        self._predictions.clear()
        self._drift_status = "green"
        self._save()

    def snapshot(self) -> dict:
        return {
            "psi":               round(self._last_psi, 4),
            "drift_status":      self._drift_status,
            "consecutive_red":   self._consecutive_red,
            "rolling_accuracy":  round(self.rolling_accuracy(), 4),
            "should_retrain":    self.should_retrain(),
            "last_retrain_ts":   self._last_retrain_ts,
            "predictions_window": len(self._predictions),
        }
