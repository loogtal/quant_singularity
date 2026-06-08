"""
LightGBM direction predictor — 18-feature regime-conditional classifier.

Features: momentum (1/5/10/20-bar), EMA ratios (9/21/50), EMA crossover,
          RSI-14, Bollinger Band position, ATR%, volume ratio, session hour (sin/cos),
          binary flags (above EMA50, fast>slow, RSI>50).

Labels:  next-3-bar return > +0.1% = LONG (2), < -0.1% = SHORT (0), else HOLD (1).

One model per regime (bull/bear/sideways/volatile) + a global fallback.
Auto-retrain triggered when holdout accuracy drops below ACCURACY_THRESHOLD.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import STORAGE_DIR

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

MODEL_DIR   = STORAGE_DIR / "models" / "lgbm"
ACCURACY_THRESHOLD = 0.52
MIN_TRAIN_SAMPLES  = 200
FORWARD_BARS       = 6       # 1.5 h at 15m — clearer directional signal
LABEL_THRESHOLD    = 0.002   # 0.2% move — reduces noise in low-vol regimes
BARS_PER_DAY       = 96      # 15m candles: 96 per day

REGIMES = ("global", "bull", "bear", "sideways", "volatile")
CLASSES = ("SHORT", "HOLD", "LONG")   # label 0, 1, 2

FEATURE_NAMES = [
    # ── price momentum ──────────────────────────────────────
    "mom1", "mom5", "mom10", "mom20",
    # ── trend structure ─────────────────────────────────────
    "ema9_ratio", "ema21_ratio", "ema50_ratio",
    "cross9_21", "trend21_50",
    # ── oscillators ─────────────────────────────────────────
    "rsi14_norm", "bb_pos",
    # ── volatility ──────────────────────────────────────────
    "atr_pct",
    "atr_pct_pctile",   # NEW: ATR relative to its own 100-bar history (0-1)
    # ── volume ──────────────────────────────────────────────
    "vol_ratio",
    "vol_trend",        # NEW: is volume rising? (slope over last 10 bars, normalised)
    # ── time ────────────────────────────────────────────────
    "hour_sin", "hour_cos",
    # ── binary flags ────────────────────────────────────────
    "above_ema50", "fast_above_slow", "rsi_above_50",
    # ── vol-adjusted momentum ────────────────────────────────
    "mom5_atr_norm",    # NEW: 5-bar return / ATR — true signal-to-noise ratio
    "mom20_atr_norm",   # NEW: 20-bar return / ATR
]
N_FEATURES = len(FEATURE_NAMES)   # 22


# ── feature helpers ───────────────────────────────────────────────────────────

def _ema_scalar(arr: np.ndarray, period: int) -> float:
    if len(arr) == 0:
        return float(arr[-1]) if len(arr) else 0.0
    k = 2.0 / (period + 1)
    v = float(arr[0])
    for x in arr[1:]:
        v = float(x) * k + v * (1 - k)
    return v


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = float(np.mean(gains))  + 1e-9
    al = float(np.mean(losses)) + 1e-9
    return float(100 - 100 / (1 + ag / al))


def _bb_position(closes: np.ndarray, period: int = 20) -> float:
    if len(closes) < period:
        return 0.5
    w   = closes[-period:]
    mid = float(np.mean(w))
    std = float(np.std(w))
    if std < 1e-9:
        return 0.5
    pos = (float(closes[-1]) - (mid - 2 * std)) / (4 * std + 1e-9)
    return float(np.clip(pos, 0.0, 1.0)) - 0.5   # centred at 0


def extract_features(
    closes: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    hour: Optional[int] = None,
) -> np.ndarray:
    if hour is None:
        hour = datetime.now(timezone.utc).hour

    last = float(closes[-1])

    def safe_ret(n: int) -> float:
        if len(closes) < n or closes[-n] == 0:
            return 0.0
        return float((last - closes[-n]) / closes[-n])

    mom1, mom5, mom10, mom20 = safe_ret(2), safe_ret(5), safe_ret(10), safe_ret(20)

    ema9  = _ema_scalar(closes, 9)
    ema21 = _ema_scalar(closes, 21)
    ema50 = _ema_scalar(closes, 50)

    ema9_ratio  = (last  - ema9)  / (ema9  + 1e-9)
    ema21_ratio = (last  - ema21) / (ema21 + 1e-9)
    ema50_ratio = (last  - ema50) / (ema50 + 1e-9)
    cross9_21   = (ema9  - ema21) / (ema21 + 1e-9)
    trend21_50  = (ema21 - ema50) / (ema50 + 1e-9)

    rsi14_norm = (_rsi(closes, 14) - 50.0) / 50.0
    bb_pos     = _bb_position(closes)

    # Current ATR (std of last 20 returns)
    rets    = np.diff(closes[-21:]) / closes[-21:-1] if len(closes) >= 21 else np.array([0.0])
    atr_pct = float(np.std(rets)) if len(rets) > 1 else 0.001

    # ATR percentile: where is current ATR vs last 100 periods?
    # Values near 1 = unusually volatile; near 0 = unusually calm.
    atr_pct_pctile = 0.5
    if len(closes) >= 121:
        atr_history = []
        for j in range(1, 101):
            window = closes[-(21 + j):-(j)] if j > 0 else closes[-21:]
            if len(window) >= 2:
                r = np.diff(window) / window[:-1]
                atr_history.append(float(np.std(r)))
        if atr_history:
            atr_pct_pctile = float(np.mean(np.array(atr_history) <= atr_pct))

    # Volume ratio (current bar vs 20-bar avg)
    vol_ratio = 0.0
    if volumes is not None and len(volumes) >= 20:
        avg = float(np.mean(volumes[-20:]))
        if avg > 0:
            vol_ratio = float(volumes[-1] / avg) - 1.0

    # Volume trend: is volume increasing? (OLS slope of last 10 bars, normalised)
    vol_trend = 0.0
    if volumes is not None and len(volumes) >= 10:
        v10 = np.array(volumes[-10:], dtype=float)
        v_mean = float(np.mean(v10)) + 1e-9
        x = np.arange(10, dtype=float) - 4.5
        vol_trend = float(np.dot(x, v10 / v_mean) / float(np.dot(x, x) + 1e-9))
        vol_trend = float(np.clip(vol_trend, -2.0, 2.0))

    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))

    # Volatility-adjusted momentum: return / ATR — true signal-to-noise ratio
    # High value = strong move relative to noise; model can trust it more.
    mom5_atr_norm  = float(np.clip(mom5  / (atr_pct + 1e-9), -5.0, 5.0))
    mom20_atr_norm = float(np.clip(mom20 / (atr_pct + 1e-9), -5.0, 5.0))

    return np.array([
        mom1, mom5, mom10, mom20,
        ema9_ratio, ema21_ratio, ema50_ratio,
        cross9_21, trend21_50,
        rsi14_norm, bb_pos,
        atr_pct, atr_pct_pctile,
        vol_ratio, vol_trend,
        hour_sin, hour_cos,
        float(last > ema50),
        float(ema9 > ema21),
        float(rsi14_norm > 0),
        mom5_atr_norm, mom20_atr_norm,
    ], dtype=np.float32)


# ── dataset builder ───────────────────────────────────────────────────────────

def _build_dataset(
    closes: np.ndarray,
    volumes: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(200, len(closes) - FORWARD_BARS):
        # Cycle through 24h using bar index — each 15m bar advances 15 min.
        # Offset by current hour so alignment matches live data approximately.
        hour = (i % BARS_PER_DAY) // 4
        feat    = extract_features(closes[:i], volumes[:i] if volumes is not None else None, hour)
        fwd_ret = (closes[i + FORWARD_BARS] - closes[i]) / (closes[i] + 1e-9)
        if fwd_ret > LABEL_THRESHOLD:
            label = 2
        elif fwd_ret < -LABEL_THRESHOLD:
            label = 0
        else:
            label = 1
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── predictor class ───────────────────────────────────────────────────────────

class LGBMPredictor:
    """
    Drop-in upgrade over MLPredictor.
    Uses LightGBM when available; falls back to HOLD if library missing.
    One model per regime + global fallback.
    """

    def __init__(self):
        self._models:   dict[str, "lgb.Booster"] = {}
        self._accuracy: dict[str, float]          = {}
        self.trained = False
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if _LGB_OK:
            self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _model_path(self, regime: str) -> Path:
        return MODEL_DIR / f"lgbm_{regime}.txt"

    def _meta_path(self) -> Path:
        return MODEL_DIR / "lgbm_meta.json"

    def _load(self) -> None:
        meta = self._meta_path()
        if meta.exists():
            try:
                self._accuracy = json.loads(meta.read_text()).get("accuracy", {})
            except Exception:
                pass
        for regime in REGIMES:
            p = self._model_path(regime)
            if p.exists():
                try:
                    self._models[regime] = lgb.Booster(model_file=str(p))
                    self.trained = True
                except Exception:
                    pass

    def _save_meta(self) -> None:
        self._meta_path().write_text(json.dumps({"accuracy": self._accuracy}, indent=2))

    # ── training ─────────────────────────────────────────────────────────────

    def _fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime: str,
    ) -> float:
        """
        Core LightGBM fit on pre-built (X, y).
        Splits 80/20, trains with early stopping, saves model.
        Returns validation accuracy.
        """
        if len(X) < MIN_TRAIN_SAMPLES:
            return 0.0

        # Temporal split: last 20% as holdout (preserves time-order)
        split = int(len(X) * 0.80)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        params = {
            "objective":        "multiclass",
            "num_class":        3,
            "metric":           "multi_logloss",
            "num_leaves":       31,
            "learning_rate":    0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "min_data_in_leaf": 20,
            "verbose":          -1,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        cbs    = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]

        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=cbs,
        )

        preds    = np.argmax(model.predict(X_val), axis=1)
        accuracy = float(np.mean(preds == y_val))

        model.save_model(str(self._model_path(regime)))
        self._models[regime] = model
        self._accuracy[regime] = round(accuracy, 4)
        self.trained = True
        self._save_meta()
        return accuracy

    def train(
        self,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        regime: str = "global",
    ) -> float:
        """Train on a single symbol's OHLCV series. Returns holdout accuracy."""
        if not _LGB_OK:
            return 0.0
        X, y = _build_dataset(closes, volumes)
        return self._fit_model(X, y, regime)

    def train_multi(
        self,
        data: list[tuple[np.ndarray, Optional[np.ndarray]]],
        regime: str = "global",
    ) -> float:
        """
        Train on data from multiple symbols concatenated and shuffled.

        Providing BTC+ETH+SOL+BNB trains a more generalised model that handles
        altcoin regimes the BTC-only model never sees, improving accuracy on
        all active trades.

        data : list of (closes, volumes) tuples — one per symbol.
        Returns holdout accuracy on the combined shuffled dataset.
        """
        if not _LGB_OK:
            return 0.0

        all_X, all_y = [], []
        for closes, volumes in data:
            X, y = _build_dataset(closes, volumes)
            if len(X) >= MIN_TRAIN_SAMPLES // 2:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return 0.0

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Shuffle to prevent the model from learning "this is always BTC then always ETH"
        rng = np.random.default_rng(seed=42)
        idx = rng.permutation(len(X_combined))
        return self._fit_model(X_combined[idx], y_combined[idx], regime)

    # ── inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        regime: str = "global",
    ) -> dict:
        fallback = {"direction": "HOLD", "confidence": 0.0, "prob_up": 0.5, "trained": False}
        if not _LGB_OK or len(closes) < 50:
            return fallback

        model = self._models.get(regime) or self._models.get("global")
        if model is None:
            return fallback

        feat  = extract_features(closes, volumes).reshape(1, -1)
        probs = model.predict(feat)[0]   # [P_SHORT, P_HOLD, P_LONG]
        p_short, p_hold, p_long = float(probs[0]), float(probs[1]), float(probs[2])

        if p_long > p_short and p_long > 0.40:
            direction  = "LONG"
            confidence = p_long
        elif p_short > p_long and p_short > 0.40:
            direction  = "SHORT"
            confidence = p_short
        else:
            direction  = "HOLD"
            confidence = p_hold

        return {
            "direction":  direction,
            "confidence": round(confidence, 4),
            "prob_up":    round(p_long, 4),
            "prob_down":  round(p_short, 4),
            "trained":    True,
            "regime":     regime,
            "accuracy":   self._accuracy.get(regime, self._accuracy.get("global", 0.0)),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def needs_retrain(self, regime: str = "global") -> bool:
        if not self.trained or (regime not in self._models and "global" not in self._models):
            return True
        acc = self._accuracy.get(regime) or self._accuracy.get("global", 0.0)
        return acc < ACCURACY_THRESHOLD

    def get_accuracy(self, regime: str = "global") -> float:
        return self._accuracy.get(regime, self._accuracy.get("global", 0.0))

    # Backward-compat shim so callers can use same interface as MLPredictor
    def extract_features(
        self,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return extract_features(closes, volumes)
