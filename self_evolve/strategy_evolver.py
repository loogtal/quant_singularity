"""
Strategy Evolution Engine
=========================
Mutates passive and active strategy parameters, validates via mini-backtest,
and promotes the best variant if it outperforms the current configuration.

Uses Bayesian Optimisation (Gaussian Process + Expected Improvement) instead of
random mutation so each evolution run builds on prior observations and converges
faster toward better Sharpe ratios.

Flow (called by DualEngine every N cycles):
    evolver.maybe_evolve("passive", symbol, market_state)
    evolver.maybe_evolve("active",  symbol, market_state)

Best params are persisted to storage/evolved_params.json and loaded at startup
by PassiveStrategy / ActiveStrategy.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np

from config.settings import STORAGE_DIR

PARAMS_FILE   = STORAGE_DIR / "evolved_params.json"
BO_HIST_FILE  = STORAGE_DIR / "bo_history.json"

# ── parameter search space ────────────────────────────────────────────────────

BOUNDS: dict[str, dict[str, tuple]] = {
    "passive": {
        "take_profit":   (0.06, 0.20),
        "stop_loss":     (0.03, 0.10),
        "ema_fast":      (30,   70),
        "ema_slow":      (150,  250),
        "max_hold_days": (5,    21),
    },
    "active": {
        "take_profit": (0.008, 0.030),
        "stop_loss":   (0.003, 0.015),
        "ema_fast":    (5,     15),
        "ema_slow":    (15,    35),
    },
}

DEFAULTS: dict[str, dict] = {
    "passive": {
        "take_profit": 0.10,
        "stop_loss": 0.05,
        "ema_fast": 50,
        "ema_slow": 200,
        "max_hold_days": 14,
    },
    "active": {
        "take_profit": 0.015,
        "stop_loss": 0.007,
        "ema_fast": 9,
        "ema_slow": 21,
    },
}

EVOLUTION_CYCLES  = 80      # how often DualEngine triggers evolution
CANDIDATES_PER_RUN = 6      # BO iterations per evolution run
MIN_IMPROVEMENT   = 0.05    # Sharpe must improve by this much to promote
EVAL_BARS_PASSIVE = 200     # 1D bars for passive mini-backtest
EVAL_BARS_ACTIVE  = 400     # 1H bars for active mini-backtest

PASSIVE_EVAL_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
ACTIVE_EVAL_SYMBOLS  = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"]

OOS_SPLIT  = 0.70

# BO hyper-parameters
_BO_GRID_SIZE  = 1500   # random grid size for EI maximisation
_BO_MAX_OBS    = 50     # sliding window — keeps GP fast over long runs
_BO_INIT_RAND  = 4      # pure-random seed points before GP kicks in
_BO_NOISE      = 0.05   # GP observation noise (regularisation)
_BO_LS         = 0.30   # RBF length scale in normalised [0,1]^d space


# ── shared maths ─────────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(values[-period:], weights, mode="valid")[-1])


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 5:
        return 15.0
    dm_plus, dm_minus, tr_list = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i - 1]
        ld = lows[i - 1] - lows[i]
        dm_plus.append(hd if hd > ld and hd > 0 else 0.0)
        dm_minus.append(ld if ld > hd and ld > 0 else 0.0)
        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))

    def _smooth(arr, p):
        s = float(np.sum(arr[:p])); out = [s]
        for x in arr[p:]: s = s - s / p + x; out.append(s)
        return out

    if len(tr_list) < period:
        return 15.0
    sm_tr  = _smooth(tr_list,   period)
    sm_dmp = _smooth(dm_plus,   period)
    sm_dmn = _smooth(dm_minus,  period)
    dx_list = []
    for tr, dmp, dmn in zip(sm_tr, sm_dmp, sm_dmn):
        if tr <= 0: continue
        di_p = 100 * dmp / tr; di_m = 100 * dmn / tr
        denom = di_p + di_m
        if denom > 0: dx_list.append(100 * abs(di_p - di_m) / denom)
    return float(np.mean(dx_list[-period:])) if dx_list else 15.0


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains)) if gains.any() else 0.0
    avg_l  = float(np.mean(losses)) if losses.any() else 1e-9
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))


def _sharpe(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
    std = float(np.std(rets))
    if std <= 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252))


# ── Gaussian Process Bayesian Optimiser ───────────────────────────────────────

class _BayesOpt:
    """
    Minimal numpy-only GP surrogate with Expected Improvement acquisition.

    Normalises each parameter dimension to [0, 1] so the RBF length scale
    is the same across all parameters.  Maintains a sliding window of the
    most recent _BO_MAX_OBS observations so GP cost stays bounded.
    """

    def __init__(self, bounds: dict[str, tuple], n_init: int = _BO_INIT_RAND):
        self.bounds  = bounds
        self.keys    = list(bounds.keys())
        self._n_init = n_init
        self._X: list[np.ndarray] = []   # normalised points in [0,1]^d
        self._y: list[float]      = []   # observed Sharpe values

    # ── serialise / deserialise ───────────────────────────────────────────────

    def to_list(self) -> list[dict]:
        return [{"x": x.tolist(), "y": y} for x, y in zip(self._X, self._y)]

    def from_list(self, records: list[dict]) -> None:
        for r in records[-_BO_MAX_OBS:]:
            x = np.asarray(r["x"], dtype=float)
            if len(x) == len(self.keys):
                self._X.append(x)
                self._y.append(float(r["y"]))

    # ── normalise / denormalise ───────────────────────────────────────────────

    def _norm(self, params: dict) -> np.ndarray:
        out = []
        for k in self.keys:
            lo, hi = self.bounds[k]
            span = float(hi - lo)
            out.append((float(params.get(k, (lo + hi) / 2)) - lo) / max(span, 1e-9))
        return np.clip(np.array(out), 0.0, 1.0)

    def _denorm(self, x: np.ndarray) -> dict:
        result = {}
        for i, k in enumerate(self.keys):
            lo, hi = self.bounds[k]
            val = float(x[i]) * (hi - lo) + lo
            if isinstance(lo, int):
                result[k] = int(round(max(lo, min(hi, val))))
            else:
                result[k] = round(max(float(lo), min(float(hi), val)), 4)
        return result

    # ── kernel + GP predict ───────────────────────────────────────────────────

    @staticmethod
    def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diff = A[:, None, :] - B[None, :, :]      # (n, m, d)
        return np.exp(-0.5 * np.sum(diff ** 2, axis=-1) / _BO_LS ** 2)

    def _gp_predict(
        self, X_obs: np.ndarray, y_obs: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(X_obs)
        K    = self._rbf(X_obs, X_obs) + (_BO_NOISE + 1e-6) * np.eye(n)
        K_s  = self._rbf(X_obs, X_test)          # (n, m)
        try:
            L     = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu    = K_s.T @ alpha
            v     = np.linalg.solve(L, K_s)      # (n, m)
            var   = np.ones(len(X_test)) - np.sum(v ** 2, axis=0)
            std   = np.sqrt(np.maximum(var, 1e-9))
        except np.linalg.LinAlgError:
            mu  = np.full(len(X_test), float(np.mean(y_obs)))
            std = np.full(len(X_test), max(float(np.std(y_obs)), 1e-3))
        return mu, std

    # ── normal CDF / PDF (Abramowitz & Stegun 26.2.17, max error 7.5e-8) ─────

    @staticmethod
    def _ncdf(x: np.ndarray) -> np.ndarray:
        ax  = np.abs(x)
        t   = 1.0 / (1.0 + 0.2316419 * ax)
        p   = t * (0.319381530 + t * (-0.356563782 + t * (
              1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        phi = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        cdf = 1.0 - phi * p
        return np.where(x >= 0, cdf, 1.0 - cdf)

    @staticmethod
    def _npdf(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    # ── Expected Improvement ─────────────────────────────────────────────────

    def _ei(
        self, mu: np.ndarray, std: np.ndarray, f_best: float, xi: float = 0.01
    ) -> np.ndarray:
        z  = (mu - f_best - xi) / (std + 1e-9)
        ei = (mu - f_best - xi) * self._ncdf(z) + std * self._npdf(z)
        return np.maximum(ei, 0.0)

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, params: dict, score: float) -> None:
        self._X.append(self._norm(params))
        self._y.append(float(score))
        # Sliding window: keep only the most recent observations
        if len(self._X) > _BO_MAX_OBS:
            self._X = self._X[-_BO_MAX_OBS:]
            self._y = self._y[-_BO_MAX_OBS:]

    def suggest(self) -> dict:
        """Return the next candidate dict to evaluate."""
        d = len(self.keys)
        n = len(self._X)

        if n < self._n_init:
            # Pure exploration: random sample
            return self._denorm(np.random.uniform(0, 1, d))

        X_obs = np.array(self._X)
        y_obs = np.array(self._y, dtype=float)
        f_best = float(np.max(y_obs))

        # Generate candidate grid + random perturbations around best observed
        X_grid = np.random.uniform(0, 1, (_BO_GRID_SIZE, d))
        # Add neighbours of best point for local search
        best_x = X_obs[int(np.argmax(y_obs))]
        neighbours = np.clip(
            best_x + np.random.normal(0, 0.05, (50, d)), 0, 1
        )
        X_candidates = np.vstack([X_grid, neighbours])

        mu, std = self._gp_predict(X_obs, y_obs, X_candidates)
        ei = self._ei(mu, std, f_best)
        return self._denorm(X_candidates[int(np.argmax(ei))])


# ── mini-backtests ────────────────────────────────────────────────────────────

def _backtest_passive(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, params: dict
) -> float:
    fast = int(params["ema_fast"])
    slow = int(params["ema_slow"])
    tp_mult  = float(params["take_profit"])
    sl_mult  = float(params["stop_loss"])
    max_hold = int(params["max_hold_days"])
    capital  = 10_000.0
    cash     = capital
    equity_curve: list[float] = []
    position = None

    warmup = slow + 10
    for i in range(warmup, len(closes)):
        c = closes[:i + 1]
        ema_f = _ema(c, fast)
        ema_s = _ema(c, slow)
        price = closes[i]

        if position:
            h, l    = highs[i], lows[i]
            hold    = i - position["open_bar"]
            closed  = False
            pnl     = 0.0
            if position["side"] == "LONG":
                if l <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * position["size"]; closed = True
                elif h >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["size"]; closed = True
            else:
                if h >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"]; closed = True
                elif l <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"]; closed = True
            if not closed and hold >= max_hold:
                pnl = (
                    (price - position["entry"]) * position["size"]
                    if position["side"] == "LONG"
                    else (position["entry"] - price) * position["size"]
                )
                closed = True
            if closed:
                cash += position["position_value"] + pnl
                position = None

        if position is None:
            side = "LONG" if ema_f > ema_s else "SHORT" if ema_f < ema_s else None
            if side and len(c) >= slow + 14:
                adx = _adx(highs[:i + 1], lows[:i + 1], c, period=14)
                if adx >= 20:
                    ema50_val = _ema(c, fast)
                    dev = (price - ema50_val) / ema50_val if ema50_val > 0 else 0.0
                    pullback_ok = dev <= 0.05 if side == "LONG" else dev >= -0.05
                    if pullback_ok:
                        value = min(cash, 3_333.0)
                        size  = value / price
                        tp    = price * (1 + tp_mult if side == "LONG" else 1 - tp_mult)
                        sl    = price * (1 - sl_mult if side == "LONG" else 1 + sl_mult)
                        cash -= value
                        position = {"side": side, "entry": price, "size": size,
                                    "position_value": value, "tp": tp, "sl": sl, "open_bar": i}

        unrealized = 0.0
        if position:
            unrealized = (
                (price - position["entry"]) * position["size"]
                if position["side"] == "LONG"
                else (position["entry"] - price) * position["size"]
            )
        equity_curve.append(cash + (position["position_value"] if position else 0.0) + unrealized)

    return _sharpe(equity_curve)


def _backtest_active(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, params: dict
) -> float:
    fast     = int(params["ema_fast"])
    slow     = int(params["ema_slow"])
    tp_mult  = float(params["take_profit"])
    sl_mult  = float(params["stop_loss"])
    capital  = 10_000.0
    cash     = capital
    equity_curve: list[float] = []
    position = None

    warmup = slow + 5
    for i in range(warmup, len(closes)):
        c           = closes[:i + 1]
        ema_f       = _ema(c, fast)
        ema_s       = _ema(c, slow)
        trend_bias  = float(np.mean(c[-8:]))
        price       = closes[i]

        if position:
            h, l   = highs[i], lows[i]
            closed = False
            pnl    = 0.0
            if position["side"] == "LONG":
                if l <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * position["size"]; closed = True
                elif h >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["size"]; closed = True
            else:
                if h >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"]; closed = True
                elif l <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"]; closed = True
            if closed:
                cash += position["position_value"] + pnl
                position = None

        if position is None:
            side = None
            rsi  = _rsi(c, 14)
            if ema_f > ema_s and price > trend_bias and rsi < 35:
                side = "LONG"
            elif ema_f < ema_s and price < trend_bias and rsi > 65:
                side = "SHORT"
            if side is None:
                bar_range = (highs[i] - lows[i]) / price if price > 0 else 0
                if ema_f > ema_s and price > trend_bias and bar_range > 0.008:
                    side = "LONG"
                elif ema_f < ema_s and price < trend_bias and bar_range > 0.008:
                    side = "SHORT"
            if side:
                value    = min(cash, 3_333.0)
                size     = value / price
                tp       = price * (1 + tp_mult if side == "LONG" else 1 - tp_mult)
                sl       = price * (1 - sl_mult if side == "LONG" else 1 + sl_mult)
                cash    -= value
                position = {"side": side, "entry": price, "size": size,
                            "position_value": value, "tp": tp, "sl": sl}

        unrealized = 0.0
        if position:
            unrealized = (
                (price - position["entry"]) * position["size"]
                if position["side"] == "LONG"
                else (position["entry"] - price) * position["size"]
            )
        equity_curve.append(cash + (position["position_value"] if position else 0.0) + unrealized)

    return _sharpe(equity_curve)


# ── main evolver class ────────────────────────────────────────────────────────

class StrategyEvolver:
    """
    AI-driven parameter evolution using Bayesian Optimisation.

    Each call to _run_evolution:
      1. Evaluates the current params (baseline Sharpe, OOS multi-symbol).
      2. Runs CANDIDATES_PER_RUN BO iterations, each guided by the GP posterior.
      3. Promotes the best candidate if it beats the baseline by MIN_IMPROVEMENT.

    BO history persists across restarts (bo_history.json) so the surrogate
    becomes more accurate over time without wasting live-trading evaluations.
    """

    def __init__(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._params: dict[str, dict] = self._load_params()
        self._call_count: dict[str, int] = {"passive": 0, "active": 0}
        self._evolution_log: list[dict] = []
        # One BO instance per strategy, loaded with persisted history
        self._bo: dict[str, _BayesOpt] = {
            s: _BayesOpt(BOUNDS[s]) for s in BOUNDS
        }
        self._load_bo_history()

    # ─── persistence ─────────────────────────────────────────────────────────

    def _load_params(self) -> dict[str, dict]:
        if PARAMS_FILE.exists():
            try:
                data = json.loads(PARAMS_FILE.read_text())
                result = {}
                for strat in ("passive", "active"):
                    base = dict(DEFAULTS[strat])
                    base.update(data.get(strat, {}))
                    result[strat] = base
                return result
            except Exception:
                pass
        return {s: dict(DEFAULTS[s]) for s in DEFAULTS}

    def _save_params(self) -> None:
        PARAMS_FILE.write_text(json.dumps(self._params, indent=2))

    # Keep backward-compat alias
    def _save(self) -> None:
        self._save_params()

    def _load_bo_history(self) -> None:
        if BO_HIST_FILE.exists():
            try:
                data = json.loads(BO_HIST_FILE.read_text())
                for strat, records in data.items():
                    if strat in self._bo and isinstance(records, list):
                        self._bo[strat].from_list(records)
            except Exception:
                pass

    def _save_bo_history(self) -> None:
        try:
            BO_HIST_FILE.write_text(json.dumps(
                {s: self._bo[s].to_list() for s in self._bo}, indent=2
            ))
        except Exception:
            pass

    # ─── evaluation ──────────────────────────────────────────────────────────

    def _fetch(self, strategy: str, symbol: str, market):
        if strategy == "passive":
            df = market.get_ohlcv_df(symbol, timeframe="1d", limit=EVAL_BARS_PASSIVE + 10)
        else:
            df = market.get_ohlcv_df(symbol, timeframe="1h", limit=EVAL_BARS_ACTIVE + 10)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        split  = int(len(closes) * OOS_SPLIT)
        return (
            closes[:split], highs[:split], lows[:split],
            closes[split:], highs[split:], lows[split:],
        )

    def _eval_symbols(self, strategy: str, params: dict, market) -> float:
        symbols = PASSIVE_EVAL_SYMBOLS if strategy == "passive" else ACTIVE_EVAL_SYMBOLS
        fn      = _backtest_passive if strategy == "passive" else _backtest_active
        sharpes = []
        for sym in symbols:
            try:
                _, _, _, oos_c, oos_h, oos_l = self._fetch(strategy, sym, market)
                if len(oos_c) < 20:
                    continue
                sharpes.append(fn(oos_c, oos_h, oos_l, params))
            except Exception:
                continue
        return float(np.median(sharpes)) if sharpes else 0.0

    def _evaluate(self, strategy: str, symbol: str, market) -> float:
        return self._eval_symbols(strategy, self._params[strategy], market)

    def _evaluate_candidate(self, strategy: str, symbol: str, market, candidate: dict) -> float:
        return self._eval_symbols(strategy, candidate, market)

    # ─── BO candidate constraint repair ──────────────────────────────────────

    @staticmethod
    def _fix_ema(candidate: dict, strategy: str) -> dict:
        """Ensure ema_fast < ema_slow with a meaningful gap (minimum 5 bars)."""
        MIN_EMA_GAP = 5
        fast = candidate.get("ema_fast", 0)
        slow = candidate.get("ema_slow", 999)
        if fast >= slow - MIN_EMA_GAP:
            lo = int(BOUNDS[strategy]["ema_fast"][0])
            candidate["ema_fast"] = max(lo, slow - max(10, MIN_EMA_GAP + 1))
        return candidate

    def _mutate(self, strategy: str) -> dict:
        """
        Generate a random candidate within BOUNDS with ema_fast < ema_slow.
        Used by validation tests and as a BO warm-start fallback.
        """
        rng = np.random.default_rng()
        candidate = {}
        for key, (lo, hi) in BOUNDS[strategy].items():
            if isinstance(lo, int) and isinstance(hi, int):
                candidate[key] = int(rng.integers(lo, hi + 1))
            else:
                candidate[key] = float(rng.uniform(lo, hi))
                candidate[key] = round(candidate[key], 4)
        return self._fix_ema(candidate, strategy)

    # ─── main entry point ────────────────────────────────────────────────────

    def maybe_evolve(self, strategy: str, symbol: str, market) -> dict | None:
        self._call_count[strategy] = self._call_count.get(strategy, 0) + 1
        if self._call_count[strategy] % EVOLUTION_CYCLES != 0:
            return None
        return self._run_evolution(strategy, symbol, market)

    def _run_evolution(self, strategy: str, symbol: str, market) -> dict | None:
        t0 = time.time()
        bo = self._bo[strategy]

        baseline_sharpe = self._evaluate(strategy, symbol, market)
        bo.observe(self._params[strategy], baseline_sharpe)

        best_sharpe    = baseline_sharpe
        best_candidate: dict | None = None

        for _ in range(CANDIDATES_PER_RUN):
            candidate = self._fix_ema(bo.suggest(), strategy)
            sharpe    = self._evaluate_candidate(strategy, symbol, market, candidate)
            bo.observe(candidate, sharpe)
            if sharpe > best_sharpe:
                best_sharpe    = sharpe
                best_candidate = candidate

        self._save_bo_history()

        elapsed = round(time.time() - t0, 1)
        if best_candidate is not None and (best_sharpe - baseline_sharpe) >= MIN_IMPROVEMENT:
            best_candidate["sharpe"]     = round(best_sharpe, 4)
            best_candidate["evolved_at"] = datetime.now(timezone.utc).isoformat()
            best_candidate["source"]     = "bayesian_opt"
            self._params[strategy].update(best_candidate)
            self._save_params()
            event = {
                "strategy":       strategy,
                "symbol":         symbol,
                "baseline_sharpe": round(baseline_sharpe, 4),
                "new_sharpe":     round(best_sharpe, 4),
                "improvement":    round(best_sharpe - baseline_sharpe, 4),
                "new_params":     best_candidate,
                "elapsed_s":      elapsed,
                "ts":             datetime.now(timezone.utc).isoformat(),
                "bo_obs":         len(bo._X),
            }
            self._evolution_log.insert(0, event)
            self._evolution_log = self._evolution_log[:20]
            return best_candidate

        return None

    # ─── accessors ───────────────────────────────────────────────────────────

    def current_params(self, strategy: str) -> dict:
        return dict(self._params.get(strategy, DEFAULTS[strategy]))

    def evolution_log(self) -> list[dict]:
        return list(self._evolution_log)

    def bo_status(self) -> dict:
        return {
            s: {"observations": len(bo._X), "best_y": round(max(bo._y), 4) if bo._y else 0.0}
            for s, bo in self._bo.items()
        }
