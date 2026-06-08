"""
Strategy Lab — validates AI Brain suggestions via mini-backtest before promoting.

Calls the same OOS multi-symbol backtests used by StrategyEvolver.
Only promotes AI-suggested params if they improve median Sharpe by at least
MIN_LAB_IMPROVEMENT over the current params.

Usage:
    lab = StrategyLab(evolver)
    promoted, delta = lab.test_and_promote("passive", ai_adj, market)
    if promoted:
        passive_strategy.update_params(evolver.current_params("passive"))
"""

from datetime import datetime, timezone

from self_evolve.strategy_evolver import (
    BOUNDS,
    MIN_IMPROVEMENT,
    StrategyEvolver,
)

MIN_LAB_IMPROVEMENT = MIN_IMPROVEMENT   # same threshold as Bayesian evolver


class StrategyLab:
    """Paper-tests AI Brain parameter suggestions before they go live."""

    def __init__(self, evolver: StrategyEvolver):
        self._evolver = evolver

    def _sanitize(self, strategy: str, adjustments: dict) -> dict:
        """
        Merge AI suggestions onto current params, clamp to BOUNDS,
        and enforce ema_fast < ema_slow.
        """
        bounds  = BOUNDS[strategy]
        current = self._evolver.current_params(strategy)
        result  = {}
        for key, (lo, hi) in bounds.items():
            raw = adjustments.get(key)
            if raw is None:
                result[key] = current[key]   # keep current if AI left it null
                continue
            if isinstance(lo, int):
                result[key] = int(max(lo, min(hi, int(round(float(raw))))))
            else:
                result[key] = round(float(max(lo, min(hi, float(raw)))), 4)
        # Structural constraint
        if result.get("ema_fast", 0) >= result.get("ema_slow", 999):
            result["ema_fast"] = max(int(bounds["ema_fast"][0]), result["ema_slow"] - 10)
        return result

    def test_and_promote(
        self,
        strategy: str,
        adjustments: dict,
        market,
    ) -> tuple[bool, float]:
        """
        Backtest AI-suggested adjustments.
        Returns (promoted: bool, sharpe_improvement: float).
        """
        if not adjustments:
            return False, 0.0

        # Filter out null values before sanitising
        non_null = {k: v for k, v in adjustments.items() if v is not None}
        if not non_null:
            return False, 0.0

        candidate = self._sanitize(strategy, non_null)
        baseline  = self._evolver._eval_symbols(
            strategy, self._evolver.current_params(strategy), market
        )
        candidate_sharpe = self._evolver._eval_symbols(strategy, candidate, market)
        improvement = candidate_sharpe - baseline

        if improvement >= MIN_LAB_IMPROVEMENT:
            candidate["sharpe"]     = round(candidate_sharpe, 4)
            candidate["evolved_at"] = datetime.now(timezone.utc).isoformat()
            candidate["source"]     = "ai_brain_lab"
            self._evolver._params[strategy].update(candidate)
            self._evolver._save_params()
            # Also feed this tested point into the BO surrogate
            self._evolver._bo[strategy].observe(candidate, candidate_sharpe)
            self._evolver._save_bo_history()
            return True, improvement

        return False, improvement

    def preview(self, strategy: str, adjustments: dict, market) -> dict:
        """Dry-run: return expected Sharpe before/after without promoting."""
        non_null  = {k: v for k, v in adjustments.items() if v is not None}
        candidate = self._sanitize(strategy, non_null) if non_null else self._evolver.current_params(strategy)
        baseline  = self._evolver._eval_symbols(
            strategy, self._evolver.current_params(strategy), market
        )
        candidate_sharpe = self._evolver._eval_symbols(strategy, candidate, market)
        return {
            "strategy":   strategy,
            "baseline_sharpe":   round(baseline, 4),
            "candidate_sharpe":  round(candidate_sharpe, 4),
            "improvement":       round(candidate_sharpe - baseline, 4),
            "would_promote":     (candidate_sharpe - baseline) >= MIN_LAB_IMPROVEMENT,
            "candidate_params":  candidate,
        }
