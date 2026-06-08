"""
Regime-Specific Strategy Router for Dual Engine.

Maps (regime, volatility, UTC session) to concrete modes for passive and active
strategies, plus a size multiplier applied to every position opened that cycle.

Session-based active mode bias (layered on top of regime):
  Asian   0-7  UTC  → mean_reversion  (choppy, range-bound)
  London  8-11 UTC  → breakout        (compression breakouts begin with London open)
  Overlap 12-17 UTC → regime decides  (highest liquidity, all modes valid)
  Late US 18-22 UTC → mean_reversion  (positions unwind, fade extremes)
  Night   23-7 UTC  → funding_arb     (low volume, funding dominates)

Sideways regime NEVER uses momentum (historical win rate: 9%) or vwap_reversal (43%).
Only mean_reversion and funding_arb are safe in sideways/ranging markets.

Passive modes : "long"  | "short" | "auto" | "skip"
Active modes  : "momentum" | "mean_reversion" | "funding_arb" | "vwap_reversal" |
                "breakout" | "skip"
"""

from datetime import datetime, timezone

from config.constants import REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS, REGIME_VOLATILE

HIGH_VOL         = 0.72
STRONG_TREND_VOL = 0.50

# Modes that are safe in ranging/sideways/bear markets — proven by bandit win rates
_SIDEWAYS_SAFE = frozenset({"mean_reversion", "funding_arb"})
# Modes that require a real trend to work — dangerous in sideways
_TREND_ONLY    = frozenset({"momentum"})


class DualRegimeRouter:
    """
    Called once per DualEngine cycle; returns a routing dict that drives
    which signal flavour each strategy uses and how large positions are.
    """

    @staticmethod
    def _session_mode_bias() -> str | None:
        """
        Returns preferred active mode for the current UTC session.
        None = let regime decide freely (London/NY overlap — all modes valid).

        Momentum removed from London session: bandit data showed 9% win rate
        in sideways markets when session bias forced momentum. Replaced with
        breakout (compression setups occur at major session opens).
        """
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 7:    # Asian: choppy, mean reversion works well
            return "mean_reversion"
        if 8 <= hour < 12:   # London open: range breakouts, not trend momentum
            return "breakout"
        if 12 <= hour < 18:  # London/NY overlap: let regime decide
            return None
        if 18 <= hour < 23:  # Late US: fade extremes back toward mean
            return "mean_reversion"
        return "funding_arb"   # deep night: funding rate dominates

    @staticmethod
    def _blend_mode(regime_mode: str, session_mode: str | None,
                    safe_only: bool = False) -> str:
        """
        Merge regime recommendation and session bias.

        safe_only=True: restrict to _SIDEWAYS_SAFE modes regardless of session.
        """
        if regime_mode in ("skip", "funding_arb") or session_mode is None:
            return regime_mode
        candidate = session_mode
        if safe_only and candidate not in _SIDEWAYS_SAFE:
            return "mean_reversion"
        return candidate

    def route(self, market_state: dict) -> dict:
        """
        Returns:
            passive_mode  : str
            active_mode   : str
            size_mult     : float
            reason        : str
        """
        regime = market_state.get("regime", REGIME_SIDEWAYS)
        vol    = float(market_state.get("volatility", 0.5))
        hour   = datetime.now(timezone.utc).hour
        sess   = self._session_mode_bias()

        # ── Volatile regime or high-vol ───────────────────────────────────────
        if regime == REGIME_VOLATILE or vol > HIGH_VOL:
            return {
                "passive_mode": "skip",
                "active_mode":  "funding_arb",
                "size_mult":    0.40,
                "reason":       f"VOLATILE (vol={vol:.2f})",
            }

        # ── Bull trend ────────────────────────────────────────────────────────
        # Router always recommends momentum in trending markets.
        # The bandit overrides with session-optimal mode after enough real trades.
        if regime == REGIME_BULL:
            size = 1.15 if vol < STRONG_TREND_VOL else 1.0
            return {
                "passive_mode": "long",
                "active_mode":  "momentum",
                "size_mult":    size,
                "reason":       f"BULL_TREND (vol={vol:.2f} sess={hour}h)",
            }

        # ── Bear trend ────────────────────────────────────────────────────────
        # Backtest confirmed: momentum SHORT in bear = Sharpe 1.79 (best mode).
        # Direction guard in dual_engine blocks LONGs from momentum in bear.
        # mean_reversion BANNED (backtest: Sharpe -6.45 in bear).
        if regime == REGIME_BEAR:
            size = 0.85 if vol < STRONG_TREND_VOL else 0.70
            return {
                "passive_mode": "short",
                "active_mode":  "momentum",
                "size_mult":    size,
                "reason":       f"BEAR_TREND (vol={vol:.2f} sess={hour}h)",
            }

        # ── Sideways: funding_arb only ────────────────────────────────────────
        # Backtest results for sideways:
        #   momentum:       Sharpe -3.63 ❌
        #   mean_reversion: Sharpe -1.43 ❌  (BANNED — confirmed negative edge)
        #   vwap_reversal:  Sharpe -3.39 ❌  (BANNED — confirmed negative edge)
        #   funding_arb:    structural income, untestable historically but rational
        # Default: funding_arb. If no funding setup, skip (don't force bad trades).
        return {
            "passive_mode": "skip",
            "active_mode":  "funding_arb",
            "size_mult":    0.65,
            "reason":       f"SIDEWAYS (vol={vol:.2f} sess={hour}h→funding_arb)",
        }
