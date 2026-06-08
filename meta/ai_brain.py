"""
Claude AI Brain — market analysis, strategy adjustment, and regime override.

Called every 4 h from DualEngine. Uses claude-sonnet-4-6 for deeper reasoning.

Capabilities:
  1. Analyse market context and suggest parameter adjustments (validated in StrategyLab)
  2. Suggest active mode priority order for the next period
  3. Provide regime override when macro signals conflict with EMA classifier
  4. Recommend drawdown action when equity is under stress
  5. Flag individual symbols to avoid (correlation clusters, manipulation patterns)

Falls back silently when ANTHROPIC_API_KEY is absent or anthropic not installed.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import STORAGE_DIR

_BRAIN_FILE   = STORAGE_DIR / "ai_brain_last.json"
_COOLDOWN_SEC = 4 * 3600   # analyse every 4 h (was 23 h)

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

_SYSTEM_PROMPT = """\
You are the AI brain of an autonomous Binance USDT-M futures trading bot.
Two strategies run in parallel:
  • Passive  : multi-day EMA trend-following (4H+1D, ATR stops, ADX filter)
  • Active   : intraday scalping across four modes:
               momentum | mean_reversion | vwap_reversal | funding_arb

Goal: maximise risk-adjusted returns over years. Protect capital first.
      Be conservative with changes; never move a parameter more than 25% in one step.

=== Output: valid JSON ONLY — no prose, no markdown fences ===

Required schema:
{
  "analysis": "<2-3 sentence market summary>",
  "regime_override": "bull"|"bear"|"sideways"|"volatile"|null,
  "regime_override_reason": "<why the EMA classifier might be wrong>"|null,

  "passive_adjustments": {
    "take_profit":   <float|null>,   // bounds: 0.06–0.20
    "stop_loss":     <float|null>,   // bounds: 0.03–0.10
    "ema_fast":      <int|null>,     // bounds: 30–70
    "ema_slow":      <int|null>,     // bounds: 150–250
    "max_hold_days": <int|null>      // bounds: 5–21
  },

  "active_adjustments": {
    "take_profit": <float|null>,     // bounds: 0.008–0.030
    "stop_loss":   <float|null>,     // bounds: 0.003–0.015
    "ema_fast":    <int|null>,       // bounds: 5–15
    "ema_slow":    <int|null>        // bounds: 15–35
  },

  "active_mode_priority": ["funding_arb","mean_reversion","momentum","vwap_reversal"]|null,
  // Reorder all 4 modes by preference for the next period. null = no change.

  "drawdown_action": "none"|"reduce_size"|"pause_active"|"close_passive_weakest",
  // Suggested action when equity is under stress. "none" if no action needed.

  "symbols_to_avoid": ["SYM1","SYM2"]|[],
  // Symbols to skip for the next 24 h (manipulation, thin liquidity, news risk).

  "confidence": <0.0–1.0>,
  "rationale": "<brief reason for main changes>"
}

Rules:
  - Set confidence < 0.40 when uncertain; leave adjustments null.
  - Never suggest take_profit ≤ stop_loss.
  - Prefer funding_arb and mean_reversion in sideways/bear markets.
  - Prefer momentum in bull markets with high breadth (>65).
  - If LGBM accuracy < 0.45, suggest "pause_active" unless funding_arb is profitable.
  - If drawdown > 15%, suggest "reduce_size" or "pause_active".
  - If fear_greed < 20 (extreme fear) with bear trend, suggest mean_reversion LONGs
    or funding_arb (shorts crowded → fade)."""


class AIBrain:
    """Wraps Claude API for 4-hourly strategy analysis."""

    def __init__(self):
        self._client: Optional[object] = None
        self._last_run: float   = 0.0
        self._last_output: dict = {}
        self._load_state()

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if _ANTHROPIC_OK and api_key:
            try:
                self._client = _anthropic.Anthropic(api_key=api_key)
            except Exception:
                pass

    # ── state persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if _BRAIN_FILE.exists():
            try:
                data = json.loads(_BRAIN_FILE.read_text())
                self._last_run    = float(data.get("last_run_ts", 0))
                self._last_output = data.get("output", {})
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            _BRAIN_FILE.write_text(json.dumps({
                "last_run_ts": self._last_run,
                "output":      self._last_output,
                "saved_at":    datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._client is not None

    def needs_run(self) -> bool:
        return time.time() - self._last_run > _COOLDOWN_SEC

    def last_output(self) -> dict:
        return dict(self._last_output)

    def analyze(
        self,
        regime: str,
        market_state: dict,
        portfolio_status: dict,
        strategy_stats: dict,
        evolved_params: dict,
        market_intel: dict | None = None,
        lgbm_accuracy: dict | None = None,
        recent_trades: list | None = None,
        performance_history: dict | None = None,
    ) -> dict:
        """
        Call Claude for strategy analysis. Returns JSON dict.
        Respects 4-h cooldown — returns cached output if called early.
        """
        if not self.is_available():
            return {}
        if not self.needs_run():
            return self._last_output

        context = self._build_context(
            regime, market_state, portfolio_status, strategy_stats,
            evolved_params, market_intel, lgbm_accuracy, recent_trades,
            performance_history,
        )
        try:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            raw = msg.content[0].text.strip()

            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else parts[0]
                if raw.startswith("json"):
                    raw = raw[4:]

            output = json.loads(raw.strip())
            output["generated_at"] = datetime.now(timezone.utc).isoformat()
            self._last_run    = time.time()
            self._last_output = output
            self._save_state()
            return output

        except Exception as exc:
            return {"error": str(exc), "generated_at": datetime.now(timezone.utc).isoformat()}

    # ── context builder ───────────────────────────────────────────────────────

    def _build_context(
        self,
        regime: str,
        market_state: dict,
        portfolio_status: dict,
        strategy_stats: dict,
        evolved_params: dict,
        market_intel: dict | None,
        lgbm_accuracy: dict | None,
        recent_trades: list | None,
        performance_history: dict | None = None,
    ) -> str:
        lines = [
            f"Date/time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            f"Current regime:  {regime}",
            f"Volatility:      {market_state.get('volatility', 'unknown')}",
            f"BTC 24h change:  {market_state.get('btc_change_24h', 0):+.2%}",
            "",
            "=== Portfolio ===",
            f"  Total equity:   {portfolio_status.get('equity', 'N/A')} USDT",
            f"  Realized PnL:   {portfolio_status.get('realized_pnl', 0):+.2f} USDT",
            f"  Drawdown:       {portfolio_status.get('drawdown', 0):.1%}",
            f"  Open positions: {portfolio_status.get('positions', 0)}",
        ]

        if market_intel:
            lines += [
                "",
                "=== Market Intelligence ===",
                f"  Fear & Greed:   {market_intel.get('fear_greed', 50):.0f}/100"
                f"  (source: {market_intel.get('fear_greed_source', 'proxy')})",
                f"  Market breadth: {market_intel.get('breadth', 50):.0f}% above EMA50",
                f"  Dominant trend: {market_intel.get('dominant_trend', 'mixed')}",
                f"  Agg. funding:   {market_intel.get('aggregate_funding', 0):.6f}",
                f"  BTC dominant:   {market_intel.get('btc_dominant', False)}",
                f"  Alt season:     {market_intel.get('alt_season', False)}",
            ]
            cycle = market_intel.get("btc_cycle", {})
            if cycle:
                lines += [
                    f"  BTC cycle phase:{cycle.get('cycle_phase', 'unknown')}",
                    f"  Cycle progress: {cycle.get('cycle_progress', 0):.0%}",
                    f"  Price/200wMA:   {cycle.get('price_vs_200wma', 1.0):.2f}×",
                    f"  Days to halving:{cycle.get('days_to_next_halving', 'N/A')}",
                ]

        if lgbm_accuracy:
            lines += [
                "",
                "=== ML Model Accuracy ===",
            ]
            for k, v in lgbm_accuracy.items():
                if v and v > 0:
                    flag = " ⚠ BELOW THRESHOLD" if v < 0.50 else ""
                    lines.append(f"  {k:10s}: {v:.3f}{flag}")

        lines += [
            "",
            "=== Per-strategy stats (all time) ===",
        ]
        for strat, stats in strategy_stats.items():
            t  = stats.get("trades", 0) or 0
            w  = stats.get("wins",   0) or 0
            wr = w / t if t else 0.0
            lines.append(
                f"  {strat:8s}: trades={t:3d}  winrate={wr:.0%}"
                f"  pnl={stats.get('pnl', 0):+.2f}"
            )

        if recent_trades:
            lines += ["", "=== Last 10 closed trades ==="]
            for t in recent_trades[-10:]:
                lines.append(
                    f"  {t.get('strategy','?'):7s} {t.get('symbol','?'):20s} "
                    f"{t.get('side','?'):5s} pnl={t.get('pnl', 0):+.2f} "
                    f"mode={t.get('signal_mode', '-')}"
                )

        if performance_history and performance_history.get("days_tracked", 0) >= 3:
            ph = performance_history
            lines += [
                "",
                "=== Rolling Performance (time series) ===",
                f"  Days tracked:       {ph.get('days_tracked', 0)}",
                f"  Sharpe 14d:         {ph.get('sharpe_14d', 0):.3f}",
                f"  Sharpe 7d:          {ph.get('sharpe_7d', 0):.3f}",
                f"  Active WR 7d:       {ph.get('active_winrate_7d', 0):.0%}",
                f"  Passive WR 7d:      {ph.get('passive_winrate_7d', 0):.0%}",
                f"  Equity trend 5d:    {ph.get('equity_trend_5d', 'unknown')}",
                f"  ML accuracy trend:  {ph.get('accuracy_trend_5d', 'unknown')}",
                f"  Winrate trend 5d:   {ph.get('winrate_trend_5d', 'unknown')}",
            ]

        lines += [
            "",
            "=== Currently evolved params ===",
            f"  Passive: {json.dumps(evolved_params.get('passive', {}), separators=(',',':'))}",
            f"  Active:  {json.dumps(evolved_params.get('active',  {}), separators=(',',':'))}",
            "",
            "Analyse the above and return your JSON response.",
        ]
        return "\n".join(lines)
