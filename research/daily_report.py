"""Daily autopilot report — dual-engine automated end-of-day summary.

Provides DailyAutopilot.generate(), fired at 23:00 UTC by DualEngine._report().
"""

from collections import defaultdict
from datetime import datetime, timezone


class DailyAutopilot:
    """Analyses closed trades for the day and returns a plain-English summary."""

    def generate(
        self,
        trade_history: list,
        market_intel: dict,
        evolved_params: dict,
    ) -> dict:
        """
        Parameters
        ----------
        trade_history  : list of dicts from TradeManager.get_recent_trades()
                         Fields: symbol, side, pnl, signal_mode, confidence, regime, strategy
        market_intel   : latest market intelligence snapshot
        evolved_params : current evolved params {passive: {...}, active: {...}}

        Returns
        -------
        dict with keys:
            date, summary, winning_modes, losing_modes, best_symbol, worst_symbol,
            regime_accuracy, tomorrow_focus, confidence_calibration, action_items
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if not trade_history:
            return self._empty_report(today)

        mode_pnl:    dict[str, float] = defaultdict(float)
        mode_trades: dict[str, int]   = defaultdict(int)
        sym_pnl:     dict[str, float] = defaultdict(float)

        regime_total   = 0
        regime_correct = 0
        win_confs:  list[float] = []
        lose_confs: list[float] = []
        total_pnl = 0.0

        for t in trade_history:
            pnl        = float(t.get("pnl", 0.0))
            mode       = str(t.get("signal_mode") or t.get("passive_mode") or "unknown")
            symbol     = str(t.get("symbol", "unknown"))
            side       = str(t.get("side", ""))
            regime_raw = str(t.get("regime") or "unknown").lower()
            conf       = float(t.get("confidence", 0.5))

            mode_pnl[mode]    += pnl
            mode_trades[mode] += 1
            sym_pnl[symbol]   += pnl
            total_pnl         += pnl

            # Regime accuracy: LONG in bull, SHORT in bear, neutral in sideways
            if side in ("LONG", "SHORT"):
                regime_total += 1
                if "bull" in regime_raw and side == "LONG":
                    regime_correct += 1
                elif "bear" in regime_raw and side == "SHORT":
                    regime_correct += 1
                elif "sideways" in regime_raw:
                    regime_correct += 1

            if pnl > 0:
                win_confs.append(conf)
            else:
                lose_confs.append(conf)

        winning_modes = [m for m, p in mode_pnl.items() if p > 0 and mode_trades[m] >= 2]
        losing_modes  = [m for m, p in mode_pnl.items() if p < 0 and mode_trades[m] >= 2]

        best_symbol  = max(sym_pnl, key=sym_pnl.get) if sym_pnl else "N/A"
        worst_symbol = min(sym_pnl, key=sym_pnl.get) if sym_pnl else "N/A"

        regime_accuracy = round(regime_correct / regime_total, 4) if regime_total > 0 else 0.0

        avg_win_conf  = round(sum(win_confs)  / len(win_confs),  4) if win_confs  else 0.0
        avg_lose_conf = round(sum(lose_confs) / len(lose_confs), 4) if lose_confs else 0.0
        confidence_calibration = round(avg_win_conf - avg_lose_conf, 4)

        tomorrow_focus = sorted(sym_pnl, key=lambda s: abs(sym_pnl[s]), reverse=True)[:3]

        n_trades = len(trade_history)
        n_wins   = sum(1 for t in trade_history if float(t.get("pnl", 0)) > 0)
        winrate  = round(n_wins / n_trades, 2) if n_trades > 0 else 0.0
        direction = "positive" if total_pnl >= 0 else "negative"
        best_mode = max(mode_pnl, key=mode_pnl.get) if mode_pnl else "N/A"
        summary = (
            f"Today closed {n_trades} trades with a {winrate:.0%} win rate "
            f"and {direction} PnL of {total_pnl:+.2f} USDT. "
            f"Best mode: {best_mode} ({mode_pnl.get(best_mode, 0):+.2f} USDT)."
        )

        action_items: list[str] = []
        if avg_lose_conf > 0 and avg_lose_conf > avg_win_conf and len(lose_confs) >= 3:
            action_items.append(
                "Raise min confidence to 0.70 (too many low-conf losses)"
            )
        if regime_accuracy < 0.60 and regime_total >= 5:
            action_items.append(
                f"Regime accuracy low ({regime_accuracy:.0%}) — review direction guard."
            )
        if losing_modes and not winning_modes:
            action_items.append(
                f"All modes lost today ({', '.join(losing_modes)}) — "
                "consider pausing active until regime stabilises."
            )
        if not action_items:
            action_items.append("Performance on track — no parameter changes needed.")

        return {
            "date":                   today,
            "summary":                summary,
            "winning_modes":          winning_modes,
            "losing_modes":           losing_modes,
            "best_symbol":            best_symbol,
            "worst_symbol":           worst_symbol,
            "regime_accuracy":        regime_accuracy,
            "tomorrow_focus":         tomorrow_focus,
            "confidence_calibration": confidence_calibration,
            "action_items":           action_items,
            "total_pnl":              round(total_pnl, 4),
            "n_trades":               n_trades,
            "winrate":                winrate,
        }

    @staticmethod
    def _empty_report(today: str) -> dict:
        return {
            "date":                   today,
            "summary":                "No trades were closed today.",
            "winning_modes":          [],
            "losing_modes":           [],
            "best_symbol":            "N/A",
            "worst_symbol":           "N/A",
            "regime_accuracy":        0.0,
            "tomorrow_focus":         [],
            "confidence_calibration": 0.0,
            "action_items":           ["No trades today — check if bot is running correctly."],
            "total_pnl":              0.0,
            "n_trades":               0,
            "winrate":                0.0,
        }
