"""Score 0–100: how ready the system is for real capital."""

from config.settings import DAILY_TARGET_PCT, INITIAL_CAPITAL, MAX_DRAWDOWN


class LiveReadiness:
    def score(self, perf: dict, portfolio_drawdown: float) -> dict:
        points = 0
        notes = []

        trades = perf.get("total_trades", 0)
        winrate = perf.get("winrate", 0)
        daily_pct = perf.get("daily_pnl_pct", 0)
        total_pnl = perf.get("pnl", 0)

        if trades >= 20:
            points += 25
            notes.append("Enough trade history (20+)")
        elif trades >= 10:
            points += 15
            notes.append("Some history (10+ trades)")
        else:
            notes.append(f"Need more trades ({trades}/20)")

        if winrate >= 0.5:
            points += 25
        elif winrate >= 0.45:
            points += 15
        else:
            notes.append(f"Winrate low ({winrate:.1%})")

        if daily_pct >= DAILY_TARGET_PCT:
            points += 20
            notes.append("Daily target reached today")
        elif daily_pct >= 0:
            points += 10

        if total_pnl > 0:
            points += 15
        elif total_pnl > -INITIAL_CAPITAL * 0.02:
            points += 5

        if portfolio_drawdown < 0.05:
            points += 15
        elif portfolio_drawdown < MAX_DRAWDOWN * 0.5:
            points += 8
        else:
            notes.append(f"Drawdown high ({portfolio_drawdown:.1%})")

        score = min(100, points)
        if score >= 70:
            verdict = "READY_FOR_SMALL_LIVE"
        elif score >= 50:
            verdict = "KEEP_PAPER"
        else:
            verdict = "NOT_READY"

        return {
            "score": score,
            "verdict": verdict,
            "notes": notes,
            "target_daily_pct": DAILY_TARGET_PCT,
        }
