"""
Kelly Criterion position sizing.
Uses half-Kelly for safety — bet in proportion to edge, not blindly fixed.
"""


class KellyCriterion:
    MIN_TRADES = 10      # need at least this many trades before using Kelly
    MIN_FRACTION = 0.01  # floor (never risk less than 1%)
    MAX_FRACTION = 0.20  # ceiling (never risk more than 20% on any single trade)

    def half_kelly_fraction(
        self,
        winrate: float,
        win_pct: float,
        loss_pct: float,
    ) -> float:
        """
        Compute half-Kelly fraction of equity to commit per trade.

        winrate : historical win rate  (e.g. 0.45)
        win_pct : average win as decimal (e.g. 0.10 for 10%)
        loss_pct: average loss as decimal (e.g. 0.05 for 5%)
        """
        if loss_pct <= 0 or win_pct <= 0 or not (0 < winrate < 1):
            return self.MIN_FRACTION

        # b = reward/risk ratio
        b = win_pct / loss_pct
        p = winrate
        q = 1.0 - winrate

        kelly = (p * b - q) / b          # full Kelly
        half = kelly * 0.5               # half-Kelly = conservative

        return float(max(self.MIN_FRACTION, min(self.MAX_FRACTION, half)))

    def position_value(
        self,
        winrate: float,
        win_pct: float,
        loss_pct: float,
        equity: float,
        trades_so_far: int = 0,
        default_fraction: float = 0.08,
    ) -> float:
        """
        Return the dollar value to allocate to this position.

        Falls back to default_fraction when not enough trade history yet.
        """
        if trades_so_far < self.MIN_TRADES:
            return equity * default_fraction

        fraction = self.half_kelly_fraction(winrate, win_pct, loss_pct)
        return equity * fraction
