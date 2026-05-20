"""Runtime metrics snapshot for monitoring."""

from dataclasses import dataclass, asdict


@dataclass
class CycleMetrics:
    cycle: int
    equity: float
    drawdown: float
    positions: int
    winrate: float
    daily_pnl_pct: float
    regime: str
    meta_mode: str

    def to_dict(self) -> dict:
        return asdict(self)
