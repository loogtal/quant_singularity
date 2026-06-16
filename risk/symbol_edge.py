"""
Per-symbol edge tracker — adapts position sizing to each symbol's historical performance.

Maintains a rolling window of the last 20 closed trades per symbol, tracking both
win/loss boolean and average PnL percentage. Uses Kelly criterion (half-Kelly) when
enough trade history is available, falling back to a simple win-rate multiplier.

edge_mult() returns a multiplier (0.40–1.40) used by DualEngine to scale
position size up for consistently profitable symbols and down for losers.
"""

import json
from collections import deque

from config.settings import STORAGE_DIR
from risk.kelly import KellyCriterion

_EDGE_FILE = STORAGE_DIR / "symbol_edge.json"
_WINDOW    = 20   # rolling trade history per symbol
_KELLY_MIN_TRADES = 12  # minimum trades before switching to Kelly sizing


class SymbolEdgeTracker:
    """
    Track win/loss + PnL history per symbol.
    Uses half-Kelly for position sizing once enough history exists.
    """

    def __init__(self):
        # {symbol: deque of {"won": 0/1, "pnl_pct": float}}
        self._history: dict[str, deque] = {}
        self._kelly = KellyCriterion()
        self._load()

    def _load(self) -> None:
        if _EDGE_FILE.exists():
            try:
                data = json.loads(_EDGE_FILE.read_text())
                for sym, vals in data.items():
                    entries = []
                    for v in list(vals)[-_WINDOW:]:
                        # Support old format (plain int) and new format (dict)
                        if isinstance(v, dict):
                            entries.append(v)
                        else:
                            entries.append({"won": int(v), "pnl_pct": 0.0})
                    self._history[sym] = deque(entries, maxlen=_WINDOW)
            except Exception:
                pass

    def _save(self) -> None:
        try:
            _EDGE_FILE.write_text(
                json.dumps({s: list(h) for s, h in self._history.items()}, indent=2)
            )
        except Exception:
            pass

    def record(self, symbol: str, won: bool, pnl_pct: float = 0.0) -> None:
        """Record a closed trade. pnl_pct = PnL as fraction of position value."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=_WINDOW)
        self._history[symbol].append({"won": int(won), "pnl_pct": float(pnl_pct)})
        self._save()

    def _stats(self, symbol: str) -> dict:
        """Compute win rate, avg win %, avg loss % from history."""
        h = self._history.get(symbol)
        if not h or len(h) < 3:
            return {"trades": 0, "winrate": 0.5, "avg_win": 0.015, "avg_loss": 0.007}
        wins   = [e["pnl_pct"] for e in h if e["won"]]
        losses = [abs(e["pnl_pct"]) for e in h if not e["won"]]
        return {
            "trades":   len(h),
            "winrate":  len(wins) / len(h),
            "avg_win":  float(sum(wins)  / len(wins))  if wins   else 0.015,
            "avg_loss": float(sum(losses) / len(losses)) if losses else 0.007,
        }

    def edge_mult(self, symbol: str) -> float:
        """
        Half-Kelly position-size multiplier for this symbol (0.40 – 1.40).

        With ≥ 12 trades: uses half-Kelly (win_rate, avg_win, avg_loss).
        With < 12 trades: simple win-rate lookup table (conservative default).
        """
        s = self._stats(symbol)
        if s["trades"] < 5:
            return 1.0   # no data — neutral

        if s["trades"] >= _KELLY_MIN_TRADES:
            fraction = self._kelly.half_kelly_fraction(
                s["winrate"], s["avg_win"], s["avg_loss"]
            )
            # Map fraction (0.01–0.20) linearly to multiplier (0.40–1.40):
            #   fraction 0.08 → 0.80×, 0.12 → 1.0×, 0.20 → 1.40×.
            # A neutral 50%-winrate symbol yields fraction ≈0.13 → ≈1.07×.
            mult = 0.40 + (fraction / 0.20) * 1.0
            return round(float(min(1.40, max(0.40, mult))), 3)

        # Simple table for early trades
        wr = s["winrate"]
        if wr >= 0.80: return 1.30
        if wr >= 0.65: return 1.15
        if wr >= 0.55: return 1.05
        if wr >= 0.45: return 1.00
        if wr >= 0.35: return 0.80
        if wr >= 0.25: return 0.65
        return 0.50

    def summary(self) -> dict:
        return {
            sym: {
                "winrate":    round(self._stats(sym)["winrate"], 3),
                "trades":     self._stats(sym)["trades"],
                "avg_win":    round(self._stats(sym)["avg_win"], 4),
                "avg_loss":   round(self._stats(sym)["avg_loss"], 4),
                "edge_mult":  self.edge_mult(sym),
            }
            for sym in self._history
            if self._history[sym] and len(self._history[sym]) >= 3
        }
