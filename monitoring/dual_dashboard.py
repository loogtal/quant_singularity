"""Helpers for dual strategy dashboard rendering."""


def normalize_dual_state(state: dict) -> dict:
    """Ensure the dual dashboard data structure is present and stable."""
    if not isinstance(state, dict):
        return state

    dual = state.get("dual", {})

    # Normalise nested passive / active sub-dicts
    def _norm_side(key):
        d = dual.get(key, {})
        return {
            "today":         d.get("realized_pnl", 0.0),
            "equity":        d.get("equity", 0.0),
            "positions":     d.get("positions", 0),
            "realized_pnl":  d.get("realized_pnl", 0.0),
            "unrealized_pnl": d.get("unrealized_pnl", 0.0),
            "cash":          d.get("cash", 0.0),
        }

    normalized = {
        "passive":              _norm_side("passive"),
        "active":               _norm_side("active"),
        "total_portfolio_value": dual.get("total_portfolio_value", 0.0),
        "capital_utilization":  dual.get("capital_utilization", {}),
        "capital_allocation":   dual.get("capital_allocation", {}),
        "strategy_stats":       dual.get("strategy_stats", {}),
        "conflicts":            dual.get("conflict_log", []),
        "daily_target":         dual.get("daily_target", {}),
        # Fields that were previously stripped — pass through as-is
        "regime_route":         dual.get("regime_route", {}),
        "evolution_log":        dual.get("evolution_log", []),
        "evolved_params":       dual.get("evolved_params", {}),
        "market_intelligence":  dual.get("market_intelligence",
                                         state.get("market_intelligence", {})),
    }
    state["dual"] = normalized
    return state
