"""Pre-trade checks for live capital."""

from config.settings import (
    INITIAL_CAPITAL,
    LIVE_CONFIRM,
    LIVE_MODE,
    MAX_LIVE_ORDER_USDT,
    MAX_POSITION_FRACTION,
)


class LiveSafety:
    def validate_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        equity: float,
    ) -> tuple[bool, str]:
        if not LIVE_MODE:
            return True, "paper"

        if not LIVE_CONFIRM:
            return False, "Set QS_LIVE_CONFIRM=true to enable live orders"

        notional = size * price
        if notional > MAX_LIVE_ORDER_USDT:
            return False, f"Order {notional:.0f} USDT > max {MAX_LIVE_ORDER_USDT:.0f}"

        cap = equity * MAX_POSITION_FRACTION
        if notional > cap * 1.05:
            return False, f"Order exceeds position cap ({cap:.0f} USDT)"

        if equity < INITIAL_CAPITAL * 0.5:
            return False, "Equity below 50% of initial — live blocked"

        return True, "ok"
