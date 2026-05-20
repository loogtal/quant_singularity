class PortfolioEngine:

    def __init__(
        self,
        initial_cash=10000
    ):

        self.initial_cash = initial_cash

        self.cash = initial_cash

        self.equity = initial_cash

        self.positions = []

        self.realized_pnl = 0

        self.unrealized_pnl = 0

        self.drawdown = 0

    def has_open_position(self, symbol):

        for p in self.positions:

            if p["symbol"] == symbol:
                return True

        return False

    def get_position(self, symbol):

        for p in self.positions:

            if p["symbol"] == symbol:
                return p

        return None

    def add_position(self, position):

        self.positions.append(position)

        self.cash -= position["position_value"]

    def close_position(self, symbol, pnl):

        pos = self.get_position(symbol)

        if not pos:
            return

        self.cash += pos["position_value"] + pnl

        self.realized_pnl += pnl

        self.positions.remove(pos)

        self.update_equity()

    def update_position_price(
        self,
        symbol,
        current_price
    ):

        pos = self.get_position(symbol)

        if not pos:
            return

        pos["current_price"] = current_price

        if pos["side"] == "LONG":

            pnl = (
                current_price
                - pos["entry_price"]
            ) * pos["size"]

        else:

            pnl = (
                pos["entry_price"]
                - current_price
            ) * pos["size"]

        pos["unrealized_pnl"] = pnl

        self.update_equity()

    def update_equity(self):

        total_unrealized = sum([
            p.get("unrealized_pnl", 0)
            for p in self.positions
        ])

        self.unrealized_pnl = total_unrealized

        self.equity = (
            self.cash
            + total_unrealized
            + sum([
                p["position_value"]
                for p in self.positions
            ])
        )

        peak = max(
            self.initial_cash,
            self.equity
        )

        self.drawdown = max(
            0,
            (peak - self.equity) / peak
        )

    def current_drawdown(self) -> float:
        return float(self.drawdown)

    def restore(self, cash: float, positions: list, realized_pnl: float = 0) -> None:
        self.cash = float(cash)
        self.positions = list(positions)
        self.realized_pnl = float(realized_pnl)
        self.update_equity()

    def status(self):

        return {

            "cash": round(self.cash, 2),

            "equity": round(self.equity, 2),

            "positions": len(self.positions),

            "realized_pnl": round(
                self.realized_pnl,
                2
            ),

            "unrealized_pnl": round(
                self.unrealized_pnl,
                2
            ),

            "drawdown": round(
                self.drawdown,
                4
            )
        }