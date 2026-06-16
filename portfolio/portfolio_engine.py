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

        self._peak_equity = float(initial_cash)  # running high-water mark

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
        leverage = float(position.get("leverage", 1)) or 1.0
        margin   = round(position["position_value"] / leverage, 4)
        position["margin"] = margin          # stored for close/half-close
        self.positions.append(position)
        self.cash -= margin

    def close_position(self, symbol, pnl):

        pos = self.get_position(symbol)

        if not pos:
            return

        leverage = float(pos.get("leverage", 1)) or 1.0
        margin = pos.get("margin", pos["position_value"] / leverage)
        self.cash += margin + pnl

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
                p.get("margin", p["position_value"])
                for p in self.positions
            ])
        )

        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

        self.drawdown = max(
            0,
            (self._peak_equity - self.equity) / max(self._peak_equity, 1.0)
        )

    def current_drawdown(self) -> float:
        return float(self.drawdown)

    def restore(self, cash: float, positions: list, realized_pnl: float = 0) -> None:
        self.cash = float(cash)
        self.positions = list(positions)
        self.realized_pnl = float(realized_pnl)
        self.update_equity()
        # Don't reset peak on restore — keep the high-water mark
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

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