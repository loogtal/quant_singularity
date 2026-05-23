import random
import time

from data.market_data import MarketData


class PaperBroker:

    def __init__(self, price_feed=None):

        self.market = price_feed or MarketData()

    # ============================================================
    # GET PRICE
    # ============================================================

    def get_price(self, symbol):

        return self.market.get_price(symbol)

    # ============================================================
    # EXECUTE ORDER
    # ============================================================

    def execute_order(
        self,
        symbol,
        side,
        size
    ):

        market_price = self.get_price(symbol)

        slippage = random.uniform(
            -0.001,
            0.001
        )

        fill_price = market_price * (
            1 + slippage
        )

        return {

            "symbol": symbol,

            "side": side,

            "size": size,

            "price": round(fill_price, 4),

            "status": "FILLED",

            "timestamp": time.time()
        }

    # ============================================================
    # CLOSE POSITION
    # ============================================================

    def close_position(self, position):

        exit_price = self.get_price(
            position["symbol"]
        )

        entry = position["entry_price"]

        size = position["size"]

        side = position["side"]

        if side == "LONG":

            pnl = (
                exit_price - entry
            ) * size

        else:

            pnl = (
                entry - exit_price
            ) * size

        return pnl
