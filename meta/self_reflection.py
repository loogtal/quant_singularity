class SelfReflection:

    def evaluate(self, metrics):

        winrate = metrics["winrate"]

        sharpe = metrics["sharpe"]

        pnl = metrics["pnl"]

        total_trades = metrics.get("total_trades", 0)

        equity = metrics.get("equity", 10000)

        # Paper: reduce size only when losses matter vs current equity
        loss_threshold = equity * 0.01  # 1% of equity

        if total_trades >= 5 and winrate < 0.35 and pnl < -loss_threshold:

            return {
                "status": "WEAK",
                "action": "REDUCE_ALLOCATION"
            }

        if total_trades >= 8 and pnl < -(equity * 0.05):

            return {
                "status": "WEAK",
                "action": "REDUCE_ALLOCATION"
            }

        if pnl > 0 and sharpe > 1.5:

            return {
                "status": "STRONG",
                "action": "INCREASE_ALLOCATION"
            }

        return {
            "status": "NEUTRAL",
            "action": "MONITOR"
        }
