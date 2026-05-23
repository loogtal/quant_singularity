import numpy as np


class AlphaFactors:

    def momentum_factor(self, closes):

        return (
            closes[-1] - closes[-20]
        ) / closes[-20]

    def volatility_factor(self, closes):

        returns = np.diff(closes) / closes[:-1]

        return np.std(returns)

    def trend_strength(self, closes):

        ma_fast = np.mean(closes[-20:])

        ma_slow = np.mean(closes[-100:])

        return (ma_fast - ma_slow) / ma_slow
