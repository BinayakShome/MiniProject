import numpy as np


class MarketRegimeAgent:

    def __init__(self, log_returns):
        self.log_returns = log_returns

    def detect_regime(self):
        rolling_vol = self.log_returns.std() * np.sqrt(252)
        avg_vol = rolling_vol.mean()
        rolling_mean = self.log_returns.mean() * 252
        avg_return = rolling_mean.mean()
        if avg_vol > 0.20:
            return "High Volatility"
        if avg_return < 0:
            return "Bear Market"
        if avg_return > 0.15 and avg_vol < 0.15:
            return "Bull Market"
        return "Sideways Market"

    def adjust_for_regime(self, weights_dict, regime):

        adjusted = weights_dict.copy()
        if regime == "High Volatility":
            # Reduce equity by 20%
            for asset in ["NIF100BEES.NS", "MID150BEES.NS"]:
                adjusted[asset] *= 0.8
        elif regime == "Bear Market":
            # Cut equity by 30%
            for asset in ["NIF100BEES.NS", "MID150BEES.NS"]:
                adjusted[asset] *= 0.7
        elif regime == "Bull Market":
            # Increase equity by 10%
            for asset in ["NIF100BEES.NS", "MID150BEES.NS"]:
                adjusted[asset] *= 1.1
        # Normalize back to 1
        total = sum(adjusted.values())
        for k in adjusted:
            adjusted[k] /= total
        return adjusted