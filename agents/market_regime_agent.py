import numpy as np


class MarketRegimeAgent:

    def __init__(self, log_returns):
        self.log_returns = log_returns

    def detect_regime(self):
        if self.log_returns.empty:
            return "Sideways Market"

        rolling_vol = self.log_returns.std() * np.sqrt(252)
        avg_vol = rolling_vol.mean()
        rolling_mean = self.log_returns.mean() * 252
        avg_return = rolling_mean.mean()

        # Guard: NaN metrics (e.g. all-NaN input) → fall back to Sideways
        if not np.isfinite(avg_vol) or not np.isfinite(avg_return):
            return "Sideways Market"

        if avg_vol > 0.20:
            return "High Volatility"
        if avg_return < 0:
            return "Bear Market"
        if avg_return > 0.15 and avg_vol < 0.15:
            return "Bull Market"
        return "Sideways Market"

    def adjust_for_regime(self, weights_dict, regime):

        adjusted = weights_dict.copy()
        equity_assets = ["NIF100BEES.NS", "MID150BEES.NS"]

        # Only touch equity tickers that actually exist in the portfolio
        present_equity = [a for a in equity_assets if a in adjusted]

        if regime == "High Volatility":
            for asset in present_equity:
                adjusted[asset] *= 0.8
        elif regime == "Bear Market":
            for asset in present_equity:
                adjusted[asset] *= 0.7
        elif regime == "Bull Market":
            for asset in present_equity:
                adjusted[asset] *= 1.1

        # Normalize back to 1
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total

        return adjusted