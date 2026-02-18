import numpy as np


class DecisionAgent:

    def __init__(self, rebalance_threshold=0.10):
        self.rebalance_threshold = rebalance_threshold
        self.previous_weights = None

    def should_rebalance(self, current_weights, new_weights):
        if self.previous_weights is None:
            self.previous_weights = new_weights
            return True

        turnover = np.sum(np.abs(new_weights - self.previous_weights))

        if turnover > self.rebalance_threshold:
            self.previous_weights = new_weights
            return True

        return False

    def evaluate_market_state(self, vol, sharpe):
        if vol > 0.18:
            return "High Volatility"
        elif sharpe < 0.5:
            return "Weak Risk-Adjusted Return"
        else:
            return "Stable"
