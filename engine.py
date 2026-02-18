import numpy as np
import pandas as pd
from quant.metrics import compute_log_returns
from quant.optimization import max_sharpe_ratio


def rolling_backtest(prices, lookback_days=756):  # 3 years approx
    log_returns = compute_log_returns(prices)
    dates = log_returns.index
    portfolio_values = [1.0]
    equal_values = [1.0]

    current_weights = np.array([1/len(prices.columns)] * len(prices.columns))

    for i in range(lookback_days, len(log_returns)):

        window_returns = log_returns.iloc[i - lookback_days:i]

        mean_returns = window_returns.mean() * 252
        cov_matrix = window_returns.cov() * 252

        # Optimize weights
        optimal_weights = max_sharpe_ratio(mean_returns, cov_matrix)

        # Next day return
        next_return = log_returns.iloc[i]

        # Portfolio return
        port_return = np.dot(optimal_weights, next_return)
        eq_return = np.mean(next_return)

        portfolio_values.append(
            portfolio_values[-1] * np.exp(port_return)
        )

        equal_values.append(
            equal_values[-1] * np.exp(eq_return)
        )

    result_df = pd.DataFrame({
        "Optimized": portfolio_values,
        "EqualWeight": equal_values
    })

    return result_df