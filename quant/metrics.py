import numpy as np

TRADING_DAYS = 252


def compute_log_returns(price_df):
    log_returns = np.log(price_df / price_df.shift(1)).dropna()

    # Remove extreme outliers (>20% daily move)
    log_returns = log_returns[(log_returns.abs() < 0.2).all(axis=1)]

    return log_returns


def annualized_return(log_returns):
    return log_returns.mean() * TRADING_DAYS


def annualized_volatility(log_returns):
    return log_returns.std() * np.sqrt(TRADING_DAYS)


def covariance_matrix(log_returns):
    return log_returns.cov() * TRADING_DAYS


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.06):
    weights = np.array(weights)

    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Guard: zero volatility makes Sharpe undefined — return 0.0 instead of ±inf/NaN
    if port_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (port_return - risk_free_rate) / port_vol

    return port_return, port_vol, sharpe