import numpy as np

TRADING_DAYS = 252


def compute_cagr(portfolio_values):
    total_return = portfolio_values[-1] / portfolio_values[0]
    years = len(portfolio_values) / TRADING_DAYS
    return total_return ** (1 / years) - 1


def compute_max_drawdown(portfolio_values):
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return np.min(drawdown)


def compute_calmar(cagr, max_dd):
    if max_dd == 0:
        return 0
    return cagr / abs(max_dd)


def compute_annual_volatility(log_returns):
    return log_returns.std() * np.sqrt(TRADING_DAYS)