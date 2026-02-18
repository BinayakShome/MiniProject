import numpy as np
from scipy.optimize import minimize


# =====================================================
# Max Sharpe Ratio Optimization
# =====================================================

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.06):
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        weights = np.array(weights)
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Numerical safety
        if port_vol == 0:
            return 1e6

        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe

    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets])

    result = minimize(
        negative_sharpe,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# =====================================================
# Minimum Volatility Optimization
# =====================================================

def min_volatility(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def portfolio_vol(weights):
        weights = np.array(weights)
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets])

    result = minimize(
        portfolio_vol,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# =====================================================
# Efficient Frontier (Monte Carlo Sampling)
# =====================================================

def generate_efficient_frontier(mean_returns, cov_matrix,
                                num_portfolios=5000,
                                risk_free_rate=0.06):

    results = []
    weights_record = []

    num_assets = len(mean_returns)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if port_vol == 0:
            continue

        sharpe = (port_return - risk_free_rate) / port_vol

        results.append([port_return, port_vol, sharpe])
        weights_record.append(weights)

    return np.array(results), weights_record