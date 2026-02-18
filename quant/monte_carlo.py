import numpy as np

TRADING_DAYS = 252


def monte_carlo_simulation(
    weights,
    log_returns,
    years=5,
    simulations=5000
):

    days = years * TRADING_DAYS
    returns_array = log_returns.values
    results = []

    for _ in range(simulations):
        sampled_days = returns_array[
            np.random.randint(0, len(returns_array), days)
        ]

        portfolio_returns = np.dot(sampled_days, weights)
        portfolio_growth = np.exp(np.cumsum(portfolio_returns))
        results.append(portfolio_growth[-1])

    results = np.array(results)

    expected_value = np.mean(results)
    var_5 = np.percentile(results, 5)

    # CVaR (Expected Shortfall)
    worst_5_percent = results[results <= var_5]
    cvar_5 = np.mean(worst_5_percent)

    prob_loss = np.mean(results < 1)

    metrics = {
        "Expected Final Value": round(expected_value, 3),
        "5% VaR": round(var_5, 3),
        "5% CVaR (Expected Shortfall)": round(cvar_5, 3),
        "Probability of Loss": round(prob_loss, 3)
    }

    return results, metrics