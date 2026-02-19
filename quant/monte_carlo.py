import numpy as np

TRADING_DAYS = 252


def monte_carlo_simulation(
    weights,
    log_returns,
    years=5,
    simulations=5000
):
    if simulations <= 0:
        raise ValueError(f"simulations must be > 0, got {simulations}")

    days = years * TRADING_DAYS
    if days <= 0:
        raise ValueError(f"years must be > 0, got {years}")

    returns_array = log_returns.values
    if len(returns_array) == 0:
        raise ValueError("log_returns is empty — cannot run Monte Carlo simulation")

    results = []

    for _ in range(simulations):
        sampled_days = returns_array[
            np.random.randint(0, len(returns_array), days)
        ]

        portfolio_returns = np.dot(sampled_days, weights)

        # Guard against overflow from extreme cumulative returns
        cumsum = np.cumsum(portfolio_returns)
        # Clip before exp to avoid inf (exp(710) overflows float64)
        cumsum = np.clip(cumsum, -700, 700)
        portfolio_growth = np.exp(cumsum)
        results.append(portfolio_growth[-1])

    results = np.array(results)

    expected_value = np.mean(results)
    var_5 = np.percentile(results, 5)

    # CVaR (Expected Shortfall)
    worst_5_percent = results[results <= var_5]
    cvar_5 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_5

    prob_loss = np.mean(results < 1)

    metrics = {
        "Expected Final Value": round(float(expected_value), 3),
        "5% VaR": round(float(var_5), 3),
        "5% CVaR (Expected Shortfall)": round(float(cvar_5), 3),
        "Probability of Loss": round(float(prob_loss), 3)
    }

    return results, metrics