import numpy as np

TRADING_DAYS = 252


def compute_cagr(portfolio_values):
    start = portfolio_values[0]
    end = portfolio_values[-1]
    years = len(portfolio_values) / TRADING_DAYS

    # Guard: zero/negative start or zero time span are undefined
    if start <= 0:
        raise ValueError(f"Portfolio start value must be positive, got {start}")
    if years == 0:
        raise ValueError("Cannot compute CAGR over zero time span")

    total_return = end / start
    # Negative total_return (end < 0) would make fractional power undefined in reals
    if total_return <= 0:
        raise ValueError(f"Total return must be positive to compute CAGR, got {total_return}")

    return total_return ** (1 / years) - 1


def compute_max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values, dtype=float)
    cumulative_max = np.maximum.accumulate(portfolio_values)

    # Guard: avoid 0/0 when all values are zero
    if (cumulative_max == 0).any():
        # Replace zeros in denominator with NaN so we skip those positions
        with np.errstate(invalid="ignore", divide="ignore"):
            drawdown = np.where(
                cumulative_max == 0,
                0.0,
                (portfolio_values - cumulative_max) / cumulative_max
            )
    else:
        drawdown = (portfolio_values - cumulative_max) / cumulative_max

    return float(np.min(drawdown))


def compute_calmar(cagr, max_dd):
    if max_dd == 0:
        return 0
    return cagr / abs(max_dd)


def compute_annual_volatility(log_returns):
    # ddof=1 gives NaN for a single observation; fall back to 0.0 (no vol measurable)
    if len(log_returns) < 2:
        return 0.0
    return log_returns.std() * np.sqrt(TRADING_DAYS)