from memory.state_manager import StateManager
from agents.risk_profile_agent import RiskProfileAgent
from agents.market_regime_agent import MarketRegimeAgent
from agents.decision_agent import DecisionAgent
from agents.explanation_agent import ExplanationAgent

from data.data_loader import DataLoader
from quant.metrics import *
from quant.optimization import max_sharpe_ratio
from quant.performance import (
    compute_cagr,
    compute_max_drawdown,
    compute_calmar
)
from quant.monte_carlo import monte_carlo_simulation

import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Asset Universe
# ==============================

tickers = [
    "NIF100BEES.NS",
    "MID150BEES.NS",
    "GOLDBEES.NS",
    "LIQUIDBEES.NS"
]

# ==============================
# Load Data
# ==============================

loader = DataLoader(tickers)
prices = loader.fetch_data()
prices = prices[tickers]

log_returns = compute_log_returns(prices)
mean_returns = annualized_return(log_returns)
cov_matrix = covariance_matrix(log_returns)

# ==============================
# Step 1: Quant Optimization
# ==============================

optimal_weights = max_sharpe_ratio(mean_returns, cov_matrix)

base_weights = {
    ticker: float(round(weight, 4))
    for ticker, weight in zip(tickers, optimal_weights)
}

# ==============================
# Step 2: User Risk Profile
# ==============================

age = int(input("Enter your age: "))
horizon = int(input("Investment horizon (years): "))
tolerance = input("Risk tolerance (Low/Medium/High): ")

risk_agent = RiskProfileAgent(age, horizon, tolerance)
risk_adjusted_weights, risk_score = risk_agent.adjust_allocation(base_weights.copy())

# ==============================
# Step 3: Market Regime Detection
# ==============================

regime_agent = MarketRegimeAgent(log_returns)
regime = regime_agent.detect_regime()

final_weights = regime_agent.adjust_for_regime(
    risk_adjusted_weights.copy(),
    regime
)

# ==============================
# Step 4: Final Performance Calculation
# ==============================

final_array = np.array([final_weights[t] for t in tickers])

ret, vol, sharpe = portfolio_performance(
    final_array,
    mean_returns,
    cov_matrix
)

portfolio_daily = np.dot(log_returns.values, final_array)
portfolio_values = np.exp(np.cumsum(portfolio_daily))

cagr = compute_cagr(portfolio_values)
max_dd = compute_max_drawdown(portfolio_values)
calmar = compute_calmar(cagr, max_dd)

performance = {
    "Return": float(round(ret, 4)),
    "Volatility": float(round(vol, 4)),
    "Sharpe": float(round(sharpe, 4)),
    "CAGR": float(round(cagr, 4)),
    "Max Drawdown": float(round(max_dd, 4)),
    "Calmar Ratio": float(round(calmar, 4))
}

# ==============================
# Step 5: Monte Carlo + CVaR
# ==============================

mc_results_array, mc_metrics = monte_carlo_simulation(
    final_array,
    log_returns,
    years=horizon,
    simulations=5000
)

# ==============================
# Plot Monte Carlo Distribution
# ==============================

plt.figure(figsize=(10, 6))
plt.hist(mc_results_array, bins=50, alpha=0.7)

plt.axvline(mc_metrics["5% VaR"], linestyle='--', label="5% VaR")
plt.axvline(mc_metrics["5% CVaR (Expected Shortfall)"], linestyle='--', label="5% CVaR")

plt.title(f"Monte Carlo {horizon}-Year Wealth Distribution")
plt.xlabel("Final Portfolio Value (Wealth Multiple)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ==============================
# Persistent Memory + Turnover
# ==============================

state_manager = StateManager()
previous_state = state_manager.load_state()
previous_allocation = previous_state.get("allocation", {})

turnover = None
if previous_allocation:
    try:
        prev_array = np.array([previous_allocation[t] for t in tickers])
        turnover = float(np.sum(np.abs(final_array - prev_array)))
    except:
        turnover = None

# ==============================
# Step 6: Decision Layer
# ==============================

decision_agent = DecisionAgent()

rebalance = decision_agent.should_rebalance(
    np.array([previous_allocation[t] for t in tickers]) if previous_allocation else None,
    final_array
)

# ==============================
# Console Output
# ==============================

print("\n==============================")
print("Autonomous Portfolio Decision")
print("==============================")

print("Risk Score:", risk_score)
print("Market Regime:", regime)
print("Turnover:", turnover)
print("Rebalance Recommended:", rebalance)

print("\nFinal Allocation:")
for k, v in final_weights.items():
    print(f"  {k}: {round(v,4)}")

print("\nPerformance Metrics:")
for k, v in performance.items():
    print(f"  {k}: {v}")

print(f"\nMonte Carlo Projection ({horizon}-Year Forward):")
for k, v in mc_metrics.items():
    print(f"  {k}: {v}")

# Save state AFTER decision
state_manager.save_state(
    final_weights,
    regime,
    performance,
    risk_score
)

# ==============================
# Step 7: LLM Investment Memo
# ==============================

explanation_agent = ExplanationAgent()

print("\nGenerating Investment Memo...\n")

memo = explanation_agent.generate_report(
    regime,
    final_weights,
    {**performance, **mc_metrics},
    horizon
)

print("===== AI Portfolio Memo =====\n")
print(memo)