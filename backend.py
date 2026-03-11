from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

# Existing modules from your project
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

app = FastAPI()

# ==============================
# Asset Universe
# ==============================

tickers = [
    "NIF100BEES.NS",
    "MID150BEES.NS",
    "GOLDBEES.NS",
    "LIQUIDBEES.NS"
]

allocation_labels = {
    "NIF100BEES.NS": "Nifty 100 ETF",
    "MID150BEES.NS": "Midcap 150 ETF",
    "GOLDBEES.NS": "Gold ETF",
    "LIQUIDBEES.NS": "Liquid Fund"
}

allocation_colors = {
    "NIF100BEES.NS": "#C8A96E",
    "MID150BEES.NS": "#8BBFCC",
    "GOLDBEES.NS": "#E8C547",
    "LIQUIDBEES.NS": "#3A4260"
}


# ==============================
# Request Model
# ==============================

class AnalyzeRequest(BaseModel):
    age: int
    horizon: int
    tolerance: str
    generate_memo: bool = False


# ==============================
# Serve Dashboard
# ==============================

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ==============================
# Load Previous State
# ==============================

@app.get("/api/state")
def get_state():

    state_manager = StateManager()
    state = state_manager.load_state()

    return {
        "has_state": bool(state),
        "state": state
    }


# ==============================
# Main Portfolio Analysis API
# ==============================

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):

    age = req.age
    horizon = req.horizon
    tolerance = req.tolerance

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
    # Step 2: Risk Profile Agent
    # ==============================

    risk_agent = RiskProfileAgent(age, horizon, tolerance)
    risk_adjusted_weights, risk_score = risk_agent.adjust_allocation(base_weights.copy())

    # ==============================
    # Step 3: Market Regime Agent
    # ==============================

    regime_agent = MarketRegimeAgent(log_returns)
    regime = regime_agent.detect_regime()

    final_weights = regime_agent.adjust_for_regime(
        risk_adjusted_weights.copy(),
        regime
    )

    final_array = np.array([final_weights[t] for t in tickers])

    # ==============================
    # Portfolio Performance
    # ==============================

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
        "MaxDrawdown": float(round(max_dd, 4)),
        "CalmarRatio": float(round(calmar, 4))
    }

    # ==============================
    # Monte Carlo Simulation
    # ==============================

    mc_results_array, mc_metrics = monte_carlo_simulation(
        final_array,
        log_returns,
        years=horizon,
        simulations=1500
    )

    # Create histogram structure for UI

    counts, edges = np.histogram(mc_results_array, bins=40)

    monte_carlo_output = {
        "ExpectedFinalValue": float(round(mc_metrics.get("Expected Final Value", np.mean(mc_results_array)), 4)),
        "ProbabilityOfLoss": float(round(np.mean(mc_results_array < 1.0), 4)),
        "VaR5": float(mc_metrics.get("5% VaR", 0)),
        "CVaR5": float(mc_metrics.get("5% CVaR (Expected Shortfall)", 0)),
        "histogram": {
            "edges": edges.tolist(),
            "counts": counts.tolist()
        }
    }

    # ==============================
    # State Memory + Turnover
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
    # Decision Agent
    # ==============================

    decision_agent = DecisionAgent()

    rebalance = decision_agent.should_rebalance(
        np.array([previous_allocation[t] for t in tickers]) if previous_allocation else None,
        final_array
    )

    # ==============================
    # Save State
    # ==============================

    state_manager.save_state(
        final_weights,
        regime,
        performance,
        risk_score
    )

    # ==============================
    # AI Memo
    # ==============================

    memo = None

    if req.generate_memo:
        explanation_agent = ExplanationAgent()

        memo = explanation_agent.generate_report(
            regime,
            final_weights,
            {**performance, **mc_metrics},
            horizon
        )

    # ==============================
    # Return API Response
    # ==============================

    return {
        "regime": regime,
        "riskScore": risk_score,
        "rebalanceRecommended": rebalance,
        "turnover": turnover,
        "allocation": final_weights,
        "allocationLabels": allocation_labels,
        "allocationColors": allocation_colors,
        "performance": performance,
        "monteCarlo": monte_carlo_output,
        "historicalValues": portfolio_values.tolist(),
        "memo": memo
    }
#Run python -m uvicorn backend:app --reload
