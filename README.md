<div align="center">

# 🧠 AURA
### Autonomous Regime-Aware Portfolio Intelligence Engine

*A production-grade, multi-agent quantitative portfolio management system for Indian ETF markets*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-5000%20Paths-FF6B6B?style=for-the-badge)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
[![Agents](https://img.shields.io/badge/Multi--Agent-Architecture-6C3483?style=for-the-badge)](https://github.com/BinayakShome/MiniProject)
[![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-2ECC71?style=for-the-badge)](https://miniproject-odye.onrender.com/)

<br/>

> **AURA** is a fully autonomous, multi-agent portfolio intelligence platform that combines quantitative finance, market regime detection, Monte Carlo simulation, and local AI reasoning to deliver real-time, personalized ETF portfolio allocations — no human intervention required.

<br/>

**🌐 Live Demo:** [https://miniproject-odye.onrender.com/](https://miniproject-odye.onrender.com/)

---

</div>

## 📖 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗 System Architecture](#-system-architecture)
- [📂 Project Structure](#-project-structure)
- [🧠 Multi-Agent Framework](#-multi-agent-framework)
- [📊 Quantitative Engine](#-quantitative-engine)
- [🎲 Monte Carlo Simulation Engine](#-monte-carlo-simulation-engine)
- [🔬 Stress Testing & Reliability](#-stress-testing--reliability)
- [⚙️ Installation](#️-installation)
- [🚀 Running the Project](#-running-the-project)
- [🌐 API Documentation](#-api-documentation)
- [📈 Sample Output](#-sample-output)
- [🛡 Error Handling & Fault Tolerance](#-error-handling--fault-tolerance)
- [🔮 Future Roadmap](#-future-roadmap)
- [👥 Team](#-team)
- [🏆 Technical Highlights](#-technical-highlights)
- [📜 License](#-license)

---

## 🎯 Project Overview

### What is AURA?

**AURA (Autonomous Regime-Aware Portfolio Intelligence Engine)** is a fully autonomous, end-to-end portfolio management system designed for the Indian ETF market. It integrates modern portfolio theory, regime-aware asset allocation, probabilistic risk modeling, and optional on-device AI narration — all within a modular multi-agent architecture.

### The Business Problem

Retail and institutional investors face three critical challenges:

| Challenge | Industry Reality | AURA Solution |
|-----------|-----------------|---------------|
| **Static Allocations** | Most portfolios ignore current market conditions | Dynamic regime-based rebalancing |
| **One-Size-Fits-All Risk** | Generic risk profiles ignore personal parameters | Personalized risk scoring per user |
| **Opaque Decisions** | Black-box recommendations lack reasoning | AI-generated institutional investment memos |

### Why Multi-Agent Architecture?

Traditional monolithic financial software cannot adapt in real time to multiple concurrent signals — risk profile, market regime, portfolio drift, and performance metrics. AURA decomposes this problem into **four specialized autonomous agents**, each independently responsible for a single decision domain, orchestrated by a central runtime.

```
User Input → Risk Agent → Regime Agent → Quant Engine → Decision Agent → Explanation Agent → Output
```

This separation of concerns enables independent testing, upgradeability, and graceful degradation (e.g., if the LLM is offline, all other agents continue functioning normally).

### Target Users

- 🏦 **Quantitative Analysts** — for backtesting and optimization workflows
- 🎓 **Finance Students & Researchers** — for portfolio theory experimentation
- 💼 **Retail Investors** — for guided ETF allocation decisions
- 🏢 **Fintech Developers** — as a backend engine for wealth management apps

---

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 📊 **Quantitative Optimization** | Max-Sharpe Ratio via SciPy SLSQP on the efficient frontier | ✅ Production |
| 🗺 **Market Regime Detection** | Classifies Bull / Bear / Sideways / High-Volatility regimes | ✅ Production |
| 🧑‍💼 **User Risk Profiling** | Personalized risk scoring based on age, horizon, and tolerance | ✅ Production |
| 🎲 **Monte Carlo Simulation** | 5,000-path bootstrap wealth projection with VaR & CVaR | ✅ Production |
| 🔄 **Portfolio Rebalancing** | Turnover-triggered rebalance recommendations (>10% drift) | ✅ Production |
| 💾 **State Persistence** | JSON-based session memory with numpy-safe serialization | ✅ Production |
| 🤖 **AI Investment Memo** | Institutional-grade narrative via Ollama / phi3:mini (optional) | ✅ Production |
| 🌐 **FastAPI Backend** | REST API with data caching and HTML dashboard serving | ✅ Production |
| 🖥 **Web Dashboard** | Interactive browser-based portfolio UI | ✅ Production |
| 🔬 **Stress Testing** | 67-test suite covering edge cases, NaN/Inf/overflow conditions | ✅ Production |
| 📉 **Backtesting Engine** | Historical performance validation module | ✅ Production |

---

## 🏗 System Architecture

AURA is structured across six distinct layers, each with a single, well-defined responsibility:

```
╔══════════════════════════════════════════════════════════════════════╗
║                    AURA — System Architecture                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  🎨  PRESENTATION LAYER                                      │    ║
║  │  FastAPI Backend (backend.py) + HTML Dashboard (index.html) │    ║
║  └────────────────────────────┬────────────────────────────────┘    ║
║                               │ REST API / HTML                      ║
║  ┌────────────────────────────▼────────────────────────────────┐    ║
║  │  🤖  AGENT LAYER                                             │    ║
║  │  RiskProfileAgent → MarketRegimeAgent → ExplanationAgent     │    ║
║  └────────────────────────────┬────────────────────────────────┘    ║
║                               │ Adjusted Weights / Regime            ║
║  ┌────────────────────────────▼────────────────────────────────┐    ║
║  │  ⚖️   DECISION LAYER                                         │    ║
║  │  DecisionAgent (Turnover Analysis → Rebalance Trigger)       │    ║
║  └────────────────────────────┬────────────────────────────────┘    ║
║                               │ Final Weights                        ║
║  ┌────────────────────────────▼────────────────────────────────┐    ║
║  │  📊  QUANTITATIVE LAYER                                      │    ║
║  │  Metrics → Optimization (SLSQP) → Monte Carlo → Performance  │    ║
║  └────────────────────────────┬────────────────────────────────┘    ║
║                               │ OHLCV Price Data                     ║
║  ┌────────────────────────────▼────────────────────────────────┐    ║
║  │  📥  DATA LAYER                                              │    ║
║  │  DataLoader → yfinance → NSE ETF Close Prices (2015–Today)  │    ║
║  └────────────────────────────┬────────────────────────────────┘    ║
║                               │ Portfolio State                      ║
║  ┌────────────────────────────▼────────────────────────────────┐    ║
║  │  💾  MEMORY LAYER                                            │    ║
║  │  StateManager → portfolio_state.json (numpy-safe JSON)       │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### ETF Asset Universe

| Ticker | Name | Asset Class |
|--------|------|-------------|
| `NIF100BEES.NS` | Nifty 100 ETF | Large-Cap Equity |
| `MID150BEES.NS` | Nifty Midcap 150 ETF | Mid-Cap Equity |
| `GOLDBEES.NS` | Gold ETF | Commodity |
| `LIQUIDBEES.NS` | Liquid / Overnight ETF | Cash Equivalent |

---

## 📂 Project Structure

```text
📦 AURA — Autonomous Regime-Aware Portfolio Intelligence Engine
│
├── 🚀 main.py                      # CLI entry point — orchestrates the full pipeline
├── 🌐 backend.py                   # FastAPI REST server — serves dashboard & API
├── 📋 requirements.txt             # Python dependency manifest
├── 🧪 stress_test.py               # 67-case system validation suite
├── 💾 state_manager.py             # Root-level state manager (deployment copy)
│
├── 📁 agents/                      # Autonomous decision agents
│   ├── 🤖 risk_profile_agent.py    # User risk scoring & equity cap adjustment
│   ├── 📈 market_regime_agent.py   # Bull/Bear/Sideways/High-Vol regime classifier
│   ├── ⚖️  decision_agent.py        # Turnover-based portfolio rebalance trigger
│   └── 📝 explanation_agent.py     # Local LLM investment memo generator (Ollama)
│
├── 📁 quant/                       # Quantitative finance computation layer
│   ├── 📊 metrics.py               # Log returns, annualized metrics, covariance
│   ├── 🎯 optimization.py          # Max-Sharpe ratio via SciPy SLSQP
│   ├── 🎲 monte_carlo.py           # Bootstrap Monte Carlo simulation (5,000 paths)
│   └── 📉 performance.py           # CAGR, Max Drawdown, Calmar Ratio
│
├── 📁 data/                        # Data ingestion and preprocessing
│   └── 📥 data_loader.py           # yfinance NSE ETF OHLCV fetcher (2015–present)
│
├── 📁 memory/                      # Persistent session state
│   ├── 💾 state_manager.py         # JSON-based portfolio state with NumpyEncoder
│   └── 🗄️  portfolio_state.json     # Last-known allocation, metrics & regime
│
├── 📁 backtest/                    # Historical performance validation
│   └── 🔍 engine.py                # Walk-forward backtesting engine
│
└── 📁 templates/                   # Frontend web assets
    └── 🎨 index.html               # Interactive portfolio dashboard UI
```

---

## 🧠 Multi-Agent Framework

AURA's intelligence is distributed across four specialized agents. Each agent operates independently, receives clearly defined inputs, and produces structured outputs consumed by the next layer.

---

### 🤖 Risk Profile Agent

**File:** `agents/risk_profile_agent.py`

**Purpose:** Translates a user's personal financial profile into a quantitative risk score and adjusts the optimized portfolio weights accordingly.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | Age (int), Investment Horizon (years), Risk Tolerance (`Low` / `Medium` / `High`) |
| **Outputs** | Risk Score `[0.0, 1.0]`, Risk-Adjusted Weight Dictionary |
| **Algorithm** | Weighted composite of age factor, horizon factor, and tolerance mapping |
| **Equity Cap** | Maximum equity exposure = `0.8 × risk_score` |
| **Redistribution** | Excess equity reallocated to `LIQUIDBEES.NS` (or nearest liquid asset) |

**Risk Score Formula:**

```
risk_score = 0.4 × age_factor + 0.4 × horizon_factor + 0.2 × tolerance_factor

where:
  age_factor     = max(0, min(1, (60 - age) / 40))
  horizon_factor = min(horizon / 20, 1)
  tolerance_map  = { low: 0.2, medium: 0.5, high: 0.8 }
```

**Input Validation & Error Handling:**
- `ValueError` raised for negative age
- `ValueError` raised for negative investment horizon
- `ValueError` raised for unrecognised tolerance strings
- Safe redistribution when `LIQUIDBEES.NS` is absent from portfolio

---

### 📈 Market Regime Agent

**File:** `agents/market_regime_agent.py`

**Purpose:** Classifies the current market environment into one of four regimes using rolling volatility and annualized return signals, then tilts the portfolio accordingly.

| Regime | Detection Criteria | Equity Adjustment |
|--------|-------------------|-------------------|
| 🟢 **Bull Market** | High avg return, moderate volatility | **+10%** equity tilt |
| 🔴 **Bear Market** | Negative avg return | **−30%** equity reduction |
| 🟡 **High Volatility** | Volatility exceeds threshold | **−20%** equity reduction |
| ⚪ **Sideways Market** | Neither bull nor bear conditions | No change |

**Inputs:** Log returns DataFrame
**Outputs:** Regime string, Regime-adjusted weight dictionary

**Robustness:**
- NaN/Inf guard using `np.isfinite()` before all comparisons
- Falls back to `"Sideways Market"` on empty or all-NaN input
- Only adjusts tickers that are actually present in the portfolio

---

### ⚖️ Decision Agent

**File:** `agents/decision_agent.py`

**Purpose:** Determines whether a portfolio rebalance is warranted by measuring the turnover between the current allocation and the newly recommended allocation.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | Current weights dict, New optimal weights dict |
| **Output** | `True` (rebalance) / `False` (hold), Turnover value |
| **Threshold** | Default 10% cumulative weight drift |
| **Formula** | `turnover = Σ |new_weight - current_weight|` |

**Logic:** On first run (no prior state), a rebalance is always recommended to establish the baseline allocation.

---

### 📝 Explanation Agent

**File:** `agents/explanation_agent.py`

**Purpose:** Generates an institutional-grade investment memo by prompting a local LLM with structured portfolio data, regime context, and performance metrics.

| Attribute | Detail |
|-----------|--------|
| **Model** | `phi3:mini` via Ollama (local inference) |
| **Endpoint** | `http://localhost:11434/api/generate` |
| **Inputs** | Market regime, horizon, weights dict, performance metrics |
| **Output** | Formatted institutional investment narrative |
| **Fallback** | Graceful plain-text summary when Ollama is offline |

**Prompt Discipline:** Strict instructions prevent the LLM from hallucinating numbers — all figures are injected directly into the prompt with explicit rules:
> *"Use ONLY the numbers provided. Do NOT modify, approximate, or invent values."*

---

## 📊 Quantitative Engine

**Module:** `quant/`

### Log Returns

Daily log returns are computed to handle compounding correctly and ensure statistical normality:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Outliers exceeding ±20% daily moves are filtered to eliminate data errors from corporate events.

### Annualized Return & Volatility

$$\mu_{annual} = \bar{r} \times 252$$

$$\sigma_{annual} = \sigma_r \times \sqrt{252}$$

### Covariance Matrix

$$\Sigma = \text{Cov}(r_1, r_2, \ldots, r_n) \times 252$$

Used to measure cross-asset correlation and portfolio-level variance.

### Sharpe Ratio

$$\text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}$$

Where $r_f = 6\%$ (Indian risk-free rate). Returns `0.0` when $\sigma_p = 0$ to prevent division errors.

### Maximum Sharpe Optimization

The optimizer uses **SciPy SLSQP** (Sequential Least Squares Programming) to find the tangency portfolio on the efficient frontier:

$$\max_w \quad \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}$$

$$\text{subject to:} \quad \sum_i w_i = 1, \quad w_i \geq 0 \; \forall i$$

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **CAGR** | $(V_T / V_0)^{1/T} - 1$ | Compound Annual Growth Rate |
| **Max Drawdown** | Peak-to-trough percentage decline | Worst historical loss from peak |
| **Calmar Ratio** | `CAGR / |Max Drawdown|` | Return-to-risk efficiency ratio |

---

## 🎲 Monte Carlo Simulation Engine

**File:** `quant/monte_carlo.py`

### Bootstrap Sampling

Rather than assuming a parametric return distribution, AURA uses **empirical bootstrap sampling** — drawing directly from the historical daily return record to simulate future paths:

$$\tilde{r}_t \sim \text{Uniform}\{r_1, r_2, \ldots, r_T\}$$

This preserves fat tails and real-world clustering effects present in the data.

### Wealth Projection

For each of 5,000 simulation paths over the user's investment horizon:

$$W_H^{(i)} = \exp\left(\sum_{t=1}^{H \times 252} \tilde{r}_t^{(i)} \cdot w^\top\right)$$

Cumulative log-returns are clipped to `[-700, 700]` before exponentiation to prevent IEEE 754 overflow.

### Risk Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Expected Final Value** | Mean of all simulated terminal wealth | Average wealth multiple after horizon |
| **5% VaR** | 5th percentile of simulated terminal wealth | Worst outcome in 1-in-20 scenarios |
| **5% CVaR (Expected Shortfall)** | Mean of paths below VaR | Average loss in extreme tail scenarios |
| **Probability of Loss** | Fraction of paths ending below 1.0× | Likelihood of losing capital |

### Interpretation Guide

```
Expected Final Value > 1.0  → Portfolio is projected to grow
5% VaR > 1.0                → Even worst-case scenario shows growth
Probability of Loss = 0.0   → No simulated path resulted in a loss
```

---

## 🔬 Stress Testing & Reliability

**File:** `stress_test.py`

AURA ships with a comprehensive stress testing suite designed to validate correctness, numerical stability, and fault tolerance across all modules.

### Test Results

```
Total Tests    : 67
Passed         : 56  ✅
Bugs Found     :  0  ✅
Design Warnings: 11  ⚠️  (documented & non-critical)
```

### Coverage Map

| Module | Test Cases | Edge Cases Covered |
|--------|-----------|-------------------|
| `quant/metrics.py` | 8 | Zero prices, NaN values, single-row data |
| `quant/optimization.py` | 7 | Singular covariance, degenerate weights, zero vol |
| `quant/monte_carlo.py` | 10 | Zero simulations, zero years, empty returns, overflow |
| `quant/performance.py` | 9 | Zero start value, all-zero portfolio, single observation |
| `agents/risk_profile_agent.py` | 11 | Negative age, negative horizon, invalid tolerance |
| `agents/market_regime_agent.py` | 8 | NaN returns, empty DataFrame, all-zero vol |
| `memory/state_manager.py` | 7 | Corrupted JSON, numpy scalars, missing file |
| `agents/explanation_agent.py` | 7 | Ollama offline, empty response, timeout |

### Key Bug Fixes (v2)

| Module | Bug | Resolution |
|--------|-----|------------|
| `quant/metrics.py` | Sharpe = `-inf` when vol = 0 | Returns `0.0` safely |
| `quant/monte_carlo.py` | `IndexError` on 0 simulations | Raises `ValueError` with message |
| `quant/monte_carlo.py` | `exp()` overflow on extreme returns | Clips to `[-700, 700]` |
| `quant/performance.py` | CAGR = `inf` on zero start value | Raises `ValueError` |
| `quant/performance.py` | `max_drawdown = NaN` on all-zero portfolio | Returns `0.0` |
| `agents/risk_profile_agent.py` | Negative inputs silently accepted | Raises `ValueError` |
| `agents/market_regime_agent.py` | `KeyError` on missing ticker | Guards with `in` check |
| `agents/market_regime_agent.py` | NaN comparisons fall through | Guards with `np.isfinite()` |
| `memory/state_manager.py` | `numpy.float64` not JSON-serialisable | Added `_NumpyEncoder` |
| `agents/explanation_agent.py` | Full crash when Ollama offline | `try/except` with fallback |

---

## ⚙️ Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| pip | Latest | Required |
| Ollama | Any | Optional — for AI memo |
| Internet | Active | For yfinance data |

---

### 1. Clone the Repository

```bash
git clone https://github.com/BinayakShome/MiniProject.git
cd MiniProject
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:

```
fastapi       — REST API framework
uvicorn       — ASGI server
numpy         — Numerical computing
scipy         — Scientific optimization (SLSQP)
pandas        — Data manipulation
matplotlib    — Visualization
yfinance      — NSE market data
requests      — HTTP client (Ollama)
pydantic      — Request validation
```

### 4. (Optional) Install Ollama for AI Memos

```bash
# Install Ollama from https://ollama.com
# Then pull the phi3:mini model:

ollama serve
ollama pull phi3:mini
```

> **Note:** If Ollama is not installed or not running, AURA completes all pipeline stages normally. The AI memo section gracefully prints a plain-text fallback summary instead of crashing.

---

## 🚀 Running the Project

### CLI Mode (Terminal)

```bash
cd MiniProject
python main.py
```

You will be prompted for three inputs:

```
Enter your age: 30
Investment horizon (years): 10
Risk tolerance (Low/Medium/High): High
```

---

### FastAPI Backend (Web Server)

```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

The server starts at: **http://localhost:8000**

---

### Web Dashboard

Once the FastAPI server is running, open your browser and navigate to:

```
http://localhost:8000
```

The interactive dashboard loads automatically, powered by `templates/index.html`.

---

### Stress Tests

```bash
python stress_test.py
```

Expected output:

```
Running AURA Stress Test Suite...
[67/67] Tests Executed
✅ 56 Passed  |  ⚠️ 11 Warnings  |  ❌ 0 Failed
```

---

## 🌐 API Documentation

The FastAPI backend exposes three endpoints:

---

### `GET /`

Serves the HTML dashboard.

**Response:** `text/html` — the portfolio dashboard UI.

---

### `GET /api/state`

Returns the last saved portfolio state from the JSON store.

**Response:**

```json
{
  "has_state": true,
  "state": {
    "allocation": {
      "NIF100BEES.NS": 0.053,
      "MID150BEES.NS": 0.3258,
      "GOLDBEES.NS": 0.6212,
      "LIQUIDBEES.NS": 0.0
    },
    "regime": "Sideways Market",
    "risk_score": 0.67,
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

---

### `POST /api/analyze`

Runs the full AURA pipeline and returns portfolio analysis results.

**Request Body:**

```json
{
  "age": 30,
  "horizon": 10,
  "tolerance": "High",
  "generate_memo": true
}
```

**Response:**

```json
{
  "risk_score": 0.67,
  "regime": "Sideways Market",
  "turnover": 0.0147,
  "rebalance_recommended": true,
  "allocation": {
    "NIF100BEES.NS": 0.053,
    "MID150BEES.NS": 0.3258,
    "GOLDBEES.NS": 0.6212,
    "LIQUIDBEES.NS": 0.0
  },
  "performance": {
    "return": 0.212,
    "volatility": 0.1276,
    "sharpe": 1.1914,
    "cagr": 0.2357,
    "max_drawdown": -0.1933,
    "calmar": 1.2193
  },
  "monte_carlo": {
    "expected_final_value": 9.061,
    "var_5pct": 4.278,
    "cvar_5pct": 3.623,
    "probability_of_loss": 0.0
  },
  "memo": "Institutional Investment Memo: ..."
}
```

---

## 📈 Sample Output

### CLI Output — Full Pipeline Run

```
Downloading data...
[*********************100%***********************]  4 of 4 completed
Download complete
Assets downloaded: ['GOLDBEES.NS', 'LIQUIDBEES.NS', 'MID150BEES.NS', 'NIF100BEES.NS']
Data Shape: (1626, 4)

Enter your age: 30
Investment horizon (years): 10
Risk tolerance (Low/Medium/High): High

==============================
  Autonomous Portfolio Decision
==============================
Risk Score             : 0.67
Market Regime          : Sideways Market
Portfolio Turnover     : 0.0147
Rebalance Recommended  : Yes

Final Allocation:
  NIF100BEES.NS (Nifty 100 ETF)     :  5.30%
  MID150BEES.NS (Midcap 150 ETF)    : 32.58%
  GOLDBEES.NS   (Gold ETF)           : 62.12%
  LIQUIDBEES.NS (Liquid Fund)        :  0.00%

Performance Metrics:
  Annualized Return    :  21.20%
  Annualized Volatility:  12.76%
  Sharpe Ratio         :  1.1914
  CAGR                 :  23.57%
  Max Drawdown         : -19.33%
  Calmar Ratio         :  1.2193

Monte Carlo Projection (10-Year Forward, 5,000 Paths):
  Expected Final Value      : 9.06x  (1 unit → 9.06 units)
  5% Value at Risk          : 4.28x
  5% CVaR (Exp. Shortfall)  : 3.62x
  Probability of Loss       : 0.00%
```

---

### AI Investment Memo (Ollama Output)

```
===== AURA AI Portfolio Memo =====

Institutional Investment Memo: ETF Portfolio Allocation & Outlook

Subject: 10-Year Horizon — Diversified Indian ETF Portfolio

Allocation Rationale:
In a Sideways Market environment, the portfolio is optimally positioned
with a 62.12% allocation to GOLDBEES as a defensive anchor, 32.58% to
MID150BEES for growth exposure, and 5.30% to NIF100BEES for large-cap
stability. Zero allocation to LIQUIDBEES reflects confidence in the
identified long-term trend.

Risk-Adjusted Performance:
  Sharpe Ratio: 1.1914 — superior risk-adjusted return profile
  Calmar Ratio: 1.2193 — strong return-to-drawdown efficiency
  Max Drawdown: -19.33% — within acceptable bounds for growth investors

Monte Carlo Outlook (10-Year):
  Expected Final Value: 9.061x — compelling long-term compounding
  5% VaR: 4.278x — even adverse scenarios project positive growth
  Probability of Loss: 0.0% — no simulated path ended below capital

Respectfully submitted,
AURA Portfolio Intelligence Engine
```

---

## 🛡 Error Handling & Fault Tolerance

AURA is engineered for production-grade resilience. Every failure mode is handled explicitly:

| Failure Scenario | Module | Handling Strategy |
|-----------------|--------|------------------|
| **Negative age / horizon** | `risk_profile_agent.py` | Raises `ValueError` with descriptive message |
| **Invalid tolerance string** | `risk_profile_agent.py` | Raises `ValueError` listing valid options |
| **Missing ticker in portfolio** | `risk_profile_agent.py` | Redistributes to any present liquid asset |
| **Empty price data** | `data_loader.py` | Raises `ValueError` — "No data downloaded" |
| **All-NaN returns** | `market_regime_agent.py` | Falls back to `"Sideways Market"` |
| **NaN vol/return comparisons** | `market_regime_agent.py` | `np.isfinite()` guard before logic |
| **Zero portfolio volatility** | `metrics.py` | Returns Sharpe `= 0.0`, avoids division error |
| **Singular covariance matrix** | `optimization.py` | SLSQP handles via penalty `1e6` |
| **Zero simulations / years** | `monte_carlo.py` | Raises `ValueError` before execution |
| **exp() overflow in simulation** | `monte_carlo.py` | Clips log-returns to `[-700, 700]` |
| **Zero start portfolio value** | `performance.py` | Raises `ValueError` — CAGR undefined |
| **All-zero portfolio** | `performance.py` | Returns `max_drawdown = 0.0` |
| **Single-observation returns** | `performance.py` | Returns `annual_vol = 0.0` |
| **Ollama offline / unreachable** | `explanation_agent.py` | `try/except` → plain-text fallback memo |
| **numpy.float64 in JSON** | `state_manager.py` | Custom `_NumpyEncoder` handles all numpy types |
| **Corrupted JSON state file** | `state_manager.py` | Returns `{}` and initialises fresh state |

---

## 🔮 Future Roadmap

| Phase | Feature | Priority |
|-------|---------|----------|
| 🟢 **v2.1** | Live NSE WebSocket data feed | High |
| 🟢 **v2.1** | Multi-asset universe (bonds, REITs, international ETFs) | High |
| 🟡 **v2.2** | Reinforcement Learning rebalancing agent (PPO) | Medium |
| 🟡 **v2.2** | Sentiment Analysis Agent (FinBERT on NSE news) | Medium |
| 🟡 **v2.2** | Portfolio Explainability Dashboard (SHAP-based) | Medium |
| 🔵 **v3.0** | Cloud deployment on AWS / GCP with CI/CD pipeline | Medium |
| 🔵 **v3.0** | Multi-user authentication and portfolio isolation | Low |
| 🔵 **v3.0** | Tax-loss harvesting optimization module | Low |
| 🔵 **v3.0** | ESG-aware portfolio scoring layer | Low |
| ⚪ **v4.0** | Mobile application (React Native) | Planned |

---

## 👥 Team

<div align="center">

| Name | Role | Responsibility |
|------|------|----------------|
| **Debdyuti Chakraborty** | Team Lead & Quant Engineer | Optimization, Efficient Frontier |
| **Binayak Shome** |  Agent Architecture | System Design |
| **Atharva Pratap Singh** | Performance Analyst | Backtesting, Performance Metrics |
| **Aryan Sinha** | Risk Engineer | Monte Carlo Simulation, VaR/CVaR |
| **Aryan Yadav** | Data Engineer | Data Pipeline, yfinance Integration |
| **Deepjyoti Bhattacharya** | Systems Engineer | Memory & State Management |

</div>

<br/>

> 🎓 **6th Semester Mini Project**
> **Project Guide:** Prof. Himanshu Ranjan
> **Institution:** KIIT University

---

## 🏆 Technical Highlights

```
✅  Multi-agent architecture with 4 specialized autonomous agents
✅  Max-Sharpe portfolio optimization via SciPy SLSQP on efficient frontier
✅  Regime-aware dynamic allocation across 4 market states
✅  5,000-path empirical bootstrap Monte Carlo with VaR, CVaR, and loss probability
✅  Graceful LLM integration via Ollama with full offline fallback
✅  Production-safe numerical handling (NaN, Inf, overflow, singularity guards)
✅  Custom NumpyEncoder for numpy-safe JSON serialization
✅  FastAPI REST backend with data caching and HTML dashboard
✅  67-case stress test suite with 0 bugs and full edge-case coverage
✅  Live deployed demo on Render
```

---

## 📜 License

```
MIT License
Copyright (c) 2026 AURA Team — KIIT University
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

See [LICENSE](LICENSE) for the full license text.

---

<div align="center">

**Built with 🧠 intelligence, 📊 precision, and ⚡ autonomy**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-miniproject--odye.onrender.com-4A90E2?style=for-the-badge)](https://miniproject-odye.onrender.com/)
[![GitHub](https://img.shields.io/badge/GitHub-BinayakShome%2FMiniProject-181717?style=for-the-badge&logo=github)](https://github.com/BinayakShome/MiniProject)

*AURA — Where Quantitative Finance Meets Autonomous Intelligence*

</div>
