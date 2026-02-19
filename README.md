# 🎯 AURA — Autonomous Regime-Aware Portfolio Intelligence Engine

> 6th Semester Mini Project

---

## 👥 Team Members

| Name                   | Role                        |
| ---------------------- | --------------------------- |
| Debdyuti Chakraborty   | Quant Engine & Optimization |
| Binayak Shome          | Agent Architecture          |
| Atharva Pratap Singh   | Backtesting & Performance   |
| Aryan Sinha            | Monte Carlo & Risk          |
| Aryan Yadav            | Data Pipeline               |
| Deepjyoti Bhattacharya | Memory & State Management   |

**Project Guide:** Prof. Himanshu Ranjan

---

## 📌 What is AURA?

AURA is a fully autonomous, multi-agent portfolio management system for Indian ETFs. It combines quantitative optimization, market regime detection, user risk profiling, and Monte Carlo simulation to produce real-time, personalized portfolio allocations — with an optional AI-generated investment memo via a local LLM.

---

## 🏗️ Architecture

```
main.py
├── data/
│   └── data_loader.py          — Fetches OHLCV data via yfinance
├── quant/
│   ├── metrics.py              — Log returns, Sharpe, covariance
│   ├── optimization.py         — Max-Sharpe & Min-Vol (SciPy SLSQP)
│   ├── monte_carlo.py          — Bootstrap Monte Carlo simulation
│   └── performance.py          — CAGR, Max Drawdown, Calmar Ratio
├── agents/
│   ├── risk_profile_agent.py   — User risk scoring & equity cap
│   ├── market_regime_agent.py  — Bull / Bear / Sideways / High-Vol detection
│   ├── decision_agent.py       — Turnover-based rebalance trigger
│   └── explanation_agent.py    — LLM investment memo (Ollama / phi3:mini)
└── memory/
    └── state_manager.py        — JSON-based persistent portfolio state
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install numpy scipy pandas matplotlib yfinance requests
```

### (Optional) AI Memo via Ollama

```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull phi3:mini
```

> If Ollama is not running, AURA still completes fully — the memo section prints a plain-text summary instead of crashing.

---

## ▶️ Running the Program

```bash
cd MiniProject-main
python main.py
```

You will be prompted for:

```
Enter your age: 30
Investment horizon (years): 10
Risk tolerance (Low/Medium/High): High
```

### Example Output

```
==============================
Autonomous Portfolio Decision
==============================
Risk Score: 0.77
Market Regime: Sideways Market
Rebalance Recommended: True

Final Allocation:
  NIF100BEES.NS:  0.0530   ← Large-cap ETF
  MID150BEES.NS:  0.3258   ← Mid-cap ETF
  GOLDBEES.NS:    0.6212   ← Gold ETF
  LIQUIDBEES.NS:  0.0000   ← Liquid ETF

Performance Metrics:
  Return:       0.2120
  Volatility:   0.1276
  Sharpe:       1.1914
  CAGR:         0.2357
  Max Drawdown: -0.1933
  Calmar Ratio: 1.2193

Monte Carlo Projection (20-Year Forward):
  Expected Final Value:        81.077
  5% VaR:                      27.859
  5% CVaR (Expected Shortfall): 22.595
  Probability of Loss:          0.0
```

---

## 🧠 How It Works

### 1. Data Loading

Downloads daily Close prices for 4 Indian ETFs from Yahoo Finance (NSE):

- `NIF100BEES.NS` — Nifty 100 (large-cap equity)
- `MID150BEES.NS` — Nifty Midcap 150 (mid-cap equity)
- `GOLDBEES.NS` — Gold ETF
- `LIQUIDBEES.NS` — Liquid / overnight ETF

### 2. Quantitative Optimization

Computes annualised log returns and covariance matrix, then runs **Max Sharpe Ratio** optimization (SciPy SLSQP) to find the efficient frontier allocation.

### 3. Risk Profile Agent

Scores the user on a `[0, 1]` scale from age, investment horizon, and risk tolerance. Caps equity exposure at `0.8 × risk_score` and redistributes excess to liquid assets.

**Input validation (v2):** Raises `ValueError` on negative age, negative horizon, or unrecognised tolerance string.

### 4. Market Regime Detection

Classifies the current market into one of four regimes based on rolling volatility and mean return, then tilts the allocation accordingly:

| Regime          | Equity Adjustment |
| --------------- | ----------------- |
| Bull Market     | +10% equity       |
| High Volatility | −20% equity       |
| Bear Market     | −30% equity       |
| Sideways Market | No change         |

### 5. Monte Carlo Simulation

Bootstraps 5,000 paths over the user's investment horizon. Reports:

- **Expected Final Value** — mean wealth multiple
- **5% VaR** — worst-case at 5th percentile
- **5% CVaR** — expected shortfall below VaR
- **Probability of Loss** — fraction of paths ending below 1×

### 6. Decision Agent

Compares current weights to the last saved allocation. Recommends rebalancing if turnover exceeds 10%.

### 7. State Manager

Saves the final allocation, regime, performance metrics, and risk score to `memory/portfolio_state.json` for use on the next run.

### 8. AI Investment Memo (optional)

Sends all metrics to a local `phi3:mini` model via Ollama to generate an institutional-style investment memo. Degrades gracefully if Ollama is offline.

---

## 🔬 Stress Testing

A full stress test suite is included (`stress_test.py`) covering 67 tests across all modules:

```bash
python stress_test.py
```

Tests cover zero/negative/NaN/Inf prices, degenerate weight vectors, singular covariance matrices, zero-simulation Monte Carlo, CAGR on zero-start portfolios, missing ticker keys in agents, corrupted JSON state, and more.

**Latest results:** 56 passed · 0 bugs · 11 design warnings

---

## 🐛 Bug Fixes (v2)

| Module                          | Bug                                                      | Fix                                                        |
| ------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| `quant/metrics.py`              | Sharpe = `-inf` when portfolio vol = 0                   | Returns `0.0` when vol is zero                             |
| `quant/monte_carlo.py`          | `IndexError` crash when `simulations=0` or `years=0`     | Raises `ValueError` with clear message                     |
| `quant/monte_carlo.py`          | `exp()` overflow on extreme returns                      | Cumulative returns clipped to `[-700, 700]` before `exp()` |
| `quant/performance.py`          | `CAGR = inf` when start value is 0                       | Raises `ValueError`                                        |
| `quant/performance.py`          | `max_drawdown = NaN` on all-zero portfolio               | Returns `0.0` safely                                       |
| `quant/performance.py`          | `annual_vol = NaN` for single observation                | Returns `0.0` when `len < 2`                               |
| `agents/risk_profile_agent.py`  | Negative age/horizon accepted silently                   | Raises `ValueError` on invalid inputs                      |
| `agents/risk_profile_agent.py`  | Unknown tolerance silently treated as medium             | Raises `ValueError`                                        |
| `agents/risk_profile_agent.py`  | `KeyError` when `LIQUIDBEES.NS` absent                   | Redistributes excess to any present non-equity asset       |
| `agents/market_regime_agent.py` | `KeyError` on missing ticker in `adjust_for_regime`      | Only adjusts tickers present in the portfolio              |
| `agents/market_regime_agent.py` | NaN comparisons in `detect_regime` silently fall through | Guards with `np.isfinite()` before comparisons             |
| `memory/state_manager.py`       | `numpy.float64` not JSON-serialisable                    | Added `_NumpyEncoder` custom JSON encoder                  |
| `agents/explanation_agent.py`   | Full crash when Ollama is offline                        | `try/except` with graceful fallback message                |

---

## 📁 Project Structure

```
MiniProject-main/
├── main.py
├── README.md
├── stress_test.py
├── data/
│   ├── __init__.py
│   └── data_loader.py
├── quant/
│   ├── metrics.py
│   ├── optimization.py
│   ├── monte_carlo.py
│   └── performance.py
├── agents/
│   ├── risk_profile_agent.py
│   ├── market_regime_agent.py
│   ├── decision_agent.py
│   └── explanation_agent.py
└── memory/
    ├── state_manager.py
    └── portfolio_state.json
```

---

## 📄 License

See [LICENSE](LICENSE)
